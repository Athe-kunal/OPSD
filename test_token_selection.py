"""
Quick smoke-test for all five token_selection_mode variants.

Builds a tiny in-memory dataset, instantiates the data collator + mask logic,
and verifies that:
  - baseline  : all non-pad labels are active
  - first_sentence : at least some labels are active, fewer than baseline
  - middle_sentences : similar
  - last_sentence : similar
  - paragraph_first_token : very few labels active (one per paragraph)

Run from the repo root:
    python test_token_selection.py

No GPU or training is required — this only exercises the mask computation path.
"""

import torch
from datasets import Dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Tiny dataset — 2 examples with multi-paragraph solutions
# ---------------------------------------------------------------------------
# Each paragraph intentionally has >=3 sentences so 'middle_sentences' can select some.
SAMPLES = [
    {
        "problem": "Solve x^2 - 5x + 6 = 0.",
        "solution": (
            "We need to factor the quadratic x^2 - 5x + 6. "
            "Factoring a quadratic means rewriting it as a product of two linear terms. "
            "This is possible when we find two numbers whose product equals the constant term and whose sum equals the linear coefficient.\n\n"
            "Notice that 2 * 3 = 6 and 2 + 3 = 5. "
            "So we can write x^2 - 5x + 6 = (x-2)(x-3). "
            "We can verify this by expanding: (x-2)(x-3) = x^2 - 5x + 6, which confirms the factoring.\n\n"
            "Setting each factor to zero: x - 2 = 0 gives x = 2. "
            "Similarly, x - 3 = 0 gives x = 3. "
            "Therefore the solutions are x = 2 and x = 3."
        ),
    },
    {
        "problem": "What is the derivative of sin(x)?",
        "solution": (
            "The derivative of sin(x) is a standard result from calculus. "
            "We approach this using the first-principles limit definition. "
            "This method makes no assumptions and applies to any differentiable function.\n\n"
            "Using the limit definition, d/dx sin(x) = lim_{h->0} [sin(x+h)-sin(x)]/h. "
            "We expand sin(x+h) using the addition formula: sin(x+h) = sin(x)cos(h) + cos(x)sin(h). "
            "Substituting back and grouping terms gives sin(x)[cos(h)-1]/h + cos(x)sin(h)/h.\n\n"
            "As h->0, [cos(h)-1]/h -> 0 and sin(h)/h -> 1. "
            "These are standard limits that can be proved geometrically. "
            "Therefore d/dx sin(x) = cos(x)."
        ),
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_PATH = "Qwen/Qwen3-1.7B"

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_fake_labels_and_texts(tokenizer, completion_texts, prompt_len, max_comp_len=128):
    """Simulate what training_step builds: labels tensor + completion token lists."""
    pad_id = tokenizer.pad_token_id
    batch = len(completion_texts)

    # Encode each completion
    comp_ids_list = []
    for text in completion_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_comp_len]
        comp_ids_list.append(ids)

    # Pad to same length
    max_len = max(len(ids) for ids in comp_ids_list)
    padded = []
    for ids in comp_ids_list:
        padded.append(ids + [pad_id] * (max_len - len(ids)))

    # Build fake full sequence: [prompt_placeholder] + [completion]
    prompt_placeholder = [[pad_id] * prompt_len] * batch
    full_ids = torch.tensor(
        [p + c for p, c in zip(prompt_placeholder, padded)], dtype=torch.long
    )

    # Labels: mask prompt and pad tokens
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    labels[labels == pad_id] = -100

    return labels, comp_ids_list


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_test(mode: str, top_k: int, tokenizer, trainer_stub):
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}  |  top_k: {top_k}")
    print(f"{'='*60}")

    completion_texts = [s["solution"] for s in SAMPLES]
    prompt_len = 32

    labels, comp_ids_list = make_fake_labels_and_texts(tokenizer, completion_texts, prompt_len)
    baseline_active = (labels != -100).sum().item()

    if mode != "baseline":
        trainer_stub.token_selection_mode = mode
        trainer_stub.token_selection_top_k = top_k
        sel_masks = trainer_stub._compute_token_selection_mask(comp_ids_list, completion_texts)

        for i, sel_mask in enumerate(sel_masks):
            comp_start = prompt_len
            comp_end = comp_start + len(sel_mask)
            deselected = ~sel_mask.to(labels.device)
            labels[i, comp_start:comp_end][deselected] = -100

    active = (labels != -100).sum().item()
    print(f"  Active label positions : {active} / {baseline_active} (baseline)")

    if mode == "baseline":
        assert active == baseline_active, "baseline should keep all positions"
    elif mode == "paragraph_first_token":
        # Expect very few active positions (one per paragraph per example)
        assert active > 0, "must have at least one active position"
        assert active < baseline_active, "must be sparser than baseline"
        print(f"  OK — sparse as expected ({active} positions)")
    else:
        assert active > 0, "must have at least one active position"
        assert active <= baseline_active, "must not exceed baseline"
        print(f"  OK")

    return active


def _compute_token_selection_mask(self, completion_ids_list: list, completion_texts: list) -> list:
    """Verbatim copy of OPSDTrainer._compute_token_selection_mask for standalone testing."""
    import nltk

    tokenize = self.processing_class
    mode = self.token_selection_mode
    masks = []

    for comp_ids, comp_text in zip(completion_ids_list, completion_texts):
        n_tokens = len(comp_ids)
        mask = torch.zeros(n_tokens, dtype=torch.bool)

        raw_paragraphs = comp_text.split("\n\n")
        token_pos = 0

        for para_idx, para in enumerate(raw_paragraphs):
            para_with_sep = para if para_idx == len(raw_paragraphs) - 1 else para + "\n\n"
            para_ids = tokenize.encode(para_with_sep, add_special_tokens=False)
            para_len = len(para_ids)

            if mode == "paragraph_first_token":
                if token_pos < n_tokens:
                    mask[token_pos] = True
            else:
                sentences = nltk.sent_tokenize(para) if para.strip() else []
                n_sents = len(sentences)
                sent_token_pos = token_pos

                for sent_idx, sent in enumerate(sentences):
                    sent_ids = tokenize.encode(sent, add_special_tokens=False)
                    sent_len = len(sent_ids)

                    is_first = sent_idx == 0
                    is_last = sent_idx == n_sents - 1
                    is_middle = not is_first and not is_last

                    selected = (
                        (mode == "first_sentence" and is_first)
                        or (mode == "last_sentence" and is_last)
                        or (mode == "middle_sentences" and is_middle)
                    )

                    if selected:
                        end = min(sent_token_pos + sent_len, n_tokens)
                        mask[sent_token_pos:end] = True

                    sent_token_pos += sent_len

            token_pos = min(token_pos + para_len, n_tokens)

        masks.append(mask)

    return masks


class TrainerStub:
    """Minimal stub exposing only _compute_token_selection_mask."""

    def __init__(self, tokenizer, mode="baseline", top_k=1):
        self.processing_class = tokenizer
        self.token_selection_mode = mode
        self.token_selection_top_k = top_k

    _compute_token_selection_mask = _compute_token_selection_mask


if __name__ == "__main__":
    import nltk
    # Download punkt tokenizer data if not already present
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading nltk punkt_tab...")
        nltk.download("punkt_tab", quiet=True)

    print(f"Loading tokenizer from {MODEL_PATH} ...")
    tokenizer = load_tokenizer()
    stub = TrainerStub(tokenizer)

    results = {}
    for mode, top_k in [
        ("baseline", 1),
        ("first_sentence", 1),
        ("first_sentence", 2),
        ("middle_sentences", 1),
        ("last_sentence", 1),
        ("paragraph_first_token", 1),
    ]:
        results[(mode, top_k)] = run_test(mode, top_k, tokenizer, stub)

    print("\n" + "="*60)
    print("All tests passed!")
    print("\nSummary (active label positions):")
    for (mode, k), count in results.items():
        print(f"  {mode:<25} top_k={k}  ->  {count} positions")
