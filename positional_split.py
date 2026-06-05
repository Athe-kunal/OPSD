"""
positional_split.py

Paragraph-level token selection based on equal thirds.

Findings from analyze_kl_by_position.py (KL proxy = log p_teacher - log p_student):
  - last_third  : mean=+0.132, std=0.360  — highest, most consistent teacher signal
  - middle_third: mean=+0.123, std=0.457  — close second
  - first_third : mean=+0.030, std=2.227  — noisy; student overconfident at paragraph starts

Recommended modes for distillation: 'last_third' or 'last_two_thirds'.
"""

import torch

MIN_THIRD_TOKENS = 3  # paragraphs with fewer tokens are merged before splitting into thirds

THIRDS_MODES = {"first_third", "middle_third", "last_third", "last_two_thirds"}


def merge_short_paragraphs(paragraphs: list[str], tokenizer, min_tokens: int = MIN_THIRD_TOKENS) -> list[str]:
    """Merge paragraphs that are too short to split into thirds.

    Short paragraphs (typically bare LaTeX delimiters like '\\[' or '\\end{align*}')
    are merged forward into the next paragraph.  If the short paragraph is the last
    one, it is merged backward into the previous one instead.

    Args:
        paragraphs: raw list from completion_text.split('\\n\\n')
        tokenizer:  processing_class (used only to count tokens)
        min_tokens: paragraphs with fewer tokens than this are merged (default: 3)

    Returns:
        New list of paragraphs where every entry tokenizes to >= min_tokens.
    """
    merged: list[str] = []
    for para in paragraphs:
        tok_len = len(tokenizer.encode(para, add_special_tokens=False))
        if tok_len < min_tokens:
            if merged:
                # short paragraph — append to previous (merge backward)
                merged[-1] = merged[-1] + "\n\n" + para
            else:
                # very first paragraph is short — keep pending
                merged.append(para)
        else:
            if merged and len(tokenizer.encode(merged[-1], add_special_tokens=False)) < min_tokens:
                # previous was short and had no predecessor — merge it forward into this one
                merged[-1] = merged[-1] + "\n\n" + para
            else:
                merged.append(para)
    return merged


def compute_thirds_mask(
    completion_ids: list[int],
    completion_text: str,
    tokenizer,
    mode: str,
) -> torch.Tensor:
    """Return a boolean mask over completion token positions for equal-thirds selection.

    Each paragraph is divided into equal thirds by token count.  Short paragraphs are
    pre-merged via merge_short_paragraphs so every paragraph has at least MIN_THIRD_TOKENS
    tokens before splitting.  Any remainder from integer division falls into the last third.

    Args:
        completion_ids:   list of token ids for the completion (no prompt)
        completion_text:  decoded text of the completion (used for paragraph splitting)
        tokenizer:        processing_class
        mode:             one of 'first_third' | 'middle_third' | 'last_third' | 'last_two_thirds'

    Returns:
        Boolean tensor of shape (len(completion_ids),).
        True  = position contributes to the JSD loss.
        False = position is masked out (labels set to -100).
    """
    if mode not in THIRDS_MODES:
        raise ValueError(f"Unknown thirds mode '{mode}'. Expected one of {THIRDS_MODES}.")

    n_tokens = len(completion_ids)
    mask = torch.zeros(n_tokens, dtype=torch.bool)

    raw_paragraphs = completion_text.split("\n\n")
    paragraphs = merge_short_paragraphs(raw_paragraphs, tokenizer)

    token_pos = 0
    for para_idx, para in enumerate(paragraphs):
        # Re-add '\n\n' separator for all but the last paragraph so cumulative
        # token counts stay aligned with the actual completion encoding.
        para_with_sep = para if para_idx == len(paragraphs) - 1 else para + "\n\n"
        para_len = len(tokenizer.encode(para_with_sep, add_special_tokens=False))

        para_start = token_pos
        para_end = min(token_pos + para_len, n_tokens)
        actual_len = para_end - para_start

        if actual_len > 0:
            third = max(1, actual_len // 3)
            t1 = para_start + third       # boundary between first and middle third
            t2 = para_start + 2 * third   # boundary between middle and last third

            if mode == "first_third":
                mask[para_start:t1] = True
            elif mode == "middle_third":
                mask[t1:t2] = True
            elif mode == "last_third":
                mask[t2:para_end] = True
            elif mode == "last_two_thirds":
                mask[t1:para_end] = True

        token_pos = para_end

    return mask
