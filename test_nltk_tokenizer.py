"""
Test nltk sentence tokenizer on random examples from the OPSD dataset.
Visualizes how the reference solution (teacher completion) is split into paragraphs
and sentences, mirroring the logic in OPSDTrainer._compute_token_selection_mask.
"""

import random
import textwrap
import nltk
from datasets import load_dataset

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

N_EXAMPLES = 5
WRAP_WIDTH = 100


def visualize_tokenization(solution: str, example_idx: int):
    print(f"\n{'='*100}")
    print(f"EXAMPLE {example_idx + 1}")
    print(f"{'='*100}")

    paragraphs = solution.split("\n\n")
    print(f"Total paragraphs: {len(paragraphs)}\n")

    for para_idx, para in enumerate(paragraphs):
        print(f"  [Paragraph {para_idx + 1}]")
        if not para.strip():
            print("    <empty paragraph>")
            print()
            continue

        # Print the raw paragraph (truncated for readability)
        raw_preview = para.replace("\n", "\\n")
        if len(raw_preview) > 120:
            raw_preview = raw_preview[:120] + "..."
        print(f"    Raw: {raw_preview}")

        sentences = nltk.sent_tokenize(para)
        print(f"    Sentences found: {len(sentences)}")

        for sent_idx, sent in enumerate(sentences):
            position = (
                "FIRST" if sent_idx == 0 and len(sentences) > 1
                else "LAST" if sent_idx == len(sentences) - 1 and len(sentences) > 1
                else "ONLY" if len(sentences) == 1
                else "MIDDLE"
            )
            sent_preview = sent.replace("\n", " ")
            if len(sent_preview) > 90:
                sent_preview = sent_preview[:90] + "..."
            print(f"      [{position}] {sent_preview}")
        print()


def check_selection_modes(solution: str):
    """Show which tokens would be selected under each token_selection_mode."""
    modes = ["first_sentence", "middle_sentences", "last_sentence", "paragraph_first_token"]
    paragraphs = solution.split("\n\n")

    counts = {m: 0 for m in modes}
    total_sentences = 0

    for para in paragraphs:
        if not para.strip():
            continue
        sentences = nltk.sent_tokenize(para)
        n = len(sentences)
        total_sentences += n

        for i, sent in enumerate(sentences):
            sent_len = len(sent.split())  # approximate word count
            if i == 0:
                counts["first_sentence"] += sent_len
            if i == n - 1:
                counts["last_sentence"] += sent_len
            if 0 < i < n - 1:
                counts["middle_sentences"] += sent_len

        counts["paragraph_first_token"] += 1  # just 1 token per paragraph

    total_words = sum(len(s.split()) for p in paragraphs for s in nltk.sent_tokenize(p) if p.strip())

    print(f"  Total paragraphs (non-empty): {sum(1 for p in paragraphs if p.strip())}")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Approx total words: {total_words}")
    print(f"  Words selected per mode (approx):")
    for mode, count in counts.items():
        pct = 100 * count / total_words if total_words > 0 else 0
        print(f"    {mode:<25}: {count:4d} words  ({pct:.1f}% of total)")


def main():
    print("Loading dataset siyanzhao/Openthoughts_math_30k_opsd ...")
    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
    train = dataset["train"]

    indices = random.sample(range(len(train)), N_EXAMPLES)

    for i, idx in enumerate(indices):
        example = train[idx]
        solution = example["solution"]
        problem_preview = example["problem"][:120].replace("\n", " ")

        print(f"\n{'#'*100}")
        print(f"DATASET INDEX: {idx}")
        print(f"Problem: {problem_preview}...")
        print(f"Solution length: {len(solution)} chars")

        print(f"\n--- FULL REFERENCE SOLUTION ---\n")
        print(solution)
        print(f"\n--- END OF SOLUTION ---")

        visualize_tokenization(solution, i)

        print(f"  --- Token Selection Mode Coverage ---")
        check_selection_modes(solution)

    print(f"\n{'='*100}")
    print("SUMMARY: Check the above to verify:")
    print("  1. Paragraphs split correctly on \\n\\n boundaries")
    print("  2. Sentences within each paragraph look semantically correct")
    print("  3. FIRST/LAST/MIDDLE labels match your expectations")
    print("  4. Coverage percentages make sense for your intended selection mode")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
