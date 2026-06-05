"""
Test pysbd sentence segmenter on random examples from the OPSD dataset.
Mirrors test_nltk_tokenizer.py but uses pysbd instead of nltk.sent_tokenize.
"""

import random
import pysbd
from datasets import load_dataset

N_EXAMPLES = 5

segmenter = pysbd.Segmenter(language="en", clean=False)


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

        raw_preview = para.replace("\n", "\\n")
        if len(raw_preview) > 120:
            raw_preview = raw_preview[:120] + "..."
        print(f"    Raw: {raw_preview}")

        sentences = segmenter.segment(para) if para.strip() else []
        print(f"    Sentences found: {len(sentences)}")

        for sent_idx, sent in enumerate(sentences):
            n = len(sentences)
            position = (
                "FIRST" if sent_idx == 0 and n > 1
                else "LAST" if sent_idx == n - 1 and n > 1
                else "ONLY" if n == 1
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
        sentences = segmenter.segment(para)
        n = len(sentences)
        total_sentences += n

        for i, sent in enumerate(sentences):
            sent_len = len(sent.split())
            if i == 0:
                counts["first_sentence"] += sent_len
            if i == n - 1:
                counts["last_sentence"] += sent_len
            if 0 < i < n - 1:
                counts["middle_sentences"] += sent_len

        counts["paragraph_first_token"] += 1

    total_words = sum(
        len(s.split())
        for p in paragraphs
        for s in (segmenter.segment(p) if p.strip() else [])
    )

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
    print("  5. Compare vs nltk: numbered items (4.), LaTeX (\\end{align*}) handled correctly?")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
