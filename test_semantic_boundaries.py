"""
Test semantic paragraph boundary detection using Option 4:
  1. Split on \n\n (coarse boundaries)
  2. Embed each paragraph with a small sentence transformer
  3. Compute cosine similarity between adjacent paragraphs
  4. Merge adjacent paragraphs whose similarity exceeds a threshold
     (they are semantically continuous — e.g. a prose claim + its LaTeX block)

Run:
    python test_semantic_boundaries.py [--threshold 0.6] [--n 5] [--model all-MiniLM-L6-v2]
"""

import argparse
import re
import random
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ── helpers ──────────────────────────────────────────────────────────────────

def strip_latex(text: str) -> str:
    """Remove display-math blocks and inline math before embedding.
    Keeps prose so the embedding reflects meaning, not symbol noise.
    """
    # Remove display math \[ ... \] and $$ ... $$
    text = re.sub(r'\\\[.*?\\\]', ' <MATH> ', text, flags=re.DOTALL)
    text = re.sub(r'\$\$.*?\$\$', ' <MATH> ', text, flags=re.DOTALL)
    # Remove \begin{...} ... \end{...} environments
    text = re.sub(r'\\begin\{[^}]*\}.*?\\end\{[^}]*\}', ' <MATH> ', text, flags=re.DOTALL)
    # Remove inline math $...$
    text = re.sub(r'\$[^$]+\$', '<MATH>', text)
    # Remove remaining LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def merge_paragraphs(paragraphs: list[str], embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Return groups of paragraph indices that should be merged.
    Adjacent paragraphs with cosine similarity >= threshold are merged.
    """
    if len(paragraphs) == 0:
        return []

    groups = [[0]]
    for i in range(1, len(paragraphs)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim >= threshold:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


# ── visualization ─────────────────────────────────────────────────────────────

def visualize(solution: str, model: SentenceTransformer, threshold: float, example_idx: int):
    print(f"\n{'='*100}")
    print(f"EXAMPLE {example_idx + 1}")
    print(f"{'='*100}")

    # Step 1: coarse split
    raw_paragraphs = solution.split("\n\n")
    non_empty = [(i, p) for i, p in enumerate(raw_paragraphs) if p.strip()]

    print(f"\nCoarse \\n\\n split → {len(raw_paragraphs)} blocks "
          f"({len(non_empty)} non-empty)\n")

    if len(non_empty) == 0:
        print("  <no content>")
        return

    indices, paragraphs = zip(*non_empty)

    # Step 2: embed (strip LaTeX first)
    stripped = [strip_latex(p) for p in paragraphs]
    embeddings = model.encode(stripped, show_progress_bar=False)

    # Step 3: similarities between adjacent paragraphs
    print("  Pairwise cosine similarities between adjacent non-empty paragraphs:")
    sims = []
    for i in range(len(paragraphs) - 1):
        sim = cosine_sim(embeddings[i], embeddings[i + 1])
        sims.append(sim)
        verdict = "MERGE ↓" if sim >= threshold else "SPLIT  |"
        p_preview = paragraphs[i].replace("\n", " ")[:60]
        n_preview = paragraphs[i + 1].replace("\n", " ")[:60]
        print(f"    [{i}↔{i+1}]  sim={sim:.3f}  {verdict}")
        print(f"           A: {p_preview}...")
        print(f"           B: {n_preview}...")

    # Step 4: merge
    groups = merge_paragraphs(list(paragraphs), embeddings, threshold)

    print(f"\n  After merging (threshold={threshold}): {len(groups)} semantic paragraphs\n")

    for g_idx, group in enumerate(groups):
        merged_text = "\n\n".join(paragraphs[i] for i in group)
        preview = merged_text.replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "..."

        n = len(groups)
        position = (
            "FIRST" if g_idx == 0 and n > 1
            else "LAST" if g_idx == n - 1 and n > 1
            else "ONLY" if n == 1
            else None
        )
        if position is None:
            continue
        block_label = f"blocks {group}" if len(group) > 1 else f"block {group[0]}"
        print(f"  [{position}]  ({block_label})")
        print(f"    {preview}")
        print()


def check_selection_coverage(solution: str, model: SentenceTransformer, threshold: float):
    """Show token coverage per selection mode after semantic merging."""
    raw_paragraphs = solution.split("\n\n")
    non_empty = [p for p in raw_paragraphs if p.strip()]

    if not non_empty:
        return

    stripped = [strip_latex(p) for p in non_empty]
    embeddings = model.encode(stripped, show_progress_bar=False)
    groups = merge_paragraphs(non_empty, embeddings, threshold)

    # Approximate word counts per merged paragraph
    word_counts = [
        sum(len(non_empty[i].split()) for i in g)
        for g in groups
    ]
    total = sum(word_counts)
    n = len(groups)

    first_words = word_counts[0] if n >= 1 else 0
    last_words  = word_counts[-1] if n >= 1 else 0
    selected    = first_words + (last_words if n > 1 else 0)

    print(f"  Semantic paragraphs after merge: {n}")
    print(f"  Approx total words: {total}")
    print(f"  Words selected (first + last):")
    for label, count in [("first_paragraph", first_words), ("last_paragraph", last_words if n > 1 else 0)]:
        pct = 100 * count / total if total > 0 else 0
        print(f"    {label:<25}: {count:4d} words  ({pct:.1f}% of total)")
    pct_total = 100 * selected / total if total > 0 else 0
    print(f"    {'TOTAL selected':<25}: {selected:4d} words  ({pct_total:.1f}% of total)")
    skipped = total - selected
    print(f"    {'middle (skipped)':<25}: {skipped:4d} words  ({100 - pct_total:.1f}% of total)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Cosine similarity threshold for merging adjacent paragraphs (default: 0.6)")
    parser.add_argument("--n", type=int, default=5, help="Number of examples to test (default: 5)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    print(f"Loading sentence transformer: {args.model} ...")
    model = SentenceTransformer(args.model)

    print(f"Loading dataset siyanzhao/Openthoughts_math_30k_opsd ...")
    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
    train = dataset["train"]

    indices = random.sample(range(len(train)), args.n)

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

        visualize(solution, model, args.threshold, i)

        print(f"  --- Token Selection Mode Coverage ---")
        check_selection_coverage(solution, model, args.threshold)

    print(f"\n{'='*100}")
    print("SUMMARY: Check the above to verify:")
    print("  1. Adjacent blocks that belong together (prose + LaTeX) are merged")
    print("  2. Topic shifts produce clean SPLIT boundaries")
    print("  3. FIRST/LAST labels fall on semantically meaningful units (middle skipped)")
    print("  4. Try --threshold 0.5 (more merging) or 0.7 (less merging) to tune")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
