#!/usr/bin/env python3
"""
analyze_kl_by_position.py

For each student-generated completion, measures the per-token log-prob difference
    delta_i = log p_teacher(t_i | teacher_ctx) - log p_student(t_i | student_ctx)
broken down by sentence position within each \\n\\n-separated paragraph
(first / middle / last / single-sentence paragraphs).

Teacher context:  problem + ground-truth solution (privileged info).
                  Chat template with enable_thinking=True  (matches run_opsd_1b.sh).
Student context:  problem only.
                  Chat template with enable_thinking=False.

KL proxy: delta_i is the per-token contribution to the reverse log-likelihood ratio.
Positive delta → teacher assigns higher probability than student (student is uncertain).
We aggregate mean delta per sentence position to identify WHERE the student diverges most.

Requirements:
    pip install openai datasets transformers scipy nltk tqdm

Usage:
    python analyze_kl_by_position.py \\
        --base_url http://localhost:8000/v1 \\
        --model Qwen/Qwen3-1.7B \\
        --num_samples 100 \\
        --max_new_tokens 512
"""

import argparse
import json
from collections import defaultdict

import nltk
import numpy as np
from datasets import load_dataset
from openai import OpenAI
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Prompt construction — mirrors data_collator.py exactly
# ---------------------------------------------------------------------------

_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


def build_student_prompt(tokenizer, problem: str) -> str:
    msg = (
        f"Problem: {problem}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_teacher_prompt(tokenizer, problem: str, solution: str) -> str:
    msg = (
        f"Problem: {problem}\n\n"
        "Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"{_TRANSITION_PROMPT}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


# ---------------------------------------------------------------------------
# Sentence-position classifier — same cursor logic as _compute_token_selection_mask
# ---------------------------------------------------------------------------

# Each paragraph is divided into equal thirds for approach 5.


def classify_tokens(
    tokenizer, completion_text: str, n_tokens: int
) -> tuple[list[str], list[bool], list[int], list[str], list[str], list[int]]:
    """
    Returns (labels, is_para_first_token, first_sent_tok_idx, approach1_labels, approach5_labels, para_ids).

    labels              : sentence-position label per token ('first','middle','last','single','unknown')
    is_para_first_token : True for the opening token of each paragraph
    first_sent_tok_idx  : within-sentence index for tokens in a paragraph's first sentence, else -1
    approach1_labels    : 'boundary_first' | 'boundary_last' | 'other'
                          (the very first and very last token of each paragraph)
    approach5_labels    : 'ratio_first' | 'ratio_last' | 'ratio_middle'
                          (equal thirds of each paragraph by token count)
    para_ids            : paragraph index (0-based) for each token position
    """
    labels = ["unknown"] * n_tokens
    is_para_first_token = [False] * n_tokens
    first_sent_tok_idx = [-1] * n_tokens
    approach1_labels = ["other"] * n_tokens
    approach5_labels = ["ratio_middle"] * n_tokens
    para_ids = [-1] * n_tokens

    MIN_THIRD_TOKENS = 3  # paragraphs with fewer tokens than this are merged

    raw_paragraphs = completion_text.split("\n\n")

    # Merge short paragraphs: forward into next, or backward if last.
    paragraphs: list[str] = []
    for i, para in enumerate(raw_paragraphs):
        tok_len = len(tokenizer.encode(para, add_special_tokens=False))
        if tok_len < MIN_THIRD_TOKENS and paragraphs:
            # too short — append to the previous paragraph (merge forward-looking content back)
            # but if this is the last raw paragraph just append to previous
            paragraphs[-1] = paragraphs[-1] + "\n\n" + para
        elif tok_len < MIN_THIRD_TOKENS and not paragraphs:
            # first paragraph is short — keep it; will be merged when next paragraph is processed
            paragraphs.append(para)
        else:
            if paragraphs and len(tokenizer.encode(paragraphs[-1], add_special_tokens=False)) < MIN_THIRD_TOKENS:
                # previous paragraph was short and had no predecessor to merge into — merge it forward into this one
                paragraphs[-1] = paragraphs[-1] + "\n\n" + para
            else:
                paragraphs.append(para)

    token_pos = 0

    for para_idx, para in enumerate(paragraphs):
        para_with_sep = para if para_idx == len(paragraphs) - 1 else para + "\n\n"
        para_len = len(tokenizer.encode(para_with_sep, add_special_tokens=False))

        para_start = token_pos
        para_end = min(token_pos + para_len, n_tokens)
        actual_len = para_end - para_start

        if actual_len > 0:
            # --- Paragraph ID ---
            for k in range(para_start, para_end):
                para_ids[k] = para_idx

            # --- Approach 1: boundary tokens only ---
            approach1_labels[para_start] = "boundary_first"
            approach1_labels[para_end - 1] = "boundary_last"

            # --- Approach 5: equal thirds ---
            # Short paragraphs are pre-merged above, so third >= 1 is guaranteed.
            third = max(1, actual_len // 3)
            t1 = para_start + third          # end of first third (exclusive)
            t2 = para_start + 2 * third      # end of second third (exclusive)
            # last third absorbs any remainder from integer division
            for k in range(para_start, t1):
                approach5_labels[k] = "ratio_first"
            for k in range(t1, t2):
                approach5_labels[k] = "ratio_middle"
            for k in range(t2, para_end):
                approach5_labels[k] = "ratio_last"

        # --- Existing sentence-position labels ---
        sentences = nltk.sent_tokenize(para) if para.strip() else []
        n_sents = len(sentences)
        sent_pos = token_pos

        for sent_idx, sent in enumerate(sentences):
            sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
            if n_sents == 1:
                pos_label = "single"
            elif sent_idx == 0:
                pos_label = "first"
            elif sent_idx == n_sents - 1:
                pos_label = "last"
            else:
                pos_label = "middle"

            end = min(sent_pos + sent_len, n_tokens)
            for k in range(sent_pos, end):
                labels[k] = pos_label
            if sent_idx == 0 and sent_pos < n_tokens:
                is_para_first_token[sent_pos] = True
                for k in range(sent_pos, end):
                    first_sent_tok_idx[k] = k - sent_pos
            sent_pos += sent_len

        token_pos = para_end

    return labels, is_para_first_token, first_sent_tok_idx, approach1_labels, approach5_labels, para_ids


# ---------------------------------------------------------------------------
# vLLM API helpers
# ---------------------------------------------------------------------------

def generate_student_batch(
    client: OpenAI,
    model: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k_sample: int,
) -> list[tuple[str, list[float]]]:
    """
    Generates completions for a batch of student prompts in one API call.
    Returns list of (completion_text, per_token_logprobs) in prompt order.
    vLLM processes all prompts in parallel on the GPU.
    """
    resp = client.completions.create(
        model=model,
        prompt=prompts,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"top_k": top_k_sample},
        logprobs=1,
    )
    # vLLM returns choices in any order; sort by index to match input order
    choices = sorted(resp.choices, key=lambda c: c.index)
    return [(c.text, c.logprobs.token_logprobs) for c in choices]


def get_teacher_logprobs_batch(
    client: OpenAI,
    model: str,
    teacher_prompts: list[str],
    completion_texts: list[str],
    tokenizer,
) -> list[list[float | None]]:
    """
    Scores completion tokens under teacher context for an entire batch in one API call.

    For each example: concatenates teacher_prompt + completion_text, sends to vLLM
    with echo=True so we get logprobs for all prompt positions, then slices out the
    completion portion using a local teacher-prompt token count.

    Returns list of per-token logprob lists (one list per example).
    """
    teacher_token_counts = [
        len(tokenizer.encode(tp, add_special_tokens=False)) for tp in teacher_prompts
    ]
    full_texts = [tp + ct for tp, ct in zip(teacher_prompts, completion_texts)]

    resp = client.completions.create(
        model=model,
        prompt=full_texts,
        max_tokens=1,       # 1 dummy token so vLLM accepts the request
        temperature=0.0,
        echo=True,
        logprobs=1,
    )
    choices = sorted(resp.choices, key=lambda c: c.index)

    results = []
    for choice, offset in zip(choices, teacher_token_counts):
        all_lps = choice.logprobs.token_logprobs
        # Slice completion portion; drop the 1 generated dummy token at the very end
        teacher_lps = all_lps[offset:-1] if len(all_lps) > offset else []
        results.append(teacher_lps)
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def summarize(name: str, vals: list[float]):
    if len(vals) < 2:
        print(f"  {name}: insufficient data (n={len(vals)})")
        return
    a = np.array(vals, dtype=np.float64)
    mean = a.mean()
    std = a.std(ddof=1)
    se = std / np.sqrt(len(a))
    lo, hi = stats.t.interval(0.95, df=len(a) - 1, loc=mean, scale=se)
    print(f"  {name:6s}: mean={mean:.5f}  std={std:.5f}  95% CI=[{lo:.5f}, {hi:.5f}]  (n={len(a)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_url", default="http://localhost:8007/v1", help="vLLM OpenAI-compatible server URL")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model name as registered in vLLM")
    p.add_argument("--api_key", default="EMPTY", help="API key (vLLM default: EMPTY)")
    p.add_argument("--num_samples", type=int, default=1000, help="Number of dataset examples to process")
    p.add_argument("--dataset_split", default="train", help="Dataset split to sample from")
    p.add_argument("--max_new_tokens", type=int, default=2048, help="Max student completion tokens")
    p.add_argument("--temperature", type=float, default=1.1, help="Sampling temperature (matches run_opsd_1b.sh)")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k_sample", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128,
                   help="Number of examples to send to vLLM in one API call (student generation and teacher scoring run as a single batched request each)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_json", default=None, help="Optional path to save raw per-token results as JSON")
    return p.parse_args()


def main():
    args = parse_args()

    # NLTK punkt tokenizer
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading nltk punkt_tab...", flush=True)
        nltk.download("punkt_tab", quiet=True)

    print(f"Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset: siyanzhao/Openthoughts_math_30k_opsd", flush=True)
    ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split=args.dataset_split)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)
    samples = [ds[int(i)] for i in indices]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Accumulate per-sentence-position log-prob deltas
    position_deltas: dict[str, list[float]] = defaultdict(list)
    # 'first', 'middle', 'last', 'single' → list of per-token deltas across all examples
    para_first_token_deltas: list[float] = []  # delta for each paragraph's first token
    # Within the first sentence of each paragraph: token-0 delta vs rest
    first_sent_tok0_deltas: list[float] = []   # delta of token index 0 in each first sentence
    first_sent_rest_deltas: list[float] = []   # deltas of all other tokens in first sentences
    # Per-sentence: did token-0 have the max delta in that sentence?
    first_sent_tok0_is_max: list[bool] = []

    # Approach 1: boundary token deltas (every boundary token)
    approach1_deltas: dict[str, list[float]] = defaultdict(list)  # 'boundary_first', 'boundary_last', 'other'
    # Approach 5: per-paragraph MAX delta (highest-entropy token) per segment
    # Each entry = max delta of one paragraph's segment, enabling a fair per-paragraph Mann-Whitney test
    approach5_max_deltas: dict[str, list[float]] = defaultdict(list)  # 'ratio_first', 'ratio_last', 'ratio_middle'

    raw_records = []  # for optional JSON dump

    # Split samples into batches
    batches = [samples[i : i + args.batch_size] for i in range(0, len(samples), args.batch_size)]

    for batch in tqdm(batches, desc=f"Batches (size={args.batch_size})"):
        problems  = [ex["problem"]  for ex in batch]
        solutions = [ex["solution"] for ex in batch]

        student_prompts = [build_student_prompt(tokenizer, p)          for p in problems]
        teacher_prompts = [build_teacher_prompt(tokenizer, p, s)       for p, s in zip(problems, solutions)]

        # --- Batched student generation (single API call, all prompts in parallel) ---
        try:
            student_results = generate_student_batch(
                client, args.model, student_prompts,
                args.max_new_tokens, args.temperature, args.top_p, args.top_k_sample,
            )
        except Exception as e:
            tqdm.write(f"[WARN] Student generation batch failed: {e}")
            continue

        completion_texts = [r[0] for r in student_results]
        student_lps_batch = [r[1] for r in student_results]

        # --- Batched teacher scoring (single API call, all teacher+completion pairs in parallel) ---
        try:
            teacher_lps_batch = get_teacher_logprobs_batch(
                client, args.model, teacher_prompts, completion_texts, tokenizer
            )
        except Exception as e:
            tqdm.write(f"[WARN] Teacher scoring batch failed: {e}")
            continue

        # --- Per-example processing ---
        for problem, completion_text, student_lps, teacher_lps in zip(
            problems, completion_texts, student_lps_batch, teacher_lps_batch
        ):
            if not completion_text.strip():
                continue

            n_tokens = len(student_lps)
            if n_tokens == 0:
                continue

            # Align lengths (tokenization boundary effects can cause ±1 difference)
            n_aligned = min(n_tokens, len(teacher_lps))
            if n_aligned == 0:
                continue

            s_lps = student_lps[:n_aligned]
            t_lps = teacher_lps[:n_aligned]

            # --- Classify tokens by sentence position ---
            pos_labels, is_para_first, first_sent_idx, a1_labels, a5_labels, para_ids = classify_tokens(
                tokenizer, completion_text, n_aligned
            )

            # --- Accumulate deltas ---
            record_tokens = []
            cur_first_sent_deltas: dict[int, list] = defaultdict(list)
            # per-paragraph, per-segment bucket for approach 5 max computation
            para_seg_deltas: dict[tuple[int, str], list[float]] = defaultdict(list)

            for i, (s_lp, t_lp) in enumerate(zip(s_lps, t_lps)):
                if s_lp is None or t_lp is None:
                    continue
                delta = t_lp - s_lp

                # Existing sentence-position
                label = pos_labels[i]
                if label != "unknown":
                    position_deltas[label].append(delta)
                if is_para_first[i]:
                    para_first_token_deltas.append(delta)
                widx = first_sent_idx[i]
                if widx >= 0:
                    cur_first_sent_deltas[i - widx].append((widx, delta))
                    if widx == 0:
                        first_sent_tok0_deltas.append(delta)
                    else:
                        first_sent_rest_deltas.append(delta)

                # Approach 1: boundary tokens (every boundary token contributes)
                approach1_deltas[a1_labels[i]].append(delta)

                # Approach 5: collect per-paragraph per-segment deltas for max extraction
                if para_ids[i] >= 0:
                    para_seg_deltas[(para_ids[i], a5_labels[i])].append(delta)

                record_tokens.append({
                    "pos": i, "label": label,
                    "para_first": is_para_first[i],
                    "a1": a1_labels[i], "a5": a5_labels[i],
                    "s_lp": s_lp, "t_lp": t_lp, "delta": delta,
                })

            # Flush approach 5: take the max-delta (highest-entropy) token per paragraph per segment
            for (_, seg), deltas in para_seg_deltas.items():
                if deltas:
                    approach5_max_deltas[seg].append(max(deltas))

            # Per-sentence: check if tok0 had the max delta
            for sent_entries in cur_first_sent_deltas.values():
                if not sent_entries:
                    continue
                max_delta = max(d for _, d in sent_entries)
                tok0_delta = next((d for w, d in sent_entries if w == 0), None)
                if tok0_delta is not None:
                    first_sent_tok0_is_max.append(tok0_delta >= max_delta)

            raw_records.append({
                "problem": problem[:120],
                "completion_len": n_aligned,
                "tokens": record_tokens,
            })

    # --- Optional JSON dump ---
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(raw_records, f, indent=2)
        print(f"\nRaw results saved to {args.output_json}")

    # --- Report ---
    start_kls  = position_deltas.get("first",  [])
    middle_kls = position_deltas.get("middle", [])
    end_kls    = position_deltas.get("last",   [])
    single_kls = position_deltas.get("single", [])

    print(f"\nProcessed {len(raw_records)} examples")
    print(f"\nMean per-token KL proxy  (log p_teacher - log p_student)")
    summarize("Start",  start_kls)
    summarize("Middle", middle_kls)
    summarize("End",    end_kls)
    if single_kls:
        summarize("Single", single_kls)

    # --- Pairwise Mann-Whitney U tests ---
    print("\nMann-Whitney U tests (one-sided: row > col)")
    pairs = [("Start", start_kls), ("Middle", middle_kls), ("End", end_kls)]
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            n1, v1 = pairs[i]
            n2, v2 = pairs[j]
            if len(v1) < 2 or len(v2) < 2:
                print(f"  {n1} > {n2}: insufficient data")
                continue
            _, p = stats.mannwhitneyu(v1, v2, alternative="greater")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {n1} > {n2}: p={p:.2e}  {sig}")

    # --- Is token-0 of the first sentence the highest-entropy token in that sentence? ---
    print("\nFirst-sentence token-0 vs rest (within first sentences only):")
    if len(first_sent_tok0_deltas) >= 2 and len(first_sent_rest_deltas) >= 2:
        summarize("Tok-0 ", first_sent_tok0_deltas)
        summarize("Rest  ", first_sent_rest_deltas)
        _, p = stats.mannwhitneyu(first_sent_tok0_deltas, first_sent_rest_deltas, alternative="greater")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  Mann-Whitney (tok-0 > rest): p={p:.2e}  {sig}")
    if first_sent_tok0_is_max:
        n_max = sum(first_sent_tok0_is_max)
        n_tot = len(first_sent_tok0_is_max)
        print(f"  Fraction of first sentences where tok-0 has max delta: {n_max}/{n_tot} = {n_max/n_tot:.3f}")
    else:
        print("  insufficient data")

    # --- High-entropy analysis for paragraph-first tokens ---
    print("\nParagraph-first-token high-entropy analysis:")
    if len(para_first_token_deltas) >= 2:
        all_deltas = []
        for v in position_deltas.values():
            all_deltas.extend(v)
        if all_deltas:
            q75 = np.percentile(all_deltas, 75)
            q90 = np.percentile(all_deltas, 90)
            pft = np.array(para_first_token_deltas)
            n_total = len(pft)
            n_hi75 = int((pft > q75).sum())
            n_hi90 = int((pft > q90).sum())
            print(f"  Paragraph-first tokens sampled : {n_total}")
            print(f"  Global delta 75th percentile   : {q75:.5f}")
            print(f"  Global delta 90th percentile   : {q90:.5f}")
            print(f"  Fraction above p75 (high-entropy): {n_hi75}/{n_total} = {n_hi75/n_total:.3f}")
            print(f"  Fraction above p90 (high-entropy): {n_hi90}/{n_total} = {n_hi90/n_total:.3f}")
            summarize("ParaFirst", para_first_token_deltas)
            # one-sided Mann-Whitney: para_first > all_other
            _, p = stats.mannwhitneyu(para_first_token_deltas, all_deltas, alternative="greater")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  Mann-Whitney (para_first > all): p={p:.2e}  {sig}")
    else:
        print("  insufficient data")

    # --- Approach 1: boundary token analysis ---
    bf = approach1_deltas.get("boundary_first", [])
    bl = approach1_deltas.get("boundary_last",  [])
    ot = approach1_deltas.get("other",          [])

    print(f"\n{'='*60}")
    print(f"APPROACH 1: Paragraph boundary tokens only")
    print(f"  (very first token and very last token of each \\n\\n paragraph)")
    print(f"{'='*60}")
    summarize("BoundaryFirst", bf)
    summarize("BoundaryLast ", bl)
    summarize("Other        ", ot)

    print("\nMann-Whitney U tests (one-sided: row > Other):")
    for name, vals in [("BoundaryFirst", bf), ("BoundaryLast", bl)]:
        if len(vals) < 2 or len(ot) < 2:
            print(f"  {name} > Other: insufficient data")
            continue
        _, p = stats.mannwhitneyu(vals, ot, alternative="greater")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {name} > Other: p={p:.2e}  {sig}")

    print("\nMann-Whitney (BoundaryFirst > BoundaryLast):")
    if len(bf) >= 2 and len(bl) >= 2:
        _, p = stats.mannwhitneyu(bf, bl, alternative="greater")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  p={p:.2e}  {sig}")

    # --- Approach 5: ratio-based analysis (max-delta token per paragraph per segment) ---
    rf  = approach5_max_deltas.get("ratio_first",  [])
    rl  = approach5_max_deltas.get("ratio_last",   [])
    rm  = approach5_max_deltas.get("ratio_middle", [])

    print(f"\n{'='*60}")
    print(f"APPROACH 5: Equal-thirds split — highest-entropy token per paragraph segment")
    print(f"  (first third / middle third / last third; one max-delta value per paragraph)")
    print(f"{'='*60}")
    summarize("RatioFirst ", rf)
    summarize("RatioLast  ", rl)
    summarize("RatioMiddle", rm)

    print("\nMann-Whitney U tests (one-sided: row > RatioMiddle):")
    for name, vals in [("RatioFirst", rf), ("RatioLast", rl)]:
        if len(vals) < 2 or len(rm) < 2:
            print(f"  {name} > RatioMiddle: insufficient data")
            continue
        _, p = stats.mannwhitneyu(vals, rm, alternative="greater")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {name} > RatioMiddle: p={p:.2e}  {sig}")

    print("\nMann-Whitney (RatioFirst > RatioLast):")
    if len(rf) >= 2 and len(rl) >= 2:
        _, p = stats.mannwhitneyu(rf, rl, alternative="greater")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  p={p:.2e}  {sig}")

    # --- Summary counts ---
    print("\nToken counts per position:")
    for label in ("first", "middle", "last", "single"):
        print(f"  {label:6s}: {len(position_deltas.get(label, []))}")
    print(f"  {'para_first':10s}: {len(para_first_token_deltas)}")
    print(f"\nApproach 1 token counts:")
    for label in ("boundary_first", "boundary_last", "other"):
        print(f"  {label:15s}: {len(approach1_deltas.get(label, []))}")
    print(f"\nApproach 5 paragraph counts (equal thirds, one max-delta per paragraph):")
    for label in ("ratio_first", "ratio_last", "ratio_middle"):
        print(f"  {label:15s}: {len(approach5_max_deltas.get(label, []))}")


if __name__ == "__main__":
    main()
