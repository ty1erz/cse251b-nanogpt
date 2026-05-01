#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from collections import Counter

import numpy as np

try:
    import tiktoken
except Exception:
    tiktoken = None


EOT_TOKEN = 50256


DOMAIN_PATTERNS = {
    "wiki_reference": [
        r"\bcategory:\b",
        r"\breferences\b",
        r"\bexternal links\b",
        r"\bsee also\b",
        r"\baccording to\b",
        r"\bencyclopedia\b",
        r"\bwikipedia\b",
    ],
    "educational_textbook": [
        r"\bin this chapter\b",
        r"\blearning objectives\b",
        r"\bexercise\b",
        r"\bdefinition\b",
        r"\bexample\b",
        r"\bsummary\b",
        r"\bstudents?\b",
        r"\blesson\b",
    ],
    "math_science": [
        r"\bequation\b",
        r"\btheorem\b",
        r"\bproof\b",
        r"\bmatrix\b",
        r"\bderivative\b",
        r"\bintegral\b",
        r"\blemma\b",
        r"\balgorithm\b",
        r"\$\$",
        r"\\\(",
        r"\\\)",
    ],
    "qa_forum": [
        r"\bq:\b",
        r"\ba:\b",
        r"\bquestion\b",
        r"\banswer\b",
        r"\bstack overflow\b",
        r"\basked\b",
        r"\bposted\b",
        r"\bcomment\b",
    ],
    "fiction_story": [
        r"\bsaid\b",
        r"\basked\b",
        r"\"",
        r"\bchapter\b",
        r"\bcharacter\b",
        r"\blooked at\b",
        r"\bsuddenly\b",
    ],
    "web_article": [
        r"https?://",
        r"\bnewsletter\b",
        r"\bcookie\b",
        r"\bsubscribe\b",
        r"\bprivacy policy\b",
        r"\bwebsite\b",
        r"\bblog\b",
        r"\bposted on\b",
    ],
    "code_technical": [
        r"\bdef\s+\w+\(",
        r"\bclass\s+\w+",
        r"\bimport\s+\w+",
        r"</?[A-Za-z][^>]*>",
        r"\bconsole\.log\b",
        r"\bprintf\s*\(",
        r"\bfunction\s+\w+\(",
        r"```",
    ],
}


def split_docs(tokens):
    docs = []
    start = 0
    for i, tok in enumerate(tokens):
        if tok == EOT_TOKEN:
            if i > start:
                docs.append(tokens[start:i])
            start = i + 1
    if start < len(tokens):
        docs.append(tokens[start:])
    return docs


def safe_decode(enc, toks):
    if enc is None:
        return " ".join(map(str, toks[:256]))
    try:
        return enc.decode(list(map(int, toks)))
    except Exception:
        return ""


def score_text(text):
    text_l = text.lower()
    scores = {}
    for domain, patterns in DOMAIN_PATTERNS.items():
        score = 0
        for pat in patterns:
            score += len(re.findall(pat, text_l))
        scores[domain] = score
    return scores


def summarize_docs(decoded_docs):
    domain_counter = Counter()
    aggregate_scores = Counter()
    url_domains = Counter()
    char_lengths = []
    token_lengths = []

    for text, ntok in decoded_docs:
        char_lengths.append(len(text))
        token_lengths.append(ntok)
        scores = score_text(text)
        aggregate_scores.update(scores)
        top_domain, top_score = max(scores.items(), key=lambda kv: kv[1])
        if top_score > 0:
            domain_counter[top_domain] += 1
        else:
            domain_counter["unclear"] += 1

        for m in re.findall(r"https?://([^/\s]+)", text.lower()):
            url_domains[m] += 1

    def mean(xs):
        return sum(xs) / max(1, len(xs))

    return {
        "doc_count": len(decoded_docs),
        "mean_chars": mean(char_lengths),
        "mean_tokens": mean(token_lengths),
        "domain_vote_counts": dict(domain_counter),
        "aggregate_pattern_scores": dict(aggregate_scores),
        "top_url_domains": url_domains.most_common(20),
    }


def main():
    parser = argparse.ArgumentParser(description="Heuristic domain analysis for val.bin")
    parser.add_argument(
        "--data",
        type=str,
        default="/data/fengfei/cse251b-nanogpt/val.bin",
        help="Path to uint16 tokenized validation bin",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/fengfei/cse251b-nanogpt/build-nanogpt/val_domain_report_v1",
        help="Where to save text samples and JSON summary",
    )
    parser.add_argument("--num_samples", type=int, default=50, help="Number of decoded sample docs/passages")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--max_decode_tokens",
        type=int,
        default=512,
        help="Maximum tokens to decode per sampled document/passage",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if tiktoken is None:
        raise RuntimeError("tiktoken is required for decoding GPT-2 token ids")
    enc = tiktoken.get_encoding("gpt2")

    tokens = np.memmap(args.data, dtype=np.uint16, mode="r")
    docs = split_docs(tokens)
    rng = random.Random(args.seed)

    # If EOT boundaries are sparse or absent, also support window sampling fallback.
    use_windows = len(docs) < max(10, args.num_samples // 2)
    decoded_docs = []

    if use_windows:
        max_start = max(0, len(tokens) - args.max_decode_tokens)
        for _ in range(args.num_samples):
            start = rng.randint(0, max_start) if max_start > 0 else 0
            chunk = tokens[start : start + args.max_decode_tokens]
            text = safe_decode(enc, chunk)
            decoded_docs.append((text, len(chunk)))
    else:
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        for idx in indices[: args.num_samples]:
            chunk = docs[idx][: args.max_decode_tokens]
            text = safe_decode(enc, chunk)
            decoded_docs.append((text, len(chunk)))

    summary = summarize_docs(decoded_docs)
    summary.update(
        {
            "data_path": args.data,
            "total_tokens": int(len(tokens)),
            "eot_split_doc_count": int(len(docs)),
            "sampling_mode": "windows" if use_windows else "documents",
            "num_samples": args.num_samples,
            "max_decode_tokens": args.max_decode_tokens,
        }
    )

    sample_txt = os.path.join(args.output_dir, "decoded_samples.txt")
    with open(sample_txt, "w", encoding="utf-8") as f:
        for i, (text, ntok) in enumerate(decoded_docs):
            f.write(f"===== SAMPLE {i:03d} | tokens={ntok} =====\n")
            f.write(text.strip() + "\n\n")

    summary_json = os.path.join(args.output_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"EOT-split docs: {summary['eot_split_doc_count']:,}")
    print(f"Sampling mode: {summary['sampling_mode']}")
    print(f"Mean chars/sample: {summary['mean_chars']:.1f}")
    print(f"Mean tokens/sample: {summary['mean_tokens']:.1f}")
    print("Domain vote counts:")
    for k, v in sorted(summary["domain_vote_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:20s} {v}")
    print("Top URL domains:")
    for dom, cnt in summary["top_url_domains"][:10]:
        print(f"  {dom:30s} {cnt}")
    print(f"Saved samples to: {sample_txt}")
    print(f"Saved summary to: {summary_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
