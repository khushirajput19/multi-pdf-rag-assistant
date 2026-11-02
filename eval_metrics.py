"""
eval_metrics.py  -  evaluation harness for the PDF RAG chatbot.

Run from the project root (same folder as app.py / config.py):

    python eval_metrics.py --validate-only            # sanity-check the dataset (no models needed)
    python eval_metrics.py                            # retrieval ablation + threshold sweep
    python eval_metrics.py --check-llm                    # test the Groq connection (1 tiny call)
    python eval_metrics.py --generate --limit 3           # + answer checks on the first 3 questions
    python eval_metrics.py --generate-only --sleep 5      # only answer checks (needs GROQ_API_KEY)

What it measures
  1. Retrieval ablation (dense / BM25 / hybrid RRF / +rerank / production) using the REAL
     chunking, MMR, BM25 tokenizer, reranker and format_docs from your code.
     Metrics: P@k, R@k, Hit@k, MRR, nDCG@k, Context keyword recall, latency (+95% bootstrap CI).
  2. Breakdown by question type (lexical / paraphrase / multi_page).
  3. Score-threshold sweep, so `score_threshold` is chosen from data (answerable vs unanswerable).
  4. (optional) Generation checks: false-refusal rate, correct abstention on unanswerable
     questions, keyword coverage and citation-page correctness.
"""
import argparse
import json
import os
import re
import threading
import time
import uuid

import numpy as np
import pandas as pd

RRF_C = 60.0
POOL = 35  # same candidate pool as hybrid_search
THRESHOLD_SWEEP = [-4.0, -3.0, -2.0, -1.2, -0.5, 0.0, 1.0, 2.0]
ABSTAIN_RE = re.compile(r"couldn.t find the answer", re.IGNORECASE)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# =============================================================================
# Dataset loading + validation (no ML models needed)
# =============================================================================
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for it in data:
        for key in ("id", "file", "question"):
            if key not in it:
                raise SystemExit(f"Dataset item {it.get('id', '?')} is missing '{key}'")
        it.setdefault("answerable", True)
        it.setdefault("type", "unknown")
        it.setdefault("target_pages", [])
        it.setdefault("answer_keywords", [])
        if it["answerable"] and not it["target_pages"]:
            raise SystemExit(f"{it['id']}: answerable question needs target_pages")
    return data


def validate_dataset(dataset, pdf_dir):
    """Checks pages exist and every keyword really occurs on at least one target page."""
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    page_text = {}
    problems = []
    for it in dataset:
        fname = it["file"]
        path = os.path.join(pdf_dir, fname)
        if not os.path.exists(path):
            problems.append(f"{it['id']}: PDF '{fname}' not found in {pdf_dir}")
            continue
        if fname not in page_text:
            page_text[fname] = {i + 1: norm(p.get_text()) for i, p in enumerate(pymupdf.open(path))}
        pages = page_text[fname]

        for p in it["target_pages"]:
            if p not in pages:
                problems.append(f"{it['id']}: target page {p} does not exist (PDF has {len(pages)} pages)")
        target_text = " ".join(pages.get(p, "") for p in it["target_pages"])
        for kw in it["answer_keywords"]:
            if norm(kw) not in target_text:
                problems.append(f"{it['id']}: keyword '{kw}' not found on target pages {it['target_pages']}")

    n_ans = sum(1 for d in dataset if d["answerable"])
    print(f"Dataset: {len(dataset)} questions ({n_ans} answerable, {len(dataset) - n_ans} unanswerable)")
    types = pd.Series([d["type"] for d in dataset]).value_counts().to_dict()
    print(f"By type: {types}")
    if problems:
        print(f"\n[!] {len(problems)} dataset problem(s):")
        for p in problems:
            print("   -", p)
    else:
        print("Validation OK: all pages exist and all keywords found on their target pages.")
    return problems


# =============================================================================
# Build the real pipeline
# =============================================================================
class LocalUpload:
    """Minimal stand-in for Streamlit's UploadedFile."""

    def __init__(self, path):
        self.name = os.path.basename(path)
        self._path = path

    def getvalue(self):
        with open(self._path, "rb") as f:
            return f.read()


def build_pipeline(pdf_dir):
    from langchain_chroma import Chroma
    from core.document_loader import process_multiple_documents
    from core.retriever import create_hybrid_retriever

    pdf_paths = sorted(
        os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")
    )
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    print(f"\nIndexing {len(pdf_paths)} PDF(s): {[os.path.basename(p) for p in pdf_paths]}")
    chunks, meta, embeddings = process_multiple_documents([LocalUpload(p) for p in pdf_paths])
    print(f"Child chunks: {len(chunks)} | per file: { {k: v['num_chunks'] for k, v in meta.items()} }")

    # In-memory Chroma: benchmark never touches ./chroma_db
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"eval_{uuid.uuid4().hex[:8]}",
        ids=[c.metadata["chunk_id"] for c in chunks],
    )
    return create_hybrid_retriever(vector_store, chunks, k=POOL)


# =============================================================================
# Retrieval strategies (ablations)
# =============================================================================
def dedupe(docs):
    seen, out = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            out.append(d)
    return out


def dense_only(q, v, b, k):
    return dedupe(v.invoke(q))[:k]


def bm25_only(q, v, b, k):
    return dedupe(b.invoke(q))[:k]


def hybrid_rrf(q, v, b, k):
    """Same RRF fusion as hybrid_search, WITHOUT the cross-encoder."""
    content_to_doc, scores = {}, {}
    for retriever in (v, b):
        seen = set()
        for rank, doc in enumerate(retriever.invoke(q), start=1):
            c = doc.page_content
            if c in seen:
                continue
            seen.add(c)
            content_to_doc.setdefault(c, doc)
            scores[c] = scores.get(c, 0.0) + 1.0 / (RRF_C + rank)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [content_to_doc[c] for c in ranked][:k]


def make_strategies(hybrid_search):
    def hybrid_rerank(q, v, b, k):
        return hybrid_search(q, v, b, top_k=k, score_threshold=float("-inf"))

    def production(q, v, b, k):
        return hybrid_search(q, v, b, top_k=k)

    return {
        "dense_only (MMR)": dense_only,
        "bm25_only": bm25_only,
        "hybrid_rrf": hybrid_rrf,
        "hybrid_rrf+rerank": hybrid_rerank,
        "production (+threshold)": production,
    }


# =============================================================================
# Metrics
# =============================================================================
def score_query(docs, targets):
    """targets: set of (file, page)."""
    retrieved = [(d.metadata.get("source_file"), d.metadata.get("page_number")) for d in docs]
    k = len(retrieved)
    hits = [1 if r in targets else 0 for r in retrieved]

    precision = sum(hits) / k if k else 0.0
    recall = len(set(retrieved) & targets) / len(targets)
    hit = 1.0 if any(hits) else 0.0
    rr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)

    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    # Relevance is per chunk (many chunks can sit on one target page), so the ideal
    # ranking is "all k results relevant". This keeps nDCG within [0, 1].
    idcg = sum(1.0 / np.log2(i + 2) for i in range(k))
    ndcg = dcg / idcg if idcg else 0.0
    return {"P@k": precision, "R@k": recall, "Hit@k": hit, "MRR": rr, "nDCG@k": ndcg,
            "retrieved": [p for _, p in retrieved]}


def keyword_recall(text, keywords):
    """Fraction of answer keywords present in `text` (None if no keywords)."""
    if not keywords:
        return None
    t = norm(text)
    return sum(1 for kw in keywords if norm(kw) in t) / len(keywords)


def bootstrap_ci(values, n=2000, seed=42):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# =============================================================================
# 1 + 2. Retrieval ablation
# =============================================================================
def run_retrieval_eval(answerable, v, b, k, strategies, format_docs):
    rows, per_query = [], []
    metric_names = ("P@k", "R@k", "Hit@k", "MRR", "nDCG@k", "CtxKW")

    for name, fn in strategies.items():
        vals = {m: [] for m in metric_names}
        latencies, n_returned = [], []
        for it in answerable:
            targets = {(it["file"], p) for p in it["target_pages"]}
            t0 = time.perf_counter()
            docs = fn(it["question"], v, b, k)
            latencies.append(time.perf_counter() - t0)
            n_returned.append(len(docs))

            res = score_query(docs, targets)
            ctx_kw = keyword_recall(format_docs(docs), it["answer_keywords"])
            res["CtxKW"] = ctx_kw if ctx_kw is not None else np.nan
            for m in metric_names:
                vals[m].append(res[m])
            per_query.append({
                "strategy": name, "id": it["id"], "type": it["type"],
                "target": sorted(it["target_pages"]), "retrieved_pages": res["retrieved"],
                **{m: round(res[m], 3) if not np.isnan(res[m]) else None for m in metric_names},
            })

        row = {"Strategy": name}
        for m in metric_names:
            arr = np.array(vals[m], dtype=float)
            arr = arr[~np.isnan(arr)]
            lo, hi = bootstrap_ci(arr)
            row[m] = f"{arr.mean():.3f} [{lo:.2f}-{hi:.2f}]"
        row["avg chunks"] = f"{np.mean(n_returned):.2f}"
        row["Latency(s)"] = f"{np.mean(latencies):.3f}"
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(per_query)


# =============================================================================
# 3. Threshold sweep
# =============================================================================
def threshold_analysis(dataset, v, b, k, get_reranker):
    reranker = get_reranker()
    cache = {}
    for it in dataset:
        cands = hybrid_rrf(it["question"], v, b, POOL)
        if not cands:
            cache[it["id"]] = []
            continue
        scores = reranker.predict([[it["question"], d.page_content] for d in cands])
        cache[it["id"]] = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)

    ans = [d for d in dataset if d["answerable"] and cache[d["id"]]]
    una = [d for d in dataset if not d["answerable"] and cache[d["id"]]]
    top1 = lambda d: float(cache[d["id"]][0][1])

    print("\nReranker top-1 score distribution")
    if ans:
        a = [top1(d) for d in ans]
        print(f"  answerable   : median {np.median(a):+.2f}  min {min(a):+.2f}  max {max(a):+.2f}")
    if una:
        u = [top1(d) for d in una]
        print(f"  unanswerable : median {np.median(u):+.2f}  min {min(u):+.2f}  max {max(u):+.2f}")

    rows = []
    for t in THRESHOLD_SWEEP:
        hits, ndocs, cov = [], [], []
        for d in ans:
            scored = cache[d["id"]]
            kept = [doc for doc, s in scored if s >= t] or [scored[0][0]]
            kept = kept[:k]
            targets = {(d["file"], p) for p in d["target_pages"]}
            res = score_query(kept, targets)
            hits.append(res["Hit@k"])
            ndocs.append(len(kept))
        row = {
            "threshold": t,
            "Hit@k (answerable)": f"{np.mean(hits):.3f}" if hits else "-",
            "avg chunks kept": f"{np.mean(ndocs):.2f}" if ndocs else "-",
            "answerable top-1 below t": f"{np.mean([top1(d) < t for d in ans]) * 100:.0f}%" if ans else "-",
            "unanswerable top-1 below t": f"{np.mean([top1(d) < t for d in una]) * 100:.0f}%" if una else "-",
        }
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 4. Generation checks (optional)
# =============================================================================
def parse_cited_pages(answer):
    """Extract page numbers from citations like '(source: X.pdf, p. 4)', 'pp. 4-5', 'Page 7'."""
    pages = set()
    pattern = r"\b(?:pp?\.|pages?)\s*(\d+(?:\s*(?:[-\u2013,&]|and)\s*\d+)*)"
    for m in re.finditer(pattern, answer, flags=re.IGNORECASE):
        seg = m.group(1)
        nums = [int(x) for x in re.findall(r"\d+", seg)]
        if re.search(r"[-\u2013]", seg) and len(nums) == 2 and nums[0] <= nums[1]:
            nums = list(range(nums[0], nums[1] + 1))
        pages.update(nums)
    return pages


def evaluate_answer(item, answer):
    abstained = bool(ABSTAIN_RE.search(answer))
    out = {"id": item["id"], "type": item["type"], "answerable": item["answerable"], "abstained": abstained}
    if item["answerable"]:
        cited = parse_cited_pages(answer)
        out["kw_coverage"] = keyword_recall(answer, item["answer_keywords"])
        out["has_citation"] = bool(cited)
        out["citation_correct"] = bool(cited & set(item["target_pages"]))
        out["cited_pages"] = sorted(cited)
    return out


def _call_with_timeout(fn, timeout):
    """Run fn() in a daemon thread; raise TimeoutError if it does not finish in time."""
    box = {}

    def runner():
        try:
            box["value"] = fn()
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"no response within {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _load_llm():
    from core.rag_engine import get_llm
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set (check your .env file); cannot use the LLM")
    return get_llm(key)


def check_llm(timeout):
    """Single tiny call to see whether the Groq key, model name and network work."""
    llm = _load_llm()
    print(f"Calling the LLM once (timeout {timeout}s) ...")
    t0 = time.time()
    try:
        out = _call_with_timeout(lambda: llm.invoke("Reply with the single word: ok").content, timeout)
        print(f"OK in {time.time() - t0:.1f}s. Model replied: {out[:80]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {str(e)[:300]}")


def run_generation_eval(dataset, v, b, sleep, timeout=90, limit=None):
    from core.rag_engine import RAGChatbot

    bot = RAGChatbot(v, b, _load_llm())
    items = dataset[:limit] if limit else dataset
    results, consecutive_fail = [], 0
    print(f"Running {len(items)} questions (timeout {timeout}s each, {sleep}s pause between calls).")
    print("Press Ctrl+C to stop early; partial results will still be summarised.\n")
    try:
        for n, it in enumerate(items, start=1):
            t0 = time.time()
            try:
                answer = _call_with_timeout(
                    lambda: "".join(bot.stream(it["question"], [], "")[0]), timeout)
            except Exception as e:  # rate limits, bad model name, network, timeout
                consecutive_fail += 1
                print(f"  [{n}/{len(items)}] {it['id']} FAILED after {time.time() - t0:.1f}s: "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                if consecutive_fail >= 3:
                    print("\n3 failures in a row: check GROQ_API_KEY, the model name in "
                          "core/rag_engine.py (get_llm) and your network. Try --check-llm.")
                    break
                time.sleep(sleep)
                continue
            consecutive_fail = 0
            res = evaluate_answer(it, answer)
            results.append(res)
            tag = "abstained" if res["abstained"] else "answered"
            print(f"  [{n}/{len(items)}] {it['id']} {tag} in {time.time() - t0:.1f}s", flush=True)
            time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nInterrupted by user; summarising partial results.")
    if not results:
        return "No answers were collected (see the errors above).", pd.DataFrame()
    return summarize_generation(results), pd.DataFrame(results)


def summarize_generation(results):
    ans = [r for r in results if r["answerable"]]
    una = [r for r in results if not r["answerable"]]
    lines = []
    if ans:
        kw = [r["kw_coverage"] for r in ans if r["kw_coverage"] is not None]
        lines += [
            f"Answerable (n={len(ans)})",
            f"  false-refusal rate        : {np.mean([r['abstained'] for r in ans]) * 100:.1f}%",
            f"  keyword coverage in answer: {np.mean(kw):.3f}  (proxy for completeness)",
            f"  answers with a citation   : {np.mean([r['has_citation'] for r in ans]) * 100:.1f}%",
            f"  citation page correct     : {np.mean([r['citation_correct'] for r in ans]) * 100:.1f}%",
        ]
    if una:
        lines += [
            f"Unanswerable (n={len(una)})",
            f"  correct abstention rate   : {np.mean([r['abstained'] for r in una]) * 100:.1f}%"
            f"  (hallucination rate = {100 - np.mean([r['abstained'] for r in una]) * 100:.1f}%)",
        ]
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="./benchmark_pdfs")
    ap.add_argument("--dataset", default="eval_dataset.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="eval_results.csv")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--generate", action="store_true", help="also run LLM answer checks (uses Groq)")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between LLM calls (rate limits)")
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for one LLM answer")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions in --generate mode")
    ap.add_argument("--generate-only", action="store_true", help="skip retrieval tables, run only the answer checks")
    ap.add_argument("--check-llm", action="store_true", help="test the Groq connection with one tiny call and exit")
    args = ap.parse_args()

    if args.check_llm:
        check_llm(args.timeout)
        return

    dataset = load_dataset(args.dataset)
    problems = validate_dataset(dataset, args.pdf_dir)
    if args.validate_only:
        raise SystemExit(1 if problems else 0)
    if problems:
        print("\nFix the dataset problems above (or ignore them knowingly) before trusting results.\n")

    v, b = build_pipeline(args.pdf_dir)

    if args.generate_only:
        print("\n" + "=" * 110 + "\nGENERATION CHECKS (production pipeline, single-turn)\n" + "=" * 110)
        text, df = run_generation_eval(dataset, v, b, args.sleep, args.timeout, args.limit)
        print("\n" + text)
        if not df.empty:
            df.to_csv(args.out.replace(".csv", "_generation.csv"), index=False)
        return

    from core.retriever import hybrid_search, format_docs, get_reranker
    strategies = make_strategies(hybrid_search)
    answerable = [d for d in dataset if d["answerable"]]

    hybrid_search("warm up", v, b, top_k=args.k)  # exclude model load from latency

    summary, per_query = run_retrieval_eval(answerable, v, b, args.k, strategies, format_docs)
    print("\n" + "=" * 110)
    print(f"RETRIEVAL ABLATION  (mean [95% bootstrap CI], n={len(answerable)} answerable, k={args.k})")
    print("CtxKW = fraction of answer keywords present in the final context sent to the LLM")
    print("Note: P@k is only comparable between strategies returning the same number of chunks (see 'avg chunks').")
    print("=" * 110)
    print(summary.to_string(index=False))

    print("\nHit@k / MRR by question type")
    by_type = per_query.groupby(["strategy", "type"])[["Hit@k", "MRR"]].mean().round(3).unstack("type")
    print(by_type.to_string())

    print("\nProduction misses (Hit@k = 0):")
    miss = per_query[(per_query.strategy == "production (+threshold)") & (per_query["Hit@k"] == 0)]
    print(miss[["id", "type", "target", "retrieved_pages"]].to_string(index=False) if not miss.empty else "  none")

    sweep = threshold_analysis(dataset, v, b, args.k, get_reranker)
    print("\nSCORE-THRESHOLD SWEEP (current default = -1.2)")
    print(sweep.to_string(index=False))

    per_query.to_csv(args.out, index=False)
    summary.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    print(f"\nSaved {args.out} and {args.out.replace('.csv', '_summary.csv')}")

    if args.generate:
        print("\n" + "=" * 110 + "\nGENERATION CHECKS (production pipeline, single-turn)\n" + "=" * 110)
        text, df = run_generation_eval(dataset, v, b, args.sleep, args.timeout, args.limit)
        print("\n" + text)
        if not df.empty:
            df.to_csv(args.out.replace(".csv", "_generation.csv"), index=False)

    if len(answerable) < 30:
        print(f"\n[!] Only {len(answerable)} answerable questions: CIs are wide.")


if __name__ == "__main__":
    main()