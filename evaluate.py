"""
AURA-ED Evaluation Script
Computes AUROC, F1, and accuracy for:
  1. Clinical risk scores (MEWS, NEWS, NEWS2, REMS, CART, CCI) — runs immediately
  2. LLM-generated risk tiers — requires GEMINI_API_KEY in .env or Ollama running

Usage:
  python evaluate.py                          # score-based eval only
  python evaluate.py --llm gemini             # score + LLM eval (needs GEMINI_API_KEY)
  python evaluate.py --llm ollama             # score + LLM eval (needs Ollama running)
  python evaluate.py --llm ollama --n 300     # limit LLM eval to N patients
  python evaluate.py --split train            # evaluate on train set instead
"""

import os, sys, re, argparse, traceback
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

TEST_PATH  = os.path.join(os.path.dirname(__file__), "dataset", "test_master_dataset.csv")
TRAIN_PATH = os.path.join(os.path.dirname(__file__), "dataset", "train_master_dataset.csv")

# Primary outcomes of clinical interest
PRIMARY_OUTCOMES = [
    "outcome_hospitalization",
    "outcome_critical",
    "outcome_icu_transfer_12h",
    "outcome_sepsis",
    "outcome_aki",
    "outcome_acs_mi",
    "outcome_stroke",
    "outcome_ahf",
    "outcome_pneumonia_all",
    "outcome_pe",
]

# Risk score columns and their "high-risk" binary thresholds
SCORE_COLS = {
    "score_MEWS":  3,   # ≥3 = elevated risk
    "score_NEWS":  5,   # ≥5 = elevated risk
    "score_NEWS2": 5,   # ≥5 = escalate care
    "score_REMS":  8,   # ≥8 = elevated risk
    "score_CART":  5,   # ≥5 = elevated risk
    "score_CCI":   2,   # ≥2 = elevated comorbidity burden
}

# LLM tier → numeric score mapping
TIER_SCORE = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

# Max retries when LLM returns no parseable risk tier
MAX_RETRIES = 2

# Default model names per provider
DEFAULT_MODELS = {
    "gemini": "gemini-3-flash-preview",
    "ollama": "gemma3:12b",
}


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load a patient dataset CSV and cast stay_id to string.

    Args:
        path: Absolute path to the CSV file.

    Returns:
        DataFrame with stay_id as string type.
    """
    df = pd.read_csv(path, low_memory=False)
    df["stay_id"] = df["stay_id"].astype(str)
    return df


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_score_cont, threshold: float) -> dict:
    """Compute AUROC, F1, and accuracy from a continuous score and a binary threshold.

    Args:
        y_true:        Array-like of binary ground-truth labels (0/1).
        y_score_cont:  Array-like of continuous scores (NaN values are dropped).
        threshold:     Score value above which a prediction is treated as positive.

    Returns:
        Dict with keys: n, prevalence, auroc, f1, accuracy.
        If only one class is present, auroc/f1/accuracy are None.
    """
    valid    = ~np.isnan(y_score_cont)
    y_true_v = np.array(y_true)[valid]
    y_score_v = np.array(y_score_cont)[valid]
    y_pred_v  = (y_score_v >= threshold).astype(int)

    if len(np.unique(y_true_v)) < 2:
        return {
            "n": int(valid.sum()), "auroc": None, "f1": None, "accuracy": None,
            "prevalence": float(y_true_v.mean()), "note": "Only one class present",
        }

    return {
        "n":          int(valid.sum()),
        "prevalence": round(float(y_true_v.mean()), 4),
        "auroc":      round(roc_auc_score(y_true_v, y_score_v), 4),
        "f1":         round(f1_score(y_true_v, y_pred_v, zero_division=0), 4),
        "accuracy":   round(accuracy_score(y_true_v, y_pred_v), 4),
    }


# ── Display ────────────────────────────────────────────────────────────────────

def print_table(title: str, rows: list[dict], columns: list[str]):
    """Print a formatted metrics table to stdout.

    Args:
        title:   Header line printed above the table.
        rows:    List of dicts, each with a 'label' key plus one key per column.
        columns: Ordered list of column names to display.
    """
    col_w  = max(max(len(c) for c in columns), 6)
    row_w  = max(len(r["label"]) for r in rows)
    header = f"{'':>{row_w}}  " + "  ".join(f"{c:>{col_w}}" for c in columns)

    print(f"\n{'─'*len(header)}")
    print(title)
    print('─'*len(header))
    print(header)
    print('─'*len(header))
    for r in rows:
        vals = "  ".join(
            f"{r.get(c, '—'):>{col_w}}"         if isinstance(r.get(c), str)
            else f"{r.get(c):>{col_w}.4f}"      if r.get(c) is not None
            else f"{'—':>{col_w}}"
            for c in columns
        )
        print(f"{r['label']:>{row_w}}  {vals}")
    print('─'*len(header))


# ── Score-based evaluation ─────────────────────────────────────────────────────

def eval_scores(df: pd.DataFrame, split_name: str):
    """Evaluate all traditional clinical scores against every primary outcome.

    Iterates over SCORE_COLS, binarises each score at its threshold, and prints
    AUROC / F1 / accuracy / prevalence for every outcome in PRIMARY_OUTCOMES.

    Args:
        df:          Patient dataset (full split).
        split_name:  Label used in the section header (e.g. 'TEST SET').
    """
    print(f"\n{'='*70}")
    print(f"SCORE-BASED EVALUATION  —  {split_name}  ({len(df):,} patients)")
    print(f"{'='*70}")

    for score_col, threshold in SCORE_COLS.items():
        if score_col not in df.columns:
            print(f"\n[skip] {score_col} not in dataset")
            continue

        rows = []
        for outcome in PRIMARY_OUTCOMES:
            if outcome not in df.columns:
                continue
            m = compute_metrics(
                df[outcome].fillna(0).astype(int),
                df[score_col].values,
                threshold,
            )
            rows.append({
                "label":      outcome.replace("outcome_", ""),
                "prevalence": m["prevalence"],
                "auroc":      m["auroc"],
                "f1":         m["f1"],
                "accuracy":   m["accuracy"],
            })

        print_table(
            f"{score_col}  (binary threshold ≥ {threshold})",
            rows,
            ["auroc", "f1", "accuracy", "prevalence"],
        )


# ── LLM evaluation ─────────────────────────────────────────────────────────────

def parse_risk_tier(brief: str) -> str | None:
    """Extract the risk tier keyword from an AURA brief.

    First looks for the tier inside the '## Overall Risk Assessment' section;
    falls back to the first tier keyword anywhere in the text.

    Args:
        brief: Raw text output from the LLM.

    Returns:
        One of 'LOW', 'MODERATE', 'HIGH', 'CRITICAL', or None if not found.
    """
    match = re.search(
        r"##\s*Overall Risk Assessment[^\n]*\n[^\n]*\b(CRITICAL|HIGH|MODERATE|LOW)\b",
        brief, re.IGNORECASE,
    )
    if not match:
        match = re.search(r"\b(CRITICAL|HIGH|MODERATE|LOW)\b", brief, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _build_llm_caller(provider: str, model: str):
    """Construct a provider-specific callable that sends a prompt and returns text.

    Args:
        provider: 'gemini' or 'ollama'.
        model:    Model identifier string.

    Returns:
        A callable call_llm(prompt: str) -> str.
    """
    if provider == "gemini":
        from google import genai
        import time
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")
        client = genai.Client(api_key=api_key)

        def call_llm(prompt):
            wait = 20
            for attempt in range(5):
                try:
                    return client.models.generate_content(model=model, contents=prompt).text
                except Exception as e:
                    code = getattr(e, "status_code", None) or getattr(e, "code", None)
                    if attempt < 4 and code in (429, 503):
                        print(f"\n  [Gemini {code} — retrying in {wait}s…]")
                        time.sleep(wait)
                        wait = min(wait * 2, 120)
                    else:
                        raise

    else:  # ollama
        import ollama as ollama_lib
        client = ollama_lib.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

        def call_llm(prompt):
            return client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            ).message.content

    return call_llm


_aura_app_cache = None

def _load_aura_app():
    """Load the AURA-ED Streamlit app as a plain module, cached after first call.

    Returns:
        The loaded app module with extract_patient_summary and build_prompt available.
    """
    global _aura_app_cache
    if _aura_app_cache is None:
        import importlib.util, unittest.mock as mock
        spec = importlib.util.spec_from_file_location(
            "aura_app", os.path.join(os.path.dirname(__file__), "AURA-ED.py")
        )
        with mock.patch.dict(sys.modules, {"streamlit": mock.MagicMock()}):
            _aura_app_cache = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_aura_app_cache)
    return _aura_app_cache


def eval_llm(df: pd.DataFrame, provider: str, model: str, n: int, split_name: str):
    """Evaluate an LLM by parsing its risk tier predictions against ground-truth outcomes.

    Samples n patients, sends each through the AURA-ED prompt, parses the assigned
    risk tier (LOW/MODERATE/HIGH/CRITICAL), and computes AUROC / F1 / accuracy for
    every outcome in PRIMARY_OUTCOMES using HIGH/CRITICAL as the positive class.

    Args:
        df:          Patient dataset to sample from.
        provider:    'gemini' or 'ollama'.
        model:       Model identifier (e.g. 'gemma3:12b', 'gemini-3-flash-preview').
        n:           Number of patients to evaluate.
        split_name:  Label used in the section header (e.g. 'TEST SET').
    """
    app = _load_aura_app()

    try:
        call_llm = _build_llm_caller(provider, model)
    except EnvironmentError as e:
        print(f"\n[ERROR] {e} — skipping LLM eval")
        return

    sample = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    print(f"\n{'='*70}")
    print(f"LLM-BASED EVALUATION  —  {split_name}  ({provider}/{model}, n={len(sample)})")
    print(f"{'='*70}")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Build prompts upfront (CPU-bound, fast)
    rows_data = []
    for _, row in sample.iterrows():
        summary = app.extract_patient_summary(row)
        summary["outcomes"] = {}   # blind eval — strip ground truth
        rows_data.append((app.build_prompt(summary), row))

    def process_row(args):
        prompt, row = args
        tier = None
        for _ in range(MAX_RETRIES + 1):
            tier = parse_risk_tier(call_llm(prompt))
            if tier is not None:
                break
        return tier, row

    tiers, true_labels = [], {o: [] for o in PRIMARY_OUTCOMES}
    errors, done = 0, 0

    # Parallelise LLM calls — Ollama handles concurrent requests
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process_row, rd): rd for rd in rows_data}
        for future in as_completed(futures):
            try:
                tier, row = future.result()
                tiers.append(TIER_SCORE.get(tier, np.nan) if tier else np.nan)
                for o in PRIMARY_OUTCOMES:
                    v = row.get(o)
                    true_labels[o].append(int(v) if pd.notna(v) else 0)
            except Exception as e:
                if errors == 0:
                    print(f"\n  [ERROR] {e}")
                    traceback.print_exc()
                errors += 1
            finally:
                done += 1
                if done % 10 == 0:
                    print(f"  {done}/{len(sample)} completed …")

    if errors:
        print(f"  [{errors}/{len(sample)} rows skipped due to errors — excluded from metrics]")

    y_score = np.array(tiers, dtype=float)
    rows = []
    for outcome in PRIMARY_OUTCOMES:
        if outcome not in df.columns:
            continue
        m = compute_metrics(np.array(true_labels[outcome]), y_score, threshold=2.0)
        rows.append({
            "label":      outcome.replace("outcome_", ""),
            "prevalence": m["prevalence"],
            "auroc":      m["auroc"],
            "f1":         m["f1"],
            "accuracy":   m["accuracy"],
        })

    print_table(
        "LLM risk tier  (HIGH/CRITICAL = positive, threshold ≥ 2)",
        rows,
        ["auroc", "f1", "accuracy", "prevalence"],
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    """Parse CLI arguments and run score-based and/or LLM-based evaluation."""
    parser = argparse.ArgumentParser(description="AURA-ED evaluation")
    parser.add_argument("--llm",   choices=["gemini", "ollama"], default=None,
                        help="Enable LLM-based evaluation")
    parser.add_argument("--model", default=None,
                        help="Model name (default: gemini-3-flash-preview / gemma3:12b)")
    parser.add_argument("--n",     type=int, default=100,
                        help="Number of test patients for LLM eval (default: 100)")
    parser.add_argument("--split", choices=["test", "train", "both"], default="test",
                        help="Which dataset split to evaluate (default: test)")
    args = parser.parse_args()

    splits = []
    if args.split in ("test", "both"):
        splits.append((TEST_PATH, "TEST SET"))
    if args.split in ("train", "both"):
        splits.append((TRAIN_PATH, "TRAIN SET"))

    for path, name in splits:
        print(f"\nLoading {name} from {path} …")
        df = load_data(path)
        eval_scores(df, name)

        if args.llm:
            model = args.model or DEFAULT_MODELS[args.llm]
            eval_llm(df, args.llm, model, args.n, name)

    print("\nDone.")


if __name__ == "__main__":
    main()
