"""
AURA-ED — AI-powered Utilization & Risk Analysis in the Emergency Department
Converts raw ED clinical data into an executive-level patient risk brief.
"""

import os
import sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import ollama
from google import genai

load_dotenv()

# Configure Ollama client with host from .env (falls back to localhost default)
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_ollama_client = ollama.Client(host=_OLLAMA_HOST)

# Configure Gemini client
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_gemini_client = genai.Client(api_key=_GEMINI_API_KEY) if _GEMINI_API_KEY else None

sys.path.insert(0, os.path.dirname(__file__))
from guardrails.ClinicalReviewGuardrail import clinicalReviewGuardrail
from guardrails.FairnessGuardrail import FairnessGuardrail
from guardrails.RAGGuardrail import RAGGuardrail

st.set_page_config(
    page_title="AURA-ED",
    page_icon="🩺",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "master_dataset.csv")

VITAL_REFS = {
    "triage_temperature":  ("Temperature (°C)",   35.0,  38.5),
    "triage_heartrate":    ("Heart Rate (bpm)",    60,    100),
    "triage_resprate":     ("Resp Rate (/min)",    12,    20),
    "triage_o2sat":        ("O₂ Saturation (%)",   95,    100),
    "triage_sbp":          ("Systolic BP (mmHg)",  90,    140),
    "triage_dbp":          ("Diastolic BP (mmHg)", 60,    90),
    "triage_MAP":          ("MAP (mmHg)",           70,    100),
}

KEY_LABS = [
    "CREATININE", "BLOOD UREA NITROGEN (BUN)", "GLUCOSE",
    "LACTATE", "POC:LACTATE, ISTAT",
    "HIGH SENSITIVITY TROPONIN", "TROPONIN I",
    "BNP, NT-PRO", "WHITE BLOOD CELLS (WBC)", "HEMOGLOBIN",
    "PLATELET COUNT (PLT)", "ANION GAP", "CO2",
    "SODIUM", "Sodium (Combined)", "Potassium (Combined)",
    "INR", "ALBUMIN", "BILIRUBIN, TOTAL",
]

CCI_MAP = {
    "cci_MI": "Myocardial Infarction", "cci_CHF": "Congestive Heart Failure",
    "cci_PVD": "Peripheral Vascular Disease", "cci_Stroke": "Stroke/TIA",
    "cci_Dementia": "Dementia", "cci_Pulmonary": "COPD",
    "cci_Rheumatic": "Rheumatic Disease", "cci_PUD": "Peptic Ulcer Disease",
    "cci_Liver1": "Mild Liver Disease", "cci_DM1": "Diabetes (uncomplicated)",
    "cci_DM2": "Diabetes (end-organ damage)", "cci_Paralysis": "Hemiplegia/Paraplegia",
    "cci_Renal": "Chronic Kidney Disease", "cci_Cancer1": "Cancer (non-metastatic)",
    "cci_Liver2": "Moderate/Severe Liver Disease", "cci_Cancer2": "Metastatic Cancer",
    "cci_HIV": "HIV/AIDS",
}

ECI_MAP = {
    "eci_Arrhythmia": "Cardiac Arrhythmia", "eci_CHF": "Heart Failure",
    "eci_Coagulopathy": "Coagulopathy", "eci_FluidsLytes": "Fluid/Electrolyte Disorders",
    "eci_HTN1": "Hypertension (uncomplicated)", "eci_HTN2": "Hypertension (complicated)",
    "eci_Anemia": "Anemia", "eci_DM1": "Diabetes (uncomplicated)",
    "eci_DM2": "Diabetes (complicated)", "eci_Obesity": "Obesity",
    "eci_Renal": "Renal Failure", "eci_Liver": "Liver Disease",
    "eci_Pulmonary": "Pulmonary Circulation Disorders",
    "eci_Depression": "Depression", "eci_PHTN": "Pulmonary Hypertension",
    "eci_Tumor1": "Solid Tumor (no metastasis)", "eci_Tumor2": "Lymphoma",
    "eci_WeightLoss": "Weight Loss", "eci_Alcohol": "Alcohol Abuse",
    "eci_Drugs": "Drug Abuse",
}

OUTCOME_LABELS = {
    "outcome_sepsis": "Sepsis", "outcome_aki": "Acute Kidney Injury",
    "outcome_acs_mi": "ACS / MI", "outcome_stroke": "Stroke",
    "outcome_ards": "ARDS", "outcome_pe": "Pulmonary Embolism",
    "outcome_pneumonia_all": "Pneumonia", "outcome_ahf": "Acute Heart Failure",
    "outcome_copd_exac": "COPD Exacerbation",
    "outcome_hospitalization": "Hospitalized", "outcome_critical": "Critical Care",
    "outcome_icu_transfer_12h": "ICU Transfer (12h)",
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["stay_id"] = df["stay_id"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    return df


# ── Helper: extract structured patient summary ────────────────────────────────
def extract_patient_summary(row: pd.Series) -> dict:
    """Pull the key clinical signals from a single row."""

    # --- Vitals & flags
    vitals = {}
    for col, (label, low, high) in VITAL_REFS.items():
        val = row.get(col)
        if pd.notna(val):
            flag = "NORMAL"
            if val < low:
                flag = "LOW"
            elif val > high:
                flag = "HIGH"
            vitals[label] = {"value": round(float(val), 1), "flag": flag}

    # --- Risk scores
    scores = {}
    score_cols = {
        "score_MEWS": "MEWS", "score_NEWS": "NEWS", "score_NEWS2": "NEWS2",
        "score_REMS": "REMS", "score_CART": "CART", "score_CCI": "CCI (Charlson)",
    }
    for col, label in score_cols.items():
        val = row.get(col)
        if pd.notna(val):
            scores[label] = round(float(val), 1)

    # --- Key labs (non-null only)
    labs = {}
    for lab in KEY_LABS:
        val = row.get(lab)
        if pd.notna(val):
            labs[lab] = round(float(val), 2) if isinstance(val, float) else val

    # --- Active comorbidities
    comorbidities = []
    for col, name in {**CCI_MAP, **ECI_MAP}.items():
        val = row.get(col)
        if pd.notna(val) and int(val) == 1:
            comorbidities.append(name)
    comorbidities = sorted(set(comorbidities))

    # --- Visit history
    history = {}
    for col in ["n_ed_30d", "n_ed_90d", "n_hosp_30d", "n_hosp_90d",
                "n_icu_30d", "n_icu_90d", "n_med"]:
        val = row.get(col)
        if pd.notna(val):
            history[col] = int(val)

    # --- Known outcomes (ground truth, for context in demo)
    outcomes = {}
    for col, label in OUTCOME_LABELS.items():
        val = row.get(col)
        if pd.notna(val) and int(val) == 1:
            outcomes[label] = True

    return {
        "demographics": {
            "age": row.get("age"), "gender": row.get("gender"),
            "race": row.get("race"), "acuity_level": row.get("triage_acuity"),
        },
        "chief_complaint": row.get("CC", "Not recorded"),
        "vitals": vitals,
        "risk_scores": scores,
        "labs": labs,
        "comorbidities": comorbidities,
        "visit_history": history,
        "outcomes": outcomes,
    }


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(summary: dict) -> str:
    d = summary["demographics"]
    vitals_text = "\n".join(
        f"  • {k}: {v['value']}  [{v['flag']}]"
        for k, v in summary["vitals"].items()
    ) or "  Not available"

    scores_text = "\n".join(
        f"  • {k}: {v}" for k, v in summary["risk_scores"].items()
    ) or "  Not available"

    labs_text = "\n".join(
        f"  • {k}: {v}" for k, v in summary["labs"].items()
    ) or "  Not available"

    comorbidities_text = (
        "\n".join(f"  • {c}" for c in summary["comorbidities"])
        or "  None documented"
    )

    h = summary["visit_history"]
    history_text = (
        f"  ED visits (30d/90d): {h.get('n_ed_30d',0)} / {h.get('n_ed_90d',0)}\n"
        f"  Hospitalizations (30d/90d): {h.get('n_hosp_30d',0)} / {h.get('n_hosp_90d',0)}\n"
        f"  ICU stays (30d/90d): {h.get('n_icu_30d',0)} / {h.get('n_icu_90d',0)}\n"
        f"  Active medications on reconciliation: {h.get('n_med','?')}"
    )

    outcomes_text = (
        ", ".join(summary["outcomes"].keys()) if summary["outcomes"]
        else "None flagged"
    )

    return f"""You are a clinical decision-support AI generating an executive-level "Early Risk Profile" brief for a physician reviewing an Emergency Department case.

PATIENT SNAPSHOT
Age / Sex: {d.get('age')} yo {d.get('gender', '?')}
Race/Ethnicity: {d.get('race', 'Unknown')}
Triage Acuity (ESI): {d.get('acuity_level', '?')}
Chief Complaint: {summary['chief_complaint']}

TRIAGE VITALS
{vitals_text}

RISK SCORES
{scores_text}

LABORATORY RESULTS (Available)
{labs_text}

ACTIVE COMORBIDITIES (CCI / ECI)
{comorbidities_text}

UTILIZATION HISTORY
{history_text}

DOCUMENTED OUTCOMES (Ground-truth, for context)
{outcomes_text}

TASK
Generate a concise, executive-level "Early Risk Profile" brief. Structure your output EXACTLY as follows:

## Overall Risk Assessment
One-line summary: assign a risk tier (LOW / MODERATE / HIGH / CRITICAL) and a single-sentence rationale.

## Key Drivers of Concern
Bullet list of the 3–6 most clinically significant findings, ranked by importance. Each bullet should state the finding AND its clinical implication. Be specific — cite actual values.

## Narrative Summary
2–3 sentences a senior physician could read in under 30 seconds. Integrate demographics, chief complaint, vitals, scores, and comorbidities into a coherent story.

## Recommended Watch-Points
Bullet list of 2–4 things the clinical team should monitor or act on in the next 1–4 hours.

## Confidence Note
One sentence on any data gaps or limitations that may affect this assessment.

Tone: clinical, precise, direct. No unnecessary hedging. Assume the reader is a physician.
"""


def generate_brief(summary: dict, provider: str, model: str) -> str:
    prompt = build_prompt(summary)
    if provider == "Gemini":
        response = _gemini_client.models.generate_content(model=model, contents=prompt)
        return response.text
    else:
        response = _ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content


def run_guardrails(brief: str, summary: dict, row: pd.Series) -> dict:
    """Run all three guardrails against the generated AURA brief.

    Returns a dict with each guardrail's result keyed by name.
    """
    scores = summary["risk_scores"]
    vitals = summary["vitals"]

    # Build truth dict for ClinicalReviewGuardrail
    truth = {
        "HR":   vitals.get("Heart Rate (bpm)", {}).get("value"),
        "RR":   vitals.get("Resp Rate (/min)", {}).get("value"),
        "Temp": vitals.get("Temperature (°C)", {}).get("value"),
        "SBP":  vitals.get("Systolic BP (mmHg)", {}).get("value"),
        "SpO2": vitals.get("O₂ Saturation (%)", {}).get("value"),
        "MEWS": scores.get("MEWS"),
    }
    truth = {k: v for k, v in truth.items() if v is not None}

    # Derive age group for FairnessGuardrail
    age = summary["demographics"].get("age")
    if age is not None:
        age = int(age)
        if age < 18:
            age_group = "age_under_18"
        elif age <= 30:
            age_group = "age_18-30"
        elif age <= 50:
            age_group = "age_31-50"
        elif age <= 70:
            age_group = "age_51-70"
        else:
            age_group = "age_71-100"
    else:
        age_group = None

    clinical_result = clinicalReviewGuardrail().auto_score(brief, truth)

    fairness_guardrail = FairnessGuardrail()
    fairness_result = (
        fairness_guardrail.check_pediatric(brief, age)
        if age is not None and age < 18
        else {"applicable": False, "note": "Adult patient — pediatric check skipped"}
    )

    rag_result = RAGGuardrail().validate_grounding(brief, patient_row=row)

    return {
        "clinical_review": clinical_result,
        "fairness":        fairness_result,
        "rag_grounding":   rag_result,
    }


def flag_color(flag: str) -> str:
    return {"HIGH": "🔴", "LOW": "🔵", "NORMAL": "🟢"}.get(flag, "⚪")


def render_vitals_table(vitals: dict):
    rows = [
        {"Vital Sign": k, "Value": v["value"], "Status": flag_color(v["flag"]) + " " + v["flag"]}
        for k, v in vitals.items()
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Vital Sign"), use_container_width=True)
    else:
        st.info("No triage vitals available.")


def render_scores(scores: dict):
    if not scores:
        st.info("No risk scores available.")
        return
    cols = st.columns(len(scores))
    thresholds = {
        "MEWS": (3, 5), "NEWS": (5, 9), "NEWS2": (5, 9),
        "REMS": (8, 12), "CART": (5, 9),
    }
    for col, (label, val) in zip(cols, scores.items()):
        short = label.split()[0]
        lo, hi = thresholds.get(short, (4, 8))
        color = "normal" if val < lo else ("off" if val < hi else "inverse")
        col.metric(label=label, value=val, delta=None)


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.title("🩺 AURA-ED")
    st.caption("AI-powered Utilization & Risk Analysis in the Emergency Department")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        provider = st.radio("Model Provider", ["Ollama", "Gemini"], horizontal=True)

        if provider == "Ollama":
            active_model = st.text_input(
                "Ollama Model",
                value=os.getenv("OLLAMA_MODEL", "llama3.2"),
                placeholder="e.g. llama3.2, mistral, gemma3",
                help="Run `ollama pull <model>` to download a model.",
            )
            if st.button("Pull Model", use_container_width=True):
                with st.spinner(f"Pulling {active_model}… this may take a few minutes."):
                    try:
                        _ollama_client.pull(active_model.strip())
                        st.success(f"'{active_model}' pulled successfully.")
                    except Exception as e:
                        st.error(f"Pull failed: {e}")
        else:
            active_model = st.selectbox(
                "Gemini Model",
                [
                    "gemini-3-flash-preview",
                ],
            )
            if not _GEMINI_API_KEY:
                st.warning("GEMINI_API_KEY not set in .env")
        st.divider()
        st.header("Patient Selection")

        df = load_data()

        # Search by stay_id
        search_id = st.text_input(
            "Enter Stay ID",
            placeholder="e.g. 99354408",
            help="Look up a specific ED stay.",
        )

        st.markdown("**— or —**")
        st.caption("Browse a random sample")
        sample_size = st.slider("Sample size", 10, 200, 50)
        seed = st.number_input("Random seed", value=42, step=1)

        df_sample = df.sample(n=sample_size, random_state=int(seed))

        if search_id.strip():
            matches = df[df["stay_id"] == search_id.strip()]
            if matches.empty:
                st.warning("Stay ID not found.")
                selected_row = None
            else:
                selected_row = matches.iloc[0]
                st.success(f"Found stay {search_id}")
        else:
            # Let user pick from sample
            options = df_sample["stay_id"].tolist()
            chosen = st.selectbox("Select a Stay ID", options)
            selected_row = df_sample[df_sample["stay_id"] == chosen].iloc[0]

        st.divider()
        generate_btn = st.button(
            "Generate Brief ▶",
            type="primary",
            disabled=(selected_row is None or not active_model.strip()),
            use_container_width=True,
        )

    # ── Main panel ────────────────────────────────────────────────────────────
    if selected_row is None:
        st.info("Select a patient from the sidebar to begin.")
        return

    summary = extract_patient_summary(selected_row)
    d = summary["demographics"]

    # Patient header
    st.subheader(
        f"Stay {selected_row['stay_id']}  |  "
        f"{d.get('age')} yo {d.get('gender', '?')}  |  "
        f"ESI {d.get('acuity_level', '?')}"
    )
    st.markdown(f"**Chief Complaint:** {summary['chief_complaint']}")
    st.markdown("---")

    # Two-column data view
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Triage Vitals")
        render_vitals_table(summary["vitals"])

        st.markdown("#### Risk Scores")
        render_scores(summary["risk_scores"])

    with col_right:
        st.markdown("#### Key Labs")
        if summary["labs"]:
            lab_df = pd.DataFrame(
                [{"Test": k, "Result": v} for k, v in summary["labs"].items()]
            ).set_index("Test")
            st.dataframe(lab_df, use_container_width=True)
        else:
            st.info("No lab results available.")

        st.markdown("#### Active Comorbidities")
        if summary["comorbidities"]:
            for c in summary["comorbidities"]:
                st.markdown(f"- {c}")
        else:
            st.markdown("_None documented_")

    # Utilization history
    with st.expander("Utilization History & Medications"):
        h = summary["visit_history"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ED Visits (30d)", h.get("n_ed_30d", "—"))
        col2.metric("ED Visits (90d)", h.get("n_ed_90d", "—"))
        col3.metric("Hospitalizations (30d)", h.get("n_hosp_30d", "—"))
        col4.metric("Active Meds", h.get("n_med", "—"))

    # Outcomes (ground truth — hidden by default)
    with st.expander("Documented Outcomes (Ground Truth)"):
        if summary["outcomes"]:
            st.error("⚠️ Confirmed outcomes: " + ", ".join(summary["outcomes"].keys()))
        else:
            st.success("No adverse outcomes documented for this visit.")

    # ── Brief Generation ──────────────────────────────────────────────────────
    st.markdown("### 📋 Early Risk Profile")

    if not active_model.strip():
        st.warning("Select a model in the sidebar to generate a brief.")
        return

    if generate_btn:
        with st.spinner("Generating Early Risk Profile…"):
            try:
                brief = generate_brief(summary, provider, active_model.strip())
                guardrail_results = run_guardrails(brief, summary, selected_row)
                st.session_state["last_brief"] = brief
                st.session_state["last_guardrails"] = guardrail_results
                st.session_state["last_stay"] = selected_row["stay_id"]
            except ollama.ResponseError as e:
                st.error(f"Ollama error: {e.error}. Is the model pulled and Ollama running?")
            except Exception as e:
                st.error(f"Error generating brief: {e}")

    if "last_brief" in st.session_state and st.session_state.get("last_stay") == selected_row["stay_id"]:
        # Normalize inline • bullets to proper markdown list items, strip divider lines
        import re
        brief_display = re.sub(r"\s*•\s*", "\n- ", st.session_state["last_brief"])
        brief_display = re.sub(r"^─+.*\n?", "", brief_display, flags=re.MULTILINE)
        st.markdown(brief_display)
        st.download_button(
            label="⬇ Download Brief (Markdown)",
            data=st.session_state["last_brief"],
            file_name=f"AURA_stay_{selected_row['stay_id']}.md",
            mime="text/markdown",
        )

        # ── Guardrail Panel ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Guardrail Checks")
        gr = st.session_state.get("last_guardrails", {})

        gr_col1, gr_col2, gr_col3 = st.columns(3)

        with gr_col1:
            cr = gr.get("clinical_review", {})
            action = cr.get("action", "—")
            icon = "✅" if action == "AUTO-APPROVE" else "⚠️"
            st.metric("Clinical Review", f"{icon} {action}", delta=f"Score: {cr.get('average', '—')}")
            if cr.get("needs_human_review"):
                st.warning("Flagged for human review")
            with st.expander("Score breakdown"):
                for k, v in cr.get("scores", {}).items():
                    st.write(f"**{k}**: {v}")

        with gr_col2:
            rag = gr.get("rag_grounding", {})
            rag_action = rag.get("action", "—")
            rag_icon = "✅" if rag.get("grounded") else "🔴"
            st.metric("RAG Grounding", f"{rag_icon} {rag_action}")
            if not rag.get("grounded") and rag.get("unsourced_claims"):
                with st.expander("Unsourced claims"):
                    for claim in rag["unsourced_claims"]:
                        st.write(f"- {claim}")

        with gr_col3:
            fair = gr.get("fairness", {})
            if fair.get("applicable") is False:
                st.metric("Fairness", "⚪ N/A", delta=fair.get("note", ""))
            else:
                warnings = fair.get("warnings", [])
                fair_icon = "✅" if not warnings else "⚠️"
                st.metric("Fairness (Pediatric)", f"{fair_icon} {'Pass' if not warnings else 'Warnings'}")
                if warnings:
                    with st.expander("Pediatric warnings"):
                        for w in warnings:
                            st.write(f"- {w}")
    else:
        st.info("Click **Generate Brief ▶** in the sidebar to create the AI-powered risk profile.")


if __name__ == "__main__":
    main()
