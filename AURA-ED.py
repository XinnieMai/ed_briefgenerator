# AURA-ED — AI risk briefs for ED patients

import os, re, sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import config
from config import ollama_client, gemini_client, OLLAMA_MODEL, GEMINI_MODEL
import ollama

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from guardrails.ClinicalReviewGuardrail import clinicalReviewGuardrail
from guardrails.FairnessGuardrail import FairnessGuardrail
from guardrails.RAGGuardrail import RAGGuardrail

st.set_page_config(page_title="AURA-ED", page_icon="🩺", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset/master_dataset.csv")

VITAL_REFS = {
    "triage_temperature": ("Temperature (°C)",   35.0, 38.5),
    "triage_heartrate":   ("Heart Rate (bpm)",   60,   100),
    "triage_resprate":    ("Resp Rate (/min)",   12,   20),
    "triage_o2sat":       ("O₂ Saturation (%)",  95,   100),
    "triage_sbp":         ("Systolic BP (mmHg)", 90,   140),
    "triage_dbp":         ("Diastolic BP (mmHg)",60,   90),
    "triage_MAP":         ("MAP (mmHg)",          70,   100),
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

COMORBIDITY_MAP = {**CCI_MAP, **ECI_MAP}

SCORE_THRESHOLDS = {
    "MEWS": (3, 5), "NEWS": (5, 9), "NEWS2": (5, 9), "REMS": (8, 12), "CART": (5, 9),
}


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df[["stay_id", "subject_id"]] = df[["stay_id", "subject_id"]].astype(str)
    return df


def extract_patient_summary(row: pd.Series) -> dict:
    vitals = {
        label: {"value": round(float(row[col]), 1), "flag": "LOW" if row[col] < lo else "HIGH" if row[col] > hi else "NORMAL"}
        for col, (label, lo, hi) in VITAL_REFS.items()
        if pd.notna(row.get(col))
    }

    scores = {
        label: round(float(row[col]), 1)
        for col, label in {
            "score_MEWS": "MEWS", "score_NEWS": "NEWS", "score_NEWS2": "NEWS2",
            "score_REMS": "REMS", "score_CART": "CART", "score_CCI": "CCI (Charlson)",
        }.items()
        if pd.notna(row.get(col))
    }

    labs = {
        lab: round(float(row[lab]), 2) if isinstance(row[lab], float) else row[lab]
        for lab in KEY_LABS if pd.notna(row.get(lab))
    }

    comorbidities = sorted({
        name for col, name in COMORBIDITY_MAP.items()
        if pd.notna(row.get(col)) and int(row[col]) == 1
    })

    history = {col: int(row[col]) for col in
               ["n_ed_30d", "n_ed_90d", "n_hosp_30d", "n_hosp_90d", "n_icu_30d", "n_icu_90d", "n_med"]
               if pd.notna(row.get(col))}

    outcomes = {label: True for col, label in OUTCOME_LABELS.items()
                if pd.notna(row.get(col)) and int(row[col]) == 1}

    return {
        "demographics": {
            "age": row.get("age"), "gender": row.get("gender"),
            "race": row.get("race"), "acuity_level": row.get("triage_acuity"),
        },
        "chief_complaint": row.get("CC", "Not recorded"),
        "vitals": vitals, "risk_scores": scores, "labs": labs,
        "comorbidities": comorbidities, "visit_history": history, "outcomes": outcomes,
    }


def _bullets(items) -> str:
    return "\n".join(f"  • {k}: {v}" for k, v in items) or "  Not available"


def build_prompt(summary: dict) -> str:
    d = summary["demographics"]
    h = summary["visit_history"]

    vitals_text      = "\n".join(f"  • {k}: {v['value']}  [{v['flag']}]" for k, v in summary["vitals"].items()) or "  Not available"
    scores_text      = _bullets(summary["risk_scores"].items())
    labs_text        = _bullets(summary["labs"].items())
    comorbidity_text = "\n".join(f"  • {c}" for c in summary["comorbidities"]) or "  None documented"
    outcomes_text    = ", ".join(summary["outcomes"]) or "None flagged"
    history_text     = (
        f"  ED visits (30d/90d): {h.get('n_ed_30d',0)} / {h.get('n_ed_90d',0)}\n"
        f"  Hospitalizations (30d/90d): {h.get('n_hosp_30d',0)} / {h.get('n_hosp_90d',0)}\n"
        f"  ICU stays (30d/90d): {h.get('n_icu_30d',0)} / {h.get('n_icu_90d',0)}\n"
        f"  Active medications: {h.get('n_med','?')}"
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
{comorbidity_text}

UTILIZATION HISTORY
{history_text}

DOCUMENTED OUTCOMES (Ground-truth, for context)
{outcomes_text}

SCORE INTERPRETATION GUIDE
Use these validated thresholds when assigning the risk tier:
  • MEWS  ≥ 5 → CRITICAL  |  3–4 → HIGH  |  1–2 → MODERATE  |  0 → LOW
  • NEWS  ≥ 7 → CRITICAL  |  5–6 → HIGH  |  1–4 → MODERATE  |  0 → LOW
  • NEWS2 ≥ 7 → CRITICAL  |  5–6 → HIGH  |  1–4 → MODERATE  |  0 → LOW
  • REMS  ≥ 12 → CRITICAL  |  8–11 → HIGH  |  <8 → MODERATE/LOW
  • CART  ≥ 9 → CRITICAL  |  5–8 → HIGH  |  <5 → MODERATE/LOW
  • CCI   ≥ 4 → very high comorbidity burden (elevates tier by one level)
  Sepsis flag (qSOFA ≥ 2): RR ≥ 22  OR  SBP ≤ 100  OR  altered mentation — any two present with suspected infection → escalate to HIGH minimum.
  Lactate > 2 mmol/L with suspected infection → sepsis concern, escalate tier.

TASK
Step 1 — REASON: Silently evaluate each score against the thresholds above and note which vitals or labs are abnormal. Consider whether qSOFA or sepsis criteria are met.
Step 2 — ASSIGN: Based on that reasoning, select ONE overall tier: LOW / MODERATE / HIGH / CRITICAL. Use the highest tier justified by any single score or clinical flag.
Step 3 — WRITE: Generate the brief using the structure below.

## Overall Risk Assessment
One-line summary: state the assigned tier (LOW / MODERATE / HIGH / CRITICAL) and a single-sentence rationale citing the primary driver.

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
        return gemini_client.models.generate_content(model=model, contents=prompt).text
    return ollama_client.chat(model=model, messages=[{"role": "user", "content": prompt}]).message.content


def run_guardrails(brief: str, summary: dict, row: pd.Series) -> dict:
    vitals, scores = summary["vitals"], summary["risk_scores"]
    age = summary["demographics"].get("age")
    if age is not None:
        age = int(age)

    truth = {}
    for key, path in [
        ("HR", ("Heart Rate (bpm)",)), ("RR", ("Resp Rate (/min)",)),
        ("Temp", ("Temperature (°C)",)), ("SBP", ("Systolic BP (mmHg)",)),
        ("SpO2", ("O₂ Saturation (%)",)),
    ]:
        val = vitals.get(path[0], {}).get("value")
        if val is not None:
            truth[key] = val
    for key in ("MEWS", "NEWS", "NEWS2"):
        if scores.get(key) is not None:
            truth[key] = scores[key]

    if age is not None and age < 18:
        fairness_result = FairnessGuardrail().check_pediatric(brief, age)
    else:
        fairness_result = {"applicable": False, "note": "Adult patient — pediatric check skipped"}

    return {
        "clinical_review": clinicalReviewGuardrail().auto_score(brief, truth),
        "fairness": fairness_result,
        "rag_grounding": RAGGuardrail().validate_grounding(brief, patient_row=row),
    }


def flag_color(flag: str) -> str:
    return {"HIGH": "🔴", "LOW": "🔵", "NORMAL": "🟢"}.get(flag, "⚪")


def render_vitals_table(vitals: dict):
    if not vitals:
        st.info("No triage vitals available.")
        return
    rows = [{"Vital Sign": k, "Value": v["value"], "Status": f"{flag_color(v['flag'])} {v['flag']}"} for k, v in vitals.items()]
    st.dataframe(pd.DataFrame(rows).set_index("Vital Sign"), use_container_width=True)


def render_scores(scores: dict):
    if not scores:
        st.info("No risk scores available.")
        return
    for col, (label, val) in zip(st.columns(len(scores)), scores.items()):
        col.metric(label=label, value=val)


def main():
    st.title("AURA-ED")
    st.caption("AI-powered Utilization & Risk Analysis in the Emergency Department")

    with st.sidebar:
        st.header("Configuration")
        provider = st.radio("Model Provider", ["Ollama", "Gemini"], horizontal=True)

        if provider == "Ollama":
            active_model = st.text_input("Ollama Model", value=OLLAMA_MODEL,
                                         placeholder="e.g. llama3.2, mistral, gemma3",
                                         help="Run `ollama pull <model>` to download a model.")
            if st.button("Pull Model", use_container_width=True):
                with st.spinner(f"Pulling {active_model}…"):
                    try:
                        ollama_client.pull(active_model.strip())
                        st.success(f"'{active_model}' pulled successfully.")
                    except Exception as e:
                        st.error(f"Pull failed: {e}")
        else:
            active_model = st.selectbox("Gemini Model", [GEMINI_MODEL])
            if not config.GEMINI_API_KEY:
                st.warning("GEMINI_API_KEY not set in .env")

        st.divider()
        st.header("Patient Selection")
        df = load_data()

        if "random_stay_id" not in st.session_state:
            st.session_state.random_stay_id = None

        search_id = st.text_input("Enter Stay ID", placeholder="e.g. 99354408")
        st.markdown("**— or —**")
        chosen = st.selectbox("Select a Stay ID", df.sample(n=50)["stay_id"].tolist(),
                              on_change=lambda: st.session_state.update(random_stay_id=None))
        st.markdown("**— or —**")
        if st.button("Next →  Random Patient", use_container_width=True):
            st.session_state.random_stay_id = df.sample(n=1).iloc[0]["stay_id"]

        if search_id.strip():
            matches = df[df["stay_id"] == search_id.strip()]
            if matches.empty:
                selected_row = None
                st.warning("Stay ID not found.")
            else:
                selected_row = matches.iloc[0]
                st.success(f"Found stay {search_id}")
        elif st.session_state.random_stay_id:
            selected_row = df[df["stay_id"] == st.session_state.random_stay_id].iloc[0]
        else:
            selected_row = df[df["stay_id"] == chosen].iloc[0]

        st.divider()
        generate_btn = st.button("Generate Brief ▶", type="primary",
                                 disabled=(selected_row is None or not active_model.strip()),
                                 use_container_width=True)

    if selected_row is None:
        st.info("Select a patient from the sidebar to begin.")
        return

    summary = extract_patient_summary(selected_row)
    d = summary["demographics"]

    st.subheader(f"Stay {selected_row['stay_id']}  |  {d.get('age')} yo {d.get('gender', '?')}  |  ESI {d.get('acuity_level', '?')}")
    st.markdown(f"**Chief Complaint:** {summary['chief_complaint']}")
    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Triage Vitals")
        render_vitals_table(summary["vitals"])
        st.markdown("#### Risk Scores")
        render_scores(summary["risk_scores"])

    with col_right:
        st.markdown("#### Key Labs")
        if summary["labs"]:
            st.dataframe(pd.DataFrame([{"Test": k, "Result": v} for k, v in summary["labs"].items()]).set_index("Test"), use_container_width=True)
        else:
            st.info("No lab results available.")
        st.markdown("#### Active Comorbidities")
        st.markdown("\n".join(f"- {c}" for c in summary["comorbidities"]) or "_None documented_")

    with st.expander("Utilization History & Medications"):
        h = summary["visit_history"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ED Visits (30d)", h.get("n_ed_30d", "—"))
        c2.metric("ED Visits (90d)", h.get("n_ed_90d", "—"))
        c3.metric("Hospitalizations (30d)", h.get("n_hosp_30d", "—"))
        c4.metric("Active Meds", h.get("n_med", "—"))

    with st.expander("Documented Outcomes (Ground Truth)"):
        if summary["outcomes"]:
            st.error("Confirmed outcomes: " + ", ".join(summary["outcomes"]))
        else:
            st.success("No adverse outcomes documented for this visit.")

    st.markdown("### Early Risk Profile")
    if not active_model.strip():
        st.warning("Select a model in the sidebar to generate a brief.")
        return

    if generate_btn:
        with st.spinner("Generating Early Risk Profile…"):
            try:
                brief = generate_brief(summary, provider, active_model.strip())
                st.session_state.update({
                    "last_brief": brief,
                    "last_guardrails": run_guardrails(brief, summary, selected_row),
                    "last_stay": selected_row["stay_id"],
                })
            except ollama.ResponseError as e:
                st.error(f"Ollama error: {e.error}. Is the model pulled and Ollama running?")
            except Exception as e:
                st.error(f"Error generating brief: {e}")

    if "last_brief" in st.session_state and st.session_state.get("last_stay") == selected_row["stay_id"]:
        brief_display = re.sub(r"\s*•\s*", "\n- ", st.session_state["last_brief"])
        brief_display = re.sub(r"^─+.*\n?", "", brief_display, flags=re.MULTILINE)
        st.markdown(brief_display)
        st.download_button("Download Brief (Markdown)", data=st.session_state["last_brief"],
                           file_name=f"AURA_stay_{selected_row['stay_id']}.md", mime="text/markdown")

        st.markdown("---")
        st.markdown("### Guardrail Checks")
        gr = st.session_state.get("last_guardrails", {})
        gr_col1, gr_col2, gr_col3 = st.columns(3)

        with gr_col1:
            cr = gr.get("clinical_review", {})
            action = cr.get("action", "—")
            st.metric("Clinical Review", f"{'✅' if action == 'AUTO-APPROVE' else '⚠️'} {action}", delta=f"Score: {cr.get('average', '—')}")
            if cr.get("needs_human_review"):
                st.warning("Flagged for human review")
            with st.expander("Score breakdown"):
                for k, v in cr.get("scores", {}).items():
                    st.write(f"**{k}**: {v}")

        with gr_col2:
            rag = gr.get("rag_grounding", {})
            st.metric("RAG Grounding", f"{'✅' if rag.get('grounded') else '🔴'} {rag.get('action', '—')}")
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
                st.metric("Fairness (Pediatric)", f"{'✅' if not warnings else '⚠️'} {'Pass' if not warnings else 'Warnings'}")
                if warnings:
                    with st.expander("Pediatric warnings"):
                        for w in warnings:
                            st.write(f"- {w}")
    else:
        st.info("Click **Generate Brief ▶** in the sidebar to create the AI-powered risk profile.")


if __name__ == "__main__":
    main()
