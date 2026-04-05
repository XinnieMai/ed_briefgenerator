# AURA-ED: AI-powered Utilization & Risk Analysis in the Emergency Department

**Course: DS 5003: Healthcare Data Science**

AURA-ED is a clinical decision-support tool designed to convert fragmented Emergency Department (ED) data into structured, actionable "Early Risk Profiles". By leveraging Large Language Models (LLMs) and the Stanford MC-MED dataset, this tool reduces clinicians' cognitive load by synthesizing vital signs, laboratory results, and medical histories into concise risk narratives.

# Project Purpose: 
Emergency physicians spend significant portions of their shifts on indirect care tasks, leading to professional burnout and potential delays in risk identification. AURA-ED aims to:
- Automate Clinical Synthesis: Transform raw data (vitals, labs, history) into a human-readable brief.
- Improve Early Detection: Reliably flag patients at risk for critical outcomes like sepsis, stroke, or ICU transfer.
- Enhance Efficiency: Potential to reclaim clinical hours by providing an immediate, high-fidelity understanding of a patient's risk profile.

# Dataset Access: 
- This project utilizes the Stanford MultiModal Clinical Monitoring in the Emergency Department (MC-MED) dataset, which includes 118,385 adult ED visits.
Request Access: The dataset is hosted on PhysioNet. You must complete the required CITI training to gain access.
- MC-MED database is not provided with this repository and is **required** for this workflow. 
## Generating the Master Dataset:
To generate the `master_dataset.csv` required by `brief_generator_llama.py`, follow these steps based on the MC-MED processing pipeline:
The structure of this repository is detailed as follows:

- `benchmark_scripts/...` contains the scripts for benchmark dataset generation (master_data.csv).

**Master Dataset Workflow**
Before proceeding, download and set up the MC-MED repository locally. 

### 1. Benchmark Data Generation
~~~
python extract_master_dataset.py 
~~~

**Arguements**:
- `VISITS_PATH` : Path to the directory containing the patient's ED visit data.
- `MEDS_PATH ` : Path to the directory containing the patient's home medication records.
- `PMH_PATH` : Path to the directory containing the patient's past medical history.
- `LABS_PATH` : Path to the directory containing the patient's laboratory tests and results.
- `output_path` : Path to output directory

**Output**:
`master_dataset.csv` output to `output_path`

# Dataset
- Source: [https://physionet.org/content/mc-med/1.0.0/](https://physionet.org/content/mc-med/1.0.0/) (Publicly available)
- Composition: A total of **216** variables are included in `master_dataset.csv`
- Protocol: 80/20 train-test split.

# Running the App

## Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

Ollama must be running locally with at least one model pulled:
```bash
ollama serve
ollama pull gemma3:12b   # or any supported model
```

## Environment Variables

Create a `.env` file in the project root (optional — defaults shown):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TIMEOUT` | `120` | Request timeout in seconds |
| `OLLAMA_MODEL` | `gemma3:12b` | Default Ollama model |
| `AUTO_PULL` | `true` | Auto-pull model if not found locally |
| `GEMINI_API_KEY` | *(empty)* | Optional Gemini API key |

## Launch

```bash
streamlit run AURA-ED.py
```

The app will open at `http://localhost:8501`. Select a patient record and choose a model from the dropdown to generate an Early Risk Profile.

# Inference Pipeline
To generate the Clinical Risk Briefs, the system utilizes a Retrieval-Augmented Generation (RAG) architecture.
- Employs a zero-shot chain-of-thought prompting strategy to force the model to weigh acute physiological instability against chronic disease burden.
- The engine integrates triage vitals, high-priority labs (Lactate, Troponin), and validated risk scores (NEWS2, CCI, ECI) into a constrained output schema.

~~~
python brief_generator_llama.py --input master_dataset.csv --model llama3.1:8b
~~~

# Evaluation Results

Models were evaluated on 300 held-out patients from the MC-MED test set. Metrics include AUROC, F1, Accuracy, and Outcome Prevalence for each clinical outcome.

## llama3.2

| Outcome | AUROC | F1 | Accuracy | Prevalence |
|---|---|---|---|---|
| hospitalization | 0.5661 | 0.3423 | 0.5833 | 0.2333 |
| critical | 0.6353 | 0.0615 | 0.5933 | 0.0233 |
| icu_transfer_12h | 0.6353 | 0.0615 | 0.5933 | 0.0233 |
| sepsis | 0.3729 | 0.0000 | 0.5867 | 0.0033 |
| aki | 0.8800 | 0.0320 | 0.5967 | 0.0067 |
| stroke | 0.4734 | 0.0157 | 0.5833 | 0.0133 |
| ahf | 0.7778 | 0.0476 | 0.6000 | 0.0100 |
| pe | 0.5432 | 0.0317 | 0.5933 | 0.0100 |

## llama3.2:1b

| Outcome | AUROC | F1 | Accuracy | Prevalence |
|---|---|---|---|---|
| hospitalization | 0.5283 | 0.3593 | 0.4596 | 0.2576 |
| critical | 0.3798 | 0.0492 | 0.4141 | 0.0303 |
| icu_transfer_12h | 0.3798 | 0.0492 | 0.4141 | 0.0303 |
| sepsis | 0.8756 | 0.0171 | 0.4912 | 0.0051 |
| aki | 0.4375 | 0.0169 | 0.4141 | 0.0101 |
| stroke | 0.7803 | 0.0504 | 0.4293 | 0.1520 |
| ahf | 0.4375 | 0.0169 | 0.4141 | 0.0101 |
| pe | 0.5812 | 0.0171 | 0.4193 | 0.0051 |

## llama3.1:8b

| Outcome | AUROC | F1 | Accuracy | Prevalence |
|---|---|---|---|---|
| hospitalization | 0.7019 | 0.4000 | 0.7867 | 0.2333 |
| critical | 0.6902 | 0.1231 | 0.8100 | 0.0233 |
| icu_transfer_12h | 0.6902 | 0.1231 | 0.8100 | 0.0233 |
| sepsis | 0.9933 | 0.0339 | 0.8100 | 0.0033 |
| aki | 0.4765 | 0.0000 | 0.8000 | 0.0067 |
| stroke | 0.5823 | 0.0323 | 0.8000 | 0.0067 |
| ahf | 0.8990 | 0.0984 | 0.8167 | 0.0100 |
| pe | 0.8990 | 0.0984 | 0.8167 | 0.0100 |

## gemma3:12b

| Outcome | AUROC | F1 | Accuracy | Prevalence |
|---|---|---|---|---|
| hospitalization | 0.6809 | 0.4912 | 0.6133 | 0.2333 |
| critical | 0.7087 | 0.0727 | 0.4900 | 0.0233 |
| icu_transfer_12h | 0.7087 | 0.0727 | 0.4900 | 0.0233 |
| sepsis | 0.7358 | 0.0126 | 0.4767 | 0.0033 |
| aki | 0.7366 | 0.0250 | 0.4800 | 0.0067 |
| stroke | 0.7382 | 0.0494 | 0.4867 | 0.0133 |
| ahf | 0.7374 | 0.0373 | 0.4833 | 0.0100 |
| pe | 0.7374 | 0.0373 | 0.4833 | 0.0100 |

# Limitations
Pilot Sample Size: While the initial plan involved the full 80/20 split, computational constraints (8-hour run times for 300 patients) limited the pilot evaluation to a representative sample of 300 ED encounters.

HIPAA Compliance: To protect patient privacy, all names are omitted; users must search via SubjectID.

Model Blind Spots: No AI should operate without physician oversight. For example, llama3.1:8b showed sub-chance performance for Acute Kidney Injury (AKI), indicating where human adjudication is irreplaceable.

# Future Work
Infrastructure: Migrating from local inference to a Virtual Private Server (VPS) or Cloud environment (e.g., AWS Bedrock) to achieve sub-second inference speeds.

Integration: Transitioning from a Streamlit prototype to a secure, FHIR-compliant hospital plugin.

Equity: Implementing real-time auditing for Algorithmic Equity to ensure consistent risk narratives across diverse demographics.

# References

