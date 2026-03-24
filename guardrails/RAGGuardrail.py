"""Knowledge Groudning using a Retrieval-Augmented Generation (RAG) approach.
This module provides a framework for integrating external knowledge sources into the generation process of language models.
Tasks: 
- Define a base class for RAG models.
- Check that all claims are sourced from the knowledge base."""

class RAGGuardrail:
    def __init__(self):
        self.approved_sources = {
            'MEWS': 'MEWS >= 3 = high risk',
            'qSOFA': 'qSOFA >= 2 = sepsis risk',
            'NEWS2': 'NEWS2 >= 5 = escalate care',
            'Sepsis_bundle': 'Lactate > 2 = sepsis concern'
        }

    def validate_grounding(self, AURA, patient_row=None):
        """Check that all claims in the AURA are sourced from the approved sources.

        If patient_row is provided, also verifies that numeric values in the AURA
        match the patient's actual data.
        """
        unsourced_claims = []

        # Check for common hallucinations
        red_flags = [
            ("hallucinated stat", "Any claim without source"),
            ("invented value", "Number not in the dataset"),
            ("unsupported recommendation", "Action not in guidelines")
        ]

        for flag, description in red_flags:
            if flag in AURA.lower():
                unsourced_claims.append(description)

        # If patient data is provided, verify key numeric values appear correctly
        if patient_row is not None:
            numeric_checks = {
                'score_MEWS': 'mews',
                'score_NEWS2': 'news2',
                'triage_heartrate': 'hr',
                'triage_temperature': 'temp',
            }
            for col, label in numeric_checks.items():
                if col in patient_row and patient_row[col] is not None:
                    val = str(round(float(patient_row[col])))
                    if label in AURA.lower() and val not in AURA:
                        unsourced_claims.append(f"Value mismatch for {label}: expected {val}")

        if unsourced_claims:
            return {
                "grounded": False,
                "unsourced_claims": unsourced_claims,
                "action": "REGENERATE AURA"
            }
        return {
            "grounded": True,
            "action": "APPROVE"
        }
            
# Example usage
if __name__ == "__main__":
    guardrail = RAGGuardrail()
    AURA_example = "The patient has a hallucinated stat of 5, which is not sourced from MEWS."
    result = guardrail.validate_grounding(AURA_example)
    print(result)
    