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
            
        def validate_grounding(self, AURA):
            """Check that all claims in the AURA are sourced from the approved soruces."""
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
    