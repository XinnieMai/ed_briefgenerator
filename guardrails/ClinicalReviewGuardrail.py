"""Clinical Review Guardrails.  """

class clinicalReviewGuardrail:
    def __init__(self):
        self.review_rubric = {
            "factual_accuracy": 0,
            "risk_tier_justified": 0,
            "actionable": 0,
            "safe": 0,
            "fair": 0
        }
        
    def auto_score(self, AURA, truth):
        """Automatically score before human review."""
        scores = {}
        
        # Factual Accuracy
        if "110" in AURA and truth['HR'] == "110":
            scores['factual_accuracy'] = 1.0
        else:
            scores['factual_accuracy'] = 0.0
            
        # Risk Tier Justification
        if "HIGH" in AURA and truth['MEWS'] >= 3:
            scores['risk_tier_justified'] = 1.0
        else:
            scores['risk_tier_justified'] = 0.5
            
        # Average score
        avg_score = sum(scores.values()) / len(scores)
        
        # Flag for reviews if score < 0.8
        needs_review = avg_score < 0.8
        
        return {
            "scores": scores,
            "average": avg_score,
            "needs_human_review": needs_review,
            "action": "FLAG FOR REVIEW" if needs_review else "AUTO-APPROVE"
        }
        
# Example usage
if __name__ == "__main__":
    guardrail = clinicalReviewGuardrail()
    AURA_output = "Patient has HR 110 and MEWS 4, categorized as HIGH risk."
    truth_data = {"HR": "110", "MEWS": 4}
    
    result = guardrail.auto_score(AURA_output, truth_data)
    print(result)