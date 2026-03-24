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
        """Automatically score before human review.

        Args:
            AURA: The generated AURA brief text.
            truth: Dict of patient ground-truth values. Expected keys:
                   'HR' (int/float), 'MEWS' (int/float), 'risk_tier' (str: 'HIGH'/'MODERATE'/'LOW').
        """
        scores = {}

        # Factual Accuracy — verify key vitals mentioned in AURA match truth
        vital_checks = {k: v for k, v in truth.items() if k in ('HR', 'RR', 'Temp', 'SBP', 'SpO2')}
        if vital_checks:
            matched = sum(1 for k, v in vital_checks.items() if str(round(float(v))) in AURA)
            scores['factual_accuracy'] = round(matched / len(vital_checks), 2)
        else:
            scores['factual_accuracy'] = 0.5  # can't verify without vitals

        # Risk Tier Justification — MEWS thresholds: >=5 HIGH, 3-4 MODERATE, <3 LOW
        mews = truth.get('MEWS')
        expected_tier = truth.get('risk_tier')
        if mews is not None:
            if mews >= 5:
                expected_tier = expected_tier or 'HIGH'
            elif mews >= 3:
                expected_tier = expected_tier or 'MODERATE'
            else:
                expected_tier = expected_tier or 'LOW'

        if expected_tier and expected_tier.upper() in AURA.upper():
            scores['risk_tier_justified'] = 1.0
        else:
            scores['risk_tier_justified'] = 0.0

        # Actionable — brief should contain at least one recommended action
        action_keywords = ['monitor', 'consult', 'order', 'administer', 'transfer', 'notify', 'initiate']
        scores['actionable'] = 1.0 if any(kw in AURA.lower() for kw in action_keywords) else 0.0

        # Safe — brief must not recommend contraindicated or dangerous actions
        unsafe_keywords = ['ignore', 'discharge immediately', 'no further workup']
        scores['safe'] = 0.0 if any(kw in AURA.lower() for kw in unsafe_keywords) else 1.0

        # Average score
        avg_score = round(sum(scores.values()) / len(scores), 2)

        # Flag for review if score < 0.8
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
    AURA_output = "Patient has HR 110 and MEWS 4, categorized as MODERATE risk. Monitor closely and consult cardiology."
    truth_data = {"HR": 110, "MEWS": 4, "risk_tier": "MODERATE"}

    result = guardrail.auto_score(AURA_output, truth_data)
    print(result)