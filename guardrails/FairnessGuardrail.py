"""Fairness evaluation for classification models."""

class FairnessGuardrail:
    def __init__(self):
        self.demographic_baselines = {
            'age_under_18': {
                'sensitivity': 0.84,
                'specificity': 0.75,
                'group_name': 'Pediatric (under 18)'
            },
            'age_18-30': {
                'sensitivity': 0.80,
                'specificity': 0.70,
                'group_name': 'Young Adults (18-30)'
            },
            'age_31-50': {
                'sensitivity': 0.78,
                'specificity': 0.72,
                'group_name': 'Adults (31-50)'
            },
            'age_51-70': {
                'sensitivity': 0.75,
                'specificity': 0.70,
                'group_name': 'Middle-aged Adults (51-70)'
            },
            'age_71-100': {
                'sensitivity': 0.70,
                'specificity': 0.65,
                'group_name': 'Older Adults (71-100)'
            }
        }
        
        # How much variance is acceptable
        self.variance_threshold = 0.05
        
        def check_drift(self, demographic_group, current_perf):
            """Check if performance degraded for group."""
            baseline = self.demographic_baselines.get(demographic_group)
            if not baseline:
                return {"checked": False, "error": f"Unknown demographic group: {demographic_group}"}
            
            sensitivity_drift = abs(current_perf['sensitivity'] - baseline['sensitivity'])
            
            if sensitivity_drift > self.variance_threshold:
                return {
                    "fair": False,
                    "group": demographic_group,
                    "action": "FLAG"
                }
            return {
                "fair": True,
                "group": demographic_group,
                "action": "APPROVE"
            }
            
        def check_pediatric(self, AURA, age_years):
            """Pediatric-specific check."""
            if age_years >= 18:
                return {"applicable": False, "error": "Patient is not pediatric"}
            
            warnings = []
            if "infection" in AURA.lower() and "fever" not in AURA.lower():
                warnings.append("Sepsis without fever - atypical in pediatrics")
            if "lactate" in AURA.lower():
                warnings.append("Elevated lactate - concerning in pediatrics")
            return {"applicable": True, "warnings": warnings}
        
# Example usage
if __name__ == "__main__":
    guardrail = FairnessGuardrail()
    demographic_group = "age_18-30"
    current_perf = {
        "sensitivity": 0.80,
        "specificity": 0.70
    }
    result = guardrail.check_drift(demographic_group, current_perf)
    print(result)