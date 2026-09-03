"""Run the whole decision analysis and write `reports/decision/`.

One entry point, so the README, the model card and the figures cannot cite different runs.
"""

from credexp.modeling.decision_report import run_decision_analysis

if __name__ == "__main__":
    payload = run_decision_analysis()
    print(f"holdout: {payload['n_holdout']} rows, default rate {payload['default_rate']:.4f}")
    print(f"cost interval: {payload['cost_interval']}")
    print("wrote reports/decision/")
