# Explainable AI (XAI)

The "black box problem" refers to models - especially complex ones like
gradient-boosted trees or deep neural networks - whose internal decision
process cannot be understood just by inspection, unlike a simple model such
as logistic regression, where each input's contribution is a direct,
readable coefficient.

Two widely used techniques address this:

- SHAP (SHapley Additive exPlanations): assigns each input feature a value
  representing how much it pushed a specific prediction up or down,
  based on cooperative game theory. Gives exact, consistent explanations
  for tree-based models like XGBoost.
- LIME (Local Interpretable Model-agnostic Explanations): approximates a
  complex model's behavior near one specific prediction with a simpler,
  interpretable model, to explain that one decision.

Both are "post-hoc" - applied after training - as opposed to using an
inherently interpretable model from the start.
