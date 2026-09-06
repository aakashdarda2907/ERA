# Fairness and Bias in AI

Bias in an AI model can come from several sources: skewed training data
(some groups underrepresented), historical bias baked into past decisions
the data reflects, or a model latching onto a feature that correlates with
a protected attribute even without using that attribute directly.

Common fairness metrics include:

- Demographic parity difference: do different groups get a positive
  prediction (e.g. "high risk") at similar rates, regardless of whether
  that prediction is correct?
- Equalized odds difference: do different groups get CORRECT predictions
  at similar rates - a stricter, outcome-aware check than demographic
  parity.

Bias mitigation techniques include re-sampling training data, re-weighting
underrepresented groups, or post-processing model outputs to equalize error
rates across groups. No single metric captures "fairness" completely - a
model can satisfy one fairness definition while violating another.
