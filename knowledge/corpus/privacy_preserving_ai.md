# Privacy-Preserving AI

Two major techniques address the tension between using data to train useful
models and protecting the individuals that data describes:

- Differential privacy: adds carefully calibrated statistical noise during
  training or querying, so no individual's data can be confidently
  reconstructed from the model's outputs, while still allowing useful
  aggregate patterns to be learned.
- Federated learning: trains a shared model across multiple institutions
  (e.g. several hospitals) without any institution's raw data ever leaving
  its own servers - only model updates are shared and combined centrally.

Both trade off some model accuracy or convenience for stronger privacy
guarantees, and are especially relevant in healthcare and finance, where
data is sensitive and often legally restricted from being centralized.
