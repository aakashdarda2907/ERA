# Sustainable AI

Sustainable AI concerns the environmental footprint of building and running
AI systems, alongside their social and economic sustainability.

- Training cost: large models require significant compute, translating to
  substantial electricity use and carbon emissions, concentrated in a short,
  intensive training period.
- Inference cost: once deployed, a widely-used model's cumulative energy use
  from serving predictions can exceed its one-time training cost over its
  lifetime.
- Efficiency techniques: model compression, quantization, and knowledge
  distillation reduce a model's compute footprint with limited accuracy loss
  - relevant to this project's own choice of TF-IDF over a heavier embedding
    model for retrieval, and a compact model (XGBoost) over a larger one.
- Reporting: some researchers advocate disclosing a model's estimated energy
  use and carbon footprint alongside its accuracy, similar to a nutrition
  label, so environmental cost is part of the standard evaluation.

Sustainability also covers whether an AI system's benefits justify its
resource cost for a given use case - not every problem needs a large model.
