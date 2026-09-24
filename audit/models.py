from django.db import models

class Patient(models.Model):
    encounter_id = models.BigIntegerField(unique=True, primary_key=True)
    patient_nbr = models.BigIntegerField()
    race = models.CharField(max_length=50, blank=True, null=True)
    gender = models.CharField(max_length=20)
    age_bracket = models.CharField(max_length=20)  # e.g. "[70-80)"
    time_in_hospital = models.IntegerField()
    num_lab_procedures = models.IntegerField()
    num_procedures = models.IntegerField()
    num_medications = models.IntegerField()
    number_outpatient = models.IntegerField()
    number_emergency = models.IntegerField()
    number_inpatient = models.IntegerField()
    number_diagnoses = models.IntegerField()
    max_glu_serum = models.CharField(max_length=20, blank=True, null=True)
    a1c_result = models.CharField(max_length=20, blank=True, null=True)
    change_med = models.CharField(max_length=10)
    diabetes_med = models.CharField(max_length=10)
    readmitted_raw = models.CharField(max_length=10)   # "NO" / ">30" / "<30"
    readmitted_label = models.IntegerField()            # 0 or 1

class Prediction(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='predictions')
    model_version = models.CharField(max_length=50)     # e.g. "xgboost-v0.1"
    risk_score = models.FloatField()                    # predicted probability
    predicted_label = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('patient', 'model_version')

class ShapExplanation(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='shap_values')
    model_version = models.CharField(max_length=50)
    feature_name = models.CharField(max_length=100)
    feature_value = models.CharField(max_length=100)
    shap_value = models.FloatField()

    class Meta:
        indexes = [models.Index(fields=['patient', 'model_version'])]


class ModelMetric(models.Model):
    """Global (non-group) model performance metrics, e.g. ROC-AUC on the held-out
    test set. Kept separate from FairnessMetric because these describe overall
    predictive performance rather than a per-protected-attribute fairness gap,
    and separate from Prediction because they're one row per model, not per patient.
    """
    model_version = models.CharField(max_length=50)   # e.g. "xgboost-v0.1"
    metric_name = models.CharField(max_length=100)     # e.g. "roc_auc"
    value = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('model_version', 'metric_name')

    def __str__(self):
        return f'{self.model_version} {self.metric_name}={self.value:.3f}'


class FairnessMetric(models.Model):
    model_version = models.CharField(max_length=50)
    protected_attribute = models.CharField(max_length=50)   # 'race', 'gender', 'age_bracket'
    metric_name = models.CharField(max_length=100)           # 'demographic_parity_difference', etc.
    value = models.FloatField()
    detail = models.JSONField(blank=True, null=True)         # per-group breakdown
    created_at = models.DateTimeField(auto_now_add=True)

class ExplanationDisparity(models.Model):
    """Stores how differently the model 'reasons' about a feature for one
    demographic subgroup vs. the overall population - measured via Wasserstein
    distance between the subgroup's SHAP-contribution distribution for that
    feature and the global distribution. This is a fairness check on the
    EXPLANATIONS themselves, not on the predictions (which FairnessMetric
    already covers) - a model can look fair on outcomes while still reasoning
    very differently about different groups.
    """
    model_version = models.CharField(max_length=50)
    feature_name = models.CharField(max_length=100)       # base feature, e.g. "number_inpatient", "age"
    protected_attribute = models.CharField(max_length=50)  # 'race', 'gender', 'age_bracket'
    subgroup = models.CharField(max_length=100)             # e.g. "AfricanAmerican", "Female", "70-80)"
    subgroup_mean_shap = models.FloatField()
    global_mean_shap = models.FloatField()
    wasserstein_distance = models.FloatField()
    n = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['model_version', 'protected_attribute', 'feature_name'])]


class ExplanationStability(models.Model):
    """Stores how much a patient's SHAP-based explanation changes under a
    small, realistic perturbation to their data - independent of whether the
    final prediction changes. This is the STABILITY axis of explanation
    quality, complementing ExplanationDisparity's FAIRNESS axis: disparity
    asks "does the model reason differently by WHO the patient is?", this
    asks "does the model's reasoning hold up under tiny changes to WHAT the
    patient's data says?"
    """
    model_version = models.CharField(max_length=50)
    encounter_id = models.IntegerField()
    perturbation = models.CharField(max_length=100)  # e.g. "number_inpatient +1"
    original_score = models.FloatField()
    perturbed_score = models.FloatField()
    score_shift = models.FloatField()
    classification_flipped = models.BooleanField()
    original_top3 = models.CharField(max_length=200)   # comma-joined base feature names
    perturbed_top3 = models.CharField(max_length=200)
    shared_top3_count = models.IntegerField()            # 0-3: how many top factors survived
    jaccard_similarity = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['model_version', 'shared_top3_count'])]