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