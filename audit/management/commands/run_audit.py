"""
Management command: computes SHAP explanations and Fairlearn group-fairness
metrics for the XGBoost model, and caches both to the database.

Run with: python manage.py run_audit  (after train_model has already run)

This is the command that actually produces the evidence the chatbot cites —
SHAP answers "why did the model decide this for one patient", Fairlearn
answers "is the model systematically unfair to a group". Together they cover
the Explainable AI and Fairness/Bias syllabus units with real numbers rather
than the model narrating its own reasoning in free text.
"""
import numpy as np
import joblib
import shap
from fairlearn.metrics import (
    MetricFrame, selection_rate,
    demographic_parity_difference, equalized_odds_difference,
)
from django.core.management.base import BaseCommand
from audit.models import Patient, ShapExplanation, FairnessMetric
from audit.pipeline import load_and_clean, build_features, split

# How many top SHAP features to store per patient. Storing all ~100+ one-hot
# columns per patient would bloat the DB for no benefit — only the strongest
# contributors are ever surfaced in an answer, so 6 gives headroom without waste.
TOP_K_FEATURES = 6


class Command(BaseCommand):
    help = 'Compute SHAP explanations (XGBoost) and Fairlearn group fairness metrics, cache to DB'

    def handle(self, *args, **options):
        # Rebuilding the exact same split (same random_state, same pipeline
        # functions) as train_model.py — this is why pipeline.py exists
        # instead of duplicating this logic in each command.
        self.stdout.write('Reloading data + rebuilding the same train/test split ...')
        df, dropped = load_and_clean()
        X, y = build_features(df)
        X_train, X_test, y_train, y_test, df_train, df_test = split(df, X, y)

        xgb = joblib.load('audit/ml_artifacts/xgboost_v0.1.pkl')

        # --- Explainability: per-patient SHAP values ---
        # TreeExplainer is used (not the generic/model-agnostic explainer)
        # because it's exact and fast specifically for tree ensembles like
        # XGBoost, rather than approximating via sampling.
        self.stdout.write('Computing SHAP values for XGBoost (may take a minute) ...')
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_test)

        ShapExplanation.objects.filter(model_version='xgboost-v0.1').delete()

        feature_names = list(X_test.columns)
        records = []
        for row_i, (idx, row) in enumerate(df_test.iterrows()):
            try:
                patient = Patient.objects.get(encounter_id=row['encounter_id'])
            except Patient.DoesNotExist:
                # Should only happen if run_audit is run before train_model
                # has successfully cached patients — skip rather than crash.
                continue
            row_shap = shap_values[row_i]
            # argsort by absolute value so the biggest pushes in EITHER
            # direction (toward high-risk or low-risk) are kept, not just
            # the biggest positive ones.
            top_idx = np.argsort(np.abs(row_shap))[::-1][:TOP_K_FEATURES]
            for fi in top_idx:
                records.append(ShapExplanation(
                    patient=patient,
                    model_version='xgboost-v0.1',
                    feature_name=feature_names[fi],
                    feature_value=str(X_test.iloc[row_i, fi]),
                    shap_value=float(row_shap[fi]),
                ))
        ShapExplanation.objects.bulk_create(records, batch_size=1000)
        self.stdout.write(self.style.SUCCESS(f'Stored {len(records)} SHAP entries.'))

        # --- Fairness: group-level bias metrics ---
        self.stdout.write('Computing Fairlearn group fairness metrics ...')
        xgb_preds = (xgb.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

        FairnessMetric.objects.filter(model_version='xgboost-v0.1').delete()

        # Three protected attributes checked per syllabus Unit IV (sources of
        # bias / algorithmic discrimination). Race/gender are the classic
        # legally-protected categories; age is added because early testing
        # showed it had the largest fairness gap of the three.
        sensitive_cols = {
            'race': df_test['race'].fillna('Unknown'),
            'gender': df_test['gender'],
            'age_bracket': df_test['age'],
        }

        fairness_records = []
        for attr_name, sensitive_series in sensitive_cols.items():
            # Demographic parity: do groups get flagged high-risk at similar
            # rates, regardless of whether the flag is correct?
            dpd = demographic_parity_difference(y_test, xgb_preds, sensitive_features=sensitive_series)
            # Equalized odds: do groups get flagged CORRECTLY at similar
            # rates? This is the stricter, outcome-aware fairness check.
            eod = equalized_odds_difference(y_test, xgb_preds, sensitive_features=sensitive_series)

            # Per-group breakdown (e.g. flag rate for each age bracket
            # individually), stored as JSON so the chatbot can cite exact
            # group numbers later, not just the single gap statistic.
            mf = MetricFrame(
                metrics=selection_rate, y_true=y_test, y_pred=xgb_preds,
                sensitive_features=sensitive_series,
            )
            group_detail = {str(k): float(v) for k, v in mf.by_group.items()}

            fairness_records.append(FairnessMetric(
                model_version='xgboost-v0.1', protected_attribute=attr_name,
                metric_name='demographic_parity_difference', value=float(dpd), detail=group_detail,
            ))
            fairness_records.append(FairnessMetric(
                model_version='xgboost-v0.1', protected_attribute=attr_name,
                metric_name='equalized_odds_difference', value=float(eod), detail=None,
            ))
            self.stdout.write(f'  {attr_name}: DP diff={dpd:.3f}, EO diff={eod:.3f}')

        FairnessMetric.objects.bulk_create(fairness_records)
        self.stdout.write(self.style.SUCCESS(f'Stored {len(fairness_records)} fairness metric rows.'))