"""
Management command: measures Explanation Stability - whether a patient's
top SHAP-driven factors survive a small, realistic perturbation to their
data (number_inpatient +1), independent of whether the final classification
changes.

This is grounded in known XAI-fragility findings (Ghorbani et al. 2019,
"Interpretation of Neural Networks is Fragile"; Alvarez-Melis & Jaakkola
2018): explanations can shift dramatically under tiny input changes even
when the prediction barely moves - meaning a convincing-looking explanation
is not automatically a trustworthy one.

Run with: python manage.py analyze_explanation_stability
(after train_model and run_audit have already run)
"""
import numpy as np
import joblib
import shap
from django.core.management.base import BaseCommand
from audit.models import ExplanationStability
from audit.pipeline import load_and_clean, build_features, split
from chat.responder import _base_key

MODEL_VERSION = 'xgboost-v0.1'
SAMPLE_SIZE = 1500       # patients sampled for this audit - full test set would be slower with no added insight
PERTURBATION_COL = 'number_inpatient'
PERTURBATION_DESC = 'number_inpatient +1'
RANDOM_SEED = 42


def top3_base_features(shap_row, feature_names):
    """Aggregate one row's SHAP values into base features (undoing one-hot
    encoding, same grouping used everywhere else in the app) and return the
    3 base features with the largest absolute contribution.
    """
    agg = {}
    for j, fname in enumerate(feature_names):
        base = _base_key(fname)
        agg[base] = agg.get(base, 0.0) + shap_row[j]
    ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [name for name, _ in ranked[:3]]


class Command(BaseCommand):
    help = 'Measure explanation stability under a small realistic perturbation'

    def handle(self, *args, **options):
        self.stdout.write('Reloading data + rebuilding the same train/test split ...')
        df, dropped = load_and_clean()
        X, y = build_features(df)
        X_train, X_test, y_train, y_test, df_train, df_test = split(df, X, y)

        feature_names = list(X_test.columns)
        xgb = joblib.load('audit/ml_artifacts/xgboost_v0.1.pkl')

        rng = np.random.RandomState(RANDOM_SEED)
        n = min(SAMPLE_SIZE, len(X_test))
        sample_positions = rng.choice(len(X_test), size=n, replace=False)

        X_sample = X_test.iloc[sample_positions].reset_index(drop=True)
        df_sample = df_test.iloc[sample_positions].reset_index(drop=True)

        X_perturbed = X_sample.copy()
        X_perturbed[PERTURBATION_COL] = X_perturbed[PERTURBATION_COL] + 1

        self.stdout.write(f'Scoring {n} sampled patients (original + perturbed) ...')
        original_scores = xgb.predict_proba(X_sample)[:, 1]
        perturbed_scores = xgb.predict_proba(X_perturbed)[:, 1]

        self.stdout.write('Computing SHAP for original and perturbed versions (may take a minute) ...')
        explainer = shap.TreeExplainer(xgb)
        shap_original = explainer.shap_values(X_sample)
        shap_perturbed = explainer.shap_values(X_perturbed)

        ExplanationStability.objects.filter(model_version=MODEL_VERSION, perturbation=PERTURBATION_DESC).delete()

        records = []
        changed_count = 0
        flipped_count = 0

        for i in range(n):
            encounter_id = int(df_sample.iloc[i]['encounter_id'])
            orig_score = float(original_scores[i])
            pert_score = float(perturbed_scores[i])
            flipped = (orig_score >= 0.5) != (pert_score >= 0.5)

            top3_before = top3_base_features(shap_original[i], feature_names)
            top3_after = top3_base_features(shap_perturbed[i], feature_names)

            shared = len(set(top3_before) & set(top3_after))
            union = len(set(top3_before) | set(top3_after))
            jaccard = shared / union if union else 1.0

            if shared < 3:
                changed_count += 1
            if flipped:
                flipped_count += 1

            records.append(ExplanationStability(
                model_version=MODEL_VERSION,
                encounter_id=encounter_id,
                perturbation=PERTURBATION_DESC,
                original_score=orig_score,
                perturbed_score=pert_score,
                score_shift=abs(pert_score - orig_score),
                classification_flipped=flipped,
                original_top3=', '.join(top3_before),
                perturbed_top3=', '.join(top3_after),
                shared_top3_count=shared,
                jaccard_similarity=jaccard,
            ))

        ExplanationStability.objects.bulk_create(records, batch_size=1000)

        pct_changed = 100 * changed_count / n
        pct_flipped = 100 * flipped_count / n
        self.stdout.write(self.style.SUCCESS(f'Stored {n} stability records.'))
        self.stdout.write(self.style.SUCCESS(
            f'\n{pct_changed:.1f}% of sampled patients had at least one top-3 factor change '
            f'from a +1 change in {PERTURBATION_COL}.'
        ))
        self.stdout.write(self.style.SUCCESS(
            f'{pct_flipped:.1f}% had their actual risk classification flip from the same tiny change.'
        ))