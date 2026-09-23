"""
Management command: computes Explanation Disparity - how differently the
model's SHAP-based reasoning behaves for each demographic subgroup, per
feature, compared to the overall population.

This is distinct from Fairlearn's fairness audit (run_audit.py), which checks
whether PREDICTIONS are fair across groups. This command checks whether
EXPLANATIONS are fair - whether the model weighs the same factor very
differently depending on who the patient is, even when outcome-level fairness
looks acceptable.

Run with: python manage.py analyze_explanation_disparity
(after train_model and run_audit have already run)
"""
import numpy as np
import pandas as pd
import joblib
import shap
from scipy.stats import wasserstein_distance
from django.core.management.base import BaseCommand
from audit.models import ExplanationDisparity
from audit.pipeline import load_and_clean, build_features, split
from chat.responder import _base_key, RAW_NUMERIC_COLS, CATEGORICAL_BASES

MODEL_VERSION = 'xgboost-v0.1'
MIN_SUBGROUP_SIZE = 20  # skip subgroups too small for a meaningful distribution comparison


class Command(BaseCommand):
    help = 'Compute per-subgroup SHAP-explanation disparity (Explanation Disparity Analysis)'

    def handle(self, *args, **options):
        self.stdout.write('Reloading data + rebuilding the same train/test split ...')
        df, dropped = load_and_clean()
        X, y = build_features(df)
        X_train, X_test, y_train, y_test, df_train, df_test = split(df, X, y)

        xgb = joblib.load('audit/ml_artifacts/xgboost_v0.1.pkl')
        feature_names = list(X_test.columns)

        self.stdout.write('Computing full SHAP matrix for the test set (may take a minute) ...')
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_test)  # shape: (n_patients, n_onehot_features)

        # --- Collapse one-hot SHAP columns back to interpretable base features ---
        # e.g. all 'age_50-60)', 'age_60-70)', ... columns get summed into one
        # 'age' contribution per patient, matching the same grouping already
        # used to explain individual patients in chat/responder.py.
        self.stdout.write('Aggregating SHAP values by base feature ...')
        base_names = sorted(set(_base_key(f) for f in feature_names))
        base_contrib = pd.DataFrame(0.0, index=X_test.index, columns=base_names)

        for j, fname in enumerate(feature_names):
            base = _base_key(fname)
            base_contrib[base] = base_contrib[base].values + shap_values[:, j]

        # --- Group patients by each protected attribute ---
        sensitive_cols = {
            'race': df_test['race'].fillna('Unknown').values,
            'gender': df_test['gender'].values,
            'age_bracket': df_test['age'].values,
        }

        ExplanationDisparity.objects.filter(model_version=MODEL_VERSION).delete()
        records = []

        for attr_name, groups in sensitive_cols.items():
            groups = np.array(groups)
            unique_subgroups = np.unique(groups)

            for base_feature in base_names:
                all_values = base_contrib[base_feature].values
                global_mean = float(np.mean(all_values))

                for subgroup in unique_subgroups:
                    mask = groups == subgroup
                    n = int(mask.sum())
                    if n < MIN_SUBGROUP_SIZE:
                        continue

                    subgroup_values = all_values[mask]
                    distance = float(wasserstein_distance(subgroup_values, all_values))

                    records.append(ExplanationDisparity(
                        model_version=MODEL_VERSION,
                        feature_name=base_feature,
                        protected_attribute=attr_name,
                        subgroup=str(subgroup),
                        subgroup_mean_shap=float(np.mean(subgroup_values)),
                        global_mean_shap=global_mean,
                        wasserstein_distance=distance,
                        n=n,
                    ))

            self.stdout.write(f'  {attr_name}: done')

        ExplanationDisparity.objects.bulk_create(records, batch_size=1000)
        self.stdout.write(self.style.SUCCESS(f'Stored {len(records)} explanation-disparity rows.'))

        # Print the single most divergent finding straight to the terminal
        top = max(records, key=lambda r: r.wasserstein_distance)
        self.stdout.write(self.style.SUCCESS(
            f'\nTop finding: feature "{top.feature_name}" diverges most for '
            f'{top.protected_attribute}="{top.subgroup}" '
            f'(distance={top.wasserstein_distance:.4f}, n={top.n})'
        ))