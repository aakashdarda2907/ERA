"""
Management command: trains both models and caches results to the database.

Run with: python manage.py train_model

Trains two models on purpose, not one — this pair is the concrete evidence
for the "black box problem" (syllabus Unit IV): LogisticRegression is
interpretable by construction (you can read its coefficients directly) but
weaker; XGBoost is stronger but opaque, which is exactly why the SHAP layer
in run_audit.py exists — to make the stronger model explainable too.
"""
import os
import joblib
import pandas as pd
from django.db import transaction
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from django.core.management.base import BaseCommand
from audit.models import Patient, Prediction
from audit.pipeline import load_and_clean, build_features, split


class Command(BaseCommand):
    help = 'Load diabetic_data.csv, clean it, train LogReg + XGBoost, cache predictions to DB'

    def handle(self, *args, **options):
        self.stdout.write('Loading data/diabetic_data.csv ...')
        df, dropped = load_and_clean()
        self.stdout.write(f'Dropped {dropped} death/hospice rows, {len(df)} remain')

        X, y = build_features(df)
        X_train, X_test, y_train, y_test, df_train, df_test = split(df, X, y)

        os.makedirs('audit/ml_artifacts', exist_ok=True)

        # --- Interpretable baseline ---
        self.stdout.write('Training LogisticRegression ...')
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)
        logreg_probs = logreg.predict_proba(X_test)[:, 1]
        self.stdout.write(f'LogReg ROC-AUC: {roc_auc_score(y_test, logreg_probs):.3f}')
        joblib.dump(logreg, 'audit/ml_artifacts/logreg_v0.1.pkl')

        # --- Black-box model, the one the chatbot actually explains via SHAP ---
        self.stdout.write('Training XGBoost ...')
        xgb = XGBClassifier(eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)
        xgb_probs = xgb.predict_proba(X_test)[:, 1]
        self.stdout.write(f'XGBoost ROC-AUC: {roc_auc_score(y_test, xgb_probs):.3f}')
        joblib.dump(xgb, 'audit/ml_artifacts/xgboost_v0.1.pkl')

        # Column order must match exactly when SHAP re-loads this model later,
        # since sklearn/XGBoost matches inputs by position, not by name.
        joblib.dump(list(X.columns), 'audit/ml_artifacts/feature_columns.pkl')

        # --- Cache every test-set patient + both models' predictions to the DB ---
        # Only the test set is cached (not train) because those are the only
        # patients the chatbot should ever discuss — discussing a training
        # patient would let the model "know the answer" in a way that
        # misrepresents what it can actually generalize to.
        self.stdout.write('Caching patients + predictions to DB (test set only) ...')
        Patient.objects.all().delete()
        Prediction.objects.all().delete()

        # transaction.atomic() batches ~20,000 rows into a single commit
        # instead of one commit per row — without it, SQLite's per-write
        # fsync makes this loop take several minutes longer.
        with transaction.atomic():
            for i, (idx, row) in enumerate(df_test.iterrows()):
                patient, _ = Patient.objects.update_or_create(
                    encounter_id=row['encounter_id'],
                    defaults=dict(
                        patient_nbr=row['patient_nbr'],
                        race=row['race'] if pd.notna(row['race']) else None,
                        gender=row['gender'],
                        age_bracket=row['age'],
                        time_in_hospital=row['time_in_hospital'],
                        num_lab_procedures=row['num_lab_procedures'],
                        num_procedures=row['num_procedures'],
                        num_medications=row['num_medications'],
                        number_outpatient=row['number_outpatient'],
                        number_emergency=row['number_emergency'],
                        number_inpatient=row['number_inpatient'],
                        number_diagnoses=row['number_diagnoses'],
                        max_glu_serum=row['max_glu_serum'] if pd.notna(row['max_glu_serum']) else None,
                        a1c_result=row['A1Cresult'] if pd.notna(row['A1Cresult']) else None,
                        change_med=row['change'],
                        diabetes_med=row['diabetesMed'],
                        readmitted_raw=row['readmitted'],
                        readmitted_label=row['readmitted_label'],
                    )
                )
                Prediction.objects.update_or_create(
                    patient=patient, model_version='logreg-v0.1',
                    defaults=dict(risk_score=float(logreg_probs[i]), predicted_label=bool(logreg_probs[i] >= 0.5)),
                )
                Prediction.objects.update_or_create(
                    patient=patient, model_version='xgboost-v0.1',
                    defaults=dict(risk_score=float(xgb_probs[i]), predicted_label=bool(xgb_probs[i] >= 0.5)),
                )
                if i % 2000 == 0:
                    self.stdout.write(f'  ...{i} rows cached')

        self.stdout.write(self.style.SUCCESS(f'Done. {Patient.objects.count()} patients cached.'))