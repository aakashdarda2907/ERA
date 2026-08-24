"""
Shared data-loading and preprocessing logic for the readmission-risk audit engine.

Both train_model.py and run_audit.py import from here, and both use the SAME
random_state when splitting — this guarantees the test set is identical across
commands, so a SHAP explanation computed in run_audit.py always lines up with
the correct cached Patient row from train_model.py. If these two files each
built their own split independently, SHAP values could silently attach to the
wrong patient.
"""
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# Discharge codes meaning the patient died or was moved to hospice care.
# A patient who died cannot be "readmitted" — leaving these rows in would
# corrupt the target variable (readmitted_label), so they're dropped entirely
# rather than treated as a normal "NO" outcome.
DEATH_HOSPICE_CODES = {11, 13, 14, 19, 20, 21}

# Columns used as model inputs. Deliberately excludes 'weight' (~97% missing
# in this dataset) and identifier columns (encounter_id, patient_nbr) which
# carry no predictive signal and would just be memorized as noise.
FEATURE_COLS = [
    'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses',
    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
]


def load_and_clean(csv_path='data/diabetic_data.csv'):
    """Load the raw CSV, drop death/hospice discharges, binarize the target.

    The UCI dataset encodes missing values as the literal string '?' rather
    than leaving cells blank, so na_values='?' is required or pandas would
    treat '?' as a normal category value instead of missing data.

    readmitted_label collapses the original 3-way column ('NO', '>30', '<30')
    into a binary target: 1 if readmitted within 30 days, else 0. This is the
    standard framing used in published work on this dataset, since <30-day
    readmission is the clinically actionable threshold hospitals are scored on.
    """
    df = pd.read_csv(csv_path, na_values='?', low_memory=False)
    before = len(df)
    df = df[~df['discharge_disposition_id'].isin(DEATH_HOSPICE_CODES)].copy()
    dropped = before - len(df)
    df['readmitted_label'] = (df['readmitted'] == '<30').astype(int)
    return df, dropped


def build_features(df):
    """One-hot encode categorical columns and sanitize names for XGBoost.

    XGBoost's DMatrix rejects '[', ']', '<' in feature names. Several raw
    values (e.g. the age column's '[70-80)' brackets) contain these
    characters, so after one-hot encoding produces column names like
    'age_[70-80)', we strip the offending characters — this only affects the
    column LABEL, not the underlying data, so it's safe for both models.
    """
    X_raw = df[FEATURE_COLS].copy()
    y = df['readmitted_label']

    cat_cols = X_raw.select_dtypes(include='object').columns
    X_raw[cat_cols] = X_raw[cat_cols].fillna('Unknown')
    X = pd.get_dummies(X_raw, columns=list(cat_cols), drop_first=False)

    X.columns = [re.sub(r'[\[\]<]', '', str(c)) for c in X.columns]
    return X, y


def split(df, X, y, test_size=0.2, random_state=42):
    """80/20 stratified split, keeping y balanced across train/test.

    df is split alongside X/y (not separately) so df_test rows stay
    index-aligned with X_test/y_test — this is what lets run_audit.py map
    each SHAP row back to the correct encounter_id.
    """
    return train_test_split(X, y, df, test_size=test_size, random_state=random_state, stratify=y)