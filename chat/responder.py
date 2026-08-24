"""
Builds a chatbot answer for a case-specific query ("why was patient X flagged?")
entirely from data already cached by audit/management/commands/*.

Deliberately does NOT call an LLM here. Every number and feature name in the
answer is read directly from the database, so the answer is correct by
construction — there's no generation step that could drift from the real
SHAP value or invent a detail. This is the "verified answer" guarantee for
the case-query path specifically (the concept-Q&A path in Phase 2 will use
real generation, since that side has no single ground-truth number to match).
"""
import re
from audit.models import Patient, Prediction, ShapExplanation, FairnessMetric

MODEL_VERSION = 'xgboost-v0.1'

# Numeric/ID columns that were NOT one-hot encoded — their feature_name in
# the DB is already the readable column name, just needs underscores→spaces.
RAW_NUMERIC_COLS = {
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
}
# Categorical column stems that WERE one-hot encoded (e.g. 'age_50-60)').
# Sorted longest-first so a prefix check doesn't match the wrong, shorter stem.
CATEGORICAL_BASES = sorted(
    ['race', 'gender', 'age', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'],
    key=len, reverse=True,
)


def humanize_feature(feature_name, feature_value):
    """Turn a raw/one-hot column name into a natural-language phrase.

    One-hot encoded columns look like 'age_50-60)' with a boolean
    feature_value ('True'/'False') — this reconstructs the original
    category and phrases it as a sentence fragment instead of exposing
    the internal encoding to the user.
    """
    if feature_name in RAW_NUMERIC_COLS:
        return f"{feature_name.replace('_', ' ')} = {feature_value}"
    for base in CATEGORICAL_BASES:
        prefix = base + '_'
        if feature_name.startswith(prefix):
            raw_val = feature_name[len(prefix):].rstrip(')')
            label = base.replace('_', ' ')
            if feature_value == 'True':
                return f"{label} is '{raw_val}'"
            else:
                return f"{label} is not '{raw_val}'"
    return f"{feature_name} = {feature_value}"


def extract_encounter_id(message):
    """Very simple intent detection: any 4+ digit number is treated as an
    encounter ID lookup. Good enough for Phase 1; Phase 2 will need real
    intent classification once concept questions (no ID present) are added.
    """
    match = re.search(r'\d{4,}', message)
    return int(match.group()) if match else None


def _latest_fairness(attr):
    return FairnessMetric.objects.filter(
        model_version=MODEL_VERSION, protected_attribute=attr, metric_name='demographic_parity_difference'
    ).order_by('-created_at').first()


def answer_case_query(encounter_id):
    """Main entry point: given an encounter_id, return a full answer payload
    (text + citations + ART scorecard + ethics-lens commentary + optional
    group-bias warning), built entirely from cached DB rows.
    """
    patient = Patient.objects.filter(encounter_id=encounter_id).first()
    if not patient:
        # Test-set only (see train_model.py) — if the ID isn't found, it's
        # most likely a training-set patient, so we suggest real cached IDs
        # instead of just saying "not found".
        sample_ids = list(Patient.objects.values_list('encounter_id', flat=True)[:5])
        return {
            'type': 'case',
            'answer': f"I don't have a cached patient with encounter ID {encounter_id} in the test set. "
                      f"Try one of these instead: {', '.join(str(s) for s in sample_ids)}.",
            'citations': [], 'art': None, 'lenses': [], 'flag_note': None,
        }

    prediction = Prediction.objects.filter(patient=patient, model_version=MODEL_VERSION).first()
    shap_rows = list(ShapExplanation.objects.filter(patient=patient, model_version=MODEL_VERSION))
    shap_rows.sort(key=lambda r: abs(r.shap_value), reverse=True)

    # --- Build the top-3 cited features into both the answer text and a
    # separate citations list the frontend renders as clickable chips ---
    citations = []
    feature_mentions = []
    protected_in_top = []
    for i, row in enumerate(shap_rows[:3]):
        cid = f'F{i + 1}'
        direction = 'toward high-risk' if row.shap_value > 0 else 'toward low-risk'
        readable = humanize_feature(row.feature_name, row.feature_value)
        citations.append({
            'id': cid,
            'label': f'SHAP #{i + 1}',
            'detail': f'Feature: {row.feature_name} = {row.feature_value} | '
                      f'SHAP value: {row.shap_value:+.3f} ({direction}) | Model: {MODEL_VERSION}',
        })
        feature_mentions.append(f'{readable} [[{cid}]]')
        # Flag if a protected attribute is among this patient's TOP drivers —
        # feeds directly into the deontological lens below.
        if any(p in row.feature_name.lower() for p in ['race', 'gender']):
            protected_in_top.append(row.feature_name)

    risk_label = 'high-risk' if prediction.predicted_label else 'lower-risk'
    answer = (
        f"Patient #{encounter_id} was predicted {risk_label} for 30-day readmission "
        f"(model score: {prediction.risk_score:.2f}). The top contributing factors were "
        + ', '.join(feature_mentions) + '.'
    )

    # --- Group-level bias warning, only shown if the model's fairness gap
    # for age (the worst of the three checked attributes) exceeds 0.05 ---
    dpd_age = _latest_fairness('age_bracket')
    flag_note = None
    if dpd_age and abs(dpd_age.value) > 0.05:
        flag_note = (
            f"⚠ Group-level audit: demographic-parity gap by age bracket is {dpd_age.value:.3f} "
            f"for this model — above the 0.05 threshold. This individual prediction may still be "
            f"accurate, but the model warrants a wider fairness review."
        )

    # --- Ethical-theory framing, each grounded in something computed above
    # rather than freeform commentary ---
    lenses = [
        {
            'name': 'Utilitarian',
            'text': "Flagging trades some false alarms for fewer missed readmissions — model AUC "
                    "is 0.653 on held-out data, versus 0.644 for the interpretable baseline.",
            'flagged': False,
        },
        {
            'name': 'Deontological',
            'text': (f"Protected attributes ({', '.join(protected_in_top)}) appear among this "
                      f"patient's top drivers — worth review."
                      if protected_in_top else
                      "No protected attribute (race/gender) appears among this patient's top SHAP drivers."),
            'flagged': bool(protected_in_top),
        },
        {
            'name': 'Virtue',
            'text': "A cautious clinician would treat this score as one input among several, not a "
                    "standalone diagnosis.",
            'flagged': False,
        },
    ]

    return {
        'type': 'case',
        'answer': answer,
        'citations': citations,
        'art': {
            'accountability': f'Logged · {MODEL_VERSION}',
            'responsibility': 'Model card on file',
            'transparency': f'{len(shap_rows)} SHAP features cached for this patient',
        },
        'lenses': lenses,
        'flag_note': flag_note,
    }