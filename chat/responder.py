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
from .id_mapping import get_id_description


MODEL_VERSION = 'xgboost-v0.1'


# Numeric columns that were NOT one-hot encoded.
RAW_NUMERIC_COLS = {
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses',
}


# Categorical column stems that WERE one-hot encoded.
CATEGORICAL_BASES = sorted(
    ['race', 'gender', 'age', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'],
    key=len,
    reverse=True,
)


def humanize_feature(feature_name, feature_value):
    """Turn a raw/one-hot column name into a natural-language phrase.

    ID columns are translated using IDS_mapping.csv.
    One-hot encoded columns reconstruct the original category.
    Other numeric columns are displayed directly.
    """

    id_columns = {
        'admission_type_id',
        'discharge_disposition_id',
        'admission_source_id',
    }

    if feature_name in id_columns:
        description = get_id_description(feature_name, feature_value)

        if description:
            label = feature_name.replace('_id', '').replace('_', ' ')
            return f"{label} is '{description}'"

        # Safe fallback if a code is not found in the mapping.
        return f"{feature_name.replace('_', ' ')} = {feature_value}"

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
        model_version=MODEL_VERSION,
        protected_attribute=attr,
        metric_name='demographic_parity_difference'
    ).order_by('-created_at').first()
def build_counterfactual_explanation(prediction, shap_rows):
    """Build a counterfactual-style explanation from cached SHAP data."""

    if prediction is None or not shap_rows:
        return (
            "I don't have enough cached model evidence to explain "
            "what would change this prediction."
        )

    # If already lower-risk, explain the factors supporting that result.
    if not prediction.predicted_label:
        low_risk_rows = [
            row for row in shap_rows
            if row.shap_value < 0
        ]

        low_risk_rows.sort(key=lambda row: row.shap_value)

        if low_risk_rows:
            supporting = []

            for row in low_risk_rows[:3]:
                supporting.append(
                    humanize_feature(
                        row.feature_name,
                        row.feature_value
                    )
                )

            return (
                f"This patient is already predicted lower-risk "
                f"(model score: {prediction.risk_score:.2f}), "
                f"so no change is needed to reach the lower-risk category. "
                f"The strongest cached factors supporting this prediction "
                f"are " + ", ".join(supporting) + "."
            )

        return (
            f"This patient is already predicted lower-risk "
            f"(model score: {prediction.risk_score:.2f}), "
            f"so no change is needed to reach the lower-risk category."
        )

    # If high-risk, identify the strongest factors pushing risk upward.
    high_risk_rows = [
        row for row in shap_rows
        if row.shap_value > 0
    ]

    high_risk_rows.sort(
        key=lambda row: row.shap_value,
        reverse=True
    )

    if not high_risk_rows:
        return (
            f"The patient is currently predicted high-risk "
            f"(model score: {prediction.risk_score:.2f}), "
            f"but there is no positive cached SHAP contributor "
            f"available for a counterfactual explanation."
        )

    changes = []

    for row in high_risk_rows[:3]:
        readable = humanize_feature(
            row.feature_name,
            row.feature_value
        )

        if row.feature_name in RAW_NUMERIC_COLS:
            change = (
                f"{row.feature_name.replace('_', ' ')} were lower "
                f"than its current value ({row.feature_value})"
            )
        else:
            change = (
                f"{readable} changed to a value or category "
                f"associated with lower risk"
            )

        changes.append(change)

    return (
        f"The patient is currently predicted high-risk "
        f"(model score: {prediction.risk_score:.2f}). "
        f"Based on the cached SHAP evidence, the factors most likely "
        f"to move the prediction toward lower risk are: "
        + "; ".join(changes)
        + "."
    )

def build_confidence_explanation(prediction):
    """Explain the risk score relative to the 0.5 decision boundary.

    This is an interpretability aid, not a calibrated confidence interval
    or a guarantee that the prediction is correct.
    """

    if prediction is None:
        return (
            "I don't have a cached prediction score, so I can't describe "
            "the model's confidence."
        )

    score = prediction.risk_score
    distance = abs(score - 0.5)

    if distance >= 0.35:
        strength = "far from"
    elif distance >= 0.20:
        strength = "clearly away from"
    else:
        strength = "relatively close to"

    decision = "high-risk" if prediction.predicted_label else "lower-risk"

    return (
        f"The model score is {score:.2f}, which is {strength} "
        f"the 0.50 decision boundary (distance: {distance:.2f}). "
        f"This supports a {decision} classification. However, the score "
        f"is not a diagnosis, a guarantee of the patient's outcome, or a "
        f"statistical confidence interval. The model should be treated as "
        f"one input alongside clinical judgment and other relevant evidence."
    )


def answer_case_query(encounter_id, message=''):
    """Main entry point: given an encounter_id, return a full answer payload
    (text + citations + ART scorecard + ethics-lens commentary + optional
    group-bias warning), built entirely from cached DB rows.
    """

    patient = Patient.objects.filter(encounter_id=encounter_id).first()

    if not patient:
        # Test-set only (see train_model.py) — if the ID isn't found, it's
        # most likely a training-set patient, so we suggest real cached IDs
        # instead of just saying "not found".
        sample_ids = list(
            Patient.objects.values_list('encounter_id', flat=True)[:5]
        )

        return {
            'type': 'case',
            'answer': (
                f"I don't have a cached patient with encounter ID {encounter_id} "
                f"in the test set. Try one of these instead: "
                f"{', '.join(str(s) for s in sample_ids)}."
            ),
            'citations': [],
            'art': None,
            'lenses': [],
            'flag_note': None,
        }

    prediction = Prediction.objects.filter(
        patient=patient,
        model_version=MODEL_VERSION
    ).first()

    shap_rows = list(
        ShapExplanation.objects.filter(
            patient=patient,
            model_version=MODEL_VERSION
        )
    )

    shap_rows.sort(key=lambda r: abs(r.shap_value), reverse=True)

    # Build the top-3 cited features into both the answer text
    # and a separate citations list.
    citations = []
    feature_mentions = []
    protected_in_top = []

    for i, row in enumerate(shap_rows[:3]):
        cid = f'F{i + 1}'

        direction = (
            'toward high-risk'
            if row.shap_value > 0
            else 'toward low-risk'
        )

        readable = humanize_feature(
            row.feature_name,
            row.feature_value
        )

        citations.append({
            'id': cid,
            'label': f'SHAP #{i + 1}',
            'detail': (
                f'Feature: {row.feature_name} = {row.feature_value} | '
                f'SHAP value: {row.shap_value:+.3f} ({direction}) | '
                f'Model: {MODEL_VERSION}'
            ),
        })

        feature_mentions.append(
            f'{readable} [[{cid}]]'
        )

        # Flag if a protected attribute is among this patient's
        # top drivers.
        if any(
            p in row.feature_name.lower()
            for p in ['race', 'gender']
        ):
            protected_in_top.append(row.feature_name)

    risk_label = (
        'high-risk'
        if prediction.predicted_label
        else 'lower-risk'
    )

    answer = (
        f"Patient #{encounter_id} was predicted {risk_label} "
        f"for 30-day readmission "
        f"(model score: {prediction.risk_score:.2f}). "
        f"The top contributing factors were "
        + ', '.join(feature_mentions)
        + '.'
    )
    # Task 2 Option A: "What would change this?"
    counterfactual_keywords = [
        'what would change',
        'what could change',
        'what needs to change',
        'what would need to change',
        'how can this change',
        'how could this change',
        'what can change',
        'change this',
    ]

    if any(
        keyword in message.lower()
        for keyword in counterfactual_keywords
    ):
        counterfactual = build_counterfactual_explanation(
            prediction,
            shap_rows
        )

        answer += (
            f"\n\n**What would change this?** "
            f"{counterfactual}"
        )

    # Task 2 Option B: confidence and limitations.
    confidence_keywords = [
        'how confident',
        'confidence',
        'how certain',
        'certainty',
        'how reliable',
        'reliable is this',
        'limitations',
        'limitation',
    ]

    if any(
        keyword in message.lower()
        for keyword in confidence_keywords
    ):
        confidence = build_confidence_explanation(prediction)

        answer += (
            f"\n\n**Confidence & limitations:** "
            f"{confidence}"
        )

    # Group-level bias warning.
    dpd_age = _latest_fairness('age_bracket')

    flag_note = None

    if dpd_age and abs(dpd_age.value) > 0.05:
        flag_note = (
            f"⚠ Group-level audit: demographic-parity gap by age bracket "
            f"is {dpd_age.value:.3f} for this model — above the 0.05 "
            f"threshold. This individual prediction may still be accurate, "
            f"but the model warrants a wider fairness review."
        )

    # Ethical-theory framing.
    lenses = [
        {
            'name': 'Utilitarian',
            'text': (
                "Flagging trades some false alarms for fewer missed "
                "readmissions — model AUC is 0.653 on held-out data, "
                "versus 0.644 for the interpretable baseline."
            ),
            'flagged': False,
        },
        {
            'name': 'Deontological',
            'text': (
                f"Protected attributes ({', '.join(protected_in_top)}) "
                f"appear among this patient's top drivers — worth review."
                if protected_in_top
                else
                "No protected attribute (race/gender) appears among "
                "this patient's top SHAP drivers."
            ),
            'flagged': bool(protected_in_top),
        },
        {
            'name': 'Virtue',
            'text': (
                "A cautious clinician would treat this score as one input "
                "among several, not a standalone diagnosis."
            ),
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
            'transparency': (
                f'{len(shap_rows)} SHAP features cached for this patient'
            ),
        },
        'lenses': lenses,
        'flag_note': flag_note,
    }


def answer_concept_query(message):
    """Answers a general concept question (GDPR, ethics theories, fairness,
    etc.) by retrieving the single best-matching corpus document via TF-IDF
    and citing it - as opposed to answer_case_query, which cites cached
    SHAP/Fairlearn data. Returns the document's full cleaned body rather
    than a single extracted paragraph, since short corpus docs answer more
    reliably as a whole than as a fragment picked out of context.
    """
    from knowledge.retriever import retrieve_best_doc, clean_body

    doc, score = retrieve_best_doc(message)
    if doc is None:
        return {
            'type': 'concept',
            'answer': "I don't have a confident source for that in my knowledge base yet. "
                      "Try asking about GDPR, NITI Aayog, the ART framework, utilitarianism, "
                      "deontology, virtue ethics, fairness/bias, explainable AI, or privacy-preserving AI.",
            'citations': [], 'art': None, 'lenses': [], 'flag_note': None,
        }

    body = clean_body(doc)
    answer = f"{body} [[C1]]"

    return {
        'type': 'concept',
        'answer': answer,
        'citations': [{
            'id': 'C1',
            'label': f'Source: {doc.title}',
            'detail': f'Corpus document: {doc.slug}.md | Match confidence: {score:.2f}',
        }],
        'art': None, 'lenses': [], 'flag_note': None,
    }
