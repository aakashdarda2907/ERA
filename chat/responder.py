"""
Builds a chatbot answer for a case-specific query ("why was patient X flagged?")
entirely from data already cached by audit/management/commands/*.

Deliberately does NOT call an LLM here. Every number and feature name in the
answer is read directly from the database, so the answer is correct by
construction - there's no generation step that could drift from the real
SHAP value or invent a detail. This is the "verified answer" guarantee for
the case-query path specifically.
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

ID_COLUMNS = {
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
}

# Plain-English label for every feature the model can cite - this is the
# single biggest lever for readability: nobody outside a data science
# class knows what "num_lab_procedures" means, but everyone understands
# "how many lab tests were done during this stay".
FEATURE_PLAIN = {
    'number_inpatient': 'how many times they were admitted to a hospital before',
    'number_emergency': 'how many emergency room visits they have had',
    'number_outpatient': 'how many outpatient visits they have had',
    'num_medications': 'how many medications they were given during this stay',
    'num_procedures': 'how many medical procedures were done during this stay',
    'num_lab_procedures': 'how many lab tests were done during this stay',
    'time_in_hospital': 'how many days they stayed in the hospital',
    'number_diagnoses': 'how many different diagnoses were recorded for them',
    'admission_type_id': 'how they were admitted to the hospital',
    'discharge_disposition_id': 'where they were sent after this hospital stay',
    'admission_source_id': 'where this admission was referred from',
    'age': 'their age group',
    'race': 'their recorded race',
    'gender': 'their gender',
    'max_glu_serum': 'their glucose test result',
    'A1Cresult': 'their A1C (blood sugar) test result',
    'change': 'whether their diabetes medicine was changed during this stay',
    'diabetesMed': 'whether they are on diabetes medication',
}


def humanize_feature(feature_name, feature_value):
    """Turn a raw/one-hot column name into a natural-language phrase.
    Used for the technical citation detail text (shown on click), not the
    main plain-English answer - see _plain_reason for that.
    """
    if feature_name in ID_COLUMNS:
        description = get_id_description(feature_name, feature_value)
        if description:
            label = feature_name.replace('_id', '').replace('_', ' ')
            return f"{label} is '{description}'"
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


def _base_key(feature_name):
    """Map a raw/one-hot column name back to its FEATURE_PLAIN key."""
    if feature_name in ID_COLUMNS or feature_name in RAW_NUMERIC_COLS:
        return feature_name
    for base in CATEGORICAL_BASES:
        if feature_name.startswith(base + '_'):
            return base
    return feature_name


def _value_clause(feature_name, feature_value):
    """The 'value' half of a plain-English sentence, e.g. '0' for a count,
    or "'Discharged to home'" for a mapped ID, or "'50-60'" for a bracket.
    """
    if feature_name in ID_COLUMNS:
        desc = get_id_description(feature_name, feature_value)
        return f"'{desc}'" if desc else str(feature_value)

    if feature_name in RAW_NUMERIC_COLS:
        return str(feature_value)

    for base in CATEGORICAL_BASES:
        prefix = base + '_'
        if feature_name.startswith(prefix):
            raw_val = feature_name[len(prefix):].rstrip(')')
            if feature_value == 'True':
                return f"'{raw_val}'"
            return f"something other than '{raw_val}'"

    return str(feature_value)


def _plain_reason(feature_name, feature_value, shap_value, cid):
    """Builds one plain-English bullet explaining a single SHAP-driven
    reason: what the factor is, its value, and how strongly/which way it
    pushed the prediction - no SHAP jargon, no raw column names.
    """
    base = _base_key(feature_name)
    label = FEATURE_PLAIN.get(base, base.replace('_', ' '))
    value_clause = _value_clause(feature_name, feature_value)

    magnitude = abs(shap_value)
    if magnitude >= 0.25:
        strength = 'strongly'
    elif magnitude >= 0.10:
        strength = 'moderately'
    else:
        strength = 'slightly'
    direction_word = 'raised' if shap_value > 0 else 'lowered'

    if feature_name in RAW_NUMERIC_COLS:
        clause = f"{label.capitalize()}: {value_clause}."
    else:
        clause = f"{label.capitalize()} is {value_clause}."

    return f"{clause} This {strength} {direction_word} their overall risk score. [[{cid}]]"

def extract_encounter_id(message):
    """Very simple intent detection: any 4+ digit number is treated as an
    encounter ID lookup.
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
    - a plain-English headline + risk level, a short list of plain-English
    reasons (each backed by a clickable technical citation), an ART
    scorecard, ethics-lens commentary, and an optional group-bias warning.
    """
    patient = Patient.objects.filter(encounter_id=encounter_id).first()

    if not patient:
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
            'citations': [], 'art': None, 'lenses': [], 'flag_note': None,
        }

    prediction = Prediction.objects.filter(
        patient=patient, model_version=MODEL_VERSION
    ).first()

    shap_rows = list(
        ShapExplanation.objects.filter(patient=patient, model_version=MODEL_VERSION)
    )
    shap_rows.sort(key=lambda r: abs(r.shap_value), reverse=True)

    citations = []
    reasons = []
    protected_in_top = []

    for i, row in enumerate(shap_rows[:3]):
        cid = f'F{i + 1}'
        direction = 'toward high-risk' if row.shap_value > 0 else 'toward low-risk'

        citations.append({
            'id': cid,
            'label': f'Evidence #{i + 1}',
            'detail': (
                f'Feature: {row.feature_name} = {row.feature_value} | '
                f'SHAP value: {row.shap_value:+.3f} ({direction}) | '
                f'Model: {MODEL_VERSION}'
            ),
        })

        reasons.append(_plain_reason(row.feature_name, row.feature_value, row.shap_value, cid))

        if any(p in row.feature_name.lower() for p in ['race', 'gender']):
            protected_in_top.append(row.feature_name)

    risk_percent = round(prediction.risk_score * 100)
    if prediction.predicted_label:
        risk_level, risk_class = 'HIGH RISK', 'warn'
        headline_detail = f"About a {risk_percent}% estimated chance of being readmitted within 30 days."
    else:
        risk_level, risk_class = 'LOW RISK', 'ok'
        headline_detail = f"About a {risk_percent}% estimated chance of being readmitted within 30 days."
    # Task 2 (Sanskruti): answer optional follow-up question types using
    # the prediction/SHAP data already computed above.
    counterfactual_keywords = [
        'what would change', 'what could change', 'what needs to change',
        'what would need to change', 'how can this change',
        'how could this change', 'what can change', 'change this',
    ]
    if any(keyword in message.lower() for keyword in counterfactual_keywords):
        reasons.append(build_counterfactual_explanation(prediction, shap_rows))

    confidence_keywords = [
        'how confident', 'confidence', 'how certain', 'certainty',
        'how reliable', 'reliable is this', 'limitations', 'limitation',
    ]
    if any(keyword in message.lower() for keyword in confidence_keywords):
        reasons.append(build_confidence_explanation(prediction))

    # Group-level bias warning, plain English.
    dpd_age = _latest_fairness('age_bracket')
    flag_note = None
    if dpd_age and abs(dpd_age.value) > 0.05:
        flag_note = (
            "⚠ Overall, this model tends to flag some age groups as high-risk "
            "more often than others, by a margin larger than we consider "
            "acceptable. This specific prediction may still be accurate, but "
            "the model as a whole should be reviewed for age-related bias."
        )

    lenses = [
        {
            'name': 'Utilitarian',
            'text': (
                "This model is tuned to catch more real readmission risks than "
                "it misses, even if that means a few extra false alarms - the "
                "goal is fewer missed cases overall, not a perfect score on "
                "every single patient."
            ),
            'flagged': False,
        },
        {
            'name': 'Deontological',
            'text': (
                "This decision was influenced in part by a protected trait "
                "like race or gender - that deserves a closer look."
                if protected_in_top else
                "This decision was based on medical history and hospital-stay "
                "details, not on protected traits like race or gender."
            ),
            'flagged': bool(protected_in_top),
        },
        {
            'name': 'Virtue',
            'text': (
                "Treat this as one extra piece of information for hospital "
                "staff, not a diagnosis. A careful clinician would double-check "
                "it against the full picture before acting on it."
            ),
            'flagged': False,
        },
    ]

    return {
        'type': 'case',
        'headline': f"Patient #{encounter_id} — {risk_level}",
        'headline_detail': headline_detail,
        'risk_class': risk_class,
        'reasons': reasons,
        'citations': citations,
        'art': {
            'accountability': f'Logged · {MODEL_VERSION}',
            'responsibility': 'Model card on file',
            'transparency': f'{len(shap_rows)} pieces of evidence cached for this patient',
        },
        'lenses': lenses,
        'flag_note': flag_note,
    }


def answer_concept_query(message):
    """Answers a general concept question by retrieving the single
    best-matching corpus document via TF-IDF and citing it.
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

def extract_comparison_ids(message):
    """Extract two patient encounter IDs from a comparison request."""
    ids = re.findall(r'\d{4,}', message)
    if len(ids) >= 2:
        return int(ids[0]), int(ids[1])
    return None


def answer_comparison_query(first_id, second_id):
    """Return the full case-query result for two patients side by side.
    Reuses answer_case_query as-is, so comparison answers stay just as
    evidence-grounded as single-patient ones - nothing new is computed.
    """
    return {
        'type': 'comparison',
        'patients': [
            {'encounter_id': first_id, 'data': answer_case_query(first_id)},
            {'encounter_id': second_id, 'data': answer_case_query(second_id)},
        ],
        'citations': [], 'art': {}, 'lenses': [], 'flag_note': '',
    }