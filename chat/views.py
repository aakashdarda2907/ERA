"""
HTTP layer: renders the chat page and handles the /ask/ POST endpoint.
Routes to case-query (patient ID present), comparison-query (two IDs),
or concept-query (no ID - general knowledge-base question) - all actual
answer-building logic lives in responder.py.

Remembers the last patient ID AND the last concept-doc topic discussed in
this browser session, so a follow-up question that doesn't repeat an ID or
share much vocabulary with the original question still routes correctly
instead of falling through to the wrong path or a flat refusal.
"""
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from audit.models import Patient, Prediction
from .responder import (
    answer_case_query,
    answer_comparison_query,
    answer_concept_query,
    extract_comparison_ids,
    extract_encounter_id,
)

MODEL_VERSION = 'xgboost-v0.1'

# Keywords that only make sense as a follow-up about a specific patient -
# used to decide whether a message with no ID should still reuse the last
# discussed encounter ID from this session.
FOLLOWUP_KEYWORDS = [
    'what would change', 'what could change', 'what needs to change',
    'what would need to change', 'how can this change', 'how could this change',
    'what can change', 'change this',
    'how confident', 'confidence', 'how certain', 'certainty',
    'how reliable', 'reliable is this', 'limitations', 'limitation',
    'why', 'this patient', 'this prediction', 'this case',
]


def index(request):
    """Serves the chat UI page.

    Seeds the suggested-question chips with THREE real cached patients -
    one clearly high-risk, one near the 0.5 decision boundary, and one
    clearly low-risk - instead of three random IDs, so a first-time visitor
    immediately sees the full range of what the model actually predicts.
    """
    base_qs = Prediction.objects.filter(model_version=MODEL_VERSION).select_related('patient')

    high = base_qs.order_by('-risk_score').first()
    low = base_qs.order_by('risk_score').first()

    if base_qs.exists():
        medium = (
            base_qs.filter(risk_score__gte=0.45, risk_score__lte=0.55).first()
            or base_qs.filter(risk_score__gte=0.35, risk_score__lte=0.65).first()
            or base_qs.order_by('risk_score')[base_qs.count() // 2]
        )
    else:
        medium = None

    level_meta = {
        'high': ('\U0001F534 High-risk example', 'warn'),
        'medium': ('\U0001F7E1 Medium-risk example', 'mid'),
        'low': ('\U0001F7E2 Low-risk example', 'ok'),
    }

    suggestions = []
    for level, pred in [('high', high), ('medium', medium), ('low', low)]:
        if not pred:
            continue
        label_text, css_class = level_meta[level]
        suggestions.append({
            'label': f'{label_text} ({round(pred.risk_score * 100)}%)',
            'question': f'Why was patient {pred.patient.encounter_id} flagged?',
            'css_class': css_class,
        })

    return render(request, 'chat/index.html', {'suggestions': suggestions})


@csrf_exempt  # dev-only convenience - re-enable proper CSRF handling before any real deployment
def ask(request):
    """Receives a user message and routes it to the right query handler."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    body = json.loads(request.body)
    message = body.get('message', '')

    comparison_ids = extract_comparison_ids(message)
    if comparison_ids:
        payload = answer_comparison_query(*comparison_ids)
        return JsonResponse(payload)

    encounter_id = extract_encounter_id(message)
    if encounter_id:
        request.session['last_encounter_id'] = encounter_id
        payload = answer_case_query(encounter_id, message)
    else:
        last_id = request.session.get('last_encounter_id')
        looks_like_followup = any(kw in message.lower() for kw in FOLLOWUP_KEYWORDS)
        if last_id and looks_like_followup:
            payload = answer_case_query(last_id, message)
        else:
            last_topic = request.session.get('last_concept_topic')
            payload = answer_concept_query(message, last_topic=last_topic)
            if payload.get('topic'):
                request.session['last_concept_topic'] = payload['topic']

    return JsonResponse(payload)


def random_patient(request):
    """Returns a random cached patient's encounter ID as JSON.

    Backs the "Random patient" button on the frontend so the user doesn't
    have to guess a valid encounter ID.
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    patient = Patient.objects.order_by('?').first()
    if not patient:
        return JsonResponse({'error': 'No cached patients available'}, status=404)

    return JsonResponse({'encounter_id': patient.encounter_id})