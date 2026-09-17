"""
HTTP layer: renders the chat page and handles the /ask/ POST endpoint.
Routes to case-query (patient ID present) or concept-query (no ID -
general knowledge-base question) - all actual answer-building logic lives
in responder.py.

Remembers the last patient ID discussed in this browser session, so a
follow-up like "how confident is this?" (with no ID repeated) still routes
to the case-query path instead of falling through to concept Q&A.
"""
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .responder import answer_case_query, answer_concept_query, extract_encounter_id

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
    """Serves the chat UI page."""
    return render(request, 'chat/index.html')


@csrf_exempt  # dev-only convenience - re-enable proper CSRF handling before any real deployment
def ask(request):
    """Receives a user message and routes it to the right query handler."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    body = json.loads(request.body)
    message = body.get('message', '')
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
            payload = answer_concept_query(message)

    return JsonResponse(payload)