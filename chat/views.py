"""
HTTP layer: renders the chat page and handles the /ask/ POST endpoint.
Routes to case-query (patient ID present) or concept-query (no ID -
general knowledge-base question) - all actual answer-building logic lives
in responder.py.
"""
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .responder import answer_case_query, answer_concept_query, extract_encounter_id


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
        payload = answer_case_query(encounter_id)
    else:
        payload = answer_concept_query(message)
    return JsonResponse(payload)