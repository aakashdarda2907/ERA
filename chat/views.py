"""
HTTP layer: renders the chat page and handles the /ask/ POST endpoint.
All actual answer-building logic lives in responder.py — this file only
handles request/response plumbing.
"""
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .responder import answer_case_query, extract_encounter_id


def index(request):
    """Serves the chat UI page."""
    return render(request, 'chat/index.html')


@csrf_exempt  # dev-only convenience — re-enable proper CSRF handling before any real deployment
def ask(request):
    """Receives a user message, routes it to a query handler, returns JSON.

    Currently only handles case-specific queries (message contains an
    encounter ID). Concept questions (no ID present) return a placeholder —
    that path gets built in Phase 2 once the RAG/knowledge-base layer exists.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    body = json.loads(request.body)
    message = body.get('message', '')
    encounter_id = extract_encounter_id(message)
    if encounter_id:
        payload = answer_case_query(encounter_id)
    else:
        payload = {
            'type': 'unknown',
            'answer': 'Concept Q&A (GDPR, ethics theories, etc.) isn\'t wired up yet — that\'s Phase 2. '
                      'Try asking about a specific patient, e.g. "why was patient 12522 flagged?"',
            'citations': [], 'art': None, 'lenses': [], 'flag_note': None,
        }
    return JsonResponse(payload)