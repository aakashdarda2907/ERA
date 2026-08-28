"""
HTTP layer: renders the chat page and handles the /ask/ POST endpoint.
All actual answer-building logic lives in responder.py — this file only
handles request/response plumbing.
"""
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from audit.models import Patient
from .responder import answer_case_query, extract_encounter_id


def index(request):
    """Serves the chat UI page.

    Also pulls a few real cached encounter IDs to seed the suggested-question
    chips, so the chips are guaranteed to hit a valid patient instead of
    pointing at an ID that may not exist in this environment's cache.
    """
    sample_ids = list(
        Patient.objects.order_by('?').values_list('encounter_id', flat=True)[:3]
    )
    suggestions = [f'Why was patient {eid} flagged?' for eid in sample_ids]
    return render(request, 'chat/index.html', {'suggestions': suggestions})


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