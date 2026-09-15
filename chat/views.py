
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
from audit.models import Patient
from .responder import (
    answer_case_query,
    answer_comparison_query,
    answer_concept_query,
    extract_comparison_ids,
    extract_encounter_id,
)


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


@csrf_exempt
def ask(request):
    """Receives a user message and routes it to the right query handler."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    body = json.loads(request.body)
    message = body.get('message', '')

    comparison_ids = extract_comparison_ids(message)

    if comparison_ids:
        payload = answer_comparison_query(*comparison_ids)
    else:
        encounter_id = extract_encounter_id(message)

        if encounter_id:
            payload = answer_case_query(encounter_id)
        else:
            payload = answer_concept_query(message)

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