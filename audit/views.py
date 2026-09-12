"""
HTTP layer for the /report/ page: a read-only audit summary showing
model comparison (ROC-AUC) and the full Fairlearn fairness results.

This view only reads from the database (ModelMetric, FairnessMetric) —
it never re-runs training, SHAP, or Fairlearn. Those are populated by the
`train_model` and `run_audit` management commands. This keeps the page
fast and consistent with the project's design principle (see phase1.txt):
every number shown is a specific, precomputed value, not something
generated on the fly.
"""
from django.shortcuts import render
from .models import ModelMetric, FairnessMetric

# Model version strings, matching the constants used in
# chat/responder.py and the management commands.
LOGREG_VERSION = 'logreg-v0.1'
XGBOOST_VERSION = 'xgboost-v0.1'

# Attributes audited by `run_audit`, in the order they should be displayed.
# The dict maps the stored protected_attribute value to a display label.
PROTECTED_ATTRIBUTES = [
    ('race', 'Race'),
    ('gender', 'Gender'),
    ('age_bracket', 'Age bracket'),
]

# Same fairness-gap threshold used by the chatbot's flag_note logic
# (chat/responder.py) — kept in sync so the report and the chatbot agree
# on what counts as a "materially large" gap.
FAIRNESS_FLAG_THRESHOLD = 0.05


def _model_roc_auc():
    """Return {model_version: roc_auc value} for the two trained models,
    or None for a model whose metric hasn't been cached yet (i.e.
    `python manage.py train_model` hasn't been run since this feature
    was added).
    """
    cached = {
        m.model_version: m.value
        for m in ModelMetric.objects.filter(
            model_version__in=[LOGREG_VERSION, XGBOOST_VERSION],
            metric_name='roc_auc',
        )
    }
    return {
        'logreg': {'version': LOGREG_VERSION, 'label': 'Logistic Regression', 'roc_auc': cached.get(LOGREG_VERSION)},
        'xgboost': {'version': XGBOOST_VERSION, 'label': 'XGBoost', 'roc_auc': cached.get(XGBOOST_VERSION)},
    }


def _fairness_breakdown():
    """Build one row per protected attribute with both Fairlearn metrics
    plus the per-group selection-rate breakdown stored in
    FairnessMetric.detail (only present on the demographic_parity_difference
    row — see run_audit.py, which sets detail=None for equalized_odds).
    """
    metrics = FairnessMetric.objects.filter(model_version=XGBOOST_VERSION)
    by_attr = {}
    for m in metrics:
        by_attr.setdefault(m.protected_attribute, {})[m.metric_name] = m

    rows = []
    for attr_key, attr_label in PROTECTED_ATTRIBUTES:
        entry = by_attr.get(attr_key, {})
        dpd = entry.get('demographic_parity_difference')
        eod = entry.get('equalized_odds_difference')

        detail = None
        if dpd and dpd.detail:
            # Sort by selection rate descending so the highest-flagged-rate
            # group is easy to spot at a glance.
            detail = sorted(dpd.detail.items(), key=lambda kv: kv[1], reverse=True)

        flagged = any(
            metric is not None and abs(metric.value) > FAIRNESS_FLAG_THRESHOLD
            for metric in (dpd, eod)
        )

        rows.append({
            'key': attr_key,
            'label': attr_label,
            'dpd': dpd.value if dpd else None,
            'eod': eod.value if eod else None,
            'detail': detail,
            'flagged': flagged,
        })
    return rows


def report(request):
    """Renders the standalone /report/ page: ROC-AUC comparison, the full
    Fairlearn results for race/gender/age bracket, and the per-group
    breakdown. Entirely separate from the chat views — nothing here is
    reachable from, or reused by, the chatbot.
    """
    context = {
        'models': _model_roc_auc(),
        'fairness_rows': _fairness_breakdown(),
        'threshold': FAIRNESS_FLAG_THRESHOLD,
        'xgboost_version': XGBOOST_VERSION,
    }
    return render(request, 'audit/report.html', context)
