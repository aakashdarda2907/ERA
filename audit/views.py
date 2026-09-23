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

from django.db.models import Avg, Count
from django.db.models.functions import Abs
from .models import ModelMetric, FairnessMetric, ShapExplanation

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

def _global_feature_importance(limit=12):
    """Mean absolute SHAP value per feature across all cached XGBoost SHAP
    explanations, ranked descending — a global, model-wide view built by
    aggregating the same per-patient evidence run_audit already stored.
    No new SHAP computation happens here.

    Caveat: run_audit only stores each patient's top 6 contributing
    features (TOP_K_FEATURES), not all ~100 one-hot columns per patient.
    So each feature's average is only over the patients where it was
    significant enough to be a top-6 contributor — this is shown as `n`
    alongside each bar so the report doesn't overstate precision.
    """
    qs = (
        ShapExplanation.objects
        .filter(model_version=XGBOOST_VERSION)
        .values('feature_name')
        .annotate(avg_abs_shap=Avg(Abs('shap_value')), n=Count('id'))
        .order_by('-avg_abs_shap')[:limit]
    )
    rows = list(qs)
    if not rows:
        return []
    max_val = rows[0]['avg_abs_shap']
    return [
        {
            'feature': r['feature_name'],
            'avg_abs_shap': r['avg_abs_shap'],
            'n': r['n'],
            'pct': round((r['avg_abs_shap'] / max_val) * 100, 1) if max_val else 0,
        }
        for r in rows
    ]

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
        'global_importance': _global_feature_importance(),
    }
    return render(request, 'audit/report.html', context)



def risk_assessment(request):
    """Renders /report/risk-assessment/ — a narrative AI Risk & Impact
    Assessment: what the model is for, who it affects, known risks, and
    mitigations. The numbers cited (ROC-AUC, fairness gaps) come from the
    same cached ModelMetric/FairnessMetric evidence as the /report/ page —
    only the surrounding narrative (purpose, stakeholders, mitigations)
    is authored documentation rather than computed evidence.
    """
    fairness_rows = _fairness_breakdown()
    context = {
        'models': _model_roc_auc(),
        'fairness_rows': fairness_rows,
        'flagged_rows': [r for r in fairness_rows if r['flagged']],
        'threshold': FAIRNESS_FLAG_THRESHOLD,
        'xgboost_version': XGBOOST_VERSION,
    }
    return render(request, 'audit/risk_assessment.html', context)

from audit.models import ExplanationDisparity


def _color_for_distance(distance, max_distance):
    """Maps a distance value to a red-intensity background color for the
    heatmap table - higher divergence = more saturated red.
    """
    if max_distance <= 0:
        return '#F2F4F3'
    ratio = min(distance / max_distance, 1.0)
    r = 255
    g = int(244 - ratio * 140)
    b = int(243 - ratio * 180)
    return f'rgb({r},{g},{b})'


def explanation_disparity(request):
    """Renders the Explanation Disparity heatmap and the top-finding
    comparison chart, entirely from ExplanationDisparity rows already
    computed and stored by analyze_explanation_disparity.py.
    """
    MODEL_VERSION = 'xgboost-v0.1'
    rows = list(ExplanationDisparity.objects.filter(model_version=MODEL_VERSION))

    if not rows:
        return render(request, 'audit/disparity.html', {'has_data': False})

    max_distance = max(r.wasserstein_distance for r in rows)

    # --- Heatmap: one table per protected attribute ---
    heatmaps = {}
    for attr in ['race', 'gender', 'age_bracket']:
        attr_rows = [r for r in rows if r.protected_attribute == attr]
        max_per_feature = {}
        for r in attr_rows:
            max_per_feature[r.feature_name] = max(max_per_feature.get(r.feature_name, 0), r.wasserstein_distance)
        features = sorted(max_per_feature, key=max_per_feature.get, reverse=True)
        subgroups = sorted(set(r.subgroup for r in attr_rows))

        lookup = {(r.feature_name, r.subgroup): r for r in attr_rows}
        table = []
        for feat in features:
            row_cells = []
            for sg in subgroups:
                r = lookup.get((feat, sg))
                if r:
                    row_cells.append({
                        'value': f'{r.wasserstein_distance:.3f}',
                        'color': _color_for_distance(r.wasserstein_distance, max_distance),
                    })
                else:
                    row_cells.append({'value': '\u2014', 'color': '#F2F4F3'})
            table.append({'feature': feat, 'cells': row_cells})

        heatmaps[attr] = {'subgroups': subgroups, 'rows': table}

    # --- Top finding + comparison chart ---
    top = max(rows, key=lambda r: r.wasserstein_distance)
    same_feature_attr = [
        r for r in rows
        if r.feature_name == top.feature_name and r.protected_attribute == top.protected_attribute
    ]
    high_group = max(same_feature_attr, key=lambda r: r.subgroup_mean_shap)
    low_group = min(same_feature_attr, key=lambda r: r.subgroup_mean_shap)

    def top_features_for(subgroup_name, attr):
        candidates = [r for r in rows if r.protected_attribute == attr and r.subgroup == subgroup_name]
        candidates.sort(key=lambda r: abs(r.subgroup_mean_shap), reverse=True)
        return [r.feature_name for r in candidates[:3]]

    union_features = list(dict.fromkeys(
        top_features_for(high_group.subgroup, top.protected_attribute) +
        top_features_for(low_group.subgroup, top.protected_attribute)
    ))

    lookup_top = {(r.feature_name, r.subgroup): r for r in same_feature_attr + rows}
    chart_labels = union_features
    chart_high = []
    chart_low = []
    for feat in union_features:
        rh = next((r for r in rows if r.feature_name == feat and r.protected_attribute == top.protected_attribute and r.subgroup == high_group.subgroup), None)
        rl = next((r for r in rows if r.feature_name == feat and r.protected_attribute == top.protected_attribute and r.subgroup == low_group.subgroup), None)
        chart_high.append(round(rh.subgroup_mean_shap, 4) if rh else 0)
        chart_low.append(round(rl.subgroup_mean_shap, 4) if rl else 0)

    ranked = sorted(rows, key=lambda r: r.wasserstein_distance, reverse=True)[:10]
    leaderboard_labels = [f'{r.feature_name} \u00d7 {r.subgroup} ({r.protected_attribute})' for r in ranked]
    leaderboard_values = [round(r.wasserstein_distance, 4) for r in ranked]
    leaderboard_n = [r.n for r in ranked]

    context = {
        'has_data': True,
        'heatmaps': heatmaps,
        'top_feature': top.feature_name,
        'top_attribute': top.protected_attribute,
        'top_distance': round(top.wasserstein_distance, 4),
        'high_group': high_group.subgroup,
        'low_group': low_group.subgroup,
        'chart_labels': chart_labels,
        'chart_high': chart_high,
        'chart_low': chart_low,
                'leaderboard_labels': leaderboard_labels,
        'leaderboard_values': leaderboard_values,
        'leaderboard_n': leaderboard_n,
    }
    return render(request, 'audit/disparity.html', context)