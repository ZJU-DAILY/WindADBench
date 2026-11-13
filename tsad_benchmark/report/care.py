import numpy as np
import pandas as pd
from tsad_benchmark.common.operating_points import TRACK1_ANOMALY_RATIOS
from tsad_benchmark.evaluation.care_protocol import PROTOCOL, LABELS, CURVES
from tsad_benchmark.evaluation.score_timing import normalize_model_name

RATIOS = list(TRACK1_ANOMALY_RATIOS)
PPH = 6
TIME_METRICS = ['mean_lead_time', 'mean_detection_delay', 'actionable_lead_time', 'mtbfa']
COSTS = ['fit_time','fit_gpu_peak_memory_mb','fit_cpu_usage_percent','inference_time',
         'inference_gpu_peak_memory_mb','inference_cpu_usage_percent','n_params','model_size_mb']
MEAN_METRICS = [c for c in (*LABELS, *CURVES) if c not in {
    'accuracy','false_alarms_per_turbine_day','mtbfa','correct_points','total_points',
    'false_alarm_events','turbine_days','eligible_event_count','early_detection_count',
    'lead_support_count','actionable_lead_support_count'}]
GROUPS = ['model','family','paper_family','timing_status','pending_global_dependency','legacy_identity']


def is_current(frame):
    if 'evaluation_protocol' not in frame:
        return False
    if not frame.evaluation_protocol.eq(PROTOCOL).all():
        raise ValueError('Cannot merge different CARE evaluation protocols.')
    return True


def prepare_rows(frame):
    if not is_current(frame):
        raise ValueError('Current CARE evaluation rows are required.')
    from experiments.cross_domain.model_catalog import MODELS
    families = {key: family for family, key, *_ in MODELS}
    paper = dict(non_learning='NL', machine_learning='ML', deep_learning='DL',
                 llm_based='LLM', ts_pretrained='TSP', finetune_llm='DLLM')
    frame = frame.copy()
    if 'model' not in frame:
        frame['model'] = frame.model_name.map(normalize_model_name)
    if 'family' not in frame:
        frame['family'] = frame.model.map(families)
    if 'paper_family' not in frame:
        frame['paper_family'] = frame.family.map(paper)
    frame['ratio_percent'] = pd.to_numeric(frame.typical_anomaly_ratio)
    keys = ['model','file_name','ratio_percent']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate current-run sequence/operating point.')
    for _, group in frame.groupby(['model','file_name']):
        if set(group.ratio_percent) != set(RATIOS):
            raise ValueError('CARE requires all eight operating points per sequence.')
    if frame[GROUPS[:3]].isna().any().any():
        raise ValueError('Unknown model family.')
    return frame


def pooled_mtbfa(group):
    count=group.false_interval_count.sum()
    return float(group.false_interval_sum.sum()/count) if count else np.nan

def _base_summarize_farm(group, across_grid):
    anomalies = group[group.event_label == 'anomaly']
    normal = group[group.event_label == 'normal']
    events = anomalies.groupby('file_name').mean(numeric_only=True)
    result = {c: events[c].mean() for c in MEAN_METRICS}
    total = normal.total_points.sum()
    result['acc'] = normal.correct_points.sum()/total if total else np.nan
    result['normal_point_fpr'] = normal.normal_alert_points.sum()/normal.normal_points.sum() if len(normal) else np.nan
    result['false_alarms_per_turbine_day'] = normal.false_alarm_events.sum()/normal.turbine_days.sum() if len(normal) else np.nan
    intervals = np.asarray([pooled_mtbfa(g) for _,g in normal.groupby('ratio_percent')])
    result['mtbfa'] = float(np.nanmean(intervals)) if np.isfinite(intervals).any() else np.nan
    for c in TIME_METRICS:
        result[c] /= PPH
    result.update(n_anomaly_events=len(events), n_normal_sequences=normal.file_name.nunique(),
                  score_curve_event_support=int(events.auc_pr.notna().sum()),
                  score_curve_farm_support=int(events.auc_pr.notna().any()),
                  n_history_eligible=int(events.eligible.sum()),
                  n_publication_opportunity=int(events.publication_opportunity.sum()),
                  early_detection_count_grid_mean=float(events.early_detection_count.sum()),
                  lead_event_support=int(events.mean_lead_time.notna().sum()),
                  actionable_lead_event_support=int(events.actionable_lead_time.notna().sum()),
                  lead_event_ratio_support=int(anomalies.lead_support_count.sum()),
                  actionable_lead_event_ratio_support=int(anomalies.actionable_lead_support_count.sum()),
                  normal_clock_points_grid_mean=total/(len(RATIOS) if across_grid else 1),
                  normal_available_clock_fraction=normal.available_clock_points.sum()/total if total else np.nan,
                  anomaly_available_clock_fraction=anomalies.available_clock_points.sum()/anomalies.total_points.sum() if len(anomalies) else np.nan,
                  available_warning_updates_mean=events.available_warning_updates.mean())
    return result

def summarize_farm(group,across_grid):
    result=_base_summarize_farm(group,across_grid)
    normal=group[group.event_label=='normal'];anom=group[group.event_label=='anomaly']
    events=anom.groupby('file_name').mean(numeric_only=True)
    intervals=[pooled_mtbfa(g) for _,g in normal.groupby('ratio_percent')]
    result['mtbfa']=float(np.nanmean(intervals))/6 if np.isfinite(intervals).any() else np.nan
    for metric in ['observable_early_detection_rate','affiliation_precision_unmasked_reference',
                   'affiliation_recall_unmasked_reference','affiliation_f1_unmasked_reference']:
        result[metric]=float(events[metric].mean())
    result.update(n_observable_eligible=int(events.observable_eligible_event_count.sum()),
        normal_available_clock_fraction=normal.valid_available_clock_points.sum()/normal.total_points.sum(),
        anomaly_available_clock_fraction=anom.valid_available_clock_points.sum()/anom.total_points.sum(),
        normal_raw_points_grid_mean=normal.raw_test_points.sum()/(8 if across_grid else 1),
        normal_valid_points_grid_mean=normal.total_points.sum()/(8 if across_grid else 1),
        status_evaluation_version='care_bc_valid_support_v1')
    return result

def aggregate(rows,ratio=False):
    keys=GROUPS+['farm_id']+(['ratio_percent'] if ratio else [])
    result=[]
    for key,g in rows.groupby(keys,dropna=False,sort=True):
        result.append(dict(zip(keys,key),**summarize_farm(g,not ratio)))
    return pd.DataFrame(result)

def macro(farms, ratio=False):
    keys = GROUPS+(['ratio_percent'] if ratio else [])
    numeric = [c for c in farms.select_dtypes(include='number').columns if c not in keys]
    result = farms.groupby(keys, dropna=False)[numeric].mean().reset_index()

    for column in [c for c in numeric if c.startswith('n_') or c.endswith('_support') or c.endswith('_grid_mean')]:
        result[column] = farms.groupby(keys, dropna=False)[column].sum().to_numpy()
    result['farm_count'] = farms.groupby(keys, dropna=False).size().to_numpy()
    result['lead_farm_support'] = farms.groupby(keys, dropna=False).mean_lead_time.count().to_numpy()
    return result


def main_tables(frame):
    rows = prepare_rows(frame)
    farms, ratios = aggregate(rows), aggregate(rows, True)
    tables = dict(latest_main_by_farm=farms, by_farm_ratio=ratios,
                  latest_main_macro=macro(farms), macro_ratio=macro(ratios, True))
    costs = rows.drop_duplicates(['model', 'file_name']).groupby(['model', 'farm_id'])[COSTS].mean().reset_index()
    tables['latest_cost_by_farm'] = costs
    tables['latest_cost_macro'] = costs.groupby('model')[COSTS].mean().reset_index()
    for name in ('latest_main_by_farm', 'by_farm_ratio', 'latest_main_macro', 'macro_ratio'):
        tables[name]['time_units'] = 'hours'
    names = rows[['model', 'model_name']].drop_duplicates()
    for source, target in [('latest_main_macro', 'overall_leaderboard'),
                            ('latest_main_by_farm', 'by_farm_leaderboard')]:
        board = tables[source].merge(names, on='model', validate='many_to_one')
        board['accuracy'] = board['acc']
        for column in TIME_METRICS:
            board[column] *= PPH
        board['time_units'] = 'sampling_points'
        tables[target] = board
    tables['operational_summary'] = operational_board(tables['overall_leaderboard'])
    tables['by_farm_operational_summary'] = operational_board(tables['by_farm_leaderboard'])
    return tables


def operational_board(board):
    board = board.copy()
    board['accuracy_normal'] = board['acc']
    board['false_alarms_per_turbine_day_normal'] = board['false_alarms_per_turbine_day']
    board['mtbfa_normal_days'] = board['mtbfa']/144.
    return board
