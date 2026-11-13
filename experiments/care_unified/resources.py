from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/care_unified_20261107'
IDENTITY=['model','source_farm','target_farm','target_file_name']
FIT_FIELDS=['model_name','model_path','model_params','seed','protocol','source_run_path',
            'checkpoint_path','checkpoint_sha256','source_asset_count','source_event_count',
            'source_train_rows','fit_seconds_source']
SUM_FIELDS={
    'inference_seconds_event':'inference_seconds',
    'inference_points_event':'inference_points',
    'checkpoint_load_seconds_event':'checkpoint_load_seconds',
    'calibration_seconds_event':'calibration_seconds',
    'calibration_checkpoint_load_seconds_event':'calibration_checkpoint_load_seconds',
    'evaluation_wall_seconds_event':'evaluation_wall_seconds',
}
PEAK_FIELDS=['inference_peak_memory_mb','inference_peak_gpu_memory_mb',
             'calibration_peak_memory_mb','calibration_peak_gpu_memory_mb']
REFERENCE={
    'full_fleet':ROOT/'outputs/care_v6_reevaluation_20261104/cross_full_fleet/all_per_event.csv',
    'cross_turbine':ROOT/'outputs/care_bc_affiliation_20261107/cross_turbine/all_per_event.csv',
}
SCOPES={'full_fleet':'full_fleet','cross_turbine':'cross_turbine',
        'paired/same_farm':'full_fleet','paired/cross_turbine':'cross_turbine'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _constant(g,column):
    values=g[column]
    if values.isna().all():return np.nan
    if values.isna().any() or values.nunique(dropna=False)!=1:
        raise ValueError('source fit has inconsistent '+column)
    return values.iloc[0]


def _source_path(value):
    p=Path(str(value))
    return str((p if p.is_absolute() else ROOT/p).resolve())


def summarize(frame,scope):
    frame=frame[frame.track.eq('label')].copy()
    if frame.empty:raise ValueError('empty label resource scope')
    if frame.duplicated(IDENTITY).any():raise ValueError('duplicate label event resource row')
    required=[*IDENTITY,'fit_seconds_source','source_run_path','checkpoint_path','model_params',
              'inference_seconds_event','inference_points_event']
    if set(required)-set(frame):raise ValueError('missing resource fields '+str(set(required)-set(frame)))
    for field in list(SUM_FIELDS)+PEAK_FIELDS+['fit_seconds_source']:
        if field not in frame:continue
        frame[field]=pd.to_numeric(frame[field],errors='raise')
        if np.isinf(frame[field].dropna()).any() or (frame[field].dropna()<0).any():
            raise ValueError('invalid resource '+field)
    fits=[];directions=[]
    for (model,source),g in frame.groupby(['model','source_farm'],sort=True):
        fit=dict(scope=scope,model=model,source_farm=source)
        for field in FIT_FIELDS:
            if field in g:fit[field]=_constant(g,field)
        fit['source_run_path']=_source_path(fit['source_run_path'])
        physical=json.dumps([model,source,fit['source_run_path'],str(fit['checkpoint_path']),
                             str(fit.get('checkpoint_sha256',''))],ensure_ascii=False)
        fit_id=hashlib.sha256(physical.encode()).hexdigest()[:24]
        fit.update(fit_id=fit_id,accounting='shared_source_fit_once',target_direction_count=g.target_farm.nunique(),
                   evaluated_sequence_count=len(g),scope_fit_cost_is_reference=True)
        fits.append(fit)
        for target,h in g.groupby('target_farm',sort=True):
            row=dict(scope=scope,model=model,source_farm=source,target_farm=target,fit_id=fit_id,
                     model_params=fit['model_params'],sequence_count=len(h),
                     target_asset_count=int(h.target_asset_id.nunique()) if 'target_asset_id' in h else np.nan,
                     transfer_type='same_farm' if source==target else 'cross_farm',
                     accounting='event_inference_summed_once_source_fit_referenced',
                     point_basis='raw_inference_points_not_status_filtered')
            for field,name in SUM_FIELDS.items():
                if field not in h:continue
                defined=int(h[field].notna().sum())

                row[name]=float(h[field].sum()) if defined==len(h) else np.nan
                row[name+'_defined_sequences']=defined
            for field in PEAK_FIELDS:
                if field not in h:continue
                defined=int(h[field].notna().sum())
                row[field]=float(h[field].max()) if defined==len(h) else np.nan
                row[field+'_defined_sequences']=defined
            seconds=row['inference_seconds'];points=row['inference_points']
            row['throughput_points_per_second']=points/seconds if np.isfinite(seconds) and seconds>0 and np.isfinite(points) else np.nan
            directions.append(row)
    return pd.DataFrame(fits),pd.DataFrame(directions)


def _read(path):
    columns=set(IDENTITY+FIT_FIELDS+list(SUM_FIELDS)+PEAK_FIELDS+['track','target_asset_id'])
    return pd.read_csv(path,usecols=lambda c:c in columns,float_precision='round_trip',low_memory=False)


def _verify_reference(frame,reference):
    frame=frame[frame.track.eq('label')].set_index(IDENTITY).sort_index()
    reference=reference[reference.track.eq('label')].set_index(IDENTITY)
    if reference.index.has_duplicates or frame.index.has_duplicates:raise ValueError('duplicate original resource identity')
    reference=reference.loc[frame.index]
    cells=0
    for c in frame.columns:
        a=frame[c];b=reference[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            np.testing.assert_allclose(a.to_numpy(float),b.to_numpy(float),rtol=0,atol=0,equal_nan=True,err_msg=c)
        elif not a.fillna('<NA>').astype(str).equals(b.fillna('<NA>').astype(str)):
            raise ValueError('resource metadata changed '+c)
        cells+=len(a)
    return cells


def run(output=OUTPUT):
    output=Path(output).resolve()
    if output!=OUTPUT and OUTPUT not in output.parents:raise ValueError('outside new result namespace')
    paths={scope:output/scope/'all_per_event.csv' for scope in SCOPES}
    inputs={str(p):sha(p) for p in [*paths.values(),*REFERENCE.values()]}
    refs={scope:_read(path) for scope,path in REFERENCE.items()}
    fits=[];directions=[];checks={};model_sets=[]
    for scope,path in paths.items():
        frame=_read(path);checks[scope]=_verify_reference(frame,refs[SCOPES[scope]])
        model_sets.append(set(frame.model));fit,direction=summarize(frame,scope)
        fits.append(fit);directions.append(direction)
    if any(s!=model_sets[0] for s in model_sets):raise ValueError('resource scope model coverage differs')
    f=pd.concat(fits,ignore_index=True);d=pd.concat(directions,ignore_index=True)

    for fit_id,g in f.groupby('fit_id'):
        for c in ['model','source_farm','source_run_path','checkpoint_path','model_params','fit_seconds_source']:
            _constant(g,c)
    target=output/'resources';target.mkdir(exist_ok=True)
    f.to_csv(target/'source_fit.csv',index=False)
    d.to_csv(target/'direction_resources.csv',index=False)
    if any(sha(path)!=digest for path,digest in inputs.items()):raise ValueError('input changed during resource export')
    info=dict(status='complete',models=len(model_sets[0]),scopes=list(SCOPES),
              source_fit_scope_rows=len(f),unique_physical_source_fits=f.fit_id.nunique(),direction_rows=len(d),
              original_resource_cells_verified=checks,input_sha256=inputs,code_sha256=sha(Path(__file__)),
              outputs={p.name:sha(p) for p in [target/'source_fit.csv',target/'direction_resources.csv']},
              accounting='scope rows are overlapping views; count unique fit_id once across selected directions/scopes',
              missing_resource_policy='unmeasured values remain NaN; incomplete direction totals/peaks remain NaN with support',
              status_filter_policy='compute measured on raw inference workload; never replace inference_points_event by valid scoring points',
              model_params_policy='preserved per original source fit, including USAD configuration differences')
    (target/'manifest.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:info[k] for k in ['status','models','source_fit_scope_rows','unique_physical_source_fits','direction_rows']},ensure_ascii=False))
    return info


def _self_test():
    assert 'summarize' in globals(), 'resource summary implementation is required'
    base=dict(model='x',source_farm='A',target_farm='B',track='label',source_run_path='fit-A',
              checkpoint_path='checkpoint.pt',checkpoint_sha256='abc',model_params='{"window": 12}',
              fit_seconds_source=10.,source_asset_count=2,source_event_count=3,source_train_rows=100,
              inference_peak_memory_mb=np.nan,inference_peak_gpu_memory_mb=4.)
    a=dict(base,target_file_name='B_1.csv',target_asset_id=1,inference_seconds_event=2.,inference_points_event=50)
    b=dict(base,target_file_name='B_2.csv',target_asset_id=1,inference_seconds_event=3.,inference_points_event=70)
    frame=pd.DataFrame([a,b,dict(a,track='score')])
    fits,directions=summarize(frame,'test')
    assert len(fits)==1 and fits.iloc[0].fit_seconds_source==10
    assert len(directions)==1 and directions.iloc[0].inference_seconds==5
    assert directions.iloc[0].inference_points==120 and directions.iloc[0].sequence_count==2
    assert pd.isna(directions.iloc[0].inference_peak_memory_mb)
    assert 'fit_seconds_source' not in directions
    assert directions.iloc[0].fit_id==fits.iloc[0].fit_id
    c=dict(a,target_farm='C',target_file_name='C_1.csv')
    fs,ds=summarize(pd.DataFrame([a,b,c]),'test')
    assert len(fs)==1 and len(ds)==2 and fs.iloc[0].fit_seconds_source==10
    pf,pd_=summarize(pd.DataFrame([a]),'paired/test')
    assert pf.iloc[0].fit_id==fs.iloc[0].fit_id
    bad=pd.DataFrame([a,dict(b,fit_seconds_source=11)])
    try:summarize(bad,'test')
    except ValueError:pass
    else:raise AssertionError('inconsistent shared fit cost must fail')
    duplicate=pd.DataFrame([a,a])
    try:summarize(duplicate,'test')
    except ValueError:pass
    else:raise AssertionError('duplicate label event must fail')
    print('Resource ledger counterexample checks passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:_self_test()
    else:run()
