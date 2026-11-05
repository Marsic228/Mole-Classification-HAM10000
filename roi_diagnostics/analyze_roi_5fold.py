#!/usr/bin/env python3
"""Reproduce the five-fold per-image reference-mask vs predicted-mask ROI audit.

Usage:
    python analyze_roi_5fold.py roi_5fold_diagnostics.zip --out audit

Requires pandas and scikit-learn. Does not load images, masks, or models.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

CLASSES = ('akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc')


def read_fold_tables(zip_path: Path) -> pd.DataFrame:
    frames = []
    with ZipFile(zip_path) as archive:
        for fold in range(1, 6):
            name = f'classifier_predicted_vs_reference_fold{fold}.csv'
            matches = [p for p in archive.namelist() if Path(p).name == name]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {name}; found {len(matches)}')
            with archive.open(matches[0]) as stream:
                part = pd.read_csv(stream, dtype={'image_id': str, 'dx': str})
            required = {'image_id', 'dx', 'true_label', 'reference_prediction',
                        'predicted_mask_prediction'}
            if not required.issubset(part.columns):
                raise ValueError(f'Invalid columns for fold {fold}: {part.columns}')
            if part.image_id.duplicated().any():
                raise ValueError(f'Duplicate image_id in fold {fold}')
            if part[['true_label', 'reference_prediction',
                     'predicted_mask_prediction']].isna().any().any():
                raise ValueError(f'Missing labels or predictions in fold {fold}')
            if not all(part[c].isin(range(len(CLASSES))).all() for c in
                       ['true_label', 'reference_prediction', 'predicted_mask_prediction']):
                raise ValueError(f'Out-of-range predictions or labels in fold {fold}')
            if (part.dx != part.true_label.map(dict(enumerate(CLASSES)))).any():
                raise ValueError(f'Class/label mismatch in fold {fold}')
            part['fold'] = fold
            frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    if len(df) != 8530 or df.image_id.nunique() != len(df):
        raise ValueError(f'Expected 8530 distinct image IDs; got {len(df)} / {df.image_id.nunique()}')
    df['reference_class'] = df.reference_prediction.map(dict(enumerate(CLASSES)))
    df['predicted_class'] = df.predicted_mask_prediction.map(dict(enumerate(CLASSES)))
    df['ref_correct'] = df.true_label.eq(df.reference_prediction)
    df['pred_correct'] = df.true_label.eq(df.predicted_mask_prediction)
    df['changed'] = df.reference_prediction.ne(df.predicted_mask_prediction)
    df['outcome'] = np.select([
        df.changed & ~df.ref_correct & df.pred_correct,
        df.changed & df.ref_correct & ~df.pred_correct,
        df.changed & ~df.ref_correct & ~df.pred_correct,
    ], ['corrected', 'regression', 'wrong_to_wrong'], default='unchanged')
    return df


def run(zip_path: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    df = read_fold_tables(zip_path)
    changed = df[df.changed].copy()
    changed[['fold', 'image_id', 'dx', 'true_label', 'reference_prediction',
             'predicted_mask_prediction', 'reference_class', 'predicted_class',
             'outcome']].sort_values(['fold','outcome','dx','image_id']).to_csv(
                out/'roi_5fold_changed_predictions.csv', index=False)
    summary=[]
    for fold, d in df.groupby('fold'):
        n=d.outcome.value_counts()
        summary.append(dict(fold=fold,images=len(d),changed=int(d.changed.sum()),
                            corrected=int(n.get('corrected',0)),regressions=int(n.get('regression',0)),
                            wrong_to_wrong=int(n.get('wrong_to_wrong',0)),
                            reference_accuracy=accuracy_score(d.true_label,d.reference_prediction),
                            predicted_accuracy=accuracy_score(d.true_label,d.predicted_mask_prediction),
                            reference_macro_f1=f1_score(d.true_label,d.reference_prediction,
                                                      labels=range(7),average='macro'),
                            predicted_macro_f1=f1_score(d.true_label,d.predicted_mask_prediction,
                                                      labels=range(7),average='macro')))
    summ=pd.DataFrame(summary)
    summ['delta_macro_f1']=summ.predicted_macro_f1-summ.reference_macro_f1
    summ['delta_accuracy']=summ.predicted_accuracy-summ.reference_accuracy
    summ.to_csv(out/'roi_fold_summary.csv',index=False)
    classes=[]
    for fold, d in [('pooled',df),*[(str(fold), x) for fold,x in df.groupby('fold')]]:
        rp=precision_score(d.true_label,d.reference_prediction,labels=range(7),average=None,zero_division=0)
        pp=precision_score(d.true_label,d.predicted_mask_prediction,labels=range(7),average=None,zero_division=0)
        rr=recall_score(d.true_label,d.reference_prediction,labels=range(7),average=None,zero_division=0)
        pr=recall_score(d.true_label,d.predicted_mask_prediction,labels=range(7),average=None,zero_division=0)
        rf=f1_score(d.true_label,d.reference_prediction,labels=range(7),average=None,zero_division=0)
        pf=f1_score(d.true_label,d.predicted_mask_prediction,labels=range(7),average=None,zero_division=0)
        for i,c in enumerate(CLASSES):
            classes.append(dict(fold=fold,true_class=c,n=int((d.true_label==i).sum()),
                                reference_precision=rp[i],predicted_precision=pp[i],
                                reference_recall=rr[i],predicted_recall=pr[i],
                                reference_f1=rf[i],predicted_f1=pf[i],delta_f1=pf[i]-rf[i],
                                corrected=int(((d.dx==c)&(d.outcome=='corrected')).sum()),
                                regressions=int(((d.dx==c)&(d.outcome=='regression')).sum())))
    classdf=pd.DataFrame(classes)
    classdf.to_csv(out/'roi_per_class_metrics.csv',index=False)
    trans=changed.groupby(['fold','dx','reference_class','predicted_class','outcome'],
                          as_index=False).size().rename(columns={'size':'n'})
    trans.sort_values(['fold','n'],ascending=[True,False]).to_csv(
        out/'roi_changed_transitions.csv', index=False)
    fold5=df[df.fold==5]
    priority=[]
    # Include all critical-class regressions and corrections, plus selected nv->mel regressions.
    for true_class in ['df','mel']:
        priority.extend(fold5[(fold5.dx==true_class)&(fold5.outcome=='regression')].to_dict('records'))
        priority.extend(fold5[(fold5.dx==true_class)&(fold5.outcome=='corrected')].to_dict('records'))
    priority.extend(fold5[(fold5.dx=='nv')&(fold5.reference_class=='nv')&
                          (fold5.predicted_class=='mel')].head(4).to_dict('records'))
    priority=pd.DataFrame(priority)
    priority[['fold','image_id','dx','reference_class','predicted_class','outcome']].to_csv(
        out/'roi_fold5_priority_cases.csv',index=False)
    print('DATA:',len(df),'unique IDs; changed:',len(changed))
    print('FOLDS:\n',summ[['fold','reference_macro_f1','predicted_macro_f1','delta_macro_f1','changed','corrected','regressions']].to_string(index=False))
    print('MEAN DELTA:',summ.delta_macro_f1.mean(),'SAMPLE SD:',summ.delta_macro_f1.std(ddof=1))
    print('PRIORITY CASES:',len(priority))
    return df, summ, classdf, priority


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('zipfile',type=Path)
    ap.add_argument('--out',type=Path,default=Path('roi_diagnostic_audit'))
    args=ap.parse_args()
    run(args.zipfile,args.out)

if __name__=='__main__':main()
