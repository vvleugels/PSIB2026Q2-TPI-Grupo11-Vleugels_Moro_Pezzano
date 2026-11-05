"""Fixed classical Otsu grid and patient-only selection; no trained models."""
from copy import deepcopy
from itertools import product
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from .lesion_eval import _optimal_pairs
from .detection_validation import summarize

PROFILES = [
    dict(opening=False, closing=False, fill_holes=False, radius=1, min_area=0),
    dict(opening=False, closing=False, fill_holes=False, radius=1, min_area=10),
    dict(opening=False, closing=True, fill_holes=True, radius=1, min_area=10),
    dict(opening=True, closing=True, fill_holes=True, radius=1, min_area=0),
    dict(opening=True, closing=True, fill_holes=True, radius=1, min_area=10),
    dict(opening=True, closing=True, fill_holes=True, radius=2, min_area=10),
]


def configurations(base):
    result = []
    for center_width, sigma, equalization in product(
            ((30, 150), (45, 100), (100, 200)), (0, .3, .6, 1.),
            ('none', 'histogram_equalization', 'clahe')):
        for profile, cleanup in enumerate(PROFILES):
            cfg = deepcopy(base)
            cfg['detection'] = {'enabled': False}
            cfg['edge_analysis'] = {'operator': 'none'}
            cfg['preprocessing'] = dict(hu_window=dict(zip(('center', 'width'), center_width)),
                denoising=dict(method='gaussian', gaussian_sigma=sigma) if sigma else dict(method='none'),
                enhancement=dict(method=equalization, clahe_clip_limit=.01, clahe_kernel_size=8))
            cfg['segmentation'] = dict(method='otsu', footprint='disk', **cleanup)
            result.append(dict(configuration_id=f'otsu_{len(result):03d}', profile=f'P{profile}', config=cfg))
    return result


def score_mask(pred, prepared_reference, criteria=(.1, .25)):
    """Count matches without per-component DataFrames; same cardinality/IoU rule."""
    rlabels, rsizes, diameters = prepared_reference
    pred = np.asarray(pred, dtype=bool)
    plabels, npred = ndi.label(pred, np.ones((3, 3), bool))
    nref = len(rsizes)-1
    psizes = np.bincount(plabels.ravel(), minlength=npred+1)
    both = (rlabels > 0) & pred
    intersections = np.bincount(rlabels[both]*(npred+1)+plabels[both],
        minlength=(nref+1)*(npred+1)).reshape(nref+1,npred+1)[1:,1:]
    unions = rsizes[1:,None]+psizes[None,1:]-intersections
    ious = np.divide(intersections, unions, out=np.zeros_like(intersections,dtype=float), where=unions>0)
    small = {r-1 for r,d in diameters.items() if d < 10}
    ref_area, pred_area, intersection = int(rsizes[1:].sum()), int(pred.sum()), int(both.sum())
    counts = dict(reference_area=ref_area,pred_area=pred_area,intersection=intersection,
                  union=ref_area+pred_area-intersection)
    rows=[]
    for criterion in criteria:
        pairs = _optimal_pairs(ious, (intersections>0)&(ious>=criterion))
        tp=len(pairs); small_tp=sum(r in small for r,p in pairs)
        rows.append(dict(**counts,min_iou=criterion,TP=tp,FP=npred-tp,FN=nref-tp,
            small_TP=small_tp,small_FN=len(small)-small_tp,
            reference_coverage=intersection/ref_area if ref_area else np.nan,
            dice=2*intersection/(ref_area+pred_area) if ref_area+pred_area else 1.,
            iou=intersection/counts['union'] if counts['union'] else 1.))
    return rows


def summarize_search(frame):
    result=summarize(frame)
    tp,fp,fn=(result[k] for k in ('TP','FP','FN'))
    result['F2']=5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.
    counts=frame.groupby('case_id')[['intersection','reference_area']].sum()
    coverage=counts.intersection/counts.reference_area.replace(0,np.nan)
    result['reference_coverage_mean']=float(coverage.mean())
    result['reference_coverage_n_cases']=int(coverage.notna().sum())
    result['pixel_FP']=int((frame.pred_area-frame.intersection).sum())
    result['pixel_precision']=float(frame.intersection.sum()/frame.pred_area.sum()) if frame.pred_area.sum() else np.nan
    return result


def ranking(score, ids, objective='F2'):
    finite=lambda v: v if np.isfinite(v) else -1.
    return (-finite(score[objective]),-finite(score['small_recall']),
            score['fp_per_slice'],-finite(score['dice_mean']),tuple(ids))


def choose(ids, case_ids, evaluate, objective='F2'):
    """The evaluator receives only the permitted patient IDs."""
    scored=[]
    for cid in ids:
        score=summarize_search(evaluate((cid,),case_ids))
        scored.append((ranking(score,(cid,),objective),cid,score))
    _,cid,score=min(scored)
    return (cid,),score


def greedy_union(ids, case_ids, evaluate, start=None):
    members,score=choose(ids,case_ids,evaluate) if start is None else start
    trace=[dict(members=list(members),**score)]
    while len(members)<3:
        trials=[]
        for cid in ids:
            if cid in members: continue
            candidate=tuple(sorted((*members,cid)))
            s=summarize_search(evaluate(candidate,case_ids))
            trials.append((ranking(s,candidate),candidate,s))
        if not trials: break
        _,candidate,s=min(trials)
        if s['F2'] <= score['F2']: break
        members,score=candidate,s
        trace.append(dict(members=list(members),**score))
    return members,score,trace


def validate_folds(folds, cases):
    tests=[]
    for f in folds:
        train,test=set(f['train']),set(f['test'])
        inner=[c for group in f['inner_validation'] for c in group]
        if train & test or train|test != set(cases) or len(test)!=2 or len(train)!=8:
            raise ValueError('Invalid outer partition')
        if len(f['inner_validation'])!=4 or any(len(g)!=2 for g in f['inner_validation']) or sorted(inner)!=sorted(train):
            raise ValueError('Invalid inner partition')
        tests.extend(test)
    if len(folds)!=5 or sorted(tests)!=sorted(cases):
        raise ValueError('Outer test patients must occur exactly once')
