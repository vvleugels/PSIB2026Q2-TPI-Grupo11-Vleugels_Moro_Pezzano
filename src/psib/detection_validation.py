"""Patient-disjoint internal validation for fixed, non-learned detector variants."""
import numpy as np
import pandas as pd


def make_folds(case_ids, seed=2026):
    cases = np.array(sorted(set(map(int, case_ids))))
    if len(cases) != 10:
        raise ValueError('Nested protocol requires exactly 10 patients')
    rng = np.random.default_rng(seed)
    pairs = np.array_split(rng.permutation(cases), 5)
    folds = []
    for i, test in enumerate(pairs):
        train = np.array(sorted(set(cases) - set(test)))
        inner = [list(map(int, v)) for v in np.array_split(rng.permutation(train), 4)]
        folds.append(dict(fold=i, test=list(map(int, test)), train=list(map(int, train)), inner_validation=inner))
    return folds


def summarize(frame):
    tp, fp, fn = (int(frame[c].sum()) for c in ('TP', 'FP', 'FN'))
    n = len(frame)
    small_tp, small_fn = int(frame.small_TP.sum()), int(frame.small_FN.sum())
    counts = frame.groupby('case_id')[['intersection', 'pred_area', 'reference_area', 'union']].sum()
    total = counts.pred_area + counts.reference_area
    dice = np.divide(2 * counts.intersection, total, out=np.ones(len(total)), where=total > 0)
    iou = np.divide(counts.intersection, counts.union, out=np.ones(len(total)), where=counts.union > 0)
    negative = frame[frame.reference_area == 0]
    return dict(TP=tp, FP=fp, FN=fn, n_slices=n, recall=tp/(tp+fn) if tp+fn else np.nan,
        precision=tp/(tp+fp) if tp+fp else np.nan, F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
        fp_per_slice=fp/n if n else np.nan, small_recall=small_tp/(small_tp+small_fn) if small_tp+small_fn else np.nan,
        small_TP=small_tp, small_FN=small_fn, dice_mean=float(np.mean(dice)), iou_mean=float(np.mean(iou)),
        negative_slices=len(negative), negative_slices_with_fp=int((negative.FP > 0).sum()),
        fp_per_negative_slice=float(negative.FP.mean()) if len(negative) else np.nan)


def select_variant(validation, budget):
    scores = pd.DataFrame([dict(configuration_id=cid, **summarize(g))
                           for cid, g in validation.groupby('configuration_id')])
    feasible = scores[scores.fp_per_slice <= budget].copy()
    if feasible.empty:
        return None  # Abstain explicitly; never use outer outcomes to choose.
    return str(feasible.sort_values(['recall', 'fp_per_slice', 'dice_mean', 'configuration_id'],
                    ascending=[False, True, False, True], na_position='last').iloc[0].configuration_id)


def nested_evaluation(metrics, folds, budgets=(1, 2, 4, 8)):
    """Inner held-out patients select fixed parameters; outer patients only score.

    There is no fitted model. Concatenating inner validation folds is equivalent
    to evaluating each fixed configuration once on each training patient.
    """
    chosen, outer_rows, inner_rows = [], [], []
    primary = metrics[metrics.min_iou == 0.10]
    for fold in folds:
        train, test = set(fold['train']), set(fold['test'])
        if train & test or len(train) != 8 or len(test) != 2:
            raise ValueError('Invalid outer folds')
        inner_ids = [v for group in fold['inner_validation'] for v in group]
        if len(inner_ids) != len(set(inner_ids)) or set(inner_ids) != train:
            raise ValueError('Inner validation must partition outer training patients')
        for j, ids in enumerate(fold['inner_validation']):
            for cid, group in primary[primary.case_id.isin(ids) & primary.configuration_id.str.startswith('det_')].groupby('configuration_id'):
                inner_rows.append(dict(outer_fold=fold['fold'], inner_fold=j, configuration_id=cid, **summarize(group)))
        validation = primary[primary.case_id.isin(inner_ids) & primary.configuration_id.str.startswith('det_')]
        for budget in budgets:
            cid = select_variant(validation, budget)
            chosen.append(dict(fold=fold['fold'], budget=budget, configuration_id=cid,
                               status='selected' if cid else 'no_feasible_configuration'))
            for criterion in (0.10, 0.25):
                subset = metrics[(metrics.min_iou == criterion) & metrics.case_id.isin(test)]
                if cid:
                    selected = subset[subset.configuration_id == cid].copy()
                else:
                    selected = subset[subset.configuration_id == 'historical'].copy()
                    selected['FN'] = selected.TP + selected.FN
                    selected['small_FN'] = selected.small_TP + selected.small_FN
                    selected[['TP', 'FP', 'small_TP', 'pred_area', 'intersection']] = 0
                    selected['union'] = selected.reference_area
                    selected['dice'] = (selected.reference_area == 0).astype(float)
                    selected['iou'] = selected['dice']
                    selected['configuration_id'] = 'abstain'
                selected['family'], selected['budget'], selected['outer_fold'] = 'detector', budget, fold['fold']
                outer_rows.append(selected)
                for baseline in ('historical', 'otsu'):
                    b = subset[subset.configuration_id == baseline].copy()
                    b['family'], b['budget'], b['outer_fold'] = baseline, budget, fold['fold']
                    outer_rows.append(b)
    outer = pd.concat(outer_rows, ignore_index=True)
    summary = pd.DataFrame([dict(family=family, budget=budget, min_iou=iou, **summarize(g))
        for (family, budget, iou), g in outer.groupby(['family', 'budget', 'min_iou'])])
    per_case = pd.DataFrame([dict(family=family, budget=budget, min_iou=iou, case_id=case, **summarize(g))
        for (family, budget, iou, case), g in outer.groupby(['family', 'budget', 'min_iou', 'case_id'])])
    target = summary[(summary.budget == 8) & (summary.min_iou == .1)].set_index('family')
    promoted = False
    if not target.empty:
        d = target.loc['detector']
        promoted = all(d.recall > target.loc[b].recall and d.small_recall >= target.loc[b].small_recall
                       for b in ('historical', 'otsu')) and d.fp_per_slice <= 8
    decision = dict(eligible_for_promotion=bool(promoted), defaults_changed=False,
        criterion='Outer recall > both baselines; small recall >= both; outer FP/slice <= 8, IoU >= 0.10',
        interpretation='Small internal evaluation on reference-selected slices; not external validation.')
    return pd.DataFrame(chosen), outer, summary, per_case, pd.DataFrame(inner_rows), decision
