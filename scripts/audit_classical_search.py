"""Post-hoc audit of saved classical-search masks; never modifies the source run."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from psib.classical_search import summarize_search, ranking
from psib.lesion_eval import prepare_reference, match_lesions_one_to_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'results/classical_search/classical-search-20261104')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.run / 'individual_per_slice.csv')
    frame = frame[frame.min_iou == .1]
    candidates = frame[frame.configuration_id != 'default']
    folds = json.loads((args.run / 'folds.json').read_text())
    choices, outer = [], []
    for fold in folds:
        scores = {cid: summarize_search(g) for cid, g in
                  candidates[candidates.case_id.isin(fold['train'])].groupby('configuration_id')}
        for objective in ('F2', 'dice_mean', 'recall', 'small_recall'):
            cid = min(scores, key=lambda c: ranking(scores[c], (c,), objective))
            choices.append(dict(fold=fold['fold'], objective=objective, configuration_id=cid))
            outer.append(candidates[(candidates.configuration_id == cid) &
                                    candidates.case_id.isin(fold['test'])].assign(objective=objective, fold=fold['fold']))
    outer = pd.concat(outer, ignore_index=True)
    summary = pd.DataFrame([dict(objective=o, **summarize_search(g)) for o, g in outer.groupby('objective')])
    pd.DataFrame(choices).to_csv(args.output / 'choices.csv', index=False)
    outer.to_csv(args.output / 'outer_posthoc.csv', index=False)
    summary.to_csv(args.output / 'summary_posthoc.csv', index=False)

    # Paired ablation: exactly the same preprocessing, P0 versus P5 cleanup.
    # Reference IDs identify gains/losses; net TP alone would hide substitutions.
    rows, unique = [], []
    hashes = [hashlib.sha256() for _ in range(216)]
    for path in sorted((args.run / 'inputs').glob('*.npz')):
        case, k = int(path.stem.split('_')[1]), int(path.stem.split('_')[3])
        with np.load(path) as data:
            ref, spacing = data['reference'], data['spacing']
        prepared = prepare_reference(ref, spacing[:2])
        packed = np.load(args.run / 'predictions' / (path.stem + '.npy'))
        unique.append(dict(case_id=case, slice_index=k, unique_masks=len(np.unique(packed[:216], axis=0))))
        for index, digest in enumerate(hashes):
            digest.update(packed[index].tobytes())
        for start in (36, 48, 54, 108):
            found = []
            for index in (start, start + 5):
                pred = np.unpackbits(packed[index], count=ref.size).reshape(ref.shape).astype(bool)
                matches = match_lesions_one_to_one(pred, ref, spacing[:2], .1, prepared_reference=prepared)
                found.append(set(matches.loc[matches.status == 'TP', 'reference_id']))
            for rid, diameter in prepared[2].items():
                rows.append(dict(case_id=case, slice_index=k, raw=f'otsu_{start:03d}',
                                 cleaned=f'otsu_{start+5:03d}', reference_id=rid,
                                 diameter_mm=diameter, raw_detected=rid in found[0],
                                 cleaned_detected=rid in found[1]))
    pd.DataFrame(rows).to_csv(args.output / 'cleanup_per_lesion.csv', index=False)
    pd.DataFrame(unique).to_csv(args.output / 'unique_masks.csv', index=False)
    metadata = dict(source=str(args.run.resolve()), unique_cohort_predictions=len({h.hexdigest() for h in hashes}),
                    interpretation='Post-hoc diagnostic using previously explored patients; not independent validation.')
    (args.output / 'audit.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)
    print(metadata, flush=True)


if __name__ == '__main__':
    main()
