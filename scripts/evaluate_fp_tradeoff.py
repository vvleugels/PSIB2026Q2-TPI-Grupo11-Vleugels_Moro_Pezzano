"""Post-hoc sensitivity/FP analysis of saved predictions; never changes defaults."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from psib.detection_validation import nested_evaluation, summarize

BUDGETS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, float('inf'))


def evaluate(run, out):
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(run / 'detection_per_slice.csv')
    folds = json.loads((run / 'folds.json').read_text())
    chosen, outer, summary, per_case, inner, _ = nested_evaluation(metrics, folds, BUDGETS)
    for name, frame in [('selected_configurations', chosen), ('outer_per_slice', outer),
                        ('summary', summary), ('per_case', per_case), ('inner_validation', inner)]:
        frame.to_csv(out / f'{name}.csv', index=False)
    raw = pd.DataFrame([dict(configuration_id=cid, min_iou=iou, **summarize(g))
        for (cid, iou), g in metrics.groupby(['configuration_id', 'min_iou'])])
    raw['pareto'] = False
    for i, row in raw.iterrows():
        peers = raw[raw.min_iou == row.min_iou]
        dominated = ((peers.recall >= row.recall) & (peers.fp_per_slice <= row.fp_per_slice)
                     & ((peers.recall > row.recall) | (peers.fp_per_slice < row.fp_per_slice))).any()
        raw.loc[i, 'pareto'] = not dominated
    raw.to_csv(out / 'exploratory_frontier.csv', index=False)
    (out / 'analysis.json').write_text(json.dumps(dict(
        source=str(run.resolve()), budgets=[v if np.isfinite(v) else 'unlimited' for v in BUDGETS],
        selection='Inner recall at IoU 0.10; ties: fewer FP, greater Dice, configuration ID',
        post_hoc=True, defaults_changed=False,
        note='Same frozen patients, folds and masks; budgets examined after original results. No new independent validation.'), indent=2))

    primary = summary[summary.min_iou == .1]
    det = primary[primary.family == 'detector']
    lines = ['# Sensibilidad frente a falsos positivos: reevaluación', '',
        'Análisis posterior al experimento original, sin regenerar máscaras ni cambiar defaults. '
        'Se conservan los diez pacientes, 50 cortes y particiones congeladas. '
        'Cada límite selecciona parámetros usando únicamente los ocho pacientes internos; '
        'los dos pacientes restantes evalúan esa selección. Sin límite se maximiza recall '
        'y se desempata por menos FP y mayor Dice. Los límites son puntos exploratorios, no criterios clínicos.','',
        '**No es una nueva validación independiente:** se decidió explorar estos límites después '
        'de observar el primer resultado. Las observaciones lesionales son componentes 2D.','',
        '| Límite interno FP/corte | Folds factibles | Sensibilidad | <10 mm | FP/corte observados | Precisión | FP/corte negativo |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in det.itertuples():
        feasible = int(chosen[chosen.budget == r.budget].configuration_id.notna().sum())
        label = f'{r.budget:g}' if np.isfinite(r.budget) else 'Sin límite'
        precision = f'{r.precision:.1%}' if pd.notna(r.precision) else '—'
        lines.append(f'| {label} | {feasible}/5 | {r.recall:.1%} | {r.small_recall:.1%} | '
                     f'{r.fp_per_slice:.2f} | {precision} | {r.fp_per_negative_slice:.2f} |')
    lines += ['', 'Los folds sin configuración factible abstienen: su sensibilidad cero no describe '
              'las máscaras de las variantes. Los límites son internos; los FP observados en pacientes '
              'retenidos pueden superarlos. Los puntos no son umbrales de un único detector y no '
              'tienen por qué ser monótonos.', '', '## Baselines fijos y selección sin límite', '',
              '| Método | IoU mínimo | TP / referencias | Sensibilidad | <10 mm | FP/corte | Dice por paciente |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary[summary.budget == np.inf].itertuples():
        lines.append(f'| {r.family} | {r.min_iou:.2f} | {r.TP}/{r.TP+r.FN} | {r.recall:.1%} | '
                     f'{r.small_recall:.1%} | {r.fp_per_slice:.2f} | {r.dice_mean:.3f} |')
    lines += ['', '## Resultados por paciente, IoU ≥0,10', '',
              '| Paciente | Recall Otsu | Recall sin límite | FP/corte Otsu | FP/corte sin límite |',
              '|---|---:|---:|---:|---:|']
    pc = per_case[(per_case.budget == np.inf) & (per_case.min_iou == .1)]
    for case, group in pc.groupby('case_id'):
        g = group.set_index('family'); a, b = g.loc['otsu'], g.loc['detector']
        fmt = lambda x: f'{x:.1%}' if pd.notna(x) else 'Sin lesiones'
        lines.append(f'| {case} | {fmt(a.recall)} | {fmt(b.recall)} | {a.fp_per_slice:.2f} | {b.fp_per_slice:.2f} |')
    lines += ['', '## Selección interna sin límite', '',
              '| Fold | Configuración | Pacientes retenidos |', '|---|---|---|']
    for r in chosen[chosen.budget == np.inf].itertuples():
        f = next(f for f in folds if f['fold'] == r.fold)
        lines.append(f'| {r.fold} | {r.configuration_id} | {f["test"]} |')
    lines += ['', 'La tabla `exploratory_frontier.csv` marca las variantes no dominadas en sensibilidad '
              'y FP usando toda la cohorte, incluidos los baselines. Es descriptiva y no selecciona '
              'el ganador de evaluación. `per_case.csv` y `outer_per_slice.csv` conservan todos los conteos.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    feasible_budgets = chosen.groupby('budget').configuration_id.count()
    for ax, metric, title in zip(axes, ['recall', 'small_recall'], ['Todas las lesiones', 'Lesiones <10 mm']):
        valid = det[det.budget.isin(feasible_budgets[feasible_budgets == 5].index)]
        ax.scatter(valid.fp_per_slice, valid[metric], label='Selección por paciente', color='C0')
        for (fp, recall), group in valid.groupby(['fp_per_slice', metric]):
            label = '/'.join(f'{b:g}' if np.isfinite(b) else 'Sin límite' for b in group.budget)
            ax.annotate(label, (fp, recall), xytext=(3, 4), textcoords='offset points', fontsize=8)
        for family in ['historical', 'otsu']:
            r = primary[primary.family == family].iloc[0]
            ax.scatter(r.fp_per_slice, r[metric], marker='s', label=family)
        ax.set(xlabel='FP/corte en pacientes retenidos', ylabel='Sensibilidad (IoU ≥0,10)', ylim=(0, 1), title=title)
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle('Límites internos exploratorios; solo puntos con 5/5 folds factibles')
    fig.tight_layout(); fig.savefig(out / 'tradeoff.png', dpi=160); plt.close(fig)
    print((out / 'report.md').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run, args.output)
