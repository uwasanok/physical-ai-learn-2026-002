"""Generate reproducible local-pass experiments; not a robotic simulation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cleaning_model import scene, sweep, residual_area_fraction

out = Path(__file__).resolve().parents[1] / 'output'
out.mkdir(exist_ok=True)
results = []
fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey=True)
for row, seed in enumerate([42, 123, 999]):
    model, patches = scene(seed)
    history = []
    for round_id in range(6):
        if round_id:
            sweep(model)
        history.append(dict(raster_round=round_id, remaining_area_fraction=residual_area_fraction(model),
                            dirty_cells=int(model.remaining.sum())))
        if round_id in [0, 1, 2, 5]:
            ax = axes[row, [0, 1, 2, 5].index(round_id)]
            ax.scatter(model.uv[:, 0], model.uv[:, 1], c='#dfe5e9', s=20, marker='s')
            m = model.remaining
            ax.scatter(model.uv[m, 0], model.uv[m, 1], c=model.required[m], vmin=1, vmax=5,
                       cmap='YlOrBr', s=20, marker='s')
            ax.set_title(f'Seed {seed} | round {round_id}\nResidual {history[-1]["remaining_area_fraction"]:.1%}', fontsize=10)
            ax.set_aspect('equal'); ax.set_xlim(-.12, .12); ax.set_ylim(.02, .16)
            if row == 2: ax.set_xlabel('Surface u [m]')
            if round_id == 0: ax.set_ylabel('Surface v [m]')
    results.append(dict(seed=seed, patches=patches, history=history))
fig.suptitle('CURVED-SURFACE CLEANING SURROGATE | random locations, required local passes n = 1 / 3 / 5')
fig.text(.5, .015, 'Surface-coordinate map (not a camera image). One raster round can contain multiple local passes. No physics or AI.', ha='center', fontsize=9)
fig.tight_layout(rect=[0,.04,1,.95]); fig.savefig(out/'cleaning_study.png', dpi=150)
(out/'cleaning_results.json').write_text(json.dumps(dict(status='geometric_surrogate_only',
    contact='distance and normal alignment; no force simulation',
    removal='binary disappearance after n completed local passes',
    scenes=results), ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
