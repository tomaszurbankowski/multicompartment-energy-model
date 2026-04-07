from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

PHENOTYPE_LABELS = {
    'compliance_dominant': 'Compliance-dominant',
    'resistance_dominant': 'Resistance-dominant',
    'mixed': 'Mixed',
}
WAVEFORM_LABELS = {
    'square': 'Square',
    'decelerating': 'Decelerating',
    'sinusoidal': 'Sinusoidal',
}
PHENOTYPE_ORDER = ['compliance_dominant', 'resistance_dominant', 'mixed']
WAVEFORM_ORDER = ['square', 'decelerating', 'sinusoidal']
TI_ORDER = [0.6, 1.0, 1.5]
PAUSE_ORDER = [0.0, 0.2]


def load_inputs(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sim = pd.read_csv(base_dir / 'simulation_results.csv')
    mpd = pd.read_csv(base_dir / 'mp_matched_divergence.csv')
    psi = pd.read_csv(base_dir / 'phenotype_sensitivity.csv')
    defs = pd.read_csv(base_dir / 'phenotype_definitions.csv')
    return sim, mpd, psi, defs


def save_all_formats(fig: plt.Figure, stem: str) -> None:
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT_DIR / f'{stem}.{ext}', bbox_inches='tight')


def draw_phenotype_schematic(ax: plt.Axes, compliances: list[float], resistances: list[float], title: str) -> None:
    """Draw a cleaner phenotype schematic for Figure 1.

    Key fixes:
    - more vertical space for top and bottom annotations
    - 'Airway opening' no longer overlaps compartment labels
    - PEEP / R / compliance values are clearly separated
    """
    n = len(compliances)
    x_positions = np.arange(n)
    c = np.asarray(compliances, dtype=float)
    r = np.asarray(resistances, dtype=float)
    c_norm = (c - c.min()) / (c.max() - c.min() + 1e-12)
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-12)

    # More room above and below to avoid label collisions.
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-0.18, 1.18)

    top_y = 0.86
    bottom_y = 0.10
    upper_stem_top = 0.80
    lower_stem_bottom = 0.18

    # Trunk lines.
    ax.plot([-0.35, n - 0.65], [top_y, top_y], lw=1.2)
    ax.plot([-0.35, n - 0.65], [bottom_y, bottom_y], lw=1.2)

    for i, x in enumerate(x_positions):
        height = 0.28 + 0.42 * c_norm[i]
        width = 0.16 + 0.18 * r_norm[i]
        ax.plot([x, x], [top_y, upper_stem_top], lw=1.2)
        ax.plot([x, x], [lower_stem_bottom, bottom_y], lw=1.2)
        rect = Rectangle((x - width / 2, 0.50 - height / 2), width, height, fill=False, lw=1.5)
        ax.add_patch(rect)

        # Top labels higher than airway label.
        ax.text(x, 1.02, f'C{i+1}', ha='center', va='bottom', fontsize=8)
        # Bottom resistance labels clearly separated from PEEP/compliance values.
        ax.text(x, -0.005, f'R={r[i]:.0f}', ha='center', va='top', fontsize=7.5)
        ax.text(x, -0.075, f'{c[i]:.3f}', ha='center', va='top', fontsize=7.5)

    # Offset left labels to avoid overlap with C1.
    ax.text(-0.52, 0.95, 'Airway opening', fontsize=7.5, ha='left', va='bottom')
    ax.text(-0.52, 0.045, 'PEEP', fontsize=7.5, ha='left', va='center')

    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_xlabel('Compliance (L/cmH$_2$O)', labelpad=10)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure1(sim: pd.DataFrame, defs: pd.DataFrame) -> None:
    ref_cols = [c for c in sim.columns if c.startswith('ref_')]
    baseline = sim[(sim['waveform'] == 'square') & (sim['ti_s'] == 1.0) & (sim['pause_fraction'] == 0.0)].copy()
    baseline['phenotype'] = pd.Categorical(baseline['phenotype'], PHENOTYPE_ORDER, ordered=True)
    baseline = baseline.sort_values('phenotype')

    fig = plt.figure(figsize=(11.2, 7.4))
    # Increased hspace so panel D has more breathing room.
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.48, wspace=0.30)

    for i, phenotype in enumerate(PHENOTYPE_ORDER):
        row = defs[defs['phenotype'] == phenotype].iloc[0]
        compliances = [row[f'C_{j}'] for j in range(1, 7)]
        resistances = [row[f'R_{j}'] for j in range(1, 7)]
        ax = fig.add_subplot(gs[0, i])
        draw_phenotype_schematic(ax, compliances, resistances, PHENOTYPE_LABELS[phenotype])
        ax.text(-0.18, 1.08, chr(ord('A') + i), transform=ax.transAxes, fontsize=12, fontweight='bold')

    ax4 = fig.add_subplot(gs[1, :])
    x = np.arange(1, len(ref_cols) + 1)
    markers = {'compliance_dominant': 'o', 'resistance_dominant': 's', 'mixed': '^'}
    for phenotype in PHENOTYPE_ORDER:
        row = baseline[baseline['phenotype'] == phenotype].iloc[0]
        y = row[ref_cols].to_numpy(dtype=float)
        ax4.plot(x, y, marker=markers[phenotype], linewidth=1.8, markersize=5, label=PHENOTYPE_LABELS[phenotype])
    ax4.set_xlabel('Compartment')
    ax4.set_ylabel('Regional energy fraction (REF)')
    ax4.set_xticks(x)
    ax4.set_title('Baseline regional inspiratory energy distribution\n(reference condition: square waveform, inspiratory time 1.0 s, pause fraction 0.0)', pad=10)
    ax4.legend(frameon=False, ncol=3, loc='upper left')
    ax4.grid(alpha=0.25, linewidth=0.5)
    ax4.text(-0.03, 1.04, 'D', transform=ax4.transAxes, fontsize=12, fontweight='bold')

    save_all_formats(fig, 'Figure1_phenotypes_baseline_ref')
    plt.close(fig)


def figure2(sim: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), constrained_layout=True)
    vmin = sim['eii_cv'].min()
    vmax = sim['eii_cv'].max()
    im = None

    for idx, phenotype in enumerate(PHENOTYPE_ORDER):
        ax = axes[idx]
        subset = sim[sim['phenotype'] == phenotype].copy()
        mat = np.zeros((len(WAVEFORM_ORDER), len(TI_ORDER) * len(PAUSE_ORDER)))
        col_labels = []
        for ti in TI_ORDER:
            for p in PAUSE_ORDER:
                col_labels.append(f'Ti={ti}\nP={p}')
        for i, wf in enumerate(WAVEFORM_ORDER):
            k = 0
            for ti in TI_ORDER:
                for p in PAUSE_ORDER:
                    val = subset[(subset['waveform'] == wf) & (subset['ti_s'] == ti) & (subset['pause_fraction'] == p)]['eii_cv'].iloc[0]
                    mat[i, k] = val
                    k += 1
        im = ax.imshow(mat, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(PHENOTYPE_LABELS[phenotype])
        ax.set_yticks(np.arange(len(WAVEFORM_ORDER)))
        ax.set_yticklabels([WAVEFORM_LABELS[w] for w in WAVEFORM_ORDER])
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_xlabel('Inspiratory time and pause fraction')
        if idx == 0:
            ax.set_ylabel('Waveform')
        ax.text(-0.18, 1.06, chr(ord('A') + idx), transform=ax.transAxes, fontsize=12, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label('EII (coefficient of variation of REF)')
    save_all_formats(fig, 'Figure2_EII_heatmaps')
    plt.close(fig)


def figure3(sim: pd.DataFrame, mpd: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)

    # Panel A
    ax1 = axes[0]
    markers = {'compliance_dominant': 'o', 'resistance_dominant': 's', 'mixed': '^'}
    for phenotype in PHENOTYPE_ORDER:
        subset = sim[sim['phenotype'] == phenotype]
        ax1.scatter(
            subset['mechanical_power_j_min'],
            subset['dces'],
            s=30,
            alpha=0.85,
            marker=markers[phenotype],
            label=PHENOTYPE_LABELS[phenotype],
        )
    ax1.set_xlabel('Mechanical power (J/min)')
    ax1.set_ylabel('Dominant compartment energy share (DCES)')
    ax1.set_title('Phenotype-specific clustering of MP and DCES')
    ax1.grid(alpha=0.25, linewidth=0.5)
    ax1.legend(frameon=False, loc='lower right')
    ax1.text(-0.14, 1.04, 'A', transform=ax1.transAxes, fontsize=12, fontweight='bold')

    # Panel B - matched to the corrected standalone Figure 3 code
    ax2 = axes[1]
    if 'anchor_phenotype' in mpd.columns:
        for phenotype in PHENOTYPE_ORDER:
            subset = mpd[mpd['anchor_phenotype'] == phenotype]
            if len(subset) == 0:
                continue
            ax2.scatter(
                subset['matched_eii_range'],
                subset['matched_dces_range'],
                s=30,
                alpha=0.85,
                marker=markers[phenotype],
                label=PHENOTYPE_LABELS[phenotype],
            )
    else:
        ax2.scatter(
            mpd['matched_eii_range'],
            mpd['matched_dces_range'],
            s=30,
            alpha=0.85,
        )

    ax2.set_xlabel('Range of EII within MP-matched sets')
    ax2.set_ylabel('Range of DCES within MP-matched sets')
    ax2.set_title('Divergence within ±5% MP-matched sets')
    ax2.grid(alpha=0.25, linewidth=0.5)
    if 'anchor_phenotype' in mpd.columns:
        ax2.legend(frameon=False, loc='lower right')
    ax2.text(-0.14, 1.04, 'B', transform=ax2.transAxes, fontsize=12, fontweight='bold')

    save_all_formats(fig, 'Figure3_MP_DCES_divergence')
    plt.close(fig)


def figure4(psi: pd.DataFrame) -> None:
    psi = psi.copy()
    psi['phenotype'] = pd.Categorical(psi['phenotype'], PHENOTYPE_ORDER, ordered=True)
    psi = psi.sort_values('phenotype')

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    x = np.arange(len(psi))
    labels = [PHENOTYPE_LABELS[p] for p in psi['phenotype']]

    ax1 = axes[0]
    ax1.bar(x, psi['psi_eii'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right')
    ax1.set_ylabel('Phenotype sensitivity index (ΔEII)')
    ax1.set_title('Sensitivity based on EII range')
    ax1.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax1.text(-0.14, 1.04, 'A', transform=ax1.transAxes, fontsize=12, fontweight='bold')

    ax2 = axes[1]
    ax2.bar(x, psi['psi_dces'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right')
    ax2.set_ylabel('Phenotype sensitivity index (ΔDCES)')
    ax2.set_title('Sensitivity based on DCES range')
    ax2.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax2.text(-0.14, 1.04, 'B', transform=ax2.transAxes, fontsize=12, fontweight='bold')

    save_all_formats(fig, 'Figure4_sensitivity_summary')
    plt.close(fig)


def write_summary(sim: pd.DataFrame, mpd: pd.DataFrame, psi: pd.DataFrame) -> None:
    summary = {
        'n_simulations': int(len(sim)),
        'phenotypes': [PHENOTYPE_LABELS[p] for p in PHENOTYPE_ORDER],
        'waveforms': [WAVEFORM_LABELS[w] for w in WAVEFORM_ORDER],
        'ti_values_s': TI_ORDER,
        'pause_fractions': PAUSE_ORDER,
        'max_mp_matched_eii_range': float(mpd['matched_eii_range'].max()) if len(mpd) else None,
        'max_mp_matched_dces_range': float(mpd['matched_dces_range'].max()) if len(mpd) else None,
        'psi_eii': {PHENOTYPE_LABELS[row['phenotype']]: float(row['psi_eii']) for _, row in psi.iterrows()},
        'psi_dces': {PHENOTYPE_LABELS[row['phenotype']]: float(row['psi_dces']) for _, row in psi.iterrows()},
    }
    with open(OUT_DIR / 'figure_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    sim, mpd, psi, defs = load_inputs(BASE_DIR)
    figure1(sim, defs)
    figure2(sim)
    figure3(sim, mpd)
    figure4(psi)
    write_summary(sim, mpd, psi)
    print(f'Generated figures in: {OUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
