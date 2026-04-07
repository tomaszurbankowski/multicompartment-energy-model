# multicompartment-energy-model

Python code for simulating inspiratory energy distribution in multicompartment lung phenotypes and generating publication-ready figures.

## Overview

This repository contains a simple multicompartment lung model designed to study how inspiratory temporal patterning influences regional inspiratory energy distribution across predefined mechanical phenotypes.

The code includes:
- a simulation model of parallel lung compartments with different compliance and resistance profiles,
- predefined virtual phenotypes representing compliance-dominant, resistance-dominant, and mixed heterogeneity,
- analysis of inspiratory energy distribution using regional energy fraction (REF), energy inequality index (EII), and dominant compartment energy share (DCES),
- generation of publication-ready figures in PNG, SVG, and PDF formats.

## Repository contents

- `virtual_phenotypes_energy_model.py`  
  Runs the simulations, computes summary metrics, and exports result tables.

- `generate_publication_figures.py`  
  Generates publication-ready figures from the exported simulation outputs.

- `run_full_pipeline.py`  
  Runs the full workflow sequentially in one step.

## Model summary

The model represents the lung as a set of parallel compartments exposed to the same airway opening pressure. Each phenotype is defined by a different distribution of compartmental compliance and resistance. For each scenario, the code simulates one breath under volume-controlled conditions and calculates inspiratory energy distribution metrics.

The default scenario grid includes:
- 3 phenotypes,
- 3 inspiratory waveform shapes: square, decelerating, and sinusoidal,
- 3 inspiratory times: 0.6 s, 1.0 s, and 1.5 s,
- 2 pause fractions: 0.0 and 0.2.

## Requirements

Tested with Python 3.

Required Python packages:
- numpy
- pandas
- matplotlib

You can install them with:

```bash
pip install numpy pandas matplotlib
```

## How to run

Place all files in the same folder.

### Run the full pipeline

```bash
python3 run_full_pipeline.py
```

### Or run step by step

First run the simulations:

```bash
python3 virtual_phenotypes_energy_model.py
```

Then generate the figures:

```bash
python3 generate_publication_figures.py
```

## Output files

Running the model script generates:
- `simulation_results.csv`
- `phenotype_sensitivity.csv`
- `mp_matched_divergence.csv`
- `phenotype_definitions.csv`

It also generates intermediate exploratory figures:
- `figure_baseline_ref.png`
- `figure_eii_heatmap_*.png`
- `figure_mp_vs_dces.png`
- `figure_phenotype_sensitivity.png`

Running the figure-generation script creates publication-ready figures:
- `Figure1_phenotypes_baseline_ref.png/.svg/.pdf`
- `Figure2_EII_heatmaps.png/.svg/.pdf`
- `Figure3_MP_DCES_divergence.png/.svg/.pdf`
- `Figure4_sensitivity_summary.png/.svg/.pdf`
- `figure_summary.json`

All outputs are saved to the same folder as the scripts.

## Notes

This code is intended for research and educational use. It is a simplified in silico model and is not intended for clinical decision-making.

## Citation

If you use this code in academic work, please cite the associated manuscript when available.

## License

This repository is distributed under the MIT License. See the `LICENSE` file for details.
