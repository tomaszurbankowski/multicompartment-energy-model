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
