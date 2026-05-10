# Intro to Math Modeling Final Project

This repository contains a final project modeling the first reported COVID-19 wave in Hubei, China with SIR-type models.

## Project Question

Can a simple SIR model approximate the first COVID-19 wave in Hubei, and how does allowing the transmission rate to change over time affect the model's interpretation?

## Contents

- `hubei_sir_project.ipynb`: main notebook with data processing, SIR fitting, counterfactual simulations, and time-varying beta analysis.
- `hubei_sir_analysis.py`: script version of the analysis.
- `figures/`: generated figures used in the report.
- `results/`: generated CSV outputs with fitted parameters and scenario results.
- `report/main.tex`: LaTeX report source.
- `report/references.bib`: bibliography file.
- `report/figures/`: figures copied for Overleaf.

## Data Source

The analysis uses the Johns Hopkins CSSE COVID-19 confirmed cases time series:

https://github.com/CSSEGISandData/COVID-19

## How to Run

Install the required Python packages if needed:

```bash
pip install numpy pandas matplotlib scipy
```

Run the script:

```bash
python hubei_sir_analysis.py
```

Or open and run:

```text
hubei_sir_project.ipynb
```

## Report

The report is written in LaTeX. To compile it in Overleaf, upload:

- `report/main.tex`
- `report/references.bib`
- `report/figures/`
