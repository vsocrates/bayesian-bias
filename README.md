# Bayesian Bias in Clinical Reasoning

A research project examining whether large language models (LLMs) exhibit racial bias in Bayesian diagnostic reasoning — mirroring the well-documented bias observed in human clinicians.

## Background

This project builds on [J. Brush et al.'s work](https://doi.org/10.1097/MLR.0000000000001533) which showed that physicians update disease probabilities differently based on a patient's stated race, even when all clinical information is identical. We asked: do LLMs exhibit the same pattern?

The core task presents a clinical vignette with a placeholder `{race}` field. For each vignette, the model is asked:

1. **Pretest probability** — "What is the probability this patient has disease X?"
2. **Posttest probability** — "Given this lab result, what is the updated probability?"

We compare estimated probabilities across racial groups and against the mathematically correct Bayesian update computed from the vignette's stated likelihood ratios.

## Experiments

### Race Bias (`brush_llm.py`)

Runs each vignette six times — once per racial group — across five trials. Measures whether pretest and posttest probability estimates differ by race after controlling for the clinical content.

**Conditions:**

| Experiment | LR info | CoT reasoning |
|------------|---------|---------------|
| `baseline` | Provided from vignette | Yes |
| `sensspec` | Model estimates sens/spec itself | Yes |
| `noLR`     | Not provided | Yes |

### SMDM Variant (`brush_llm_smdm.py`)

Presented at SMDM (Society for Medical Decision Making). Same setup without the race manipulation — all vignettes run with no specified race. Adds a `noCoT` condition to test whether chain-of-thought prompting affects calibration.

**Additional condition:**

| Experiment | LR info | CoT reasoning |
|------------|---------|---------------|
| `noCoT`    | Provided from vignette | No (suppressed) |

### Ebell CAP Experiment (`ebell_llm_funcs.py`)

A secondary experiment using Ebell et al.'s Community-Acquired Pneumonia (CAP) vignettes. The model estimates pretest probability, makes a management decision, then is shown the clinical prediction rule probability and asked whether it would change its decision.

## Repository Structure

```
.
├── src/                        # Source code
│   ├── brush_llm.py            # Main race bias experiment script
│   ├── brush_llm_smdm.py       # SMDM variant (no race, adds noCoT)
│   ├── brush_llm_funcs.py      # Shared helpers: data loading, case runner, result extraction
│   ├── ebell_llm_funcs.py      # Ebell CAP experiment functions
│   ├── llm_funcs.py            # Shared utilities (retry logic, Bayesian math)
│   └── prompts/
│       └── prompts.py          # All prompt templates and reasoning instruction strings
├── data/
│   ├── all_cases_clean.csv     # J. Brush clinical vignettes (pipe-separated)
│   ├── Vignette_Score_Annotations.csv  # Human annotation scores
│   └── prompts.csv             # Earlier prompt iteration reference
├── outputs/
│   ├── all_cases_*_v*.csv      # LLM experiment output CSVs (versioned runs)
│   └── output_dictionary.txt   # Notes on what each output file contains
├── images/
│   ├── difference_sensspec_v5.png              # Overall probability difference by race
│   └── difference_by_condition_sensspec_v5.png # Differences broken down by condition
├── notebooks/
│   ├── GenerateAnnotationDataset.ipynb         # Builds the annotation dataset
│   ├── Test_RunnableWithMessageHistory.ipynb   # LangChain conversation history testing
│   ├── v1/                                     # Early race bias analysis notebooks
│   ├── v2/                                     # Refined analysis with Bayes LR plots
│   └── smdm_plot_notebooks/                    # SMDM presentation plots and results
└── scripts/
    ├── jbrush_race.slurm       # SLURM job for race bias experiment (HPC cluster)
    ├── jbrush_smdm.slurm       # SLURM job for SMDM variant
    └── smdm_scripts/           # Additional SLURM scripts for SMDM conditions
```

## Setup

```bash
conda env create -f langchain.yml
conda activate langchain
```

Set the following environment variables (or add them to a `.env` file):

```
AZURE_OPENAI_KEY_EAST1=...
AZURE_OPENAI_ENDPOINT_EAST1=...
```

## Running

```bash
# Race bias experiment — baseline condition, test mode
python src/brush_llm.py baseline -t

# Race bias experiment — full run, sensspec condition
python src/brush_llm.py sensspec -m decile-gpt-4o --temperature 0.8

# SMDM variant — noCoT condition
python src/brush_llm_smdm.py noCoT -m decile-gpt-4o

# Custom data and output paths
python src/brush_llm.py baseline --data data/all_cases_clean.csv --output-dir outputs/
```

## Key Findings

Pretest probability estimates were relatively stable across racial groups. However, posttest probability updates showed statistically significant differences by race in some conditions, with the model applying different Bayesian weight to identical lab results depending on the patient's stated race — mirroring the anchoring and adjustment bias documented in human physicians.

The `sensspec` condition (where the model self-estimates sensitivity/specificity) showed larger racial disparities than the `baseline` condition where LRs were provided, suggesting that the bias is amplified when the model must draw on learned priors rather than given statistics.
