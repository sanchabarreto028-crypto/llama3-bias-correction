# Demographic Bias Detection and Correction in LLMs for Clinical Decision-Making

**MSc Data Science Dissertation — Newcastle University, 2025–26**  
**Author:** Sancha Barreto | **Supervisor:** Dr Vlad González-Zelaya

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://python.org)
[![Model](https://img.shields.io/badge/model-LLaMA--3--8B--Instruct-orange)](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

## Overview

Large language models are increasingly deployed in clinical decision-support tools, yet growing evidence shows they produce systematically different outputs for identical clinical cases based solely on patient demographics. This project investigates **whether LLaMA-3-8B-Instruct exhibits measurable racial and gender bias in clinical question answering**, and applies **Direct Preference Optimisation (DPO) with LoRA fine-tuning** to correct it.

Key result: DPO reduced the proxy counterfactual fairness gap by **52.8%** on BiasMedQA without compromising clinical accuracy.

---

## Method at a Glance

```
BiasMedQA (759 cases × 8 demographics = 6,072 variants)
CPV       (1,202 cases × 5 ethnicities × 2 genders = 12,310 variants)
        │
        ▼
Demographic Attribute Injection (race + gender substitution)
        │
        ▼
LLaMA-3-8B-Instruct  (temperature=0, greedy decoding)
        │
        ▼
Fairness Metric Computation
  • Proxy Counterfactual Fairness Gap
  • Demographic Parity Difference
  • Equalised Odds Difference
  • Minimax Fairness
        │
        ▼
DPO Preference Dataset Construction
  BiasMedQA: 52 biased questions → 421 preference pairs
  CPV:      110 biased cases    → 1,317 preference pairs
        │
        ▼
DPO Fine-Tuning + LoRA (r=16, α=32, β=0.1, 1 epoch)
        │
        ▼
Post-Correction Re-Evaluation (same test sets)
```

---

## Results Summary

### BiasMedQA

| Metric | Before DPO | After DPO | Δ |
|---|---|---|---|
| Overall Accuracy | 0.4107 | 0.4157 | +0.50% |
| Proxy CF Gap | 0.1397 | 0.0659 | **−52.8%** |
| Demographic Parity Difference | 0.0171 | 0.0132 | −22.8% |

### CPV (Counterfactual Patient Variations)

| Metric | Before DPO | After DPO | Δ |
|---|---|---|---|
| Overall Accuracy | 0.5224 | 0.5245 | +0.21% |
| Proxy CF Gap | 0.0077 | 0.1040 | +0.096 |
| Demographic Parity Difference | 0.0077 | 0.0065 | −15.6% |

> **Note on CPV proxy CF gap:** The baseline CPV CF gap was already near-zero (0.0077), indicating the model was highly consistent (though not always correct) across demographic variants. The post-DPO increase reflects reduced consistency after alignment, not increased systematic group disadvantage — ethnicity differences remained statistically non-significant (χ², p>0.95). Gender differences in CPV were significant both before and after DPO (p=0.005), indicating this axis of bias was not addressed by the current alignment strategy.

### Statistical Significance (Chi-Squared Tests)

| Dataset | Axis | Before | After |
|---|---|---|---|
| BiasMedQA | Ethnicity | p=0.9965, n.s. | p=0.9985, n.s. |
| BiasMedQA | Gender | p=0.9792, n.s. | p=0.8964, n.s. |
| CPV | Ethnicity | p=0.9839, n.s. | p=0.9949, n.s. |
| CPV | Gender | **p=0.0046, sig.** | **p=0.0056, sig.** |

---

## Repository Structure

```
llama3-bias-correction/
├── llama3_bias_correction.ipynb   # Main pipeline notebook (Google Colab)
├── results/
│   ├── baseline_metrics.json      # Pre-DPO BiasMedQA fairness metrics
│   ├── cpv_metrics.json           # Pre-DPO CPV fairness metrics
│   ├── corrected_metrics.json     # Post-DPO BiasMedQA fairness metrics
│   ├── corrected_cpv_metrics.json # Post-DPO CPV fairness metrics
|   ├── preference_pairs_combined.json  # DPO training preference pairs
│   └── extended_analysis.json     # Gender breakdown + chi-squared results
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** Raw inference outputs (`*.jsonl`) and model checkpoints are excluded from version control due to file size. Aggregated metrics in `results/` are sufficient to reproduce all reported figures.

---

## Reproducing the Results

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (A100 recommended; experiments run on Google Colab Pro+)
- HuggingFace account with access granted to [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

### Setup

```bash
git clone https://github.com/sanchabarreto028-crypto/llama3-bias-correction.git
cd llama3-bias-correction
pip install -r requirements.txt
```

Set your HuggingFace token:

```bash
export HF_TOKEN=your_token_here
```

### Running in Google Colab

Open `llama3_bias_correction.ipynb` in Google Colab. The notebook is structured into self-contained sections — each section saves its outputs to Google Drive before the next section begins, so inference can be resumed after runtime disconnections.

**Section order:**
1. Setup & data preparation
2. BiasMedQA dataset construction and demographic injection
3. Baseline inference (BiasMedQA + CPV)
4. Fairness metric computation
5. Preference dataset construction
6. DPO fine-tuning with LoRA
7. Post-correction inference and evaluation
8. Extended analysis (gender breakdown, chi-squared tests)

---

## Datasets

| Dataset | Source | Licence |
|---|---|---|
| BiasMedQA | [Schmidgall et al., 2024](https://arxiv.org/abs/2408.03948) | MIT |
| CPV (medqa-cpv) | [Benkirane et al., 2024](https://arxiv.org/abs/2410.16574) | CC BY 4.0 |

BiasMedQA was filtered to exclude staff and administrative scenarios (759 of 1,273 original questions retained). Demographic injection was applied counterfactually: race and gender attributes were substituted while all clinical content remained unchanged.

---

## Fairness Metrics

**Proxy Counterfactual Fairness Gap (proxy CF gap):** Proportion of clinical cases where the model produces different answers across demographic variants of the same scenario. A value of 0 indicates perfect counterfactual fairness.

**Demographic Parity Difference (DPD):** Maximum accuracy gap between any two demographic groups.

**Equalised Odds Difference:** Maximum difference in true positive rates across demographic groups.

**Minimax Fairness:** Same as DPD for binary correct/incorrect outcomes.

---

## Limitations

- Quantised 4-bit inference may introduce rounding effects not present in full-precision deployment
- BiasMedQA uses USMLE-style MCQs; findings may not generalise to free-text clinical generation
- DPO was trained on 1 epoch with a balanced 842-pair dataset; scaling to the full 1,738 pairs may yield stronger generalisation
- CPV gender bias (p=0.005) persisted post-DPO, suggesting demographic axes of bias do not respond uniformly to alignment

---

## Citation

If you use this code or build on this work, please cite:

```
Barreto, S. S. (2026). Detecting and Correcting Demographic and Gender Bias in Large
Language Models for Clinical Decision Support. MSc Dissertation, Newcastle University.
```

---

## References

- Schmidgall et al. (2024). BiasMedQA. *npj Digital Medicine*. https://doi.org/10.1038/s41746-024-01283-6
- Benkirane et al. (2024). CPV. arXiv:2410.16574
- Poulain et al. (2024). Aligning (Medical) LLMs for (Counterfactual) Fairness. arXiv:2408.12055
- Rafailov et al. (2023). Direct Preference Optimization. *NeurIPS*
- Hu et al. (2021). LoRA. arXiv:2106.09685
- Zack et al. (2024). GPT-4 racial and gender biases in healthcare. *Lancet Digital Health*

---

## Licence

Code: MIT | Data outputs: CC BY 4.0 | See `LICENSE` for full terms.
