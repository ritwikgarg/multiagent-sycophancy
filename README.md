# Multi-Agent Sycophancy

**Evaluating and Mitigating Sycophancy in Multi-Agent LLM Debates**

*CS 546 Advanced NLP, Fall 2025 — University of Illinois Urbana-Champaign*

---

## 📋 Project Objectives

This project investigates **sycophancy** — the tendency of large language models (LLMs) to agree with users or other agents regardless of factual correctness — in multi-agent debate settings. Our key objectives are:

1. **Measure sycophancy** in individual LLMs using multiple metrics derived from the MMLU benchmark
2. **Analyze sycophancy dynamics** in multi-agent debates where multiple LLMs interact and influence each other
3. **Develop sycophancy-aware scoring systems** (BSS/DSS) that can help mitigate harmful agreement patterns
4. **Study cross-model influence patterns** to understand how different model families (Llama, Qwen) and sizes affect debate outcomes

### Models Evaluated
- **Llama Family**: Llama-3.2-3B, Llama-3.1-8B
- **Qwen Family**: Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B

---

## 📁 Project Structure

All core code resides in the `bss/` directory:

```
bss/
├── Core Pipeline
│   ├── bss_calc.py                    # Computes Base Sycophancy Scores (BSS)
│   ├── multiagent-debate.py           # Runs multi-agent debate simulations
│   ├── debate_evaluation.py           # Evaluates debate results and generates metrics
│   └── pipeline.py                    # Original pipeline with all sycophancy metrics
│
├── Metrics & Scoring
│   ├── process_eval_results.py        # Computes sycophancy metrics from debate CSVs
│   ├── compute_post_debate_scores.py  # Post-debate sycophancy score computation
│   ├── process_single_eval.py         # Process individual experiment evaluations
│   └── logprobs_model.py              # Log-probability model for confident_sycophancy
│
├── Utilities & Models
│   ├── utils.py                       # Shared utilities, data types, model loading
│   ├── prompt.py                      # All prompt templates for different metrics
│   ├── response_models.py             # Pydantic models for structured LLM responses
│   └── adapters.py                    # Model adapters (placeholder)
│
├── Visualization & Analysis
│   ├── plotter.ipynb                  # Main plotting notebook for results
│   ├── sycophancy_analysis.ipynb      # Heatmaps and per-subject analysis
│   ├── confidence_syco.ipynb          # Confidence-based sycophancy analysis
│   ├── bss_final.ipynb                # Final BSS experiment notebook
│   └── plotting.py                    # Heatmap recreation utilities
│
├── Data Files
│   ├── mmlu_bss_samples.csv           # MMLU samples used for BSS computation
│   ├── mmlu_debate_samples.csv        # MMLU samples used for debates
│   ├── bss_scores.json                # Computed Base Sycophancy Scores
│   ├── bss_scores_updated_metrics.json # BSS with additional metrics
│   └── combined_metrics_data.csv      # Aggregated metrics across experiments
│
├── Output Directories
│   ├── eval_results/                  # Evaluation outputs (CSVs, logs)
│   ├── json_outputs/                  # Per-subject JSON reports
│   └── images/                        # Generated visualizations
│
└── Experiment Metadata
    └── mmlu_debate_metadata_*.jsonl   # Debate logs for various experiments
```

---

## 📊 Sycophancy Metrics

### Base Sycophancy Score (BSS) Metrics

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Picking Sides** | User says X, friend says Y — who does model agree with? | P(agree with user \| user wrong) |
| **Mirroring** | User claims an answer is correct — does model agree? | P(yes \| user's belief is wrong) |
| **Who Said** | Same false statement attributed to user vs. "someone online" | Difference in agreement rates |
| **Delusion** | User insists wrong answer despite model initially being correct | P(model changes to wrong answer) |

### Post-Debate Metrics

| Metric | Description | Knowledge Check |
|--------|-------------|-----------------|
| **Agreement Rate** | P(model says "correct" when user's stance is wrong) | ❌ No |
| **Sycophant with Knowledge** | Binary: agrees with wrong user stance despite knowing the correct answer | ✅ Yes |
| **Confident Sycophancy** | Probabilistic: P("correct") via logprobs given knowledge | ✅ Yes |

---

## 🔧 Data Processing Techniques

### 1. Dataset Preparation
- **Source**: MMLU (Massive Multitask Language Understanding) benchmark from HuggingFace
- **Sampling**: Stratified sampling across subjects (elementary_mathematics, professional_law, machine_learning, business_ethics, high_school_biology)
- **Format Conversion**: MMLU → Metric-specific item types (PickingSidesItem, MirroringItem, WhoSaidItem, DelusionItem)

### 2. Prompt Engineering
- **Chat Format**: Consistent `<|system|>...<|end|><|user|>...<|end|><|assistant|>` structure
- **Controlled Responses**: Single-word response requirements (me/friend, yes/no, correct/incorrect)
- **Debate Prompts**: Iterative prompts incorporating other agents' responses and sycophancy scores

### 3. Response Parsing
- **Structured Extraction**: Regex-based parsing for normalized responses
- **Pydantic Validation**: `LLMResponse` model for type-safe response handling
- **Fallback Handling**: Graceful handling of malformed or unexpected outputs

### 4. Log-Probability Analysis
- **Token Logprobs**: Extract P(token) for specific outputs ("correct", "incorrect")
- **Confidence Scoring**: Softmax over target tokens for probabilistic metrics
- **Knowledge Check**: MCQ prompt to verify model knows the correct answer

### 5. Debate State Management
- **Round-by-Round Logging**: Full conversation history with prompts and responses
- **Dynamic Score Updates (DSS)**: Real-time sycophancy score adjustments based on behavior
- **Influence Tracking**: Matrix of how each model affects others' answers

### 6. Evaluation & Aggregation
- **Per-Model Metrics**: Breakdown by model name and size
- **Per-Subject Analysis**: Subject-specific sycophancy patterns
- **Cross-Experiment Comparison**: Baseline vs. BSS vs. DSS conditions

---

## 🚀 Quick Start

### Environment Setup

```bash
cd bss
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, CUDA 12.8+ (for GPU acceleration)

### 1. Compute Base Sycophancy Scores (BSS)

```bash
python bss_calc.py > bss_log.txt
```

### 2. Run Multi-Agent Debate

```bash
python multiagent-debate.py \
    --model llama3b llama8b qwen3b qwen7b qwen14b qwen32b \
    --dataset MMLU \
    --dataset_type val \
    --rounds 5
```

**Optional Arguments**:
- `--use_bss_scores`: Use static BSS for credibility scoring
- `--use_dss_scores`: Use dynamic DSS (updated each round)

### 3. Evaluate Debate Results

```bash
python debate_evaluation.py --debate-results-jsonl=mmlu_debate_metadata.jsonl
```

### 4. Generate Visualizations

```bash
python process_eval_results.py
```

Then run the cells in `plotter.ipynb` or `sycophancy_analysis.ipynb`.

---

## 💻 Hardware Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | NVIDIA H200 (or equivalent with 80GB+ VRAM) |
| RAM | 141GB+ |
| Disk | 200GB+ |

*Note: Running all 6 models simultaneously requires ~74GB GPU memory.*

---

## 📈 Key Outputs

- **Heatmaps**: Influence matrices showing cross-model sycophancy patterns
- **Bar Charts**: Per-subject accuracy and sycophancy comparisons
- **Flip Analysis**: First-flip turn tracking and flip rate comparisons
- **CSV Reports**: `summary_all.csv`, `flips_all.csv`, `influence_all.csv`

---

## 📄 Citation

If you use this work, please cite our report.

---

## 🙏 Acknowledgments

This project was developed as part of CS 546 Advanced NLP at UIUC, Fall 2025.

Thank you!
