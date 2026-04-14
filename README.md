# Code Summarization with `stable-code-3b` + QLoRA

Fine-tune [`stabilityai/stable-code-3b`](https://huggingface.co/stabilityai/stable-code-3b) for automatic code summarization using QLoRA. Supports training on Python (CodeSearchNet) and Java (FunCom) datasets, with full evaluation and mechanistic interpretability analysis.

> Designed and tested on **Kaggle** (single GPU, ~16GB VRAM). Should also run on Google Colab (A100/V100).

---

## Pretrained Weights

| Dataset | Link |
|---------|------|
| FunCom (Java) | [Download from Google Drive](https://drive.google.com/file/d/1BlZcjAdocLn4wk-3Jt90WjR24DAR0eXT/view?usp=sharing) |
| CodeSearchNet (Python) | [Download from Google Drive](https://drive.google.com/file/d/1RB-tAVqzdn94YEBJLOkITrQ00KS2dLal/view?usp=sharing) |

Download and place the weights in `./results_funcom/final_model/` before running evaluation cells. Similarly for CodeSearchNet.

To download directly in the notebook:
```python
# pip install gdown
import gdown
gdown.download(
    "https://drive.google.com/uc?id=1BlZcjAdocLn4wk-3Jt90WjR24DAR0eXT",
    "./results_funcom/final_model/adapter_model.safetensors",
    quiet=False
)
```

---

## Requirements

### Hardware
- GPU with **≥ 16GB VRAM** (e.g. Kaggle P100/T4, Colab A100)
- ~20GB disk space for model weights and checkpoints

### Installation

Run **Cell 1** of the notebook, or manually:

```bash
pip install -q --upgrade transformers>=4.40.0
pip install -q peft>=0.10.0 trl>=0.8.0
pip install -q bitsandbytes>=0.43.0
pip install -q datasets>=2.19.0 accelerate>=0.30.0
pip install -q evaluate rouge_score nltk sentencepiece protobuf
pip install -q bert_score sacrebleu
pip install -q matplotlib seaborn scikit-learn

# Required on Kaggle — removes triton which conflicts with eager attention
pip uninstall -y triton triton-nightly 2>/dev/null || true
```

> **Note:** The notebook uses `attn_implementation="eager"` to avoid `flash_attn`/`triton` dependency issues on Kaggle.

---

## Quickstart

Open `NLP_Project.ipynb` and run cells in order. Each cell has an estimated runtime:

| Cell | Description | Est. Time |
|------|-------------|-----------|
| 1 | Install dependencies | 3–5 min |
| 2 | Imports & setup | 30 sec |
| 3 | Load CodeSearchNet | 3–5 min |
| 4 | Load FunCom | 2–3 min |
| 5 | Tokenizer & preprocessing | 5–8 min |
| 6 | Load model with QLoRA | 3–5 min |
| 7 | Fine-tune on CodeSearchNet | 30–45 min |
| 8 | Plot CSN training curves | 10 sec |
| 9 | Fine-tune on FunCom | 35–50 min |
| 10 | Plot FunCom training curves | 10 sec |
| 11 | Evaluation utilities | 30 sec |
| 12 | Evaluate on CodeSearchNet | 10–15 min |
| 13 | Evaluate on FunCom | 10–15 min |
| 14 | Results table & comparison chart | 10 sec |
| 15 | Hyperparameter experiments | 60–90 min |
| 16 | Plot hyperparameter results | 10 sec |
| 17 | Attention pattern visualization | 5–8 min |
| 18 | Hidden state / PCA analysis | 3–5 min |
| 19 | Attention head specialization | 3–5 min |
| 20 | Gradient-based token attribution | 3–5 min |
| 21 | LoRA weight analysis | 2–3 min |
| 22 | Error analysis | 5 min |
| 23 | Generate summary report | 10 sec |

**Total runtime:** ~4–6 hours end-to-end on a single GPU.

---

## Configuration

All key constants are set in **Cell 2**:

```python
MODEL_NAME        = "stabilityai/stable-code-3b"
MAX_SEQ_LENGTH    = 512    # Max tokens per example
TRAIN_SUBSET_SIZE = 5000   # Training examples per dataset
VAL_SUBSET_SIZE   = 500    # Validation examples
TEST_SUBSET_SIZE  = 500    # Test examples
SEED              = 42
```

---

## Datasets

### CodeSearchNet — Python (Cell 3)
```python
csn_dataset = load_dataset("code_search_net", "python", trust_remote_code=True)
# Fields used: func_code_string, func_documentation_string
```

### FunCom — Java (Cell 4)
```python
funcom_raw = load_dataset("apcl/funcom-java-long")
# Fields used: source (code), summary
# Automatically falls back to CodeSearchNet Java if HF load fails
```

Both datasets are subsampled to `TRAIN_SUBSET_SIZE / VAL_SUBSET_SIZE / TEST_SUBSET_SIZE`.

---

## Prompt Format

Every example is formatted as:

```
Below is a code snippet. Write a concise summary of what the code does.

### Code:
{code}

### Summary:
{summary}
```

At inference time, `### Summary:` is left empty and the model completes it. Label masking ensures the training loss is computed **only on summary tokens**, not on the prompt.

---

## Model & QLoRA Setup (Cell 6)

The base model is loaded in **4-bit NF4** quantization and fine-tuned with LoRA adapters:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

This results in **~25M trainable parameters** (~0.89% of the 3B total).

---

## Training (Cells 7 & 9)

Both datasets use identical training arguments:

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    fp16=True,
    max_grad_norm=0.3,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    # + EarlyStoppingCallback(early_stopping_patience=3)
)
```

Checkpoints are saved to `./results_csn/` and `./results_funcom/` respectively. Final best models are saved under `{output_dir}/final_model/`.

---

## Evaluation (Cells 11–14)

Models are evaluated with **greedy decoding** (`do_sample=False`) on 100 test samples. Metrics:

- **BLEU** (SacreBLEU)
- **ROUGE-1, ROUGE-2, ROUGE-L**
- **METEOR**

To generate a summary for a single snippet:

```python
summary = generate_summary(model, code_text, max_new_tokens=128)
```

---

## Hyperparameter Experiments (Cell 15)

Six LoRA configurations are tested on a 1,000-sample CSN subset (1 epoch each):

| Config | LR | LoRA Rank | Dropout |
|--------|-----|-----------|---------|
| baseline | 2e-4 | 16 | 0.05 |
| lr=1e-4 | 1e-4 | 16 | 0.05 |
| lr=5e-4 | 5e-4 | 16 | 0.05 |
| rank=8 | 2e-4 | 8 | 0.05 |
| rank=32 | 2e-4 | 32 | 0.05 |
| dropout=0.0 | 2e-4 | 16 | 0.00 |
| dropout=0.1 | 2e-4 | 16 | 0.10 |

Results are plotted as validation loss per config and trainable parameter count vs. validation loss.

---

## Interpretability Analysis (Cells 17–21)

After training, five mechanistic analyses are run on the CodeSearchNet model:

| Cell | Analysis |
|------|----------|
| 17 | Attention heatmaps — layers L0, L8, L16, L24, L31 × heads H0, H16, H31 |
| 18 | PCA of hidden state trajectories + L2 norm growth across layers |
| 19 | Attention head specialization by token category (keywords, identifiers, operators, brackets, etc.) |
| 20 | Gradient-based token attribution via L2 norm of input embedding gradients |
| 21 | LoRA adapter weight magnitudes per layer |

---

## Output Files

| File | Cell | Description |
|------|------|-------------|
| `csn_training_curves.png` | 8 | Train/val loss — CodeSearchNet |
| `funcom_training_curves.png` | 10 | Train/val loss — FunCom |
| `combined_training_curves.png` | 14 | Overlaid loss curves for both datasets |
| `evaluation_comparison.png` | 14 | Bar chart of all evaluation metrics |
| `hyperparameter_experiments.png` | 16 | HP sweep results |
| `attention_patterns.png` | 17 | Attention heatmaps |
| `hidden_state_analysis.png` | 18 | PCA + L2 norm plots |
| `head_specialization.png` | 19 | Token-category attention by head |
| `token_attribution.png` | 20 | Gradient attribution scores |
| `lora_weight_analysis.png` | 21 | LoRA adapter weight magnitudes |
| `error_analysis.png` | 22 | BLEU distribution + length correlations |
| `./results_csn/final_model/` | 7 | Saved CodeSearchNet LoRA adapters |
| `./results_funcom/final_model/` | 9 | Saved FunCom LoRA adapters |

---

## Loading a Saved Model

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stable-code-3b",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

model = PeftModel.from_pretrained(base_model, "./results_csn/final_model")
model.eval()
```

---

## Known Issues

- **Triton conflict on Kaggle:** Cell 1 uninstalls `triton` and the model is loaded with `attn_implementation="eager"` throughout.
- **FunCom fallback:** If `apcl/funcom-java-long` fails to load, Cell 4 automatically falls back to `code_search_net` (Java subset).
- **Hyperparameter sweep interruption:** Cell 15 may time out on Kaggle — partial results are still captured and plotted.
- **Memory management:** Cells 9 and 12 explicitly call `del model`, `gc.collect()`, and `torch.cuda.empty_cache()` between runs to prevent OOM errors.

---

## References

1. Stability AI. [stable-code-3b](https://huggingface.co/stabilityai/stable-code-3b)
2. Husain et al. *CodeSearchNet Challenge.* arXiv:1909.09436, 2019.
3. LeClair et al. *FunCom: Java Method Summarization.* ICSE 2019.
4. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021.
5. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LMs.* NeurIPS 2023.
