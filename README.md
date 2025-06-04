*Work in progress*

## Table of Contents

## About The Project

The code implements *SEMAFAUZ*. *SEMAFAUZ* is a semantics-aware fairness testing approach.
*SEMAFAUZ* automatically generates semantically valid fairness tests by synergistically combining mutation analysis and NLP methods. It performs semantics preserving input mutations to generate test suites that expose
individual fairness violations.

The approach uses a bias dictionary produced from [SBIC](https://paperswithcode.com/dataset/sbic) to generate a test suite containing bias mutations from texts.

The repository evaluates *SEMAFAUZ* using [LexGLUE](https://github.com/coastalcph/lex-glue) Benchmark **four** legal datasets (*Ecthr*, *Scotus*, *Eurlex*, *Ledgar*) and **four** LLM architectures (*BERT*, *Legal-BERT*, *RoBERTA*, *DeBERTA*) resulting in **16** models to evaluate **three** sensitive attributes (*race*, *gender*, *body*). In addition, we use Llama2 and GPT3.5 in our experiments, and IMDB, for a total of **18** models and **five** datasets.

Details of the performance of our fine-tuned models versus the original from [LexGLUE](https://github.com/coastalcph/lex-glue).

## Getting Started

Reproducing the testing takes weeks, and even the processing of the results requires time as it involves opening and processing a multitude of files.

This section explains how to install the necessary components and launch the experiments.

**Important:**

Due to a known issue with Python's hash seed, it is recommended to run all scripts with the following command to ensure consistency and reproducibility:

```
PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 python your_script.py
```

### Prerequisites

The experiments were tested on both Windows and Linux (Ubuntu 20.04).

- Python 3.8.
- At least one non-hierarchical model trained from [LexGLUE](https://github.com/coastalcph/lex-glue) (all 16 combinations of BERT models/datasets cited above to reproduce everything).
- Access to the Llama model `meta-llama/Llama-2-7b-chat-hf` on Hugging Face. You may need to complete a form to request access from Meta [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
- Access to OpenAI's `gpt-3.5-turbo-0125` model. Ensure sufficient credits in your OpenAI account. You can manage credits and billing [here](https://platform.openai.com/settings/organization/billing/overview) and find specific billing details [here](https://platform.openai.com/docs/models/gpt-3-5#gpt-3-5-turbo).
- Tokens for accessing both models will be required during testing.

It is recommended to use **separate virtual environments** for BERT models and Llama/GPT models to avoid dependency conflicts. This ensures a smoother setup and operation for each model type.

## Usage

### BERT Models

#### Setup

- Ensure all dependencies in `requirements_bert_mutation.txt` are installed. It is best to use a dedicated virtual environment for BERT-related experiments.

#### Training

Train the BERT models using the datasets specified in [LexGLUE](https://github.com/coastalcph/lex-glue).

#### Testing

To test a BERT model, simply launch `src/smart_replacement.py` with the appropriate parameters. Be sure to test a model for biases using the datasets it was trained with.

- `model`: Path to the model you want to test.
- `dataset_path`: Hugging Face path to the dataset.
- `set`: Dataset split to use: e.g., `"train"`, `"validation"`, or `"test"`.

## Useful Links

### Datasets
- [ECTHR Dataset](https://github.com/coastalcph/lex-glue#ecthr-a): Evaluating fairness in European Court of Human Rights cases.
- [EURLex Dataset](https://github.com/coastalcph/lex-glue#eurlex): Providing legal texts for language model evaluation.
- [LEDGAR Dataset](https://github.com/coastalcph/lex-glue#ledgar): Resources for legal contract analysis.
- [SCOTUS Dataset](https://case.law/): Supreme Court legal cases used in testing.
- [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews): Sentiment analysis dataset.
- [SBIC Dataset](https://paperswithcode.com/dataset/sbic): Source for the bias dictionary.

### Models
- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf): Hosted on Hugging Face, requires access approval.
- [gpt-3.5-turbo-0125](https://platform.openai.com/docs/models/gpt-3-5#gpt-3-5-turbo): Available via OpenAI, requires sufficient credits.
- [LexGLUE Models](https://github.com/coastalcph/lex-glue): Includes BERT, Legal-BERT, RoBERTA, and DeBERTA architectures.

### Other
- [LexGLUE Benchmark](https://github.com/coastalcph/lex-glue): Legal language benchmarks.
- [Best README Template](https://github.com/othneildrew/Best-README-Template/tree/master): Format inspiration for this README.

## Contact

Anonymous
