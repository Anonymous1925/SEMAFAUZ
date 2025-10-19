## About The Project

The code implements *SEMAFAUZ*. *SEMAFAUZ* is a semantics-aware fairness testing approach.
*SEMAFAUZ* automatically generates semantically valid fairness tests by synergistically combining mutation analysis and NLP methods. It performs semantics preserving input mutations to generate test suites that expose
individual fairness violations.

The approach uses a bias dictionary produced from [SBIC](https://paperswithcode.com/dataset/sbic) to generate a test suite containing bias mutations from texts.

The repository evaluates *SEMAFAUZ* using [LexGLUE](https://github.com/coastalcph/lex-glue) Benchmark **four** legal datasets (*Ecthr*, *Scotus*, *Eurlex*, *Ledgar*) and **four** LLM architectures (*BERT*, *Legal-BERT*, *RoBERTA*, *DeBERTA*) resulting in **16** models to evaluate **three** sensitive attributes (*race*, *gender*, *body*). In addition, we use Llama2 and GPT3.5 in our experiments, and IMDB, for a total of **18** models and **five** datasets.

Supplementary materials for the paper can be found [here](supplementary_materials.pdf).

Details of the performance of our fine-tuned models versus the original from [LexGLUE](https://github.com/coastalcph/lex-glue).


## User Study

We conducted a user study with 95 domain experts (lawyers and computer scientists) to evaluate the semantic validity of SEMAFAUZ-generated test inputs. The study included around 220 questions across grammar, semantics, and legal reasoning. Participants were asked to assess texts blindly and with source disclosure, rate similarity and correctness, and identify potential biases.
All materials related to the user study can be found in the [User Study/](User%20Study/) folder.

- The questionnaire presented to participants is available here:  
  [User Study on the Validity of Automatically Generated Legal Texts - General Questions](User%20Study/User%20Study%20on%20the%20Validity%20of%20Automatically%20Generated%20Legal%20Texts%20-%20General%20Questions.pdf)  
  This includes all grammar, semantic, and legal validity assessment questions, as well as validation and background checks.

- Quantitative responses and their aggregated results are available in:  
  [Public_user_study_results](User%20Study/Public_user_study_results.xlsx)

- Aggregated summaries of qualitative (open-ended) responses can be found in:  
  [Public_result_qualitative_survey](User%20Study/Public_result_qualitative_survey.xlsx)

> ⚠️ **Note:**  
> The full coding for individual qualitative responses is not included in this repository. This coding spans multiple files, and carefully anonymizing each would be error-prone. Any risk of deanonymization would violate the confidentiality requirements of our user study and compromise the double-blind review process.  
> To preserve participant anonymity and ensure compliance with ethical and publication guidelines, the coding artifacts will be released **after** the paper is accepted for publication.


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
- NVIDIA GPU with at least 24GB VRAM
- At least one non-hierarchical model trained from [LexGLUE](https://github.com/coastalcph/lex-glue) (all 16 combinations of BERT models/datasets cited above to reproduce everything).
- Access to the Llama model `meta-llama/Llama-2-7b-chat-hf` on Hugging Face. You may need to complete a form to request access from Meta [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
- Access to OpenAI's `gpt-3.5-turbo-0125` model. Ensure sufficient credits in your OpenAI account. You can manage credits and billing [here](https://platform.openai.com/settings/organization/billing/overview) and find specific billing details [here](https://platform.openai.com/docs/models/gpt-3-5#gpt-3-5-turbo).
- Tokens for accessing both models will be required during testing.

It is recommended to use **separate virtual environments** for BERT models and Llama/GPT models to avoid dependency conflicts. This ensures a smoother setup and operation for each model type.

## Usage

---

### BERT Models

#### Setup

- Ensure all dependencies in `requirements_bert_mutation.txt` are installed. It is best to use a dedicated virtual environment for BERT-related experiments.

- You will need CUDA 12 installed
#### Training

Train the BERT models using the datasets specified in [LexGLUE](https://github.com/coastalcph/lex-glue).

#### Testing

To test a BERT model, run the following script with the appropriate parameters:

```
python3 src/smart_replacement.py <model> <dataset_path> <set> [options]
```

E.g This code will run the generation of the mutants for ECTHR dataset, with bert-base-uncased's truncation size. And ignore Structural Similarity Check.
```
python3 src/smart_replacement.py ../models/ecthr_a/bert-base-uncased/seed_1/ lex_glue test --comment="checking_ablation" --checking_ablation --data_path="../data" --output_path="../output"
```

To reproduce the results of the paper, it is required to run this for each dataset, model and ablation setting.

##### Required Parameters

- `<model>`: Path to the model to test.  
  Must be one of:  `'bert-base-uncased'`, `'microsoft/deberta-base'`, `'roberta-base'`, `'nlpaueb/legal-bert-base-uncased'`

- `<dataset_path>`: Hugging Face path to the dataset.  
  Must be one of:  `"ecthr_a"`, `"scotus"`, `"ledgar"`, `"eurlex"`

- `<set>`: Dataset split to use: `"train"`, `"validation"`, or `"test"`.

##### Optional Parameters

- `--comment`: Short comment added to the results (default: `'default'`).  
  **⚠️ When using an ablation flag, the comment must exactly match the ablation name** for future processing steps to function correctly.

- `--length`: Maximum sequence length used for label truncation (default: `512`).

- `--depen_ablation`: Enable dependency-based ablation.
- `--single_ablation`: Enable single-unit ablation.
- `--checking_ablation`: Enable checking-related ablation.
- `--coref_ablation`: Enable only and only coreference, this flag will activate the other flags and use coreference alone.
- `--sememe_ablation`: Enable sememe-based ablation.

- `--mutation_only`: If set, only mutated inputs will be generated and no testing will be performed.

- `--output_path`: Path to the output folder where the mutants and results will be generated (default: `../output/`).
- `--data_path`: Path to the data folder containing the dictionnary (default: `../data/`).

##### Example

```bash
python3 src/smart_replacement.py ../models/ecthr_a/bert-base-uncased/seed_1/ lex_glue test --comment="checking_ablation" --checking_ablation --data_path="../data" --output_path="../output"
```

In this example:
- Mutants are generated for the `lex_glue` dataset using the `test` split.
- The model is loaded from the local path: `../models/ecthr_a/bert-base-uncased/seed_1/`.
- The model used is `bert-base-uncased`, and the input will be truncated using its default length (512 tokens).
- The dataset used is `ecthr_a`.
- The `checking_ablation` technique is activated, Structural Similarity Check is **ignored** (not used).
- The output will be saved in `../output`, and data will be read from `../data`.
- The `--comment` is explicitly set to `"checking_ablation"` to match the ablation type — this is **required** for downstream processing.

> ✅ **Important**: When using an ablation flag (e.g., `--checking_ablation`), the `--comment` must match its name exactly (e.g., `--comment="checking_ablation"`).


##### Output Files

The script will generate result files inside the directory specified by the `--output_path` parameter (default is `../output/`). The folder structure is as follows:

```
<output_path>/
└── <dataset>/
    └── <model>/
        └── <set>/
            ├── base_prediction.pkl
            ├── truncated_text.pkl
            └── error_details/
                └── <ablation>_mutants.pkl
```

##### Parameters:
- `<output_path>`: The root of the output folder, defined by the `--output_path` argument (default: `../output/`)
- `<dataset>`: The name of the dataset (e.g., `ecthr_a`)
- `<model>`: The model used (e.g., `bert-base-uncased`)
- `<set>`: The dataset split (`train`, `validation`, or `test`)
- `<ablation>`: The ablation method used (e.g., `checking_ablation`, `single_ablation`, etc.)

##### Description of Output Files

- `base_prediction.pkl`: The model’s predictions on the original, unmodified inputs.
- `truncated_text.pkl`: The input texts after truncation based on the `--length` parameter.
- `<ablation>_mutants.pkl`: File containing mutated inputs and predictions.

#### Run Every Bert Models

To run the full set of experiments for BERT models across all datasets and ablation settings, you can use the provided script:

```bash
./script_test_bert.sh <data_path> <output_path>
```

Where:
- `<data_path>` is the path to the data folder (e.g., `../data`)
- `<output_path>` is the destination folder for the output (e.g., `../output`)

> ⚠️ **Note:**  
> This script will launch a large number of experiments and can take **several days** to complete.



---

### LLAMA / GPT Models

#### Setup

- To run experiments using LLaMA and GPT models, it is recommended to use a **dedicated virtual environment** to avoid dependency conflicts with the BERT setup.

- Ensure the dependencies in `requirements_llms.txt` are installed.


#### Truncation Generation

Generate the truncated versions of the input cases of each datasets for both LLAMA/GPT using the following script:

```bash
python3 src/generate_llms_truncation.py [--data_path <path>] [--output_path <path>]
```

**Optional Parameters:**

- `--data_path`: Path to the folder containing the source data files to process.  
  Default is `./data/`.

- `--output_path`: Path to the folder where the truncated cases will be saved.  
  Default is `./output/`.

#### Testing Original Inputs on LLAMA / GPT

> ⚠️ **Warning: Running this script will trigger a large number of OpenAI API calls, and significant costs.**  

To evaluate the original inputs on LLaMA and GPT models, use the provided script:

```bash
./script_evaluate_llms.sh <LLAMA_TOKEN> <GPT_TOKEN>
```

Where:
- `<LLAMA_TOKEN>` is your Hugging Face access token for `meta-llama/Llama-2-7b-chat-hf`.
- `<GPT_TOKEN>` is your OpenAI API key for accessing `gpt-3.5-turbo-0125`.

This script runs `test_llms_originals.py` across all combinations of datasets and models.

#### Generating Mutants for LLAMA / GPT

To generate mutated inputs (fairness test cases) for LLaMA and GPT models across datasets, use the following script:

```bash
./src/script_mutants_llms.sh
```

This script launches the mutant generation pipeline across all combinations of datasets and models, and take several days.

#### Testing Mutants for LLAMA / GPT

> ⚠️ **Warning: Running this script will trigger a large number of OpenAI API calls, and significant costs.**  


To test all fairness mutants generated for LLaMA and GPT across all datasets and ablation settings:

```bash
./script_test_mutants_llms.sh <LLAMA_TOKEN> <GPT_TOKEN>
```

This script launches the mutants testing pipeline across all combinations of datasets and models, and take several days.

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
