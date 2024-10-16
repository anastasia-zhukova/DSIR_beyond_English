# Data Selection for Language Models via Importance Resampling (DSIR) beyond English

The repository revisits [the DSIR paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b9aa8f418bde2840d5f4ab7a02f663b-Abstract-Conference.html) (see the [original code](https://github.com/p-lambda/dsir)) applied to collect text data given target data from the large multilingual text collection. The example below evaluates the methodology applied to German Literature domain. 


# DSIR German Literature
This repository contains two Jupyter notebooks for the project "Automated Collection of Multilingual Domain-Related Text Data for Continual Pretraining":

## Notebooks

1. **`dsir.ipynb`**
   - This notebook collects *n* domain-related text data from large raw datasets.

2. **`continual_pretraining.ipynb`**
   - This notebook performs continual pretraining on models with the collected text data.

     
## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.x
- `nltk` package
- `data_selection` package

You can install the required packages using pip:

```python
pip install nltk data_selection
```



## How to use DSIR

**`dsir.ipynb`** contains a Python script for selecting domain-related data using the `HashedNgramDSIR` class from the `data_selection` package. The script includes how to fit an importance estimator, compute importance weights, resample data, and save/load intermediate results.

### 1. **`raw_datasets`**

- **Purpose:** Contains the initial large raw data, e.g. the Pile, cc-100.
- **Format:** Must be converted to JSONL format. 

### 2. **`target_datasets`**

- **Purpose:** Contains domain-related data that you want to collect more from the large raw dataset, e.g. German literature.
- **Format:** Must be converted to JSONL format. 

**Conversion to JSONL:**

If your data is not in JSONL format, below is an example of how to convert txt to JSONL:

```python
import json

input_file = 'raw_data.txt'
output_file = 'raw_data.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        json_line = json.dumps({'text': line.strip()})
        outfile.write(json_line + '\n')
```
### 3. **`num_to_sample`**
- Specify how many lines of text data you want to extract from the raw dataset. 


## Continual Pretraining

- The notebook uses a collection of German texts as input for continual. The data collection can be found under this OneDrive folder: https://1drv.ms/f/c/376dc8dd0db77e08/Ek_edEfOgrNHu2MHtKZdpUMBpLEjn9mlqSwp5j1DR2Oe6g?e=lChvNx
- The tokenizer, **`BertTokenizer`**, is used to tokenize the dataset, and special tokens are handled automatically.
- The **`mask_tokens`** function applies the MLM objective by randomly masking 15% of the input tokens.
- **`TextDataset`** turns the input text into encodings and lables.
- The project conducted continual pretraining on pretrained gbert-large, thus loading the model using **`BertForMaskedLM.from_pretrained`**
```python
model = BertForMaskedLM.from_pretrained('deepset/gbert-large')
```
- The following code trains the model continually with extra data and the model can later be save to desired location.
```python
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
``` 
