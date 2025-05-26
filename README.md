# Neural Network Theory and Implementation Project WS 2023/2024

## Project Overview: Multilingual LLM Evaluation and Parameter Efficient Finetuning (PEFT)

This project explores the capabilities and limitations of Large Language Models (LLMs) in multilingual contexts through three main tasks:

### Task 1: Language Model Inference
- Evaluation of pre-trained LLMs (XGLM and GPT-2) on multilingual text
- Implementation using HuggingFace's transformers library
- Focus on model loading, tokenization, and inference
- Performance comparison across different languages

### Task 2: Hidden Representation Analysis
- Visualization and analysis of model's internal representations
- Implementation of dimensionality reduction techniques (PCA, t-SNE)
- Cross-lingual representation analysis
- Understanding model's language embeddings

### Task 3: Model Adaptation Techniques
- Implementation of efficient finetuning strategies:
  - LoRA (Low-Rank Adaptation)
  - iA3 (Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning)
- Comparative analysis of adaptation methods
- Performance evaluation across multiple languages

## Project Structure
```
llm-finetuning/
├── final_submission/
│   ├── Code/
│   │   ├── Task 1/        # Language model inference
│   │   ├── Task 2/        # Hidden representation analysis
│   │   └── Task 3/        # Model adaptation techniques
│   └── Report/            # Project documentation and results
├── notebooks/             # Development notebooks
├── scripts/              # Utility scripts and implementations
└── tasks/                # Task descriptions and requirements
```

## Getting Started

### Prerequisites
```bash
python>=3.8
torch>=1.10.0
transformers>=4.30.0
datasets>=2.12.0
numpy>=1.17
matplotlib>=3.3.0
scikit-learn>=0.24.0
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-finetuning.git
cd llm-finetuning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Task 1: Language Model Inference
```bash
cd final_submission/Code/Task\ 1
jupyter notebook "Task 1.ipynb"
```

### Task 2: Hidden Representation Analysis
```bash
cd final_submission/Code/Task\ 2
jupyter notebook task2_cheatVersion.ipynb
```

### Task 3: Model Adaptation
```bash
cd final_submission/Code/Task\ 3
jupyter notebook task3-vis.ipynb
```

## Languages Covered

The project evaluates models on a diverse set of languages including:
- English (eng_Latn)
- Spanish (spa_Latn)
- Italian (ita_Latn)
- German (deu_Latn)
- Arabic (arb_Arab)
- Telugu (tel_Telu)
- Tamil (tam_Taml)
- Quechua (quy_Latn)

## Models Used

- XGLM-564M: Primary multilingual model
- GPT-2: Comparison baseline
- Custom adapted versions using LoRA, iA3, and DoRA

## Evaluation Metrics

- Cross-entropy loss
- Hidden representation analysis
- Adaptation efficiency metrics

## Report Submission Guidelines




## License

This project is part of the Neural Network Theory and Implementation course at Saarland University. All rights reserved.

## Acknowledgments

- UdS NNTI Course instructors and teaching assistants
- HuggingFace team for the transformers library
- Facebook AI for the FLORES dataset and XGLM model