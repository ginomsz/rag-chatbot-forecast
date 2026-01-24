---
title: Rag Forecast Chatbot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: false	
python_version: 3.12.3
---

# 📈 RAG Chatbot Forecast

This is a Retrieval-Augmented Generation (RAG) chatbot advisor specialized in **Forecasting and Time Series** topics for business and technical questions. 
Built with LlamaIndex, ChromaDB, and OpenAI.

### Key Features

- **Domain-Specific Knowledge Base**: Built from public forecasting literature (Papers, articles, and online resources). It can include several full books but are not included in this deployment to avoid copyright issues.
- **Hybrid Search**: Combines vector similarity + keyword search using Reciprocal Rank Fusion (RRF)
- **Cohere Reranker**: Cross-encoder reranking for improved retrieval relevance
- **Dynamic Few-Shot Prompting**: Automatically selects the best examples based on user query
- **Evaluation Suite**: Includes dataset and scripts to measure retrieval performance (Hit Rate, MRR)

## Optional Functionalities used in this project
1. Uses dynamic few-shot prompting, where the best examples are selected according to the user query.
2. There’s code for RAG evaluation in the folder, and the README contains the evaluation results. The folder must also contain the evaluation dataset and the evaluation scripts.
3. The app is designed for a specific goal/domain that is not a tutor about AI. For example, it could be about finance, healthcare, etc.
4. You have shown evidence of collecting at least two data sources beyond those provided in our course.
5. Use a reranker in your RAG pipeline. It can be a fine-tuned version (your choice).
6. Use hybrid search in your RAG pipeline.

## 🚀 Quick Start

```bash
# 1. Clone and enter directory
git clone https://github.com/yourusername/rag-forecast-chatbot.git
cd rag-forecast-chatbot

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Add your PDF documents to data/ folder

# 6. Run the chatbot
python app.py
```

Open `http://localhost:7860` in your browser.

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `GOOGLE_API_KEY` | Optional | For evaluation script |
| `COHERE_API_KEY` | Optional | For reranking |

## 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Main chatbot application |
| `app_rag_retriever_eval_vf.py` | Retriever evaluation script |
| `rag_eval_dataset_v0.json`      | Evaluation dataset (it will be regenerated with the evaluation script)  
|`rag_eval_dataset_v0_results.csv`| Evaluation results (it will be updated with the evaluation script)  

## (Optional) RAG Forecast Chatbot Evaluation

### 🧪 How to Run the Evaluation
1. **Prepare the Environment**
	 - Ensure your ChromaDB knowledge base is built (run the chatbot app at least once).
	 - Set your API keys in the environment:
		 - `OPENAI_API_KEY` (required)
		 - `GOOGLE_API_KEY` (required for question generation)
		 - `COHERE_API_KEY` (optional, for reranker evaluation)

2. **Run the Evaluation Script**
	 ```bash
	 python app_rag_retriever_eval_vf.py --force_regenerate --workers 1
	 ```
	 - `--force_regenerate`: (optional) Forces regeneration of the evaluation dataset.
	 - `--workers 1`: Use 1 worker to avoid HNSW issues.

3. **What Happens**
	 - The script extracts all nodes from your ChromaDB collection.
	 - It generates evaluation questions for each node using Gemini (Google LLM).
	 - It evaluates three retriever types:
		 - Vector-only
		 - Hybrid (vector + keyword)
		 - Hybrid + Reranker (if Cohere API key is set)
	 - It computes metrics: **Hit Rate** and **MRR** for each retriever and top_k value.

4. **Datasets and Results**
	 - The evaluation dataset is saved as:
		 - `rag_eval_dataset_v4.json`
	 - The results (metrics for each retriever and top_k) are saved as:
		 - `rag_eval_dataset_v4_results.csv`
	 - A summary of results is printed in the terminal and can be found in the README.

## Evaluation results

[RESULTS] HIT RATE COMPARISON:
------------------------------------------------------------
| retriever_type | vector_only | hybrid   | hybrid_reranker |
|:--------------:|:-----------:|:--------:|:---------------:|
| top_k = 2      | 0.565333    | 0.528000 | 0.805333        |
| top_k = 4      | 0.637333    | 0.629333 | 0.805333        |
| top_k = 6      | 0.693333    | 0.674667 | 0.805333        |
| top_k = 8      | 0.722667    | 0.701333 | 0.805333        |
| top_k = 10     | 0.741333    | 0.728000 | 0.805333        |

[RESULTS] MRR COMPARISON:
------------------------------------------------------------
| retriever_type | vector_only | hybrid   | hybrid_reranker |
|:--------------:|:-----------:|:--------:|:---------------:|
| top_k = 2      | 0.504000    | 0.485333 | 0.539062        |
| top_k = 4      | 0.525556    | 0.516889 | 0.539062        |
| top_k = 6      | 0.536044    | 0.526844 | 0.539062        |
| top_k = 8      | 0.540044    | 0.528283 | 0.539062        |
| top_k = 10     | 0.542059    | 0.533764 | 0.539062        |

[IMPROVE] IMPROVEMENTS (vs Vector-only baseline):
------------------------------------------------------------
| k  | Hybrid vs Vector | Hybrid+Reranker vs Vector |
|----|------------------|--------------------------|
| 2  | -6.6%            | +42.5%                   |
| 4  | -1.3%            | +26.4%                   |
| 6  | -2.7%            | +16.2%                   |
| 8  | -3.0%            | +11.4%                   |
| 10 | -1.8%            | +8.6%                    |

✓ Results saved to ./rag_eval_dataset_v4_results.csv

## 💲 Total Estimated Cost per Query

| Component      | Tokens | Price per 1M tokens | Cost per query |
|---------------|--------|---------------------|----------------|
| Embedding     | 500    | $0.02               | $0.00001       |
| LLM Input     | 500    | $0.15               | $0.000075      |
| LLM Output    | 500    | $0.60               | $0.0003        |
| **Total**     |        |                     | **$0.000385**  |

*Assumes 500 tokens for embedding, LLM input, and LLM output per query. Adjust values as needed for your use case.*
