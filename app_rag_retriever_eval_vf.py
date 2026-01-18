"""Retriever evaluation script for the RAG app - Final Version

This script:
1. Extracts nodes from the SAME ChromaDB index that will be evaluated
2. Generates evaluation questions for those nodes (preserving IDs)
3. Evaluates the retriever - ensuring ID matching
4. Compares: Vector-only vs Hybrid vs Hybrid+Reranker

Usage:
    python app_rag_retriever_eval_vf.py --force_regenerate --workers 1
    
Environment Variables:
    GOOGLE_API_KEY      - Required for Gemini LLM (question generation)
    OPENAI_API_KEY      - Required for embeddings
    COHERE_API_KEY      - Optional for reranker evaluation
    CHROMA_DB_PATH      - Optional, path to ChromaDB (default: ./data/forecast_expert_knowledge)
    CHROMA_COLLECTION   - Optional, collection name (default: forecast_expert_knowledge)
    EMBEDDING_MODEL     - Optional, embedding model (default: text-embedding-3-small)
"""

import os
import argparse
import asyncio
import time
from typing import List, Any

import pandas as pd
from dotenv import load_dotenv
from llama_index.core.evaluation import generate_question_context_pairs, RetrieverEvaluator
from llama_index.llms.google_genai import GoogleGenAI
import google.genai.types as types
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.llama_dataset.legacy.embedding import EmbeddingQAFinetuneDataset
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

# Import HybridRetriever and reranker from the main app
from app_rag_forecast_vf import (
    HybridRetriever,
    get_cohere_reranker,
    retrieve_all_nodes_from_vector_index
)
from llama_index.core.retrievers import KeywordTableSimpleRetriever
from llama_index.core import SimpleKeywordTableIndex

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION (can be overridden by environment variables)
# =============================================================================

# Default paths - relative to script location for portability
DEFAULT_CHROMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "data2", 
    "forecast_expert_knowledge"
)

# Configuration from environment variables with defaults
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", DEFAULT_CHROMA_PATH)
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "forecast_expert_knowledge")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# =============================================================================
# CONFIGURE EMBEDDING MODEL (must match the one used to create the index)
# =============================================================================
Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

# =============================================================================
# API KEY DIAGNOSTICS
# =============================================================================
def check_api_keys():
    """Check and display status of required API keys."""
    print("\n" + "="*60)
    print("[KEY] CHECKING API KEYS")
    print("="*60)
    
    # Check Google API Key
    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        print(f"GOOGLE_API_KEY: [OK] Found ({google_key[:8]}...{google_key[-4:]})")
    else:
        print("GOOGLE_API_KEY: [X] Not found")
    
    # Check Cohere API Key - try multiple sources
    cohere_key = os.environ.get("COHERE_API_KEY")
    
    # If not found in environment, try to load from Windows User environment
    if not cohere_key:
        try:
            import subprocess
            result = subprocess.run(
                ['powershell', '-Command', '[System.Environment]::GetEnvironmentVariable("COHERE_API_KEY", "User")'],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                cohere_key = result.stdout.strip()
                os.environ["COHERE_API_KEY"] = cohere_key
                print(f"COHERE_API_KEY: [OK] Loaded from Windows User env ({cohere_key[:8]}...{cohere_key[-4:]})")
        except Exception as e:
            print(f"COHERE_API_KEY: [!] Could not load from Windows env: {e}")
    
    if cohere_key and "COHERE_API_KEY" not in os.environ:
        os.environ["COHERE_API_KEY"] = cohere_key
    
    if cohere_key:
        if "Loaded from Windows" not in str(locals().get('_', '')):
            print(f"COHERE_API_KEY: [OK] Found ({cohere_key[:8]}...{cohere_key[-4:]})")
    else:
        print("COHERE_API_KEY: [X] Not found")
        print("\n[!] Reranker evaluation will be SKIPPED without COHERE_API_KEY")
        print("   To enable reranking, set the key:")
        print("   PowerShell: $env:COHERE_API_KEY = 'your_key_here'")
    
    print("="*60 + "\n")
    
    return {
        "google": bool(google_key),
        "cohere": bool(cohere_key)
    }


def extract_nodes_from_chroma(chroma_collection, max_retries=10) -> List[TextNode]:
    """
    Extract all documents from ChromaDB and convert them to TextNode objects.
    This ensures the node IDs match what the retriever will return.
    """
    print("Extracting nodes from ChromaDB collection...")
    
    for attempt in range(1, max_retries + 1):
        try:
            # Get all documents from the collection
            results = chroma_collection.get(include=["documents", "metadatas"])
            break
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to extract data after {max_retries} attempts: {e}")
            wait_time = min(2 ** attempt, 60)
            print(f"  Attempt {attempt} failed: {e}; retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    nodes = []
    ids = results["ids"]
    documents = results["documents"]
    metadatas = results["metadatas"] or [{}] * len(ids)
    
    for doc_id, doc_text, metadata in zip(ids, documents, metadatas):
        if doc_text:  # Skip empty documents
            # Create TextNode with the EXACT same ID as in ChromaDB
            node = TextNode(
                id_=doc_id,  # Critical: Use the same ID!
                text=doc_text,
                metadata=metadata or {}
            )
            nodes.append(node)
    
    print(f"✓ Extracted {len(nodes)} nodes from ChromaDB")
    return nodes


def generate_eval_dataset_from_nodes(
    nodes: List[TextNode],
    output_path: str,
    llm,
    num_questions_per_chunk: int = 1
) -> EmbeddingQAFinetuneDataset:
    """
    Generate evaluation dataset from nodes, preserving their IDs.
    """
    print(f"Generating evaluation questions for {len(nodes)} nodes using Gemini...")
    print("This may take several minutes...")
    
    # generate_question_context_pairs preserves node IDs
    rag_eval_dataset = generate_question_context_pairs(
        nodes, 
        llm=llm, 
        num_questions_per_chunk=num_questions_per_chunk
    )
    
    print(f"✓ Generated {len(rag_eval_dataset.queries)} questions")
    print(f"Saving evaluation dataset to {output_path}...")
    rag_eval_dataset.save_json(output_path)
    
    return rag_eval_dataset


def display_results_retriever(name: str, eval_results: List[Any]) -> pd.DataFrame:
    """Display results from evaluate and return a simple DataFrame."""
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = getattr(eval_result, "metric_vals_dict", None) or eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)
    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
    )
    return metric_df


def main(output_path, num_questions_per_chunk=1, workers=1, force_regenerate=False):
    # Check API keys first (this will also load COHERE_API_KEY from Windows if needed)
    api_keys = check_api_keys()
    
    # Get Google API key
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Check if reranker evaluation is possible
    cohere_available = api_keys.get("cohere", False)
    if cohere_available:
        print("[OK] Cohere Reranker evaluation ENABLED")
    else:
        print("[!] Cohere Reranker evaluation DISABLED (no API key)")

    # ChromaDB path - use configurable path from environment or default
    vector_db_path = CHROMA_DB_PATH
    
    # Validate that ChromaDB exists
    if not os.path.exists(vector_db_path):
        raise FileNotFoundError(
            f"ChromaDB path not found: {vector_db_path}\n"
            f"Please either:\n"
            f"  1. Run app_rag_forecast_vf.py first to create the knowledge base\n"
            f"  2. Set CHROMA_DB_PATH environment variable to your ChromaDB location\n"
            f"  3. Copy your existing ChromaDB to: {DEFAULT_CHROMA_PATH}"
        )
    
    print(f"Loading ChromaDB from {vector_db_path}...")
    db = chromadb.PersistentClient(path=vector_db_path)
    chroma_collection = db.get_collection(CHROMA_COLLECTION_NAME)
    print(f"✓ Collection loaded with {chroma_collection.count()} documents")
    
    # Create index for retrieval
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    # Get current ChromaDB IDs first
    print("\n" + "="*60)
    print("CHECKING CHROMADB IDS")
    print("="*60)
    
    chroma_data = chroma_collection.get()
    chroma_ids = set(chroma_data["ids"])
    print(f"ChromaDB contains {len(chroma_ids)} documents")
    print(f"  Sample IDs: {list(chroma_ids)[:3]}")
    
    # Helper function to check if dataset IDs match ChromaDB
    def ids_match(dataset, chroma_ids_set):
        if dataset is None:
            return False
        dataset_ids = set(dataset.corpus.keys())
        common = dataset_ids.intersection(chroma_ids_set)
        return len(common) == len(dataset_ids) and len(common) == len(chroma_ids_set)
    
    # Load existing dataset if available
    rag_eval_dataset = None
    needs_regeneration = force_regenerate
    
    if os.path.exists(output_path) and not force_regenerate:
        print(f"\nLoading existing evaluation dataset from {output_path}...")
        rag_eval_dataset = EmbeddingQAFinetuneDataset.from_json(output_path)
        print(f"✓ Loaded {len(rag_eval_dataset.queries)} queries")
        
        # Verify IDs match
        dataset_ids = set(rag_eval_dataset.corpus.keys())
        common_ids = dataset_ids.intersection(chroma_ids)
        
        print(f"\n[CHECK] ID VERIFICATION:")
        print(f"   Dataset IDs: {len(dataset_ids)}")
        print(f"   ChromaDB IDs: {len(chroma_ids)}")
        print(f"   IDs in common: {len(common_ids)}")
        
        if len(common_ids) == 0:
            print("\n[!] NO IDs match! Dataset is stale and needs regeneration.")
            needs_regeneration = True
        elif len(common_ids) < len(dataset_ids) or len(common_ids) < len(chroma_ids):
            print(f"\n[!] Only {len(common_ids)}/{max(len(dataset_ids), len(chroma_ids))} IDs match.")
            print("   Dataset is out of sync and needs regeneration.")
            needs_regeneration = True
        else:
            print("\n[OK] All IDs match! Dataset is valid.")
    else:
        if not os.path.exists(output_path):
            print(f"\nNo existing dataset found at {output_path}")
        needs_regeneration = True
    
    # Regenerate dataset if needed
    if needs_regeneration:
        print("\n" + "="*60)
        print("[REGEN] REGENERATING EVALUATION DATASET FROM CHROMADB NODES")
        print("="*60)
        
        # Extract nodes FROM ChromaDB (this ensures IDs match!)
        nodes = extract_nodes_from_chroma(chroma_collection)
        
        # Setup Gemini LLM
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=512,
            temperature=0.3,
        )
        llm = GoogleGenAI(model="gemini-2.5-flash", generation_config=config)
        
        # Generate dataset
        rag_eval_dataset = generate_eval_dataset_from_nodes(
            nodes=nodes,
            output_path=output_path,
            llm=llm,
            num_questions_per_chunk=num_questions_per_chunk
        )
        
        # Verify the new dataset
        dataset_ids = set(rag_eval_dataset.corpus.keys())
        common_ids = dataset_ids.intersection(chroma_ids)
        print(f"\n[OK] New dataset generated with {len(common_ids)}/{len(chroma_ids)} matching IDs")
    
    print("="*60)
    
    # ==========================================================================
    # SETUP RETRIEVERS FOR COMPARISON
    # ==========================================================================
    
    print("\n" + "="*60)
    print("SETTING UP RETRIEVERS FOR COMPARISON")
    print("="*60)
    
    # Extract all nodes for keyword index (using imported function)
    print("\nExtracting nodes for keyword search...")
    all_nodes = retrieve_all_nodes_from_vector_index(index)
    print(f"✓ Extracted {len(all_nodes)} nodes")
    
    # Create keyword index
    print("Building keyword index...")
    keyword_index = SimpleKeywordTableIndex(nodes=all_nodes)
    keyword_retriever = KeywordTableSimpleRetriever(keyword_index)
    print("✓ Keyword index built")
    
    # Initialize Cohere reranker only if API key is available
    reranker = None
    if cohere_available:
        print("\nInitializing Cohere reranker...")
        reranker = get_cohere_reranker(top_n=10)
        if reranker:
            print("[OK] Cohere reranker initialized successfully")
        else:
            print("[!] Cohere reranker initialization failed")
    else:
        print("\n[!] Skipping Cohere reranker (no COHERE_API_KEY)")
    
    # ==========================================================================
    # EVALUATE ALL THREE RETRIEVER CONFIGURATIONS
    # ==========================================================================
    
    print("\n" + "="*60)
    print("RUNNING RETRIEVER EVALUATION")
    print("="*60)
    print("\n[EVAL] Comparing: Vector-only vs Hybrid vs Hybrid+Reranker")
    
    results_all = []
    
    for k in [2, 4, 6, 8, 10]:
        print(f"\n{'='*60}")
        print(f"TOP_K = {k}")
        print("="*60)
        
        # ---------------------------------------------------------------------
        # 1. Vector-only retriever (baseline)
        # ---------------------------------------------------------------------
        print(f"\n  [1/3] Vector-only (top_k={k})...")
        vector_retriever = index.as_retriever(similarity_top_k=k)
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], 
            retriever=vector_retriever
        )
        eval_results = asyncio.run(
            retriever_evaluator.aevaluate_dataset(rag_eval_dataset, workers=workers)
        )
        df_vector = display_results_retriever(f"Vector-only (k={k})", eval_results)
        print(f"        Hit Rate: {df_vector['Hit Rate'].values[0]:.4f} | MRR: {df_vector['MRR'].values[0]:.4f}")
        
        results_all.append({
            "top_k": k,
            "retriever_type": "vector_only",
            "hit_rate": df_vector["Hit Rate"].values[0],
            "mrr": df_vector["MRR"].values[0],
        })
        
        # ---------------------------------------------------------------------
        # 2. Hybrid retriever (vector + keyword, NO reranker)
        # ---------------------------------------------------------------------
        print(f"\n  [2/3] Hybrid (top_k={k}, no reranker)...")
        hybrid_retriever_no_rerank = HybridRetriever(
            vector_retriever=index.as_retriever(similarity_top_k=k*2),  # Get more candidates
            keyword_retriever=keyword_retriever,
            max_retrieve=k,
            reranker=None  # No reranking
        )
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], 
            retriever=hybrid_retriever_no_rerank
        )
        eval_results = asyncio.run(
            retriever_evaluator.aevaluate_dataset(rag_eval_dataset, workers=workers)
        )
        df_hybrid = display_results_retriever(f"Hybrid (k={k})", eval_results)
        print(f"        Hit Rate: {df_hybrid['Hit Rate'].values[0]:.4f} | MRR: {df_hybrid['MRR'].values[0]:.4f}")
        
        results_all.append({
            "top_k": k,
            "retriever_type": "hybrid",
            "hit_rate": df_hybrid["Hit Rate"].values[0],
            "mrr": df_hybrid["MRR"].values[0],
        })
        
        # ---------------------------------------------------------------------
        # 3. Hybrid + Reranker (if reranker is available)
        # ---------------------------------------------------------------------
        if reranker is not None:
            print(f"\n  [3/3] Hybrid + Reranker (top_k={k})...")
            # Create reranker with correct top_n
            reranker_k = get_cohere_reranker(top_n=k)
            
            hybrid_retriever_with_rerank = HybridRetriever(
                vector_retriever=index.as_retriever(similarity_top_k=20),  # Get many candidates
                keyword_retriever=keyword_retriever,
                max_retrieve=20,  # Let reranker do the final selection
                reranker=reranker_k
            )
            retriever_evaluator = RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"], 
                retriever=hybrid_retriever_with_rerank
            )
            eval_results = asyncio.run(
                retriever_evaluator.aevaluate_dataset(rag_eval_dataset, workers=workers)
            )
            df_rerank = display_results_retriever(f"Hybrid+Reranker (k={k})", eval_results)
            print(f"        Hit Rate: {df_rerank['Hit Rate'].values[0]:.4f} | MRR: {df_rerank['MRR'].values[0]:.4f}")
            
            results_all.append({
                "top_k": k,
                "retriever_type": "hybrid_reranker",
                "hit_rate": df_rerank["Hit Rate"].values[0],
                "mrr": df_rerank["MRR"].values[0],
            })
        else:
            print(f"\n  [3/3] Hybrid + Reranker - SKIPPED (no COHERE_API_KEY)")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(results_all)
    
    # Pivot tables for better readability
    print("\n[RESULTS] HIT RATE COMPARISON:")
    print("-"*60)
    pivot_hr = results_df.pivot(index='top_k', columns='retriever_type', values='hit_rate')
    # Reorder columns for clarity
    cols_order = ['vector_only', 'hybrid', 'hybrid_reranker']
    cols_present = [c for c in cols_order if c in pivot_hr.columns]
    pivot_hr = pivot_hr[cols_present]
    print(pivot_hr.to_string())
    
    print("\n[RESULTS] MRR COMPARISON:")
    print("-"*60)
    pivot_mrr = results_df.pivot(index='top_k', columns='retriever_type', values='mrr')
    pivot_mrr = pivot_mrr[cols_present]
    print(pivot_mrr.to_string())
    
    # Calculate improvements
    print("\n[IMPROVE] IMPROVEMENTS (vs Vector-only baseline):")
    print("-"*60)
    
    for k in [2, 4, 6, 8, 10]:
        k_results = results_df[results_df['top_k'] == k]
        vector_hr = k_results[k_results['retriever_type'] == 'vector_only']['hit_rate'].values[0]
        hybrid_hr = k_results[k_results['retriever_type'] == 'hybrid']['hit_rate'].values[0]
        
        hybrid_improvement = ((hybrid_hr - vector_hr) / vector_hr) * 100 if vector_hr > 0 else 0
        
        line = f"k={k:2d}: Hybrid vs Vector = {hybrid_improvement:+.1f}%"
        
        if 'hybrid_reranker' in k_results['retriever_type'].values:
            rerank_hr = k_results[k_results['retriever_type'] == 'hybrid_reranker']['hit_rate'].values[0]
            rerank_improvement = ((rerank_hr - vector_hr) / vector_hr) * 100 if vector_hr > 0 else 0
            line += f" | Hybrid+Reranker vs Vector = {rerank_improvement:+.1f}%"
        
        print(line)
    
    # Save results
    results_csv = output_path.replace('.json', '_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to {results_csv}")
    print(f"\n✓ Results saved to {results_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Retriever Evaluation Script (v4 - Fixed ID matching)")
    parser.add_argument("--output", default="./rag_eval_dataset_v4.json", help="Output dataset JSON path")
    parser.add_argument("--num_questions", type=int, default=1, help="Questions per chunk")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (keep at 1 to avoid HNSW issues)")
    parser.add_argument("--force_regenerate", action="store_true", help="Force regeneration of evaluation dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        output_path=args.output,
        num_questions_per_chunk=args.num_questions,
        workers=args.workers,
        force_regenerate=args.force_regenerate
    )
