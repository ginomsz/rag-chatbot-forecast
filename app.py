# =============================================================================
# IMPORTS - Import necessary libraries
# =============================================================================

"""RAG Forecast Chatbot - Final Version

A Retrieval-Augmented Generation chatbot specialized in forecasting knowledge.
Uses ChromaDB for vector storage, OpenAI for embeddings/LLM, and Cohere for reranking.

Usage:
    python app_rag_forecast_vf.py                    # Run full application

Environment Variables:
    OPENAI_API_KEY      - Required for LLM and embeddings
    COHERE_API_KEY      - Optional for reranker (improves retrieval quality)
    DATA_DIR            - Optional, PDF source directory (default: ./data)
    CHROMA_DB_PATH      - Optional, ChromaDB path (default: ./data/forecast_expert_knowledge)
    EMBEDDING_MODEL     - Optional, embedding model (default: text-embedding-3-small)
    LLM_MODEL           - Optional, LLM model (default: gpt-4o-mini)
"""

# Standard Library Imports
import logging  # To configure and handle system logs
import os  # For operating system operations (files, directories)
import glob  # To search files using specific patterns
import csv  # To handle CSV files
import asyncio  # For async operations
from datetime import datetime  # To handle dates and timestamps

# Third-party Imports
from dotenv import load_dotenv  # To load environment variables from a .env file
import chromadb  # Vector database to store embeddings
import gradio as gr  # Framework to create interactive web interfaces
import numpy as np  # For numerical operations and cosine similarity
import nest_asyncio  # To allow nested async event loops

# Website Crawling Imports
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode  # Web crawler for extracting website content

# LlamaIndex (formerly GPT Index) Imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext  # Core LlamaIndex components
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever, BaseRetriever  # To retrieve similar documents
from llama_index.core import SimpleKeywordTableIndex, QueryBundle
from llama_index.core.schema import NodeWithScore, Document  # Document class for web content
from llama_index.core import SimpleKeywordTableIndex, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore  # ChromaDB integration
from llama_index.core.llms import MessageRole  # Message roles in conversations
from llama_index.core.memory import ChatSummaryMemoryBuffer  # Conversation memory buffer
from llama_index.core.tools import RetrieverTool, ToolMetadata  # Tools for the agent
from llama_index.agent.openai import OpenAIAgent  # OpenAI agent
from llama_index.embeddings.openai import OpenAIEmbedding  # OpenAI embeddings model
from llama_index.llms.openai import OpenAI  # OpenAI language model
from llama_index.core import Settings  # Global LlamaIndex settings
from llama_index.postprocessor.cohere_rerank import CohereRerank  # Cohere reranker for improved relevance

# Load environment variables from a .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)  # Create a module-specific logger
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce httpx verbosity

# =============================================================================
# CONFIGURATION (can be overridden by environment variables)
# =============================================================================

# Base directory - relative to script location for portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths configuration
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
DATA_DIR2 = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data2"))
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", os.path.join(DATA_DIR2, "forecast_expert_knowledge"))
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "forecast_expert_knowledge")

# Model configuration
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# =============================================================================
# FEW-SHOT EXAMPLES LIBRARY
# =============================================================================

def get_few_shot_examples():
    """
    Returns a library of few-shot examples covering Forecasting concepts,
    models, and implementation across 3 technical levels.
    Each example is a dict with: User, Assistant
    """
    examples = [
        # Business-Oriented Examples
        {
            "User": "What is demand forecasting and why is it important for my business?",
            "Assistant": "Demand forecasting is an estimation of future sales for a product or service. It's crucial because it helps optimize inventory, reduce storage costs, improve customer satisfaction by avoiding stockouts, and plan production efficiently. In practical terms, a good forecast can reduce operational costs by 10-30% and increase product availability."
        },
        {
            "User": "How many months of historical data do I need to make a reliable forecast?",
            "Assistant": "Generally, you should have at least 12-24 months of historical data to capture seasonal patterns. However, this depends on the type of business: products with high seasonality (like toys at Christmas) require more data, while stable products can work with 6-12 months. The key is having enough data to identify trends and seasonality."
        },
        {
            "User": "What's the difference between short-term and long-term forecasting?",
            "Assistant": "Short-term forecasts (1-3 months) focus on daily operations: inventory management, weekly production planning. They're more accurate but require frequent updates. Long-term forecasts (6-24 months) are used for strategic planning: annual budgets, capacity expansion, investment decisions. They're less accurate but provide strategic direction."
        },
        {
            "User": "How can I measure if my forecast is good or bad?",
            "Assistant": "Key metrics are: MAPE (Mean Absolute Percentage Error) - ideally less than 20% for retail; Bias - should be close to 0 to avoid systematic over/under-estimation; and Accuracy - percentage of times the forecast is within an acceptable range. It's also important to measure business impact: reduction in dead inventory, improvement in product availability, and cost optimization."
        },
        {
            "User": "What external factors should I consider when making forecasts?",
            "Assistant": "Key factors include: seasonality (Christmas, summer, annual events), promotions and marketing campaigns, economic trends, competition, changes in consumer behavior, and unique events (pandemic, natural disasters). It's important to maintain a record of these events and adjust forecasts when significant changes occur in the environment."
        },
        {
            "User": "What is the typical cost of implementing a forecasting system?",
            "Assistant": "Costs vary by complexity: basic solutions with Excel/simple software can cost $5,000-$20,000 annually. Intermediate systems with tools like Tableau or Power BI: $20,000-$100,000. Advanced solutions with custom ML: $100,000-$500,000+ including development, infrastructure, and maintenance. Typical ROI is 3-5x in 12-18 months through inventory reduction and availability improvement."
        },
        {
            "User": "How can I improve my forecast accuracy without hiring experts?",
            "Assistant": "Practical steps: 1) Collect complete and clean historical data, 2) Identify and document special events (promotions, seasons), 3) Use accessible tools like Excel with forecast functions or cloud software like Forecast Pro, 4) Compare multiple simple methods (moving average, exponential smoothing), 5) Review and adjust monthly based on past errors, 6) Involve the sales team to capture tacit knowledge."
        },
        {
            "User": "What is the typical accuracy I should expect from a forecasting system?",
            "Assistant": "Accuracy expectations vary by industry and product type. For retail: MAPE of 15-25% is good, 10-15% is excellent. For stable products (staples, utilities): 5-10% MAPE is achievable. For fashion/trendy items: 30-40% MAPE may be acceptable. The key is comparing against baseline (naive forecast) - if you're 20-30% better than baseline, that's meaningful improvement. Focus on business impact, not just numbers."
        },
        {
            "User": "How often should I update my forecasts?",
            "Assistant": "Update frequency depends on business needs: High-frequency (daily/weekly) for fast-moving items, perishables, or when external factors change rapidly. Medium-frequency (monthly) for most retail products - balances accuracy with operational efficiency. Low-frequency (quarterly) for stable products or strategic planning. Rule of thumb: update when new data would meaningfully change decisions. Most businesses find monthly updates optimal."
        },
        {
            "User": "What are the main benefits of implementing a forecasting system?",
            "Assistant": "Key benefits: 1) Inventory optimization - reduce excess stock by 20-40%, 2) Stockout reduction - improve service levels by 5-15%, 3) Cost savings - reduce holding costs, markdowns, and expedited shipping, 4) Better planning - align production, procurement, and staffing with expected demand, 5) Competitive advantage - respond faster to market changes. Typical ROI: 3-5x within first year through inventory reduction alone."
        },
        {
            "User": "How do I know if my business needs a forecasting system?",
            "Assistant": "You need forecasting if: 1) You have inventory management challenges (overstock or stockouts), 2) Demand varies significantly over time, 3) You make purchasing/production decisions in advance, 4) You have seasonal patterns or trends, 5) You're scaling operations and need better planning. Signs you need it: frequent stockouts, high inventory write-offs, difficulty planning for promotions, or manual forecasting taking too much time."
        },
        {
            "User": "What is the difference between forecasting and planning?",
            "Assistant": "Forecasting predicts what will likely happen based on historical data and patterns. Planning uses forecasts plus business goals, constraints, and strategies to decide what should happen. Forecasting answers 'What will demand be?' Planning answers 'How should we respond?' Good planning requires accurate forecasts, but also considers capacity, budgets, marketing strategies, and business objectives. Both are essential for effective operations."
        },
        {
            "User": "Can I use forecasting for new products with no historical data?",
            "Assistant": "Yes, but with different approaches: 1) Use similar products as proxy - find products with similar characteristics, 2) Market research and expert judgment - combine industry data with internal knowledge, 3) Analogous forecasting - use launch patterns from similar products, 4) Start conservative - begin with lower forecasts and adjust based on early sales, 5) Use qualitative methods - Delphi method, sales force composite. As data accumulates, transition to quantitative methods."
        },
        {
            "User": "What is a baseline forecast and why is it important?",
            "Assistant": "A baseline forecast is a simple, naive method used as a comparison point (e.g., 'next period equals last period' or 'next period equals same period last year'). It's important because: 1) It sets a minimum performance bar - your model should beat it, 2) It helps measure improvement - shows value of sophisticated methods, 3) It's a fallback - if complex models fail, baseline still works, 4) It's interpretable - stakeholders understand simple methods. Always compare advanced models against baseline."
        },
        {
            "User": "How do promotions and marketing campaigns affect forecasts?",
            "Assistant": "Promotions significantly impact demand: typically increase sales by 20-200% during the event, but may cause pre/post promotion dips. To handle: 1) Document all promotions with dates, discount levels, and channels, 2) Create separate forecasts for promotional vs regular periods, 3) Use historical promotion lift data to adjust baseline forecasts, 4) Consider cannibalization effects on other products, 5) Account for post-promotion hangover (reduced demand). Many businesses create promotion calendars and adjust forecasts accordingly."
        },
        {
            "User": "What is forecast bias and how does it impact my business?",
            "Assistant": "Forecast bias is systematic over- or under-forecasting. Over-forecasting leads to excess inventory, markdowns, and holding costs. Under-forecasting causes stockouts, lost sales, and customer dissatisfaction. To measure: calculate average error - positive means over-forecasting, negative means under-forecasting. Ideal bias is near zero. Causes: optimistic/pessimistic assumptions, ignoring trends, or not accounting for external factors. Regular bias tracking helps identify and correct systematic errors."
        },
        
        # Intermediate Examples
        {
            "User": "When should I use ARIMA versus exponential smoothing methods?",
            "Assistant": "Use ARIMA when: you have data with complex trends and seasonality, need to explicitly model autocorrelation, and have sufficient data (minimum 50-100 points). Use exponential smoothing (ETS) when: you seek simplicity and interpretability, have clear seasonal patterns, or need quick forecasts with less data. ARIMA is more flexible but requires more parameters; ETS is more robust with less data. For most business cases, ETS is a good starting point."
        },
        {
            "User": "How do I implement temporal cross-validation to evaluate my forecast model?",
            "Assistant": "Use rolling-origin validation: 1) Split data into training (70-80%) and test (20-30%), 2) Train model on initial window, forecast next period, 3) Add that period to training, re-train, forecast next, 4) Repeat until covering entire test period, 5) Calculate metrics (MAE, RMSE, MAPE) on all predictions. This simulates real conditions where the model is updated periodically. Use Python with scikit-learn TimeSeriesSplit or implement manually with pandas."
        },
        {
            "User": "What is MAPE and when should I not use it?",
            "Assistant": "MAPE (Mean Absolute Percentage Error) measures average error as a percentage: MAPE = (1/n) * Σ|actual - forecast|/actual * 100. Advantages: easy to interpret, scale-independent. Limitations: doesn't work well with values close to zero (division by zero), penalizes errors more in small values, can be asymmetric. Alternatives: sMAPE (symmetric MAPE), wMAPE (weighted MAPE), or MAE/RMSE for small values. Use MAPE when all values are significantly greater than zero."
        },
        {
            "User": "How do I handle missing data in time series for forecasting?",
            "Assistant": "Strategies by case: 1) Linear interpolation for small gaps (<5% of data), 2) Forward fill or backward fill if pattern is stable, 3) Imputation with moving average of neighboring values, 4) Models that handle missing values natively (Prophet, some deep learning models), 5) If many missing (>20%), consider aggregating to weekly/monthly level. Always document the method used and evaluate impact on accuracy. In Python: pandas.interpolate(), fillna(), or scikit-learn SimpleImputer."
        },
        {
            "User": "What's the difference between Prophet and ARIMA for forecasting?",
            "Assistant": "Prophet (Facebook) is easier to use, automatically handles seasonality, holidays, and outliers. Ideal for data with clear patterns, multiple seasonalities (daily, weekly, annual), and when you need quick results without extensive tuning. ARIMA requires more statistical knowledge, manual parameter tuning (p,d,q), but offers more control and can be more accurate with clean data and simple patterns. Use Prophet for business cases with complex seasonality; ARIMA for deep statistical analysis or when you need parameter interpretability."
        },
        {
            "User": "How do I implement external features (promotions, events) in my forecast model?",
            "Assistant": "Step 1: Create dummy/binary variables for events (1 if promotion, 0 if not). Step 2: Add continuous variables (discount %, marketing spend). Step 3: Use models that support external regressors: SARIMAX (ARIMA with regressors), Prophet (add_regressor), XGBoost/LightGBM (features as columns), or deep learning models (LSTM with features). Step 4: Validate that features are significant (p-value < 0.05). Step 5: Monitor feature impact in production. Python example: statsmodels SARIMAX or Prophet with add_regressor()."
        },
        {
            "User": "What is overfitting in forecasting and how do I avoid it?",
            "Assistant": "Overfitting occurs when the model fits too closely to historical data and fails on new data. Signs: very low training error but high validation error, model with many parameters. Prevention: 1) Use temporal cross-validation (not standard k-fold), 2) Limit model complexity (fewer parameters in ARIMA), 3) Use regularization (L1/L2 in ML models), 4) Compare simple vs complex models, 5) Keep separate test set. Practical rule: if adding complexity doesn't improve validation, use simpler model. Prefer interpretable models (ETS, ARIMA) over deep learning unless you have lots of data."
        },
        {
            "User": "How do I choose between univariate and multivariate forecasting models?",
            "Assistant": "Use univariate models (ARIMA, ETS, Prophet) when: you only have historical demand data, relationships with other variables are unclear, or you need simplicity. Use multivariate models (VAR, VECM, ML with features) when: you have strong leading indicators (e.g., marketing spend, economic indicators), you need to model relationships between multiple time series, or external factors significantly impact demand. Start univariate, add variables if they improve accuracy. Multivariate requires more data and careful feature selection."
        },
        {
            "User": "What is the difference between point forecasts and prediction intervals?",
            "Assistant": "Point forecasts give a single number (e.g., 'demand will be 100 units'). Prediction intervals provide a range with confidence (e.g., 'demand will be 85-115 units with 95% confidence'). Point forecasts are simpler but don't show uncertainty. Prediction intervals help with: inventory safety stock calculation, risk assessment, and decision-making under uncertainty. Most business decisions benefit from intervals - use them for critical items or when uncertainty is high. Calculate using bootstrap, simulation, or analytical methods."
        },
        {
            "User": "How do I handle seasonality in my forecast model?",
            "Assistant": "Seasonality handling: 1) Identify seasonality - use autocorrelation plots, decomposition, or domain knowledge, 2) Choose method - additive (constant variation) vs multiplicative (proportional variation), 3) Model options: SARIMA (seasonal ARIMA), Prophet (automatic seasonality), ETS with seasonal components, or Fourier terms, 4) Validate - check if seasonal patterns are consistent across years, 5) Adjust for changing seasonality - some models adapt automatically. Test multiple seasonal periods (weekly, monthly, yearly) if patterns exist at different scales."
        },
        {
            "User": "What is the difference between MAE, RMSE, and MAPE metrics?",
            "Assistant": "MAE (Mean Absolute Error): average absolute difference, easy to interpret, treats all errors equally. RMSE (Root Mean Squared Error): penalizes large errors more, sensitive to outliers, in same units as data. MAPE (Mean Absolute Percentage Error): percentage-based, scale-independent, but problematic with values near zero. Use MAE for interpretability, RMSE when large errors are costly, MAPE for comparing across different scales (but avoid with small values). Often report multiple metrics for comprehensive evaluation."
        },
        {
            "User": "How do I implement hierarchical forecasting for product families?",
            "Assistant": "Hierarchical forecasting handles multiple levels (e.g., total → category → product). Approaches: 1) Bottom-up - forecast each product, sum to higher levels, 2) Top-down - forecast total, allocate to products, 3) Middle-out - forecast middle level, allocate up and down, 4) Optimal reconciliation - use statistical methods to ensure consistency. Bottom-up preserves product-level patterns but may not sum correctly. Top-down ensures consistency but loses product detail. Optimal reconciliation balances both. Use hierarchical methods when you need forecasts at multiple levels simultaneously."
        },
        {
            "User": "What is exponential smoothing and when should I use it?",
            "Assistant": "Exponential smoothing gives more weight to recent observations, with weights decreasing exponentially for older data. Types: Simple (no trend/seasonality), Holt's (adds trend), Holt-Winters (adds seasonality). Use when: data shows trends or seasonality, you need interpretable models, you have limited data (works with 20+ points), or you need quick forecasts. Advantages: simple, robust, handles missing values. Disadvantages: assumes patterns are stable. Good default choice for many business forecasting problems."
        },
        {
            "User": "How do I detect and handle outliers in time series data?",
            "Assistant": "Outlier detection: 1) Statistical methods - Z-scores, IQR method, or time series specific (e.g., STL decomposition residuals), 2) Domain knowledge - identify known events (promotions, disasters), 3) Visualization - time plots, box plots, residual analysis. Handling: 1) Remove if clearly erroneous, 2) Cap/floor extreme values, 3) Model as separate component (Prophet handles automatically), 4) Use robust methods (median-based, robust regression), 5) Create indicator variables. Don't remove without understanding cause - outliers may contain important information about rare events."
        },
        {
            "User": "What is the difference between stationary and non-stationary time series?",
            "Assistant": "Stationary series have constant mean and variance over time. Non-stationary series have trends, changing variance, or structural breaks. Most economic/business data is non-stationary. Why it matters: many models (ARIMA) assume stationarity. Solutions: 1) Differencing - take differences to remove trends, 2) Transformation - log to stabilize variance, 3) Detrending - remove trend component, 4) Use models that handle non-stationarity (Prophet, some ML models). Test stationarity with ADF test or visual inspection. Most forecasting models handle non-stationarity through differencing or transformations."
        },
        {
            "User": "How do I implement demand sensing for short-term forecasts?",
            "Assistant": "Demand sensing uses recent signals (point-of-sale data, web traffic, social media) to adjust short-term forecasts. Implementation: 1) Collect high-frequency data (daily/hourly), 2) Identify leading indicators (sales velocity, search trends), 3) Use short-term models (moving averages, exponential smoothing with low alpha), 4) Combine with baseline forecast (weighted average), 5) Update frequently (daily or real-time). Benefits: captures sudden changes, improves 1-7 day forecasts. Limitations: requires data infrastructure, may overreact to noise. Best for fast-moving, high-variability products."
        },
        {
            "User": "What is the difference between deterministic and probabilistic forecasts?",
            "Assistant": "Deterministic forecasts give single values (point forecasts). Probabilistic forecasts provide distributions (prediction intervals, quantiles, full distributions). Deterministic: simpler, easier to communicate, but ignores uncertainty. Probabilistic: richer information, enables risk-based decisions, but more complex. Use deterministic for operational planning, simple reporting. Use probabilistic for inventory optimization (safety stock), risk management, or when uncertainty matters. Most modern forecasting systems can generate both - choose based on decision needs."
        },
        {
            "User": "How do I evaluate forecast accuracy across different products or SKUs?",
            "Assistant": "Multi-SKU evaluation: 1) Aggregate metrics - average MAPE/MAE across all SKUs, 2) Weighted metrics - weight by volume/value (wMAPE), 3) Distribution analysis - percentiles of accuracy (median, 75th percentile), 4) Segment analysis - group by characteristics (fast/slow moving, high/low variability), 5) Hit rate - percentage within acceptable range. Avoid simple averages if SKUs vary greatly - use weighted or segmented analysis. Focus improvement efforts on high-value or problematic segments."
        },
        {
            "User": "What is the difference between forecast accuracy and forecast value?",
            "Assistant": "Forecast accuracy measures how close forecasts are to actuals (MAE, MAPE, etc.). Forecast value measures business impact (inventory reduction, stockout reduction, cost savings). A model can be accurate but not valuable if it doesn't change decisions, or less accurate but more valuable if it improves critical decisions. Always measure both: track accuracy metrics AND business KPIs (inventory levels, service levels, costs). Value depends on how forecasts are used in decision-making, not just statistical accuracy."
        },
        {
            "User": "How do I handle intermittent demand (many zeros) in forecasting?",
            "Assistant": "Intermittent demand (sparse, irregular) requires special methods: 1) Croston's method - separate forecast of demand size and interval, 2) TSB (Teunter-Syntetos-Babai) - improved version of Croston's, 3) Aggregate to higher level - forecast at category level, allocate down, 4) Use count models - Poisson, negative binomial for demand frequency, 5) Machine learning - classification for 'will demand occur?' plus regression for 'how much?'. Standard methods (ARIMA, ETS) often fail with intermittent demand. Test Croston's or TSB first."
        },
        {
            "User": "What is the difference between forecast horizon and forecast frequency?",
            "Assistant": "Forecast horizon is how far ahead you predict (e.g., 1 week, 1 month, 1 year). Forecast frequency is how often you update forecasts (e.g., daily, weekly, monthly). They're related but different: short horizons (1-7 days) need frequent updates, long horizons (1 year) may need less frequent updates. Choose horizon based on decision lead time. Choose frequency based on data availability and how quickly patterns change. Many systems use rolling horizons - forecast multiple periods ahead, update regularly."
        },
        
        # Technical/Expert Examples
        {
            "User": "How do I implement an N-BEATS model for multi-horizon forecasting with attention to architecture?",
            "Assistant": "N-BEATS (Neural Basis Expansion Analysis) uses stacked blocks with backcast/forecast. Implementation: 1) Architecture: stacked blocks (4-8), each block with FC layers (128-512 units), basis expansion (polynomial or trigonometric), 2) Training: loss function (MAE or MAPE), Adam optimizer with learning rate 1e-3, early stopping, 3) Multi-horizon: final block output generates multiple horizons simultaneously, 4) Regularization: dropout (0.1-0.3), weight decay, 5) Features: normalize inputs, add temporal features (day of week, month). Code: PyTorch or use darts library (NBEATSModel). Key hyperparameters: num_stacks, num_blocks, num_layers, layer_widths. Validate with walk-forward validation."
        },
        {
            "User": "What's the difference between data drift vs concept drift in forecast models, and how do I monitor them?",
            "Assistant": "Data drift: change in input feature distribution (e.g., change in sales volume). Concept drift: change in input-output relationship (e.g., same volume but different demand). Monitoring: 1) Statistical tests: KS test, PSI (Population Stability Index) for data drift, 2) Model performance: track MAE/RMSE in moving windows, alert if degradation >20%, 3) Residual analysis: Q-Q plots, Ljung-Box test for autocorrelation, 4) Feature importance shifts: compare feature importance between periods, 5) Implementation: Python with evidently.ai, alibi-detect, or custom with scipy.stats. Thresholds: retrain if PSI > 0.25 or error increases >30% for 3 consecutive periods."
        },
        {
            "User": "How do I optimize hyperparameters for SARIMAX using Bayesian methods vs grid search?",
            "Assistant": "Grid search: exhaustive but slow, good for small spaces (<100 combinations). Bayesian optimization (Optuna, Hyperopt): more efficient, learns from previous evaluations. For SARIMAX: 1) Search space: p,d,q in [0-3], P,D,Q,s in [0-2, s=12 for monthly], 2) Metric: AICc (corrected for small samples) or BIC, 3) Validation: time series cross-validation, 4) Optuna implementation: define search space, objective function that trains SARIMAX and returns metric, 5) Pruning: eliminate promising trials early. Example: optuna.create_study(direction='minimize'), study.optimize(objective, n_trials=100). Bayesian is 3-5x faster than grid search and finds better parameters."
        },
        {
            "User": "How do I implement a forecast ensemble combining statistical and ML models, and what combination method should I use?",
            "Assistant": "Ensemble improves robustness. Steps: 1) Train diverse models: ARIMA, ETS, Prophet, XGBoost, LSTM, 2) Generate individual forecasts, 3) Combination: a) Simple average (baseline), b) Weighted average (weights by historical accuracy), c) Stacking (meta-model learns to combine), d) Bayesian Model Averaging (BMA) with probabilistic weights, 4) Implementation: calculate weights as inverse of error (w_i = 1/MAE_i normalized), or train regressor (Ridge/Lasso) on predictions as features, 5) Validation: evaluate ensemble vs best individual models. Python: statsmodels for statistical, sklearn for stacking. Typical improvement: 5-15% error reduction vs best individual model."
        },
        {
            "User": "How do I implement advanced feature engineering for forecast models with lag features, rolling statistics, and Fourier transformations?",
            "Assistant": "Key feature engineering: 1) Lag features: demand t-1, t-7, t-30 (captures autocorrelation), 2) Rolling statistics: 7/30-day moving average, moving std, min/max (captures local trends), 3) Fourier features: sin/cos with seasonal frequencies (day of year, week of year) to capture cycles, 4) Target encoding: historical average by category (product, region), 5) Interactions: lag * seasonality, 6) Transformations: log, box-cox to stabilize variance. Python implementation: pandas shift(), rolling(), numpy for Fourier. Validate importance with feature_importances_ (XGBoost) or coefficients (regression). Typically 10-30 well-designed features outperform 100+ generic features."
        },
        {
            "User": "How do I design an MLOps system for forecasts in production with automatic retraining, A/B testing, and rollback?",
            "Assistant": "MLOps architecture: 1) Pipeline: Airflow/Prefect for orchestration, 2) Retraining: automatic trigger (weekly/monthly) or by drift detection, 3) Validation: automatic tests (accuracy > threshold, latency < SLA), 4) A/B testing: deploy new model to % of traffic, compare metrics (MAE, business KPIs) vs baseline, 5) Rollback: maintain previous versions, revert if degradation >20%, 6) Monitoring: Prometheus/Grafana for metrics, alerts on degradation, 7) Feature store: Feast/Tecton for consistent features, 8) Model registry: MLflow for versioning, 9) CI/CD: GitHub Actions for tests, automatic deployment. Stack: Python (models) + Docker + Kubernetes + MLflow + monitoring. Typical retraining time: 1-4 hours depending on data size."
        },
        {
            "User": "How do I implement Temporal Fusion Transformer (TFT) for multi-horizon forecasting with static and dynamic features?",
            "Assistant": "TFT handles multiple input types: static (product category), known future (holidays), observed past (demand). Implementation: 1) Architecture: encoder-decoder with temporal attention, variable selection networks, 2) Features: encode static (embedding), known future (time features), observed (lag features, rolling stats), 3) Training: quantile loss for uncertainty, teacher forcing, 4) Hyperparameters: hidden size (64-256), attention heads (4-8), dropout (0.1-0.3), 5) Use PyTorch Forecasting library or implement from paper. Advantages: handles mixed data types, provides quantile forecasts, interpretable attention. Requires substantial data (1000+ samples) and tuning."
        },
        {
            "User": "How do I implement quantile regression for probabilistic forecasting?",
            "Assistant": "Quantile regression predicts specific percentiles (e.g., 10th, 50th, 90th) instead of just mean. Implementation: 1) Loss function - pinball/quantile loss for each quantile, 2) Models - linear quantile regression, quantile random forests, quantile gradient boosting (LightGBM, XGBoost), or neural networks with quantile outputs, 3) Training - train separate models per quantile or single model with multiple outputs, 4) Validation - check quantile coverage (actual % within predicted intervals), 5) Use libraries: scikit-learn, statsmodels, or PyTorch. Applications: inventory optimization (safety stock from 95th percentile), risk management, robust planning."
        },
        {
            "User": "How do I implement online learning for forecasts that adapt in real-time?",
            "Assistant": "Online learning updates models incrementally as new data arrives. Approaches: 1) Incremental algorithms - exponential moving average, online gradient descent, 2) Adaptive models - ARIMA with recursive parameter updates, online Prophet, 3) Ensemble methods - update weights based on recent performance, 4) Concept drift detection - monitor performance, retrain when degradation detected, 5) Implementation: use River library (Python) or custom incremental updates. Trade-offs: adapts quickly to changes but may overreact to noise. Use for high-frequency data, rapidly changing patterns, or when batch retraining is expensive."
        },
        {
            "User": "How do I implement transfer learning for forecasting across similar products or regions?",
            "Assistant": "Transfer learning uses knowledge from source domains (products/regions with data) to improve forecasts for target domains (limited data). Methods: 1) Feature transfer - use shared feature representations, 2) Model transfer - pre-train on source, fine-tune on target, 3) Meta-learning - learn to learn (MAML, Model-Agnostic Meta-Learning), 4) Hierarchical - leverage group-level patterns, 5) Implementation: PyTorch for neural approaches, or statistical pooling. Use when: new products/regions have little data but similar products/regions have rich data. Can improve accuracy by 10-30% for low-data scenarios."
        },
        {
            "User": "How do I implement causal inference methods to measure promotion and marketing impact on forecasts?",
            "Assistant": "Causal methods isolate true causal effects from correlations. Approaches: 1) Difference-in-differences - compare treated vs control groups, 2) Regression discontinuity - exploit threshold rules, 3) Instrumental variables - use instruments correlated with treatment but not outcome, 4) Synthetic control - construct counterfactual from similar units, 5) Causal forests - machine learning for heterogeneous effects. Implementation: use DoWhy, EconML libraries. Applications: measure true promotion lift (not just correlation), adjust forecasts for marketing actions, optimize marketing spend. More rigorous than simple correlation analysis."
        },
        {
            "User": "How do I implement state-space models (Kalman filters) for forecasting with uncertainty?",
            "Assistant": "State-space models decompose series into unobserved components (level, trend, seasonality) with uncertainty. Kalman filter provides optimal estimates and prediction intervals. Implementation: 1) Define state space - observation and state equations, 2) Initialize - prior mean and covariance, 3) Filter - update estimates as data arrives, 4) Smooth - backward pass for full-sample estimates, 5) Forecast - extrapolate state with uncertainty. Use statsmodels (SARIMAX uses state space), PyKalman, or custom. Advantages: handles missing data, provides uncertainty, adapts to changes. Good for: irregular data, structural breaks, or when uncertainty quantification is critical."
        },
        {
            "User": "How do I implement automated model selection and hyperparameter tuning for time series?",
            "Assistant": "Automated selection: 1) Model candidates - ARIMA, ETS, Prophet, ML models, 2) Search strategy - grid search (small spaces), random search, Bayesian optimization (Optuna, Hyperopt), 3) Validation - time series cross-validation (walk-forward), 4) Selection criteria - AIC/BIC for statistical, cross-validation error for ML, 5) AutoML tools - AutoTS, PyCaret, or custom pipeline. Implementation: use Optuna for Bayesian optimization, scikit-learn for ML tuning, statsmodels auto_arima. Best practices: validate on multiple metrics, use temporal CV, avoid data leakage, test on holdout set. Can automate entire forecasting pipeline."
        },
        {
            "User": "How do I implement distributed forecasting for thousands of SKUs efficiently?",
            "Assistant": "Distributed forecasting handles large-scale problems: 1) Parallelization - forecast multiple SKUs simultaneously (multiprocessing, Dask, Spark), 2) Model selection - use same model class for similar SKUs, cluster SKUs by patterns, 3) Hierarchical methods - forecast at aggregate level, disaggregate, 4) Incremental updates - only retrain changed SKUs, 5) Infrastructure - use cloud (AWS, GCP) with auto-scaling, batch processing. Implementation: Dask for parallel pandas, Spark for very large scale, or cloud ML services. Optimize: cache features, use efficient algorithms, batch API calls. Can reduce computation time from days to hours."
        },
        {
            "User": "How do I implement explainable AI (XAI) for forecast model interpretability?",
            "Assistant": "XAI methods explain model predictions: 1) Feature importance - SHAP values, permutation importance, 2) Partial dependence plots - show feature effects, 3) Attention mechanisms - for neural models (TFT, Transformer), 4) Local explanations - LIME for individual predictions, 5) Model-agnostic - use post-hoc methods on any model. Implementation: SHAP library, LIME, or built-in methods (XGBoost feature_importances_). Applications: understand drivers, debug models, build trust, regulatory compliance. Balance: simpler models (ARIMA) are inherently interpretable, complex models (deep learning) need XAI methods."
        },
        {
            "User": "How do I implement counterfactual forecasting to answer 'what-if' scenarios?",
            "Assistant": "Counterfactual forecasting predicts under alternative scenarios (e.g., 'what if we increase price 10%?'). Methods: 1) Causal models - structural models with identified causal effects, 2) Scenario analysis - run models with different inputs, 3) Simulation - Monte Carlo with different assumptions, 4) Intervention analysis - measure impact of past interventions, extrapolate. Implementation: use DoWhy for causal inference, or run standard models with modified inputs. Applications: pricing decisions, promotion planning, capacity planning. Requires: understanding of causal relationships, not just correlations. Validate with historical interventions when possible."
        },
        {
            "User": "How do I implement federated learning for forecasting across multiple business units without sharing data?",
            "Assistant": "Federated learning trains models across decentralized data without centralizing it. Steps: 1) Local training - each unit trains on own data, 2) Aggregation - central server combines model updates (FedAvg), 3) Distribution - send aggregated model back to units, 4) Iterate - repeat until convergence. Implementation: PySyft, TensorFlow Federated, or custom. Use cases: multiple subsidiaries, privacy-sensitive data, regulatory constraints. Challenges: non-IID data, communication costs, heterogeneity. Benefits: privacy preservation, leverages all data without sharing, can improve accuracy through diversity."
        },
        {
            "User": "How do I implement reinforcement learning for dynamic pricing and demand forecasting?",
            "Assistant": "RL learns optimal policies (pricing, inventory) through interaction. Setup: 1) State - current inventory, demand, market conditions, 2) Actions - price levels, order quantities, 3) Reward - profit, revenue, or business objectives, 4) Algorithm - Q-learning, policy gradient, or actor-critic, 5) Environment - simulation or real A/B testing. Implementation: use OpenAI Gym for environment, stable-baselines3 for algorithms. Applications: dynamic pricing, inventory optimization, promotion timing. Advantages: learns optimal policies, adapts to feedback. Challenges: requires exploration, reward design, simulation accuracy. Often combined with forecasting for state prediction."
        },
        {
            "User": "How do I implement graph neural networks for forecasting products with relationships?",
            "Assistant": "GNNs model relationships between products (substitutes, complements, categories). Architecture: 1) Graph construction - nodes (products), edges (relationships), 2) Message passing - aggregate neighbor information, 3) Temporal component - combine with time series models (LSTM, Transformer), 4) Training - predict demand using both product features and graph structure. Implementation: PyTorch Geometric, DGL libraries. Use cases: retail with product relationships, supply chains, multi-product portfolios. Captures: substitution effects, category trends, cross-product influences. Requires: graph structure definition, more complex than standard time series models."
        },
        {
            "User": "How do I implement conformal prediction for forecast uncertainty with guaranteed coverage?",
            "Assistant": "Conformal prediction provides prediction intervals with guaranteed coverage (e.g., 90% of actuals within 90% intervals). Methods: 1) Split conformal - use calibration set to adjust intervals, 2) Cross-conformal - use cross-validation, 3) Adaptive - adjust for heteroscedasticity. Implementation: 1) Train model, get residuals on calibration set, 2) Calculate quantile of residuals, 3) Add to predictions for intervals. Libraries: MAPIE (Model Agnostic Prediction Interval Estimation). Advantages: distribution-free, finite-sample guarantees, works with any model. Applications: risk management, inventory safety stock, robust planning. More reliable than parametric uncertainty estimates."
        }
    ]
    return examples


# =============================================================================
# DYNAMIC FEW-SHOT PROMPTING FUNCTIONS
# =============================================================================

def compute_cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def get_top_similar_examples(user_query, examples, embed_model, top_k=12):
    """
    Find the top K most similar examples to the user query using cosine similarity.
    
    Args:
        user_query: The user's query string
        examples: List of example dictionaries with 'User' key
        embed_model: Embedding model to convert text to vectors
        top_k: Number of top examples to return (default: 12)
    
    Returns:
        List of top K example dictionaries sorted by similarity
    """
    try:
        # Get embedding for user query
        query_embedding = np.array(embed_model.get_query_embedding(user_query))
        
        # Compute similarities
        similarities = []
        for example in examples:
            # Get embedding for example question
            example_embedding = np.array(embed_model.get_query_embedding(example["User"]))
            similarity = compute_cosine_similarity(query_embedding, example_embedding)
            similarities.append((similarity, example))
        
        # Sort by similarity (descending) and get top K
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_examples = [example for _, example in similarities[:top_k]]
        
        logging.info(f"Selected {len(top_examples)} few-shot examples with similarities: {[f'{sim:.3f}' for sim, _ in similarities[:top_k]]}")
        
        return top_examples
    
    except Exception as e:
        logging.error(f"Error in get_top_similar_examples: {str(e)}")
        # Fallback: return first top_k examples
        return examples[:top_k]


def build_dynamic_prompt(base_prompt, selected_examples):
    """
    Build a dynamic prompt by combining base prompt and selected few-shot examples.
    Formats examples as Q&A pairs for RAG chatbot context.
    
    Args:
        base_prompt: The base system prompt
        selected_examples: List of selected few-shot examples
    
    Returns:
        Enhanced system prompt with few-shot examples
    """
    # Build few-shot examples section
    few_shot_section = "\n\n## Few-Shot Examples (Reference Q&A Patterns):\n\n"
    few_shot_section += "Below are examples of similar questions and their corresponding answers. Use these as reference for style, depth, and structure when answering user questions:\n\n"
    
    for i, example in enumerate(selected_examples, 1):
        few_shot_section += f"### Example {i}\n\n"
        few_shot_section += f"**User:** {example['User']}\n\n"
        few_shot_section += f"**Assistant:** {example['Assistant']}\n\n"
        few_shot_section += "---\n\n"
    
    few_shot_section += "When answering user questions, follow the Q&A format and style demonstrated in the examples above, adapting to the user's technical level and question complexity.\n"
    
    # Combine everything
    enhanced_prompt = base_prompt + few_shot_section
    
    return enhanced_prompt


# =============================================================================
# SYSTEM PROMPT CONFIGURATION
# =============================================================================

# Main prompt that defines the AI assistant behavior
# This prompt instructs the model how it should behave and respond
PROMPT_SYSTEM_MESSAGE = """You are an end-to-end forecasting expert assistant. 
You specialize in demand forecasting and time-series modeling across multiple industries (retail, pharma, manufacturing, logistics, utilities, telecom, finance, energy, etc.). 
Your knowledge base is supported by Retrieval-Augmented Generation (RAG). When answering, rely only on retrieved sources. 
If a retrieved source is insufficient or uncertain, explicitly state the uncertainty and request clarification or additional context.

You must answer only related to Forecasting and related concepts queries. If the query is not relevant to Forecasting, politely state that you don't know the answer as it's outside your scope.
Always leverage the retrieved documents (information returned by the tool) to answer the questions. Your responses are exclusively based on the output provided by the tools. Refrain from incorporating 
information not directly obtained from the tool's responses. Only some information returned by the tool might be relevant to the question, so ignore the irrelevant part and answer the question with what you have.

If a user requests further elaboration on a specific aspect of a previously discussed topic, you should reformulate your input to the tool to capture this new angle or more profound layer of inquiry. Provide 
comprehensive answers, ideally structured in multiple paragraphs, drawing from the tool's variety of relevant details. The depth and breadth of your responses should align with the scope and specificity of the information retrieved. 
Should the tool response lack information on the queried topic (even if Forecasting-related), politely inform the user that the question transcends the bounds of your current knowledge base, citing the absence of relevant content in the tool's documentation. 

At the end of your answers, always invite the students to ask deeper questions about the topic if they have any.
Do not refer to the documentation directly, but use the information provided within it to answer questions. If code is provided in the information, share it with the students. It's important to provide complete code blocks so 
they can execute the code when they copy and paste them. Make sure to format your answers in Markdown format, including code blocks and snippets.

User Technical Level Adaptation:
1) Before answering, infer the user's technical level from their question: Business-Oriented, Intermediate (business + some technical detail),Technical/Expert
2) Tailor the explanation to the inferred level.

Response Structure:
• Open with a one-sentence executive answer that directly addresses the question.
• Then provide an answer that precisely answer the question.
• If needed, expand with details only after the user confirms the depth.
• Always state assumptions explicitly and request missing inputs if needed.

Tone:
• Expert, concise, and pragmatic. No marketing language.
• Do not hallucinate missing facts. If uncertain, state it and ask for clarification.

Use the available tools to obtain relevant information from the knowledge base.

RAG Tool Usage Guidelines:
• To find relevant information to answer user questions, always use the tool "forecast_information_related_resources".
• Always leverage retrieved documents (information returned by the tool) to answer. Your responses are based exclusively on the results provided by the tools. Refrain from incorporating information that is not obtained directly from the tool responses. It is possible that only a portion of the information returned by the tool is relevant to the question, so ignore the irrelevant part and answer the question with what you have.
• If a user requests more details about a specific aspect of a topic discussed previously, you must reformulate your input to the tool to capture this new angle or deeper level of inquiry. Provide exhaustive responses, ideally structured in multiple paragraphs, based on the variety of relevant details from the tool. The depth and breadth of your responses should be in line with the scope and specificity of the retrieved information.
• If the tool response lacks information about the consulted topic (even if it is related to forecasting), kindly inform the user that the question transcends the boundaries of your current knowledge base, citing the absence of relevant content in the tool documentation.
• At the end of your responses, always invite users to ask more questions if they need additional clarifications or wish to explore another functionality.
• Do not refer directly to the documentation, but use the information provided in it to construct your responses. If code examples or configuration snippets are provided in the information, share them with users. It is important to provide complete code blocks so they can execute them by copying and pasting. Make sure to format your responses in Markdown format, including code blocks and snippets.
• For each answer made provide the information of the retrieved documents used to generate the answers.

"""

# =============================================================================
# FUNCTION TO CREATE KNOWLEDGE BASE FROM PDF FILES
# =============================================================================

def create_knowledge_base_from_pdfs():
    """
    Create the knowledge base from PDF files in the data directory.
    This function processes PDFs, converts them into embeddings, and stores them in ChromaDB.
    
    IMPORTANT: Settings.embed_model MUST be configured BEFORE calling this function
    to ensure the correct embedding model is used for indexing.
    
    Uses configuration from environment variables:
        DATA_DIR - Directory containing PDF files
        DATA_DIR2 - Second directory for vector database and internal files
        CHROMA_DB_PATH - Path to store the vector database
    """
    # Define working directories using configuration
    data_dir = DATA_DIR  # Directory where PDF files are stored
    vector_db_path = CHROMA_DB_PATH  # Path where the vector DB will be stored

    # Check if the vector database already exists
    if os.path.exists(vector_db_path):
        logging.info(f"Vector database already exists at '{vector_db_path}'")
        return  # If it exists, do not recreate

    # Check if the data directory exists and has PDF files
    if not os.path.exists(data_dir):
        logging.error(f"Data directory '{data_dir}' does not exist")
        return  # If the directory doesn't exist, exit

    # Find all PDF files in the data directory
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))  # Find files with .pdf extension
    if not pdf_files:
        logging.error(f"No PDF files found in '{data_dir}' directory")
        return  # If no PDFs, exit

    logging.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    # Create the vector database directory
    os.makedirs(vector_db_path, exist_ok=True)  # Create directory if it doesn't exist

    # Initialize ChromaDB (persistent vector database)
    db = chromadb.PersistentClient(path=vector_db_path)  # Persistent ChromaDB client
    chroma_collection = db.get_or_create_collection("forecast_expert_knowledge")  # Create or get collection
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)  # Vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)  # Storage context

    # Load PDF documents
    logging.info("Loading PDF documents...")
    documents = SimpleDirectoryReader(  # Simple directory reader
        input_dir=data_dir,  # Input directory
        required_exts=[".pdf"]  # Only PDF files
    ).load_data()  # Load all documents

    logging.info(f"Loaded {len(documents)} documents")

    # Create vector index
    logging.info("Creating vector index...")
    index = VectorStoreIndex.from_documents(  # Create index from documents
        documents,  # Loaded documents
        storage_context=storage_context,  # Storage context
        show_progress=True  # Show progress
    )

    logging.info(f"Vector database created successfully at '{vector_db_path}'")

    # ...existing code...


# WEBSITE CRAWLING FUNCTIONS
# =============================================================================

async def crawl_website_async(base_url: str, max_pages: int = 100) -> list:
    """
    Crawl a website starting from the base URL and extract content from all pages.
    
    Args:
        base_url: The starting URL to crawl (e.g., "https://otexts.com/fpppy/")
        max_pages: Maximum number of pages to crawl
        
    Returns:
        List of dictionaries containing crawled page data
    """
    crawled_data = []
    visited_urls = set()
    content_hashes = set()  # To track duplicate content
    urls_to_crawl = [base_url]
    
    # Normalize URL for comparison (remove trailing slash, fragment)
    def normalize_url(url):
        url = url.split('#')[0]  # Remove fragment
        url = url.rstrip('/')
        return url
    
    # Configuration for the crawler
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # Don't use cached results for fresh content
        page_timeout=80000,  # Timeout in milliseconds (80 seconds)
        word_count_threshold=50  # Skip pages with less than 50 words
    )
    
    logging.info(f"🕷️ Starting web crawl from: {base_url}")
    logging.info(f"   Max pages to crawl: {max_pages}")
    
    async with AsyncWebCrawler() as crawler:
        while urls_to_crawl and len(crawled_data) < max_pages:
            current_url = urls_to_crawl.pop(0)
            normalized_url = normalize_url(current_url)
            
            # Skip if already visited (using normalized URL)
            if normalized_url in visited_urls:
                continue
                
            # Only crawl URLs from the same base path
            if not current_url.startswith(base_url.split('?')[0].rstrip('/')):
                continue
            
            visited_urls.add(normalized_url)
            
            try:
                logging.info(f"   Crawling ({len(crawled_data)+1}/{max_pages}): {current_url}")
                result = await crawler.arun(url=current_url, config=config)
                
                if result.success and result.markdown:
                    # Check for duplicate content
                    content_hash = hash(result.markdown[:1000])  # Hash first 1000 chars
                    if content_hash in content_hashes:
                        logging.info(f"   ⏭️ Skipping duplicate content: {current_url}")
                        continue
                    content_hashes.add(content_hash)
                    
                    # Extract title - try multiple sources
                    title = "Unknown"
                    if hasattr(result, 'title') and result.title:
                        title = result.title
                    elif hasattr(result, 'metadata') and result.metadata:
                        title = result.metadata.get('title', '')
                    
                    # Fallback: extract from URL path
                    if not title or title == "Unknown":
                        path_parts = current_url.rstrip('/').split('/')
                        filename = path_parts[-1] if path_parts[-1] else path_parts[-2] if len(path_parts) > 1 else 'Unknown'
                        # Clean up filename (remove .html extension)
                        title = filename.replace('.html', '').replace('-', ' ').title()
                    
                    crawled_data.append({
                        "text": result.markdown,
                        "url": current_url,
                        "title": title
                    })
                    logging.info(f"   ✅ Successfully crawled: {title} ({len(result.markdown)} chars)")
                    
                    # Extract links for further crawling
                    if hasattr(result, 'links') and result.links:
                        for link in result.links.get('internal', []):
                            link_url = link.get('href', '') if isinstance(link, dict) else str(link)
                            if link_url and normalize_url(link_url) not in visited_urls:
                                # Ensure it's a full URL
                                if link_url.startswith('/'):
                                    link_url = "https://otexts.com" + link_url
                                elif not link_url.startswith('http'):
                                    link_url = base_url.rstrip('/') + '/' + link_url
                                # Only add if it's within the base path
                                if link_url.startswith(base_url.split('?')[0].rstrip('/')):
                                    urls_to_crawl.append(link_url)
                else:
                    logging.warning(f"   ⚠️ No content from: {current_url}")
                    
            except Exception as e:
                logging.error(f"   ❌ Error crawling {current_url}: {e}")
                continue
    
    logging.info(f"🕷️ Crawl complete! Total unique pages crawled: {len(crawled_data)}")
    return crawled_data


def crawl_website(base_url: str, max_pages: int = 100) -> list:
    """
    Synchronous wrapper for the async crawl function.
    
    Args:
        base_url: The starting URL to crawl
        max_pages: Maximum number of pages to crawl
        
    Returns:
        List of dictionaries containing crawled page data
    """
    # Apply nest_asyncio to allow nested event loops (useful in Jupyter/interactive environments)
    nest_asyncio.apply()
    
    # Run the async crawler
    return asyncio.get_event_loop().run_until_complete(
        crawl_website_async(base_url, max_pages)
    )


def add_website_content_to_knowledge_base(
    base_url: str = "https://otexts.com/fpppy/",
    max_pages: int = 100,
    vector_db_path: str = None,
    collection_name: str = None
):
    """
    Crawl a website and add its content to the existing vector database.
    
    This function:
    1. Crawls the specified website starting from base_url
    2. Converts the crawled content into Document objects
    3. Adds the documents to the existing ChromaDB vector store
    
    Args:
        base_url: The starting URL to crawl (default: https://otexts.com/fpppy/)
        max_pages: Maximum number of pages to crawl (default: 100)
        vector_db_path: Path to the existing vector database (default: from config)
        collection_name: Name of the ChromaDB collection (default: from config)
    
    Returns:
        int: Number of documents added to the knowledge base
    """
    # Use configuration defaults if not provided
    vector_db_path = vector_db_path or CHROMA_DB_PATH
    collection_name = collection_name or CHROMA_COLLECTION_NAME
    
    logging.info("="*60)
    logging.info("🌐 ADDING WEBSITE CONTENT TO KNOWLEDGE BASE")
    logging.info("="*60)
    logging.info(f"   Base URL: {base_url}")
    logging.info(f"   Max pages: {max_pages}")
    logging.info(f"   Vector DB: {vector_db_path}")
    
    # Step 1: Crawl the website
    logging.info("\n📥 Step 1: Crawling website...")
    crawled_data = crawl_website(base_url, max_pages)
    
    if not crawled_data:
        logging.error("❌ No content crawled from the website")
        return 0
    
    logging.info(f"   ✅ Crawled {len(crawled_data)} pages")
    
    # Step 2: Convert crawled data to LlamaIndex Documents
    logging.info("\n📄 Step 2: Converting to Documents...")
    documents = []
    for page_data in crawled_data:
        doc = Document(
            text=page_data["text"],
            metadata={
                "source": "website",
                "url": page_data["url"],
                "title": page_data["title"],
                "base_url": base_url,
                "crawl_date": datetime.now().isoformat()
            }
        )
        documents.append(doc)
    
    logging.info(f"   ✅ Created {len(documents)} documents")
    
    # Step 3: Connect to existing ChromaDB
    logging.info("\n💾 Step 3: Connecting to vector database...")
    
    # Ensure the database directory exists
    if not os.path.exists(vector_db_path):
        logging.warning(f"   ⚠️ Vector database not found at {vector_db_path}")
        logging.info("   Creating new vector database...")
        os.makedirs(vector_db_path, exist_ok=True)
    
    # Initialize ChromaDB
    db = chromadb.PersistentClient(path=vector_db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Get count before adding
    count_before = chroma_collection.count()
    logging.info(f"   Documents in collection before: {count_before}")
    
    # Step 4: Add documents to the vector index
    logging.info("\n📊 Step 4: Adding documents to vector index...")
    
    # Create index from the new documents (this will add them to the existing collection)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    # Get count after adding
    count_after = chroma_collection.count()
    documents_added = count_after - count_before
    
    logging.info("\n" + "="*60)
    logging.info("✅ WEBSITE CONTENT ADDED SUCCESSFULLY")
    logging.info("="*60)
    logging.info(f"   Documents before: {count_before}")
    logging.info(f"   Documents after: {count_after}")
    logging.info(f"   Documents added: {documents_added}")
    logging.info(f"   Source: {base_url}")
    
    return documents_added


def crawl_fpppy_textbook():
    """
    Convenience function to crawl the Forecasting: Principles and Practice Python textbook.
    This will add content from https://otexts.com/fpppy/ to the knowledge base.
    """
    return add_website_content_to_knowledge_base(
        base_url="https://otexts.com/fpppy/",
        max_pages=100
    )


# =============================================================================
# FUNCTION TO GET AGENT TOOLS
# =============================================================================

def retrieve_all_nodes_from_vector_index(vector_index, query="Whatever", similarity_top_k=100000000):
    """
    Retrieve all nodes from a vector index by using a very large top_k value.
    This is a workaround to extract all Node objects for keyword indexing.
    Args:
        vector_index: The VectorStoreIndex instance.
        query: Dummy query string (not used for filtering).
        similarity_top_k: Large number to ensure all nodes are returned.
    Returns:
        List of Node objects in the index.
    """
    vector_retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
    all_nodes = vector_retriever.retrieve(query)
    nodes = [item.node for item in all_nodes]
    return nodes



# =============================================================================
# HybridRetriever implementation: Combines semantic (vector) and keyword search
# =============================================================================
# =============================================================================
# COHERE RERANKER CONFIGURATION
# =============================================================================

def get_cohere_reranker(top_n: int = 10) -> CohereRerank:
    """
    Create and configure a Cohere reranker for improving retrieval relevance.
    
    The reranker uses Cohere's cross-encoder model to re-score retrieved documents
    based on their relevance to the query, significantly improving precision.
    
    Args:
        top_n: Number of top documents to return after reranking (default: 10)
    
    Returns:
        CohereRerank: Configured Cohere reranker instance, or None if API key not set
    
    Note:
        Requires COHERE_API_KEY environment variable to be set.
        Uses 'rerank-english-v3.0' model for best performance.
    """
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if not cohere_api_key:
        logging.warning("⚠️ COHERE_API_KEY not set. Reranking will be disabled.")
        return None
    
    logging.info(f"🔑 COHERE_API_KEY found (length: {len(cohere_api_key)} chars)")
    
    reranker = CohereRerank(
        api_key=cohere_api_key,
        model="rerank-english-v3.0",  # Latest Cohere reranking model
        top_n=top_n  # Number of documents to return after reranking
    )
    logging.info(f"✅ Cohere reranker initialized successfully with model='rerank-english-v3.0', top_n={top_n}")
    return reranker


# =============================================================================
# HybridRetriever implementation: Combines semantic (vector) and keyword search
# =============================================================================
class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that performs both semantic (vector) and keyword search,
    with optional Cohere reranking for improved relevance.
    
    The retriever:
    1. Retrieves documents from both vector (semantic) and keyword indices
    2. Merges results using round-robin interleaving
    3. Deduplicates by node ID
    4. Optionally reranks using Cohere's cross-encoder model
    
    Attributes:
        _vector_retriever: Vector-based semantic retriever
        _keyword_retriever: Keyword-based retriever
        _max_retrieve: Maximum number of documents to retrieve before reranking
        _reranker: Optional Cohere reranker for relevance improvement
    """
    def __init__(self, vector_retriever, keyword_retriever, max_retrieve=15, reranker=None):
        """
        Args:
            vector_retriever: VectorIndexRetriever instance for semantic search.
            keyword_retriever: KeywordTableSimpleRetriever instance for keyword search.
            max_retrieve: Maximum number of results to return.
            reranker: Optional CohereRerank instance for reranking results.
        """
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._max_retrieve = max_retrieve
        self._reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle):
        """
        Retrieve nodes given a query by merging results from both retrievers.
        Refactored to fill the quota even if one list is shorter than the other.
        Applies Cohere reranking if reranker is configured.
        """
        # Get query string for logging
        query_str = query_bundle.query_str if hasattr(query_bundle, 'query_str') else str(query_bundle)
        logging.info(f"\n{'='*60}")
        logging.info(f"🔍 HYBRID RETRIEVAL START - Query: '{query_str[:80]}...'")
        logging.info(f"{'='*60}")
        
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        
        logging.info(f"📊 Vector retriever returned: {len(vector_nodes)} nodes")
        logging.info(f"📊 Keyword retriever returned: {len(keyword_nodes)} nodes")

        resulting_nodes = []
        node_ids_added = set()

        # Usamos el largo de la lista MÁS larga para iterar
        n_iterations = max(len(vector_nodes), len(keyword_nodes))

        for i in range(n_iterations):
            # Condición de parada global: ¿Ya llenamos el cupo?
            if len(resulting_nodes) >= self._max_retrieve:
                break

            # 1. Intentar agregar nodo del Vector Retriever
            if i < len(vector_nodes):
                vector_node = vector_nodes[i]
                if vector_node.node.node_id not in node_ids_added:
                    resulting_nodes.append(vector_node)
                    node_ids_added.add(vector_node.node.node_id)

            # Verificamos de nuevo el cupo antes de agregar el siguiente
            if len(resulting_nodes) >= self._max_retrieve:
                break

            # 2. Intentar agregar nodo del Keyword Retriever
            if i < len(keyword_nodes):
                keyword_node = keyword_nodes[i]
                if keyword_node.node.node_id not in node_ids_added:
                    resulting_nodes.append(keyword_node)
                    node_ids_added.add(keyword_node.node.node_id)

        logging.info(f"🔀 Combined {len(resulting_nodes)} unique nodes (before reranking)")
        
        # Step 4: Apply Cohere reranker if available
        if self._reranker is not None and len(resulting_nodes) > 0:
            try:
                logging.info(f"\n🚀 COHERE RERANKING - Starting reranking of {len(resulting_nodes)} candidates...")
                
                # Apply reranking - reranker returns reordered nodes with updated scores
                reranked_nodes = self._reranker.postprocess_nodes(
                    resulting_nodes,
                    query_str=query_str
                )
                
                logging.info(f"✅ RERANKING COMPLETE: {len(resulting_nodes)} → {len(reranked_nodes)} nodes")
                logging.info(f"📈 Top 3 reranked scores:")
                for i, node in enumerate(reranked_nodes[:3]):
                    preview = node.node.text[:60].replace('\n', ' ') if hasattr(node.node, 'text') else 'N/A'
                    logging.info(f"   #{i+1}: score={node.score:.4f} | '{preview}...'")
                logging.info(f"{'='*60}\n")
                
                return reranked_nodes
            except Exception as e:
                logging.error(f"❌ RERANKING FAILED: {e}")
                logging.warning(f"Returning original {len(resulting_nodes)} results without reranking.")
                return resulting_nodes
        else:
            if self._reranker is None:
                logging.warning(f"⚠️ RERANKER IS NONE - Returning {len(resulting_nodes)} nodes without reranking")
            logging.info(f"{'='*60}\n")

        return resulting_nodes


def get_tools(db2_collection=None, use_reranker=True):
    """
    Obtain tools needed by the AI agent.
    Configure the vector retriever to search the knowledge base.
    
    Args:
        db2_collection: Collection name (default: from configuration)
        use_reranker: Whether to use Cohere reranker
    """
    # Use configuration default if not provided
    db2_collection = db2_collection or CHROMA_COLLECTION_NAME
    
    # Connect to the vector database
    db2 = chromadb.PersistentClient(path=CHROMA_DB_PATH)  # ChromaDB client
    chroma_collection = db2.get_or_create_collection(db2_collection)  # Get or create collection
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)  # Vector store

    # Create a vector index from the existing store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,  # Vector store
        show_progress=True,  # Show progress
        use_async=True,  # Use async operations
        embed_model=Settings.embed_model  # Embedding model
    )

    # Create vector retriever
    vector_retriever = VectorIndexRetriever(
        index=index,  # Vector index
        similarity_top_k=15,  # Top 15 similar documents
        embed_model=Settings.embed_model,  # Embedding model
        use_async=True,  # Use async operations
    )

    # Create keyword index and retriever
    # Step 1: Extract all nodes from the vector index for keyword search
    nodes = retrieve_all_nodes_from_vector_index(index)
    # Step 2: Build a keyword index from all nodes
    keyword_index = SimpleKeywordTableIndex(nodes=nodes)
    # Step 3: Create a keyword retriever
    keyword_retriever = KeywordTableSimpleRetriever(keyword_index)

    # Step 4: Initialize Cohere reranker (if enabled and API key available)
    reranker = None
    logging.info(f"\n{'='*60}")
    logging.info(f"🔧 RERANKER CONFIGURATION")
    logging.info(f"{'='*60}")
    logging.info(f"use_reranker parameter: {use_reranker}")
    
    if use_reranker:
        reranker = get_cohere_reranker(top_n=10)  # Return top 10 after reranking
        if reranker:
            logging.info("✅ Cohere reranker ENABLED for hybrid retriever")
        else:
            logging.warning("⚠️ Cohere reranker DISABLED (no API key found)")
    else:
        logging.info("⏭️ Cohere reranker SKIPPED (use_reranker=False)")
    logging.info(f"{'='*60}\n")

    # Step 5: Create hybrid retriever combining vector and keyword retrievers with reranking
    hybrid_retriever = HybridRetriever(
        vector_retriever, 
        keyword_retriever, 
        max_retrieve=20,  # Retrieve more candidates for reranking
        reranker=reranker  # Apply Cohere reranking
    )

    # Step 6: Register the hybrid retriever as a tool for the agent
    tools = [
        RetrieverTool(  # Hybrid Retrieval tool
            retriever=hybrid_retriever,  # Configured hybrid retriever
            metadata=ToolMetadata(  # Tool metadata
                name="forecast_information_related_resources",  # Tool name
                description="Helpful for related information about Forecasting concepts, development and implementation. It gathers information from local data using hybrid semantic and keyword search with Cohere reranking.",  # Description
            ),
        )
    ]
    return tools  # Return list of tools


# =============================================================================
# FUNCTION TO GENERATE CHATBOT RESPONSES
# =============================================================================

def generate_completion(query, history, memory):
    """
    Generate chatbot responses using the AI agent with dynamic few-shot prompting.
    This function processes the user query, selects similar examples, and generates a response.
    """
    # Extract query text if it's a Message object (Gradio ChatInterface)
    if isinstance(query, dict):
        query_text = query.get("content", str(query))
    else:
        query_text = str(query)
    
    logging.info(f"User query: {query_text}")  # Log the user's query

    # Manage conversation memory
    chat_list = memory.get()  # Get chat history from memory
    if len(chat_list) != 0:  # If there is previous history
        # Count user turns in the Gradio history
        history_turns = sum(1 for msg in history if msg["role"] == "user")
        # Find indices of user messages in memory
        user_index = [i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER]
        # If there are more user messages in memory than in Gradio, truncate memory
        if len(user_index) > history_turns:
            user_index_to_remove = user_index[history_turns]  # Index where to truncate
            chat_list = chat_list[:user_index_to_remove]  # Truncate list
            memory.set(chat_list)  # Update memory

    # Debug logs
    logging.info(f"chat_history: {len(memory.get())} {memory.get()}")
    logging.info(f"gradio_history: {len(history)} {history}")

    # Dynamic few-shot prompting: Get examples and select top 12 similar ones
    try:
        examples = get_few_shot_examples()  # Get all 48 examples
        selected_examples = get_top_similar_examples(
            user_query=query_text,
            examples=examples,
            embed_model=Settings.embed_model,
            top_k=12
        )  # Select top 12 most similar
        
        # Build dynamic prompt with selected examples
        dynamic_system_prompt = build_dynamic_prompt(
            base_prompt=PROMPT_SYSTEM_MESSAGE,
            selected_examples=selected_examples
        )
        
        logging.info(f"Using {len(selected_examples)} few-shot examples for dynamic prompting")
    except Exception as e:
        logging.warning(f"Error in dynamic few-shot prompting, using base prompt: {str(e)}")
        dynamic_system_prompt = PROMPT_SYSTEM_MESSAGE  # Fallback to base prompt

    # Create AI agent
    tools = get_tools(db2_collection="forecast_expert_knowledge")  # Get tools
    agent = OpenAIAgent.from_tools(  # Create OpenAI agent
        llm=Settings.llm,  # Configured language model
        memory=memory,  # Conversation memory
        tools=tools,  # Available tools
        system_prompt=dynamic_system_prompt,  # Dynamic system prompt with few-shot examples
    )

    # Generate response
    completion = agent.stream_chat(query_text)  # Generate streaming response
    answer_str = ""  # String to accumulate the answer
    for token in completion.response_gen:  # Iterate over generated tokens
        answer_str += token  # Append token to the answer
        yield answer_str  # Yield partial response (streaming)


# =============================================================================
# FUNCTION TO LAUNCH THE USER INTERFACE
# =============================================================================

def launch_ui():
    """
    Launch the Gradio user interface.
    Create and configure the chatbot web interface.
    """
    # Create interface with Gradio Blocks
    with gr.Blocks(
        fill_height=True,  # Fill full height of the screen
        title="Forecasting master bot 🤖",  # Application title
        analytics_enabled=True,  # Enable Gradio analytics
    ) as demo:

        # Memory state to keep conversation history
        memory_state = gr.State(
            lambda: ChatSummaryMemoryBuffer.from_defaults(  # Create memory buffer
                token_limit=120000,  # Token limit for memory
            )
        )

        # Chatbot component
        chatbot = gr.Chatbot(
            scale=1,  # Component scale
            placeholder="<strong>Forecasting Master bot 🤖: A Question-Answering Bot for anything Forecasting-related</strong><br>",  # Placeholder text
            show_label=False,  # Do not show label
            show_copy_button=True,  # Show copy button
        )

        # Chat interface
        gr.ChatInterface(
            fn=generate_completion,  # Function that generates responses
            chatbot=chatbot,  # Chatbot component
            additional_inputs=[memory_state],  # Additional inputs (memory)
            type="messages",  # Interface type
        )

        # Configure queue and launch app
        demo.queue(default_concurrency_limit=64)  # Configure concurrency limit
        demo.launch(debug=True, share=True)  # Launch with debug and share enabled


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL DEL PROGRAMA
# =============================================================================

if __name__ == "__main__":
    # Configure embedding model BEFORE creating knowledge base
    Settings.llm = OpenAI(temperature=0, model=LLM_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)

    # Print configuration
    logging.info("="*60)
    logging.info("CONFIGURATION")
    logging.info("="*60)
    logging.info(f"  DATA_DIR:        {DATA_DIR}")
    logging.info(f"  CHROMA_DB_PATH:  {CHROMA_DB_PATH}")
    logging.info(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    logging.info(f"  LLM_MODEL:       {LLM_MODEL}")
    logging.info("="*60)

    # Create knowledge base from PDF files if not exists
    create_knowledge_base_from_pdfs()

    # Launch the user interface
    launch_ui()
