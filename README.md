# Amazon ML Challenge 2025: Smart Product Pricing

This repository contains the solution developed for the **Amazon ML Challenge 2025**. The goal of the competition was to predict the list price of products based on unstructured catalog metadata including titles, descriptions, and category information.

## 📊 Project Performance
- **Validation SMAPE:** 51.91
- **Evaluation Metric:** Symmetric Mean Absolute Percentage Error (SMAPE)
- **Model Architecture:** Gradient Boosted Decision Trees (LightGBM)

## 🛠️ Technical Implementation

### 1. Data Preprocessing & Feature Engineering
- **NLP Pipeline:** Implemented a robust text cleaning pipeline to handle noise in product titles and descriptions.
- **Vectorization:** Utilized **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform unstructured text into high-dimensional numerical vectors, capturing semantic relevance across the catalog.
- **Log-Transformation:** Applied `log1p` transformation to the target variable (Price) to handle the heavy-tailed distribution of e-commerce pricing and better align the RMSE optimization with the SMAPE metric.

### 2. Model Selection & Optimization
- **Algorithm:** Leveraged **LightGBM** for its efficiency with large-scale datasets and ability to handle sparse features from TF-IDF.
- **Tuning:** Performed hyperparameter optimization focusing on `num_leaves`, `learning_rate`, and `feature_fraction` to balance model complexity and generalization.



## 🚀 Roadmap & Future Improvisations
*To further reduce the SMAPE score below 45, the following high-impact strategies were identified for future iterations:*

- **Multimodal Fusion:** Integrating visual signals by extracting image embeddings using a pre-trained **CLIP (ViT-B/32)** model.
- **Numeric Entity Extraction:** Implementing Regex-based feature engineering to extract "Item Pack Quantity" (IPQ) and standardized units (e.g., weight, volume), which are primary drivers of product pricing.
- **Transformer Embeddings:** Replacing TF-IDF with transformer-based models like **DeBERTa-v3-Large** for superior semantic context.
- **Ensemble Learning:** Developing a weighted ensemble of XGBoost, LightGBM, and CatBoost to reduce variance and improve prediction stability.

## 📂 Project Structure
- `mlamazon.ipynb`: Main Jupyter notebook containing the full training and inference pipeline.
- `.gitignore`: Configured to exclude large datasets and temporary Jupyter checkpoints.
- `LICENSE`: MIT License.

## ⚙️ Requirements
- Python 3.x
- LightGBM
- Scikit-learn
- Pandas / Numpy
