# Datathon 2025: LIFELINE - CTG Analysis for Fetal Health

**Team:** Ricard Josse Meyer, Charles Lukas Chairos Yo, Poon Wei Lok\n
**Team ID:** TM-37
This repository contains the complete codebase and deliverables for our submission to the Datathon 2025 "Lifeline" competition. Our project focuses on developing a machine learning pipeline to classify fetal health status from Cardiotocography (CTG) data, with a strong emphasis on achieving both high performance and clinical interpretability.

## Our Approach & Key Findings

Our final solution is a **Hybrid "Glass-Box" Model**, which combines sophisticated, domain-specific feature engineering with a high-performance LightGBM classifier. This approach was chosen after comparing it against two other strong candidates: a simpler feature-engineered Random Forest and a complex "black-box" super-ensemble.

*   **Peak Performance:** Our hybrid model achieves a **96.9% accuracy** and a critical **97.1% recall** for the "Pathologic" class, matching the performance of complex ensembles while remaining fully interpretable.
*   **Clinical Interpretability:** Through SHAP analysis, we have confirmed that our model's decisions are driven by a transparent combination of raw clinical signals (like `ASTV` and `AC`) and our own engineered features (like `STV_ratio`). This makes it a trustworthy and reliable tool for clinical settings.

## Repository Structure

This repository contains all necessary files within the `/Deliverables/` directory.

*   `/Deliverables/data/`: Contains the raw `CTG.xls` dataset.
*   `/Deliverables/Scripts/`: Contains the primary Python scripts for our main pipeline (preprocessing, feature engineering, and the final hybrid model).
*   `/Deliverables/final_model_artifacts/`: Contains the saved `.pkl` artifact for our winning Hybrid Glass-Box model.
*   `/Deliverables/Unused_Models/`: Contains the scripts, artifacts, and evaluations for the other two models we benchmarked (Superblend and Feature-Engineered RF).
*   `/Deliverables/results/`: Contains evaluation reports and output plots.

## How to Reproduce Our Results

To set up the environment and run the complete pipeline, please follow these steps from your terminal, starting from the root `Datathon_24` directory.

### 1. Setup

First, clone this repository and install the required Python packages.

```
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Processing and Training Pipeline

The following scripts should be run in order. They are designed to work with the file structure within the `/Deliverables/` folder.

**Step 1: Preprocess the Raw Data**
This script cleans the raw `CTG.xls` file.

```
python Deliverables/Scripts/ctg_preprocessing_fixed.py
```

**Step 2: Create the Engineered Feature Set**
This script takes the cleaned data and generates the new, clinically-inspired features.

```
python Deliverables/Scripts/create_engineered_test_set.py
```

**Step 3: Train and Evaluate All Models**
Run the training scripts for each of the three models discussed in our report.

```
# Train our final "Hybrid Glass-Box" model
python Deliverables/Scripts/corrected_hybrid.py

# Train the "Feature-Engineered RF" baseline model
python Deliverables/Unused_Models/Training_Script/train_feature_engineered.py

# Train the "Superblend" black-box model
python Deliverables/Unused_Models/Training_Script/train_superblend.py
```

After running these commands, all model artifacts and evaluation reports will be generated, fully reproducing the results presented in our academic report.
```
