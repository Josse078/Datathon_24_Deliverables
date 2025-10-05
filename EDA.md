# Exploratory Data Analysis
**Datathon 2025 - Team TM-37**

## Overview

This document details our exploratory analysis of the UCI Cardiotocography (CTG) dataset, including correlation studies, feature relationships, and the rationale behind our design decisions.

---

## Dataset Summary

- **Total Records:** 2,126 CTG recordings
- **Features:** 24 standard CTG measurements
- **Target Variable:** NSP (Normal, Suspect, Pathological)
- **Class Distribution:** 
  - Normal: 78% (1,648 cases)
  - Suspect: 14% (293 cases)  
  - Pathological: 8% (175 cases)

**Key Challenge:** Severe class imbalance requiring specialized handling.

---

## Correlation Analysis

We performed two levels of correlation analysis to understand feature relationships and identify the strongest predictors of fetal distress.

### 1. Overall Feature Correlation Matrix

![Feature Correlation Matrix](Deliverables/results/eda/ctg_feature_correlation_matrix.png)

This heatmap reveals the correlation structure among all 24 CTG features.

#### Key Findings:

**Highly Correlated Feature Groups (r > 0.85):**

1. **Histogram Statistics Cluster**
   - Mode ↔ Mean: r = 0.89
   - Mode ↔ Median: r = 0.93
   - Mean ↔ Median: r = 0.95
   - **Interpretation:** These features capture similar information about baseline heart rate distribution

2. **Baseline Heart Rate Relationships**
   - LB ↔ Mode: r = 0.71
   - LB ↔ Mean: r = 0.72
   - LB ↔ Median: r = 0.79
   - **Interpretation:** Baseline heart rate is fundamentally linked to histogram statistics

3. **Variability Measures**
   - ASTV ↔ MSTV: r = 0.43
   - ALTV ↔ MLTV: r = 0.47
   - **Interpretation:** Short-term and mean variability measures show moderate correlation

**Independent Features:**
- UC (Uterine Contractions): Low correlation with most features (r < 0.3)
- FM (Fetal Movements): Relatively independent signal
- Deceleration features (DL, DS, DP, DR): Moderate intercorrelation

#### Clinical Implications:

The strong correlation among histogram features (Mode, Mean, Median, Variance) suggests **multicollinearity**, which could affect linear models. However:
- ✅ Tree-based models (LightGBM, Random Forest) naturally handle this through feature subsampling
- ✅ Each feature may still capture subtle variations important for edge cases
- ✅ Competition requirements mandate retaining all 24 features

---

### 2. Feature-to-Target Correlation Ranking

![Feature-Target Correlation](Deliverables/results/eda/ctg_target_correlation_ranking.png)

This vertical heatmap ranks features by their correlation with NSP (fetal health status).

#### Top Risk Indicators (Positive Correlation with NSP):

| Rank | Feature | Correlation | Clinical Meaning |
|------|---------|-------------|------------------|
| 1 | **DP** (Prolonged Decelerations) | **+0.492** | Strongest predictor - indicates severe fetal stress |
| 2 | **ASTV** (Short-term Variability) | **+0.470** | Abnormally high variability signals distress |
| 3 | **ALTV** (Long-term Variability) | **+0.422** | Loss of beat-to-beat regulation |
| 4 | Variance | +0.208 | High spread in heart rate distribution |
| 5 | LB (Baseline Heart Rate) | +0.147 | Mild association with abnormal baseline |
| 6 | DS (Severe Decelerations) | +0.132 | Additional deceleration indicator |

#### Protective Factors (Negative Correlation with NSP):

| Rank | Feature | Correlation | Clinical Meaning |
|------|---------|-------------|------------------|
| 1 | **AC** (Accelerations) | **-0.340** | Reassuring sign of fetal well-being |
| 2 | **Mode** | **-0.253** | Normal baseline mode indicates health |
| 3 | **Mean** | **-0.230** | Normal average heart rate |
| 4 | **MLTV** | **-0.226** | Stable long-term variability patterns |
| 5 | **Median** | **-0.208** | Consistent central heart rate |

#### Weak Predictors (|r| < 0.10):

- DR (Delayed Recoveries): r = 0.000 (no linear relationship)
- Nzeros, Nmax, Max, Min: Very weak correlations
- **Interpretation:** These may still capture non-linear patterns useful for tree-based models

---

## Critical Design Decision: Feature Retention

### The Dilemma

Our correlation analysis revealed significant **multicollinearity**:
- Mode, Mean, Median have r > 0.89 (near-perfect correlation)
- Standard feature selection would suggest removing redundant features
- Textbook ML would say: "Drop highly correlated features to reduce overfitting"

### Our Decision: Keep ALL 24 Features

**Why we retained all original features despite multicollinearity:**

#### 1. Competition Compliance ✅
The competition explicitly states:
> "Your model should be able to take in the original input features for a CTG time which is: {b, e, AC, FM, UC, DL, DS, DP, DR, LB, ASTV, MSTV, ALTV, MLTV, Width, Min, Max, Nmax, Nzeros, Mode, Mean, Median, Variance, Tendency}"

Removing features would violate evaluation requirements.

#### 2. LightGBM Robustness ✅
Our chosen algorithm (LightGBM) naturally handles multicollinearity through:
- **Random feature subsampling** at each tree split
- **Gradient-based learning** that automatically assigns appropriate weights
- **Regularization** preventing overfitting from redundant features

#### 3. Clinical Safety ✅
In medical applications, conservative feature retention prevents:
- Inadvertent loss of subtle diagnostic signals
- Edge cases where "redundant" features diverge
- Model failures on atypical presentations

For example: In rare cases, Mode and Mean might diverge due to bimodal distributions, and both measurements become critical.

#### 4. Non-Linear Interactions ✅
Even highly correlated features can have distinct non-linear effects:
- Mean captures central tendency
- Mode captures most frequent value (may differ in skewed distributions)
- Median is robust to outliers
- Their **interactions** may reveal complex patterns invisible to correlation analysis

### Our Strategy: Additive Feature Engineering

Instead of **replacing** original features, we **added** 9 engineered features:

Original Features (24) + Engineered Features (9) = Total Features (33)


**Engineered additions:**
- STV_ratio, LTV_ratio (variability balance)
- weighted_deceleration_score (severity weighting)
- clinical_risk_score (composite indicator)
- Baseline heart rate flags (bradycardia, tachycardia)
- Binary severity markers

This approach:
✅ Preserves all original information (competition-compliant)  
✅ Adds domain knowledge (clinical expertise)  
✅ Lets LightGBM learn optimal feature combinations automatically

---

## Impact on Model Design

These EDA findings directly informed our pipeline:

### 1. Data Augmentation Strategy
**Finding:** 8% pathological class (severe imbalance)  
**Action:** Applied Borderline-SMOTE to oversample rare pathological cases  
**Rationale:** Synthetic samples generated near class boundaries help model learn subtle distinctions

### 2. Feature Engineering Focus
**Finding:** DP, ASTV, ALTV are top predictors (r > 0.42)  
**Action:** Created `weighted_deceleration_score` emphasizing DP × 3  
**Action:** Created `STV_ratio` and `LTV_ratio` to capture variability balance  
**Rationale:** Engineering features aligned with correlation insights

### 3. Algorithm Selection
**Finding:** High multicollinearity among histogram features  
**Action:** Chose LightGBM over linear models  
**Rationale:** Tree-based learning naturally handles feature redundancy

### 4. Interpretability Approach
**Finding:** Clinical need to explain predictions  
**Action:** Integrated SHAP analysis focusing on top correlated features  
**Rationale:** Judges and clinicians need to understand why model makes decisions

---

## Validation of Approach

Our final model performance validates these EDA-driven decisions:

- **96.93% Accuracy** (highest in competition)
- **97.14% Pathological Recall** (critical safety metric)
- **99.7% Normal Recall** (minimizes false alarms)

The feature importance rankings from our trained LightGBM model closely match the correlation analysis:
1. ASTV (high feature importance + high correlation)
2. DP (high feature importance + high correlation)
3. ALTV (high feature importance + high correlation)

This alignment confirms our EDA correctly identified the key predictive signals.

---

## Conclusion

Our exploratory analysis revealed:
1. Strong multicollinearity among histogram features (requiring robust algorithm choice)
2. Clear predictive hierarchy (DP, ASTV, ALTV as top indicators)
3. Class imbalance requiring specialized handling
4. Justification for retaining all 24 original features despite redundancy

These insights shaped every subsequent modeling decision, resulting in the competition's highest-performing model while maintaining full clinical interpretability and competition compliance.

---

**For full methodology and results, see:**
- [Main README](README.md) - Pipeline overview
- [Academic Report](Datathon%201-Page%20Academic%20Report.pdf) - Complete analysis
- [Training Script](Dhttps://github.com/Josse078/Datathon_24_Deliverables/blob/main/Scripts/corrected_hybrid.py) - Implementation details
