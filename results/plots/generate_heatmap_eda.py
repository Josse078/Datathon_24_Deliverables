"""
==============================================================================
DATATHON 2025: LIFELINE - EDA CORRELATION ANALYSIS
Team TM-37

Generates two key heatmaps:
1. Overall feature-feature correlation matrix (24 CTG features)
2. Feature-to-target correlation ranking (which features predict NSP best)
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Ensure output directory exists
OUTPUT_DIR = './'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
print("Loading cleaned CTG dataset...")
DATA_PATH = '../../data/preprocessed/cleaned_data.csv'  # Adjust path if needed
df = pd.read_csv(DATA_PATH)

# Define the 24 original CTG features
ORIGINAL_FEATURES = [
    'b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR', 'LB',
    'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max',
    'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
]

# Verify all features exist
available_features = [f for f in ORIGINAL_FEATURES if f in df.columns]
print(f"Found {len(available_features)}/24 original features")

TARGET_COL = 'NSP'

# ==============================================================================
# HEATMAP 1: OVERALL FEATURE CORRELATION MATRIX
# ==============================================================================
print("\nGenerating feature correlation matrix...")

corr_matrix = df[available_features].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='RdYlBu_r',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8},
    vmin=-1,
    vmax=1
)
plt.title('CTG Features Correlation (24 Features)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

output_path_1 = os.path.join(OUTPUT_DIR, 'ctg_feature_correlation_matrix.png')
plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path_1}")
plt.close()

# ==============================================================================
# HEATMAP 2: FEATURE-TO-TARGET CORRELATION RANKING
# ==============================================================================
print("\nGenerating feature-to-target correlation ranking...")

# Calculate Pearson correlation for each feature with NSP
target_correlations = []
for feature in available_features:
    r, p_value = pearsonr(df[feature], df[TARGET_COL])
    target_correlations.append({
        'Feature': feature,
        'Correlation': r,
        'P_value': p_value
    })

# Create DataFrame and sort by absolute correlation
corr_df = pd.DataFrame(target_correlations)
corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
corr_df = corr_df.set_index('Feature')

# Create vertical heatmap
plt.figure(figsize=(6, 12))
sns.heatmap(
    corr_df[['Correlation']],
    annot=True,
    fmt='.3f',
    cmap='RdYlBu_r',
    center=0,
    cbar_kws={'shrink': 0.6},
    vmin=-0.6,
    vmax=0.6,
    linewidths=0.5
)
plt.title('Feature Correlation with NSP (24-Feature Set)',
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Pearson r', fontsize=12)
plt.ylabel('')
plt.tight_layout()

output_path_2 = os.path.join(OUTPUT_DIR, 'ctg_target_correlation_ranking.png')
plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path_2}")
plt.close()

# ==============================================================================
# PRINT KEY FINDINGS
# ==============================================================================
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

print("\nTop 5 Features Correlated with NSP (Fetal Distress):")
top_5 = corr_df.head(5)
for idx, row in top_5.iterrows():
    direction = "⬆️ Risk factor" if row['Correlation'] > 0 else "⬇️ Protective factor"
    print(f"  {idx:15s}: r = {row['Correlation']:+.3f}  {direction}")

print("\nHighly Intercorrelated Feature Groups (r > 0.85):")
# Find pairs with correlation > 0.85
high_corr_pairs = []
for i in range(len(available_features)):
    for j in range(i+1, len(available_features)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.85:
            high_corr_pairs.append((available_features[i], available_features[j], r))

if high_corr_pairs:
    for feat1, feat2, r in high_corr_pairs[:5]:  # Show top 5
        print(f"  {feat1} ↔ {feat2}: r = {r:.3f}")
else:
    print("  No feature pairs with correlation > 0.85")

print("\n" + "="*60)
print("EDA COMPLETE - 2 heatmaps generated successfully!")
print("="*60)
