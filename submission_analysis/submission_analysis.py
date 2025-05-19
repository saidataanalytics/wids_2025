import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 1. Load test_scores.xlsx
test_scores_path = Path("scripts/analysis/test_scores.xlsx")
test_scores_df = pd.read_excel(test_scores_path)

# 2. Loop over submission files and extract values
submission_dir = Path("output/NEW/Submissions")
females_submission_df = pd.DataFrame()
adhd_submission_df = pd.DataFrame()

for file in submission_dir.glob("*.csv"):
    filename = file.name
    submission_id = filename[:3]  # Keep as string
    
    df = pd.read_csv(file)
    
    # Store counts in main dataframe
    idx = test_scores_df[test_scores_df['ID'] == submission_id].index
    if not idx.empty:
        test_scores_df.loc[idx, 'Nr_ADHD'] = df['ADHD_Outcome'].sum()
        test_scores_df.loc[idx, 'Nr_Females'] = df['Sex_F'].sum()
    
    # Add to submission dataframes
    females_submission_df[submission_id] = df['Sex_F']
    adhd_submission_df[submission_id] = df['ADHD_Outcome']

# Sort columns alphabetically
females_submission_df = females_submission_df[sorted(females_submission_df.columns)]
adhd_submission_df = adhd_submission_df[sorted(adhd_submission_df.columns)]

# 3. Save updated test_scores_df and the two submission dataframes
test_scores_df.to_excel("scripts/analysis/test_scores_filled_in.xlsx", index=False)
females_submission_df.to_excel("scripts/analysis/females_submission_df.xlsx", index=False)
adhd_submission_df.to_excel("scripts/analysis/adhd_submission_df.xlsx", index=False)

# 4. Heatmap (unchanged with full labels)
plt.figure(figsize=(10, 8))
pivot = test_scores_df.pivot_table(index='Nr_Females', columns='Nr_ADHD', values='Score')
sns.heatmap(pivot, cmap='RdBu_r', cbar=True)
plt.title('Score Heatmap')
plt.xlabel('Nr_ADHD')
plt.ylabel('Nr_Females')
plt.tight_layout()
plt.savefig('scripts/analysis/score_heatmap.png')
plt.close()

# 5. 3D plot with horizontal cut and annotated range labels
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = test_scores_df['Nr_Females']
y = test_scores_df['Nr_ADHD']
z = test_scores_df['Score']

ax.scatter(x, y, z, c=z, cmap='RdBu_r', s=50)

ax.set_xlabel('Nr_Females')
ax.set_ylabel('Nr_ADHD')
ax.set_zlabel('Score')
ax.set_title('3D Score Distribution with Score > 0.72 Highlight')

# Add horizontal cut plane at Score = 0.72
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max(), 10))
zz = np.full_like(xx, 0.72)
ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

# Identify high score range and annotate min/max
high_scores = test_scores_df[test_scores_df['Score'] > 0.72]
if not high_scores.empty:
    min_fem = int(high_scores['Nr_Females'].min())
    max_fem = int(high_scores['Nr_Females'].max())
    min_adhd = int(high_scores['Nr_ADHD'].min())
    max_adhd = int(high_scores['Nr_ADHD'].max())
    
    # Annotate near plane edges
    ax.text(min_fem, min_adhd, 0.72, f"Min Fem: {min_fem}\nMin ADHD: {min_adhd}", color='black')
    ax.text(max_fem, max_adhd, 0.72, f"Max Fem: {max_fem}\nMax ADHD: {max_adhd}", color='black')

plt.tight_layout()
plt.savefig('scripts/analysis/score_3d_plot.png')
plt.close()

# 9. Create grid visualisation of adhd_submission_df
plt.figure(figsize=(14, 6))
sns.heatmap(adhd_submission_df.T, cmap='Blues', cbar=False)
plt.title("Grid of ADHD Predictions Across Submissions")
plt.xlabel("Participant Index")
plt.ylabel("Submission ID")
plt.tight_layout()
plt.savefig("scripts/analysis/adhd_prediction_grid.png")
plt.close()
