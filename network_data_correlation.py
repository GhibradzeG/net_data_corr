import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages

# Set random seed for reproducibility
np.random.seed(42)

# Generate random data
data = {
    'Traffic GB': np.random.uniform(0, 100, 200),
    'System #': np.random.randint(1, 1000, 200),
    'System Downtime': np.random.uniform(0, 24, 200),
    'Lost revenue USD': np.random.uniform(100, 10000, 200)
}

# Create DataFrame
df = pd.DataFrame(data)

# Export DataFrame to network_data.csv
df.to_csv('network_data.csv', index=False, header=False)

# Calculate Pearson's correlation
correlation_matrix = df.corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson's Correlation Matrix")
plt.tight_layout()

# Save correlation matrix to correlation_matrix.pdf
plt.savefig('correlation_matrix.pdf')

# Find the highest correlation
highest_corr = correlation_matrix.unstack().sort_values(ascending=False)
highest_corr = highest_corr[highest_corr != 1.0].head(1)

# Get the names of the two features with the highest correlation
feature1, feature2 = highest_corr.index[0]

# Export the names of two features with highest correlation to highest_correlation.pdf
with PdfPages('highest_correlation.pdf') as pdf:
    plt.figure()
    plt.text(0.5, 0.5, f"Highest Correlation:\n{feature1} and {feature2}", ha='center')
    plt.axis('off')
    pdf.savefig()
    plt.close()

    print('Process finished')