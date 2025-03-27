import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the JSON data
with open("output.json", "r") as file:
    json_data = json.load(file)

data = json_data

# Convert to DataFrame
df = pd.DataFrame(data).T

# Calculate accuracy
df["accuracy"] = df["avg_mse"]

# Filter resource_score < 100
df_filtered = df[df["resource_score"] < 100].copy()

# Compute unique identifier for color grouping
df_filtered["group"] = df_filtered["block_size"] * df_filtered["batch_parallelism"]

# Sort by resource_score and accuracy for Pareto front calculation
df_filtered = df_filtered.sort_values(by=["resource_score", "accuracy"], ascending=[True, False])

# Pareto front calculation
pareto_points = []
current_best_acc = np.inf

for _, row in df_filtered.iterrows():
    if row["accuracy"] < current_best_acc:
        pareto_points.append(row)
        current_best_acc = row["accuracy"]

pareto_df = pd.DataFrame(pareto_points)

df_filtered['avg_bw'] = (df_filtered["e_width"] + df_filtered["m_width"] * df_filtered["group"]) / df_filtered["group"]

print(df_filtered['avg_bw'])

# Plot
plt.figure(figsize=(8, 6))
plt.step(pareto_df["resource_score"], pareto_df["accuracy"], linestyle='--', color='red', label="Pareto Front")
scatter = plt.scatter(df_filtered["resource_score"], df_filtered["accuracy"],s=10 * df_filtered['group'],
                      c=df_filtered["avg_bw"], cmap="viridis", edgecolors="black")

# Plot Pareto front

# Labels and legend
plt.xlabel("Resource Score")
plt.ylabel("MSE")
plt.yscale('log')
plt.title("MSE vs. Resource Score")
plt.legend()
plt.colorbar(scatter, label="Average Bitwidth")
plt.grid(True, "major")

# Show plot
plt.show()
