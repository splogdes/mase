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
print(df)

# Calculate accuracy
df["accuracy"] = 1 - df["avg_mse"]

# Filter resource_score < 100
df_filtered = df[df["resource_score"] < 100].copy()

# Compute unique identifier for color grouping
df_filtered["group"] = df_filtered["block_size"] * df_filtered["batch_parallelism"]

# Sort by resource_score and accuracy for Pareto front calculation
df_filtered = df_filtered.sort_values(by=["resource_score", "accuracy"], ascending=[True, False])

# Pareto front: A point is Pareto-optimal if no other point has both higher accuracy and lower resource score
pareto_points = []
current_best_acc = -np.inf

for _, row in df_filtered.iterrows():
    if row["accuracy"] > current_best_acc:
        pareto_points.append(row)
        current_best_acc = row["accuracy"]

pareto_df = pd.DataFrame(pareto_points)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df_filtered["resource_score"], df_filtered["accuracy"],
                       c=df_filtered["group"], cmap="viridis", edgecolors="black", label="Configurations")

# Plot Pareto front
plt.plot(pareto_df["resource_score"], pareto_df["accuracy"], linestyle='--', color='red', label="Pareto Front")

# Labels and legend
plt.xlabel("Resource Score")
plt.ylabel("Accuracy (1 - avg_mse)")
plt.title("Accuracy vs. Resource Score")
plt.legend()
plt.colorbar(scatter, label="Block Size × Batch Parallelism")
plt.grid(True)

# Show plot
plt.show()
