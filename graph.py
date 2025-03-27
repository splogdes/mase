import pandas as pd
import matplotlib.pyplot as plt
import json

# read thor.csv (seed, time)
df = pd.read_csv("thor2.csv")

# read pareto.json (seed : params)
with open("pareto.json", "r") as f:
    pareto = json.load(f)

# Merge pareto and thor on seed
df["seed"] = df["seed"].astype(int)
df = df.merge(pd.DataFrame(pareto).T, left_on="seed", right_on="seed")

num_blocks = (128 * 128) / (df["block_size"] * df["batch_parallelism"])
bits_elements = (128 * 128) * df['m_width'] + num_blocks * df['e_width']
total_bytes = bits_elements / 8
df["throuput"] = total_bytes / df["time"]
print(df)

# Calculate the color metric
df["color_metric"] = df["block_size"] * df["batch_parallelism"]

# Map color_metric to discrete categories (powers of 2)
unique_values = sorted(df["color_metric"].unique())  # Get unique values of color_metric
value_to_category = {v: i for i, v in enumerate(unique_values)}  # Map each unique value to a category
df["color_group"] = df["color_metric"].map(value_to_category)  # Assign categories to color_metric

# Plot Resource Score vs Time with discrete color coding and legend
plt.figure()
for value, category in value_to_category.items():
    subset = df[df["color_group"] == category]
    plt.scatter(
        subset["resource_score"],
        subset["time"],
        label=f"# Elements (k): {int(value)}"
    )
    # Add seed labels to each point
    # for _, row in subset.iterrows():
    #     plt.text(row["resource_score"], row["time"], str(row["seed"]), fontsize=8)

# Add labels, title, and legend
plt.xlabel("Resource Score")
plt.ylabel("Time (us)")
plt.title("Resource Score vs Time (us)")
plt.legend(loc="best")
plt.grid(True)

# Plot Resource Score vs Accuracy with discrete color coding and legend
plt.figure()
for value, category in value_to_category.items():
    subset = df[df["color_group"] == category]
    plt.scatter(
        subset["resource_score"],
        1 - subset["avg_mse"],
        label=f"# Elements (k): {int(value)}"
    )
    # Add seed labels to each point
    for _, row in subset.iterrows():
        plt.text(row["resource_score"], 1 - row["avg_mse"], str(row["seed"]), fontsize=8)

# Add labels, title, and legend
plt.xlabel("Resource Score")
plt.ylabel("Accuracy")
plt.title("Resource Score vs Accuracy")
plt.legend(loc="best")

plt.show()

