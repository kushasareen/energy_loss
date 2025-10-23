import matplotlib.pyplot as plt
import numpy as np

# Data
fractions = [2, 1, 0.1, 0.01, 0.001]
metrics = {
    "mse": [-11.0, -10.3, -7.9, -6.2, -6.0],
    "energy_const": [-12.9, -10.3, -4.6, -4.5, -4.3],
    "energy_exp": [-16.0, -14.8, -5.2, -3.8, -4.1],
    "fape": [-17.6, -16.9, -9.2, -6.7, -6.8],
    "hybrid_const": [-7.4, -7.2, -5.3, -4.2, -4.3],
    "hybrid_exp": [-7.7, -7.4, -5.6, -4.4, -4.3],
}

# Plot
plt.figure(figsize=(8, 6))
for method, values in metrics.items():
    plt.plot(fractions, values, marker='o', label=method)

plt.xscale("log")
plt.xlabel("Dataset Fraction")
plt.ylabel("Metric")
plt.title("Dataset Fraction vs. Metric")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
plt.savefig("images/dataset_fraction_vs_metric.png")
