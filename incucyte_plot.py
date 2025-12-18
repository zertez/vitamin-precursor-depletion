# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

# Styling
sns.set_context("notebook", font_scale=1.5)
sns.set_style(
    "ticks",
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    },
)

# Experimental limit
MAX_HOURS = 72

# LOAD DATA
df_growth = pd.read_csv(
    "/Users/marcusdalakerfigenschou/Documents/osteocarcoma_model/osteo_sim/data/BIOMED320_2025_growth.csv",
    header=7,
).dropna()

df_growth["Elapsed"] = pd.to_numeric(df_growth["Elapsed"], errors="coerce")
df_growth = df_growth[df_growth["Elapsed"] <= MAX_HOURS].copy()

# Remove columns
df_growth.rename(
    columns={
        "not supplemented": "w/o B2, B3, B5",
        "not supplemented (Std Err Img)": "w/o B2, B3, B5 (Std Err Img)",
        "w/o Nam": "w/o B3",
        "w/o Nam (Std Err Img)": "w/o B3 (Std Err Img)",
        "w/o Pantoth. acid": "w/o B5",
        "w/o Pantoth. acid (Std Err Img)": "w/o B5 (Std Err Img)",
    },
    inplace=True,
)

# Conditions
mediums = ["fully supplemented", "w/o B2, B3, B5", "w/o B3", "w/o B5"]
palette = sns.color_palette("tab10", len(mediums))
colors = dict(zip(mediums, palette))


# Logistic functional
def logistic_growth(x, L, k, x0):
    """
    L  = upper asymptote (max confluence)
    k  = logistic growth rate (intrinsic parameter)
    x0 = inflection point (hours)
    """
    return L / (1 + np.exp(-k * (x - x0)))


# Growth curves with error bars
records = []
for medium in mediums:
    for _, row in df_growth.iterrows():
        records.append(
            {
                "Elapsed": row["Elapsed"],
                "Medium": medium,
                "Confluence": row[medium],
                "Std_Error": row[f"{medium} (Std Err Img)"],
            }
        )

df_long = pd.DataFrame(records)

plt.figure(figsize=(10, 6))

for medium in mediums:
    subset = df_long[df_long["Medium"] == medium]

    sns.scatterplot(
        data=subset,
        x="Elapsed",
        y="Confluence",
        color=colors[medium],
        s=70,
        label=medium,
    )

    sns.lineplot(
        data=subset,
        x="Elapsed",
        y="Confluence",
        color=colors[medium],
        linewidth=2,
    )

    plt.errorbar(
        subset["Elapsed"],
        subset["Confluence"],
        yerr=subset["Std_Error"],
        fmt="none",
        ecolor=colors[medium],
        elinewidth=1.5,
        capsize=3,
        capthick=1.5,
        alpha=0.55,
    )

plt.xlabel("Time (h)")
plt.ylabel("Phase object confluency (%)")
plt.title("Cell Growth Curves by Medium")
# plt.xlim(0, MAX_HOURS) # REMOVED: To prevent clipping
plt.ylim(0, 100)
sns.despine()
plt.tight_layout()
plt.savefig("cell_growth_curves_logistic_k.png", dpi=1200)
# plt.show()


# Logistical fitting
x = df_growth["Elapsed"]
datasets = {m: df_growth[m] for m in mediums}
results = {}

fig, axes = plt.subplots(2, 2, figsize=(15, 13))
axes = axes.flatten()

for i, (label, y) in enumerate(datasets.items()):
    initial_guess = [90, 0.05, 24]  # L, k, x0

    params, _ = curve_fit(
        logistic_growth,
        x,
        y,
        p0=initial_guess,
        maxfev=10000,
    )

    L, k, x0 = params
    results[label] = {"L": L, "k": k, "x0": x0}

    y_fit = logistic_growth(x, L, k, x0)

    sns.scatterplot(
        x=x,
        y=y,
        ax=axes[i],
        color=colors[label],
        s=80,
    )

    sns.lineplot(
        x=x,
        y=y_fit,
        ax=axes[i],
        color=colors[label],
        linestyle="--",
        linewidth=2,
    )

    axes[i].set_title(label)
    axes[i].set_xlabel("Time (h)")
    axes[i].set_ylabel("Confluency (%)")
    # axes[i].set_xlim(0, MAX_HOURS) # REMOVED: To prevent clipping
    axes[i].set_ylim(0, 100)

    # for the LaTeX part to prevent the KeyError when mixing
    # the format function with LaTeX.
    axes[i].text(
        0.05,
        0.95,
        f"k = {k:.4f} $\\mathrm{{h}}^{{-1}}$\nL = {L:.2f}%\n$x_0$ = {x0:.2f} h",
        transform=axes[i].transAxes,
        va="top",
    )

    sns.despine(ax=axes[i])

plt.tight_layout()
plt.savefig("logistic_fits_k_definition.png", dpi=1200)
# plt.show()

# Growth rate barplot
df_rates = pd.DataFrame(
    {
        "Medium": list(results.keys()),
        "Logistic_Growth_Rate_k": [results[m]["k"] for m in results],
    }
)

plt.figure(figsize=(8, 8))
sns.barplot(
    data=df_rates,
    x="Medium",
    y="Logistic_Growth_Rate_k",
    hue="Medium",
    palette=colors,
    edgecolor="black",
    alpha=0.9,
    legend=False, 
)

plt.title("U2OS")
plt.ylabel(r"Growth Rate ($\mathrm{h}^{-1}$)")
plt.xticks(rotation=45, ha="right")
sns.despine()

for idx, row in df_rates.iterrows():
    plt.text(
        idx,
        row["Logistic_Growth_Rate_k"],
        f"{row['Logistic_Growth_Rate_k']:.4f}",
        ha="center",
        va="bottom",
        fontsize=12,
    )

plt.tight_layout()
plt.savefig("logistic_growth_rates_k.png", dpi=1200)
# plt.show()

# Summary table
print("\n" + "=" * 70)
print("LOGISTIC GROWTH RATE SUMMARY (k)")
print("=" * 70)
print(f"{'Medium':<25} {'k (h⁻¹)':<15} {'L (%)':<15} {'x₀ (h)':<15}")
print("-" * 70)

for label in results:
    print(f"{label:<25} {results[label]['k']:<15.4f} {results[label]['L']:<15.2f} {results[label]['x0']:<15.2f}")

print("=" * 70)
