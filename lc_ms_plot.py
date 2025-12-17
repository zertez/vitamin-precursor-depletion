# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
# Plot styling
sns.set_context("notebook", font_scale=1)
sns.set_style("white")

MEDIUM_COLORS = {
    "fully supplemented": "#1f77b4",  # Blue
    "w/o B2, B3, B5": "#ff7f0e",  # Orange
    "w/o B3": "#2ca02c",  # Green
    "w/o B5": "#d62728",  # Red
}

# %%
# Load data
lc_ms_cells = pd.read_csv(
    "/Users/marcusdalakerfigenschou/Documents/osteocarcoma_model/osteo_sim/data/cells_nmol_mg_protein_2.csv"
)

# %%
# Data preprocessing
lc_ms_cells["Is_Concentrated"] = lc_ms_cells["Sample Name"].str.contains("Concentrated", case=False)

lc_ms_cells["Medium_Clean"] = (
    lc_ms_cells["Sample Name"]
    .str.replace("Concentrated", "", regex=False)
    .str.replace(" 24h| 72h", "", regex=True)
    .str.strip()
)

lc_ms_cells["Timepoint"] = lc_ms_cells["Days"].map({1: "24h", 3: "72h"})

rename_dict = {
    "Fully supplemented": "fully supplemented",
    "Without Nicotinamide": "w/o B3",
    "Without Pantothenic Acid": "w/o B5",
    "Not supplemented": "w/o B2, B3, B5",
}

lc_ms_cells["Medium_Clean"] = lc_ms_cells["Medium_Clean"].replace(rename_dict)

# Use concentrated samples
data = lc_ms_cells[lc_ms_cells["Is_Concentrated"]].copy()

# %%
# Metabolites
metabolites = ["AceCoA", "CoA", "ATP", "FAD", "NAD"]


# %%
def normalize_to_control(df, metabolite, timepoint, control="fully supplemented"):
    control_mean = df[(df["Medium_Clean"] == control) & (df["Timepoint"] == timepoint)][metabolite].mean()

    df_norm = df.copy()
    df_norm[metabolite] = (df_norm[metabolite] / control_mean) * 100
    return df_norm


# %%
# Plotting function
def plot_metabolite_side_by_side(
    df,
    metabolite,
    dpi_prefix,
    dpi_values=[1200],
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    timepoints = ["24h", "72h"]
    media = df["Medium_Clean"].unique()
    x = np.arange(len(media))
    width = 0.6

    ymax = 0

    for ax, timepoint in zip(axes, timepoints):
        df_norm = normalize_to_control(df, metabolite, timepoint)
        df_tp = df_norm[df_norm["Timepoint"] == timepoint]

        for i, medium in enumerate(media):
            subset = df_tp[df_tp["Medium_Clean"] == medium]
            if subset.empty:
                continue

            mean_val = subset[metabolite].mean()
            se_val = subset[metabolite].sem()
            ymax = max(ymax, mean_val + se_val)

            ax.bar(
                x[i],
                mean_val,
                width,
                yerr=se_val,
                color=MEDIUM_COLORS.get(medium, "#808080"),
                edgecolor="black",
                linewidth=1,
                alpha=0.7,
                capsize=3,
                error_kw={"linewidth": 1.5},
            )

            # Technical replicates
            for _, row in subset.iterrows():
                ax.plot(
                    x[i],
                    row[metabolite],
                    "o",
                    color="black",
                    markersize=4,
                    alpha=0.6,
                    markeredgewidth=0.5,
                )

        # Panel formatting
        ax.set_xticks(x)
        ax.set_xticklabels(media, rotation=45, ha="right")
        ax.axhline(100, linestyle="--", color="black", linewidth=1, alpha=0.6)

        # Panel label
        ax.text(
            0.02,
            0.95,
            timepoint,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

        # Axis-only frame
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)

    axes[0].set_ylabel(f"{metabolite} (% of fully supplemented)")
    fig.suptitle(metabolite, fontweight="bold", y=1.02)

    for ax in axes:
        ax.set_ylim(0, ymax * 1.25)

    plt.tight_layout()

    for dpi in dpi_values:
        plt.savefig(
            f"{dpi_prefix}_{metabolite}_24h_72h_{dpi}dpi.png",
            dpi=dpi,
            bbox_inches="tight",
        )

    plt.show()


# %%
# Generate all plots
dpi_values = [1200]

for metabolite in metabolites:
    plot_metabolite_side_by_side(
        data,
        metabolite,
        dpi_prefix="concentrated",
        dpi_values=dpi_values,
    )
