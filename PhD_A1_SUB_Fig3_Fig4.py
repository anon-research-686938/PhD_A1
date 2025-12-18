##############################################################################
## Python Libraries ##
##############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import math
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib

##############################################################################
## Variable definition ##
##############################################################################

# Choose dataset to work with
study_area = ["CAMPAN","KILPIS"][0]

# General path to access data
path_data = "/data/" 

# Name of the CSV file containing the probability of occurrence vectors
filename = "A1_hourly_probability_CAMPAN.csv"

# Select only 5 subset sizes out of the whole vector
sizes = [20, 40, 60, 80, 100]

##############################################################################
## Open the dataframe with probability of occurrence vectors ##
##############################################################################

# Open the dataset previously saved for every hour bin 
df_hourly = pd.read_csv(path_data+"probability_vectors/"+filename)
print(df_hourly.shape)

# Load the list of species in the study
species_list_final = pd.read_excel(path_data + "A1_species_list.xlsx")
print(species_list_final.shape)

# Extract stabilization predicted by the GAM model 
df_conv = species_list_final.loc[(species_list_final.dataset == study_area) & (species_list_final.selected == 1),["dataset","species","SSCT convergence - 0.5% - SD"]] # Keep only species in selected dataset

# Replace NaN in convergence by 0
df_conv = df_conv.fillna(0)

# Round the stabilization values to the nearest 2 multiple on the subset_size axis
df_conv["convergence"] = ((df_conv["SSCT convergence - 0.5% - SD"] / 2).round() * 2).astype(int)

##############################################################################
## Figure 3: Plot hourly distributions for every species ##
##############################################################################

# Filter df_results by species and subset size
df_crop = df_hourly.loc[df_hourly.subset_size.isin(sizes),:]

# Define the Okabe–Ito color palette
okabe_ito = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]

sns.set_theme(style="whitegrid")

# --- List of species ---
species_list = df_crop['species'].unique()
n_species = len(species_list)

# --- Compute grid size (square-ish layout) ---
n_cols = math.ceil(math.sqrt(n_species))
n_rows = math.ceil(n_species / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharex=True)

# In case axes is 1D
axes = axes.flatten()

# --- Create one subplot per species ---
for i, sp in enumerate(species_list):
    ax = axes[i]

    sns.lineplot(
        data=df_crop[df_crop["species"] == sp],
        x='hour',
        y='expected_abundance',
        hue='subset_size',
        errorbar=None,
        ax=ax,
        palette=okabe_ito
    )

    ax.set_title(sp)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Probability of occurrence")
    ax.grid(True)
    ax.legend(title="Subset size", loc="upper left")

# --- Turn off any empty extra subplots ---
for j in range(n_species, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()

plt.show()

##############################################################################
## Figure 4: Plot Euclidean distance for every species between probability vectors ##
##############################################################################

# --- Parameters ---
df = df_hourly.copy()
df['norm'] = (
    df.groupby(["species"])['expected_abundance']
    .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
)
species_list = sorted(df["species"].unique())

# Turn stabilization values per species into a directory
conv_dict = dict(zip(df_conv["species"], df_conv["convergence"]))

# Shared subset sizes (same for all species)
all_subset_sizes = sorted(df["subset_size"].unique())

# From the number of specie, infer the size of the subplot grid
n_species = len(species_list)
ncols = 4
nrows = int(np.ceil(n_species / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=True)
axes = axes.flatten()

# Create one subplot per species
for i, sp in enumerate(species_list):
    
    ax = axes[i]
    
    df_sp = df[df["species"] == sp]

    # Pivot: subset_size × hour -> expected abundance vector
    mat = df_sp.pivot_table(
        index="subset_size",
        columns="hour",
        values="norm",
        aggfunc="mean"
    ).fillna(0)

    # Compute Euclidean distance between probability vectors
    cos_sim = euclidean_distances(mat.values)
    cos_sim_df = pd.DataFrame(
        cos_sim,
        index=mat.index,
        columns=mat.index
    )
    
    # Plot heatmap
    sns.heatmap(
        cos_sim_df,
        ax=ax,
        cmap="Greys",
        vmin=0,
        vmax=0.3,
        square=True,
        cbar = False
    )
    
    ax.set_title(sp, fontsize=15)
    ax.set_xlabel("Subset size", fontsize=12)
    ax.set_ylabel("Subset size", fontsize=12)

    # Plot dotted lines at the subset_size where SSCT are stabilizing according to the GAM model
    if sp in conv_dict:
        conv = conv_dict[sp]
        # find the index position of the convergence subset_size
        if conv in mat.index:
            pos = list(mat.index).index(conv)
            ax.axvline(pos, color="coral", linestyle="--", linewidth=1.5)
            ax.axhline(pos, color="coral", linestyle="--", linewidth=1.5)

    subset_vals = list(mat.index)
    nvals = len(subset_vals)
    
    # Choose a indexer to plot X and Y ticks every N values
    stride = 5
    
    tick_positions = np.arange(0, nvals, stride) + 0.5
    tick_labels = [subset_vals[j] for j in range(0, nvals, stride)]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Drawing the frame
    ax.axhline(y = 0, color='k',linewidth = 5)
    ax.axhline(y = cos_sim_df.shape[1], color = 'k',
                linewidth = 5)
    
    ax.axvline(x = 0, color = 'k',
                linewidth = 5)
    
    ax.axvline(x = cos_sim_df.shape[0], 
                color = 'k', linewidth = 5)
    
# Remove empty axes if species count < nrows*ncols
for j in range(i+1, nrows*ncols):
    fig.delaxes(axes[j])

plt.subplots_adjust(right=0.88, wspace=0.3, hspace=0.3)

# Add a shared colorbar to every subplot
cbar_ax = fig.add_axes([0.90, 0.15, 0.01, 0.7])
norm = matplotlib.colors.Normalize(
    vmin=0,
    vmax=0.3
)
cbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap="Greys"),
    cax=cbar_ax
)
cbar.set_label('Euclidian Distance', fontsize=16)

plt.show()
