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
from matplotlib import gridspec

##############################################################################
## Variable definition ##
##############################################################################

# Choose dataset to work with
study_area = ["CAMPAN","KILPIS"][1]

# General path to access data
path_data = "/data/" 

# Name of the CSV file containing the probability of occurrence vectors
filename = "A1_weekly_probability_KILPIS.csv"

##############################################################################
## Open the dataframe with probability of occurrence vectors ##
##############################################################################

# Open the dataset previously saved for every hour bin 
df_weekly = pd.read_csv(path_data+"probability_vectors/"+filename)
print(df_weekly.shape)

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
## Figure S7: Plot Euclidean distance for every species between probability vectors ##
##############################################################################

# --- Parameters ---
df = df_weekly.copy()
df['norm'] = (
    df.groupby(["species"])['expected_abundance']
    .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
)
species_list = sorted(df["species"].unique())

# Turn stabilization values per species into a directory
conv_dict = dict(zip(df_conv["species"], df_conv["convergence"]))

# Shared subset sizes (same for all species)
all_subset_sizes = sorted(df["subset_size"].unique())

# From the number of species, infer the size of the subplot grid
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
        columns="week_number",
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

##############################################################################
## Retrieve the weeks of presence and absence for every species ##
##############################################################################

# Convert months to weeks 
def get_absence_weeks(arrival_month, departure_month):
    
    # Calculate approx week number for the "start" of the month
    def month_to_start_week(month):
        if month == 1:
            return 1
        # Simple cumulative approximation
        return int(np.ceil((month - 1) * (52 / 12))) + 1

    # Absence starts *after* departure month ends, and ends *before* arrival month starts.
    # Start of absence: Week after departure month's end (approx start of month + 1)
    absence_start_week = month_to_start_week(departure_month) + 4 # approx end of departure month + 1
    
    # End of absence: Week before arrival month's start
    absence_end_week = month_to_start_week(arrival_month) - 1 # approx end of month - 1
    
    # Adjust for wrap-around (absence from Oct to Mar is Oct-Dec and Jan-Feb)
    if arrival_month <= departure_month:
        # Absence is across the year boundary
        absent_weeks_1 = list(range(absence_start_week, 53)) # e.g., 40 to 52
        absent_weeks_2 = list(range(1, absence_end_week + 1)) # e.g., 1 to 10
        return set(absent_weeks_1 + absent_weeks_2)
    else:
        # Absence is within the year (e.g., May to Aug is May, Jun, Jul)
        # Note: This scenario is less common for migration in the northern hemisphere
        # but could represent a non-breeding absence period.
        return set(range(absence_start_week, absence_end_week + 1))

# Keep only species with a migratory behaviour
df_migration = species_list_final.loc[(species_list_final.dataset == "KILPIS") & (species_list_final.arrival_month.notna()) & (species_list_final.selected == 1),]

# Process the migratory data to get absence week sets
migratory_absence = {}
for index, row in df_migration.iterrows():
    migratory_absence[row['species']] = get_absence_weeks(
        row['arrival_month'], row['departure_month']
    )

# The result in migratory_absence is a dictionary mapping species name to a set of week numbers
# e.g., {'Species A': {41, 42, ..., 52, 1, 2, ..., 10}, ...}



##############################################################################
## Figure 5: Plot probability vectors across seasonal patterns ##
##############################################################################

subset_sizes = all_subset_sizes
# Species considered as migrating in KILPIS
species_list_to_plot = [
    "Snow Bunting","Redwing","Ring Ouzel",
    "European Golden-Plover","Red-throated Loon",
    "Dunnock", "Brambling","Meadow Pipit"
]

for sp in species_list_to_plot:
    # Prepare Heatmap Data
    df = df_weekly.loc[df_weekly.species == sp]
    df['norm'] = (
        df.groupby(["species"])['expected_abundance']
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    )
    
    all_weeks = range(1, 53)

    heatmap_data_full = df.pivot_table(
        index='subset_size', columns='week_number', values='norm', aggfunc='mean'
    ).reindex(index=subset_sizes, columns=all_weeks)

    # Keep only weeks that have data
    valid_weeks_mask = heatmap_data_full.notna().any(axis=0)
    heatmap_data = heatmap_data_full.loc[:, valid_weeks_mask]
    plotted_week_numbers = heatmap_data.columns.tolist()

    # Identify continuous week blocks
    diffs = np.diff(plotted_week_numbers)
    split_points = np.where(diffs > 1)[0] + 1
    segments = np.split(plotted_week_numbers, split_points)

    # Prepare figure with variable-width subplots
    widths = [len(seg) for seg in segments]  # proportional to number of weeks
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(
        nrows=1, ncols=len(segments),
        width_ratios=widths,
        wspace=0.05  # small horizontal space between boxes
    )

    vmin = np.nanmin(heatmap_data.values)
    vmax = np.nanmax(heatmap_data.values)

    axes = []
    absence_weeks = migratory_absence.get(sp, set())

    # Find the week boundaries when the species starts or ends migrating
    
    if absence_weeks:
        # Sort to ensure correct ordering
        sorted_abs = sorted(absence_weeks)
    
        # Find discontinuities in the absence week list
        diffs = np.diff(sorted_abs)
        gap_indices = np.where(diffs > 1)[0]
    
        # Default: no gap (continuous absence) - species doesn't migrate
        migration_boundaries = []
    
        if len(gap_indices) == 1:
            end_abs = sorted_abs[gap_indices[0]]      # e.g., 13
            start_abs = sorted_abs[gap_indices[0] + 1]  # e.g., 40
    
            # Boundaries just outside the absence period
            migration_boundaries = [end_abs + 1, start_abs - 1]
        elif len(gap_indices) > 1:
            gap_sizes = diffs[gap_indices]
            main_gap_idx = gap_indices[np.argmax(gap_sizes)]
            end_abs = sorted_abs[main_gap_idx]
            start_abs = sorted_abs[main_gap_idx + 1]
            migration_boundaries = [end_abs + 1, start_abs - 1]
        else:
            migration_boundaries = []
    
    else:
        migration_boundaries = []

    for i, segment_weeks in enumerate(segments):
        ax = fig.add_subplot(gs[i])
        axes.append(ax)
        seg_data = heatmap_data.loc[:, segment_weeks]

        sns.heatmap(
            seg_data,
            cmap='binary',
            vmin=vmin, vmax=vmax,   # shared color scale
            cbar=False,
            annot=False,
            linewidths=0.5,
            ax=ax
        )

        # Add an overlay for migration times - time when species is absent 
        for j, week_num in enumerate(segment_weeks):
            if week_num in absence_weeks:
                ax.axvspan(j, j + 1, color='red', alpha=0.10, zorder=2)

        # Draw migration time boundaries if within the recorded times
        if migration_boundaries:
            for i, boundary in enumerate(migration_boundaries):
                if i==0: shift = 0
                else: shift = 1

                # Find which subplot contains this week number
                for ax, segment_weeks in zip(axes, segments):
                    if boundary in segment_weeks:
                        # Compute its x position (index + 0.5 for alignment)
                        x_pos = list(segment_weeks).index(boundary) + shift
                        ax.axvline(x=x_pos, color='red', linestyle='--', linewidth=2.2, zorder=5)

        if i > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.set_yticks([])

        # Clean up x labels – only show tick labels for each segment, no xlabel
        ax.set_xticks(np.arange(len(segment_weeks)) + 0.5)
        ax.set_xticklabels(segment_weeks, rotation=45)
        ax.set_xlabel("")

        # Draw a border around each subplot, like ggplot2 theme_light()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)


    # Add labels to both axis
    fig.text(0.5, 0.04, 'Week number', ha='center', fontsize=14)
    axes[0].set_ylabel('Subset size', fontsize=14)

    # Create the colorbar
    cbar = fig.colorbar(
        axes[0].collections[0], 
        ax=axes, 
        orientation='vertical',
        fraction=0.015, 
        pad=0.02
    )
    cbar.set_label('Normalized Probability', fontsize=12)

    # Add a title
    fig.suptitle(f"{sp} – Weekly probability distributions per subset size", fontsize=16, y=0.95)

    plt.show()

















