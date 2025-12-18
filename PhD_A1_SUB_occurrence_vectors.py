##############################################################################
## Python Libraries ##
##############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns

##############################################################################
## Variable definition ##
##############################################################################

# Choose dataset to work with
study_area = ["CAMPAN","KILPIS"][1]

# General path to access data
path_data = "/data/" 

# Specific path to BirdNET raw occurrences
path_birdnet = path_data + "raw_detections/A1_birdnet_"+ study_area +".csv"

# Specific path to the list of SSCT computed for each bootstrrap replicate
path_boot = path_data + "A1_bootstrap_resampling.csv"

# Vector for the different subset sizes tested 
subset_sizes = np.arange(20,102,2)

# Filter the species to consider depending on study area 
if study_area == "CAMPAN":
    species_list = final_species = [
        "Tawny Owl", "Dunnock", "Eurasian Bullfinch", "Eurasian Wren",
        "European Robin", "European Goldfinch", "Common Chiffchaff",
        "Eurasian Treecreeper", "Common Firecrest", "Common Buzzard",
        "Eurasian Nuthatch", "Eurasian Green Woodpecker",
        "Black Woodpecker", "Red-billed Chough",
        "Great Spotted Woodpecker"]
elif study_area == "KILPIS":
    species_list =  [
        "Snow Bunting", "Common Raven", "Willow Ptarmigan", "Rock Ptarmigan",
        "Redwing", "Common Redpoll", "Willow Tit",
        "European Greenfinch", "Eurasian Bullfinch", "Ring Ouzel",
        "European Golden-Plover", "Bohemian Waxwing", "Red-throated Loon",
        "Dunnock", "Brambling", "Great Tit",
        "Meadow Pipit"]

# Sensitivity value from BirdNET to link confidence to logit scores
birdnet_sensitivity = 1.5

##############################################################################
## Open and arange datasets ##
##############################################################################

# Open BirdNET dataset
df_birdnet = pd.read_csv(path_birdnet, sep="\t")
print(df_birdnet.shape)

# Open dataset summarizing SSCT for each bootstrap replicate
df_boot = pd.read_csv(path_boot)
print(df_boot.shape)

# Keep only models from the right study_area and for one species at a time
df_boot_local = df_boot.loc[(df_boot.dataset == study_area) & (df_boot.species.isin(species_list)),].copy()

# Remove potential outliers in df_boot_local
df_boot_local = df_boot_local.drop(df_boot_local.loc[np.abs(df_boot_local.cutoff) > 7,:].index)

# Turn logit values into confidence scores for each SSCT
df_boot_local["confidence"] = np.exp(birdnet_sensitivity * df_boot_local["cutoff"]) / (1 + np.exp(birdnet_sensitivity * df_boot_local["cutoff"]))

# Similar for BirdNET, keep only species of interest
df_birdnet_local = df_birdnet.loc[df_birdnet["Common Name"].isin(species_list)]

##############################################################################
## Analyze how BirdNET occurrences react to every SSCT ##
##############################################################################

# Define the results dataframe
results = []

for sp in tqdm(species_list):
    
    # Keep only lines corresponding to the right species in both dataframes
    df_occ = df_birdnet_local.loc[df_birdnet_local["Common Name"]==sp].copy()
    df_boot_small = df_boot_local.loc[df_boot_local.species==sp,].copy()
    
    for s in subset_sizes:
        
        # Extract SSCTs for eact subset size
        conf_thresholds = df_boot_small.loc[df_boot_small.subset_size == s, 'confidence'].values
    
        # Sort the SSCTs for better efficiency
        conf_thresholds = np.sort(conf_thresholds)
    
        # For each occurrence, count how many thresholds it passes, i.e. Confidence >= threshold -> kept
        conf = df_occ['Confidence'].values
        counts = np.searchsorted(conf_thresholds, conf, side='right')
    
        # Probability that an occurrence is kept given the subset size = fraction of thresholds passed
        prob = counts / len(conf_thresholds)
    
        # Combine results
        tmp = df_occ[['Begin File']].copy()
        tmp["species"] = sp
        tmp['subset_size'] = s
        tmp['probability'] = prob
        results.append(tmp)

# Combine all subset sizes together
df_results = pd.concat(results, ignore_index=True)

##############################################################################
## Turn df_results into probability of occurrence vector ##
##############################################################################

# Get different probability vectors deopending on the studied area 
if study_area == "CAMPAN":
    
    # Create a column for hours (time variable) derived from the audio filename
    df_results["hour"] = df_results["Begin File"].str.slice(-10,-8).astype(int)
    
    # Aggregate the occurrences per hour, species and subset_size to get the "probability of occurrence vectors"
    df_hourly = (
        df_results
        .groupby(["species", "subset_size", "hour"])
        .agg(expected_abundance=("probability", "mean"))
        .reset_index())
    print(df_hourly.shape)
    
    # Save df_hourly under csv format for later 
    filename = "A1_hourly_probability_"+study_area+".csv"
    df_hourly.to_csv(path_data+"probability_vectors/"+filename,index=False)

elif study_area == "KILPIS":
    
    # Extract the date information from audio filename
    df_results["date_str"] = df_results["Begin File"].str.slice(9, 17)
    
    # Convert to the date information to a datetime format
    df_results["date"] = pd.to_datetime(df_results["date_str"], format="%Y%m%d", errors="coerce")
    
    # Extract ISO week number from the exact date
    df_results["week_number"] = df_results["date"].dt.isocalendar().week.astype("uint16")
    
    # Drop useless columns
    df_results.drop(columns=["date_str", "date"], inplace=True)
    
    # Aggregate the occurrences per week numbers, species and subset_size to get the "probability of occurrence vectors"
    df_weekly = (
        df_results
        .groupby(["species", "subset_size", "week_number"])
        .agg(expected_abundance=("probability", "mean"))
        .reset_index()
    )
    print(df_weekly.shape)
    
    # Save df_results under csv format for later 
    filename = "A1_weekly_probability_"+study_area+".csv"
    df_weekly.to_csv(path_data+"probability_vectors/"+filename,index=False)

















