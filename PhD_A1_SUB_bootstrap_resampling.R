##############################################################################
## Define R Packages ##
##############################################################################

library(readxl)
library(tidyverse)

##############################################################################
## Variable definition ##
##############################################################################

# General path to access data
path_data <- "/data/"

# Path to annotation tables from both study areas
path_annot_LAP <- paste0(path_data,"validation_table_KILPIS.csv") 
path_annot_PYR <- paste0(path_data,"validation_table_CAMPAN.csv")

pr_thresholds = c(0.9) # Probability threshold to define SSCTs

sensitivity = 1.5 # BirdNET sensitivity parameter

subset_sizes <- seq(10, 100, by = 2) # Range of sizes for bootstrap replicates

n_boot <- 3000  # Number of bootstrap replicates per subset size

##############################################################################
## Open and organize datasets ##
##############################################################################

# Open the general species list
list_sp <- read_excel(paste0(path_data,"A1_species_list.xlsx"))

# Create a species column in common for list_df and df_annot
list_sp$species_u <- gsub(" ","_",list_sp$species)

# Extract species list from list_sp by removing species with bad annotations
list_sp_filt <- list_sp %>% filter(selected==1)

# Open the validation datasets for each location
df_annot_LAP <- read.csv(path_annot_LAP)
df_annot_PYR <- read.csv(path_annot_PYR)

##############################################################################
## Run the bootstrap resampling for both datasets ##
##############################################################################

# Empty list to store results
results <- list()

for (i in 1:nrow(list_sp_filt)){
  message("Processing ", list_sp_filt$species[i], " in ",list_sp_filt$dataset[i])
  
  # Open the righ annotation folder depending on the species and dataset
  if (list_sp_filt$dataset[i]=="CAMPAN"){
    df_annot <- df_annot_PYR %>% filter(species==list_sp_filt$species_u[i])
  } else if (list_sp_filt$dataset[i]=="KILPIS"){
    df_annot <- df_annot_LAP %>% filter(species==list_sp_filt$species_u[i])
  }
  
  #############################################################################
  ## Add a confidence and a logit score column to df_annot 
  
  df_annot[df_annot$validation == "U", "validation"] = "F" # Turn "Unsure" labels into "False"
  df_annot$validation = as.integer(as.logical(df_annot$validation)) # Convert TRue/False to binary variables
  
  # Extract confidence score
  df_annot$conf_digits <- str_extract(df_annot$filename, "_conf(\\d{3,4})") %>% 
    str_remove("_conf")
  df_annot$confidence <- as.numeric(df_annot$conf_digits) / (10 ^ nchar(df_annot$conf_digits))
  
  # Transform to logit
  df_annot$logit = log(df_annot$confidence / (1 - df_annot$confidence)) / sensitivity
  
  #############################################################################
  
  set.seed(123)  # Set a fixed seed for reproducibility
  
  for (k in subset_sizes) {
    for (b in 1:n_boot) {
      # Bootstrap sample (with replacement)
      boot_dt <- df_annot[sample(nrow(df_annot), size = k, replace = TRUE), ]
      
      cutoff90.l <- NA
      intercept <- NA
      beta <- NA
      
      # Check degenerate cases
      # If only True Positives, set SSCT to the minimum confidence in the bunch
      if (all(boot_dt$validation == 1)) {
        cutoff90.l <- min(boot_dt$logit, na.rm = TRUE)
      } 
      # If only False Positives, set SSCT to the maximum confidence in the bunch
      else if (all(boot_dt$validation == 0)) {
        cutoff90.l <- max(boot_dt$logit, na.rm = TRUE)
      } 
      # Otherwise, create a logictic regression model
      else {
        # Try fitting model
        logit.model <- tryCatch(
          glm(validation ~ logit, boot_dt, family = 'binomial'),
          error = function(e) NULL
        )
        
        # Extract parameters from the model
        if (!is.null(logit.model)) {
          intercept <- logit.model$coefficients[1]
          beta <- logit.model$coefficients[2]
          
          if (!is.na(beta) && beta > 0) {
            cutoff90.l <- (log(pr_thresholds[1] / (1 - pr_thresholds[1])) - intercept) / beta
          } else {
            cutoff90.l <- NA
          }
        }
      }
      
      # Save results
      results[[length(results) + 1]] <- data.frame(
        species = list_sp_filt$species[i],
        dataset = list_sp_filt$dataset[i],
        subset_size = k,
        bootstrap_id = b,
        intercept = intercept,
        beta = beta,
        cutoff = cutoff90.l
      )
    }
  }
}

output <- bind_rows(results)

# Save results
output_path = paste0("A1_bootstrap_resampling.csv")
#write.csv(output, output_path, row.names = FALSE)
