##############################################################################
## Define R Packages ##
##############################################################################

library(glmmTMB)
library(tidyverse)
library(ggplot2)
library(mgcv)
library(DHARMa)
library(patchwork)

##############################################################################
## Variable definition ##
##############################################################################

# General path to access data
path_data <- "/data/"

sensitivity <- 1.5 # Sensitivity value set on BirdNET

##############################################################################
## Organize the data ##
##############################################################################

# Open the results from the bootstrap resampling process
output <- read.csv(paste0(path_data,"A1_bootstrap_resampling.csv"))

# Turn logit scores into confidence values
output_conf <- output %>%
  mutate(
    cutoff_conf = ifelse(
      is.na(cutoff),
      NA,
      exp(sensitivity * cutoff) / (1 + exp(sensitivity * cutoff))
    )
  )

# Get the standard deviation for each subset_zise X species X dataset
var_df <- output_conf %>%
  group_by(species, subset_size, dataset) %>%
  summarise(
    sd_cutoff = sd(cutoff_conf, na.rm = TRUE),
    n_reps = sum(!is.na(cutoff_conf)),
    .groups = "drop"
  )

# Turn categorical variables into factors
var_df$fdataset <- factor(var_df$dataset)
var_df$fspecies <- factor(var_df$species)

##############################################################################
## GAM model for KILPIS data ##
##############################################################################

# Get a dataset only with data from KILPIS
df_LAP <- var_df[var_df$dataset=="KILPIS",]

# Create a GAM statistical model with KILPIS data
mLAP <- gam(
  sd_cutoff ~ s(subset_size, k = 10, by = fspecies) + fspecies,
  family = Gamma(link = "log"),
  data = df_LAP
)

# PLot the model summary
summary(mLAP)

# Generate the DHARMa residuals
resLAP <- simulateResiduals(mLAP, n = 1000)

# QQ plot, residuals VS fitted values
plot(resLAP) 

# Plot residuals versus each covariate, here the subset_size
plotResiduals(resLAP, df_LAP$subset_size, main = "Residuals vs subset_size") # residuals VS covariates

# Plot residuals versus each covariate, here the species
plotResiduals(resLAP, df_LAP$fspecies, main = "Residuals vs species")

##############################################################################
## GAM model for CAMPAN data ##
##############################################################################

# Get a dataset only with data from CAMPAN
df_PYT <- var_df[var_df$dataset=="CAMPAN",]

# Create a GAM statistical model with CAMPAN data
mPYR <- gam(
  sd_cutoff ~ s(subset_size, k = 10, by = fspecies) + fspecies,
  family = Gamma(link = "log"),
  data = df_PYR
)

# PLot the model summary
summary(mPYR)

# Generate the DHARMa residuals
resPYR <- simulateResiduals(mPYR, n = 1000)

# QQ plot, residuals VS fitted values
plot(resPYR) 

# Plot residuals versus each covariate, here the subset_size
plotResiduals(resPYR, df_PYR$subset_size, main = "Residuals vs subset_size") # residuals VS covariates

# Plot residuals versus each covariate, here the species
plotResiduals(resPYR, df_PYR$fspecies, main = "Residuals vs species")

##############################################################################
## Function: Find the stabilization point for each species ##
##############################################################################

find_stabilization_point <- function(subset_size, pred, window = 10, threshold = 0.01) {
  
  # Define the range of values for relative difference computing
  range <- max(pred) - min(pred)
  
  n <- length(pred)

  # Store median relative differences for each sliding window
  med_rel_diff <- numeric(n - window + 1)
  
  for (i in 1:(n - window + 1)) {
    # Values inside the window
    y_win <- pred[i:(i + window - 1)]
    
    # Relative differences (pairwise)
    rel_diff <- abs(diff(y_win) / range)
    
    # Median relative difference
    med_rel_diff[i] <- median(rel_diff, na.rm = TRUE)
  }
  
  # Find first window where stability is achieved
  stable_index <- which(med_rel_diff < threshold)[1]
  
  # If no stabilization point if found within the range, return NA
  if (is.na(stable_index)) {
    return(list(
      stabilization_found = FALSE,
      stabilization_point = NA,
      details = med_rel_diff
    ))
  }
  
  return(list(
    stabilization_found = TRUE,
    stabilization_point = subset_size[stable_index],
    details = med_rel_diff
  ))
}

# Run the stabilization assessment for every species
compute_stability_per_species <- function(df, window = 10, threshold = 0.01) {
  
  df %>%
    group_by(fspecies) %>%
    arrange(subset_size, .by_group = TRUE) %>%  # ensure correctly ordered
    group_modify(~{
      
      res <- find_stabilization_point(
        subset_size = .x$subset_size,
        pred = .x$Pred,
        window = window,
        threshold = threshold
      )
      
      tibble(
        stabilization_found = res$stabilization_found,
        stabilization_point = res$stabilization_point
      )
      
    }) %>%
    ungroup()
}

##############################################################################
## Figure 2: Plot representation of both GAM models ##
##############################################################################

# Define the Okabe–Ito color palette
okabe_ito <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2")

##############################################################################
## Representation of KILPIS ##

# Predict values based on model
pred_LAP <- df_LAP %>%
  group_by(fspecies) %>%
  summarise(
    subset_size = seq(min(subset_size), max(subset_size), length.out = 100),
    .groups = "drop"
  )

P1 <- predict(mLAP, newdata = pred_LAP, type = "link", se.fit = TRUE, re.form = NA)

# Add a confidence interval around the predicted values
pred_LAP <- pred_LAP %>%
  mutate(
    fit_link = P1$fit,
    se_link = P1$se.fit,
    Pred = exp(fit_link),
    selo = exp(fit_link - 1.96 * se_link),
    seup = exp(fit_link + 1.96 * se_link)
  )

# Run the stabilization assessment for KILPIS model
results <- compute_stability_per_species(pred_LAP, window = 15, threshold = 0.005)

# Save the highest stabilization value as an indicator
x_LAP <- max(results[!is.na(results$stabilization_point),"stabilization_point"])

# Define which species to highlight
highlight_species <- c("Willow Tit",
                       "Meadow Pipit",
                       "Eurasian Bullfinch")

# Build a named vector of colors:
#  - highlighted species using okabe–ito colors
#  - other species into shades of grey
all_species <- unique(df_LAP$fspecies)
base_greys <- gray.colors(length(all_species), start = 0.2, end = 0.8)

# Create named color vector
color_map <- setNames(base_greys, all_species)

# Replace colors for highlighted species
color_map[highlight_species] <- okabe_ito[seq_along(highlight_species)]

# plot using the color mapping
p_LAP <- ggplot() +
  geom_point(data = df_LAP, 
             aes(x = subset_size, y = sd_cutoff, color = fspecies)) +
  geom_line(data = pred_LAP, 
            aes(x = subset_size, y = Pred, color = fspecies),
            linewidth = 1.2) +
  geom_ribbon(data = pred_LAP,
              aes(x = subset_size, ymax = seup, ymin = selo, color = fspecies),
              alpha = 0.2, linetype = 0) +
  geom_vline(xintercept = x_LAP,
             linetype = "dashed",
             linewidth = 1.2,
             color = "red") +
  scale_color_manual(values = color_map) +  # <-- use the custom colors
  labs(
    x = "Subset size (number of annotated samples)",
    y = "Standard Deviation of SSCT",
    title = "KILPIS",
    color = "Species - KILPIS"
  ) +
  theme_light(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold")
  )

p_LAP

##############################################################################
## Representation of CAMPAN ##

# Predict values based on model
pred_PYR <- df_PYR %>%
  group_by(fspecies) %>%
  summarise(
    subset_size = seq(min(subset_size), max(subset_size), length.out = 100),
    .groups = "drop"
  )

P1 <- predict(mPYR, newdata = pred_PYR, type = "link", se.fit = TRUE, re.form = NA)

# Add a confidence interval around the predicted values
pred_PYR <- pred_PYR %>%
  mutate(
    fit_link = P1$fit,
    se_link = P1$se.fit,
    Pred = exp(fit_link),
    selo = exp(fit_link - 1.96 * se_link),
    seup = exp(fit_link + 1.96 * se_link)
  )

# Run the stabilization assessment for CAMPAN model
results <- compute_stability_per_species(pred_PYR, window = 15, threshold = 0.005)

# Save the highest stabilization value as an indicator
x_PYR <- max(results[!is.na(results$stabilization_point),"stabilization_point"])

# Define which species to highlight
highlight_species <- c("Eurasian Bullfinch",
                       "Red-billed Chough",
                       "Common Firecrest")

# Build a named vector of colors:
#  - highlighted species using okabe–ito colors
#  - other species into shades of grey
all_species <- unique(df_PYR$fspecies)
base_greys <- gray.colors(length(all_species), start = 0.2, end = 0.8)

# Create named color vector
color_map <- setNames(base_greys, all_species)

# Replace colors for highlighted species
color_map[highlight_species] <- okabe_ito[seq_along(highlight_species)]

# plot using the color mapping
p_PYR <- ggplot() +
  geom_point(data = df_PYR, 
             aes(x = subset_size, y = sd_cutoff, color = fspecies)) +
  geom_line(data = pred_PYR, 
            aes(x = subset_size, y = Pred, color = fspecies),
            linewidth = 1.2) +
  geom_ribbon(data = pred_PYR,
              aes(x = subset_size, ymax = seup, ymin = selo, color = fspecies),
              alpha = 0.2, linetype = 0) +
  geom_vline(xintercept = x_PYR,
             linetype = "dashed",
             linewidth = 1.2,
             color = "red") +
  scale_color_manual(values = color_map) +  # <-- use the custom colors
  labs(
    x = "Subset size (number of annotated samples)",
    y = "Standard Deviation of SSCT",
    title = "CAMPAN",
    color = "Species - CAMPAN"
  ) +
  theme_light(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold")
  )

p_PYR

# Combine both figures together
p <- p_LAP/p_PYR



