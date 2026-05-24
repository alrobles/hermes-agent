---
name: sdm-pipeline
description: "Species Distribution Modeling pipeline using R (dismo, maxnet, terra)."
version: 1.0.0
author: EcoSeek
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [ecology, sdm, maxent, species-distribution, niche, modeling, r, terra]
    category: ecology
---

# Species Distribution Modeling (SDM) Pipeline

Build reproducible SDM workflows from occurrence data to habitat suitability maps.

## When to Use

Use this skill when the user asks about:
- Species distribution modeling or habitat suitability
- MaxEnt, GLM, BRT, or random forest models for species
- Predicting species ranges under climate change
- Mapping species occurrence data with environmental variables

## Workflow

### 1. Data Acquisition
```r
library(rgbif)
library(terra)

# Get occurrence records from GBIF
occ <- occ_search(scientificName = "SPECIES_NAME",
                  hasCoordinate = TRUE,
                  limit = 5000)$data

# Clean coordinates
library(CoordinateCleaner)
occ_clean <- clean_coordinates(occ,
  lon = "decimalLongitude", lat = "decimalLatitude",
  species = "species", tests = c("capitals", "centroids",
  "equal", "gbif", "institutions", "seas", "zeros"))
```

### 2. Environmental Data
```r
# WorldClim bioclimatic variables at 2.5 arc-minutes
bioclim <- geodata::worldclim_global(var = "bio", res = 2.5,
                                      path = "data/")
# Crop to study area
study_area <- ext(XMIN, XMAX, YMIN, YMAX)
env <- crop(bioclim, study_area)
```

### 3. Model Fitting
```r
library(maxnet)
library(dismo)

# Prepare presence/background data
presence <- vect(occ_clean, geom = c("decimalLongitude", "decimalLatitude"))
bg <- spatSample(env, 10000, "random", na.rm = TRUE)

# Extract environmental values
pres_env <- extract(env, presence)
bg_env <- as.data.frame(bg)

# Fit MaxEnt via maxnet
mod <- maxnet(p = rep(1, nrow(pres_env)),
              data = rbind(pres_env, bg_env),
              f = maxnet.formula(rep(1, nrow(pres_env)),
                                rbind(pres_env, bg_env),
                                classes = "lqph"))

# Predict suitability
pred <- predict(env, mod, type = "cloglog")
plot(pred, main = "Habitat Suitability")
points(presence, pch = 16, cex = 0.5)
```

### 4. Model Evaluation
```r
# Cross-validation with ENMeval
library(ENMeval)
eval <- ENMevaluate(occs = occ_coords, envs = env,
                     bg = bg_coords,
                     algorithm = "maxnet",
                     partitions = "block",
                     tune.args = list(fc = c("L", "LQ", "LQH"),
                                      rm = 1:3))
# Best model by AICc
best <- eval@results[which.min(eval@results$AICc), ]
```

### 5. Future Projections
```r
# CMIP6 future climate (SSP2-4.5, 2061-2080)
future_clim <- geodata::cmip6_world("ACCESS-CM2", "245",
                                     "2061-2080", var = "bio",
                                     res = 2.5, path = "data/")
future_pred <- predict(crop(future_clim, study_area), mod,
                        type = "cloglog")
```

## Key Packages

| Package | Purpose |
|---------|---------|
| `rgbif` | GBIF occurrence data retrieval |
| `terra` | Raster/vector spatial operations |
| `maxnet` | MaxEnt model fitting |
| `dismo` | Distribution modeling utilities |
| `ENMeval` | Model selection and evaluation |
| `geodata` | WorldClim, CMIP6 climate data |
| `CoordinateCleaner` | Occurrence record cleaning |
| `sf` | Vector spatial data |

## Tips
- Always clean occurrence records before modeling
- Use block cross-validation for spatial data (not random)
- Report AUC, TSS, and AICc for model evaluation
- Document GBIF DOIs for reproducibility
- Consider sampling bias correction (target-group background)
