---
name: niche-modeling
description: "Ecological niche modeling — ENM theory, variable selection, model comparison."
version: 1.0.0
author: EcoSeek
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ecology, niche, modeling, enm, bioclim, variable-selection, comparison]
    category: ecology
---

# Ecological Niche Modeling (ENM)

Theory, variable selection, model comparison, and best practices for ecological
niche models.

## When to Use

Use this skill when the user asks about:
- Ecological niche theory (Grinnellian vs Eltonian niche)
- Variable selection for ENMs
- Comparing modeling algorithms (MaxEnt, GLM, BRT, RF)
- Model evaluation metrics (AUC, TSS, AICc)
- Transferability and projection caveats
- Sampling bias correction

## Environmental Variables

### Bioclimatic Variables (WorldClim)

| Code | Variable | Collinearity Risk |
|------|----------|-------------------|
| BIO1 | Annual Mean Temperature | High (corr with BIO5, BIO6) |
| BIO2 | Mean Diurnal Range | Low |
| BIO4 | Temperature Seasonality | Moderate |
| BIO5 | Max Temp of Warmest Month | High |
| BIO12 | Annual Precipitation | High (corr with BIO13, BIO16) |
| BIO14 | Precipitation of Driest Month | Moderate |
| BIO15 | Precipitation Seasonality | Low |

### Variable Selection Protocol
```r
library(terra)
library(usdm)

# 1. Extract values at occurrence points
env_vals <- extract(env_stack, occ_points)

# 2. Check collinearity (VIF)
vif_result <- vifstep(env_vals, th = 10)

# 3. Or use Pearson correlation threshold
v <- vifcor(env_vals, th = 0.7)

# 4. Selected variables
selected_vars <- env_stack[[v@results$Variables]]
```

## Algorithm Comparison

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **MaxEnt** | Works with presence-only; regularization; feature classes | Background selection matters; can overfit | Most ENM studies; limited absence data |
| **GLM** | Interpretable; presence-absence | Needs true absences; assumes linearity | Well-sampled regions |
| **BRT** | Handles interactions; non-linear | Needs presence-absence; overfitting risk | Complex response surfaces |
| **Random Forest** | Robust; handles interactions | Poor extrapolation; needs absences | Classification-based studies |
| **Bioclim** | Simple; envelope approach | No interactions; poor with few variables | Quick exploration; teaching |

### Ensemble Modeling
```r
library(biomod2)

# Format data
bm_data <- BIOMOD_FormatingData(
  resp.var = occ_pa,
  expl.var = env_stack,
  resp.xy = occ_coords,
  resp.name = "species"
)

# Define models to run
bm_options <- BIOMOD_ModelingOptions()

# Run ensemble
bm_models <- BIOMOD_Modeling(
  bm_data, models = c("GLM", "GBM", "RF", "MAXENT"),
  bm_options, nb.rep = 3, data.split.perc = 80,
  metric.eval = c("TSS", "ROC")
)

# Ensemble forecast
bm_ensemble <- BIOMOD_EnsembleModeling(
  bm_models, em.by = "all",
  metric.select = c("TSS"), metric.select.thresh = c(0.7)
)
```

## Model Evaluation

### Metrics
```r
# True Skill Statistic (TSS) — threshold-dependent
# TSS = Sensitivity + Specificity - 1
# Range: -1 to 1 (>0.5 good, >0.7 very good)

# AUC — threshold-independent
# Range: 0.5 (random) to 1.0 (perfect)
# Caution: inflated with large study areas

# AICc — information criterion for model selection
# Lower = better; penalizes complexity
```

### Spatial Cross-Validation
```r
library(ENMeval)

# Block cross-validation (accounts for spatial autocorrelation)
eval <- ENMevaluate(
  occs = occ_coords, envs = env_stack,
  bg = bg_coords, algorithm = "maxnet",
  partitions = "block",       # NOT random — spatial blocks
  tune.args = list(
    fc = c("L", "LQ", "LQH", "LQHP"),
    rm = seq(0.5, 4, 0.5)
  )
)

# Best model
best <- eval@results[which.min(eval@results$delta.AICc), ]
```

## Common Pitfalls

1. **Spatial autocorrelation**: Use block CV, not random splits
2. **Sampling bias**: Correct with target-group background or bias files
3. **Collinearity**: Always check VIF (<10) or correlation (<0.7)
4. **Study area extent**: Too large inflates AUC; use ecologically relevant extent
5. **Threshold selection**: Report sensitivity + specificity, or use maxTSS
6. **Temporal mismatch**: Match occurrence dates to climate period
7. **Novel climates**: Flag extrapolation areas in future projections (MESS maps)

```r
# MESS — Multivariate Environmental Similarity Surface
library(dismo)
mess_map <- mess(future_env, reference_env_vals)
plot(mess_map)
# Negative values = novel climate combinations
```

## Tips
- Start with 4-6 uncorrelated bioclimatic variables
- Report model settings (regularization multiplier, feature classes)
- Cite GBIF download DOIs for reproducibility
- Use `kuenm` package for comprehensive MaxEnt tuning
- Consider `xsdm` for demographic approaches to SDM (stochastic growth rates)
