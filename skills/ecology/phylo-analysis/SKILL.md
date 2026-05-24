---
name: phylo-analysis
description: "Phylogenetic analysis and comparative methods with R (ape, picante, phytools)."
version: 1.0.0
author: EcoSeek
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [ecology, phylogenetics, comparative-methods, tree, evolution, r]
    category: ecology
---

# Phylogenetic Analysis and Comparative Methods

Phylogenetic tree manipulation, visualization, and comparative analyses in R.

## When to Use

Use this skill when the user asks about:
- Phylogenetic trees (reading, writing, plotting)
- Phylogenetic diversity or community phylogenetics
- Comparative methods (PGLS, ancestral state reconstruction)
- Trait evolution analyses
- Phylogenetic signal (Pagel's lambda, Blomberg's K)

## Tree Operations

### Reading and Writing Trees
```r
library(ape)

# Read Newick
tree <- read.tree("tree.nwk")

# Read Nexus
tree <- read.nexus("tree.nex")

# Get tree from Open Tree of Life
library(rotl)
taxa <- tnrs_match_names(c("Panthera onca", "Panthera pardus",
                            "Puma concolor", "Acinonyx jubatus"))
tree <- tol_induced_subtree(ott_ids = taxa$ott_id)

# Basic tree info
Ntip(tree)        # number of tips
Nnode(tree)       # number of internal nodes
is.rooted(tree)   # is tree rooted?
is.ultrametric(tree)  # all tips equidistant from root?
```

### Tree Visualization
```r
# Basic plot
plot(tree, type = "phylogram")
add.scale.bar()

# Fan layout
plot(tree, type = "fan", cex = 0.7)

# With phytools for advanced viz
library(phytools)
plotTree(tree, type = "fan", fsize = 0.7)

# Cophyloplot for host-parasite associations
cophyloplot(host_tree, parasite_tree, assoc_matrix)
```

## Community Phylogenetics

### Phylogenetic Diversity
```r
library(picante)

# Phylogenetic diversity (PD) — total branch length
pd_result <- pd(community_matrix, tree, include.root = TRUE)

# Mean pairwise distance (MPD) — average phylo distance
mpd_result <- mpd(community_matrix, cophenetic(tree))

# Mean nearest taxon distance (MNTD)
mntd_result <- mntd(community_matrix, cophenetic(tree))
```

### Null Models (NRI / NTI)
```r
# Net Relatedness Index (standardized MPD)
ses_mpd <- ses.mpd(community_matrix, cophenetic(tree),
                    null.model = "taxa.labels", runs = 999)
# NRI = -1 * ses.mpd.z  (positive = clustered, negative = overdispersed)

# Nearest Taxon Index (standardized MNTD)
ses_mntd <- ses.mntd(community_matrix, cophenetic(tree),
                      null.model = "taxa.labels", runs = 999)
```

## Comparative Methods

### Phylogenetic Signal
```r
library(phytools)

# Blomberg's K
K <- phylosig(tree, trait, method = "K", test = TRUE)

# Pagel's lambda
lambda <- phylosig(tree, trait, method = "lambda", test = TRUE)
```

### PGLS (Phylogenetic Generalized Least Squares)
```r
library(caper)

# Prepare comparative data
comp_data <- comparative.data(tree, data.frame(species, trait1, trait2),
                               names.col = "species")

# Fit PGLS
model <- pgls(trait1 ~ trait2, data = comp_data, lambda = "ML")
summary(model)
```

### Ancestral State Reconstruction
```r
library(phytools)

# Continuous trait
anc <- fastAnc(tree, trait)
contMap(tree, trait)

# Discrete trait
fit <- fitMk(tree, discrete_trait, model = "ARD")
plotTree(tree)
nodelabels(pie = fit$lik.anc, piecol = c("blue", "red"))
```

## Key Packages

| Package | Purpose |
|---------|---------|
| `ape` | Core tree manipulation, I/O, basic analyses |
| `phytools` | Comparative methods, visualization |
| `picante` | Community phylogenetics (PD, MPD, NRI) |
| `caper` | PGLS and comparative data |
| `rotl` | Open Tree of Life API |
| `ggtree` | ggplot2-based tree visualization |
| `treeio` | Tree I/O for multiple formats |
| `diversitree` | Diversification models (BiSSE, MuSSE) |

## Tips
- Always check if the tree is ultrametric before time-based analyses
- Use `drop.tip()` to prune taxa not in your community matrix
- Community matrices: rows = sites, columns = species, values = abundance/presence
- Match species names between tree tips and community matrix exactly
- For large trees (>1000 tips), `ggtree` renders faster than base `plot.phylo`
