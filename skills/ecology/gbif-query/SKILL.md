---
name: gbif-query
description: "Query GBIF biodiversity data using rgbif (R) or pygbif (Python)."
version: 1.0.0
author: EcoSeek
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ecology, gbif, biodiversity, species, occurrences, taxonomy, data]
    category: ecology
---

# GBIF Biodiversity Data Query

Retrieve and analyze biodiversity occurrence data from the Global Biodiversity
Information Facility (GBIF).

## When to Use

Use this skill when the user asks about:
- Species occurrence records or distribution data
- GBIF downloads or biodiversity databases
- Taxonomic name resolution (species, genus, family)
- Occurrence data for a specific region or country
- Biodiversity inventories or species checklists

## R Workflow (rgbif)

### Taxonomy Lookup
```r
library(rgbif)

# Resolve a species name to GBIF taxonomy
name_backbone("Panthera onca")
# Returns: usageKey, scientificName, rank, status, matchType

# Fuzzy matching
name_suggest("Pantera onca")  # handles typos
```

### Occurrence Search
```r
# Search occurrences (up to 100,000 via API, more via download)
occ <- occ_search(
  scientificName = "Panthera onca",
  hasCoordinate = TRUE,
  country = "MX",         # ISO 3166-1 alpha-2
  limit = 5000,
  fields = c("species", "decimalLatitude", "decimalLongitude",
             "year", "basisOfRecord", "datasetKey")
)
df <- occ$data

# Count by year
table(df$year)
```

### Large Downloads (for >100K records)
```r
# Requires GBIF account: usethis::edit_r_environ()
# GBIF_USER=your_username
# GBIF_PWD=your_password
# GBIF_EMAIL=your_email

download_key <- occ_download(
  pred("taxonKey", 2435099),           # Panthera onca
  pred("hasCoordinate", TRUE),
  pred_gte("year", 2000),
  format = "SIMPLE_CSV"
)

# Wait for download to complete
occ_download_wait(download_key)
d <- occ_download_get(download_key, path = "data/") |>
     occ_download_import()

# IMPORTANT: cite the DOI
occ_download_meta(download_key)$doi
```

### Species Checklists
```r
# Species in a country
checklist <- name_usage(
  datasetKey = "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c",  # GBIF backbone
  rank = "SPECIES",
  limit = 500
)

# Occurrences by dataset
occ_count(taxonKey = 2435099, country = "CR")
```

## Python Workflow (pygbif)

### Installation
```bash
pip install pygbif
```

### Basic Usage
```python
from pygbif import species, occurrences

# Taxonomy
result = species.name_backbone(name="Panthera onca")
taxon_key = result["usageKey"]

# Occurrences
occ = occurrences.search(
    taxonKey=taxon_key,
    hasCoordinate=True,
    country="MX",
    limit=300
)
records = occ["results"]

# Extract coordinates
coords = [(r["decimalLongitude"], r["decimalLatitude"])
          for r in records
          if "decimalLongitude" in r]
```

## Data Quality

Always clean occurrence data before analysis:
1. Remove records without coordinates
2. Flag suspicious coordinates (capitals, centroids, institutions)
3. Remove duplicate records (same species + location + date)
4. Check for taxonomic synonyms
5. Filter by basis of record (PRESERVED_SPECIMEN, HUMAN_OBSERVATION, etc.)

```r
library(CoordinateCleaner)
cleaned <- clean_coordinates(df,
  lon = "decimalLongitude", lat = "decimalLatitude",
  species = "species",
  tests = c("capitals", "centroids", "equal",
            "gbif", "institutions", "seas", "zeros"))
```

## Key APIs

| Endpoint | Purpose |
|----------|---------|
| `occ_search()` | Search occurrences (up to 100K) |
| `occ_download()` | Async download (any size, needs account) |
| `name_backbone()` | Resolve name to GBIF taxonomy |
| `name_suggest()` | Fuzzy name search |
| `occ_count()` | Count records by filters |
| `dataset_search()` | Find datasets |

## Tips
- Always cite GBIF downloads with their DOI
- Use `hasCoordinate = TRUE` for spatial analyses
- Prefer `occ_download()` for datasets >10K records
- Check `basisOfRecord` — museum specimens vs. observations have different biases
- Use WoRMS (`worrms` R package) for marine species taxonomy
