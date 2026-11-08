# ============================================================
# STEP 6A — BUILD MASTER MAPPING INVENTORY FOR ALL MODEL MARKERS
# ============================================================

library(data.table)
library(readxl)


# ============================================================
# 1. FILE PATHS
# ============================================================

root <- "D:/hamas123/MyModels/Model1/data/wheat2000"

model_file <- file.path(
  root,
  "genotype_markers.csv"
)

dart_file <- file.path(
  root,
  "SEQ_SNPs_Extract_iranian_2440samples_102474markers.csv"
)

moesm6_file <- file.path(
  root,
  "41467_2020_18404_MOESM6_ESM.xlsx"
)


cat("\n========================================\n")
cat("STEP 6A — MASTER MARKER INVENTORY\n")
cat("========================================\n")

cat("Model file exists :", file.exists(model_file), "\n")
cat("DArTseq file exists:", file.exists(dart_file), "\n")
cat("MOESM6 exists     :", file.exists(moesm6_file), "\n")

if (!all(
  file.exists(model_file),
  file.exists(dart_file),
  file.exists(moesm6_file)
)) {
  stop("One or more required files are missing.")
}


# ============================================================
# 2. MODEL MARKERS — ALL 17,694
# ============================================================

model_header <- fread(
  model_file,
  nrows = 0
)

model_columns <- names(model_header)

model_markers <- trimws(
  model_columns[-1]
)

master <- data.table(
  MarkerID = model_markers
)

master[, model_order := seq_len(.N)]

cat("\nModel markers:", nrow(master), "\n")


# ============================================================
# 3. RAW DArTSEQ METADATA
#
# Use existing dart_meta if available.
# Otherwise read only metadata columns.
# ============================================================

if (!exists("dart_meta")) {
  
  dart_meta <- fread(
    dart_file,
    skip = "MarkerID,AlleleSequence,SNP,SnpPosition",
    select = c(
      "MarkerID",
      "AlleleSequence",
      "SNP",
      "SnpPosition",
      "CallRate",
      "OneRatioRef",
      "OneRatioSnp",
      "FreqHomRef",
      "FreqHomSnp",
      "FreqHets",
      "AvgCountRef",
      "AvgCountSnp",
      "RepAvg"
    ),
    showProgress = TRUE
  )
}

dart_meta[, MarkerID := trimws(as.character(MarkerID))]

# Check duplicate full IDs
dart_duplicate_ids <- dart_meta[
  ,
  .N,
  by = MarkerID
][N > 1]

cat(
  "Duplicate full MarkerIDs in raw DArTseq:",
  nrow(dart_duplicate_ids),
  "\n"
)


# ============================================================
# 4. MOESM6
# ============================================================

if (!exists("moesm6")) {
  
  moesm6 <- read_excel(
    moesm6_file,
    sheet = "Hexaploid"
  )
}

moesm6_dt <- as.data.table(moesm6)

moesm6_dt[, `Marker ID` :=
            trimws(as.character(`Marker ID`))]

# Check duplicate full IDs
moesm6_duplicate_ids <- moesm6_dt[
  ,
  .N,
  by = `Marker ID`
][N > 1]

cat(
  "Duplicate full MarkerIDs in MOESM6:",
  nrow(moesm6_duplicate_ids),
  "\n"
)


# ============================================================
# 5. PREPARE RAW DArTSEQ INFORMATION
# ============================================================

dart_small <- dart_meta[
  ,
  .(
    MarkerID,
    DArT_AlleleSequence = toupper(
      gsub("\\s+", "", as.character(AlleleSequence))
    ),
    DArT_SNP = as.character(SNP),
    DArT_SnpPosition = suppressWarnings(
      as.integer(SnpPosition)
    ),
    DArT_CallRate = CallRate
  )
]


# ============================================================
# 6. PREPARE MOESM6 INFORMATION
# ============================================================

moesm6_small <- moesm6_dt[
  ,
  .(
    MarkerID = `Marker ID`,
    
    MOESM6_Sequence = toupper(
      gsub("\\s+", "", as.character(Sequence))
    ),
    
    MOESM6_Chr_original =
      as.character(Chromossome),
    
    MOESM6_Position_original =
      suppressWarnings(as.numeric(Position)),
    
    MOESM6_Annotation =
      as.character(Annotation),
    
    MOESM6_VariantImpact =
      as.character(variants_impact),
    
    MOESM6_GeneName =
      as.character(`#GeneName`),
    
    MOESM6_GeneId =
      as.character(GeneId),
    
    MOESM6_BioType =
      as.character(BioType)
  )
]


# ============================================================
# 7. MERGE RAW DArTSEQ INTO ALL MODEL MARKERS
# ============================================================

master <- merge(
  master,
  dart_small,
  by = "MarkerID",
  all.x = TRUE,
  sort = FALSE
)


# ============================================================
# 8. MERGE MOESM6 INTO ALL MODEL MARKERS
# ============================================================

master <- merge(
  master,
  moesm6_small,
  by = "MarkerID",
  all.x = TRUE,
  sort = FALSE
)


# ============================================================
# 9. RESTORE ORIGINAL MODEL MARKER ORDER
# ============================================================

setorder(
  master,
  model_order
)


# ============================================================
# 10. PRESENCE FLAGS
# ============================================================

master[
  ,
  has_raw_DArTseq :=
    !is.na(DArT_AlleleSequence) &
    DArT_AlleleSequence != ""
]

master[
  ,
  has_exact_MOESM6 :=
    !is.na(MOESM6_Position_original) &
    !is.na(MOESM6_Chr_original) &
    MOESM6_Chr_original != ""
]


# ============================================================
# 11. NORMALIZE MOESM6 CHROMOSOME NAME
# ============================================================

master[
  ,
  MOESM6_Chr :=
    sub(
      "^chr",
      "",
      MOESM6_Chr_original
    )
]


# ============================================================
# 12. KEEP BOTH PUBLISHED AND NORMALIZED COORDINATES
#
# Validation showed:
# FASTA/BLAST 1-based coordinate =
# MOESM6 Position + 1
# ============================================================

master[
  ,
  Physical_bp_1based :=
    fifelse(
      has_exact_MOESM6,
      MOESM6_Position_original + 1,
      NA_real_
    )
]


# ============================================================
# 13. EXACT-ID MAPPING STATUS
# ============================================================

master[
  ,
  mapping_status := fifelse(
    has_exact_MOESM6,
    "EXACT_MOESM6_ID",
    
    fifelse(
      has_raw_DArTseq,
      "RAW_SEQUENCE_AVAILABLE_NEEDS_MAPPING",
      "NO_RAW_SEQUENCE_YET"
    )
  )
]


# ============================================================
# 14. SUMMARY
# ============================================================

cat("\n========================================\n")
cat("MASTER INVENTORY SUMMARY\n")
cat("========================================\n")

cat(
  "Total model markers:",
  nrow(master),
  "\n"
)

cat(
  "Raw DArTseq sequence available:",
  sum(master$has_raw_DArTseq),
  "\n"
)

cat(
  "Exact MOESM6 physical mapping:",
  sum(master$has_exact_MOESM6),
  "\n"
)

cat(
  "Exact MOESM6 + raw sequence:",
  sum(
    master$has_exact_MOESM6 &
      master$has_raw_DArTseq
  ),
  "\n"
)

cat(
  "Raw sequence but no exact MOESM6 mapping:",
  sum(
    master$has_raw_DArTseq &
      !master$has_exact_MOESM6
  ),
  "\n"
)

cat(
  "Neither exact MOESM6 nor raw sequence:",
  sum(
    !master$has_exact_MOESM6 &
      !master$has_raw_DArTseq
  ),
  "\n"
)


cat("\nMapping-status counts:\n")

print(
  table(master$mapping_status)
)


# ============================================================
# 15. SAVE MASTER INVENTORY
# ============================================================

inventory_file <- file.path(
  root,
  "reference",
  "DVAGraphGP_MODEL_MARKER_MASTER_INVENTORY.csv"
)

fwrite(
  master,
  inventory_file
)

cat("\nMaster inventory saved:\n")
cat(inventory_file, "\n")


# ============================================================
# 16. SAVE MARKERS THAT STILL REQUIRE MAPPING
# ============================================================

needs_mapping <- master[
  mapping_status != "EXACT_MOESM6_ID"
]

needs_mapping_file <- file.path(
  root,
  "reference",
  "DVAGraphGP_MARKERS_NEEDING_MAPPING.csv"
)

fwrite(
  needs_mapping,
  needs_mapping_file
)

cat("\nMarkers needing further mapping saved:\n")
cat(needs_mapping_file, "\n")


cat("\n========================================\n")
cat("STEP 6A FINISHED\n")
cat("========================================\n")

