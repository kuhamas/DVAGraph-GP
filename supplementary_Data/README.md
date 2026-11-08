# Supplementary Data Guide

## DVAGraph-GP: Genomic Mapping, Gene Annotation, and Biological Interpretation

### 1. Overview

This directory contains the supplementary datasets supporting the genomic mapping, gene annotation, model-attribution analysis, perturbation evaluation, and downstream biological interpretation of DVAGraph-GP.

The supplementary datasets are organized into three complementary analytical stages:

1. **Genomic Mapping and Gene Annotation (S1–S6):** Establishment and quality control of physical genomic coordinates, mapping eligibility, and gene-level positional annotation for the original model-input markers.
2. **Model Attribution and Perturbation Analysis (S7–S8):** Trait-specific Integrated Gradients (IG) rankings and perturbation-based evaluation of prioritized markers.
3. **Post hoc Biological Interpretation (S9–S11):** Integration of prioritized markers with genomic annotations, genome-wide attribution profiles, published QTL/meta-QTL resources, and the learned latent graph.

All datasets retain the original model MarkerID as the primary identifier, allowing the results to be traced across the different analytical stages.

---

### 2. Supplementary Data Description

#### Group I. Genomic Mapping and Gene Annotation

**Supplementary Data S1 — Complete Genomic Mapping and Gene-Annotation Ledger**

- Records: 17,694 model-input features.
- Columns: 50.

This workbook provides the complete feature-level genomic-mapping ledger. It retains all original model-input features, including those without accepted high-confidence genomic mappings.

The dataset integrates:
- Original MarkerID and model-feature order.
- Physical mapping status, source, and confidence.
- Chromosome assignments and physical genomic coordinates.
- Single-SNP gene-overlap and nearest-gene annotations.
- Composite-component mapping information and genomic intervals.
- Available published marker and gene annotations from MOESM6.

**Purpose:** To provide the comprehensive record linking the original model-input features to their available genomic mapping and annotation information.

**Supplementary Data S2 — Genomic Interpretation-Ready Subset**

- Records: 6,685 features.
- Columns: 50.

This workbook contains the filtered subset of S1 that satisfied the predefined high-confidence genomic-mapping criteria.

The subset comprises:
- 6,248 high-confidence single-SNP mappings.
- 437 composite features with at least one accepted high-confidence component-tag locus.

**Purpose:** To provide the final feature population eligible for genomic localization and downstream gene-level interpretation.

S2 is a filtered subset of S1 rather than an independently generated mapping dataset.

**Supplementary Data S3 — Single-SNP Gene Annotation**

- Records: 6,248 single-SNP mappings.
- Columns: 21.

This workbook documents the gene-level positional annotation of high-confidence individual SNP mappings against the wheat reference GFF3 annotation.

It includes:
- Physical SNP coordinates.
- Overlapping gene identifiers and associated information.
- Nearest annotated genes for intergenic SNPs.
- Gene names, biotypes, and physical distances.
- Gene-annotation classifications.

All overlapping genes are retained, whereas intergenic SNPs are assigned their nearest annotated gene(s), including equal-distance ties where applicable.

**Purpose:** To document the positional gene context of individually mapped SNP markers.

**Supplementary Data S4 — Composite-Tag Mapping and Gene Annotation**

This workbook contains two worksheets representing different analytical levels.

**S4.1 — Composite_Tag_Loci**
- Records: 429 unique accepted component-tag genomic loci.
- Columns: 23.

Contains component-level genomic information, including:
- Component clone identifiers.
- Sequence and mapping identifiers.
- Accepted chromosome assignments.
- Genomic tag intervals.
- Overlapping genes.
- Nearest genes and physical distances.

**S4.2 — Composite_Features**
- Records: 437 composite model-input features.

Contains feature-level information linking original composite MarkerIDs to their accepted component-tag loci and corresponding gene annotations.

**Purpose:** To document the separate mapping and gene-annotation procedure used for multi-component markers.

Composite markers are represented through their constituent genomic tag intervals rather than being assigned artificial individual SNP coordinates.

Note that 429 unique tag loci, 437 composite features, and 440 component occurrences represent different analytical units and should not be used interchangeably.

**Supplementary Data S5 — Sequence-Based Mapping Validation**

This workbook contains two sequence-anchor validation datasets.

**S5.1 — Raw DArTseq Sequence Anchors**
- Records: 25 sequence anchors.
- Full chromosome-and-coordinate concordance: 24/25.

**S5.2 — Consensus-Map-Recovered Sequence Anchors**
- Records: 240 unique sequence anchors.
- Full chromosome-and-coordinate concordance: 233/240.

The validation compares BLAST-derived chromosome assignments and reconstructed SNP coordinates against published MOESM6 genomic positions.

**Purpose:** To evaluate the sequence-recovery and physical-coordinate reconstruction procedures used in the single-SNP mapping workflow, including the positional corroboration required for accepting eligible non-perfect alignments.

**Supplementary Data S6 — Exact-MarkerID Coordinate Consistency Quality Control**

- Records: 5,032 exact full-MarkerID matches.
- Columns: 13.

This workbook documents the comparison between physical coordinates in the final mapping ledger and the corresponding published MOESM6 chromosome and position fields.

The checks include:
- Chromosome agreement.
- Physical-position agreement.
- Coordinate normalization.
- Full-coordinate consistency.

All 5,032 records show chromosome and physical-coordinate concordance after the documented coordinate conversion.

**Purpose:** To document coordinate harmonization and consistency for exact published-marker matches.

This comparison should not be interpreted as an independent validation of BLAST-derived mappings; sequence-anchor validation is documented separately in S5.

---

#### Group II. Model Attribution and Perturbation Analysis

**Supplementary Data S7 — Integrated Gradients Consensus Rankings**

This workbook contains eight worksheets:
- README_Index.
- Top20_Attrib_Perturb.
- Six trait-specific ranking worksheets: TKW, TW, GL, GW, GH, and GP.

Each trait-specific worksheet contains all 17,694 model-input features, ranked according to their aggregated Integrated Gradients attribution values.

The rankings include:
- Mean, median, and standard deviation of absolute IG.
- Consensus marker ranks.
- Cross-fold ranking statistics.
- Top-100 and Top-177 selection frequencies.
- Marker-level genotype characteristics.

The Top20_Attrib_Perturb worksheet summarizes the 20 highest-ranked markers for each trait and includes associated perturbation information.

**Purpose:** To provide the complete trait-specific attribution rankings and cross-fold statistics for the original model-input features.

**Supplementary Data S8 — Perturbation-Based Marker Evaluation**

- Records: 1,062 trait-marker records.
- Markers: 177 per trait.
- Traits: Six.
- Outer-fold evaluations: 30 per marker.

This workbook reports perturbation-based evaluation results for the Top-177 consensus-ranked markers of each trait.

The recorded metrics include:
- Mean absolute prediction change.
- Standard deviation of prediction changes across folds.
- Mean PCC reduction.
- Mean MSE increase.
- Mean MAE increase.

**Purpose:** To provide an additional evaluation of prioritized markers based on changes in model predictions and predictive performance under the implemented perturbation procedure.

---

#### Group III. Post hoc Biological Interpretation

**Supplementary Data S9 — Genomic Annotation of Prioritized Markers**

- Records: 1,062 trait-marker instances.
- Columns: 111.

This workbook integrates the Top-177 trait-specific IG rankings with genomic mapping, gene annotation, and available perturbation information.

It combines:
- Consensus IG rankings.
- Perturbation statistics.
- Physical genomic coordinates.
- Single-SNP and composite-marker classifications.
- Gene-overlap and nearest-gene information.
- Published MOESM6 variant-consequence annotations.

Among the 1,062 trait-marker instances, 377 were eligible for genomic interpretation:
- 357 high-confidence single-SNP instances.
- 20 composite-component tag-locus instances.

The 377 instances include 131 gene-overlap contexts and 246 nearest-gene contexts.

Among the 357 single-SNP instances, 305 had exact full-MarkerID-matched MOESM6 variant-consequence annotations.

**Purpose:** To connect model-prioritized markers with their genomic and gene-level context for downstream biological interpretation.

Importantly, these are trait-marker instances, not 377 unique genomic markers.

**Supplementary Data S10 — Genome-Wide Marker Attribution Profiles**

This workbook contains six trait-specific worksheets, each with 6,161 high-confidence single-SNP markers having chromosome-resolved physical coordinates.

The datasets provide:
- Marker identifiers and genomic coordinates.
- Trait-specific IG attribution values.
- Consensus ranks.
- Cross-fold attribution statistics.
- Genotype characteristics.
- Physical plotting eligibility indicators.
- Information supporting genome-wide attribution visualization.

The Top-177 physically plottable marker counts are:
- TKW: 63.
- TW: 56.
- GL: 73.
- GW: 59.
- GH: 44.
- GP: 58.

**Purpose:** To provide the underlying data for chromosome-resolved genome-wide attribution profiles and the visualization of prioritized markers.

Composite-tag intervals are not represented as individual SNP points in these chromosome-resolved datasets.

**Supplementary Data S11 — External Genomic Concordance and Biological Interpretation of the Learned Latent Graph**

This workbook contains seven worksheets:

1. README.
2. Reference_Sets.
3. Reference_Intervals.
4. TKW_63_Markers.
5. Latent_Hub_Markers.
6. HighestConsensus_Edge_Markers.
7. Latent_Graph_Statistics.

The workbook integrates selected TKW markers with published QTL/meta-QTL intervals and gene-reference sets, and examines marker associations with selected latent nodes and stable graph edges.

It includes:
- 63 high-confidence physically mapped TKW markers.
- Published QTL and meta-QTL reference intervals.
- Published gene-reference collections.
- Marker associations with selected latent nodes.
- Marker associations with selected stable graph edges.
- Chromosome-matched permutation-based interval-overlap analyses.

The statistical assessment includes 40 tests using 10,000 permutations per test. None of the tests remained significant after Holm correction.

**Purpose:** To provide exploratory external genomic concordance analyses and the genomic context of selected latent-node and graph-edge associations.

The observed associations should be interpreted as descriptive or exploratory evidence, not as proof of causal gene effects, molecular interactions, or statistically significant biological enrichment.

---

### 3. Relationship Between the Supplementary Datasets

The datasets follow a sequential analytical structure:

**Stage 1 — Genomic Mapping and Gene Annotation**

S1 → S2 → S3 / S4 → S5 / S6

- S1 retains the complete model-input marker population.
- S2 identifies the high-confidence interpretation-ready subset.
- S3 documents single-SNP gene annotation.
- S4 documents composite-component mapping and gene annotation.
- S5 evaluates sequence-based mapping and coordinate reconstruction.
- S6 documents coordinate consistency for exact MOESM6 matches.

**Stage 2 — Model Attribution and Perturbation**

S7 → S8

- S7 provides complete trait-specific IG consensus rankings.
- S8 evaluates the Top-177 markers through perturbation analysis.

**Stage 3 — Post hoc Biological Interpretation**

S9 / S10 / S11

- S9 integrates prioritized markers with genomic and gene-level annotations.
- S10 provides data for genome-wide attribution profiles.
- S11 investigates external genomic concordance and the biological context of selected latent nodes and graph edges.

Genomic mapping and gene annotation were conducted independently of model fitting and Integrated Gradients calculations. Genomic annotations were linked to the previously established model rankings for post hoc interpretation.

---

### 4. Important Interpretation Notes

The following distinctions should be maintained when using these datasets:

1. **Mapping eligibility is not model importance.** Inclusion in S2 indicates high-confidence genomic interpretation eligibility, not high predictive importance.

2. **Gene overlap and proximity do not establish causality.** Positional annotation provides genomic context but does not demonstrate that a marker affects the associated gene or trait.

3. **Composite markers are not individual SNPs.** Their genomic information is represented through accepted component-tag intervals.

4. **IG attribution is not causal evidence.** Attribution values describe model-associated feature contributions under the implemented attribution framework.

5. **Perturbation and attribution are different analyses.** They provide complementary but non-equivalent evidence.

6. **Latent nodes and graph edges are computational representations.** Their marker associations should not be interpreted as demonstrated molecular interactions.

7. **External interval overlap is not functional validation.** The results in S11 should be interpreted in light of the reference populations, genomic intervals, and multiple-testing procedures.

---

### 5. Reference and Reproducibility Information

The genomic-mapping workflow integrates the following primary resources:

- Original CIMMYT wheat DArTseq marker data.
- Sansaloni et al. hexaploid-wheat marker resource (MOESM6).
- DArT wheat consensus genetic map v4.0.
- Chinese Spring IWGSC RefSeq v1.0 genome assembly.
- Corresponding wheat gene GFF3 annotation.

Physical genomic mapping was performed using NCBI BLAST+ (v2.17.0+), while gene-level annotation used the R packages `GenomicRanges` and `rtracklayer`.

For complete reproducibility, the associated code repository should document the exact reference FASTA and GFF3 releases, software versions, BLAST commands and parameters, and coordinate conventions.

**Reference-assembly compatibility:** The exact GFF3 release used for gene annotation should be explicitly verified against the stated IWGSC RefSeq v1.0 assembly before the repository is considered finalized.

For details of the computational implementation, refer to the corresponding genomic-mapping and model-interpretation scripts in the repository.

---

### 6. Summary

Supplementary Data S1–S11 collectively document the progression from the original model-input markers to genomic localization, gene-level annotation, model attribution, perturbation evaluation, and downstream biological interpretation.

Together, these resources provide a traceable supplementary evidence base for the methodological procedures and biological analyses reported in the DVAGraph-GP manuscript.
