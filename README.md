# Experimental Codebase

> **Important:** This paper is currently under review. Please do not distribute or share this codebase.

This repository contains the code used for the experiments and analysis presented in the paper.

---

## 1. Indexing

To index the TREC 6/7/8 collection:

```bash
python indexer_trec678rb.py
```

For other collections, import the corresponding dataset type from `definitions`.

---

## 2. Generating Different Query Expansion Methods

The following program generates expanded queries (EQs), primarily using classical query expansion techniques:

```bash
python3 gen_expanded_queries.py dataset-name
```

### HyDE-based Expansion

For HyDE-based expansion terms, we follow the implementation available at:

https://github.com/nourj98/hyde-feedback

### CEQE

For CEQE, we use the following codebase:

https://github.com/sherinaseri/ceqe-release

---

## 3. Correlation Computation

### Correlation with Ideal Queries

The following program generates ideal queries and computes correlations. Modify the parameters to run the experiment on different datasets and with different ideal queries.

```bash
python direct-correlation-idealq.py \
    --dataset msmarco_passage \
    --method dfo \
    --name dfo_trial
```

`mean_corr.py` computes the mean correlation measure across the dataset.

### Generate Only Ideal Queries

If you only want to generate the ideal queries, use:

```bash
python only_ideal_query_construction.py
```

### Correlation Between Separability and AP

To compute the correlation between separability and Average Precision (AP), use:

```bash
python direct-correlation-separability_vs_ap.py
```

---

## 4. Similarity Computation Between Ideal Queries

To compute the similarity between two ideal queries:

```bash
python3 idealq_sim.py dataset-name ideal-query-1 ideal-query-2
```

For example, to compute the similarity between the DFO-based ideal query and `leastsq_1` on the MS MARCO Passage dataset:

```bash
python3 idealq_sim.py msmarco_passage dfo leastsq_1
```


---

## 5. Generating Synthetic Queries

To generate synthetic queries, change the collection and ideal query name inside `synthetic_queries.py` and run:

```bash
python3 synthetic_queries.py
```

---

## 6. Computing AP

The following program truncates the ideal query and computes AP:

```bash
python trunc_expanded_query.py
```

---

## 7. Additional Analysis

### Restricted AP vs. Real AP

To compute the correlation between Restricted AP and Real AP:

```bash
python direct-correlation-ap.py \
    --dataset msmarco_passage
```

### Spread of Expanded Queries and Angular Range

To compute the spread of expanded queries (EQs) and their angle with respect to the ideal queries:

```bash
python angle_range.py \
    --dataset msmarco_passage_v2 \
    --name leastsq_0.1
```

The above program saves the output in the `output` directory.

Pass the resulting file to:

```bash
python narrow_band_corr.py path-to-op-file-of-angle-range
```

This program quantifies BQV/WQV across all benchmark queries.

---

## 8. Spherical Sweep

### Generate Expanded Queries

The following program generates expanded queries from the ideal query by passing through the plane defined by the EQ:

```bash
python3 spherical_sweep_new.py \
    --dataset msmarco_passage \
    --idealq_name leastsq_0.1
```

### 360-Degree Spherical Sweep

To perform a 360-degree sweep from the ideal query to the expanded queries:

```bash
python spherical_sweep.py \
    --dataset trec678rb \
    --idealq_name leastsq_0.1
```

This generates the different spherical sweeps.

### Aggregate Results Across Queries

To aggregate the spherical sweep results across different queries:

```bash
python spherical_sweep_per_dataset.py \
    --dataset trec678rb \
    --idealq_name leastsq_0.1
```

Modify the parameters to run the experiment on different datasets and with different ideal queries.

---

## 9. Output

Several of the analysis programs save their outputs in the `output` directory. Refer to the individual sections above for details on which output files are required as input to subsequent analysis programs.
