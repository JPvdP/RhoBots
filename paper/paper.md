---
title: 'Rhobots: BERTopic-Style Topic Modeling in Pure R'
tags:
  - R
  - topic modeling
  - natural language processing
  - BERTopic
  - text mining
  - computational social science
authors:
  - name: J.P.G. van der Pol
    orcid: 0000-0002-9686-9637
    affiliation: 1
affiliations:
  - name: Faculty of Geosciences, Utrecht University, The Netherlands
    index: 1
date: 26 June 2026
bibliography: rhobots.bib
---

# Summary

`Rhobots` is an R package that implements the BERTopic pipeline
[@grootendorst2022bertopic] — transformer-based sentence embedding, UMAP
dimensionality reduction [@mcinnes2018umap], HDBSCAN density clustering
[@campello2013hdbscan], and class-based TF-IDF topic extraction — without
any dependency on Python, conda, or `reticulate`. Every stage runs natively
in R through `torch`, `safetensors`, `tok`, `uwot`, and `dbscan`. The
package mirrors the accessor API of the original Python package, adds
integrated quality metrics and hyperparameter search tools, and introduces
part-of-speech (POS) filtered and C-value-ranked representation models
[@frantzi2000cvalue] that give researchers explicit control over *which
grammatical layer of language* defines each topic.

# Statement of Need

## Removing the Python dependency

BERTopic is among the most widely used neural topic models in computational
social science, digital humanities, and scientometrics. Its exclusive
availability as a Python library creates a substantial barrier for R
practitioners: a working Python installation, a compatible conda or virtual
environment, and the `reticulate` bridge are all required. This dependency
chain is version-sensitive, platform-dependent, difficult to maintain across
operating systems, and a recurring obstacle in teaching, automated pipelines,
and reproducible research workflows. `Rhobots` removes that barrier entirely.
Because every component is implemented through CRAN packages, the model can
be fitted in a single `fit_bertopic()` call from any standard R installation
with no external software.

## Quantitative model evaluation

A persistent challenge in topic modelling is justifying hyperparameter
choices transparently enough for peer review. `Rhobots` provides an
integrated suite of evaluation tools. `sweep_topics()` evaluates a
user-supplied grid of UMAP and HDBSCAN hyperparameters across five quality
metrics without re-running the encoder: silhouette score, within-topic
cohesion, between-topic centroid separation, vocabulary overlap, and noise
fraction. `topic_quality()` reports the same diagnostics for any fitted
model. `stability_analysis()` quantifies assignment robustness across
repeated runs using the Adjusted Rand Index, flagging hyperparameter settings
where topics are sensitive to random initialisation. `topic_coherence()`
computes NPMI and CV coherence directly from the corpus. Together these
functions enable documented, reproducible model selection rather than
informal inspection of word lists.

## Action-oriented topic representations

Standard topic models surface thematic noun clusters — *climate, policy,
government* — that describe *what a corpus is about*. For researchers
studying *what actors do*, this representation is analytically insufficient.

In computational journalism, social movement studies, policy discourse
analysis, and research on global or societal change, the verb structure of a
corpus is often as revealing as its noun structure. The discourse of an
activist movement is characterised not only by the concepts it invokes but
by the actions it describes: *demand*, *mobilise*, *resist*, *organise*.
News coverage of a geopolitical crisis is shaped as much by verbs —
*escalate*, *condemn*, *negotiate*, *withdraw* — as by the actors and places
it names. Research on societal change needs to identify topics of transition
and agency, not just topics of subject matter. A topic model that can only
surface noun clusters misses this layer entirely.

`Rhobots` addresses this through the `representation_model` argument of
`fit_bertopic()`. `pos_representation(pos = c("VERB"))` builds the c-TF-IDF
vocabulary from verbs alone, producing action-oriented topic labels that
describe what is happening rather than what is being discussed.
`pos_representation(pos = c("NOUN", "PROPN"))` recovers the conventional
entity-focused view; `pos = c("ADJ")` surfaces attributive framing. The
`patterns` argument extracts multi-word phrases matching a consecutive UPOS
sequence — for example `list(c("VERB", "NOUN"))` captures verb-object
constructions. POS annotation is handled by the `udpipe` package, which
supports more than 60 languages through Universal Dependencies models.

Alongside POS filtering, `cvalue_representation()` implements the C-value
method [@frantzi2000cvalue] for multi-word term recognition. It scores
candidate n-grams by their frequency discounted by how often they appear as
a sub-sequence of a longer term, ensuring that *sea level* does not compete
with *sea level rise* and that compound technical terms are surfaced as units.

# Implementation

`Rhobots` exposes the pipeline through three pluggable S3 extension points.
`dim_reduction_model` defaults to UMAP but accepts PCA or a custom object
implementing `dim_reduce()` and `dim_project()` generics. `cluster_model`
defaults to HDBSCAN but accepts k-means or agglomerative clustering.
`representation_model` defaults to document-frequency-filtered bag-of-words
and accepts `pos_representation()` or `cvalue_representation()`. Pre-trained
BERT-family encoders are loaded from the HuggingFace Hub via `hfhub` with no
Python runtime; `embed_texts_cached()` persists the embedding matrix so that
hyperparameter sweeps do not require re-running the encoder.

Additional functionality includes: out-of-sample prediction
(`transform_bertopic()`), post-fit topic manipulation (`reduce_topics()`,
`merge_topics()`, `reduce_outliers()`, `apply_mmr()`), dynamic topic
modelling over time (`topics_over_time()`, `fit_topics_over_time()`),
zero-shot topic classification (`zero_shot_topics()`), guided topic modelling
via seed-word injection (`guided_fit_bertopic()`), LLM-based label generation
(`label_topics_llm()`), structured model persistence (`save_bertopic()`,
`load_bertopic()`), and class-conditional group comparison
(`compare_topics()`). Interactive visualisations through `plotly` are
provided for all major outputs.

# Acknowledgements

The algorithm and API design follow the Python BERTopic package by
@grootendorst2022bertopic. The C-value term extraction algorithm follows
@frantzi2000cvalue. This package was developed for the *Data Analytics for
Sustainability* course at Utrecht University, Faculty of Geosciences.

# References
