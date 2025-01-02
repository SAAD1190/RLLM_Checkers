Overview
========================

.. figure:: /Documentation/images/intro.jpg
   :width: 500
   :align: center
   :alt: Image explaining Prompt Analyzer introduction

--------------------------------------------------------------

.. figure:: /Documentation/images/app_screenshots.png
   :width: 800
   :align: center
   :alt: Application Screenshots

--------------------------------------------------------------

**Prompt Analyzer** is an advanced tool designed to analyze, evaluate, and optimize prompts for natural language processing (NLP) tasks. It provides insights into prompts using a combination of semantic and syntactic metrics, empowering users to refine their NLP workflows.

Highlights
=============

- **Interactive Interface**: Powered by Streamlit for an intuitive user experience.

- **Comprehensive Metrics**: Evaluate prompts based on Semantic Richness, Vocabulary Richness, Relevance, Lexical Density, and more.

General Pipeline
===================

**Input Prompts**
-----------------

Users provide one or more natural language prompts (e.g., "Explain the applications of machine learning.").

**Semantic and Syntactic Analysis**
-----------------------------------

- **Text Preprocessing**: Prompts are tokenized, and stop words and punctuation are removed for clean processing.
- **Embedding Generation**: Each prompt is converted into dense vectors using pre-trained models (e.g., SentenceTransformer's `all-mpnet-base-v2`).
- **Metric Computation**: Various metrics, such as Semantic Diversity Score (SDS), Semantic Repetition Penalty (SRP), Vocabulary Richness (VR), and Relevance, are computed.

**Sorting and Scoring**
-----------------------

- Prompts are evaluated and scored based on the selected metric (e.g., Relevance, Semantic Vocabulary Richness).
- If relevance is chosen, a reference prompt set is used to compute hybrid relevance scores.

**Output**
-------------------

- The final results include sorted prompts and corresponding scores, displayed in an interactive table.
- Optional download of results in JSON format for further analysis.