---
layout: slide
title: "Logistic Regression"
---


| min_df |  Accuracy|
|-------------------|----------|
|      1  |0.809768|
|      2  |0.809768|
|      5 | 0.811119|
|     10|  0.816506|

min_df : float in range [0.0, 1.0] or int, default=1

    When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.



| C  |Accuracy|
|-------------------|----------|
|  0.1 | 0.740958 |
|    1 | 0.809768 |
|    5 | 0.834047 |
|   10 | 0.834029 |

C : float, optional (default=1.0)

    Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

