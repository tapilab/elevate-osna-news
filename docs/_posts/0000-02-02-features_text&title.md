---
layout: slide
title: "1. news text & title  "
---

* analyze the news text:  
`vec1 = TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1, 3), stop_words='english')`  
`x1 = vec1.fit_transform(text)`

* analyze the news title:  
`vec2 = TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1, 3), stop_words='english')`  
`x2 = vec2.fit_transform(title)`
---
`TfidfVectorizer` :  
Convert a collection of raw documents to a matrix of TF-IDF features.

