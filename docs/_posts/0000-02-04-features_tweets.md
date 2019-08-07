---
layout: slide
title: "3. related tweets "
---


* get features of the tweets:  
`df['avg_retweet'] = avg_ret`  
`df['avg_favorite'] = avg_fav`   
`df['var_time'] = var_time`  
`df['var_desc'] = var_desc`  

* analyze it  
`vecf = DictVectorizer()`
`xf = vecf.fit_transform(features)`


---
`DictVectorizer()` :  
Convert a collection of text documents to a matrix of token counts

