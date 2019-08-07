<br>

#Features：  
1. *news text & tittle*  
2. *news source*  
3. *related tweets*  

----

* combine the four features:  
`x = hstack([x1, x2, x3, xf])`  
`x = x.tocsr()`  

* analyze it  
`vecf = DictVectorizer()`
`xf = vecf.fit_transform(features)`


---
`hstack` :  
Stack arrays in sequence horizontally (column wise).  
