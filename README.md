Graph Embedding
==================
DESCRIPTION
-----------
Code is implemented according to the paper. 

Graph embedding is to represent entities and relationships of multi-relational data in low-dimensional vector space. These vectors can be used for entity mapping and in the future for knowledge completion. Five ml models are implemented in package embeddings.
1. TransE: It is the basic model for all the other four models and the fastest one. It uses the hypothesis that head - tail = relation and use max margin loss as loss function.
2. TransH: Instead of directly using the head and tail vectors, it projects them into the plane where relation is.
3. TransR/CTransR: it projects head and tail vectors into another space where relation is. CTransR, in addition, cluster each relation into different cluster-specific relations. These two methods are very complex and expensive.
4. TransD: it also projects head and tail vectors into another space where relation is, but instead of computing matrix, is uses vectors (identity matrix can be computed as vecotor). This has the best performance and can be applied to large-scale knowledge graph as vector computing is less expensive and faster.
5. TransSparse: it also projects head and tail vectors into another space where relation is, but it uses sparse matrix which aims to deal with heterogeneous and unbalanced knowledge graphs. There are two models share (head and tail share the same matrix) and separate (head and tail have their own matrices).
 
##Changelog

no changes yet

