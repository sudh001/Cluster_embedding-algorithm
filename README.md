# Cluster_embedding-algorithm
This repository demonstrates the cluster_embedding algorithm for graphs

The algorithm in high level is
1. Randomly initialize k clusters where k is the desired embedding size of a node
2. For every node, compute its embedding by calculating the cluster affinity with a particular cluster 
> * cluster_affinity(node_i, cluster_j) = sum(node_affinity(node_i, node_j) for every node_j in cluster_j if node_i != node_j)
> * node_affinity(node_i, node_j) = 1/len(shortest_path(node_i, node_j))

3. Perform k-means clustering of the nodes based on the embeddings just computed
4. Using the newly obtained clusters repeat the process unitl convergence is reached or unitl `max_iter` iterations have been done


The main idea behind this seemingly random algorithm is that if given ideal clusters of nodes, nodes in the same cluster will have very similar affinity with other clusters as these nodes are very close to each other and have the same neighbourhood. 

Hence the nodes in the same cluster will have very similar embeddings and hence k-means algorithm would cluster them together. This shows that ideal clusters would be a point of convergence

It is important to note that the clusters formed aren't stable if `max_iter` version is used

The embeddings obtained can be used for further downstream tasks like node, edge clasification and can also be passed into graph neural networks for further processing. This repo dosen't extensivly test this yet though.
