import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random as r
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def shortest_path_node_affinity(graph, n1, n2, maxVal = 10**5):
    if n1 == n2:
        return maxVal
    return min(1/nx.shortest_path_length(graph, n1, n2), maxVal)



class Cluster_embedding:
    def __init__(self, G):
        '''
        This algorithm performs 
        '''
        self.graph = G
    
    def init_clusters(self, k):
        '''
        The nodes are randomly grouped into k clusters.        
        '''
        
        num_clusters = k
        nodes = self.graph.nodes()
        
        clusters = [[] for _ in range(num_clusters)]
        for node in nodes:
            ind = int(r.random()*num_clusters)
            clusters[ind].append(node)

        return clusters
    
    def cluster_affinity(self, graph, node, cluster, node_affinity = shortest_path_node_affinity):
        affinity = 0
        for nodej in cluster:
            if nodej != node:
                affinity += node_affinity(graph, node, nodej)

        return affinity
    
    
    def compute_embedding(self, graph, clusters, node_affinity = shortest_path_node_affinity):
        '''
        Outputs a np array shaped (num_nodes, num_clusters)
        '''
        nodes = graph.nodes
        embedding_matrix = []
        for node in nodes:
            node_embedding = [self.cluster_affinity(graph, node, cluster, node_affinity) for cluster in clusters]
            embedding_matrix.append(node_embedding)

        embedding_matrix = np.array(embedding_matrix)
        return embedding_matrix
    
    def get_new_clusters(self, nodes, embeddings):
        '''
        nodes, embeddings is expected to match row_wise
        Returns a list new clusters: List[List[node]]    
        '''
        n_clusters = embeddings.shape[1]

        kmeans = KMeans(n_clusters = n_clusters).fit(embeddings)
        labels = list(kmeans.labels_)
        cluster_dict = {}
        for node, label in zip(nodes, labels):
            if label in cluster_dict:
                cluster_dict[label].append(node)
            else:
                cluster_dict[label] = [node]

        clusters = [i[1] for i in cluster_dict.items()]
        return clusters
    
    
    def node2vec_via_clustering(self,  vec_size, num_iter = 200, node_affinity = shortest_path_node_affinity):
        '''
        Output:
            Cluster as List[List[node]]
            embedding matrix with ind2node mapping done by graph.nodes
        '''
        graph = self.graph
        num_clusters = vec_size
        clusters = self.init_clusters(num_clusters)

        for iteration in range(num_iter):
            embeddings = self.compute_embedding(graph, clusters, node_affinity)
            clusters = self.get_new_clusters(graph.nodes, embeddings)

        return clusters, embeddings
    
    def labels_from_clusters(self, clusters):
        labels = []
        for node in self.graph.nodes:
            for ind, cls in enumerate(clusters):
                if node in cls:
                    labels.append(ind)
                    break
        
        return labels
    
    
    def fit(self, k, num_iter = 200, node_affinity = shortest_path_node_affinity):
        self.k = k
        self.clusters, self.embeddings = self.node2vec_via_clustering(k, num_iter, node_affinity)
        self.labels = self.labels_from_clusters(self.clusters)
    
    
    
    def assign_labels(self, labels):
        
        for i,node in enumerate(self.graph.nodes()):
            self.graph.nodes[node]['label'] = labels[i]

    def plot_graph(self, title='Graph after cluster_embedding clustering'):
        # Plots the graph   
        n_clusters = self.k
        labels = self.labels
        
        
        self.assign_labels( labels)

        colour_list = ['blue','green','red','yellow','cyan', 'magenta','lightblue','grey']
        sampled_colours = dict(zip(set(labels),r.sample(colour_list, n_clusters)))

        legend_handles = []
        for label, color in sampled_colours.items():
            colour_handle = mpatches.Patch(color=color, label=label)
            legend_handles.append(colour_handle)

        colours = [sampled_colours[i] for i in labels]

        pos_fr = nx.fruchterman_reingold_layout(self.graph)
        plt.figure(figsize=(8,8))
        plt.title(title)
        plt.legend(handles=legend_handles)
        nx.draw(self.graph, pos=pos_fr, node_size=500, node_color=colours, with_labels=True)
        plt.show()
        
    
    def get_embedding(self):
        try:
            return self.embeddings.copy()
        except:
            raise Exception("Try fitting first")
    
    def get_labels(self):
        try:
            return [i for i in self.labels]
        except:
            raise Exception("Try fitting first")
        
    
    
    
    
    
    
    
    
    