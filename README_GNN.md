
## 2. Graph Convolutional Network Model Selection: 

### 2.1 Graph Auto Encoder: 

### 2.2 Graph Convolutional Network: 
- Graph Convolutional Network (GCN): Propagates information through the graph using node connections.
GraphSAGE: Extends GCN to inductively learn embeddings for unseen nodes.
Graph Attention Network (GAT): Uses attention mechanisms to weigh neighboring nodes differently.
Node2Vec: Generates embeddings using random walks, often used with a similarity-based approach for edge prediction.

Built a two-layer GCN model for edge prediction.
Implemented a forward pass to compute adjacency probabilities.


Predicted edges using adjacency probabilities.
Developed create_mst to construct a graph using a minimum spanning tree.

## 3. Approach and Methodology

### 3.1 Data Preparation:
When designing the node embeddings for our graph, we chose random walk embeddings with Word2Vec over context embeddings.
Per Graph, a list of Random Walks were generated to train the configured Word2Vec model for Random Walks. 
The resulting parts_embeddings.model was then used to retrieve the embeddings for all Part_IDs. 

Some Parameter decisions included: 

| Parameter                  | Value | Reasoning                                                                                                                                                              |
|----------------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Number of walks per graph  | 16    | Since each starting node of the Random walk is equally likely, this addresses that for the majority of graphs (which have fewer than 8 nodes) each node is starting not of the Random Walk likely twice.|
| Walk Length               | 8     | The walk length matches the typical graph size, ensuring complete exploration. For small graphs, it avoids overtraversal of the topology and potentially duplicated segments |
| Embedding Vector Size            | 16    | A vector size of 16 captures sufficient structural information without overfitting. Between the options of embeddings, consisting of 8 or 32 values, this balanced performance and computational requirements the best.|
 

### 3.2 Edge Labels: 
During the creation of the Training Data, edge labels were created for each edge in a graph. Initially those edge labels were all positive, 
since Training only happened on the positive edges. 
As a result, the model learned to set all edge probabilities to 1. 

Therefore a negative loss term for edges, which were predicted to be highly probably, but not present in the original graph had to be included. 
However, due to these intermediary steps, the training and backpropagation loop became more and more complicated, which is why negative edge labels (encoded by 0) were later included were then included already during the Training Data Preparation steps earlier on.

During initial training after this adjustment, the model now tends to predict all edge probabilities at 0.5, 

### GCN Model Design:
### Layers: 
Two-layer GCN for feature transformation.

### Loss Function: 
Binary Cross-Entropy Loss for edge prediction.

### Prediction Workflow:
### Pass node embeddings through the trained GCN.
### Compute the probabilistic adjacency matrix.
### Apply a threshold to identify edges.



## 6. Results and Insights

### Key Observations: Highlight trends, such as performance across different thresholds.
