
# Graph Convolutional Network
As a further approach, we employ a Graph Convolutional Network (GCN) to solve the graph prediction problem, 
specifically focusing on predicting edges within a graph structure. While initial attempts utilized a Graph Autoencoder (GAE) 
architecture due to its suitability for edge prediction tasks (https://github.com/tkipf/gae), the results were not promising. 
Consequently, a GCN was implemented as a simpler, more interpretable solution.

This approach was the last, experimental approach that we took. Since during training, the results were overall not too positive
and time limitations prevented us from further pursuing this implementation, this can and should be seen as a general approach to 
solving the graph implementation problem rather than a functional, validated and performant solution to the problem. 

# Approach and Methodology

## Data Preparation

### Node Embeddings
When designing the node embeddings for our graph, we chose random walk embeddings with Word2Vec over context embeddings.
This approach captures local and global structural information of nodes in the graph.
Per Graph, a list of Random Walks were generated to train the configured Word2Vec model for Random Walks. 
The resulting parts_embeddings.model was then used to retrieve the embeddings for all Part_IDs. 

Some Parameter decisions included: 

| Parameter                  | Value | Reasoning                                                                                                                                                              |
|----------------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Number of walks per graph  | 16    | Ensures that for most graphs (fewer than 8 nodes), each node serves as the starting point approximately twice.|
| Walk Length               | 8     |Matches typical graph sizes, avoiding redundant traversal in smaller graphs.|
| Embedding Vector Size            | 16    | Captures structural information effectively, balancing between underfitting (size 8) and overfitting (size 32).|
 
### Embedding Model Training
Word2Vec was trained on the random walks with the following hyperparameters:
- Context window: 5 (captures relationships between nodes within typical graph neighborhoods).
- Epochs: 20 (ensures sufficient training without overfitting).
- Skip-Gram model: Used for its effectiveness in learning node relationships in sparse data.

Output: The trained Word2Vec model (parts_embeddings.model) provides embeddings for each Part ID.

### Edge Labels 

- Initial Setup:
Positive edge labels were generated from existing graph edges. However, during initial training, the model learned to predict all edge probabilities as 1, leading to poor performance.

- Adjustment:
Negative edge labels were introduced during the data preparation stage by:
- Generating all possible edges.
- Identifying edges absent from the original graph as negative samples (labeled 0).
This adjustment simplified the loss calculation during training and improved model performance.

## GCN Model Design

### Architecture
A two-layer GCN was designed to predict edges by transforming node features into higher-dimensional 
representations and computing adjacency probabilities.

1. input Layer
    - Takes node embeddings (size 16).
2. Hidden Layer
    - GCNConv with 32 hidden units.
    - Activation: ReLU.
3. Output Layer
    - GCNConv with 16 units.
    - Outputs latent node embeddings.

### Training
- Data Loader: Batched graphs for training using PyTorch Geometric's DataLoader.
- Loss Function with Binary Cross-Entropy Loss:
  - Positive edges: Encourages high probabilities for existing edges. 
  - Negative edges: Penalizes predictions for non-existent edges.

# Insights

Training Trends:
- Initial training, using only positive edges, caused the model to predict all edge probabilities as 1. This indicated a lack of discrimination in edge classification, likely due to imbalanced training data.
- Incorporating negative edges into the training process stabilized predictions, with most probabilities clustering near 0.5 initially. This behavior was consistent across epochs and highlighted potential issues in the model's ability to refine predictions beyond this midpoint. 
- It remains unclear whether this outcome stems from an implementation issue or reflects an inherent limitation in the model's capacity to distinguish edges effectively under the current setup.

Due to time constraints and as discussed earlier, we weren´t able to further pursue this approach and refine the implementation and training. 