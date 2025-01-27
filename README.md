# ai-lecture-project

## First Ideation

This project addresses the problem of predicting graphs that represent physical objects built from components ("parts"). Each graph node corresponds to a part, uniquely identified by part_id, and categorized into a family via family_id. Edges between nodes represent connections between these parts. The dataset contains 2271 unique parts and multiple graphs.

The task is to predict the structure of these graphs using a neural network model. Specifically, a feedforward neural network (FFNN) is trained to predict the adjacency matrix of a graph, from which a minimum spanning tree (MST) is constructed to form the final predicted graph.


## Dataset
### General
- Parts, Families, Nodes:
  - part_id: A unique identifier for each part.
  - family_id: Specifies the family to which the part belongs.
  - nodes: A graph specific reference to a specific part in a graph. 

- Graphs:
  - Undirected, unweighted, non-cyclic, connected graphs without self-loops.

### Embedding Parts 
We had several initial ideas about the best approach to encode the given graph data in both the training, testing, validation and single prediction instances. 
Each approach had it´s advantages and disadvantages in achieving the goal of encoding both the information about the overall graph structure (mainly the frequency of parts) and the current position in the graph (in order to accurately predict potential edges from this current position). 

Solely relying on a vector to encode the part frequencies in the graph omits the information about the current "position" in the graph, which quickly turned out to be a non-promising approach. 

Similarly, solely relying on one-hot encodings to encode the current position in the graph omits the overall information about the graph structure and therefore also turned out to be unfeasible. Furthermore, one-hot encoding is a binary representation where each unique part is assigned a vector with a single 1 corresponding to its identifier and 0s elsewhere. If a part occurs multiple times within a graph, one-hot encoding cannot represent this frequency since it only indicates presence or absence. A potential solution approach to this problem would be to stack the one-hot encoded vectors. However, if the vectors for all parts in a graph are stacked (one vector per part), the resulting tensor dimensions would depend on the number of parts in the graph. Since different graphs can have varying numbers of parts, this would lead to tensors with inconsistent shapes.

As a result, we opted for a combination of both frequency encodings (to capture the graph structure) and one-hot encodings (to capture the current position in the graph):

- Frequency Encoding:
  - Represents the count of each part_id in the graph as a vector of length 2271.
  - This encoding provides the network with information about the overall structure of the graph by highlighting how frequently each part appears.

- One-Hot Encoding:
  - A binary vector where one element is set to 1 for the current part, and the rest are 0.
  - This encoding helps the model focus on the specific part being processed, enabling targeted predictions for connections.

- Combined Encoding:
  - Each part's encoding combines the frequency encoding of the graph and the part's one-hot encoding.
  - This combination ensures the model simultaneously captures global graph structure and localized part-specific information.

- Learned Embeddings: 
  - This aproach might have been another feasible one. However, the embeddings would have needed to be trained as well, introducing a further point of uncertainty and information loss, which is why we ultimately decided against it in this implementation of a FNN. (During the implementation of the GNN, we applied trained embeddings.) 


## Different Solution Approaches
### Random Baseline Model
This is a simple baseline model that predicts edges randomly. This model selects all parts from the given list and randomly picks an existing node from the graph. Then it adds the part to the existing node with a new edge. As the number of possibilities is very limited due to the lack of cycles, and the performance is therefore surprisingly good, this model should serve as a baseline.
The code can be found in `predict_random.ipynb`.


### Neighbour Prediction based on mean edge probability
A global mean edge probability matrix is calculated. 
This model creates the graph based on the the maximum mean edge probabilities. Only one global adjacency matrix is therefore initialized and populated with the mean edge probabilities. The model then takes a parts list and for each part it looks at the the mean edge probability to the already existing parts/nodes in the graph. The edge with the maximum edge probability is taken as the connection between the parts and the new part is added and removed from the parts list.
The code can be found in `predict_noML.ipynb`.

 
### Graph-Wide Adjacency Prediction
A feedforward neural network takes the frequency-encoded parts as input and predicts the entire adjacency matrix for the graph. The problem is that each output has the dimension of number_of_parts x number_of_parts (2071 x 2071), which leads to RuntimeError (Invalid buffer size: 34.00 GB).
The code can be found in `predict_FNN_complete.ipynb`.
 

### Graph Convolutional Network
An extensive description can be found in `README_GNN.md` and the code can be found in `predict_GNN.ipynb`.

### Neighbor Prediction
This code can be found in `predict_neighbours.ipynb` and is our model of choice.
This approach predicts the most likely neighbors for each part using a FFNN.
With the combined encodings prepared, the task becomes a supervised learning problem where the goal is to predict edge probabilities between a source part and all other parts in the graph.

#### Data Preparation
During Data preparation, for each part, the combined encoding of size 4542 (2271 frequency + 2271 one-hot) serves as the input feature vector. The target labels represent the adjacency relationships (edges) for the part, encoded as a binary vector where a value of 1 indicates the presence of an edge.

#### Model Architecture
- Input Layer: Accepts the combined encoding of size 4542.
- Hidden Layers: Two layers with ReLU activation, configured to process and abstract the high-dimensional input.
- Output Layer: Produces a vector of probabilities for all possible connections.

#### Training
- Loss Function: Binary Cross-Entropy Loss to handle the binary adjacency labels.
- Optimizer: Adam optimizer for efficient training.

#### Adjacency Matrix and Minimum Spanning Tree Construction
After training, the model predicts the edge probabilities for each part in a set of partIds, which create the graph. 

The adjacency matrix is constructed iteratively by:

1. Edge Probability Calculation: For each part, pass its combined encoding through the trained FFNN to obtain edge probabilities.
2. Adjacency Matrix Assembly: We populate the adjacency matrix using the complementary values of the predicted probabilities for each part.
3. Graph computation: We ensure the graph remains connected and free of cycles by computing a Minimum Spanning Tree (MST).

Finally, we convert the MST into a project-specific graph. 

### Variational Autoencoder
- Variational Autoencoder (VAE)


## Metriken
### Edge Accuracy
The primary assessment metric for this project, edge accuracy, was predefined as part of the project requirements. Edge accuracy measures the proportion of correctly predicted edges (true positives and true negatives) in the adjacency matrix compared to the ground truth.

### Graph Accuracy
While edge accuracy served as the primary assessment metric for this project, an interesting complementary metric to explore would have been graph accuracy. Unlike edge accuracy, which focuses on individual connections, graph accuracy evaluates the correctness of the entire predicted graph structure compared to the ground truth graph. Unfortunately, due to time constraints, we didn´t get to implement and evaluate our implemented solution approach on that any more.











---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
This project is written with Python `3.8` based on Anaconda (https://www.anaconda.com/distribution/).
If you wish, you can upgrade to a higher Python version. 

## Getting started

The file 'requirements.txt' lists the required packages.

1. We recommend to use a virtual environment to ensure consistency, e.g.
`conda create -n ai-project python=3.8`

2. Activate the environment:
`conda activate ai-project`

3. Install the dependencies:
`conda install -c conda-forge --file requirements.txt`


## Software Tests
This project contains some software tests based on Python Unittest (https://docs.python.org/3/library/unittest.html).
Run `python -m unittest` from the command line in the repository root folder to execute the tests. This automatically searches all unittest files in modules or packages in the current folder and its subfolders that are named `test_*`.