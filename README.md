# ai-lecture-project

## Erste Gedanken
- "Undirected, unweighted, non-cyclic and connected without self-loops"
--> also keine 
- kleinste part number ist 0 --> aufpassen bei one-hot encoding und frequency encoding


## Dataset
- Embedding/Encoding von Parts 
    - one-hot encoding
        --> geht nicht
            1. man verliert die Anzahl der Parts, weil ein Part A auch mehrmals auftreten kann in einem graph
            2. wenn man für jeden Part einen one-hot encoded Vektor erstellt und diese stacked, dann ist die Dimension für jeden Tensor unterschiedlich, da abhängig von der Anzahl der Parts
    - frequency encoding
    - learning embedding wäre möglich, aber müsste mittrainiert werden



## Modellansätze
- Random --> Accuracy: 70.71%


- Feedforward NN, welches die Parts(frequency encoded) als Input erhält und ganze Adjacency Matrizen (Dimension: number_of_unique_parts x number_of_unique_parts) zurückgeben soll --> Benötigt zu viel RAM
- GNN --> Graph Convolutional Network
- Variational Autoencoder (VAE)


## Metriken
- edge accuracy vorgegeben
- weitere Möglichkeiten: node accuracy


## Unterscheidung von Node, Part und Familie
- Node: ist für den Graphen & enthält eine Referenz zu Part
- Part: beschreibt eine einzelne Komponente -> part_id
- Familie: beschreibt zu welche Familie ein Part gehört --> family_id










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