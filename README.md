# ai-lecture-project

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