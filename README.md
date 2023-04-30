pyflann  - with edits for Python 3 compatibility
=============

## 1. Introduction

pyflann is the python bindings for [FLANN - Fast Library for Approximate Nearest Neighbors](http://www.cs.ubc.ca/research/flann/).

## 2. Install

#### For python2

This package uses distutils, which is the default way of installing python modules. To install in your home directory, securely run the following:
```
git clone https://github.com/MU-HPDI/py3flann.git pyflann
cd pyflann
[sudo] python setup.py install
```

Or directly through `pip` to install it:
```
[sudo] pip install pyflann
```

#### For python3

~~Please refer to [this issuse](https://github.com/primetang/pyflann/issues/1) to modify the code.~~
This is a Python 3 compatible version!

### 3. Usage

Use it just like the following code:
```python
from pyflann import *
import numpy as np

dataset = np.array(
    [[1., 1, 1, 2, 3],
     [10, 10, 10, 3, 2],
     [100, 100, 2, 30, 1]
     ])
testset = np.array(
    [[1., 1, 1, 1, 1],
     [90, 90, 10, 10, 1]
     ])
flann = FLANN()
result, dists = flann.nn(
    dataset, testset, 2, algorithm="kmeans", branching=32, iterations=7, checks=16)
print(result)
print(dists)

dataset = np.random.rand(10000, 128)
testset = np.random.rand(1000, 128)
flann = FLANN()
result, dists = flann.nn(
    dataset, testset, 5, algorithm="kmeans", branching=32, iterations=7, checks=16)
print(result)
print(dists)
```

--- 

# A better example for building fast K-NN indexing with KD-tree

```
import pyflann as pf
import numpy as np

dataset = np.random.rand(10000, 128)
testset = np.random.rand(1000, 128)

# Build Index
FLANN_BUILD_INDEX_PARAMS = dict(
    algorithm="kdtree_simple",
)
kdtree = pf.FLANN()
kdtree.build_index(dataset, **FLANN_BUILD_INDEX_PARAMS)

# Query Index (top-25)
query = testset[0,:]

# Use the testing (query) vector to find the top-25
res, dists = kdtree.nn_index(query, num_neighbors=25)

# List out the Top-25
for d,r in zip(dists[0],res[0]):
    print("Match: Dist = {}, data={}".format(d,r))
```
