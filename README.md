# Clustering by measuring local direction centrality for data with heterogeneous density and weak connectivity (CDC)


We propose a novel Clustering algorithm by measuring Direction Centrality (CDC) locally. It adopts a density-independent metric based on the distribution of K-nearest neighbors (KNNs) to distinguish between internal and boundary points. The boundary points generate enclosed cages to bind the connections of internal points, thereby preventing cross-cluster connections and separating weakly-connected clusters. We present an interactive ***Demo*** and a brief introduction to the algorithm at ***https://zpguigroupwhu.github.io/CDC-Introduction-Website/***. This paper has been published in ***Nature Communications***, and more details can be seen https://www.nature.com/articles/s41467-022-33136-9.

![image](https://github.com/ZPGuiGroupWhu/ClusteringDirectionCentrality/blob/master/pics/cdc_algorithm.png)

# Installation
Supported python versions are ```3.9.1``` and above.

This project has been uploaded to [PyPI](https://pypi.org/), supporting direct download and installation from pypi

```
pip install cdc
```

## Automatic Installation (Recommended)
## Manual Installation
# Citation Request:
Peng, D., Gui, Z.*, Wang, D. et al. Clustering by measuring local direction centrality for data with heterogeneous density and weak connectivity. Nat. Commun. 13, 5455 (2022).
https://www.nature.com/articles/s41467-022-33136-9

# License

This project is covered under the Apache 2.0 License.
