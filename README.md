![image](https://img.shields.io/badge/R-4.1.0-brightgreen) ![image](https://img.shields.io/badge/MATLAB-R2020b-red) ![image](https://img.shields.io/badge/Python-3.9.1-yellow) [![DOI](https://zenodo.org/badge/467575519.svg)](https://zenodo.org/badge/latestdoi/467575519)

# Clustering by measuring local direction centrality for data with heterogeneous density and weak connectivity (CDC)


We propose a novel Clustering algorithm by measuring Direction Centrality (CDC) locally. It adopts a density-independent metric based on the distribution of K-nearest neighbors (KNNs) to distinguish between internal and boundary points. The boundary points generate enclosed cages to bind the connections of internal points, thereby preventing cross-cluster connections and separating weakly-connected clusters. We present an interactive ***Demo*** and a brief introduction to the algorithm at ***https://zpguigroupwhu.github.io/CDC-Introduction-Website/***. This paper has been published in ***Nature Communications***, and more details can be seen https://www.nature.com/articles/s41467-022-33136-9.

This is a toolkit for CDC cluster analysis on various applications, including ‘scRNA-seq Cluster’, ‘UCI Benchmark Test’, ‘Synthetic Data Analysis’, ‘CyTOF Cluster’, ‘Speaker Recognition’, ‘Face Recognition’. They are implemented using MATLAB, R and Python languages.

We also provide a separated code module named scRNA-seq Result Reproduction to facilitate users to quickly reproduce our results on all 13 scRNA-seq datasets in 2D UMAP space, which can be executed independently with the developed toolkit. In this module, users don’t need to specify any parameters of preprocessing steps and CDC algorithm, and only the dataset name and type of running mode (“All” and “Best” modes) are required to reproduce the exactly same results presented in our paper.

Now, a parallel version of the algorithm CDC in Java is also under developing based on High-Performance Computing (HPC) framework Apache Spark, which is nested under the folder "HPC-version".

![image](https://github.com/ZPGuiGroupWhu/ClusteringDirectionCentrality/blob/master/pics/cdc_algorithm.png)


# Citation Request:
Peng, D., Gui, Z.*, Wang, D. et al. Clustering by measuring local direction centrality for data with heterogeneous density and weak connectivity. Nat. Commun. 13, 5455 (2022).
https://www.nature.com/articles/s41467-022-33136-9

# License

This project is covered under the Apache 2.0 License.
