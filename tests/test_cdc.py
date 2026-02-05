import numpy as np
import pytest
from sklearn.datasets import make_blobs
from cdc import CDC, cdc_cluster

def test_cdc_class():
    X, y = make_blobs(n_samples=100, centers=3, random_state=42)
    cdc = CDC(n_neighbors=10, ratio=0.9)
    cdc.fit(X)
    assert hasattr(cdc, 'labels_')
    
    # Check labels start from 0
    unique_labels = np.unique(cdc.labels_)
    # Filter out noise if any (-1)
    valid_labels = unique_labels[unique_labels >= 0]
    if len(valid_labels) > 0:
        assert np.min(valid_labels) == 0
        # Check consecutive integers (optional, but good practice if expected)
        # CDC might produce gaps if not careful, but our remapping ensures consecutive 0..K-1
        assert np.all(np.diff(sorted(valid_labels)) == 1)
        
    # Check predictions match fit
    labels = cdc.fit_predict(X)
    assert np.array_equal(labels, cdc.labels_)

def test_cdc_function():
    X, y = make_blobs(n_samples=100, centers=3, random_state=42)
    labels = cdc_cluster(X, n_neighbors=10, ratio=0.9)
    
    assert len(labels) == 100
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels >= 0]
    
    if len(valid_labels) > 0:
        assert np.min(valid_labels) == 0

def test_consistency():
    X, y = make_blobs(n_samples=100, centers=3, random_state=42)
    cdc = CDC(n_neighbors=10, ratio=0.9)
    labels_class = cdc.fit_predict(X)
    labels_func = cdc_cluster(X, n_neighbors=10, ratio=0.9)
    
    assert np.array_equal(labels_class, labels_func)

def test_input_validation():
    with pytest.raises(ValueError):
        cdc_cluster(np.random.rand(10, 2), n_neighbors=-1)
        
    with pytest.raises(ValueError):
        cdc_cluster(np.random.rand(10, 2), ratio=1.5)
