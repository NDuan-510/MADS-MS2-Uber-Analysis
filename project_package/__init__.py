# project_package/__init__.py
from .modeling import (
    # Supervised
    train_classification_from_csv,
    train_regression_from_csv,
    predict,
    # Unsupervised
    run_kmeans_from_csv,
    run_isolation_forest_from_csv,
    run_pca_embeddings_from_csv,
)
