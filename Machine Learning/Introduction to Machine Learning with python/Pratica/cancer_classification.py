import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})