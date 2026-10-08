# **ProD**, a visualizable filter-feature selection method based on "prodding" the class {Pro}bability {D}ensities for overlapping.

## Install
ProD can be installed from PyPI:
<pre>
pip install prod-fs
</pre>

## Example
```python
from prod_fs import ProD

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Create random classification dataset
X, y = make_classification(
    n_samples=300, n_features=50, n_classes=3, n_informative=5,
    shuffle=False
)

# Initialize ProD object
prodRanker = ProD()

# Carry out feature selection
prodRanker.fit(X, y)

# Get top 10 features
top10Features = prodRanker.get_topnFeatures(10)

# Visualize the top feature's ability to segregate PDEs
fig, axs = plt.subplots(1, 2, sharey=True)

# Top ranked feature
prodRanker.plot_overlapAreas(top10Features[0], legend="intersection", _ax=axs[0])
axs[0].set_title("Most relevant feature", loc="left")

# Last ranked feature
prodRanker.plot_overlapAreas(49, legend="intersection", _ax=axs[1])
axs[1].set_title("Least relevant feature", loc="left")

axs[0].set_ylabel(r"Probability Density, $\hat{P}$")
for i in range(2):
    axs[i].set_xlim(-0.5, 1.5)
    axs[i].set_xticks(np.arange(-0.5, 2.0, 0.5))
```
<p align="center">
  <img src="https://github.com/RenZhen95/prod-fs/blob/main/docs/artwork/example_plot.svg" width="550">
</p>

## Citation

If you use `prod-fs` in your research, please cite our paper:

> J.C. Liaw, F. Geu Flores, W. Kowalczyk. ProD: A visualizable filter-feature selection method based on “prodding” the class {Pro}bability {D}ensities for overlapping. Machine Learning with Applications 26, 101035 (2026)

Here's an example of a BibTeX entry:

```bibtex
@article{liaw2026prod,
  title   = {{ProD: A visualizable filter-feature selection method based on “prodding” the class {Pro}bability {D}ensities for overlapping}},
  journal = {Machine Learning with Applications},
  volume  = {26},
  pages   = {101035},
  year    = {2026},
  issn    = {2666-8270},
  doi     = {10.1016/j.mlwa.2026.101035},
  url     = {https://doi.org/10.1016/j.mlwa.2026.101035},
  author  = {Liaw, Jin Cheng and {Geu Flores}, Francisco and Kowalczyk, Wojciech}
}
```
