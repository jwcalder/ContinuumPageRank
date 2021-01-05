# Continuum limit for PageRank

This GitHub repository contains the code to reproduce experiments from the paper 

Yuan, Calder, Osting. [A Continuum Limit for the PageRank Algorithm](https://arxiv.org/abs/2001.08973), arXiv Preprint, 2020.

Install the requried Python packages

```
pip install -r requirements.txt
```

The convergence rate experiment can be run with

```
python convergence_test.py > acc.csv
python plot_acc.py
```

The script convergence_test.py will take a long time to run, even with parallel processing enabled. The accuracy file acc.csv is provided in the repository for reference.

The experiment comparing the distance between the PageRank vector and the teleportation distribution as a function of the teleportation probability alpha can be run with

```
python err_vs_alpha.py
```

Finally, the experiments exploring the use of PageRank for data depth and high dimensional medians can be run with

```
python pagerank_median_simulation.py
```


## Contact and questions


Email <jwcalder@umn.edu> with any questions or comments.
