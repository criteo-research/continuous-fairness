<center><h1>Fairness-Aware Continuous Learning (FACL)</h1></center>

This repository provides an implementation of fairness-aware regularization for
notions of fairness based on (conditional) independence. This code was used to
run the experiments of the paper:
> [Fairness-Aware Learning for Continuous Attributes and Treatments, J. Mary,
C. Calauzènes, N. El Karoui, ICML 2019](http://proceedings.mlr.press/v97/mary19a/mary19a.pdf)

## Install

First, you can create two conda environments from the YML files, one for the library facl only and one for library and 
notebooks. If you wish to run the examples, you should choose 'env_library_and_notebooks.yml'.
```bash
conda env create -f env_library_and_notebooks.yml
conda activate continuous-fairness-all
```

## Examples

You can find several examples of use in the directory `examples` in the form of
jupyter notebooks.


## LICENSE
The license can be found on in the file LICENSE.

## BibTex
If you use this code please cite
```
@InProceedings{pmlr-v97-mary19a,
  title = 	 {Fairness-Aware Learning for Continuous Attributes and Treatments},
  author = 	 {Mary, Jeremie and Calauz{\`e}nes, Cl{\'e}ment and El Karoui, Noureddine},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {4382--4391},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/mary19a/mary19a.pdf},
  url = 	 {http://proceedings.mlr.press/v97/mary19a.html},
  abstract = 	 {We address the problem of algorithmic fairness: ensuring that the outcome of a classifier is not biased towards certain values of sensitive variables such as age, race or gender. As common fairness metrics can be expressed as measures of (conditional) independence between variables, we propose to use the Rényi maximum correlation coefficient to generalize fairness measurement to continuous variables. We exploit Witsenhausen’s characterization of the Rényi correlation coefficient to propose a differentiable implementation linked to $f$-divergences. This allows us to generalize fairness-aware learning to continuous variables by using a penalty that upper bounds this coefficient. Theses allows fairness to be extented to variables such as mixed ethnic groups or financial status without thresholds effects. This penalty can be estimated on mini-batches allowing to use deep nets. Experiments show favorable comparisons to state of the art on binary variables and prove the ability to protect continuous ones}
}
```
