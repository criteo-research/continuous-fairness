<center><h1>Fairness-Aware Continuous Learning (FACL)</h1></center>

This repository provides an implementation of fairness-aware regularization for
notions of fairness based on (conditional) independence. This code was used to
run the experiments of the paper:
> Fairness-Aware Learning for Continuous Attributes and Treatments, J. Mary,
C. Calauzènes, N. El Karoui, ICML 2019

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
@InProceedings{pmlr-mary19,
  title = 	 {Fairness-Aware Learning for Continuous Attributes and Treatments},
  author = 	 {J\'er\'emie Mary and Cl\'ement Clauz\`enes and Noureddine El Karoui},
  booktitle = 	 {Proceedings of the 36st International Conference on Machine Learning},
  year = 	 {2019},
  editor = 	 {Kamalika Chaudhuri and Ruslan Salakhutdinov },
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address =  {Long Beach, USA},
  month = 	 {11--13 Jun},
  publisher = 	 {PMLR},
}
```
