# Tutorial on probabilistic PCA in Python and Mathematica

## Running

* Python: `python prob_pca.py`. The figures are output to the [figures_py](figures_py) directory.
* Mathematica: Run the notebook `prob_pca.nb`. The figures are output to the [figures_ma](figures_ma) directory.

## Description

Consider 2D data such as the following:

<img src="figures_ma/data_scatter.png" alt="drawing" width="400"/>

<img src="figures_ma/data.png" alt="drawing" width="400"/>

After determining the ML parameters, we can sample the hidden units from the visible:

<img src="figures_ma/sample_hidden_from_visible.png" alt="drawing" width="400"/>

We can also draw new samples from the hidden distribution (a standard normal):

<img src="figures_ma/sample_hidden_std_normal.png" alt="drawing" width="400"/>

and then sample new visible samples from those:

<img src="figures_ma/sample_visible_from_hidden_scatter.png" alt="drawing" width="400"/>

<img src="figures_ma/sample_visible_from_hidden.png" alt="drawing" width="400"/>

Finally, we can rescale the latent variables to have any Gaussian distribution:

<img src="figures_ma/sample_hidden_from_rescaled_normal.png" alt="drawing" width="400"/>

We can simply transform the parameters and then **still** sample new valid visible samples from those:

<img src="figures_ma/rescaled_sample_visible_from_hidden_scatter.png" alt="drawing" width="400"/>

<img src="figures_ma/rescaled_sample_visible_from_hidden.png" alt="drawing" width="400"/>