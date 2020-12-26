from plot import *
from sampling import *

import numpy as np
import scipy.linalg as sl
from pathlib import Path

def sample_and_plot_visibles(
    mean_hidden : np.array,
    cov_hidden : np.array,
    data : np.array,
    weight_ml : np.array,
    mu_ml : np.array,
    var_ml : float,
    label_plt_1 : str
    ):

    no_samples = len(data)
    samples_hidden = np.random.multivariate_normal(mean_hidden,cov_hidden,size=no_samples)

    # Sample visible given hidden
    act_visible = sample_visible_given_hidden(
        weight_ml=weight_ml,
        mu_ml=mu_ml,
        var_ml=var_ml,
        hidden_samples=samples_hidden
        )

    print("Covariance visibles (data):")
    print(data_cov)
    print("Covariance visibles (sampled):")
    print(np.cov(act_visible,rowvar=False))

    print("Mean visibles (data):")
    print(np.mean(data,axis=0))
    print("Mean visibles (sampled):")
    print(np.mean(act_visible,axis=0))

    fig_1 = plot_hist_1d(samples_hidden, label_plt_1)

    fig_2 = plot_data_visible_and_activated_2d(data, act_visible)

    fig_3 = plot_scatter_visible_and_activated_2d(data, act_visible)

    return [fig_1, fig_2, fig_3]

def import_data() -> np.array:
    fname = "data.txt"
    f = open(fname,'r')
    data = []
    for line in f:
        s = line.split()
        data.append([float(s[0]),float(s[1])])
    f.close()

    return np.array(data)

########################################
# Main
########################################

if __name__ == "__main__":

    ########################################
    # Import data
    ########################################

    data = import_data()

    figs = {}
    figs["data"] = plot_hist_2d(data, "Data")

    d = data.shape[1]

    print("\n---\n")

    mu_ml = np.mean(data,axis=0)
    print("Data mean:")
    print(mu_ml)

    data_cov = np.cov(data,rowvar=False)
    print("Data cov:")
    print(data_cov)

    ########################################
    # Max. likelihood
    ########################################

    print("\n---\n")

    # No hidden variables < no visibles = d
    q = 1

    # Variance
    lambdas, eigenvecs = np.linalg.eig(data_cov)
    idx = lambdas.argsort()[::-1]   
    lambdas = lambdas[idx]
    eigenvecs = - eigenvecs[:,idx]
    print(eigenvecs)
    # print(eigenvecs @ np.diag(lambdas) @ np.transpose(eigenvecs))

    var_ml = (1.0 / (d-q)) * sum([lambdas[j] for j in range(q,d)])
    print("Var ML:")
    print(var_ml)

    # Weight matrix
    uq = eigenvecs[:,:q]
    print("uq:")
    print(uq)

    lambdaq = np.diag(lambdas[:q])
    print("lambdaq")
    print(lambdaq)

    weight_ml = uq * np.sqrt(lambdaq - var_ml * np.eye(q))
    print("Weight matrix ML:")
    print(weight_ml)

    ########################################
    # Sample hidden from visible
    ########################################
    
    act_hidden = sample_hidden_given_visible(
        weight_ml=weight_ml,
        mu_ml=mu_ml,
        var_ml=var_ml,
        visible_samples=data
        )

    figs["sample_hidden _from _visible"] = plot_hist_1d(act_hidden,"Sampled hidden from visible")

    ########################################
    # Sample visible from hidden
    ########################################

    print("\n---\n")

    # Sample new visibles
    print("\n---\n")
    figs["sampled_hidden_from_visible"], \
        figs["sampled_visible_activated_hist"], \
            figs["sampled_visible_activated_scatter"] \
                = sample_and_plot_visibles(
        mean_hidden = np.full(q,0),
        cov_hidden = np.eye(q),
        data=data,
        weight_ml=weight_ml,
        mu_ml=mu_ml,
        var_ml=var_ml,
        label_plt_1="Hidden uniform from std. normal"
        )

    ########################################
    # Rescale
    ########################################

    print("\n---\n")

    mean_hidden = np.array([120.0])
    cov_hidden = np.array([[23.0]])
    weight_ml_rescaled = weight_ml @ np.linalg.inv(sl.sqrtm(cov_hidden))
    mu_ml_rescaled = mu_ml - weight_ml_rescaled @ mean_hidden

    print("Mean ML rescaled:")
    print(mu_ml_rescaled)

    print("Weight matrix ML rescaled:")
    print(weight_ml_rescaled)

    ########################################
    # Sample visible from hidden (rescaled)
    ########################################

    print("\n---\n")
    figs["rescaled_hidden_from_visible"], \
        figs["rescaled_visible_activated_hist"], \
            figs["rescaled_visible_activated_scatter"] \
                = sample_and_plot_visibles(
        mean_hidden = mean_hidden,
        cov_hidden = cov_hidden,
        data=data,
        weight_ml=weight_ml_rescaled,
        mu_ml=mu_ml_rescaled,
        var_ml=var_ml,
        label_plt_1="Hidden uniform from rescaled normal"
        )

    # Show figures
    # plt.show()

    # Save figures
    Path("figures_py").mkdir(parents=True, exist_ok=True)
    for name, fig in figs.items():
        fig.savefig("figures_py/" + name + ".png")
