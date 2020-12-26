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
    var_ml : float
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

    fig_1 = plot_hist_1d(samples_hidden, "Hidden sampled from visible")

    fig_2 = plot_data_visible_and_activated_2d(data, act_visible)

    fig_3 = plot_scatter_visible_and_activated_2d(data, act_visible)

    return [fig_1, fig_2, fig_3]

data = [[2655, 3033], [2744, 3012], [2562, 2997], [2524, 3019], [2742, 3007], [2558, 3038], [2940, 3045], [2714, 3005], [2618, 3044], [2601, 3041], [2845, 3004], [2682, 3023], [2797, 3033], [2528, 3000], [2587, 3024], [2601, 2999], [2620, 3018], [2546, 3022], [2778, 3019], [2870, 3030], [2854, 3027], [2644, 3013], [3041, 3016], [2686, 3034], [2526, 3010], [2732, 3019], [2594, 3022], [2926, 3037], [2478, 3026], [2544, 3018], [2848, 3010], [2450, 3012], [2606, 3021], [2649, 3035], [2783, 3025], [2600, 3017], [2432, 2999], [2544, 3025], [2587, 3011], [2745, 3011], [2709, 3039], [2742, 3014], [2600, 3049], [2913, 3015], [2563, 3035], [2824, 3006], [2747, 3024], [2450, 3035], [2658, 3030], [2536, 3012], [2738, 3022], [2512, 3028], [2646, 3026], [2615, 3030], [2740, 3010], [2448, 3032], [2469, 3014], [2526, 3024], [2639, 3010], [2852, 3005], [2854, 2995], [2807, 3023], [2594, 3005], [2557, 3041], [2535, 3035], [2386, 3026], [2923, 3031], [2851, 3048], [2468, 3026], [2670, 3023], [2512, 3005], [2917, 3015], [2384, 3019], [2750, 3043], [2760, 2998], [2512, 3012], [2824, 3023], [2569, 3011], [2636, 3008], [2852, 3015], [2448, 3000], [2682, 3002], [2705, 3026], [2570, 2997], [2565, 3012], [2657, 3021], [2647, 3039], [2767, 2973], [2869, 3034], [2405, 3044], [2688, 3010], [2511, 3009], [2742, 3027], [2691, 3022], [2686, 3009], [2722, 3009], [2700, 3019], [2965, 3016], [2339, 3000], [2548, 3053], [2985, 2997], [2799, 3026], [2617, 3015], [2669, 3033], [2610, 3007], [2669, 3016], [2926, 2997], [2205, 3011], [2564, 3009], [2592, 2999], [2748, 3051], [2757, 3011], [2579, 3035], [2614, 3010], [3062, 3031], [2679, 3008], [2407, 3018], [2611, 3031], [2895, 3037], [2587, 3054], [2785, 3025], [2690, 3010], [2579, 3033], [2649, 3030], [2507, 3001], [2951, 3003], [2788, 3012], [2538, 2967], [2620, 3030], [2734, 3001], [2528, 3041], [2383, 3014], [2577, 2991], [2752, 3008], [2457, 3012], [2608, 3017], [2635, 3024], [2856, 3015], [2869, 3022], [2519, 3031], [3045, 3036], [2710, 3035], [2568, 3025], [2484, 3015], [2438, 3015], [2707, 3025], [2730, 3041], [2981, 3020], [2544, 3018], [2566, 3027]]
data = np.array(data)

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

# Sample hiddens

act_hidden = sample_hidden_given_visible(
    weight_ml=weight_ml,
    mu_ml=mu_ml,
    var_ml=var_ml,
    visible_samples=data
    )

figs["sampled_hidden_given_visible"] = plot_hist_1d(act_hidden,"Hidden samples")

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
    var_ml=var_ml
    )

# Rescale
print("\n---\n")

mean_hidden = np.array([120.0])
cov_hidden = np.array([[23.0]])
weight_ml_rescaled = weight_ml @ np.linalg.inv(sl.sqrtm(cov_hidden))
mu_ml_rescaled = mu_ml - weight_ml_rescaled @ mean_hidden

print("Mean ML rescaled:")
print(mu_ml_rescaled)

print("Weight matrix ML rescaled:")
print(weight_ml_rescaled)

# Sample new visibles
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
    var_ml=var_ml
    )

# Show figures
# plt.show()

# Save figures
Path("figures").mkdir(parents=True, exist_ok=True)
for name, fig in figs.items():
    fig.savefig("figures/" + name + ".png")
