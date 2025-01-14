from functools import partial
from itertools import chain
from tqdm import tqdm
from main import get_scores
from private_projection import PrivateOrthogonalProjectionOfLabels
from utils import save_txt
import numpy as np
from utils import seed_everything
import pandas as pd
import copy


def get_scores_dict(data_name, num_classes):
    num_dims = num_classes
    p = 1.0
    num_devices = 20
    num_repeats = 5
    channel_snr_db = 0.0
    task = "multiclass"
    main_dir = "results" # TODO: change to results
    A_t = 1.0
    
    scores = {
        "weak_private": {},
        "private": {},
        "non_private": {}
    }
    
    raw_df = {
        "weak_private": {},
        "private": {},
        "non_private": {}
    }
    
    for client_output in ["label", "belief", "weighted_belief"]:
        if client_output in ["belief", "weighted_belief"]:
            methods = ["oac", "orthogonal"]
        else:
            methods = ["oac", "orthogonal", "bestmodel"]
        epsilon = np.inf
        projector_partial = partial(PrivateOrthogonalProjectionOfLabels, epsilon=epsilon, num_classes=num_classes, num_dims=num_dims, participation_probability=p, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power")
        scores["non_private"][client_output], raw_df["non_private"][client_output] = get_scores(methods, data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial, return_all=True)

        epsilon = 1.0
        projector_partial = partial(PrivateOrthogonalProjectionOfLabels, epsilon=epsilon, num_classes=num_classes, num_dims=num_dims, participation_probability=p, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power")
        scores["private"][client_output], raw_df["private"][client_output] = get_scores(methods, data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial, return_all=True)

        epsilon = 5.0
        projector_partial = partial(PrivateOrthogonalProjectionOfLabels, epsilon=epsilon, num_classes=num_classes, num_dims=num_dims, participation_probability=p, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power")
        scores["weak_private"][client_output], raw_df["weak_private"][client_output] = get_scores(methods, data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial, return_all=True)

    dct = {
        "snr": channel_snr_db,
    }
    raw_dct = {
        "snr": channel_snr_db,
    }
    
    for k, v in scores.items():
        for method in ["oac", "orthogonal", "bestmodel"]:
            for output in ["label", "belief", "weighted_belief"]:
                if method == "bestmodel" and output in ["belief", "weighted_belief"]:
                    continue
                
                dct[f"{k}_{method}_{output}"] = v[output][f"test_macro_f1_{method}"].values[0]
                dct[f"{k}_{method}_{output}_std"] = v[output][f"test_macro_f1_{method}_std"].values[0]
                
                raw_dct[f"{k}_{method}_{output}"] = raw_df[k][output][f"test_macro_f1_{method}"]

    for k, v in scores.items():
        dct[f"{k}_max_score"] = np.amax([dct[f"{k}_{method}"] for method in ["oac_label", "oac_belief", "oac_weighted_belief", "orthogonal_label", "orthogonal_belief", "orthogonal_weighted_belief", "bestmodel_label"]])

    return dct, raw_dct

def generate_dict(shown_datasets):
    res = {}
    raw_res = {}
    
    for key, (data_name, num_classes) in tqdm(shown_datasets.items()):
        res[data_name], raw_res[data_name] = get_scores_dict(key, num_classes)

    return res, raw_res

if __name__ == "__main__":
    
    seed_everything(1)
    shown_datasets = {
        "cifar10": ("Cifar10", 10),
        "cifar100": ("Cifar100", 100),
        #"mnist": ("Mnist", 10),
        "fashionmnist": ("FashionMnist", 10),
        "food101": ("Food101", 101),
        "oxford3tpets": ("OxfordPets", 37),
        "emotion": ("Emotion", 6),
        "imdb": ("Imdb", 2),
        "multiview_oxford3tpets": ("MultiViewPets", 37),
        #"dtd": ("DTD", 47),
        #"country211": ("Country211", 211),
        #"flowers102": ("Flowers102", 102),
    }
    
    scores, raw_scores = generate_dict(shown_datasets)
    
    for key, (dataset, num_classes) in shown_datasets.items():
        for privacy_level in ["non_private", "weak_private", "private"]:
            for method in ["oac_label", "oac_belief", "oac_weighted_belief", "orthogonal_label", "orthogonal_belief", "orthogonal_weighted_belief", "bestmodel_label"]:
                is_max =  np.around(scores[dataset][f"{privacy_level}_{method}"],4) == np.around(scores[dataset][f"{privacy_level}_max_score"],4)
                
                scores[dataset][f"{privacy_level}_{method}"] = "{:.2f} ".format(scores[dataset][f"{privacy_level}_{method}"] * 100) + r" {\scriptstyle \pm " + "{:.2f}".format(scores[dataset][f"{privacy_level}_{method}_std"] * 100) + r"}"
                
                if is_max:
                    scores[dataset][f"{privacy_level}_{method}"] = r"$\mathbf{" + scores[dataset][f"{privacy_level}_{method}"] + r"}$"
                else:
                    scores[dataset][f"{privacy_level}_{method}"] = r"$" + scores[dataset][f"{privacy_level}_{method}"] + r"$"


    methods_verbose = {
        "bestmodel_label": "Best Client",
        "orthogonal_belief": "BA-Orth",
        "orthogonal_weighted_belief": "WBA-Orth",
        "orthogonal_label": "MV-Orth",
        "oac_belief": "BA-OAC",
        "oac_weighted_belief": "WBA-OAC",
        "oac_label": "MV-OAC",
    }
    
    datasets = scores.keys()

    res = r"""\begin{table*}[t]
    \centering
    \caption{Comparison of our method with the baselines in terms of Macro-F1 scores}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{cl""" + "c"*len(datasets) + r"""}
    \toprule
    $\varepsilon$ & Method & """ + r" & ".join(datasets) + r"\\ \midrule"  + "\n"
    
    privacy_level = "non_private"
    res += r"\multirow{7}{*}{$\infty$}"
    for method in ["bestmodel_label", "orthogonal_belief", "orthogonal_weighted_belief", "orthogonal_label", "oac_belief", "oac_weighted_belief", "oac_label"]:
        res += f" & {methods_verbose[method]} & "

        res += " & ".join([scores[dataset][f"{privacy_level}_{method}"] for dataset in datasets])
        res += r" \\"  + " \n "
    
    res += r"\midrule " + " \n "
    
    privacy_level = "weak_private"
    res += r"\multirow{7}{*}{$5$}"
    for method in ["bestmodel_label", "orthogonal_belief", "orthogonal_weighted_belief", "orthogonal_label", "oac_belief", "oac_weighted_belief", "oac_label"]:
        res += f" & {methods_verbose[method]} & "

        res += " & ".join([scores[dataset][f"{privacy_level}_{method}"] for dataset in datasets])
        res += r" \\ " + " \n "
    
    res += r"\midrule " + " \n "

    privacy_level = "private"
    res += r"\multirow{7}{*}{$1$}"
    for method in ["bestmodel_label", "orthogonal_belief", "orthogonal_weighted_belief", "orthogonal_label", "oac_belief", "oac_weighted_belief", "oac_label"]:
        res += f" & {methods_verbose[method]} & "

        res += " & ".join([scores[dataset][f"{privacy_level}_{method}"] for dataset in datasets])
        res += r" \\ " + " \n "
    
    res += r"""\bottomrule
            \end{tabular}}
            \label{tab:comparison}
            \end{table*}"""

    save_txt("figures", "table_comparison.tex", res)

    for privacy_level in ["non_private", "weak_private", "private"]:
        results = {}
        for method in ["bestmodel_label", "orthogonal_belief", "orthogonal_weighted_belief", "orthogonal_label", "oac_label", "oac_belief", "oac_weighted_belief"]:
                results[methods_verbose[method]] = chain(*[raw_scores[dataset][f"{privacy_level}_{method}"].values for dataset in datasets])
        
        pd.DataFrame(results).to_csv(f"figures/comparison_{privacy_level}.csv", index=False)