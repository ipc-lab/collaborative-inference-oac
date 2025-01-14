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
import itertools

all_num_devices = [5, 15, 20, 50, 100]

def get_scores_dict(data_name, num_classes):
    num_dims = num_classes
    p = 1.0
    #num_devices = 20
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
    
    for num_devices in all_num_devices:
        for client_output in ["label", "belief", "weighted_belief"]:
            if client_output in ["belief", "weighted_belief"]:
                methods = ["oac", "orthogonal"]
            else:
                methods = ["oac", "orthogonal", "bestmodel"]
            epsilon = np.inf
            projector_partial = partial(PrivateOrthogonalProjectionOfLabels, epsilon=epsilon, num_classes=num_classes, num_dims=num_dims, participation_probability=p, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power")
            scores["non_private"][f"{client_output}_{num_devices}"], raw_df["non_private"][f"{client_output}_{num_devices}"] = get_scores(methods, data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial, return_all=True)

            epsilon = 1.0
            projector_partial = partial(PrivateOrthogonalProjectionOfLabels, epsilon=epsilon, num_classes=num_classes, num_dims=num_dims, participation_probability=p, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power")
            scores["private"][f"{client_output}_{num_devices}"], raw_df["private"][f"{client_output}_{num_devices}"] = get_scores(methods, data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial, return_all=True)

            epsilon = 5.0
            projector_partial = partial(PrivateOrthogonalProjectionOfLabels, epsilon=epsilon, num_classes=num_classes, num_dims=num_dims, participation_probability=p, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power")
            scores["weak_private"][f"{client_output}_{num_devices}"], raw_df["weak_private"][f"{client_output}_{num_devices}"] = get_scores(methods, data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial, return_all=True)

    dct = {
        "snr": channel_snr_db,
    }
    raw_dct = {
        "snr": channel_snr_db,
    }
    
    for k, v in scores.items():
        for method in ["oac", "orthogonal", "bestmodel"]:
            for output in ["label", "belief", "weighted_belief"]:
                for num_users in all_num_devices:
                    if method == "bestmodel" and output in ["belief", "weighted_belief"]:
                        continue
                    
                    dct[f"{k}_{method}_{output}_{num_users}"] = v[f"{output}_{num_users}"][f"test_macro_f1_{method}"].values[0]
                    dct[f"{k}_{method}_{output}_{num_users}_std"] = v[f"{output}_{num_users}"][f"test_macro_f1_{method}_std"].values[0]
                    
                    raw_dct[f"{k}_{method}_{output}_{num_users}"] = raw_df[k][f"{output}_{num_users}"][f"test_macro_f1_{method}"]

    for k, v in scores.items():
        dct[f"{k}_max_score"] = np.amax([dct[f"{k}_{method}_{num_devices}"] for method in ["oac_label", "oac_belief", "oac_weighted_belief", "orthogonal_label", "orthogonal_belief", "orthogonal_weighted_belief", "bestmodel_label"] for num_devices in all_num_devices])

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
        #"cifar100": ("Cifar100", 100),
        #"mnist": ("Mnist", 10),
        #"fashionmnist": ("FashionMnist", 10),
        #"food101": ("Food101", 101),
        #"oxford3tpets": ("OxfordPets", 37),
        #"imdb": ("Imdb", 2),
        #"emotion": ("Emotion", 6),
        #"multiview_oxford3tpets": ("MultiViewPets", 37),
        #"dtd": ("DTD", 47),
        #"country211": ("Country211", 211),
        #"flowers102": ("Flowers102", 102),
    }
    
    scores, raw_scores = generate_dict(shown_datasets)
    
    for key, (dataset, num_classes) in shown_datasets.items():
        for privacy_level in ["non_private", "weak_private", "private"]:
            for num_devices in all_num_devices:
                for method in ["oac_label", "oac_belief", "oac_weighted_belief", "orthogonal_label", "orthogonal_belief", "orthogonal_weighted_belief", "bestmodel_label"]:
                    is_max =  np.around(scores[dataset][f"{privacy_level}_{method}_{num_devices}"],4) == np.around(scores[dataset][f"{privacy_level}_max_score"],4)
                    
                    scores[dataset][f"{privacy_level}_{method}_{num_devices}"] = "{:.2f} ".format(scores[dataset][f"{privacy_level}_{method}_{num_devices}"] * 100) + r" {\scriptstyle \pm " + "{:.2f}".format(scores[dataset][f"{privacy_level}_{method}_{num_devices}_std"] * 100) + r"}"
                    
                    if is_max:
                        scores[dataset][f"{privacy_level}_{method}_{num_devices}"] = r"$\mathbf{" + scores[dataset][f"{privacy_level}_{method}_{num_devices}"] + r"}$"
                    else:
                        scores[dataset][f"{privacy_level}_{method}_{num_devices}"] = r"$" + scores[dataset][f"{privacy_level}_{method}_{num_devices}"] + r"$"

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
    \caption{Scalability analysis of the introduced methods on CIFAR-10 in terms of Macro-F1}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{cl""" + "c"*len(all_num_devices) + r"""}
    \toprule
    $\varepsilon$ & Method & """ + r" Users & ".join(map(str, all_num_devices)) + r" Users\\ \midrule"  + "\n"
    
    privacy_level = "non_private"
    res += r"\multirow{7}{*}{$\infty$}"
    #methods = itertools.product(["bestmodel_label", "orthogonal_belief", "orthogonal_weighted_belief", "orthogonal_label", "oac_belief", "oac_weighted_belief", "oac_label"], all_num_devices)
    methods = ["bestmodel_label", "orthogonal_belief", "orthogonal_weighted_belief", "orthogonal_label", "oac_belief", "oac_weighted_belief", "oac_label"]
    dataset = list(datasets)[0]
    
    for method in methods:
        res += f" & {methods_verbose[method]} & "

        res += " & ".join([scores[dataset][f"{privacy_level}_{method}_{num_devices}"] for num_devices in all_num_devices])
        res += r" \\"  + " \n "
    
    res += r"\midrule " + " \n "
    
    privacy_level = "weak_private"
    res += r"\multirow{7}{*}{$5$}"
    for method in methods:
        res += f" & {methods_verbose[method]} & "

        res += " & ".join([scores[dataset][f"{privacy_level}_{method}_{num_devices}"] for num_devices in all_num_devices])
        res += r" \\ " + " \n "
    
    res += r"\midrule " + " \n "

    privacy_level = "private"
    res += r"\multirow{7}{*}{$1$}"
    for method in methods:
        res += f" & {methods_verbose[method]} & "

        res += " & ".join([scores[dataset][f"{privacy_level}_{method}_{num_devices}"] for num_devices in all_num_devices])
        res += r" \\ " + " \n "
    
    res += r"""\bottomrule
            \end{tabular}}
            \label{tab:ablation_numusers}
            \end{table*}"""

    save_txt("figures", "table_ablation_numusers.tex", res)

    for privacy_level in ["non_private", "weak_private", "private"]:
        results = {}
        for method in methods:
            for num_devices in all_num_devices:
                results[f"{methods_verbose[method]}_{num_devices}"] = chain(*[raw_scores[dataset][f"{privacy_level}_{method}_{num_devices}"].values])
        
        pd.DataFrame(results).to_csv(f"figures/ablation_numusers_{privacy_level}.csv", index=False)