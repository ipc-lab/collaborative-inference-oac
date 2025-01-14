
from functools import partial
from main import get_scores
from private_projection import *
from utils import save_txt
import pandas as pd
from tqdm import tqdm
import numpy as np

def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_scores_dict(p, num_dims, channel_snr_db, data_name, num_devices, num_repeats, num_classes, task, main_dir, client_output, A_t):
    scores = {}
    epsilon = np.inf
    projector_partial = partial(PrivateOrthogonalProjectionOfLabels, epsilon=epsilon, num_classes=num_classes, num_dims=num_dims, participation_probability=p, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power")
    scores["nonprivate"] = get_scores(["oac", "orthogonal"], data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial)

    epsilon = 1.0
    projector_partial = partial(PrivateOrthogonalProjectionOfLabels, epsilon=epsilon, num_classes=num_classes, num_dims=num_dims, participation_probability=p, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power")
    scores["private"] = get_scores(["oac", "orthogonal"], data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial)

    epsilon = 5.0
    projector_partial = partial(PrivateOrthogonalProjectionOfLabels, epsilon=epsilon, num_classes=num_classes, num_dims=num_dims, participation_probability=p, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power")
    scores["weakprivate"] = get_scores(["oac", "orthogonal"], data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial)

    dct = {
        "snr": channel_snr_db,
        "p": p,
        "num_dims": num_dims,
    }
    
    for privacy_level in ["nonprivate", "weakprivate", "private"]:
        for method in ["oac", "orthogonal"]:
            dct[f"{client_output}_{method}_{privacy_level}"] = scores[privacy_level][f"val_macro_f1_{method}"].values[0]
            dct[f"{client_output}_{method}_{privacy_level}_std"] = scores[privacy_level][f"val_macro_f1_{method}_std"].values[0]

    return dct

def generate_csv():
    num_devices = 20
    num_repeats = 5
    main_dir = "results" #"/home/sfy21/OneDrive/ota_backup/results" # "results"
    
    data_name = "cifar10"
    #p = 1.0
    A_t = 1.0
    #client_output = "label" # belief or label
    #epsilon = 1.0 # np.inf # 1.0 #np.inf
    #channel_snr_db = 0.0
    
    num_projection_dims = 10
    num_classes = 10
    task = "multiclass"
    methods = ["oac", "orthogonal", "bestmodel"]
    res = {}
    
    channel_snr_dbs = np.arange(-10, 5.1, 1)
    ps = np.arange(0.1, 1.01, 0.1)
    epsilons = np.arange(1, 15.1, 1)
    num_projection_dims = np.arange(1, 15, 1)
    
    res = []
    for channel_snr_db in tqdm(channel_snr_dbs):
        p = 1.0
        num_dims = 10
        dcts = []
        for client_output in ["label", "belief", "weighted_belief"]:
            dct = get_scores_dict(p, num_dims, channel_snr_db, data_name, num_devices, num_repeats, num_classes, task, main_dir, client_output, A_t)
            
            dcts.append(dct)
        
        res.append(merge_dicts(*dcts))


    df = pd.DataFrame(res)
    df .to_csv("figures/conditions_snr_vs_macro_f1.csv", index=False)
    
    res = []
    for p in tqdm(ps):
        num_dims = 10
        channel_snr_db = 0.0
        for client_output in ["label", "belief", "weighted_belief"]:
            dct = get_scores_dict(p, num_dims, channel_snr_db, data_name, num_devices, num_repeats, num_classes, task, main_dir, client_output, A_t)
            
            dcts.append(dct)
        
        res.append(merge_dicts(*dcts))
    
    df = pd.DataFrame(res)
    df .to_csv("figures/conditions_p_vs_macro_f1.csv", index=False)

    
    res = []
    for num_dims in tqdm(num_projection_dims):
        channel_snr_db = 0.0
        p = 1.0
        for client_output in ["label", "belief", "weighted_belief"]:
            dct = get_scores_dict(p, num_dims, channel_snr_db, data_name, num_devices, num_repeats, num_classes, task, main_dir, client_output, A_t)
            
            dcts.append(dct)
        
        res.append(merge_dicts(*dcts))
    
    df = pd.DataFrame(res)
    df .to_csv("figures/conditions_num_dims_vs_macro_f1.csv", index=False)

if __name__ == "__main__":
    generate_csv()