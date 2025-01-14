from functools import partial
import os
import pandas as pd
from pandas import DataFrame
from ensembler import Ensembler
from sklearn.metrics import f1_score
from private_projection import PrivateGaussianProjectionOfLabels, PrivateIdentity, PrivateGaussianProjection, PrivateOrthogonalProjectionOfLabels, PrivateOrthogonalProjectionOfLabels, PrivateRademacherProjection, PrivateRademacherProjectionOfLabels
from utils import get_results, get_y_test_beliefs, get_y_val_test_beliefs, seed_everything
from utils import calculate_score
import torch
import numpy as np


def get_scores(methods, data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial, return_all=False):
    results = get_results(data_name=data_name, num_devices=num_devices, num_repeats=num_repeats, main_dir=main_dir)
    
    scores_df = DataFrame()
    
    for seed_idx, (y_val_beliefs_dict, y_val_true, y_test_beliefs_dict, y_test_true) in get_y_val_test_beliefs(results, num_repeats).items():
        seed_everything(seed_idx)
                
        num_classes = y_val_beliefs_dict[0].shape[1]

        projector = projector_partial() #PrivateOrthogonalProjectionOfLabelsOfLabels(epsilon=epsilon, num_classes=num_classes, num_dims=num_projection_dims, normalizer=normalizer)
        
        model = Ensembler(projector, A_t, num_devices, channel_snr_db, participation_probability=p, client_output=client_output, task=task)
        
        y_pred_test = {}
        y_pred_val = {}
        for method in methods:
            y_pred_test[method] = model.forward(method, y_test_beliefs_dict, y_val_beliefs_dict, y_val_true)
            y_pred_val[method] = model.forward(method, y_val_beliefs_dict, y_val_beliefs_dict, y_val_true)
        
        seed_results = {
            #"seed_idx": seed_idx,
        }
        
        metrics = ["macro_f1"]
        y_test_true = torch.nn.functional.one_hot(y_test_true, num_classes)
        y_val_true = torch.nn.functional.one_hot(y_val_true, num_classes)
        
        for metric in metrics:
            for method in methods:
                seed_results[f"test_{metric}_{method}"] = calculate_score(y_test_true, y_pred_test[method], average="micro" if metric == "micro_f1" else "macro")
                seed_results[f"val_{metric}_{method}"] = calculate_score(y_val_true, y_pred_val[method], average="micro" if metric == "micro_f1" else "macro")
        
        scores_df = pd.concat([scores_df, pd.DataFrame(seed_results, index=[0])], ignore_index=True)

    if return_all:
        raw_df = scores_df.reindex(sorted(scores_df.columns), axis=1)

    scores_df_mean = scores_df.mean().to_frame().transpose()
    scores_df_std = scores_df.std().to_frame().transpose().add_suffix("_std")
    scores_df = pd.concat([scores_df_mean, scores_df_std], axis=1)
    scores_df = scores_df.reindex(sorted(scores_df.columns), axis=1)

    if return_all:
        return scores_df, raw_df

    return scores_df

if __name__ == "__main__":
    num_repeats = 5
    num_devices = 20
    main_dir = "results"
    
    data_name = "cifar10"
    p = 1.0
    A_t = 1.0
    client_output = "label" # belief or label
    epsilon = 1.0 # np.inf # 1.0 #np.inf
    channel_snr_db = 10.0
    normalizer = "min_power"
    
    num_projection_dims = 5
    num_classes = 10
    task = "multiclass"
    methods = ["oac", "orthogonal", "bestmodel"]
    
    projector_partial = partial(PrivateOrthogonalProjectionOfLabels, epsilon=epsilon, num_classes=num_classes, num_dims=num_projection_dims, participation_probability=1.0, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power", )

    scores_df = get_scores(methods, data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial)
    
    print(scores_df.transpose())