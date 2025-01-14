
from functools import partial

from tqdm import tqdm
from main import get_scores
from private_projection import *
from utils import save_txt

def print_items(items):
    max_score = np.max([v["oac"] for k, v in items.items()])
    max_score = "{:.2f}".format(max_score * 100)
    
    verbose_method_names = {
        "PrivateIdentity": "Identity Projection",
        "PrivateGaussianProjection": "Gaussian Projection",
        "PrivateGaussianProjectionOfLabels": "Gaussian Projection of Labels",
        "PrivateRademacherProjection": "Rademacher Projection",
        "PrivateRademacherProjectionOfLabels": "Rademacher Projection of Labels",
        "PrivateOrthogonalProjection": "Orthogonal Projection",
        "PrivateOrthogonalProjectionOfLabels": "Orthogonal Projection of Labels",
        "OrthogonalProjection": "Orthogonal Projection",
        "Identity": "Identity Projection",
        "GaussianProjection": "Gaussian Projection",
        "RademacherProjection": "Rademacher Projection",
        "PrivateOrthogonalProjectionOfLabelsRandomizedResponse": "Orthogonal Projection of RR",
    }
    
    res = ""
    for key, value in items.items():
        method_name = verbose_method_names[key.split("_")[0]]
        
        res += r" & "
        if "{:.2f}".format(value["oac"] * 100) == max_score:
            res += method_name + r" & $" + "{:.2f}".format(value["orthogonal"] * 100) + r" {\scriptstyle \pm " + "{:.2f}".format(value["orthogonal_std"] * 100) + "}$ & "
            res += "$\mathbf{"+ "{:.2f}".format(value["oac"] * 100) + r" {\scriptstyle \pm " + "{:.2f}".format(value["oac_std"] * 100) + "}}$"
        else:
            res += method_name + " & ${:.2f}".format(value["orthogonal"] * 100) + r" {\scriptstyle \pm " + "{:.2f}".format(value["orthogonal_std"] * 100) + "}$ & "
            res += "${:.2f}".format(value["oac"] * 100) + r" {\scriptstyle \pm " + "{:.2f}".format(value["oac_std"] * 100) + "}$"
            
        res += r"\\" + "\n"
        
    return res


def generate_private_table():
    num_devices = 20
    num_repeats = 5
    main_dir = "results" #"/home/sfy21/OneDrive/ota_backup/results" # "results"
    
    data_name = "cifar10"
    p = 1.0
    A_t = 1.0
    client_output = "label" # belief or label
    epsilon = 1.0 # np.inf # 1.0 #np.inf
    channel_snr_db = 0.0
    
    num_projection_dims = 10
    num_classes = 10
    task = "multiclass"
    methods = ["oac", "orthogonal", "bestmodel"]
    
    projector_classes = [
        PrivateIdentity,
        PrivateGaussianProjection,
        PrivateGaussianProjectionOfLabels,
        PrivateRademacherProjection,
        PrivateRademacherProjectionOfLabels,
        PrivateOrthogonalProjection,
        PrivateOrthogonalProjectionOfLabels,
        PrivateOrthogonalProjectionOfLabelsRandomizedResponse
    ]
    
    res = {}
    for projector_cls in tqdm(projector_classes):
        projector_partial = partial(projector_cls, epsilon=epsilon, num_classes=num_classes, num_dims=num_projection_dims, participation_probability=1.0, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power", )
        scores_df = get_scores(["oac", "orthogonal"], data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial)
        res[f"{projector_cls.__name__}_d{num_projection_dims}_snr{channel_snr_db}"] = {"oac": scores_df["val_macro_f1_oac"].values[0], "orthogonal": scores_df["val_macro_f1_orthogonal"].values[0], "oac_std": scores_df["val_macro_f1_oac_std"].values[0], "orthogonal_std": scores_df["val_macro_f1_orthogonal_std"].values[0]}
        
    return res
def generate_nonprivate_table():
    num_devices = 20
    num_repeats = 5
    main_dir = "results" #"/home/sfy21/OneDrive/ota_backup/results" # "results"
    
    data_name = "cifar10"
    p = 1.0
    A_t = 1.0
    client_output = "label" # belief or label
    epsilon = np.inf # 1.0 #np.inf
    channel_snr_db = 0.0
    
    num_projection_dims = 10
    num_classes = 10
    task = "multiclass"
    methods = ["oac", "orthogonal", "bestmodel"]
    
    projector_classes = [
        PrivateIdentity,
        PrivateGaussianProjection,
        PrivateRademacherProjection,
        PrivateOrthogonalProjection,
    ]
    
    res = {}
    for projector_cls in tqdm(projector_classes):
        projector_partial = partial(projector_cls, epsilon=epsilon, num_classes=num_classes, num_dims=num_projection_dims, participation_probability=1.0, delta=1e-6, sensitivity=np.sqrt(2), normalizer="min_power", )
        scores_df = get_scores(["oac", "orthogonal"], data_name, num_devices, num_repeats, epsilon, p, A_t, channel_snr_db, client_output, main_dir, task, projector_partial)
        res[f"{projector_cls.__name__.replace('Private', '')}_d{num_projection_dims}_snr{channel_snr_db}"] = {"oac": scores_df["val_macro_f1_oac"].values[0], "orthogonal": scores_df["val_macro_f1_orthogonal"].values[0], "oac_std": scores_df["val_macro_f1_oac_std"].values[0], "orthogonal_std": scores_df["val_macro_f1_orthogonal_std"].values[0]}
    
    return res

def generate_result():
    table = r"""\begin{table}[htbp!]
    \centering
    \caption{Ablation study of different projection methods on the validation split of CIFAR-10}
    \resizebox{\columnwidth}{!}{%
        \begin{tabular}{""" + "ll" + "cc" + r"""}"""

    table += r"""
    \toprule
    $\varepsilon$ & Projection & """ + " & ".join([r"MV-Orth (\%)",r"MV-OAC (\%)"]) + r"\\ \midrule"  + "\n"
    
    table += r""" \multirow{4}{*}{$\infty$} """ + print_items(generate_nonprivate_table()) + r""" \midrule""" + "\n"
    
    
    table += r""" \multirow{7}{*}{$1$} """ + print_items(generate_private_table()) + "\n"

    table +=  r"""\bottomrule
    \end{tabular}}
    \label{tab:ablation_projection}
    \end{table}"""

    save_txt("figures", "table_ablation_projection.tex", table)
if __name__ == "__main__":
    generate_result()