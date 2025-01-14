
import numpy as np
import pandas as pd
from private_projection import get_sigma
from tqdm import tqdm

if __name__ == "__main__":
    delta = 1e-6
    p = np.arange(0.25, 1.01, 0.25)
    eps = np.arange(1, 15.1, 3)
    n = [5, 10, 15, 20]

    sensitivity = np.sqrt(2) # TODO: check if random projection or identity at the end.

    res = []
    for p_ in tqdm(p):
        for eps_ in eps:
            for n_ in n:
                p_ = np.around(p_, 2)
                sigma = get_sigma(delta, p_, eps_, n_, sensitivity) #/ np.sqrt(p_ * n_)
                res.append({
                    "p": p_,
                    "n": n_,
                    "eps": eps_,
                    "sigma": sigma
                })

    pd.DataFrame(res).reset_index(drop=True).to_csv("figures/privacy_all.csv",index=False)
    
    res = []
    for eps_ in tqdm(eps):
        c = {
            "eps": eps_
        }
        for n_ in n:
            sigma = get_sigma(delta, 1.0, eps_, n_, sensitivity) / np.sqrt(n_)
            c[f"sigma_n{n_}"] = sigma
        res.append(c)
    
    pd.DataFrame(res).reset_index(drop=True).to_csv("figures/privacy_eps_vs_sigma_by_n.csv",index=False)
    
    res = []
    for eps_ in tqdm(eps):
        c = {
            "eps": eps_
        }
        for p_ in p:
            p_ = np.around(p_, 2)
            sigma = get_sigma(delta, p_, eps_, 20, sensitivity) #/ np.sqrt(p_ * 20)
            c[f"sigma_p{p_}"] = sigma
        res.append(c)

    pd.DataFrame(res).reset_index(drop=True).to_csv("figures/privacy_eps_vs_sigma_by_p.csv",index=False)
