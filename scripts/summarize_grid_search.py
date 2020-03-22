import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

from plot_utils import parallel_coordinates

SEP = "__"

path = "data"
if len(sys.argv) > 1:
    path = sys.argv[1]
mode = "xw"
if len(sys.argv) > 2:
    mode = sys.argv[2]
assert mode in ["xw", "daxw"]

res = {}
for f in os.listdir("{}/results".format(path)):
    temp = f.split("__")
    for i, t in enumerate(temp):
        if t == "model":
            model = temp[i + 1]
        elif t == "lamda":
            lamda = temp[i + 1]
        elif t == "m_tau":
            m_tau = temp[i + 1]
        elif t == "m_gamma":
            m_gamma = temp[i + 1]
    tmp = temp[-1].split("_")
    setting_test = (tmp[1], tmp[5], tmp[6],
                    SEP.join([model, lamda, m_gamma, m_tau]), "test")
    res[setting_test] = res.get(setting_test, [])
    for i, t in enumerate(temp):
        if t == "test":
            res[setting_test].append(float(temp[i + 1]))

results = OrderedDict()
for key in sorted(res):
    if key[0] != mode:
        continue
    temp = key[3].split(SEP)
    new_key = (float(key[1][1:]), float(key[2][1:]), temp[0], float(temp[2]),
               float(temp[3]))
    results[new_key] = res[key]
df = pd.DataFrame(results)

ms_df = df.agg(["mean", "std"]).transpose()
latex_text = ms_df.to_latex()

k = len("toprule\n")
start1 = latex_text.find("toprule\n") + k
end1 = latex_text.find("&", start1)
start2 = end1 + 1
end2 = latex_text.find("&", start2)
start3 = end2 + 1
end3 = latex_text.find("&", start3)
start4 = end3 + 1
end4 = latex_text.find("&", start4)
start5 = end4 + 1
end5 = latex_text.find("&", start5)
print(latex_text[:start1] + " $\gamma$ " + latex_text[end1:start2] +
      " $\\tau$ " + latex_text[end2:start3] + " model " +
      latex_text[end3:start4] + " $\gamma_m$ " + latex_text[end4:start5] +
      " $\\tau_m$ " + latex_text[end5:])

# plot
index_names = np.array(["gamma", "tau", "model", "m_gamma", "m_tau"])
mlp_res = {}
gcn_res = {}

for index, row in ms_df.iterrows():
    setting = "g%.2f_t%.2f" % (index[0], index[1])
    d = {}
    d["model"] = index[2]
    d["m_gamma"] = index[3]
    d["m_tau"] = index[4]
    d["score"] = row["mean"]
    if "mlp" in d["model"]:
        mlp_res[setting] = mlp_res.get(setting, [])
        mlp_res[setting].append(d)
    elif "gcn" in d["model"]:
        gcn_res[setting] = gcn_res.get(setting, [])
        gcn_res[setting].append(d)

for setting in mlp_res:
    parallel_coordinates(
        mlp_res[setting],
        "%s/%s_%s_mlp.pdf" % (path, mode, setting.replace(".", "d")))
for setting in gcn_res:
    parallel_coordinates(
        gcn_res[setting],
        "%s/%s_%s_gcn.pdf" % (path, mode, setting.replace(".", "d")))
