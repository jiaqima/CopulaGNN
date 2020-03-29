import os
import sys
from collections import OrderedDict

import pandas as pd

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
    tmp = temp[-1].split("_")
#     setting_valid = (tmp[1], tmp[5], tmp[6], "_".join([model, lamda]), "valid")
#     res[setting_valid] = res.get(setting_valid, [])
    if model == "lsm_gcn":
        model = "lsmgcn"
    setting_test = (tmp[1], tmp[5], tmp[6], "_".join([model, lamda]), "test")
    res[setting_test] = res.get(setting_test, [])
    for i, t in enumerate(temp):
#         if t == "valid":
#             res[setting_valid].append(float(temp[i + 1]))
        if t == "test":
            res[setting_test].append(float(temp[i + 1]))

results = OrderedDict()
for key in sorted(res):
    if key[0] != mode:
        continue
    new_key = (key[1][1:], key[2][1:], key[3].split("_")[0])
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
print(latex_text[:start1] + " $\gamma$ " + latex_text[end1:start2] + " $\\tau$ " + latex_text[end2:start3] + " model " + latex_text[end3:])
