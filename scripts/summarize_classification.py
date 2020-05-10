import os
import numpy as np


def parse_result_file(filename):
    # The format of result filename is:
    # vacc__tacc__model__lr__h__l2__lambda__nr__p_,
    temp = filename.split("_")
    result = {}
    plus = 0
    for j in range(0, int(len(temp) / 2) * 2, 2):
        i = j + plus
        if temp[i] in ["model", "act"]:
            if temp[i+1] in ["lsm", "sbm"]:
                result[temp[i]] = temp[i+1] + "_" + temp[i+2]
                plus = 1
            else:
                result[temp[i]] = temp[i+1]
        else:
            result[temp[i]] = float(temp[i+1])
    return result


missing = []
duplicate = []

result_folder = "results"
models = ["regressioncgcn"]
data_list = ["cora", "citeseer"]
nls = [20]

summary = {}


for nl in nls:
    summary[nl] = {}
    for d in data_list:
        summary[nl][d] = {}
        best_vacc = {}
        tacc_mean = {}
        tacc_std = {}
        num = {}
        params = {}

        settings = {}
        settings_t = {}

        for net in models:
            m = net
            best_vacc[m] = 0
            tacc_mean[m] = 0
            tacc_std[m] = 0
            num[m] = 0
            params[m] = {"lr": None, "h": None, "lambda": None, "nr": None, "hx": None, "p0": None, "p1": None}
            settings[m] = {}
            settings_t[m] = {}

        for fname in os.listdir(result_folder + "/%s/nl%d" % (d, nl)):
            res = parse_result_file(fname)

            m = res["model"]
            setting = []
            for key in sorted(params[m]):
                setting.append(res[key])
            settings[m][tuple(setting)] = settings[m].get(tuple(setting), [])
            settings[m][tuple(setting)].append(res["vacc"])

            settings_t[m][tuple(setting)] = settings_t[m].get(tuple(setting), [])
            settings_t[m][tuple(setting)].append(res["tacc"])

        summary[nl][d]["valid"] = settings
        summary[nl][d]["test"] = settings_t
        for m in settings:
            for setting in settings[m]:
                values = settings[m][setting]
                if np.mean(values) > best_vacc[m]:
                    best_vacc[m] = np.mean(values)

                    cnt = 0
                    for key in sorted(params[m]):
                        params[m][key] = setting[cnt]
                        cnt += 1
                    tacc_mean[m] = np.mean(settings_t[m][setting])
                    tacc_std[m] = np.std(settings_t[m][setting])
                    num[m] = len(settings_t[m][setting])
                    if num[m] > 10:
                        duplicate.append(params[m])
                    if num[m] < 10:
                        missing.append(params[m])
        print("---%s  %d---" % (d, nl))
        for net in models:
            m = net
            print("%s, cnt %d:\t%.5f\t%.5f\t%s" % (net, num[m], tacc_mean[m], tacc_std[m], str(params[m])))
