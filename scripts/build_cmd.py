cmds = []


def build_cmd(cmd):
    if "seed" in cmd:
        cmds.append(cmd)
        return
    if "mean_mode" not in cmd:
        for mean_mode in ["xw", "daxw"]:
            build_cmd(cmd + " --mean_mode {}".format(mean_mode))
    elif "tau" not in cmd:
        for tau in [0.1, 1., 5.]:
            build_cmd(cmd + " --tau {}".format(tau))
    elif "gamma" not in cmd:
        for gamma in [0.1]:
            build_cmd(cmd + " --gamma {}".format(gamma))
    elif "model_type" not in cmd:
        for model_type in ["gcn", "mlp", "mngcn", "mnmlp"]:
            build_cmd(cmd + " --model_type {}".format(model_type))
        for model_type in ["cgcn", "cmlp", "newcgcn", "newcmlp", "noisycgcn", "noisycmlp"]:
            for lamda in [0.1]:
                build_cmd(cmd + " --model_type {} --lamda {}".format(
                    model_type, lamda))
    elif "seed" not in cmd:
        for seed in range(20):
            build_cmd(cmd + " --seed {} --device cuda:{}".format(seed, seed % 2))


build_cmd("python main.py --verbose 0 --patience 30 --num_epochs 5000")

with open("run.sh", "w") as f:
    for cmd in cmds:
        f.write(cmd + "\n")
