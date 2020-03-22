cmds = []


def build_cmd(cmd):
    if "seed" in cmd:
        cmds.append(cmd)
        return
    if "mean_mode" not in cmd:
        for mean_mode in ["xw", "daxw"]:
            build_cmd(cmd + " --mean_mode {}".format(mean_mode))
    elif "model_type" not in cmd:
        for model_type in [
                "gcn", "mlp", "newcgcn", "newcmlp", "noisynewcgcn",
                "noisynewcmlp"
        ]:
            build_cmd(cmd + " --model_type {}".format(model_type))
    elif "tau" not in cmd:
        for tau in [1.]:
        # for tau in [0.1, 1., 5.]:
            if "cgcn" in cmd or "cmlp" in cmd:
                for m_tau in [tau / 10, tau / 2, tau, tau * 2, tau * 10]:
                # for m_tau in [tau]:
                    build_cmd(cmd + " --tau {} --m_tau {}".format(tau, m_tau))
            else:
                build_cmd(cmd + " --tau {}".format(tau))
    elif "gamma" not in cmd:
        for gamma in [0.5]:
        # for gamma in [0.1]:
            if "cgcn" in cmd or "cmlp" in cmd:
                for m_gamma in [
                        gamma / 5, gamma / 2, gamma, gamma * 2, gamma * 5
                ]:
                # for m_gamma in [gamma]:
                    build_cmd(
                        cmd +
                        " --gamma {} --m_gamma {}".format(gamma, m_gamma))
            else:
                build_cmd(cmd + " --gamma {}".format(gamma))
    elif "seed" not in cmd:
        for seed in range(100):
            build_cmd(cmd +
                      " --seed {} --device cuda:{}".format(seed, seed % 2))


build_cmd("python main.py --verbose 0 --patience 30 --num_epochs 5000")

with open("run.sh", "w") as f:
    for cmd in cmds:
        f.write(cmd + "\n")
