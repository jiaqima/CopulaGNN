from __future__ import absolute_import, division, print_function

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEP = "__"  # Avoid common signs like "_".
coolwarm_256 = sns.color_palette("coolwarm", 256)


def make_box_plot(
        df,
        filename,
        drop_columns=None,
        hue_name="model",
        x_name="metric",
        y_name="value",
):
    if drop_columns is None:
        drop_columns = []
    df_long = pd.melt(
        df.drop(columns=drop_columns),
        hue_name,
        var_name=x_name,
        value_name=y_name)
    width = 0.7 * len(pd.unique(
        df[hue_name])) * (len(df.columns) - len(drop_columns))
    sns_plot = sns.factorplot(
        x_name, hue=hue_name, y=y_name, data=df_long, kind="box")
    sns_plot.set_xticklabels(rotation=30)
    sns_plot.fig.set_size_inches(width, 5)
    sns_plot.savefig(filename)


def coolwarm_256_color_map(x):
    assert x >= 0 and x < 256
    return coolwarm_256[int(x)]


def parallel_coordinates(trials,
                         plot_filename=None,
                         scale_transformers=None,
                         color_map=coolwarm_256_color_map,
                         alpha=0.8,
                         beta=0.3):
    """Plots parallel coordinates graph.
    Inputs:
      trials: `list` of `dict` with coordinates and the `score` as the keys of
        the dict. The `score` field will be used as the color of the curves and
        all other fields will be used as coordinates. All the dicts are
        required to have the same keys.
      plot_filename: `str` indicates the filename to save the plot. Do not save
        the file if `None`.
      scale_transformers: `dict` of monotone functions that transform the scale
        of coordinates in `trials`.
      color_map: A function turns real values into color codes.
      alpha: Transparency.
      beta: Tradeoff between linear and quadratic/cubic interpolation.
    """
    if scale_transformers is None:
        scale_transformers = {}
    trial = trials[0]
    assert "score" in trial
    keys = trial.keys()
    # Assert at least 2 coordinates and a `score`
    assert len(keys) > 2
    for t in trials:
        assert t.keys() == keys

    # Digitize the values of categorical keys.
    categorical_keys = []
    for key in keys:
        type_assertion = (isinstance(trial[key], str) or
                          isinstance(trial[key], (int, np.integer)) or
                          isinstance(trial[key], float))
        assert type_assertion
        if isinstance(trial[key], str):
            categorical_keys.append(key)
    digit_map = {}
    for key in categorical_keys:
        digit_map[key] = {}
        values = sorted(list(set([t[key] for t in trials])))
        for v in values:
            digit_map[key][v] = len(digit_map[key])

    def _transform(v, key):
        """Transforms the value by digitizing and re-scaling."""
        if key in categorical_keys:
            v = digit_map[key][v]
        if key in scale_transformers:
            v = scale_transformers[key](v)
        return v

    # Align values of each coordinate
    align_min = {}
    align_max = {}
    for key in keys:
        values = [_transform(t[key], key) for t in trials]
        align_min[key] = min(values)
        align_max[key] = max(values)
        if align_max[key] == align_min[key]:
            align_max[key] += 1
            align_min[key] -= 1

    def _align(v, key):
        return (v - align_min[key]) / float(align_max[key] - align_min[key])

    # Plot axeses and ticks
    coordinates = [key for key in keys if key != "score"]
    coordinates = ["score"] + coordinates
    x = range(len(coordinates))
    fig, ax = plt.subplots(figsize=((len(x) + 0.9) * 2, 5))
    ax.set_xlim(-0.2, len(x) - 0.3)
    plt.xticks(x, coordinates)
    plt.xlabel("parameter")
    plt.yticks([], [])
    axis_color = sns.xkcd_palette(["dark grey"])[0]
    for x_i in x:
        coord = coordinates[x_i]
        ax.axvline(x=x_i, color=axis_color, linewidth=1)
        raw_values = sorted(list(set([t[coord] for t in trials])))
        n = len(raw_values)
        if n <= 10:
            index = range(n)
        else:
            index = [int(i * (n - 1) / 5.) for i in range(6)]
        ticks = [raw_values[i] for i in index]
        for tick in ticks:
            y_tick = _align(_transform(tick, coord), coord)
            label = tick
            if isinstance(label, float):
                label = "%.5g" % label
            elif isinstance(label, (int, np.integer)):
                label = "%d" % label
            ax.text(x_i, y_tick, label, size=15)
            ax.scatter(x_i, y_tick, color=axis_color, s=15)

    # Interpolate and smooth with spline
    x_interpolate = np.linspace(x[0], x[-1], len(x) * 100)
    for t in trials:
        y = []
        for coord in coordinates:
            v = _align(_transform(t[coord], coord), coord)
            y.append(v)
        score = _align(_transform(t["score"], "score"), "score")
        score *= 255
        linear = interp1d(x, y)
        if len(x) == 3:
            interpolation_kind = "quadratic"
        else:
            interpolation_kind = "cubic"
        spline = interp1d(x, y, kind=interpolation_kind)
        y_interpolate = beta * spline(x_interpolate) + (
            1 - beta) * linear(x_interpolate)
        ax.plot(
            x_interpolate,
            y_interpolate,
            color=color_map(score),
            alpha=alpha,
            linewidth=1)
    if plot_filename is not None:
        fig.savefig(plot_filename)