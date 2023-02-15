import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.cm import get_cmap
from cycler import cycler
import h5py
import scipy.io
import torch
from scipy.optimize import curve_fit

import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.colors import LogNorm


# print(plt.style.available)


# import plotly

# mpl.style.use("fast")

# mpl.rcParams["axes.prop_cycle"] = cycler(
# color=plt.style.library["ggplot"]["axes.prop_cycle"].by_key()["color"])

plt.style.use(["tableau-colorblind10", "seaborn-paper"])
mpl.rcParams["axes.prop_cycle"] = cycler(
    color=[
        "#3D65A5",
        "#E57A77",
        "#7CA1CC",
        "#F05039",
        "#1F449C",
        "#A8B6CC",
        "#EEBAB4",
    ]  # ["#F05039", "#E57A77", "#EEBAB4", "#1F449C", "#3D65A5", "#7CA1CC", "#A8B6CC"]
) + cycler(
    linestyle=["-", "--", "-.", ":", "-", "--", "-."]
)  # From: https://www.datylon.com/blog/data-visualization-for-colorblind-readers  and https://ranocha.de/blog/colors/ , respectively
# cycler(color=plt.style.library["tab10"]["axes.prop_cycle"].by_key()["color"])
# ggplot seaborn-colorblind
DEFAULT_FIGSIZE = (5.69, 3.9)
w = 1
mpl.rcParams["axes.linewidth"] = w
mpl.rcParams["xtick.major.width"] = w
mpl.rcParams["xtick.minor.width"] = w / 2
mpl.rcParams["ytick.major.width"] = w
mpl.rcParams["ytick.minor.width"] = w / 2

mpl.rcParams["lines.markersize"] = 6
mpl.rcParams["lines.linewidth"] = 3
mpl.rcParams["font.size"] = 12
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["figure.titlesize"] = 20
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["figure.figsize"] = DEFAULT_FIGSIZE  # (8, 6)
mpl.rcParams["figure.constrained_layout.use"] = True
# mpl.rcParams["axes.formatter.use_mathtext"] = True
# mpl.rcParams["text.usetex"] = True  # Use Latex
# mpl.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Palatino"],
#     }
# )

# Try this to get font similar to latex
"""
mpl.rcParams['legend.loc'] = "upper_right" # Suggestion
mpl.rcParams['']
mpl.rcParams['']
"""


# Borrow some code to create colormap from color palette https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72 and https://www.delftstack.com/howto/matplotlib/custom-colormap-using-python-matplotlib/#use-rgba-values-to-create-custom-listed-colormap-in-python


def choose_formatter(incscape=True):
    if incscape:
        mpl.rcParams["svg.fonttype"] = "none"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["axes.unicode_minus"] = False
        return
    else:
        mpl.rcParams["axes.formatter.use_mathtext"] = True
        mpl.rcParams["text.usetex"] = True  # Use Latex
        mpl.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Palatino"],
            }
        )
    return


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values"""
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values"""
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map"""
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap("XRDCT_palette_cmp", segmentdata=cdict, N=256)
    return cmp


# Diverging by using divnorm, Use TwoSlopeNorm to define min, center and max of data. Add this together with the colormap to the contourf plot.

XRDCT_palette_cmp = get_continuous_cmap(
    ["#1F449C", "#3D65A5", "#7CA1CC", "#A8B6CC", "#EEBAB4", "#E57A77", "#F05039"]
    # Might be a good choice. Need consultants # ["#F05039", "#E57A77", "#EEBAB4", "#1F449C", "#3D65A5", "#7CA1CC", "#A8B6CC"]
    # # [     "#E57A77","#EEBAB4","#7CA1CC",]
)

# XRDCT_cyclic_cmp = get_continuous_cmap(
#     [
#         "#1F449C",
#         "#3D65A5",
#         "#7CA1CC",
#         "#A8B6CC",
#         "#EEBAB4",
#         "#E57A77",
#         "#F05039",
#         "#F05039",
#         "#E57A77",
#         "#EEBAB4",
#         "#A8B6CC",
#         "#7CA1CC",
#         "#3D65A5",
#         "#1F449C",
#     ]
# )

XRDCT_cyclic_cmp = get_continuous_cmap(
    [
        "#1F449C",
        "#A8B6CC",
        "#7CA1CC",
        "#E57A77",
        "#F05039",
        "#EEBAB4",
        "A8B6CC",
        "#1F449C",
    ]
)
