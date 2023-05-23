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
from matplotlib.gridspec import GridSpec


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

mpl.rcParams["lines.markersize"] = 6 * w
mpl.rcParams["lines.linewidth"] = 2 * w
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

XRDCT_diverging_cmp = get_continuous_cmap(["#1F449C", "#FFFFFF", "#F05039"])


from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as PathEffects


def add_scalebar(ax, **kwargs):
    if "scalebar_kwargs" not in kwargs:
        size = 25 / (0.2 * 930) * 1350

        scale_kwargs = {
            "size": size,
            "label": f"25.0 mm",
            "color": "white",
            "loc": 4,
            "frameon": False,
            "size_vertical": 8,
            "label_top": False,
            # "font_properties": {"size": 12}
        }
    else:
        scale_kwargs = kwargs["scalebar_kwargs"]

    scalebar0 = AnchoredSizeBar(ax.transData, **scale_kwargs)
    scalebar0.txt_label._text.set_path_effects(
        [PathEffects.withStroke(linewidth=2, foreground="black", capstyle="round")]
    )
    ax.add_artist(scalebar0)
    return ax


def plot_slice_grid(
    imgs: list,
    titles: list,
    suptitle=None,
    savefile=None,
    savefig=False,
    fig=None,
    folder="Golden Angle",
    bar=False,
    cm=None,
    fs=None,
    ns=None,
    scalebar_kwargs=None,
):
    choose_formatter(False)
    if cm is None:
        cmap = "gray"
    else:
        cmap = cm

    def grids(n):
        if n <= 3:
            return 1, n
        n_sqrt = np.sqrt(n)
        if n_sqrt % 1 == 0:
            return int(n_sqrt), int(n_sqrt)
        else:
            assert n % 3 == 0
            return n // 3, 3

    n1, n2 = ns if ns is not None else grids(len(imgs))

    f1, f2 = fs if fs is not None else (n1, n2)

    fig = plt.figure(figsize=(f2 * DEFAULT_FIGSIZE[1], f1 * DEFAULT_FIGSIZE[1]))
    gs = GridSpec(n1, n2, figure=fig, wspace=0, hspace=0.0)
    axes = np.array(
        [[fig.add_subplot(gs[j, i]) for i in range(n2)] for j in range(n1)]
    ).reshape(-1)

    for i in range(len(imgs)):
        im = axes[i].imshow(imgs[i], cmap=cmap)
        axes[i].set_title(titles[i])
        axes[i].axis("off")
        if bar and scalebar_kwargs is not None:
            axes[i] = add_scalebar(axes[i], scalebar_kwargs=scalebar_kwargs)
        elif bar:
            axes[i] = add_scalebar(axes[i])
    if suptitle is not None:
        plt.suptitle(suptitle)
    if cm is not None:
        limits = [np.abs(img).max() for img in imgs]
        vmin = np.min([img.min() for img in imgs])
        vmax = np.max([img.max() for img in imgs])

        vmin = -np.max(limits)
        vmax = np.max(limits)

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=norm,
                cmap=cmap,
            ),
            orientation="horizontal",
            ax=[axes[i] for i in range(len(imgs))],
            label="Intensity Difference [a.u.]",
        )

        for i in range(len(imgs)):
            im = axes[i].imshow(imgs[i], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i].set_title(titles[i])
            axes[i].axis("off")

    if savefig:
        plt.savefig(f"../Results/{folder}/{savefile}.pdf", format="pdf")

    plt.show()


def plot_line_profile(
    imgs,
    labels,
    crossections,
    idxs=None,
    savefile=None,
    savefig=False,
    folder="Golden Angle",
    title="Line Profile of RoI",
):
    if idxs is None:
        x1, x2, y1, y2 = 300, 300, 300, 400
    else:
        x1, x2, y1, y2 = idxs

    fig = plt.figure(figsize=(DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1]))
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1:])
    # ax3 = fig.add_subplot(gs[3:5])

    ax1.imshow(crossections, cmap="gray")
    ax1.plot([y1, y2], [x1, x2], c="red", linewidth=2, alpha=0.75, label="RoI")
    ax1.set_title(title)
    ax1.set_axis_off()

    for i, (img, lab) in enumerate(zip(imgs, labels)):
        ax2.plot(img[x1, y1:y2], linewidth=1.5, alpha=0.75, label=lab)

    ax2.plot(
        crossections[x1, y1:y2],
        "-",
        linewidth=1.5,
        c="black",
        alpha=1,
        label="Ground Truth",
    )
    ax2.legend(
        bbox_to_anchor=(0.5, -0.25),
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=False,
    )

    ax2.set_xlabel("Pixel")
    ax2.set_ylabel("Normalised Intensity [a.u.]")

    if savefig:
        plt.savefig(f"../Results/{folder}/{savefile}.pdf", format="pdf")

    plt.show()


def plot_fsc(
    outputs,
    outputs_enhanced,
    uniques,
    fq_keys,
    filter=50,
    ylabel1="FSC Input",
    ylabel2="FSC Output",
    save=False,
    folder="Hourglass4D",
    savefile="FSC",
    xlim=None,
):
    # fig, (ax, axe) = plt.subplots(1, 2)
    fig = plt.figure(figsize=(DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1]))
    gs = fig.add_gridspec(5, 2)
    ax = fig.add_subplot(gs[1:, 0])
    axe = fig.add_subplot(gs[1:, 1])

    ax.set_xlabel("Spatial Frequency")
    ax.set_ylabel(ylabel1)
    # ax.set_xlim(0, 100)
    axe.set_xlabel("Spatial Frequency")
    axe.set_ylabel(ylabel2)

    for i, (fscr, uniques) in enumerate(outputs):
        # ax.plot(uniques, fscr.real, label="13 Projections")
        ax.plot(
            uniques[filter:-filter],
            [
                np.mean(fscr.real[i - filter : i + filter])
                for i in range(filter, len(fscr.real) - filter)
            ],
            label=fq_keys[i],
        )

    # ax.legend()

    for i, (fscr, uniques) in enumerate(outputs_enhanced):
        axe.plot(
            uniques[filter:-filter],
            [
                np.mean(fscr.real[i - filter : i + filter])
                for i in range(filter, len(fscr.real) - filter)
            ],
            label=fq_keys[i],
        )

    # axe.legend()

    ax.set_ylim(0, 1)
    axe.set_ylim(0, 1)

    # ax.legend(
    #     bbox_to_anchor=(0.35, 1.25),
    #     loc="upper left",
    #     ncol=3,
    #     fancybox=True,
    #     shadow=False,
    # )

    # fig.add_artist(
    #     ax.legend(
    #         # bbox_to_anchor=(0.35, 1.25),
    #         # loc="upper left",
    #         ncol=3,
    #         fancybox=True,
    #         shadow=False,
    #     )
    # )

    handles = ax.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1]

    ledge = fig.add_subplot(gs[0, :])  # .axis("off")
    ledge.axis("off")
    ledge.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0.5),
        loc="center",
        ncol=5,
        fancybox=True,
        shadow=False,
    )

    if xlim is not None:
        ax.set_xlim(xlim)
        axe.set_xlim(xlim)

    if save:
        plt.savefig(rf"../Results/{folder}/{savefile}.pdf", format="pdf")

    plt.show()
