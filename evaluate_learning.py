import pandas as pd
import os
from matplotlib import pyplot as plt

import matplotlib as mpl
mpl.use("pgf")
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 20,
    # "pgf.rcfonts": False,
    "legend.handlelength": 1.5,
})

def learning_curve(n_avg, file: str):
    df = pd.read_csv(file, delimiter=",", skiprows=1)
    # print(df.head())

    df = df.rolling(window=n_avg).mean()

    df_cleaned = df.dropna()
    
    ncols = 1
    nrows = 1
    k = 1
    figsize = (5 * k * ncols, 4 * k * nrows)
    dpi = 1200
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi, constrained_layout = True)
    ax.grid()
    ax.plot(df_cleaned.index, df_cleaned["r"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average return")

    plt.savefig(os.path.dirname(file) + "\\learning_curve.png")
    plt.close()
    return

def learning_curves_all_in_one(n_avg, files: list[str], titles: list[str], figpath: str):
    
    df_list = []
    for file in files:
        df = pd.read_csv(file, delimiter=",", skiprows=1)
        df_list.append(df)
    # print(df.head())
    
    df_cleaned_list = []
    for df in df_list:
        df = df.rolling(window=n_avg).mean()
        df_cleaned = df.dropna()
        df_cleaned_list.append(df_cleaned)
    
    ncols = 3
    nrows = 1
    k = 1
    figsize = (5 * k * ncols, 4 * k * nrows)
    dpi = 1200
    fig, ax = plt.subplots(sharey =  True, ncols = ncols, nrows = nrows, figsize = figsize, dpi = dpi, constrained_layout = True)
    for axis, title, df_cleaned in zip(ax, titles, df_cleaned_list):
        axis.grid()
        axis.plot(df_cleaned.index, df_cleaned["r"])
        axis.set_xlabel("Episode")
        axis.set_ylabel("Average return")

        axis.set_ylim(-20, -3)
        axis.set_xlim(0, 20000)
        axis.set_title(title)

        axis.set_yticks([-20, -15, -10, -5])
        # axis.set_yticklabels([10, 15, 20, 25, 30], fontsize = 14)
        axis.set_yticklabels([-20, -15, -10, -5])

    plt.savefig(figpath + "\\learning_curves.pgf", format = "pgf")
    # plt.savefig(figpath + "\\learning_curves.png")
    plt.close()
    return

if __name__ == "__main__":

    file_list = [
        # "D:\\recipe_opt\\data\\poly_reactor\\agent\\SB3_SAC_2_hybrid\\monitor.csv",
        # "D:\\recipe_opt\\data\\poly_reactor\\agent\\SB3_SAC_2_time\\monitor.csv",
        "data\\poly_reactor\\agent\\SB3_SAC_2_hybrid\\monitor.csv",
        "data\\poly_reactor\\agent\\SB3_SAC_2_time\\monitor.csv",
        "data\\poly_reactor\\agent\\SB3_SAC_7_det\\monitor.csv"
    ]
    title_list = [
        "Maximize product mass",
        "Minimize batch time",
        "Hybrid"
    ]
    # file = f"D:\\recipe_opt\\data\\poly_reactor\\agent\\SB3_SAC_2_time\\monitor.csv"
    # file = f"D:\\recipe_opt\\data\\poly_reactor\\agent\\SB3_SAC_2_hybrid\\monitor.csv"
    file = "data\\poly_reactor\\agent\\SB3_SAC_7_det\\monitor.csv"

    n_avg = 100
    # learning_curve(n_avg, file)

    learning_curves_all_in_one(n_avg, file_list, title_list, "data\\poly_reactor\\agent\\")
