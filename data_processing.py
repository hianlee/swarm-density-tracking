import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib import colors
import numpy as np
from matplotlib import cm
import pylatex as ptx

fig1 = True
fig2 = True
fig3 = True
fig4a = True
fig4b = True
fig4c = True
fig5 = True
fig6 = True
fig7 = True
fig8 = True
fig9 = True
figa1b = True

if fig1:
    data1 = pd.read_csv("data/1T_k10_50agents_v15_score.csv", header=None)
    data2 = pd.read_csv("data/1T_k10_40agents_v15_score.csv", header=None)
    data3 = pd.read_csv("data/1T_k10_30agents_v15_score.csv", header=None)
    data4 = pd.read_csv("data/1T_k10_20agents_v15_score.csv", header=None)

    data1a = pd.read_csv("data/1T_k15_50agents_v15_score.csv", header=None)
    data2a = pd.read_csv("data/1T_k15_40agents_v15_score.csv", header=None)
    data3a = pd.read_csv("data/1T_k15_30agents_v15_score.csv", header=None)
    data4a = pd.read_csv("data/1T_k15_20agents_v15_score.csv", header=None)

    data1b = pd.read_csv("data/1T_k25_50agents_v15_score.csv", header=None)
    data2b = pd.read_csv("data/1T_k25_40agents_v15_score.csv", header=None)
    data3b = pd.read_csv("data/1T_k25_30agents_v15_score.csv", header=None)

    plt.rcParams.update({'font.size': 20, 'lines.linewidth': 3})
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    areas = np.logspace(0.6, 2.65, num=125) ** 2
    rhos1 = 50 / areas
    rhos2 = 40 / areas
    rhos3 = 30 / areas
    rhos4 = 20 / areas

    plt.subplots_adjust(left=0.05, bottom=0.11, right=0.97, top=0.95)

    axs[0].plot(rhos1, data1, color='tab:red', label='$N=50$')
    axs[0].plot(rhos2, data2, color='tab:blue', label='$N=40$')
    axs[0].plot(rhos3, data3, color='tab:green', label='$N=30$')
    axs[0].plot(rhos4, data4, color='tab:purple', label='$N=20$')

    axs[0].set_xscale('log')
    axs[0].set_xlabel('Swarm Density ($\\rho$)')
    axs[0].set_ylabel('Tracking Performance ($\Xi$)')
    axs[0].set_title('$k=10$', fontsize=18)
    axs[0].legend(loc='center left').get_frame().set_boxstyle('Round', pad=0.2, rounding_size=1)
    axs[0].set_xlim(0.00001, 4)
    axs[0].set_ylim(0, 1.15)
    props = dict(boxstyle='round', color='white')
    axs[0].text(0.00007, 1.05, 'Low', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[0].text(0.003, 1.05, 'Transition', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[0].text(0.35, 1.05, 'High', backgroundcolor='white', bbox=props, fontsize=17.5)

    axs[0].axvspan(0, 0.002, hatch='//', facecolor='white', edgecolor='grey', linewidth=2)
    axs[0].axvspan(0.1, 4, hatch='\\\\', facecolor='white', edgecolor='grey', linewidth=2)

    axs[1].plot(rhos1, data1a, color='tab:red', label='$N=50$')
    axs[1].plot(rhos2, data2a, color='tab:blue', label='$N=40$')
    axs[1].plot(rhos3, data3a, color='tab:green', label='$N=30$')
    axs[1].plot(rhos4, data4a, color='tab:purple', label='$N=20$')

    axs[1].set_xscale('log')
    axs[1].set_xlabel('Swarm Density ($\\rho$)')
    axs[1].set_title('$k=15$', fontsize=18)

    axs[1].set_xlim(0.00001, 4)
    axs[1].set_ylim(0, 1.15)
    axs[1].text(0.00007, 1.05, 'Low', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[1].text(0.0045, 1.05, 'Transition', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[1].text(0.5, 1.05, 'High', backgroundcolor='white', bbox=props, fontsize=17.5)

    axs[1].axvspan(0, 0.002, hatch='//', facecolor='white', edgecolor='grey', linewidth=2)
    axs[1].axvspan(0.2, 4, hatch='\\\\', facecolor='white', edgecolor='grey', linewidth=2)

    axs[2].plot(rhos1, data1b, color='tab:red', label='$N=50$')
    axs[2].plot(rhos2, data2b, color='tab:blue', label='$N=40$')
    axs[2].plot(rhos3, data3b, color='tab:green', label='$N=30$')

    axs[2].set_xscale('log')
    axs[2].set_xlabel('Swarm Density ($\\rho$)')
    axs[2].set_title('$k=25$', fontsize=18)
    axs[2].text(0.00007, 1.05, 'Low', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[2].text(0.006, 1.05, 'Transition', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[2].text(0.65, 1.05, 'High', backgroundcolor='white', bbox=props, fontsize=17.5)

    axs[2].set_xlim(0.00001, 4)
    axs[2].set_ylim(0, 1.15)

    axs[2].axvspan(0, 0.002, hatch='//', facecolor='white', edgecolor='grey', linewidth=2)
    axs[2].axvspan(0.35, 4, hatch='\\\\', facecolor='white', edgecolor='grey', linewidth=2)

    plt.show()

if fig2:
    data1 = pd.read_csv("data/50agents_ne_k5_v15_density.csv", header=None)
    data2 = pd.read_csv("data/50agents_ne_k15_v15_density.csv", header=None)
    data3 = pd.read_csv("data/50agents_ne_k20_v15_density.csv", header=None)
    data4 = pd.read_csv("data/50agents_ne_k35_v15_density.csv", header=None)

    data1b = pd.read_csv("data/50agents_ne_k5_v15_score.csv", header=None)
    data2b = pd.read_csv("data/50agents_ne_k15_v15_score.csv", header=None)
    data3b = pd.read_csv("data/50agents_ne_k20_v15_score.csv", header=None)
    data4b = pd.read_csv("data/50agents_ne_k35_v15_score.csv", header=None)

    areas = np.logspace(0.6, 2.65, num=125) ** 2
    rhos = 50 / areas

    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    plt.subplots_adjust(left=0.07, bottom=0.05, right=0.93, top=0.95)

    axs[0, 0].plot(rhos, data1b, color='tab:red')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title('$k=5$')
    axs[0, 0].set_ylabel('Tracking Performance ($\Xi$)', color='tab:red')

    diff1 = data1[0] - rhos
    ax1 = axs[0, 0].twinx()
    ax1.plot(rhos, diff1, color='tab:blue')
    ax1.set_yscale('log')

    axs[0, 1].plot(rhos, data2b, color='tab:red')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_title('$k=15$')

    diff2 = data2[0] - rhos
    ax2 = axs[0, 1].twinx()
    ax2.plot(rhos, diff2, color='tab:blue')
    ax2.set_ylabel('Local Density Difference ($\Delta\\rho$)', color='tab:blue')
    ax2.set_yscale('log')

    axs[1, 0].plot(rhos, data3b, color='tab:red')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_title('$k=20$')
    axs[1, 0].set_xlabel('Swarm Density $\\rho$')
    axs[1, 0].set_ylabel('Tracking Performance ($\Xi$)', color='tab:red')

    diff3 = data3[0] - rhos
    ax3 = axs[1, 0].twinx()
    ax3.plot(rhos, diff3, color='tab:blue')
    ax3.set_yscale('log')

    axs[1, 1].plot(rhos, data4b, label='Tracking Performance', color='tab:red')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlabel('Swarm Density ($\\rho$)')
    axs[1, 1].set_title('$k=35$')

    diff4 = data4[0] - rhos
    ax4 = axs[1, 1].twinx()
    ax4.plot(rhos, diff4, color='tab:blue')
    ax4.set_ylabel('Local Density Difference ($\Delta\\rho$)', color='tab:blue')
    ax4.set_yscale('log')

    plt.show()

if fig3:
    data1a = pd.read_csv("data/50agents_ne_k5_v15_engagement.csv", header=None)
    data2a = pd.read_csv("data/50agents_ne_k15_v15_engagement.csv", header=None)
    data3a = pd.read_csv("data/50agents_ne_k20_v15_engagement.csv", header=None)
    data4a = pd.read_csv("data/50agents_ne_k35_v15_engagement.csv", header=None)

    data1b = pd.read_csv("data/50agents_ne_k5_v15_score.csv", header=None)
    data2b = pd.read_csv("data/50agents_ne_k15_v15_score.csv", header=None)
    data3b = pd.read_csv("data/50agents_ne_k20_v15_score.csv", header=None)
    data4b = pd.read_csv("data/50agents_ne_k35_v15_score.csv", header=None)

    areas = np.logspace(0.6, 2.65, num=125) ** 2
    rhos = 50 / areas

    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    plt.subplots_adjust(left=0.07, bottom=0.05, right=0.93, top=0.95)
    props = dict(boxstyle='round', color='white')

    exp1 = 1 - data1a[0]
    axs[0, 0].plot(rhos, exp1, color='tab:red', label='Exploration Ratio ($\Theta$)')
    axs[0, 0].plot(rhos, data1b, color='tab:blue', label='Tracking Performance ($\Xi$)')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title('$k=5$')
    axs[0, 0].set_ylim(-0.05, 1.14,)
    axs[0, 0].set_xlim(0.0002, 4)
    axs[0, 0].axvspan(0, 0.002, hatch='//', facecolor='white', edgecolor='grey', linewidth=2)
    axs[0, 0].axvspan(0.24, 30, hatch='\\\\', facecolor='white', edgecolor='grey', linewidth=2)
    axs[0, 0].text(0.0004, 1.05, 'Low', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[0, 0].text(0.006, 1.05, 'Transition', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[0, 0].text(0.6, 1.05, 'High', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[0, 0].legend(loc='lower right', prop={'size': 8.5}).get_frame().set_boxstyle('Round', pad=0.2, rounding_size=1)

    exp2 = 1 - data2a[0]
    axs[0, 1].plot(rhos, exp2, color='tab:red')
    axs[0, 1].plot(rhos, data2b, color='tab:blue')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_title('$k=15$')
    axs[0, 1].set_ylim(-0.05, 1.14)
    axs[0, 1].set_xlim(0.0002, 4)
    axs[0, 1].axvspan(0, 0.002, hatch='//', facecolor='white', edgecolor='grey', linewidth=2)
    axs[0, 1].axvspan(0.2, 30, hatch='\\\\', facecolor='white', edgecolor='grey', linewidth=2)
    axs[0, 1].text(0.0004, 1.05, 'Low', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[0, 1].text(0.006, 1.05, 'Transition', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[0, 1].text(0.6, 1.05, 'High', backgroundcolor='white', bbox=props, fontsize=17.5)

    exp3 = 1 - data3a[0]
    axs[1, 0].plot(rhos, exp3, color='tab:red')
    axs[1, 0].plot(rhos, data3b, color='tab:blue')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_title('$k=20$')
    axs[1, 0].set_xlabel('Swarm Density ($\\rho$)')
    axs[1, 0].set_ylim(-0.05, 1.14)
    axs[1, 0].set_xlim(0.0002, 4)
    axs[1, 0].axvspan(0, 0.002, hatch='//', facecolor='white', edgecolor='grey', linewidth=2)
    axs[1, 0].axvspan(0.35, 30, hatch='\\\\', facecolor='white', edgecolor='grey', linewidth=2)
    axs[1, 0].text(0.0004, 1.05, 'Low', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[1, 0].text(0.008, 1.05, 'Transition', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[1, 0].text(0.7, 1.05, 'High', backgroundcolor='white', bbox=props, fontsize=17.5)

    exp4 = 1 - data4a[0]
    axs[1, 1].plot(rhos, exp4, label='Exploration Ratio ($\Theta$)', color='tab:red')
    axs[1, 1].plot(rhos, data4b, label='Tracking Performance ($\Xi$)', color='tab:blue')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlabel('Swarm Density ($\\rho$)')
    axs[1, 1].set_title('$k=35$')
    axs[1, 1].set_ylim(-0.05, 1.14)
    axs[1, 1].set_xlim(0.0002, 4)
    axs[1, 1].axvspan(0, 0.002, hatch='//', facecolor='white', edgecolor='grey', linewidth=2)
    axs[1, 1].axvspan(0.6, 30, hatch='\\\\', facecolor='white', edgecolor='grey', linewidth=2)
    axs[1, 1].text(0.0004, 1.05, 'Low', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[1, 1].text(0.008, 1.05, 'Transition', backgroundcolor='white', bbox=props, fontsize=17.5)
    axs[1, 1].text(0.85, 1.05, 'High', backgroundcolor='white', bbox=props, fontsize=17.5)

    plt.show()

if fig4a:
    data1 = pd.read_csv("data/50agents_ne_k5_v15_score_e10.csv", header=None)
    data2 = pd.read_csv("data/50agents_ne_k10_v15_score_e10.csv", header=None)
    data3 = pd.read_csv("data/50agents_ne_k20_v15_score_e10.csv", header=None)
    data4 = pd.read_csv("data/50agents_ne_k30_v15_score_e10.csv", header=None)
    data5 = pd.read_csv("data/50agents_ne_k40_v15_score_e10.csv", header=None)

    areas = np.logspace(0.6, 2.65, num=125) ** 2
    rhos = 50 / areas

    plt.rcParams.update({'font.size': 22, 'lines.linewidth': 3})

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.125, right=0.95, top=0.9, bottom=0.1)

    ax.plot(rhos, data1, label='$k=5$', color='tab:red')
    ax.plot(rhos, data2, label='$k=10$', color='tab:blue')
    ax.plot(rhos, data3, label='$k=20$', color='tab:green')
    ax.plot(rhos, data4, label='$k=30$', color='tab:purple')
    ax.plot(rhos, data5, label='$k=40$', color='tab:brown')

    ax.set_xscale('log')
    ax.set_xlabel('Swarm Density ($\\rho$)')
    ax.set_ylabel('Tracking Performance ($\Xi$)')
    ax.legend(loc='lower right')

    ax.axvspan(0.01, 0.05, alpha=0.5, color='slategray')

    plt.show()

if fig4b:
    data1a = pd.read_csv("data/50agents_ne_k5_v15_engagement_e10.csv", header=None)
    data2a = pd.read_csv("data/50agents_ne_k10_v15_engagement_e10.csv", header=None)
    data3a = pd.read_csv("data/50agents_ne_k20_v15_engagement_e10.csv", header=None)
    data4a = pd.read_csv("data/50agents_ne_k30_v15_engagement_e10.csv", header=None)
    data5a = pd.read_csv("data/50agents_ne_k40_v15_engagement_e10.csv", header=None)

    areas = np.logspace(0.6, 2.65, num=125) ** 2
    rhos = 50 / areas

    data1a = 1 - data1a
    data2a = 1 - data2a
    data3a = 1 - data3a
    data4a = 1 - data4a
    data5a = 1 - data5a

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.rcParams.update({'font.size': 18, 'lines.linewidth': 3})

    ax.plot(rhos, data1a, label='$k=5$', color='tab:red')
    ax.plot(rhos, data2a, label='$k=10$', color='tab:blue')
    ax.plot(rhos, data3a, label='$k=20$', color='tab:green')
    ax.plot(rhos, data4a, label='$k=30$', color='tab:purple')
    ax.plot(rhos, data5a, label='$k=40$', color='tab:brown')

    ax.set_xscale('log')
    ax.set_ylabel('Exploration Ratio ($\Theta$)', fontsize=21)
    ax.set_xlabel('Swarm Density ($\\rho$)', fontsize=21)
    ax.legend(loc='lower left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()

if fig4c:
    data1 = pd.read_csv("data/50agents_ne25_v15_nm_score.csv")
    data2 = pd.read_csv("data/50agents_ne30_v15_nm_score.csv")
    data3 = pd.read_csv("data/50agents_ne35_v15_nm_score.csv")

    data1e = pd.read_csv("data/50agents_ne25_v15_nm_engagement.csv")
    data2e = pd.read_csv("data/50agents_ne30_v15_nm_engagement.csv")
    data3e = pd.read_csv("data/50agents_ne35_v15_nm_engagement.csv")

    alpha = [(k / 40 + 0.5) / 1.5 for k in np.linspace(2, 40, 39)]
    alpha = alpha[::-1]

    data1e = 1 - data1e
    data2e = 1 - data2e
    data3e = 1 - data3e

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.rcParams.update({'font.size': 18, 'lines.markersize' : 10})

    def scatter(ax, x, y, color, alpha_arr, **kwarg):
        r, g, b = colors.to_rgb(color)
        color = [(r, g, b, alpha) for alpha in alpha_arr]
        ax.scatter(x, y, c=color, **kwarg)


    ax.set_xlim(0.35, 1.05)
    ax.set_ylim(0, 1.25)

    scatter(ax, data1e.iloc[::-1], data1.iloc[::-1], 'darkred', alpha, label='$\\rho = 0.080$')
    scatter(ax, data2e.iloc[::-1], data2.iloc[::-1], 'darkblue', alpha, label='$\\rho = 0.056$')
    scatter(ax, data3e.iloc[::-1], data3.iloc[::-1], 'darkgreen', alpha, label='$\\rho = 0.041$')

    plt.axvline(x=0.675, linestyle='dotted', color='black')
    plt.arrow(0.6, 1.2, -.15, 0, head_width=0.02, facecolor='k')
    plt.text(0.385, 1.13, 'Exploration-Limited')
    plt.arrow(0.775, 1.2, .15, 0, head_width=0.02, facecolor='k')
    plt.text(0.745, 1.13, 'Exploitation-Limited')

    style = "Simple, tail_width=1.25, head_width=10, head_length=15"
    kw = dict(arrowstyle=style, color="k", linestyle='--')
    curved_arrow = patches.FancyArrowPatch((0.775, 0.55), (0.6, 0.575), connectionstyle="arc3,rad=.5", **kw) # Short Arrow

    plt.gca().add_patch(curved_arrow)


    def split_arrow(arrow, color_tail="C0", color_head="C0",
                    ls_tail="-", ls_head="-", lw_tail=1.5, lw_head=1.5):
        v1 = arrow.get_path().vertices[0:3, :]
        c1 = arrow.get_path().codes[0:3]
        p1 = path.Path(v1, c1)
        pp1 = patches.PathPatch(p1, color=color_tail, linestyle=ls_tail,
                                fill=False, lw=lw_tail)
        arrow.axes.add_patch(pp1)

        v2 = arrow.get_path().vertices[3:8, :]
        c2 = arrow.get_path().codes[3:8]
        c2[0] = 1
        p2 = path.Path(v2, c2)
        pp2 = patches.PathPatch(p2, color=color_head, lw=lw_head, linestyle=ls_head)
        arrow.axes.add_patch(pp2)
        arrow.remove()


    split_arrow(curved_arrow, color_tail="black", color_head="black", ls_tail="--", lw_tail=3)
    ax.text(0.685, 0.56, '$k \\nearrow $', fontsize=15)

    ax.arrow(0.635, 0.77, 0.025, -0.04, head_width=0.0075, facecolor='k')
    plt.text(0.605, 0.775, '$k=16$', fontsize=13)
    ax.arrow(0.635, 0.9, 0.028, -0.045, head_width=0.0075, facecolor='k')
    plt.text(0.605, 0.91, '$k=12$', fontsize=13)
    ax.arrow(0.645, 1.015, 0.025, -0.04, head_width=0.0075, facecolor='k')
    plt.text(0.605, 1.025, '$k=9$', fontsize=13)

    ax.set_ylabel('Tracking Performance ($\Xi$)', fontsize=21)
    ax.set_xlabel('Exploration Ratio ($\Theta$)', fontsize=21)
    ax.legend(loc='lower left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()


if fig5:
    data1 = pd.read_csv("data/1T_k10_50agents_v15_engagement.csv", header=None)
    data2 = pd.read_csv("data/1T_k10_40agents_v15_engagement.csv", header=None)
    data3 = pd.read_csv("data/1T_k10_30agents_v15_engagement.csv", header=None)
    data4 = pd.read_csv("data/1T_k10_20agents_v15_engagement.csv", header=None)

    data1a = pd.read_csv("data/1T_k15_50agents_v15_engagement.csv", header=None)
    data2a = pd.read_csv("data/1T_k15_40agents_v15_engagement.csv", header=None)
    data3a = pd.read_csv("data/1T_k15_30agents_v15_engagement.csv", header=None)
    data4a = pd.read_csv("data/1T_k15_20agents_v15_engagement.csv", header=None)

    data1b = pd.read_csv("data/1T_k25_50agents_v15_engagement.csv", header=None)
    data2b = pd.read_csv("data/1T_k25_40agents_v15_engagement.csv", header=None)
    data3b = pd.read_csv("data/1T_k25_30agents_v15_engagement.csv", header=None)

    data1 = 1 - data1
    data2 = 1 - data2
    data3 = 1 - data3
    data4 = 1 - data4

    data1a = 1 - data1a
    data2a = 1 - data2a
    data3a = 1 - data3a
    data4a = 1 - data4a

    data1b = 1 - data1b
    data2b = 1 - data2b
    data3b = 1 - data3b

    plt.rcParams.update({'font.size': 20, 'lines.linewidth': 3})
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(left=0.05, bottom=0.12, right=0.97, top=0.95)

    areas = np.logspace(0.6, 2.65, num=125) ** 2
    rhos1 = 50 / areas
    rhos2 = 40 / areas
    rhos3 = 30 / areas
    rhos4 = 20 / areas

    axs[0].plot(rhos1, data1, color='tab:red', label='$N=50$')
    axs[0].plot(rhos2, data2, color='tab:blue', label='$N=40$')
    axs[0].plot(rhos3, data3, color='tab:green', label='$N=30$')
    axs[0].plot(rhos4, data4, color='tab:purple', label='$N=20$')

    axs[0].set_xscale('log')
    axs[0].set_xlabel('Swarm Density ($\\rho$)')
    axs[0].set_ylabel('Exploration Ratio ($\Theta$)')
    axs[0].set_title('$k=10$', fontsize=18)
    axs[0].legend(loc='center left')

    axs[1].plot(rhos1, data1a, color='tab:red', label='$N=50$')
    axs[1].plot(rhos2, data2a, color='tab:blue', label='$N=40$')
    axs[1].plot(rhos3, data3a, color='tab:green', label='$N=30$')
    axs[1].plot(rhos4, data4a, color='tab:purple', label='$N=20$')

    axs[1].set_xscale('log')
    axs[1].set_xlabel('Swarm Density ($\\rho$)')
    axs[1].set_title('$k=15$', fontsize=18)

    axs[2].plot(rhos1, data1b, color='tab:red', label='$N=50$')
    axs[2].plot(rhos2, data2b, color='tab:blue', label='$N=40$')
    axs[2].plot(rhos3, data3b, color='tab:green', label='$N=30$')

    axs[2].set_xscale('log')
    axs[2].set_xlabel('Swarm Density ($\\rho$)')
    axs[2].set_title('$k=25$', fontsize=18)

    plt.show()

if fig6:
    data1 = pd.read_csv("data/50agents_ne33r5_v15_engagement.csv", header=None)
    data2 = pd.read_csv("data/40agents_ne30_v15_engagement.csv", header=None)
    data3 = pd.read_csv("data/30agents_ne26_v15_engagement.csv", header=None)
    data4 = pd.read_csv("data/20agents_ne21r2_v15_engagement.csv", header=None)

    data1 = 1 - data1
    data2 = 1 - data2
    data3 = 1 - data3
    data4 = 1 - data4

    k20 = np.linspace(1, 19, 19)
    k30 = np.linspace(1, 29, 29)
    k40 = np.linspace(1, 39, 39)
    ks_o = np.linspace(1, 49, 49)

    k20a = np.linspace(1/20, 19/20, 19)
    k30a = np.linspace(1/30, 29/30, 29)
    k40a = np.linspace(1/40, 39/40, 39)
    ks_oa = np.linspace(1/50, 49/50, 49)

    plt.rcParams.update({'font.size': 22, 'lines.linewidth': 3} )
    fig, axs = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

    axs.plot(ks_oa, data1, label='50 Agents', color='tab:red')
    axs.plot(k40a, data2, label='40 Agents', color='tab:blue')
    axs.plot(k30a, data3, label='30 Agents', color='tab:green')
    axs.plot(k20a, data4, label='20 Agents', color='tab:purple')

    axs.set_ylabel('Exploration Ratio ($\Theta$)')
    axs.set_xlabel('$k/N$')
    axs.legend(loc='upper right')

    plt.show()

if fig7:
    data1 = pd.read_csv("data/50agents_ne33r5_v15_score.csv", header=None)
    data2 = pd.read_csv("data/40agents_ne30_v15_score.csv", header=None)
    data3 = pd.read_csv("data/30agents_ne26_v15_score.csv", header=None)
    data4 = pd.read_csv("data/20agents_ne21r2_v15_score.csv", header=None)

    k20 = np.linspace(1, 19, 19)
    k30 = np.linspace(1, 29, 29)
    k40 = np.linspace(1, 39, 39)
    ks_o = np.linspace(1, 49, 49)

    k20a = np.linspace(1/20, 19/20, 19)
    k30a = np.linspace(1/30, 29/30, 29)
    k40a = np.linspace(1/40, 39/40, 39)
    ks_oa = np.linspace(1/50, 49/50, 49)

    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.1)

    axs[0].plot(ks_o, data1, label='50 Agents', color='tab:red')
    axs[0].plot(k40, data2, label='40 Agents', color='tab:blue')
    axs[0].plot(k30, data3, label='30 Agents', color='tab:green')
    axs[0].plot(k20, data4, label='20 Agents', color='tab:purple')

    axs[1].plot(ks_oa, data1, label='50 Agents', color='tab:red')
    axs[1].plot(k40a, data2, label='40 Agents', color='tab:blue')
    axs[1].plot(k30a, data3, label='30 Agents', color='tab:green')
    axs[1].plot(k20a, data4, label='20 Agents', color='tab:purple')

    axs[0].set_xlabel('$k$')
    axs[0].set_ylabel('Tracking Performance ($\Xi$)')

    axs[0].set_ylim(0.1, 0.9)
    axs[0].set_xlim(-4.5, 52)

    axs[0].arrow(10, 0.8, -7, 0, head_width=0.015, head_length=1.2, facecolor='k')
    axs[0].text(-3, 0.83, 'Exploration-Limited')
    axs[0].arrow(27, 0.8, 15, 0, head_width=0.015, head_length=1.2, facecolor='k')
    axs[0].text(25, 0.83, 'Exploitation-Limited')

    axs[1].set_xlabel('$k/N$')
    axs[1].legend(loc='lower right')

    plt.show()

if fig8:
    data1 = pd.read_csv("data/50agents_ne50_v15_long_score.csv", header=None)
    data2 = pd.read_csv("data/40agents_ne44r7_v15_long_score.csv", header=None)
    data3 = pd.read_csv("data/30agents_ne38r7_v15_long_score.csv", header=None)
    data4 = pd.read_csv("data/20agents_ne31r6_v15_long_score1.csv", header=None)

    k20 = np.linspace(1, 19, 19)
    k30 = np.linspace(1, 29, 29)
    k40 = np.linspace(1, 39, 39)
    ks_o = np.linspace(1, 49, 49)

    k20a = np.linspace(1/20, 19/20, 19)
    k30a = np.linspace(1/30, 29/30, 29)
    k40a = np.linspace(1/40, 39/40, 39)
    ks_oa = np.linspace(1/50, 49/50, 49)

    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.1)

    axs[0].plot(ks_o, data1, label='50 Agents', color='tab:red')
    axs[0].plot(k40, data2, label='40 Agents', color='tab:blue')
    axs[0].plot(k30, data3, label='30 Agents', color='tab:green')
    axs[0].plot(k20, data4, label='20 Agents', color='tab:purple')

    axs[1].plot(ks_oa, data1, label='50 Agents', color='tab:red')
    axs[1].plot(k40a, data2, label='40 Agents', color='tab:blue')
    axs[1].plot(k30a, data3, label='30 Agents', color='tab:green')
    axs[1].plot(k20a, data4, label='20 Agents', color='tab:purple')

    axs[0].set_xlabel('$k$')
    axs[0].set_ylabel('Tracking Performance ($\Xi$)')
    axs[1].set_xlabel('$k/N$')
    axs[1].legend(loc='lower right')

    plt.show()

if fig9:
    data1 = pd.read_csv("data/1T_k10_verification_v15_score.csv", header=None)
    data2 = pd.read_csv("data/constant_repulsion_k10_score.csv", header=None)

    areas = np.logspace(0.6, 2.65, num=125) ** 2
    areas1 = np.logspace(0.6, 2.35, num=125) ** 2
    rhos = 50 / areas
    plt.rcParams.update({'font.size': 22, 'lines.linewidth': 3})

    fig, ax = plt.subplots(figsize=(8.5, 8))

    ax.plot(rhos, data1, color='tab:red', label='Search and Track Strategy')
    ax.plot(rhos, data2, color='tab:blue', label='Rudimentary Swarm Strategy')

    ax.set_xscale('log')
    ax.set_xlabel('Swarm Density ($\\rho$)')
    ax.set_ylabel('Tracking Performance  ($\Xi$)')
    ax.legend(loc='upper left', fontsize=15)

    plt.show()

if figa1b:
    data1 = pd.read_csv("data/50agents_ne_k20_v15_density_e10_r3.csv", header=None)
    data2 = pd.read_csv("data/50agents_ne_k20_v15_density_e10_r4.csv", header=None)
    data3 = pd.read_csv("data/50agents_ne_k20_v15_density_e10_r6.csv", header=None)
    data4 = pd.read_csv("data/50agents_ne_k20_v15_density_e10_r10.csv", header=None)
    data5 = pd.read_csv("data/50agents_ne_k20_v15_density_e10_r16.csv", header=None)

    areas = np.logspace(0.6, 2.65, num=125) ** 2
    rhos = 50 / areas
    plt.rcParams.update({'font.size': 22, 'lines.linewidth': 3})

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(rhos, rhos, color='black', linestyle='dashed')
    ax.plot(rhos, data1, color='tab:red', label='$N=3$')
    ax.plot(rhos, data2, color='tab:blue', label='$N=4$')
    ax.plot(rhos, data3, color='tab:green', label='$N=6$')
    ax.plot(rhos, data4, color='tab:purple', label='$N=10$')
    ax.plot(rhos, data5, color='tab:Brown', label='$N=16$')

    ax.set_xscale('log')
    ax.set_xlabel('Swarm Density ($\\rho$)')
    ax.set_ylabel('Local Density ($\Delta\\rho$)')
    ax.set_yscale('log')
    ax.legend(loc='lower right')

    plt.show()

