import matplotlib.pyplot as plt

def plot_config():
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["mathtext.fontset"] = 'dejavuserif'
    plt.rcParams["font.size"] = 22
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2.2

    xtick_param = {'direction': 'in',
         'major.size': 8,
         'major.width': 2,
         'minor.size': 5,
         'minor.width': 1.5}
    ytick_param = {'direction': 'in',
         'major.size': 8,
         'major.width': 2,
         'minor.size': 5,
         'minor.width': 1.5}
    plt.rc('xtick', **xtick_param)
    plt.rc('ytick', **ytick_param)

    grid_param = {'linestyle': '--', 'alpha': 0.5}
    plt.rc('grid', **grid_param)

