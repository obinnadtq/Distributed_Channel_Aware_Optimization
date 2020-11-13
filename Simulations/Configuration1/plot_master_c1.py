import matplotlib.pyplot as plt

from Exp1 import plot_exp1

plotting_exp1 = 1

save_figures = 0    # save figures as pdfs
tikz = 0            # create tikz files
datafiles = 0       # create data files

if plotting_exp1:
    plot_exp1.plot(tikz,datafiles,save_figures)



plt.show()