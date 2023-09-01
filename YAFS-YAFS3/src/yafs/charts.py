import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

class charts:
    def trend(self, xData, yData, fileName, title, xTitle):
        x = list(map(str, xData))
        y = yData

        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.xaxis.set_minor_locator(MultipleLocator(5))

        plt.xlabel(xTitle)
        plt.ylabel("Latency (in ms)")
        plt.title(title)

        # saving graph
        folderName = 'graphs'
        fileName = fileName + ".jpeg"

        if not os.path.exists(folderName):
            os.makedirs(folderName)

        plt.savefig(os.path.join(folderName, fileName))
        plt.clf()

    def bar(self, GA_best, EP_best, fileName, title):

        objects = ('GA Fitness', 'Edge Placement Fitness')
        y_pos = np.arange(len(objects))
        fitness = [GA_best, EP_best]

        plt.bar(y_pos, fitness, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Latency (in ms)')
        plt.title(title)


        #saving graph
        folderName = 'graphs'
        fileName = fileName+".jpeg"

        if not os.path.exists(folderName):
            os.makedirs(folderName)

        plt.savefig(os.path.join(folderName, fileName))
        plt.clf()

    def twoTrend(self, xData, yData_1, yData_2, fileName, title, xTitle):
        #GA line
        x1 = list(map(str, xData))
        print("x list is",x1)
        y1 = yData_1
        # plotting the line 1 points
        plt.plot(x1, y1, label="GA")
        # line 2 points
        x2 = list(map(str, xData))
        y2 = yData_2
        # plotting the line 2 points
        plt.plot(x2, y2, label="Edge Placement")
        plt.xlabel(xTitle)
        # Set the y axis label of the current axis.
        #plt.ylabel('Computation time (in seconds)')
        plt.ylabel('Latency (in ms)')
        # Set a title of the current axes.
        plt.title(title)
        # show a legend on the plot
        plt.legend()

        # saving graph
        folderName = 'graphs'
        fileName = fileName + ".jpeg"

        if not os.path.exists(folderName):
            os.makedirs(folderName)

        plt.savefig(os.path.join(folderName, fileName))
        plt.clf()

    def twoBars(self, xData, yData_1, yData_2, fileName, title, xTitle):
        X = xData
        Ygirls = yData_1
        Zboys = yData_2

        X_axis = np.arange(len(X))

        plt.bar(X_axis - 0.2, Ygirls, 0.4, label='Solution')
        plt.bar(X_axis + 0.2, Zboys, 0.4, label='Post-simulation')


        plt.xticks(X_axis, X)

        plt.xlabel(xTitle)
        # Set the y axis label of the current axis.
        # plt.ylabel('Computation time (in seconds)')
        plt.ylabel('Latency (in ms)')
        # Set a title of the current axes.
        plt.title(title)
        # show a legend on the plot
        plt.legend()

        # saving graph
        folderName = 'graphs'
        fileName = fileName + ".jpeg"

        if not os.path.exists(folderName):
            os.makedirs(folderName)

        plt.savefig(os.path.join(folderName, fileName))
        plt.clf()

    def threeBars(self, xData, yData_1, yData_2, yData_3, fileName, title, xTitle):
        X = xData
        Y1 = yData_1
        Y2 = yData_2
        Y3 = yData_3

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

        #fig.suptitle(title)
        fig.suptitle("\n".join([title]), y=1)
        ax1.bar(X, Y1, 1, label='GA', color='blue')
        ax2.bar(X, Y2, 1, label='GA', color='green')
        ax3.bar(X, Y3, 1, label='GA', color='orange')

        ax1.set_title('5 Apps', pad=-0.5)
        ax2.set_title('8 Apps',  pad=-0.5)
        ax3.set_title('12 Apps',  pad=-0.5)

        plt.setp(ax1, xticks=X, xticklabels=X)
        plt.setp(ax2, xticks=X, xticklabels=X)
        plt.setp(ax3, xticks=X, xticklabels=X)

        # Set the y axis label of the current axis.
        ax1.set(ylabel='Latency (in ms)')
        ax2.set(xlabel='Number of modules')

        for ax in fig.get_axes():
            ax.label_outer()

        # saving graph
        folderName = 'graphs'
        fileName = fileName + ".jpeg"

        if not os.path.exists(folderName):
            os.makedirs(folderName)

        plt.savefig(os.path.join(folderName, fileName))
        plt.clf()

    def twoRowBars(self, xData, yData_1, yData_2, yData_3, yData_11, yData_21, yData_31, fileName, title, xTitle):
            X = xData
            Y1 = yData_1
            Y2 = yData_2
            Y3 = yData_3

            Y4 = yData_11
            Y5 = yData_21
            Y6 = yData_31

            fig, ax = plt.subplots(2, 3, sharey=True)

            fig.suptitle("\n\n".join([title]), y=1)
            ax[0,0].bar(X, Y1, 1, label='GA', color='blue')
            ax[0,1].bar(X, Y2, 1, label='GA', color='green')
            ax[0,2].bar(X, Y3, 1, label='GA', color='orange')

            ax[1,0].bar(X, Y4, 1, label='EPA', color='blue')
            ax[1,1].bar(X, Y5, 1, label='EPA', color='green')
            ax[1,2].bar(X, Y6, 1, label='EPA', color='orange')

            ax[0,0].set_title('5 Apps', pad=3.5)
            ax[0,1].set_title('8 Apps', pad=3.5)
            ax[0,2].set_title('12 Apps', pad=3.5)

            plt.setp(ax[1,0], xticks=X, xticklabels=X)
            plt.setp(ax[1,1], xticks=X, xticklabels=X)
            plt.setp(ax[1,2], xticks=X, xticklabels=X)

            # Set the y axis label of the current axis.
            ax[0,0].set(ylabel='GA Latency (in ms)')
            ax[1, 0].set(ylabel='EPA Latency (in ms)')
            ax[1,1].set(xlabel='Number of modules')

            for ax in fig.get_axes():
                ax.label_outer()

            # saving graph
            folderName = 'graphs'
            fileName = fileName + ".jpeg"

            if not os.path.exists(folderName):
                os.makedirs(folderName)

            plt.savefig(os.path.join(folderName, fileName))
            plt.clf()