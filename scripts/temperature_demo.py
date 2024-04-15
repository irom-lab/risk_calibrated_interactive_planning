import matplotlib.pyplot as plt
import scipy
import numpy as np
import matplotlib
from matplotlib.patches import Rectangle
def plot_bars(categories, values, highlight_bar=1, color_intent=True, default_axis=False,
              figsize=(8, 6), disp_int=False):
    # Updating the code to allow making one of the bars red

    # Creating the bar chart
    fig = plt.figure(figsize=figsize)
    bars = plt.bar(categories, values, color='cornflowerblue')

    if color_intent:
        # Change the color of the specified bar to red
        bars[highlight_bar - 1].set_color('red')

    # Adding the value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        if disp_int:
            plt.text(bar.get_x() + bar.get_width() / 2, yval, '{0:1.0f}'.format(yval), ha='center', va='bottom')
        elif yval > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, yval, '{0:.2f}'.format(yval), ha='center', va='bottom')

    # Adding labels and title
    if not default_axis:
        plt.axis([0.5, len(categories)+0.5, 0, 1])
    # plt.xlabel('Categories')
    plt.ylabel('Values')
    # plt.title('Bar Chart with Numeric Categories')

    if color_intent:
        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['r', 'cornflowerblue']]
        labels = ["True Intent"]
        plt.legend(handles, labels, loc='upper right')

    # # Show the plot
    # plt.show()

    return fig



def plot_histogram(data, highlight_bar=1, xlabel="Values"):
    # Create the histogram
    fig = plt.figure(figsize=(8, 6))
    counts, bins, bars = plt.hist(data, bins=10, color='cornflowerblue', range=(0.62, 0.72))

    # Highlight the bar with the highest frequency in red
    highest_bin = np.argmax(counts)  # Find the index of the bin with the highest count
    bars[highlight_bar].set_color('red')
    print(len(bars))
    print(counts)

    # Adding value labels on top of each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height  > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(count)}', ha='center', va='bottom')

    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    # plt.title('Histogram of Random Data')

    plt.axis([0.6, 0.85, 0, 1.3])

    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['r', 'cornflowerblue']]
    labels = ["Incorrect", "Correct"]
    plt.legend(handles, labels, loc='upper right')

    # Show the plot
    # plt.show()

    return fig

def plot_four_bars():

    p1_logits = np.array([9, 9.4, 8])
    p2_logits = 2*p1_logits
    p3_logits = np.array([1, 1, 1])
    p4_logits = 2*p3_logits
    p1 = scipy.special.softmax(p1_logits)
    p2 = scipy.special.softmax(p2_logits)
    p3 = scipy.special.softmax(p3_logits)
    p4 = scipy.special.softmax(p4_logits)

    print(p1, p2)

    categories = [1, 2, 3]

    font = {
            # 'weight': 'bold',
            'size': 26}

    matplotlib.rc('font', **font)


    f1 = plot_bars(categories, p1)
    plt.savefig('f1.png', bbox_inches='tight')
    f2 = plot_bars(categories, p2)
    plt.savefig('f2.png', bbox_inches='tight')

    f3 = plot_bars(categories, p3)
    plt.savefig('f3.png', bbox_inches='tight')
    f4 = plot_bars(categories, p4)
    plt.savefig('f4.png', bbox_inches='tight')



    non_conformities_1 = [1-p1[0], 1 - p3[0]]

    non_conformities_2 = [1-p2[0], 1 - p4[0]]

    fig = plot_histogram(non_conformities_1, highlight_bar=3, xlabel="Non-conformity Scores")

    plt.savefig('nc1.png', bbox_inches='tight')

    fig = plot_histogram(non_conformities_2, highlight_bar=8, xlabel="Non-conformity Scores")

    plt.savefig('nc2.png', bbox_inches='tight')

def run():
    plot_four_bars()


if __name__ == '__main__':
    run()
