"""
Helper function for plotting decision regions from Chapter 2 of Python Machine Learning.
"""

from matplotlib.colors import ListedColormap


def plot_decision_regions(plt, observations, labels, predict_fn, weights, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(labels))])

    # plot the decision surface
    x1_min, x1_max = observations[:, 0].min() - 1, observations[:, 0].max() + 1
    x2_min, x2_max = observations[:, 1].min() - 1, observations[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = predict_fn(np.array([xx1.ravel(), xx2.ravel()]).T, weights)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(labels)):
        plt.scatter(x=observations[labels == cl, 0], y=observations[labels == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
