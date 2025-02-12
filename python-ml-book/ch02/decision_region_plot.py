"""
Helper function for plotting decision regions from Chapter 2 of Python Machine Learning.
"""

from matplotlib.colors import ListedColormap


def plot_decision_regions(plt, observations, labels, predict_fn, weights, resolution=0.02,
                          xlabel='sepal length [cm]', ylabel='petal length [cm]'):
    # we define a number of colors and markers and create a color map from
    # the list of colors via ListedColormap
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(labels))])

    # We determine the minimum and maximum values for the two features and use those
    # feature vectors to create a pair of grid arrays xx1 and xx2 via the NumPy meshgrid function
    x1_min, x1_max = observations[:, 0].min() - 1, observations[:, 0].max() + 1
    x2_min, x2_max = observations[:, 1].min() - 1, observations[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Since we trained our perceptron classifier on two feature dimensions,
    # we need to flatten the grid arrays and create a matrix that has the same number of
    # columns as the Iris training subset so that we can use the predict method to
    # predict the class labels Z of the corresponding grid points.
    Z = predict_fn(np.array([xx1.ravel(), xx2.ravel()]).T, weights)
    Z = Z.reshape(xx1.shape)

    # After reshaping the predicted class labels Z into a grid with the same dimensions
    # as xx1 and xx2, we can now draw a contour plot via matplotlib's contourf function
    # that maps the different decision regions to different colors for each
    # predicted class in the grid array
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(labels)):
        plt.scatter(x=observations[labels == cl, 0], y=observations[labels == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
