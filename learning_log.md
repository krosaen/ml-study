Learning diary

# 3/18

Kept at it with chapter 2 of the python ML book today, which covers implementing a basic single perceptron algorithm. It starts with one of the original training algorithms where the weights are updated using a scaling factor, and follows up with an improved approach using gradient descent: something that is core to many optimization steps across ML, including within the backpropogation algorithm of neural networks.

I took my time with it and refactored the author's OO solution into what I think is a cleaner one, both in terms of using a higher order function and in having that function capture the weights that are trained for *that* solution instead of having the prediction function rely on mutable weights ([my notebook](https://github.com/krosaen/ml-study/blob/master/python-ml-book/ch02/ch02.ipynb) and [the author's](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch02/ch02.ipynb)).

One thing that tripped me up was the helper function `plot_decision_regions`. I updated the training function to return a log of the weights as they evolved during the iterations, and I wanted to plot how the decision regions got closer to 100% during execution. However, it only seems to plot properly at the final step. I think I need to read more about matplotlib's contourf function before I can get to the bottom of it. For now, I think I'll move on.

# 3/17

Switching gears today and beginning to work my way through Python Machine Learning. I read through this before but now want to go through it again more slowly, trying out each example myself and perhaps applying the techniques to new datasets as I go. I chose this book because it has a nice balance of conceptual background and practical application of libraries. It also has a great overview of the important details of applying an algorithm, including data pre-processing, dimensionality reduction, evaluating the model by comparing one trained on one portion of the dataset against an unseen segment (e.g does it seem to generalize), hyperparamter tuning etc. These concerns were also covered in Andrew NG's machine learning class that I took part of a couple of years ago and while I by no means remember everything from that course, I remember enough to know that this book does a good job covering these topics.

Working my way through this book and familiarizing myself with many of the algorithms available in scikit-learn is the 2nd of the 3 prongs in my ML curriculum, the first being stats and probability and the final being a larger project TBD (but ideas abound). 

Anyways, today it's chapters 1 and 2. Chapter 1 includes an overview of ML, how to get setup with the necessary tools and the like. I'm already setup with an install of python3, scikit-learn and Jupyter for IPython notebooks. I recommend using [anaconda](https://docs.continuum.io/anaconda/index) to quickly get setup with clean python installs, jupyter and the relevant libraries. I love jetbrains products and am using PyCharm for any python work not done directly in IPython notebooks. Every example in the book is [already available](https://github.com/rasbt/python-machine-learning-book) in notebook form, but I will work through [in my own](https://github.com/krosaen/ml-study/tree/master/python-ml-book) notebooks anyways.

## Chapter 1 notes

ML field overview:
- supervised learning: generalize from labeled data
  - classification: predicting categorical class labels (e.g spam, not spam, or is this the digit "1")
  - regression: predicting continuous value (e.g predicting house price). Note: from stats terminology, this would be predicting a quantitative "ratio" variable
- unsupervised learning: discover structure from unlabeled data
- reinforcement learning: improve performance in dynamic environment optimizing based on a reward signal

Comparing terminology from stats and ML:

- dataset aka feature matrix
- observation aka instance aka sample aka row aka x superscript i
- variable aka feature aka dimension aka attributes aka measurements aka column aka x subscript j

Predictive modeling overview

- preprocessing: feature extraction and scaling, feature selection, dimensionality reduction, sampling
- learning: 
  - model selection (e.g deciding among SVM, logistic regression, random forests...)
  - cross-validation: comparing performance on validation subset which is distinct from training subset to avoid overfitting and have a better shot at performing well in final evaluation stage
  - choosing performance metrics (e.g classification accuracy)
  - hyperparameter optimization: tuning the knobs of the model
- evaluation: how well does the tuned model perform on unseen test set?
- prediction: your model in the wild! applying tuned model to new data

One thing that's interesting to think about is how the dataset is segmented for different stages of this process. You separate training and validation sets right off the bat as you evaluate models and tune parameters. This makes sure you are not just fitting to the model that was used to generate / tune the model (e.g the weights of the nodes in a neural net). However, to make sure that you haven't overfit to the evaluation set during the tuning and model selection stages, there's one final check in the evaluation stage where you apply your model and tuned algorithm to a test set that was removed from all prior stages of the process.

It's also worth noting that hyperparameter tuning is tuning is not the same thing as optimizing the weights in whatever model you are training. From the book, "Intuitively, we can think of those hyperparameters as parameters that are not learned from the data but represent the knobs of a model that we can turn to improve its performance..."

## Chapter 2 notes

This chapter dives into implementing some basic learning algorithms based on a single perceptron. I spent a couple of hours running the same code as the book provides, but slowing down to grok it. I needed some background knowledge about numpy and pandas data structures including data frames and numpy's fast vectorized multi-dimensional arrays.

# 3/16

Today I wrapped up playing with the NBA game net rating data set in a Jupyter IPython notebook ([on github](https://github.com/krosaen/ml-study/blob/master/basic-stats/nba-games-net-rating-boxplots/NbaTeamGameNetRatingsPlots.ipynb)). Not exactly setting the world on fire but was nice to get the basics going with Jupyter notebooks and to figure out how to make it viewable on github.

I also wrapped up the "examining distributions" section of the stanford stats class.

Concepts:

- The standard deviation is the average squared delta from the mean
- similar to mean it is heavily affected by outliers and best suited for symmetric datasets, otherwise box plots are likely better
- The standard deviation rule: for a normal distribution, 68% of the data falls within 1 std deviation of the mean, 95% fall within 2 and 99.7% fall within 3.

Techniques:
- calculate the standard deviation of a data set
- report what % fall within 1, 2 and 3 standard deviations
- Given mean and std deviation, apply standard rule to answer questions like: what range will 95% of the observations fall? What % of observations will fall above 1 std deviation from mean?

# 3/15

Today I want to put one of the skills into practice by producing a box plot with some real data using a python notebook. I think it would be interesting to compare the box plots of the point delta for nba wins of various teams. 

# 3/14

Today I'm continuing to plot my big picture curriculum focusing on three areas:

- probability and statistics fundamentals
- applied ML / inference techniques
- capstone: pick something cool I want to be able to do and work backwards

I've previously gathered a lot of relevant materials, but today discovered an additional resource for basic stats: [Stanfords free course](https://lagunita.stanford.edu/courses/OLI/ProbStat/Open/about). It is a very basic intro but I believe I can learn something from it as I also work through problem sets from the text "All of Statistics" that I found from CMU's [course from the author of the](http://www.stat.cmu.edu/~larry/=stat705/) and its more introductory CS counterpart [CS 36-700](https://www.dropbox.com/s/5xf2mfd7k6w0ipk/syllabus.pdf?dl=0) (note: I found materials elsewhere online months ago that now no longer appear to be online).

One question I got wrong off the bat in this basic stats course was in determining whether the variable 'ZipCode' from a dataset was categorical or quantitative: I chose 'quantitative' without thinking too much because there are a lot of possible options and I was thinking categorical would be a smaller finite set of choices, but this is wrong; zipcodes can't be summarized quantitatively, e.g no 'average' of zipcodes, so it is a categorical variable. Kind of embarrassing I didn't get this right, but hey, this is why it's good to start with the basics even as I dive into more advanced stuff in parallel.

Another course I took another look at today was [Stanford's statistical learning course](https://lagunita.stanford.edu/courses/HumanitiesSciences/StatLearning/Winter2016/about). It covers some of the same techniques I will be looking at as I work through [Python Machine Learning](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning). I will check back to see if any of the videos help in my understanding as I proceed.

## Stats basics 

Today I covered basic exploratory analysis of categorical and continuous variables (up through measures of spread [here](https://lagunita.stanford.edu/courses/OLI/ProbStat/Open/courseware/intro/9476c98a36d34dec90e69994d367e554/). 

Concepts:
- data and variables
- summarizing data: histograms, measures of center (mean, mode, median), measures of spread and outliers

Skills & Techniques:
- determine whether a variable is categorical or quantitative (and more granularly, whether it is nominal, ordinal, interval or ratio)
- plot a histogram of a categorical variable in bar or pie form
- plot a histogram of a quantitative variable 
- produce a stem plot of a quantitative variable
- upon viewing a histogram, describe the data sets:
  - shape: uniform, unimodal, bi-modal
  - symmetry: centered, skewed right, skewed left
- determine the mode, median, mean of a quantitative variable (from the raw data or a histogram)
- have intuition about the relationship between median and mean based on its histogram (e.g skewed left will have a higher mean than median)
- determine the range of a dataset (from the raw data or a histogram)
- determine the interquartile range (IQR) of a data set
- determine whether a data point is considered an outlier based on its relationship to Q1, Q3 and the IQR (e.g if less than Q1 - 1.5 * IQR) 
- construct a box plot of a data set based in its min, q1, median, q3, max
- develop intuition for a dataset by viewing its box plot
- compare two or more datasets by plotting their box plots on the same graph next to each other
- calculate the standard deviation of a data set

### Overview 

There's a nice diagram to outline the course that puts everything about the course into context. The course covers 4 aspects of statistics:

1. Getting / producing data: sampling from a population of "all" data to get a dataset.

2. Exploratory data analysis (EDA): sizing up and summarizing the data set to get a feel for its characteristics.

3 & 4: Probability and inference: drawing conclusions about the entire population from the observed data collected in our sample.


### Data and variables

Data are pieces of information about an individual or object, and these pieces are organized into variables. A variable is a characteristic of an individual / object, such as eye color, age, number of hours spent studying for the final exam last for a particular course last fall, etc. 

Note: these are not to be confused with random variables, which assigns values to outcomes of a random experiment.

Note: I think the word I'd usually use for 'variable' here is 'feature'.

Variables can be classified as categorical or quantitative. Categorical variables classify an object into a distinct set of values which are not ordered or comparable in magnitude. Continuous variables have values that can be compared and quantified; taking the average for instance.

Variables can be further classified into 4 progressively quantifiable types by their scale of measurement.
- nominal: basic discrete categories
- ordinal: can be ordered
- interval: can be ordered, difference between two can be quantified
- ratio: has notion of zero value. Can find the mean.

### Summarizing data: examining distributions

#### One categorical variable

Histograms: number of items in each category.

#### One quantitative variable

- histograms but need to pick bins
- stem plots: an algorithm for creating a histogram like visualization by slicing by decimals, which guides you to choosing the size of the bins. Also useful because it preserves the data (each datum is stacked and visible in the plot)
- measures of center: mean, median mode. Mean is only appropriate when data is symmetric without outliers, median is better otherwise
- measures of range
  - range
  - inter-quartile range: median of top half - median of bottom half = Q3 - Q1
    - data below q1 - 1.5*IQR or above q3 + 1.5 IQR considered outliers
    - outliers shouldn't necessarily be discarded; only if they are suspected to be due to conditions that will not be repeated again or through error in collection. Outliers may be essential to the data, e.g a high magnitude earthquake.
  - box plots: visual presentation of min, q1, median, q3, max (the "five number summary") constructed by drawing dashes for each of these 5 summary numbers, and then a box around the q1 through q3.
  - standard deviation



