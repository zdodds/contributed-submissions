# TODO
# You should download this. If this dont work, try #3 or just chatgpt the right install function lol




# TODO



# TODO



# Let's start by exploring library-imports in general
# Note that notebooks are happy to interoperate with plain Python files...

# Here are the contents of morefun.py:

#
# morefun.py
#

def f():
    return 'returned from f'

def fac():
    return 'returned from fac'


# # ordinary library import
# import morefun
# print( f"{morefun.f() = }" )

# # renaming library import
# import morefun as mandatoryfun
# print( f"{mandatoryfun.fac() = }" )

# # single-function import
# from morefun import fac
# print( f"{fac() = }" )

# # single-function renaming import
# from morefun import fac as funfac   # as f
# print( f"{funfac() = }" )

# all-functions import
from morefun import *
print( f"{f() = }" )          # truly an f string...

# if a library needs to be reloaded:   # *** if you develop in more than one file, this can be very important! ***
import importlib
importlib.reload(morefun)


# TODO
# Numpy is a Python library supporting fast, memory-efficient arrays of data
# Let's try it!

# this is numpy's traditional renaming import
import numpy as np


#
# Our starting numpy examples:

print( f"{ np.arange(1,3,.1) = }\n" )
print( f"{ np.linspace(1,3,42) = }\n" )
print( f"{ np.ones( shape=(4,2) ) = }\n" )
print( f"{ np.zeros( (3, 4) ) = }\n" )



# import three random-number-generation functions that create numpy arrays...
from numpy.random import rand, randn, randint

#
# Here, we show how to convert numpy arrays to/from Python lists (and lists-of-lists, etc.)
#
R = rand(2,4)                 # uniform from 0 to 1
print(f"{R = }\n" )

Rn = randn(2,4)               # normally (bell-curve) distributed, mean 0, stdev. 1
print(f"{Rn = }\n" )

A = randint(0,10,size=(2,4))  # let's use one-digit values for ease of printing
print(f"A is\n{A}\n")

L = A.tolist()                # this converts to a Python structure!
print(f"L is\n{L}\n")

A = np.asarray(L)             # and back to a numpy array...
print(f"A is\n{A}\n")

# Notice the slight differences in printing: Python uses commas, Numpy does not


print(f"{A = }\n")



# in-class "screenshot challenge" example

# Python functions (range) and list comprehensions are still available...

L = [ list(range(low,low+6)) for low in range(0,60,10) ]     # low runs over [0,10,20,30,40,50]

A = np.asarray(L)        # convert to a numpy Array

print(f"A.shape is {A.shape}\n")   # (nrows, ncols)    symmetry hiding the difference here...

print(f"A is\n{A}\n")



A[1:3,0:2]   # showing off 2-dimensional slicing!


L = [5,6,7,8,9]
print("Slice is", L[0:2])  # [0:200]



# TODO
# Let's continue by importing the pandas library
import pandas as pd   # abbreviated "pd"


# We will import the data as a dataframe called "zillow"
zillow = pd.read_csv('./housing.csv')


# Let's see what our dataframe looks like



# We can view all of the columns available
zillow.columns


# That last "column" is not really a column, let's drop it:
ROW = 0    # this is one of the constants for defining which axis is which
COLUMN=1   # this is another such constant
zillow2 = zillow.drop(zillow.columns[-1], axis=COLUMN)   # more readable than axis=1



zillow2.columns


# Let's rename zillow
zillow = zillow2     # this is without that url-column


# We can access a column by inputting its name in brackets like so (a single column is, officially, a "Series")
zillow['SalePrice']


# We can access a single value by using its index
zillow['SalePrice'][2]


# We even slice the column, as with a Python list...
zillow['SalePrice'][0:3]


# We can use zillow.loc[n] to locate a house (one row) by its index number n (0-2929)
zillow.loc[0]


# a common pandas pattern is to create a series of Trues and Falses
zillow['SalePrice'] == 172000

# note that this applies the conditional to each of the elements of the the series, zillow['SalePrice']


# this series of booleans can be used to "subset" a data frame

# This command locates all of the houses whose sale prices are $172000
zillow.loc[ zillow['SalePrice'] == 172000 ]


#
# This gives us access to a much smaller subset...
houses_for_172k = zillow.loc[ zillow['SalePrice'] == 172000 ]
print(f"The len of houses_for_172k is {len(houses_for_172k)}")


# other boolean conditions are also welcome...

# This command locates all of the houses whose sale prices are < $172000
houses_under_172k = zillow.loc[ zillow['SalePrice'] < 172000 ]
print(f"The len of houses_under_172k is {len(houses_under_172k)}")


# By default, loc[] extracts all columns of information
# We can pass specific columns we want to view (as a list if there are more than 1)

target_info = zillow.loc[zillow['SalePrice'] == 172000, ['SalePrice','Central Air', 'Full Bath']]
target_info


# let's see the series of data that is zillow's 'Order' column:
zillow['Order']


# We can use zillow.set_index(column) to set a given column as our indices insetad of 0-2929!
# Setting the 'inplace' parameter as True causes the existing data frame to change when we call set_index (insetad of creating a new data frame)
# Setting the 'drop' parameter as False keeps the original column in the data frame (instead of deleting it)
zillow_one = zillow.set_index('Order', drop=False)   # inplace = True  (if we want to replace the original)
zillow_one


# Now we can use loc[] to find search for homes by order number
zillow_one.loc[2930]

# And you can make the index whatever you want!


z1 = zillow.loc[zillow['Screen Porch']>100]
z2 = z1.loc[z1['Pool Area']>100]
z3 = z2.loc[z2['Lot Area']>14200]

#Pool Area (Continuous): Pool area in square feet


# here is a cell to work on hw4pr1 task #1
z1 = zillow.loc[zillow['SalePrice']<300000]
z2 = z1.loc[z1['Yr Sold']==2010]
z3 = z2.loc[z2['Lot Area']>10000]
z4 = z3.loc[z3['Bedroom AbvGr']==3]
z5= z4.loc[z4['Mo Sold']==6]
z6 = z5.loc[z5['Overall Qual']==9]

# order number = 42


#TODO
# We will need data in order to make graphs! We will use pandas
import pandas as pd

# matplotlib is an essential whenever we are making graphs!
# Seaborn is simply a shortcut for using matplotlib!
import matplotlib.pyplot as plt

# Import Seaborn!
import seaborn as sns


# Next, we import import our dataframe using pandas
iris_orig = pd.read_csv('./iris.csv')
iris_orig


# we can drop a series of data (a row or a column)
# they're indicated by numeric value, row~0, col~1, but let's use readable names instead:
ROW = 0
COLUMN = 1

iris2 = iris_orig.drop('adapted from https://en.wikipedia.org/wiki/Iris_flower_data_set', axis=COLUMN)

# iris2 is a new dataframe, without that unwanted column,
# which had really just been a single element taking up a whole column...



# Those last two rows look suspicious...
iris3 = iris2.drop(142, axis=ROW)
iris4 = iris3.drop(141, axis=ROW)
iris = iris4
iris                 # our final dataframe-name


# To illustrate the beauty of data visualization...
# Let's start with perhaps the most powerful type of plot applicable to this data set: a pair plot

# The POWER of seaborn: one-line code
PairPlot = sns.pairplot(data=iris, hue='irisname')
# Making graphs has never been easier!

# Unpacking this...
# pairplot() is the function that - you guessed it - makes the pairplot
# 'data' is ... where the data comes from (our pandas data frame)
# 'hue' colors the dots based on values in the designated ('Species') column in the data frame

# What on Earth is a pair plot?
# A scatter plot compares two values (i.e. length and width)
# A pair plot simply creates a scatter plot of every possible pair of values
# You can see which values are being compared by looking at the labels!
# The diagonal plots however are simply the distribution of a single value (a 'univariate' distribution)

# Why did we name the plot?
# Without a name, an not-very-informative storage location (?) gets printed at the top of the graph...


# Now we know how striking Seaborn and data visualization can be
# Let's explore the relationship between petal length and petal width using a single scatter plot

ScatterPlot = sns.scatterplot(x='petallen', y='petalwid', data=iris, hue='irisname')

# Unpacking this...
# scatterplot() is the function that - you guessed it - makes the scatterplot
# 'x' is ... the data for the x axis
# 'y' is ... the data for the y axis

# Can you find this graph in the pair plot?


# We see that there may be some kind of linear relationship between these variables!
# We can use a lmplot (linear model) to add regression lines to the data

LmPlot = sns.lmplot(x='petallen', y='petalwid', data=iris)

# If you add the 'hue' parameter, the data will become separated by species and you can view the best-fit line for each species


# Let's try something else now...
# Which flowers have the longest petals?
# You could create something like a dot plot or a box plot, but let's try a more musical approach:

ViolinPlot = sns.violinplot(x='irisname', y='petallen', data=iris, inner='stick')

# Unpacking this...
# 'inner' draws a stick for each data point
# Notice that the plots get wider in areas with more sticks!

# A violin plot is very similar to a box-and-whiskers plot, but has more detail (and is much cooler)


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the example flights dataset and convert to CS35_Participant_2-form
flights_long = sns.load_dataset("flights")
flights = (
    flights_long
    .pivot(index="month", columns="year", values="passengers")
)

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_theme()

# Load the example flights dataset and convert to CS35_Participant_2-form
digits_orig = pd.read_csv('./digits.csv')
list_of_column_names = digits_orig.columns

ROW = 0
COLUMN = 1
digits2 = digits_orig.drop( list_of_column_names[-1], axis=COLUMN)  # drop the rightmost column - it's just a url!
digitsA = digits2.values  # get a numpy array (digitsA) from the dataframe (digits2)

row_to_show = 440   # choose the digit (row) you want to show...

pixels_as_row = digitsA[row_to_show,0:64]
print("pixels as 1d numpy array (row):\n", pixels_as_row)

pixels_as_image = np.reshape(pixels_as_row, (8,8))   # reshape into a 2d 8x8 array (image)
print("\npixels as 2d numpy array (image):\n", pixels_as_image)

# create the figure, f, and the axes, ax:
f, ax = plt.subplots(figsize=(9, 6))

# colormap choice! Fun!   www.practicalpythonfordatascience.com/ap_seaborn_palette or seaborn.pydata.org/tutorial/color_palettes.html
our_colormap = sns.color_palette("light:b", as_cmap=True)

# Draw a heatmap with the numeric values in each cell (make annot=False to remove the values)
sns.heatmap(pixels_as_image, annot=True, fmt="d", linewidths=.5, ax=ax, cmap=our_colormap)



# https://python-graph-gallery.com/scatter-plot/
# library & dataset
import seaborn as sns
df = sns.load_dataset('iris')

# use the function scatterplot() to make a scatterplot
sns.scatterplot(x=df["sepal_length"], y=df["sepal_width"])


# https://matplotlib.org/2.0.2/examples/lines_bars_and_markers/barh_demo.html
"""
====================
Horizontal bar chart
====================

This example showcases a simple horizontal bar chart.
"""
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

ax.barh(y_pos, performance, xerr=error, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

plt.show()


# https://seaborn.pydata.org/examples/smooth_bivariate_kde.html
# Credit to Michael Waskom

import seaborn as sns
sns.set_theme(style="white")

df = sns.load_dataset("penguins")

g = sns.JointGrid(data=df, x="body_mass_g", y="bill_depth_mm", space=0)
g.plot_joint(sns.kdeplot,
             fill=True, clip=((2200, 6800), (10, 25)),
             thresh=0, levels=100, cmap="rocket")
g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)


#
# seaborn gallery penguin example from
#    https://seaborn.pydata.org/examples/grouped_barplot.html

import seaborn as sns
sns.set_theme(style="whitegrid")

penguins = sns.load_dataset("penguins")

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=penguins, kind="bar",
    x="species", y="body_mass_g", hue="sex",
    errorbar="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")


# restarting the kernel... (I'm not sure why this is here. -ZD)

# from IPython.core.display import HTML
# HTML("<script>Jupyter.notebook.kernel.restart()</script>")


#
# the ribbon box example from
#     https://matplotlib.org/stable/gallery/misc/demo_ribbon_box.html
#

import numpy as np

from matplotlib import cbook, colors as mcolors
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransformTo


class RibbonBox:

    original_image = plt.imread(
        cbook.get_sample_data("Minduka_Present_Blue_Pack.png"))
    cut_location = 70
    b_and_h = original_image[:, :, 2:3]
    color = original_image[:, :, 2:3] - original_image[:, :, 0:1]
    alpha = original_image[:, :, 3:4]
    nx = original_image.shape[1]

    def __init__(self, color):
        rgb = mcolors.to_rgb(color)
        self.im = np.dstack(
            [self.b_and_h - self.color * (1 - np.array(rgb)), self.alpha])

    def get_stretched_image(self, stretch_factor):
        stretch_factor = max(stretch_factor, 1)
        ny, nx, nch = self.im.shape
        ny2 = int(ny*stretch_factor)
        return np.vstack(
            [self.im[:self.cut_location],
             np.broadcast_to(
                 self.im[self.cut_location], (ny2 - ny, nx, nch)),
             self.im[self.cut_location:]])


class RibbonBoxImage(AxesImage):
    zorder = 1

    def __init__(self, ax, bbox, color, *, extent=(0, 1, 0, 1), **kwargs):
        super().__init__(ax, extent=extent, **kwargs)
        self._bbox = bbox
        self._ribbonbox = RibbonBox(color)
        self.set_transform(BboxTransformTo(bbox))

    def draw(self, renderer, *args, **kwargs):
        stretch_factor = self._bbox.height / self._bbox.width

        ny = int(stretch_factor*self._ribbonbox.nx)
        if self.get_array() is None or self.get_array().shape[0] != ny:
            arr = self._ribbonbox.get_stretched_image(stretch_factor)
            self.set_array(arr)

        super().draw(renderer, *args, **kwargs)


def main():
    fig, ax = plt.subplots()

    years = np.arange(2004, 2009)
    heights = [7900, 8100, 7900, 6900, 2800]
    box_colors = [
        (0.8, 0.2, 0.2),
        (0.2, 0.8, 0.2),
        (0.2, 0.2, 0.8),
        (0.7, 0.5, 0.8),
        (0.3, 0.8, 0.7),
    ]

    for year, h, bc in zip(years, heights, box_colors):
        bbox0 = Bbox.from_extents(year - 0.4, 0., year + 0.4, h)
        bbox = TransformedBbox(bbox0, ax.transData)
        ax.add_artist(RibbonBoxImage(ax, bbox, bc, interpolation="bicubic"))
        ax.annotate(str(h), (year, h), va="bottom", ha="center")

    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, 10000)

    background_gradient = np.zeros((2, 2, 4))
    background_gradient[:, :, :3] = [1, 1, 0]
    background_gradient[:, :, 3] = [[0.1, 0.3], [0.3, 0.5]]  # alpha channel
    ax.imshow(background_gradient, interpolation="bicubic", zorder=0.1,
              extent=(0, 1, 0, 1), transform=ax.transAxes, aspect="auto")

    plt.show()


main()


# We import import our dataframe from a csv, using pandas
iris_orig = pd.read_csv('./iris.csv')
# iris_orig

ROW = 0
COLUMN = 1

iris2 = iris_orig.drop('adapted from https://en.wikipedia.org/wiki/Iris_flower_data_set', axis=COLUMN)
# iris2

# Those last two rows look suspicious...
iris3 = iris2.drop(142, axis=ROW)
iris4 = iris3.drop(141, axis=ROW)
iris = iris4
iris                 # our final dataframe-name


iris_sorted = iris.sort_values(by=['petalwid'])
iris_sorted


#
# let's create a new column with integer values from 0 to the length of the dataframe:

iris_sorted['x_value'] = np.arange(0,len(iris_sorted))   # this is like an "index," but it's just values
iris_sorted


#
# Now, let's plot the petalwidths against their location in the sorted list:

import seaborn as sns
sns.set_theme(style="darkgrid")

# Plot the responses for different events and regions
sns.lineplot(x="x_value",y="petalwid", hue="irisname", data=iris_sorted)



import pandas as pd

# We import import our dataframe from a csv, using pandas
boat = pd.read_csv('./titanic.csv')
# iris_orig

# print(boat)
ROW = 0
COLUMN = 1

boat_sorted = boat.sort_values(by=['pclass'])
boat_sorted

# #
# # let's create a new column with integer values from 0 to the length of the dataframe:

# iris_sorted['x_value'] = np.arange(0,len(iris_sorted))   # this is like an "index," but it's just values
# iris_sorted

# #
# # Now, let's plot the petalwidths against their location in the sorted list:

import seaborn as sns
sns.set_theme(style="darkgrid")

# # Plot the responses for different events and regions
sns.barplot(x="pclass",y="survived", hue="embarked", data=boat_sorted)


# from the plot, seems like more people from Cherbourg are more likely to survive. and pclass 1 (upperclass) people are more likely to survive too.


# distribution of the profit magins of companies and how it differs based on rank.
# Seems like there a few exceptions that do really good for themselves despite their rank.
import pandas as pd

# We import import our dataframe from a csv, using pandas
boat = pd.read_csv('fortune500.csv')
# print(boat)
ROW = 0
COLUMN = 1
boat["profitmargin%"] = 100* pd.to_numeric(boat["Profit (in millions)"], errors="coerce") / pd.to_numeric(boat["Revenue (in millions)"], errors="coerce")

# print(type(boat["Revenue (in millions)"]))
boat = boat[boat["profitmargin%"] > 0]
# boat = boat.head(200)
# boat["profitmargin%"] =boat["profitmargin%"]*(-1)
# boat['profitmargin%'] = np.arange(0,len(boat))   # this is like an "index," but it's just values

# boat_sorted = boat.sort_values(by=['Revenue'])
# boat_sorted

# # #
# # # let's create a new column with integer values from 0 to the length of the dataframe:

# # iris_sorted['x_value'] = np.arange(0,len(iris_sorted))   # this is like an "index," but it's just values
# # iris_sorted

# # #
# # # Now, let's plot the petalwidths against their location in the sorted list:

# import seaborn as sns
sns.set_theme(style="darkgrid")

# # # Plot the responses for different events and regions
sns.scatterplot(x="Rank",y="profitmargin%", data=boat)




# Load the iris data from iris_cleaned.csv
iris_df = pd.read_csv('iris_cleaned.csv')   # this is a dataframe



iris_df


# loop over all rows -- easier than trying to change how pandas displays the summary
for row in iris_df.iterrows():
    print(row)



# Create a Linear Regression model
from sklearn.linear_model import LinearRegression

# Create and fit the model
model = LinearRegression()
X = iris_df[['petallen']].values  # Input feature as 2D array
y = iris_df['sepalwid'].values    # Target variable as 1D array
model.fit(X, y)

# Make predictions using the model
y_pred = model.predict(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Linear regression')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')
plt.title('Linear Regression: Sepal Width vs Petal Length')
plt.legend()
plt.show()

# Print the model coefficients
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")



# Calculate R-squared score
r2_score = model.score(X, y)
print(f"R-squared score: {r2_score:.4f}")



def predict_sepal_width(petal_length):
    """
    Predicts sepal width based on petal length using the trained linear regression model

    Args:
        petal_length (float): The petal length value to make prediction for

    Returns:
        float: Predicted sepal width value
    """
    # Reshape input to 2D array with 1 sample, 1 feature
    X_new = [[petal_length]]

    # Use model to make prediction
    prediction = model.predict(X_new)

    # Return the predicted value
    return prediction[0]

# Test the function with a sample value
test_petal_length = 2.5
predicted_width = predict_sepal_width(test_petal_length)
print(f"For a petal length of {test_petal_length}, predicted sepal width is {predicted_width:.2f}")



def predict_sepal_width_via_avg(petal_length=4.2):
    """
    Predicts sepal width by simply returning the average sepal width,
    ignoring the input petal length

    Args:
        petal_length (float): The petal length value (ignored)

    Returns:
        float: Average sepal width value
    """
    # Calculate average sepal width from training data
    avg_sepal_width = iris_df['sepalwid'].mean()

    # Return the average, regardless of input
    return avg_sepal_width

# Test the function with same sample value
test_petal_length = 2.5
predicted_width_avg = predict_sepal_width_via_avg(test_petal_length)
print(f"For a petal length of {test_petal_length}, predicted sepal width (using average) is {predicted_width_avg:.2f}")



# Let's run a linear regression on the pearson dataset

import pandas as pd
pearson_df = pd.read_csv('./pearson_dataset.csv')
pearson_df


import matplotlib.pyplot as plt
import seaborn as sns

# Create scatter plot of heights data
plt.figure(figsize=(10,6))
sns.scatterplot(data=pearson_df, x='sheight', y='fheight', alpha=0.5)

# Fit linear regression
from sklearn.linear_model import LinearRegression
X = pearson_df[['sheight']].values
y = pearson_df['fheight'].values
reg = LinearRegression().fit(X, y)

# Plot regression line
plt.plot(X, reg.predict(X), color='red', linewidth=2)

plt.title('Father vs Son Heights with Linear Regression')
plt.xlabel('Son Height (inches)')
plt.ylabel('Father Height (inches)')

# Calculate and print R-squared, slope and intercept
r2 = reg.score(X, y)
slope = reg.coef_[0]
intercept = reg.intercept_

print(f"R-squared: {r2:.4f}")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")


# my own perdiction!
housing = pd.read_csv('housing.csv')



# loop over all rows -- easier than trying to change how pandas displays the summary
for row in housing.iterrows():
    print(row)



# Create a Linear Regression model
from sklearn.linear_model import LinearRegression

# Create and fit the model
model = LinearRegression()
X = housing[['SalePrice']].values  # Input feature as 2D array
y = housing['Lot Area'].values    # Target variable as 1D array
model.fit(X, y)

# Make predictions using the model
y_pred = model.predict(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Linear regression')
plt.xlabel('Sale Price')
plt.ylabel('Lot Area')
plt.title('Linear Regression: Lot Area vs Sale Price')
plt.legend()
plt.show()

# Print the model coefficients
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")


# Calculate R-squared score
r2_score = model.score(X, y)
print(f"R-squared score: {r2_score:.4f}")



def predict_lot_area(sale_price):
    """
    Predicts lot area based on sale price using the trained linear regression model

    Args:
        sale_price ingteger

    Returns:
        Integer: Predicted lot area
    """
    # Reshape input to 2D array with 1 sample, 1 feature
    X_new = [[sale_price]]

    # Use model to make prediction
    prediction = model.predict(X_new)

    # Return the predicted value
    return prediction[0]

# Test the function with a sample value
test_sales_price = 100000
predicted_area = predict_lot_area(test_sales_price)
print(f"For a lot area of {test_sales_price}, predicted sales price is {predicted_area:.0f}")


def predict_lot_area_via_avg(SalePrice=50000):
    """
    Predicts sales price by simply returning the average sales price,
    ignoring the input lot area

    Args:
        SalePrice: The sale price value (ignored)

    Returns:
        integer: average lot area value
    """
    # Calculate average sepal width from training data
    avg_lot_area = housing['Lot Area'].mean()

    # Return the average, regardless of input
    return avg_lot_area

# Test the function with same sample value
test_sales_price = 100000
predicted_lot_area_avg = predict_lot_area_via_avg(test_sales_price)
print(f"For a sales price of {test_sales_price}, predicted lot area (using average) is {predicted_lot_area_avg:.0f}")


