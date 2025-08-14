#
# hw5pr2digits_cleaner:  digit-data cleaning...
#


#
# *** SUGGESTION ***
#
# +++ copy-paste-and-alter from the iris- and/or births-cleaning notebooks into here +++
#
# when the data is ready to view, you might want to grab
# the digits-visualization code (it's in the gdoc hw page)
#


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


row_to_show = 42   # choose the digit (row) you want to show...


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



digits_orig


digits_orig.info


digits_orig.columns


# we can drop a series of data (a row or a column)
# the dimensions each have a numeric value, row~0, col~1, but let's use readable names we define:
ROW = 0
COLUMN = 1

# We are dropping this final column - it was just a citation...
df_clean1 = digits_orig.drop('excerpted from http://yann.lecun.com/exdb/mnist/', axis=COLUMN)
df_clean1

# df_clean1 is a new dataframe, without that unwanted column


# no rows needed to be dropped
# and, let's drop the unwanted rows:
ROW = 0
COLUMN = 1

# df_clean2 = df_clean1.drop([142,143,144], axis=ROW)
df_clean2 = df_clean1


#
# let's drop _all_ rows with data that is missing/NaN (not-a-number)
df_clean3 = df_clean2.dropna()  # drop na rows (NaN, not-a-number)
df_clean3.info()  # print the info, and
# let's see the whole table, as well:
df_clean3

# Tidy!  Our data is ready!


#
# let's keep our column names in variables, for reference
#
COLUMNS = df_clean3.columns            # "list" of columns
print(f"COLUMNS is {COLUMNS}")
  # It's a "pandas" list, called an Index
  # use it just as a Python list of strings:
print(f"COLUMNS[0] is {COLUMNS[0]}\n")

# let's create a dictionary to look up an index from its name:
COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i  # using the name (as key), look up the value (i)
print(f"COL_INDEX is {COL_INDEX}")
# print(f"COL_INDEX[ 'petallen' ] is {COL_INDEX[ 'petallen' ]}")


number_INDEX = ['0','1','2','3','4','5','6','7','8','9']

# Let's try it out...
for number in number_INDEX:
    print(f"{number} maps to {number}")


# no need


# we can "apply" to a whole column and create a new column
#   it may give a warning, but this is ok...
#

df_clean4 = df_clean3.copy()  # copy everything AND...

# # add a new column, 'irisnum'
# df_clean4['irisnum'] = df_clean3['irisname'].apply(convert_species)

# let's see...
df_clean4


#
# it's no fun fiddling with the default table formatting.
#
# Remember: we have the data itself!  If we want to see it, we can print:

(NROWS, NCOLS) = df_clean4.shape
print(f"There are {NROWS = } and {NCOLS = }")
print()

for row in range(0,NROWS,5):
    print(df_clean4[row:row+5])    # Let's print 5 at a time...


df_tidy =  df_clean4


#
# That's it!  Then, and write it out to iris_cleaned.csv
filename = 'digits.cvs'
# We'll construct the new filename:
old_filename_without_extension = filename[:-4]                      # remove the ".csv"

cleaned_filename = old_filename_without_extension + "_cleaned.csv"  # name-creating
print(f"cleaned_filename is {cleaned_filename}")

# Now, save the dataframe names df_tidy
df_tidy.to_csv(cleaned_filename, index_label=False)  # no "index" column...


#
# Let's make sure this worked, by re-reading in the data...
#

# let's re-read that file and take a look...
df_tidy_reread = pd.read_csv(cleaned_filename)   # encoding="utf-8" et al.
print(f"{cleaned_filename} : file read into a pandas dataframe.")
df_tidy_reread


#
# Let's make sure we have all of our helpful variables in one place
#
#   Since we changed the columns, this will have changed from above!
#

#
# let's keep our column names (features) in variables, for reference
#
COLUMNS = df_tidy_reread.columns            # "list" of columns
print(f"COLUMNS is {COLUMNS}\n")
  # It's a "pandas" list, called an Index
  # use it just as a Python list of strings:
print(f"COLUMNS[0] is {COLUMNS[0]}\n")

# let's create a dictionary to look up any column index by name
COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i  # using the name (as key), look up the value (i)
print(f"COL_INDEX is {COL_INDEX}\n\n")


#
# and our "species" names - these aren't columns, nor rows. They're the TARGET classifications!
#

# all of scikit-learn's ML routines need numbers, not strings
#   ... even for categories/classifications (like species!)
#   so, we will convert the flower-species to numbers:

number_INDEX = ['0','1','2','3','4','5','6','7','8','9']

# Let's try it out...
for number in number_INDEX:
    print(f"{number} maps to {number}")


