#
# hw5pr1births_cleaner:  birth classification by month + day    (above/below median: 190942)
#


# libraries!
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)


# read in our spreadsheet (csv)
filename = 'births.csv'
df = pd.read_csv(filename)        # encoding="utf-8" et al.
print(f"{filename} : file read into a pandas dataframe.")


#
# a dataframe is a "spreadsheet in Python"   (note that extra column at the right!)
#
# let's view it:



# let's look at the dataframe's "info":
df.info()


# Let's look at the dataframe's columns:
df.columns


# we can drop a series of data (a row or a column)
# the dimensions each have a numeric value, row~0, col~1, but let's use readable names we define:
ROW = 0        
COLUMN = 1     

# We are dropping this final column - it was just a citation...
df_clean1 = df.drop('from http://chmullig.com/2012/06/births-by-day-of-year/', axis=COLUMN)
df_clean1

# df_clean1 is a new dataframe, without that unwanted column


df_clean1.info() 


#
# let's drop _all_ rows with data that is missing/NaN (not-a-number)
df_clean2 = df_clean1.dropna()  # drop na rows (NaN, not-a-number)
df_clean2.info()  # print the info, and
# let's see the whole table, as well:
df_clean2

# Tidy!  Our data is ready!


#
# let's keep our column names in variables, for reference
#
COLUMNS = df_clean2.columns            # "list" of columns
print(f"COLUMNS is {COLUMNS}")  
  # It's a "pandas" list, called an Index
  # use it just as a Python list of strings:
print(f"COLUMNS[0] is {COLUMNS[0]}\n")

# let's create a dictionary to look up an index from its name:
COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i  # using the name (as key), look up the value (i)
print(f"COL_INDEX is {COL_INDEX}")
print(f"COL_INDEX[ 'month' ] is {COL_INDEX[ 'month' ]}")


# all of scikit-learn's ML routines need numbers, not strings
#   ... even for categories/classifications (like species!)
#   so, we will convert the flower-species to numbers

#
# First, let's map our different species to numeric values:

levels = ['above','below']   # int to str
levels_index = {'above':0,'below':1}  # str to int

def convert_level(levelname):
    """ return the species index (a unique integer/category) """
    return levels_index[levelname]

# Let's try it out...
for level in levels:
    print(f"{level} maps to {convert_level(level)}")


convert_level('above') 


#
# we can "apply" to a whole column and create a new column
#   it may give a warning, but this is ok...
#

df_clean3 = df_clean2.copy()  # copy everything AND...

# add a new column, 'irisnum'
df_clean3['levelnum'] = df_clean2['above/below median'].apply(convert_level)

# let's see...
df_clean3


#
# it's no fun fiddling with the default table formatting.
#
# Remember: we have the data itself!  If we want to see it, we can print:

(NROWS, NCOLS) = df_clean3.shape
print(f"There are {NROWS = } and {NCOLS = }")
print()

for row in range(0,NROWS,5):
    print(df_clean3[row:row+5])    # Let's print 5 at a time...



#
# let's call it df_tidy 
#
df_tidy =  df_clean3



#
# That's it!  Then, and write it out to iris_cleaned.csv

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
# *** SUGGESTION ***  
# 
#       +++ copy-paste-and-alter from the iris-cleaning notebook to here +++
#
# This approach has the advantage of more deeply "digesting" the iris workflow...
#      ... altering the parts that don't transfer, and taking the parts that do!
#

# eventually you'll get rid of the _births_ column, that can be here or in the modeling notebook


