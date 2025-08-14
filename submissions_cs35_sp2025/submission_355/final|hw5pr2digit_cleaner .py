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

# libraries!
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)


# read in our spreadsheet (csv)
filename = 'digits.csv'
df = pd.read_csv(filename)        # encoding="utf-8" et al.
print(f"{filename} : file read into a pandas dataframe.")


# df

# df.info()

df.columns



# we can drop a series of data (a row or a column)
# the dimensions each have a numeric value, row~0, col~1, but let's use readable names we define:
ROW = 0        
COLUMN = 1     

# We are dropping this final column - it was just a citation...
df_clean1 = df.drop('excerpted from http://yann.lecun.com/exdb/mnist/', axis=COLUMN)
df_clean1

# df_clean1 is a new dataframe, without that unwanted column


df_clean3 = df_clean1.dropna()  # drop na rows (NaN, not-a-number)
df_clean3.info()  # print the info, and
# let's see the whole table, as well:
df_clean3


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
print(f"COL_INDEX[ 'pix2' ] is {COL_INDEX[ 'pix2' ]}")



# all of scikit-learn's ML routines need numbers, not strings
#   ... even for categories/classifications (like species!)
#   so, we will convert the flower-species to numbers

#
# First, let's map our different species to numeric values:

SPECIES = [f'pix{i}' for i in range(64)]   # int to str
SPECIES_INDEX = {pixel: int(pixel[3:]) for pixel in SPECIES}  # str to int

def convert_species(speciesname):
    """ return the species index (a unique integer/category) """
    return SPECIES_INDEX[speciesname]

# Let's try it out...
for name in SPECIES:
    print(f"{name} maps to {convert_species(name)}")


# the actual digit column is already numeric, so there shouldn't need to be any changes there


#
# it's no fun fiddling with the default table formatting.
#
# Remember: we have the data itself!  If we want to see it, we can print:

df_clean4 = df_clean3

(NROWS, NCOLS) = df_clean4.shape
print(f"There are {NROWS = } and {NCOLS = }")
print()

for row in range(0,NROWS,5):
    print(df_clean4[row:row+5])    # Let's print 5 at a time...



df_tidy =  df_clean4


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


# will recreate variables in the modeling file


