#
# hw5pr1births_cleaner:  birth classification by month + day    (above/below median: 190942)
#


# libraries!
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)


#
# *** SUGGESTION ***  
# 
#       +++ copy-paste-and-alter from the iris-cleaning notebook to here +++
#
# This approach has the advantage of more deeply "digesting" the iris workflow...
#      ... altering the parts that don't transfer, and taking the parts that do!
#

# eventually you'll get rid of the _births_ column, that can be here or in the modeling notebook


# read in our spreadsheet (csv)
filename = 'births.csv'
df = pd.read_csv(filename)        # encoding="utf-8" et al.
print(f"{filename} : file read into a pandas dataframe.")


# let's look at the dataframe's "info":
df.info()


df.columns


ROW = 0        
COLUMN = 1     

# We are dropping this final column - it was just a citation...
df_clean1 = df.drop('from http://chmullig.com/2012/06/births-by-day-of-year/', axis=COLUMN)
df_clean1


df_clean1.info() 


df_clean3 = df_clean1.dropna()  # drop na rows (NaN, not-a-number)
df_clean3.info()


df_clean4 = df_clean3.drop('births', axis=COLUMN)
df_clean4


# let's keep our column names in variables, for reference
#
COLUMNS = df_clean4.columns            # "list" of columns
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


# make the above/below object a numeric 0/1

SPECIES = ['above','below']   # int to str
SPECIES_INDEX = {'above':0,'below':1}  # str to int

def convert_species(speciesname):
    """ return the species index (a unique integer/category) """
    return SPECIES_INDEX[speciesname]

# Let's try it out...
for name in SPECIES:
    print(f"{name} maps to {convert_species(name)}")


df_clean5 = df_clean4.copy()  # copy everything AND...

# add a new column
df_clean5['above/below_num'] = df_clean4['above/below median'].apply(convert_species)

# let's see...
df_clean5


(NROWS, NCOLS) = df_clean5.shape
print(f"There are {NROWS = } and {NCOLS = }")
print()

for row in range(0,NROWS,5):
    print(df_clean4[row:row+5])    # Let's print 5 at a time...


df_tidy =  df_clean5


#
# That's it!  Then, and write it out to births_cleaned.csv

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

SPECIES = ['above','below']   # int to str
SPECIES_INDEX = {'above':0,'below':1}  # str to int

def convert_species(speciesname):
    """ return the species index (a unique integer/category) """
    return SPECIES_INDEX[speciesname]

# Let's try it out...
for name in SPECIES:
    print(f"{name} maps to {convert_species(name)}")


