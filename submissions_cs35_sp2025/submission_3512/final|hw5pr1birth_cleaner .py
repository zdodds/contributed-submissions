#
# hw5pr1births_cleaner:  birth classification by month + day    (above/below median: 190942)
#


# libraries!
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)


# read in our spreadsheet (csv)
filename = 'births.csv'
bdf = pd.read_csv(filename)        # encoding="utf-8" et al.
print(f"{filename} : file read into a pandas dataframe.")





bdf.info()


bdf.columns


ROW = 0        
COLUMN = 1     

# We are dropping this final column - it was just a citation...
bdf_clean1 = bdf.drop('from http://chmullig.com/2012/06/births-by-day-of-year/', axis=COLUMN)
bdf_clean1



bdf_clean2 = bdf_clean1.drop([60, 61, 123, 185, 278, 340], axis=ROW)

def convert_median(ba):
    if ba == 'below':
        return 0
    elif ba == 'above':
        return 1
    
bdf_clean3 = bdf_clean2.copy() 

bdf_clean3['above/below one/zero'] = bdf_clean2['above/below median'].apply(convert_median)

# let's see...
bdf_clean3


bdf_clean3.info()  


COLUMNS = bdf_clean3.columns            # "list" of columns
print(f"COLUMNS is {COLUMNS}")  
  # It's a "pandas" list, called an Index
  # use it just as a Python list of strings:
print(f"COLUMNS[0] is {COLUMNS[0]}\n")

# let's create a dictionary to look up an index from its name:
COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i  # using the name (as key), look up the value (i)
print(f"COL_INDEX is {COL_INDEX}")
print(f"COL_INDEX[ 'births' ] is {COL_INDEX[ 'births' ]}")


# check that right rows were deleted... 
b = bdf_clean3['births']
print(b.min())


(NROWS, NCOLS) = bdf_clean3.shape
print(f"There are {NROWS = } and {NCOLS = }")
print()

for row in range(0,NROWS,5):
    print(bdf_clean3[row:row+5]) 


bdf_tidy =  bdf_clean3


old_filename_without_extension = filename[:-4]                      # remove the ".csv"

cleaned_filename = old_filename_without_extension + "_cleaned.csv"  # name-creating
print(f"cleaned_filename is {cleaned_filename}")

# Now, save the dataframe names df_tidy
bdf_tidy.to_csv(cleaned_filename, index_label=False)  # no "index" column...


bdf_tidy_reread = pd.read_csv(cleaned_filename)   # encoding="utf-8" et al.
print(f"{cleaned_filename} : file read into a pandas dataframe.")
bdf_tidy_reread


COLUMNS = bdf_tidy_reread.columns            # "list" of columns
print(f"COLUMNS is {COLUMNS}\n")  
  # It's a "pandas" list, called an Index
  # use it just as a Python list of strings:
print(f"COLUMNS[0] is {COLUMNS[0]}\n")

# let's create a dictionary to look up any column index by name
COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i  # using the name (as key), look up the value (i)
print(f"COL_INDEX is {COL_INDEX}\n\n")




