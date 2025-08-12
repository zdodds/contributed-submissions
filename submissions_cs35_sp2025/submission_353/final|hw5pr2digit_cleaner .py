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
# Read in the spreadsheet (CSV)

filename = 'digits.csv'
df = pd.read_csv(filename)  # Adjust encoding if necessary
print(f"{filename} : file read into a pandas dataframe.")

# Define ROW and COLUMN for clarity
ROW = 0        
COLUMN = 1     

# Drop the final column if it exists (adjust column name if different)
unwanted_column = 'excerpted from http://yann.lecun.com/exdb/mnist/'
if unwanted_column in df.columns:
    df_clean1 = df.drop(unwanted_column, axis=COLUMN)
else:
    df_clean1 = df.copy()

df_clean1

# Drop all rows with missing data (NaN values)
df_clean2 = df_clean1.dropna()
df_clean2.info()
df_clean2

# Make a copy before adding new columns
df_clean3 = df_clean2.copy()

# Drop the 'actual_digit' column as it will be used as labels in modeling
df_clean3_labels = df_clean3['actual_digit']
df_clean3.drop(columns=['actual_digit'], inplace=True)

df_tidy = df_clean3

# Construct the new filename
cleaned_filename = filename.replace('.csv', '_cleaned.csv')
print(f"cleaned_filename is {cleaned_filename}")

# Save the cleaned dataframe
df_tidy.to_csv(cleaned_filename, index=False)  # No "index" column

# Re-read and verify the cleaned data
df_tidy_reread = pd.read_csv(cleaned_filename)
print(f"{cleaned_filename} : file read into a pandas dataframe.")
df_tidy_reread


