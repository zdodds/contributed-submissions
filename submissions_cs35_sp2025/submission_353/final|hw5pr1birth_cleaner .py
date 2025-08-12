#
# hw5pr1births_cleaner:  birth classification by month + day    (above/below median: 190942)



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
# Read in the spreadsheet (CSV)
filename = 'births.csv'
df = pd.read_csv(filename)  # Adjust encoding if necessary
print(f"{filename} : file read into a pandas dataframe.")

# Define ROW and COLUMN for clarity
ROW = 0        
COLUMN = 1     

# Drop the final column if it exists (adjust column name if different)
unwanted_column = 'from http://chmullig.com/2012/06/births-by-day-of-year/'
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

# Convert 'above/below median' column to numerical representation
def convert_above_below(value):
    return 1 if value.strip().lower() == 'above' else 0

df_clean3['above_median'] = df_clean3['above/below median'].apply(convert_above_below)

df_clean3.drop(columns=['above/below median'], inplace=True)

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





