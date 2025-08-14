#
# hw5pr1births_cleaner:  birth classification by month + day    (above/below median: 190942)
#


#
# *** SUGGESTION ***  
# 
#       +++ copy-paste-and-alter from the iris-cleaning notebook to here +++
#
# This approach has the advantage of more deeply "digesting" the iris workflow...
#      ... altering the parts that don't transfer, and taking the parts that do!
#

# eventually you'll get rid of the _births_ column, that can be here or in the modeling notebook


# libraries!
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)


# read in our spreadsheet (csv)
filename = 'births.csv'
df = pd.read_csv(filename)
print(f"{filename} : file read into a pandas dataframe.")







df.info()



df_clean1 = df.drop('from http://chmullig.com/2012/06/births-by-day-of-year/', axis=1)
df_clean1


df_clean2 = df_clean1[df_clean1['births'] > 9000]
df_clean2


import calendar


def is_valid_date(row):
    try:
        # check if the date is valid
        calendar.monthrange(2023, row['month'])[1] >= row['day']
        return True
    except:
        return False


df_clean3 = df_clean2[df_clean2.apply(is_valid_date, axis=1)]
df_clean3


df_clean3['above/below median'] = df_clean3['above/below median'].map({'above': 1, 'below': 0})
df_clean3


cleaned_filename = 'births_cleaned.csv'
df_clean3.to_csv(cleaned_filename, index=False)
print(f"Cleaned data saved to {cleaned_filename}")


print(f"Minimum number of births: {df_clean3['births'].min()}")


print(df_clean3[['month', 'day']].value_counts().sort_index())



