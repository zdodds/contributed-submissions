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



import numpy as np
import pandas as pd


filename = 'digits.csv'
df = pd.read_csv(filename) 
print(f"{filename} : file read into a pandas dataframe.")





df.info()


df.columns


ROW = 0       
COLUMN = 1    


df_clean1 = df.drop('adapted from https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits', axis=COLUMN)
df_clean1


df_clean2 = df_clean1.dropna() 
df_clean2.info()  
df_clean2


df_tidy = df_clean2


old_filename_without_extension = filename[:-4]                      
cleaned_filename = old_filename_without_extension + "_cleaned.csv"  
print(f"cleaned_filename is {cleaned_filename}")



df_tidy.to_csv(cleaned_filename, index_label=False)


df_tidy_reread = pd.read_csv(cleaned_filename)  
print(f"{cleaned_filename} : file read into a pandas dataframe.")
df_tidy_reread



COLUMNS = df_tidy_reread.columns           
print(f"COLUMNS is {COLUMNS}\n")  
print(f"COLUMNS[0] is {COLUMNS[0]}\n")



COL_INDEX = {}
for i, name in enumerate(COLUMNS):
    COL_INDEX[name] = i  # using the name (as key), look up the value (i)
print(f"COL_INDEX is {COL_INDEX}\n\n")


