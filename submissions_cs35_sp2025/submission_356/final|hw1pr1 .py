# Where are we?



# what's here?



# to move around:  cd stands for "change directory" (a directory is a folder)
#    # intro_first    would move into the intro_first folder
#    # .. moves "up" to the containing directory
#    # .  doesn't move at all:  .  represents the current directory    

# For now, let's not move anywhere



# we will use a few file-handling "system" libraries. 
# These are built-in to python, so nothing to install - just to import:
import os
import os.path


#
# In fact, we can read it - it's just not a .ipynb file!
#
# Try it here, for your system:

print("+++ Contents of the file nottrue.ipynb: +++\n")

# Mac:    !cat  <filepath>  using forward slashes
#
# !cat ./intro_first/nottrue.ipynb       

# Windows:  type <filepath>  using backslashes
#
# !type .\\intro_first\\nottrue.ipynb      

L = os.walk("./intro_first/nottrue.ipynb") #root
final_list = list(L)
print(f"{final_list = }")


#
# function to return the contents of a file (as one-big-string)
#

def GET_STRING_FROM_FILE(filename_as_string):
    """ return all of the contents from the file, filename
        will error if the file is not present, then return the empty string ''
    """
    try:
        # the encoding below is a common default, but not universal...
        file_object = open(filename_as_string, "r", encoding='utf-8')    # open! (Other encodings: 'latin-1', 'utf-16', 'utf-32') 
        file_data = file_object.read()                                   # and get all its contents
        file_object.close()                                              # close the file (optional)
        #print(DATA)                                                     # if we want to see it
        return file_data                                                 # definitely want to return it!
    except FileNotFoundError:                             # it wasn't there
        print(f"file not found: {filename_as_string}")    # print error
        return ''                                         # return empty string ''
    except UnicodeDecodeError:
        print(f"decoding error: {filename_as_string}")    # encoding/decoding error  
        return ''                                         # return empty string ''


full_file_path = "./intro_first/nottrue.ipynb"
file_contents = GET_STRING_FROM_FILE(full_file_path)      # reminder: file_contents = file_data from above

# Let's print only some of this potentially large string, adapting as needed:
print("file_contents:\n\n", file_contents[0:42])          # let's then increase the 42...


####  Let's try one of the other files!  (or a non-existent file!)

full_file_path = "./intro_first/cs/file35.txt"    # how about the others?!
file_contents = GET_STRING_FROM_FILE(full_file_path)     
print("file_contents:\n\n", file_contents[0:42])


#
# Steppingstone, Version 0: does Python work?
#

import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    return 42  # just to check that it's working (v0)    

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    print(f"[[ Start! ]]\n")     # sign on

    path = "./intro_first"       # Remember: . means the current directory
    result = file_walker(path)   # Run!

    print(f"result = {result}")  # Yay for f-strings!

    print("\n[[ Fin. ]]")        # sign off



# os.walk returns the structure of a folder (directory)

# Here, we "walk" the intro_examples subfolder:
all_files = os.walk("./intro_first")

all_files     # oops! it's a "generator object"


import os
L = list( os.walk( "./intro_first" ) )  
print(f"{len(L) = }")
print(f"{L = }")


from IPython import display
#
# this is in the hw1pr1 folder
#
display.Image("./intro_first_ss_small.png")   # local image


path = "./intro_first"          # any path to any folder
result = list(os.walk(path))    # this will "walk" all of the subfolders and files

print(f"{len(result) = }")      # try c:/  (it took my machine 12.7 seconds!)
print(f"{result = }")


#
# Steppingstone, Version 1: call os.walk, return length, optionally print
#

import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    # print(f"{len(result) = }")
    # print(f"{result = }")
    num_folders = len(result)        # the len is the number of folders...
    return num_folders

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    print(f"[[ Start! ]]\n")     # sign on

    path = "./intro_first"       # Remember: . means the current directory
    result = file_walker(path)   # Run!

    print(f"result = {result}")  # Yay for f-strings!

    print("\n[[ Fin. ]]")        # sign off



#
# Steppingstone, Version 2: print all of the folder names!
#

import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        print(f"{currentpath = }")   # try non-f printing: it's worse!

    num_folders = len(result)        # the len is the number of currentpaths...
    return num_folders

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    print(f"[[ Start! ]]\n")     # sign on

    path = "./intro_first"       # Remember: . means the current directory
    result = file_walker(path)   # Run!

    print(f"result = {result}")  # Yay for f-strings!

    print("\n[[ Fin. ]]")        # sign off



#
# Steppingstone, Version 3: walk all of the files, printing each one's fullpath
#

import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...

        if '__MACOSX' in currentpath: continue         # skip the rest of _this_ loop iteration: back to top

        print(f"{currentpath = }") 
        
        for file in files:       # remember, files is a list of filenames!
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            print(f"   {fullpath = }")
            #contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            #print(f"{contents[0:42] = }")

    num_folders = len(result)        # the len is the number of currentpaths...
    return num_folders

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    print(f"[[ Start! ]]\n")     # sign on

    path = "./intro_first"       # Remember: . means the current directory
    result = file_walker(path)   # Run!

    print(f"result = {result}")  # Yay for f-strings!

    print("\n[[ Fin. ]]")        # sign off



os.path.join("/root/Users/secret_stuff" , "file_name")


#
# Steppingstone, Version 4: walk all of the files, printing (bits of) each one's contents!
#

import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            print(f"   {fullpath = }")
            contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            print(f"   {contents[0:42] = }")

    num_folders = len(result)        # the len is the number of currentpaths...
    return num_folders

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    print(f"[[ Start! ]]\n")     # sign on

    path = "./intro_first"       # Remember: . means the current directory
    result = file_walker(path)   # Run!

    print(f"result = {result}")  # Yay for f-strings!

    print("\n[[ Fin. ]]")        # sign off



#
# Rolodex lead-in, example1: counting the number of .txt files...
#

import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    
    count_txt = 0    # keep count of our .txt files

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot
            if file[-4:] == ".txt":
                print("Found a .txt file! Adding one...")
                count_txt += 1
            #contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            #print(f"   {contents[0:42] = }")

    return count_txt   # phew, we're finally returning something else!

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    print(f"[[ Start! ]]\n")     # sign on

    path = "./intro_first"       # Remember: . means the current directory
    result = file_walker(path)   # Run!

    print(f"num txt files = {result}")  # Yay for f-strings!

    print("\n[[ Fin. ]]")        # sign off



#
# Rolodex lead-in, example2: counting the number of .txt files containing 'CS' ...
#

import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    
    count_txt = 0    # keep count of our .txt files
    count_CS = 0     # keep count of 'CS' substrings found

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot

            if file[-4:] == ".txt":
                # print("Found a .txt file! Adding one...")
                count_txt += 1
                contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
                if 'CS' in contents:
                    print("        *** Found a 'CS' ... adding 1    (aka 2-True)")
                    count_CS += 1
                # print(f"   {contents[0:42] = }")

    return count_CS, count_txt   # oooh... we can return two things!

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    print(f"[[ Start! ]]\n")     # sign on

    path = "./intro_first"       # Remember: . means the current directory
    result = file_walker(path)   # Run!

    count_CS, count_txt = result
    print()
    print(f"num txt files       = {count_txt}")  
    print(f"num containing CS   = {count_CS}")  
    perc = count_CS*100/count_txt
    print(f"for a CS percentage of {perc:5.2f}%")   # :5.2f means width of 5, 2 dec. places, for a _floating pt value

    print("\n[[ Fin. ]]")        # sign off



path = "./intro_first"          # any path to any folder?!  intro_first contains _5_ folders total
# path = "./intro_second"       # any path to any folder?!  intro_second contains _12_ folders total
# path = "./files_challenge"    # this is the really large folder: it contains _23_ folders total

# path = "C:/"                  # could use C:/  on windows or "/" on MacOS  
# path = "."                    # could use "." for the current directory

result = list(os.walk(path))    # this will "walk" all of the subfolders and files

print(f"{len(result) = }")      # this took my machine 2m 47.4s seconds (for "/" with total 555695 folders)
                                # and, it asked for permission a couple of times (I said no.)
#print(f"{result = }")          # let's _not_ print it out...


cd '/Users/amandadee/Desktop/week1.1/week1_sum24/hw1pr1'


# How many .txt files are in the entire folder? 
# Hint: the starting file, above, helps a lot on this one!  It's many thousands of them...

import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    
    count_txt = 0    # keep count of our .txt files

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot
            if file[-4:] == ".txt":
                print("Found a .txt file! Adding one...")
                count_txt += 1
            #contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            #print(f"   {contents[0:42] = }")

    return count_txt   # phew, we're finally returning something else!

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    print(f"[[ Start! ]]\n")     # sign on

    path = "."       # Remember: . means the current directory
    result = file_walker(path)   # Run!

    print(f"num txt files = {result}")  # Yay for f-strings!

    print("\n[[ Fin. ]]")        # sign off


# What is the maximum depth of directories in the entire folder (in other words, what's the maximum number of times that it's possible to move deeper into a subdirectory, overall)? 
# Hint: count the number of forward-slashes and backward-slashes!
# How to count?  Here is an example:    'geese'.count('e')  returns 3.    'a/b/c'.count('/')  returns 2.
# Try small examples in a custom cell (it's tricky with the '\\' character, so you'll want to pre-debug this!
# Key:  it's discovery -- don't try to solve it, rather explore + discover your way, zig-zag'ing towards useful stuff!

import os
import os.path

def max_depth(path):
    max_depth = 0

    result = os.walk(path)     # perhaps try w/o converting to a list...

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        print(f"{currentpath = }")   # try non-f printing: it's worse!

        depth = max(currentpath.count('/'),currentpath.count('\\')) # to get separator slashes

        max_depth = max(max_depth,depth) # store new max depth if depth is greater than previously stored

    return max_depth

path = "."
result = max_depth(path)
print(f"Maximum directory depth: {result}")



# Across all of the files, how many of the phone numbers contain exactly 10 digits? 7 digits? 11 digits? 
# Hint:  Remember the function from week0 where you extracted only  the digits?  That will be helpful!
# Here, you might want a function with an extra input to indicate the number of digits to search for!

# Of the exactly-ten-digit phone numbers, how many are in the area code 909 (the area code will be the first three digits of a ten-digit number). 
# Hint: cleaning, slicing, Python - yay!

# Are there any phone numbers that have more than 10 digits? 

import os

def just_digits(s):
    ''' returns only the digits in input string s '''
    digits = []

    for i in s:
        if i.isdigit():
            digits.append(i)

    return ''.join(digits)

def file_walker(path,exact_digits = None):
    ''' returns number of phone numbers that contain exactly 7,10,11(,12,13) digits '''
    ''' also counts how many of the 10-digit numbers are in the area code 909 '''
    result = list(os.walk(path))  # perhaps try w/o converting to a list...
    
    count_digits = {7:0,10:0,11:0,12:0,13:0}  # dictionary to store counts of 7,10,11(,12,13) digit lengths
    count_909 = 0 # keep count of 909 phone numbers

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:
            continue
        
        for file in files:
            if file.startswith('.') or not file.endswith('.txt'):
                continue
            
            fullpath = os.path.join(currentpath, file)
            with open(fullpath, 'r', encoding='utf-8', errors='ignore') as f:
                contents = f.read()
                digits = just_digits(contents)
                length = len(digits)
                
                if length in count_digits:  # only count lengths if 7,10,11(,12,13) digits
                    count_digits[length] += 1

                    if length == 10 and digits.startswith("909"): # only count phone numbers if in the area code 909
                        count_909 += 1
    
    return count_digits, count_909

path = "."
count_digits, count_909 = file_walker(path)
for length in [7,10,11,12,13]:
    print(f"Total {length}-digit phone numbers: {count_digits[length]}")
print(f"Total 10-digit phone numbers in area code 909: {count_909}")


# How many people have your last name?    
# Hint#1:   use the presence of a comma to determine whether the last name is 1st or 2nd!
# Hint#2:   watch out for newline characters, '\n'    Consider s.split('\n'), s.strip(), s.startswith, …

# Choose another first name present:  How many people have that last name?

# How many people have your first name? 

# Choose another first name present: How many people have that first name?
# The answers to these 4 questions are collated in the following markdown

import os

def get_string_from_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()  # remove unnecessary whitespace
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def clean_name(name):
    ''' removes all elements other than string. '''
    return "".join([c for c in name if c.isalpha() or c == " " or c == ","]).strip() # isalpha is alphabet

def file_walker(path, target_first_name=None, target_last_name=None):
    ''' counts people by first and last names. '''
    
    first_name_count = 0
    last_name_count = 0

    for currentpath, _, files in os.walk(path):  
        if '__MACOSX' in currentpath:
            continue
        
        print(f"{currentpath = }")  

        for file in files:
            fullpath = os.path.join(currentpath, file)

            if file.startswith(".") or not file.endswith(".txt"):  
                continue  

            contents = get_string_from_file(fullpath)
            name = clean_name(contents)  

            if "," in name:
                # Format: LastName, FirstName
                parts = name.split(",")
                if len(parts) >= 2:
                    last_name = parts[0].strip().lower()
                    first_name = parts[1].strip().lower()
            else:
                # Format: FirstName LastName
                parts = name.split()
                if len(parts) >= 2:
                    first_name = parts[0].strip().lower()
                    last_name = parts[-1].strip().lower()

            # count occurrences
            if target_first_name and first_name == target_first_name.lower():
                first_name_count += 1
            if target_last_name and last_name == target_last_name.lower():
                last_name_count += 1

    return first_name_count, last_name_count

if True:
    print(f"[[ Start! ]]\n")  

    path = "."  
    target_first_name = "Emmie"  # Change this to any first name
    target_last_name = "Canel"       # Change this to any last name

    first_name_count, last_name_count = file_walker(path, target_first_name, target_last_name)

    print(f"\nTotal people with first name '{target_first_name}': {first_name_count}")
    print(f"Total people with last name '{target_last_name}': {last_name_count}")

    print("\n[[ Fin. ]]")  


# How many people have three "i"'s somewhere in their name (not necessarily consecutiiive!)
# s.count("i") is a friend!    (This can be case sensitive/insensitive: that is up to you.)

import os

def get_string_from_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()  # remove unnecessary whitespace
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def clean_name(name):
    ''' removes all elements other than string. '''
    return "".join([c for c in name if c.isalpha() or c == " " or c == ","]).strip() # isalpha is alphabet

def count_eyes(s):
    ''' returns the number of times i or I appears in the input string s '''
    result = 0
    for i in range(len(s)): 
        if s[i] in 'iI': 
            result += 1
    return result

def file_walker(path):
    '''counts names with at least 3 i or Is'''
    count_three_i = 0  # count names with at least 3 iIs

    for currentpath, _, files in os.walk(path):  
        if '__MACOSX' in currentpath:
            continue
        print(f"{currentpath = }")  

        for file in files:
            fullpath = os.path.join(currentpath, file)

            if file.startswith(".") or not file.endswith(".txt"):  
                continue  

            contents = get_string_from_file(fullpath)
            name = clean_name(contents)  

            i_count = count_eyes(name)  # Count 'i' and 'I' using provided method

            if i_count >= 3:
                print(f"{name} has {i_count} iIs")
                count_three_i += 1

    return count_three_i

path = "."  
count_three_i = file_walker(path)
print(f"Total people with at least 3 'iI's in their name: {count_three_i}")


# Chosen top-level folder
cd /Users/amandadee/Desktop/cs5


# How many files are there total?  (Choose somewhere with at least 42 files.)

import os
import os.path

def file_walker(path):
    """ Walks the entire directory structure from `path` 
        and counts the total number of files (all file types).
    """
    result = list(os.walk(path))  # Convert generator to list for processing
    
    total_files = 0  # Count of all files

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath: continue         # skip the rest of _this_ loop iteration: back to top

        print(f"{currentpath = }") 
        
        for file in files:  
            if file.startswith("."):  
                continue 
            
            fullpath = os.path.join(currentpath, file)  #  full file path
            print(f"   {fullpath = }")  
            
            total_files += 1  

    return total_files 

path = "."  
result = file_walker(path) 

print(f"Total number of files = {result}") 


# How deep was the deepest path present?

import os
import os.path

def max_depth(path):
    max_depth = 0

    result = os.walk(path)     # perhaps try w/o converting to a list...

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        print(f"{currentpath = }")   # try non-f printing: it's worse!

        depth = max(currentpath.count('/'),currentpath.count('\\')) # to get separator slashes

        max_depth = max(max_depth,depth) # store new max depth if depth is greater than previously stored

    return max_depth

path = "."
result = max_depth(path)
print(f"Maximum directory depth: {result}")


# How many ".py" files
import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    
    count_py = 0    # keep count of our .py files

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot
            if file[-3:] == ".py":
                print("Found a .py file! Adding one...")
                count_py += 1
            #contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            #print(f"   {contents[0:42] = }")

    return count_py   # phew, we're finally returning something else!

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    print(f"[[ Start! ]]\n")     # sign on

    path = "."       # Remember: . means the current directory
    result = file_walker(path)   # Run!

    print(f"num py files = {result}")  # Yay for f-strings!

    print("\n[[ Fin. ]]")        # sign off


# Which is the deepest folder 
#

import os

def deepest_folder(path):
    result = list(os.walk(path))  
    max_depth = 0
    deepest_folder = ""

    for folder_tuple in result:
        currentpath, _, _ = folder_tuple  
        depth = max(currentpath.count('/'), currentpath.count('\\'))

        if depth > max_depth:
            max_depth = depth
            deepest_folder = currentpath  

    return deepest_folder, max_depth

path = "."
deepest_folder, max_depth = deepest_folder(path)

print(f"Deepest Folder: {deepest_folder}")
print(f"Maximum Depth: {max_depth}")


# How many files have hw in the filename aka how many hw files did i submit in sophomore year !

import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    
    count_py = 0    # keep count of our .py files
    count_hw = 0     # keep count of 'hw' substrings found

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot

            if file[-3:] == ".py":
                # print("Found a .py file! Adding one...")
                count_py += 1
                
                if file.startswith("hw"):
                    print("        *** Found 'hw' in filename ... adding 1    (aka 2-True)")
                    count_hw += 1
                # print(f"   {contents[0:42] = }")

    return count_hw, count_py   # oooh... we can return two things!

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    print(f"[[ Start! ]]\n")  

    path = "."  
    count_hw, count_py = file_walker(path)  

    print()
    print(f"Total .py files       = {count_py}")  
    print(f".py files containing 'hw' in filename = {count_hw}")  

    if count_py > 0:
        perc = count_hw * 100 / count_py
        print(f"Percentage of 'hw' .py files = {perc:5.2f}%")  
    else:
        print("No .py files found, so percentage calculation is not applicable.")

    print("\n[[ Fin. ]]") 


