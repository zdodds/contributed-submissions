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

# <c:\\Users\\CS35_Participant_12\\OneDrive\\Desktop\\CS35\\week1\\starting_notebooks\\starting_notebooks>

#
# 
# !type c:\\Users\\CS35_Participant_12\\OneDrive\\Desktop\\CS35\\week1\\starting_notebooks\\starting_notebooks\\intro_first\\nottrue.ipynb


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
            fullpath = currentpath + "/" + file        # construct the full path, or, better: os.path.join(currentpath,file)
            if '.' == file[0] == '.': continue             
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


# how many .txt files are in the entire folder? (starting w/ modifying code from example 1)

import os
import os.path

def file_walker(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders and counts .txt files, returns an integer count of .txt files
    """
    result = list(os.walk(path))    
    count_txt = 0    # keep count of our .txt files

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        # print(f"{currentpath = }") 

        for file in files:      
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            # print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot
            if file[-4:] == ".txt":
                count_txt += 1

    return count_txt

if True:
    """ overall script that runs examples """

    path1 = "./intro_first"
    path2 = "./intro_second"
    path3 = "./files_challenge"
    result1 = file_walker(path1)   # Run!
    result2 = file_walker(path2)
    result3 = file_walker(path3)

    print(f"num txt files in intro_first = {result1}")  # Yay for f-strings!
    print(f"num txt files in intro_second = {result2}")
    print(f"num txt files in files_challenge = {result3}")


# What is the maximum depth of directories in the entire folder?

def file_walker(path):
    """ starting from the input, named path
        this function "walks" the whole path, including subfolders and counts forward and backslashes (counting double slashes as 1 slash) to 
        find the maximum depth of a directory in files_challenge (or other paths). it adds the depth of each file to a list and returns the maximum depth
    """
    result = list(os.walk(path))    
    depth_count = 0  
    depths = []

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        # print(f"{currentpath = }") 

        for file in files:      
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            # print(f"   {fullpath = }")
            # if file[0] == ".": continue      # skip files that start with dot
            for char in str(fullpath):
                if char == '/' or char == '\\':
                    depth_count += 1
            depths.append(depth_count)
            depth_count = 0

    return max(depths)

if True:
    """ overall script that runs examples """

    path1 = "./intro_first"
    path2 = "./intro_second"
    path3 = "./files_challenge"
    result1 = file_walker(path1)   # Run!
    result2 = file_walker(path2)
    result3 = file_walker(path3)

    print(f"max depth in intro_first = {result1}")  # Yay for f-strings!
    print(f"max depth in intro_second = {result2}")
    print(f"max depth in files_challenge = {result3}")



# of all the files, how many of the phone numbers contain exactly ten digits?

import string

def file_walker(path):
    """ starting from the input, named path
        this function "walks" the whole path, including subfolders and counts the number of digits in every .txt phonebook file. 
        it tallies the total number of these files with exactly ten digits and returns that total count
    """
    result = list(os.walk(path))    
    phone10_count = 0
    phone7_count = 0
    phone11_count = 0

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        # print(f"{currentpath = }") 

        for file in files:      
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            # print(f"   {fullpath = }")
            # if file[0] == ".": continue      # skip files that start with dot
            if file[-4:] == ".txt": # only looking at txt files 
                contents = GET_STRING_FROM_FILE(fullpath)
                for x in contents:
                    if x in string.digits:
                        count += 1
                if count == 10:
                    phone10_count += 1
                if count == 11:
                    phone11_count += 1
                if count == 7:
                    phone7_count += 1
            count = 0

    return phone10_count, phone11_count, phone7_count

if True:
    """ overall script that runs examples """

    path1 = "./intro_first"
    result1 = file_walker(path1)

    path2 = "./intro_second"
    result2 = file_walker(path2)

    path3 = "./files_challenge"
    result3 = file_walker(path3)

    print("For files_challenge:")
    print(f"10 digit phone number count = {result3[0]}")  
    print(f"11 digit phone number count = {result3[1]}")
    print(f"7 digit phone number count = {result3[2]}")

    print()
    print(f"For {path2}:")
    print(f"10 digit phone number count = {result2[0]}") 
    print(f"11 digit phone number count = {result2[1]}")
    print(f"7 digit phone number count = {result2[2]}")

    print()
    print("For intro_first:")
    print(f"10 digit phone number count = {result1[0]}")  
    print(f"11 digit phone number count = {result1[1]}")
    print(f"7 digit phone number count = {result1[2]}")



# How many of the 10 digits phone numbers start with 909?

def file_walker(path):
    """ starting from the input, named path
        this function "walks" the whole path, including subfolders and creates a string of just the digits in every .txt file. 
        it counts the total instances where the digit string is 10 digits CS35_Participant_2 and starts with 909, and returns that integer count
    """
    result = list(os.walk(path))    
    area909_count = 0

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        # print(f"{currentpath = }") 

        for file in files:      
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            # print(f"   {fullpath = }")
            # if file[0] == ".": continue      # skip files that start with dot
            if file[-4:] == ".txt": # only looking at txt files
                justDigits = '' 
                contents = GET_STRING_FROM_FILE(fullpath)
                for x in contents:
                    if x in string.digits:
                        justDigits = justDigits + x
                # print(justDigits)
                # print(justDigits[0:3])
                if len(justDigits) == 10 and justDigits[0:3] == '909':
                    area909_count += 1    
                
    return area909_count

if True:
    """ overall script that runs examples """

    path1 = "./intro_first"
    result1 = file_walker(path1)

    path2 = "./intro_second"
    result2 = file_walker(path2)

    path3 = "./files_challenge"
    result3 = file_walker(path3)

    print(f"For {path3}:")
    print(f"909 count = {result3}")  

    print()
    print(f"For {path2}:")
    print(f"909 count = {result2}") 

    print()
    print(f"For {path1}:")
    print(f"909 count = {result1}") 



# How many people have 3 i's/I's somewhere in their name?
# What is the maximum number of Is in someone's name?


def file_walker(path):
    """ starting from the input, named path
        this function "walks" the whole path, including subfolders and counts the number of I/i's in every .txt file. 
        it counts the total instances where there are exactly 3 I/i's, and returns that integer count. It also keeps track of the maximum number of Is found in 
        any file in a folder and returns that maximum value.
    """
    result = list(os.walk(path))    
    threeI_count = 0
    maxIs = 0

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple
        if '__MACOSX' in currentpath:  continue

        for file in files:      
            fullpath = currentpath + "/" + file           
            if file[-4:] == ".txt":
                iCount = 0
                contents = GET_STRING_FROM_FILE(fullpath)
                # print(contents)
                line_count = len(contents.splitlines())
                # print(line_count)
                if line_count < 5: # estimate of a cut off for where a .txt file is not a contact file
                    for x in contents:
                        if x in 'Ii':
                            iCount += 1
                    if iCount == 3:
                        threeI_count += 1
                    if iCount > maxIs:
                        maxIs = iCount    
                
    return threeI_count, maxIs

if True:
    """ overall script that runs examples """

    path1 = "./intro_first"
    result1 = file_walker(path1)

    path2 = "./intro_second"
    result2 = file_walker(path2)

    path3 = "./files_challenge"
    result3 = file_walker(path3)

    print(f"For {path3}:")
    print(f"3 Ii count = {result3[0]}")
    print(f"max Ii count in {path3}: {result3[1]}")  

    
    print()
    print(f"For {path2}:")
    print(f"3 Ii count = {result2[0]}")
    print(f"max Ii count in {path2}: {result2[1]}") 

    print()
    print(f"For {path1}:")
    print(f"3 Ii count = {result1[0]}") 
    print(f"max Ii count in {path1}: {result1[1]}")

    overallMax = max(result1[1], result2[1], result3[1])
    print()
    print(f"Overall max Ii count of all folders: {overallMax}")





# How many people have the string 'Khaby' somewhere in their name, case insensitive?

def file_walker(path):
    """ starting from the input, named path
        this function "walks" the whole path, including subfolders and counts the number of files that contain the string 'Khaby' (case insensitive), 
        and does the same for files with 'D'Amelio. 
        It returns those count values.
    """
    result = list(os.walk(path))    
    khabyCount = 0
    damCount = 0 # using this total in next part, not relevant for this block

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple
        if '__MACOSX' in currentpath:  continue

        for file in files:      
            fullpath = currentpath + "/" + file           
            if file[-4:] == ".txt":
                justLetters = ''
                contents = GET_STRING_FROM_FILE(fullpath)
                line_count = len(contents.splitlines())
                if line_count < 5: # estimate of a cut off for where a .txt file is not a contact file
                    for x in contents:
                        if x in string.ascii_letters or x == "'":
                            justLetters = justLetters + x
                    if 'khaby' in justLetters.lower(): # lowercase contents so that cases don't matter
                        khabyCount += 1
                    if "d'amelio" in justLetters.lower(): # lowercase contents so that cases don't matter
                        damCount += 1    
                
    return khabyCount, damCount

if True:
    """ overall script that runs examples """

    path1 = "./intro_first"
    result1 = file_walker(path1)

    path2 = "./intro_second"
    result2 = file_walker(path2)

    path3 = "./files_challenge"
    result3 = file_walker(path3)

    print(f"For {path3}:")
    print(f"Khaby count = {result3[0]}")
    
    print()
    print(f"For {path2}:")
    print(f"Khaby count = {result2[0]}")
 
    print()
    print(f"For {path1}:")
    print(f"Khaby count = {result1[0]}") 


    print()
    print(f"For next block, will use D'Amelio total from this also: D'Amelio count = {result3[1]}")



# How many people have the last name Khaby or D'Amelio (case insensitive)? 

import os
import string

def file_walker(path):
    """ starting from the input, named path
        this function "walks" the whole path, including subfolders and counts the number of files that contain the string 'Khaby' or 'D'amelio (case insensitive) as a last name, 
        i.e. if Khaby or D'Amelio is the second word with no comma present, or the first word if the words (names) are separated by a comma.
        It returns those count values.
    """
    result = list(os.walk(path))    
    khabyCount_L = 0 # L for last
    damCount_L = 0 
    khabyCount_F = 0 # F for first
    damCount_F = 0

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple
        if '__MACOSX' in currentpath:  continue

        for file in files:      
            fullpath = currentpath + "/" + file           
            if file[-4:] == ".txt":
                cleanedVersion = ''
                contents = GET_STRING_FROM_FILE(fullpath)
                line_count = len(contents.splitlines())
                if line_count < 5: # estimate of a cut off for where a .txt file is not a contact file
                    if ',' in contents: # the case where the contact format is Last, First
                        for x in contents:
                            if x in string.ascii_letters or x == "'":
                                cleanedVersion = cleanedVersion + x
                        cleanedVersion = cleanedVersion.lower() # lowercase contents so that cases don't matter
                        if 'khaby' in cleanedVersion[0:5]: 
                            khabyCount_L += 1
                        if "d'amelio" in cleanedVersion[0:8]:
                            damCount_L += 1


                    else: # case where format is First Last
                        for x in contents:
                            if x in string.ascii_letters or x == ' ' or x == "'": # keep letters, apostrophes, and spaces to keep names separate
                                cleanedVersion = cleanedVersion + x
                        cleanedVersion = cleanedVersion.lower() # lowercase contents so that cases don't matter
                        words = cleanedVersion.split()
                        if len(words) > 1 and 'khaby' == words[1]: 
                            khabyCount_L += 1
                        if len(words) > 1 and "d'amelio" == words[1]: 
                            damCount_L += 1

    khabyCount_F = 27 - khabyCount_L # total Khaby's minus Khaby last names = Khaby first names (total count computed in preceding block)
    damCount_F = 53 - damCount_L # same as above line

    return khabyCount_L, damCount_L, khabyCount_F, damCount_F

if True:
    """ overall script that runs examples """

    path1 = "./intro_first"
    result1 = file_walker(path1)

    path2 = "./intro_second"
    result2 = file_walker(path2)

    path3 = "./files_challenge"
    result3 = file_walker(path3)

    print(f"For {path3}:")
    print(f"Khaby last name count = {result3[0]}, Khaby first  name count = {result3[2]}, D'Amelio last name count = {result3[1]}, D'Amelio last name count = {result3[3]}")
    
    print()
    print(f"For {path2}:")
    print(f"Khaby count = {result2[0]}, D'Amelio count = {result2[1]} for both first and last names")
 
    print()
    print(f"For {path1}:")
    print(f"Khaby count = {result1[0]}, D'Amelio count = {result1[1]} for both first and last names") 




# How many people have your last name (CS35_Participant_12)?    
# Choose another last name present (Moore):  How many people have that last name?
# How many people have your first name (CS35_Participant_12)? 
# Choose another first name present (Lucas): How many people have that first name?

def file_walker(path):
    """ starting from the input, named path
        this function "walks" the whole path, including subfolders and counts the number of files that contain the string 'Khaby' or 'D'amelio (case insensitive) as a last name, 
        i.e. if Khaby or D'Amelio is the second word with no comma present, or the first word if the words (names) are separated by a comma.
        It returns those count values.
    """
    result = list(os.walk(path))    
    CS35_Participant_12 = 0
    moore = 0
    CS35_Participant_12 = 0
    lucas = 0

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple
        if '__MACOSX' in currentpath:  continue

        for file in files:      
            fullpath = currentpath + "/" + file           
            if file[-4:] == ".txt":
                cleanedVersion = ''
                contents = GET_STRING_FROM_FILE(fullpath)
                line_count = len(contents.splitlines())
                if line_count < 5: # estimate of a cut off for where a .txt file is not a contact file
                    if ',' in contents: # the case where the contact format is Last, First
                        for x in contents:
                            if x in string.ascii_letters:
                                cleanedVersion = cleanedVersion + x
                        cleanedVersion = cleanedVersion.lower() # lowercase contents so that cases don't matter
                        if 'CS35_Participant_12' in cleanedVersion[0:6]: 
                            CS35_Participant_12 += 1
                        if "moore" in cleanedVersion[0:5]:
                            moore += 1
                        words = cleanedVersion.split()
                        if len(words) > 1 and 'CS35_Participant_12' == words[1]: 
                            CS35_Participant_12 += 1
                        if len(words) > 1 and "lucas" == words[1]: 
                            lucas += 1

                    else: # case where format is First Last
                        for x in contents:
                            if x in string.ascii_letters or x == ' ': # keep letters and spaces to keep names separate
                                cleanedVersion = cleanedVersion + x
                        cleanedVersion = cleanedVersion.lower() # lowercase contents so that cases don't matter
                        words = cleanedVersion.split()
                        if len(words) > 1 and 'CS35_Participant_12' == words[1]: 
                            CS35_Participant_12 += 1
                        if len(words) > 1 and "moore" == words[1]: 
                            moore += 1
                        if 'CS35_Participant_12' in cleanedVersion[0:5]: 
                            CS35_Participant_12 += 1
                        if "lucas" in cleanedVersion[0:5]:
                            lucas += 1


    return CS35_Participant_12, moore, CS35_Participant_12, lucas


if True:
    """ overall script that runs examples """

    path1 = "./intro_first"
    result1 = file_walker(path1)

    path2 = "./intro_second"
    result2 = file_walker(path2)

    path3 = "./files_challenge"
    result3 = file_walker(path3)


print(f"For {path3}:")
print(f"CS35_Participant_12 (last) count = {result3[0]}, Moore (last) count = {result3[1]}, CS35_Participant_12 (first) count = {result3[2]}, Lucas (first) count = {result3[3]}")
    
print()
print(f"For {path2}:")
print(f"CS35_Participant_12 (last) count = {result2[0]}, Moore (last) count = {result2[1]}, CS35_Participant_12 (first) count = {result2[2]}, Lucas (first) count = {result2[3]}")
 
print()
print(f"For {path1}:")
print(f"CS35_Participant_12 (last) count = {result1[0]}, Moore (last) count = {result1[1]}, CS35_Participant_12 (first) count = {result1[2]}, Lucas (first) count = {result1[3]}")



# 3 more questions (for files in files_challenge):

# 1. Are there more NE or SW phone numbers across the whole dataset (meaning, area codes beginning with "2" or "9" respectively)?
# 2. What is the earliest name alphabetically?
# 3. What is the lowest and highest number of letters in someone's name?

def file_walker(path):
    """ starting from the input, named path
        this function "walks" the whole path, including subfolders and
        1. counts the area codes starting with 2, and those starting with 9, and returns those counts
        2. returns the earliest alphabetical name, alphabetized by last name
        3. couunts the min and max number of letters in each name and returns the highest and lowest found letters counts
    """
    result = list(os.walk(path))    
    NEcount = 0
    SWcount = 0
    minCount = 10000000000
    maxCount = 0
    alphabetFirst = 'zzzzzzzzzzzzz'
    longest_name = ''
    shortest_name = ''
    name = ''

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple
        if '__MACOSX' in currentpath:  continue

        for file in files:      
            fullpath = currentpath + "/" + file           
            if file[-4:] == ".txt":
                digitVersion = ''
                letterVersion = ''
                contents = GET_STRING_FROM_FILE(fullpath)
                line_count = len(contents.splitlines())
                if line_count < 5: # estimate of a cut off for where a .txt file is not a contact file
                    
                    # start w/ NE vs SW counting
                    for x in contents:
                        if x in string.digits:
                            digitVersion = digitVersion + x
                            # print(digitVersion)
                    if digitVersion[0:1] == '9': 
                        NEcount += 1
                    if digitVersion[0:1] == '2': 
                        SWcount += 1

                    # then max/min letter counting
                    for x in contents:
                        if x in string.ascii_letters or x in ' ':
                            letterVersion = letterVersion + x
                        words = letterVersion.split() 
                        for word in words:
                            if len(word) > maxCount:
                                maxCount = len(word)
                                longest_name = word
                            if len(word) < minCount and len(word) > 1: #word needs to be a name
                                minCount = len(word)
                                shortest_name = word

                    # then finding first alphabetical name accounting for contact file formats
                    if ',' in contents: # the case where the contact format is Last, First
                        for x in contents:
                            if x in string.ascii_letters:
                                letterVersion = letterVersion + x
                            # letterVersion = letterVersion.lower() 
                            words = letterVersion.split()
                            if len(words) > 1 and words[1] < alphabetFirst:
                                alphabetFirst = words[1]
                                name = contents

                    else: # case where format is First Last
                        for x in contents:
                            if x in string.ascii_letters or x == ' ': # keep letters and spaces to keep names separate
                                letterVersion = letterVersion + x
                            # letterVersion = letterVersion.lower() 
                            words = letterVersion.split()
                            if len(words) > 1 and words[0] < alphabetFirst:
                                alphabetFirst = words[0]
                        


    return maxCount, minCount, NEcount, SWcount, alphabetFirst, longest_name, shortest_name, name


if True:
    """ overall script that runs examples """

    path3 = "./files_challenge"
    result3 = file_walker(path3)


    print(f"The minimum letter count in a name is {result3[1]} for {result3[6]}, the maximum count is {result3[0]} for {result3[5]}.")

    print(f"There are {result3[2]} NE area codes and {result3[3]} SW area codes, so there are more SW area codes.")

    print(f"The alphabetically earliest name is {result3[4]}.")








# Moving to my own folders:

# how many files are there total?  (Choose somewhere with at least 42 files.)
# how many files of a particular type are there, in total...
# how deep was the deepest path present?  i.e.,  What's the folder's "nesting depth"

import os
import string

def file_walker(path):
    """ walks all files in an input path to count total file number, find the maximum depth of a directory, and count the total MATLAB files.
    returns counts of the above
    """
    result = list(os.walk(path))
    total_files = 0
    matlab_files = 0
    max_depth = 0

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple
        if '__MACOSX' in currentpath: continue

        # Calculate the depth based on the path
        depth = currentpath.count(os.path.sep) - path.count(os.path.sep) + 1
        if depth > max_depth:
            max_depth = depth

        for file in files:
            fullpath = os.path.join(currentpath, file)
            if file.startswith("."): continue  # skip files that start with dot
            total_files += 1  # counting towards file total
            if file.endswith(".m"):
                matlab_files += 1

    return max_depth, total_files, matlab_files

# Usage example
path = "./Spring 2023"
max_depth, total_files, matlab_files = file_walker(path)

print(f"For my {path} folder:")
print()
print(f"Maximum directory depth: {max_depth}")
print(f"Total number of files: {total_files}")
print(f"Total number of MATLAB files: {matlab_files}")


    


# 3 more questions on Spring 2023 folder:

# 1. What is the most common file type I have and how many of this file type are there?
# 2. What is the largest file size here?
# 3. Are there any duplicate file names and if so how many?

import os

def file_walker(path):
    """Walks all files in an input path to find the three most common file types,
    the size of the largest file, the count of duplicate file names, and the most duplicated file name.
    
    Returns:
    common_file_types (list of tuples): A list of tuples with the three most common file types and their counts.
    largest_file_size (int): The size of the largest file in bytes.
    duplicate_count (int): The number of duplicate file names found.
    most_duplicated_file (str): The name of the most duplicated file.
    """
    
    file_type_counts = {}
    largest_file_size = 0
    file_names = {}
    duplicate_count = 0
    most_duplicated_file = ""

    for currentpath, subfolders, files in os.walk(path):
        if '__MACOSX' in currentpath: continue

        for file in files:
            if file.startswith("."): continue  # skip files that start with dot

            # File extension
            file_extension = os.path.splitext(file)[1]
            if file_extension in file_type_counts:
                file_type_counts[file_extension] += 1
            else:
                file_type_counts[file_extension] = 1

            # File size
            fullpath = os.path.join(currentpath, file)
            file_size = os.path.getsize(fullpath)
            if file_size > largest_file_size:
                largest_file_size = file_size

            # Duplicate file name check
            if file in file_names:
                file_names[file] += 1
            else:
                file_names[file] = 1

    # Count duplicates and find the most duplicated file
    max_duplicate_count = 0
    for file_name, count in file_names.items():
        if count > 1:
            duplicate_count += (count - 1)
            if count > max_duplicate_count:
                max_duplicate_count = count
                most_duplicated_file = file_name

    # Find the three most common file types
    sorted_file_types = sorted(file_type_counts.items(), key=lambda item: item[1], reverse=True)
    common_file_types = sorted_file_types[:3]

    return common_file_types, largest_file_size, duplicate_count, most_duplicated_file

# Usage example
path = './Spring 2023'
common_file_types, largest_file_size, duplicate_count, most_duplicated_file = file_walker(path)
print(f"Three most common file types: {common_file_types}")
print(f"Largest file size: {largest_file_size} bytes")
print(f"Number of duplicate file names: {duplicate_count}")
print(f"Most duplicated file name: {most_duplicated_file}")








