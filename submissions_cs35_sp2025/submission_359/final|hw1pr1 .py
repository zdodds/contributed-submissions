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

#Mac:    !cat  <filepath>  using forward slashes
#
# !cat ./intro_first/nottrue.ipynb       

# Windows:  type <filepath>  using backslashes
#
# !type .\\intro_first\\nottrue.ipynb       


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

    path = "/Users/isabelburger"       # Remember: . means the current directory
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
        if '__MACOSX' in currentpath: continue         # skip the rest of _this_ loop iteration: back to top
        print(f"{currentpath = }")   # try non-f printing: it's worse!

    num_folders = len(result)        # the len is the number of currentpaths...
    return num_folders

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    print(f"[[ Start! ]]\n")     # sign on

    path = "./files_challenge"       # Remember: . means the current directory
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
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
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

    path = "./files_challenge"       # Remember: . means the current directory
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


#
# How many .txt files are in the entire files_challenge folder?
#

import os
import os.path

def txt_counter(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and counts the total number of .txt files
    """
    result = list(os.walk(path)) 
    
    count_txt = 0    # keep count of our .txt files

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue

        for file in files:       # remember, files is a list of filenames!
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            if file[0] == ".": continue      # skip files that start with dot
            if file[-4:] == ".txt": 
                count_txt += 1  # Found a .txt file! Adding one...

    return count_txt   

#
# when discovering, keep your data close (and your functions closer!)
#
if True:

    path = "./files_challenge"       # Remember: . means the current directory
    result = txt_counter(path)   

    print(f"num txt files = {result}")  




#
# What is the maximum depth of directories in the entire folder?
#

import os
import os.path

def count(string, e):
    count = 0
    for x in string:
        if x == e:
            count += 1
    return count 

def max_depth(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then finds the maximum depth 
    """
    result = list(os.walk(path))     
    
    maxDepth = 1    # keep count of the maximum depth

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue
        if (count(currentpath, '/') + 1) > maxDepth:
            maxDepth = count(currentpath, '/')+1
            #print(currentpath)
    return maxDepth

if True:
    path = "./files_challenge"       # Remember: . means the current directory
    result = max_depth(path)   # Run!
    print(f"The maximum depth of directories in the folder {path} is {result}")  


def getNum(contents):
    number = ''
    for i in range(len(contents)):
        if contents[i] in '0123456789':
            number += contents[i]
    return number 



#
# Across all of the files, how many of the phone numbers contain exactly 10 digits? 7 digits? 11 digits? 
#


import os
import os.path

def countDigits(path, num):
    """ starting from the input, named path and the given number of digits num 
        
        this function "walks" the whole path, including subfolders
        and counts the number of phone numbers with that number of digits

    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...

    count = 0 # keep track of the number of phone numbers of length num 

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue

        for file in files:   
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[-4:] != ".txt": 
                continue
            if file[0] == ".": continue      # skip files that start with dot 
            fullpath = currentpath + "/" + file          
            if file[0] == ".": continue      # skip files that start with dot
            contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            if (len(getNum(contents))) == num: 
                count += 1

    return count 

if True:
    
    path = "./files_challenge"       # Remember: . means the current directory
    resultSev = countDigits(path, 7)   
    resultElev = countDigits(path, 11)
    resultTen = countDigits(path, 10)

    print(f"num phone numbers of length 7 = {resultSev}")  
    print(f"num phone numbers of length 11 = {resultElev}")
    print(f"num phone numbers of length 10 = {resultTen}")



#
# How many of the 10 digit phone numbers have the 909 area code?  
#


import os
import os.path

def countAreaCode(path, s):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then... explores any questions we might want :)

        call, for example, with    file_walker("./intro_first") 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...

    countAC = 0    # keep count of phone numbers with a specific area code

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[-4:] != ".txt": 
                continue
            if file[0] == ".": continue      # skip files that start with dot
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot
            contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            phoneNum = getNum(contents)
            if len(phoneNum) == 10:
                areaCode = phoneNum[0:3]
                if areaCode == s:
                    countAC += 1

    return countAC 
#
# when discovering, keep your data close (and your functions closer!)
#
if True:

    path = "./files_challenge"       # Remember: . means the current directory
    ac = '909'
    result = countAreaCode(path, ac)   # Run!

    print(f"num phone numbers with area code {ac} = {result}")  # Yay for f-strings!



import string

def getFirstName(text):
    #
    # This function takes a text file, assuming the structure that it is two lines, with the name on the second line, and splits it into the name
    # portion, trims the edges of any whitespace, and locates and returns the first name
    #

    fName = ''
    comma = False
    commaLoc = 0
    spaceLoc = 0 
    text = text.split('\n')[1]
    text = text.strip()
    for i in range(len(text)):
                if text[i] == ',':
                    commaLoc = i
                    comma = True
                elif text[i] == ' ':
                     spaceLoc = i
    if comma == False:
        for k in range(0, spaceLoc):
            if text[k] == ' ':
                break 
            fName += text[k]
    else:
        for l in range(commaLoc+1, len(text)):
            if text[l] != ' ':
                fName += text[l]
    fName = fName.lower()
    return fName



import string

def getLastName(text):
    #
    # This function takes a text file, assuming the structure that it is two lines, with the name on the second line, and splits it into the name
    # portion, trims the edges of any whitespace, and locates and returns the last name
    #

    lName = ''
    comma = False
    commaLoc = 0
    spaceLoc = 0
    text = text.split('\n')[1]
    text = text.strip()
    for i in range(len(text)):
                if text[i] == ',':
                    commaLoc = i
                    comma = True
                elif text[i] == ' ':
                     spaceLoc = i
    if comma == True:
        for i in range(0, commaLoc): 
            if text[i] != ' ':
                lName += text[i]
    else:
        for i in range(spaceLoc, len(text)):
            if text[i] != ' ':
                lName += text[i] 
    lName = lName.lower()
    return lName



#
# How many people have the first name Khaby?  D'Amelio?
#


import os
import os.path

def countFirstName(path, name):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then counts the number of files that have name as the first name 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    name = name.lower()
    countName = 0    # keep count of phone numbers with a specific area code

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[-4:] != ".txt": 
                continue
            if file[0] == ".": continue      # skip files that start with dot
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot
            contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            firstName = getFirstName(contents)
            if firstName == name:
                countName += 1
                #print(contents)

    return countName 
#
# when discovering, keep your data close (and your functions closer!)
#
if True:

    path = "./files_challenge"       # Remember: . means the current directory
    ln = 'Khaby'
    result = countFirstName(path, ln)   # Run!
    print(f"The number of people with the first name {ln} in {path} is {result}.")  # Yay for f-strings!
    path = "./files_challenge"       # Remember: . means the current directory
    ln = 'D\'Amelio'
    result = countFirstName(path, ln)   # Run!
    print(f"The number of people with the first name {ln} in {path} is {result}.")  # Yay for f-strings
    



#
# How many people have the last name Khaby?  D'Amelio?
#


import os
import os.path

def countLastName(path, name):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then counts the number of files that have name as the first name 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    name = name.lower()
    countName = 0    # keep count of phone numbers with a specific area code

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[-4:] != ".txt": 
                continue
            if file[0] == ".": continue      # skip files that start with dot
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            lastName = getLastName(contents)
            if lastName == name:
                countName += 1
                #print(contents)

    return countName 
#
# when discovering, keep your data close (and your functions closer!)
#
if True:

    path = "./files_challenge"       # Remember: . means the current directory
    ln = 'Khaby'
    result = countLastName(path, ln)   # Run!
    print(f"The number of people with the last name {ln} in {path} is {result}!")  # Yay for f-strings!
    #path = "./files_challenge"       # Remember: . means the current directory
    ln = 'D\'Amelio'
    result = countLastName(path, ln)   # Run!
    print(f"The number of people with the last name {ln} in {path} is {result}")  # Yay for f-strings
    


#
# How many people have the name Khaby or D'Amelio in their first or last name?
#

def countName(path, name):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then counts the number of files that have name as the first or last name
    """
    fn = countFirstName(path, name)
    ln = countLastName (path, name)
    sum = fn + ln
    return sum 

if True:

    path = "./files_challenge"       # Remember: . means the current directory
    name = 'Khaby'
    result = countName(path, name)   # Run!
    print(f"The number of people with the name {name} in {path} is {result}!")  # Yay for f-strings!

    path = "./files_challenge"       # Remember: . means the current directory
    name = 'D\'Amelio'
    result = countName(path, name)   # Run!
    print(f"The number of people with the name {name} in {path} is {result}!")  # Yay for f-strings!


def countChar(path, e):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then finds the file contents that have the most characters 'e' in the name 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    maxChar = 0    # keep count of the max number of characters in a name 

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[-4:] != ".txt": 
                continue
            if file[0] == ".": continue      # skip files that start with dot
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            contents = contents.split('\n')
            if len(contents) > 4:
                continue
            if len(contents) >= 2: 
                contents = contents[1]
            else:
                contents = contents[0]
            contents = contents.strip()
            lc = contents.count(e.lower())
            uc = contents.count(e.upper())
            sum = lc + uc
            if sum >= maxChar:
                maxChar = sum
                maxName = contents
                #print(contents)
    return [maxName, maxChar]

if True:

    path = "./intro_first"       # Remember: . means the current directory
    char = 'i'
    result = countChar(path, char)   # Run!
    print(f"The name with the maximum number of \'{char}\' characters in {path} is: {result[0]} with {result[1]}!")  # Yay for f-strings!

    path = "./intro_second"       # Remember: . means the current directory
    char = 'i'
    result = countChar(path, char)   # Run!
    print(f"The name with the maximum number of \'{char}\' characters in {path} is: {result[0]} with {result[1]}!")  # Yay for f-strings!

    path = "./files_challenge"       # Remember: . means the current directory
    char = 'i'
    result = countChar(path, char)   # Run!
    print(f"The name with the maximum number of \'{char}\' characters in {path} is: {result[0]} with {result[1]}!")  # Yay for f-strings!



def numChar(path, e, N):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then finds the file contents that have N number of character 'e' in the name 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    nameList = []   # keep count of the strings that contain N number of 'e's 

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[-4:] != ".txt": 
                continue
            if file[0] == ".": continue      # skip files that start with dot
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            contents = contents.split('\n')
            if len(contents) > 4:
                continue
            if len(contents) >= 2: 
                contents = contents[1]
            else:
                contents = contents[0]
            
            contents = contents.strip()
            lc = contents.count(e.lower())
            uc = contents.count(e.upper())
            sum = lc + uc
            if sum == N:
                nameList += [contents]
    return [len(nameList), nameList] 


if True:
    path = "./intro_first"       # Remember: . means the current directory
    char = 'i'
    num = 3
    result = numChar(path, char, num)   # Run!
    print(f"There are {result[0]} names in {path} with {num} \'{char}\' characters: {result[1]}!")  # Yay for f-strings!

    path = "./intro_second"       # Remember: . means the current directory
    char = 'i'
    num = 3
    result = numChar(path, char, num)   # Run!
    print(f"There are {result[0]} names in {path} with {num} \'{char}\' characters: {result[1]}!")  # Yay for f-strings!

    path = "./files_challenge"       # Remember: . means the current directory
    char = 'i'
    num = 3
    result = numChar(path, char, num)   # Run!
    print(f"There are {result[0]} names in {path} with {num} \'{char}\' characters: {result[1]}!")  # Yay for f-strings!




if True:

    path = "./files_challenge"       # Remember: . means the current directory
    ln = 'CS35_Participant_3'
    result1 = countLastName(path, ln)   # Run!
    print(f"num people with last name {ln} = {result1}")  # Yay for f-strings!
    #path = "./files_challenge"       # Remember: . means the current directory
    fn = 'CS35_Participant_3'
    result2 = countFirstName(path, fn)   # Run!
    print(f"num people with first name {fn} = {result2}")  # Yay for f-strings


if True:

    path = "./files_challenge"       # Remember: . means the current directory
    ln = 'Julien'
    result1 = countLastName(path, ln)   # Run!
    print(f"num people with last name {ln} = {result1}")  # Yay for f-strings!
    #path = "./files_challenge"       # Remember: . means the current directory
    fn = 'CS35_Participant_9'
    result2 = countFirstName(path, fn)   # Run!
    print(f"num people with first name {fn} = {result2}")  # Yay for f-strings


#
# How many of the phone numbers contain the substring '42'?  
#


import os
import os.path

def subString_phoneNum(path, s):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and then counts the number of phone numbers containing the substring s 

    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...

    countSS = 0    # keep count of phone numbers with the substring s

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[-4:] != ".txt": 
                continue
            if file[0] == ".": continue      # skip files that start with dot
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot
            contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            phoneNum = getNum(contents)
            if s in phoneNum:
                countSS += 1

    return countSS 
#
# when discovering, keep your data close (and your functions closer!)
#
if True:

    path = "./files_challenge"       # Remember: . means the current directory
    substring = '42'
    result = subString_phoneNum(path, substring)   # Run!

    print(f"The number of phone numbers containing the substring {substring} in {path} is {result}")  # Yay for f-strings!



#
# How many of the phone numbers are 'symmetrical' and what are they?
#
import os
import os.path

def symmetrical(s):
    if len(s) < 2:
        #print('short!')
        return True
    elif s[0] == s[-1]:
        #print(s[1:len(s)-1])
        return symmetrical(s[1:len(s)-1])
    else: 
        return False

def symmetrical_phoneNums(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and counts and returns all phone numbers that are 'symmetrical'

    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    countSym = 0    # keep count of phone numbers that are symmetrical
    listSym = [] # keep count of list of phone numbers that are symmetrical 

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[-4:] != ".txt": 
                continue
            if file[0] == ".": continue      # skip files that start with dot
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot
            contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            phoneNum = getNum(contents)
            if phoneNum != '' and symmetrical(phoneNum) == True:
                countSym += 1
                listSym += [phoneNum]
                #print(f"   {fullpath = }")

    return [countSym, listSym]
#
# when discovering, keep your data close (and your functions closer!)
#
if True:

    path = "./files_challenge"       # Remember: . means the current directory
    result = symmetrical_phoneNums(path)   # Run!

    print(f"The number of phone numbers that are symmetrical in {path} is {result[0]}: the numbers {result[1][0]} and {result[1][1]}")  # Yay for f-strings!





#
# What are the most common last name in the dataset? 
#


import os
import os.path


def mostCommonLastName(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and counts and returns the last name with the most instances

    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list...
    lnameDict = {}

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[-4:] != ".txt": 
                continue
            if file[0] == ".": continue      # skip files that start with dot
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            if file[0] == ".": continue      # skip files that start with dot
            contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            lastName = getLastName(contents)
            if lastName != '':
                if lastName in lnameDict:
                      lnameDict[lastName] += 1
                else:
                    lnameDict[lastName] = 1
    maxName = max(lnameDict, key = lnameDict.get)
    maxNum = lnameDict[maxName]
    return [maxName, maxNum]
if True:

    path = "./files_challenge"       # Remember: . means the current directory
    result = mostCommonLastName(path)   # Run!

    print(f"The most common name in {path} is {result[0]} with {result[1]} instances.")  # Yay for f-strings!




#
# File counter counts all the files contained in  a given path
#

import os
import os.path

def file_counter(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and counts the total number of files
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list..
    fileCount = 0

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[0] == ".": continue      # skip files that start with dot
            fileCount += 1
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            #contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            #print(f"   {contents[0:42] = }")

    return fileCount

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    """ overall script that runs examples """
    path = "/Users/isabelburger/Documents/Previous Classes"       # Remember: . means the current directory
    result = file_counter(path)   # Run!

    print(f"There are {result} files in {path}")  # Yay for f-strings!




#
# File counter counts all the files contained in  a given path
#

import os
import os.path

def filetype_counter(path, filetype):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and counts the total number of files of type filetype
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list..
    fileCount = 0

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            
            if file[-(len(filetype)):] != filetype: 
                continue
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[0] == ".": continue      # skip files that start with dot
            fileCount += 1
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            #contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            #print(f"   {contents[0:42] = }")

    return fileCount

#
# when discovering, keep your data close (and your functions closer!)
#
if True:
    path = "/Users/isabelburger/Documents/Previous Classes" # Remember: . means the current directory
    filetype = '.ino'
    result = filetype_counter(path, filetype)   # Run!

    print(f"There are {result} files of type {filetype} in {path}")  # Yay for f-strings!



if True:
    path = "/Users/isabelburger/Documents/Previous Classes" 
    result = max_depth(path)   # Run!
    print(f"The maximum depth of directories in the folder {path} is {result}")  


#
# Most common file extension present in the given path
#

import os
import os.path

def filetype_max(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and creates a dictionary of each file type and the corresponding number of files 
    """
    result = list(os.walk(path))     # perhaps try w/o converting to a list..
    fileTypes = {}

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  # always three items, always these...
        if '__MACOSX' in currentpath:  continue
        #print(f"{currentpath = }") 

        for file in files:       # remember, files is a list of filenames!
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if file[0] == ".": continue      # skip files that start with dot
            if '.' not in file:
                continue
            else:
                for i in range(len(file)):
                    if file[i] == '.':
                        dotLoc = i
                filetype = file[dotLoc:]
            if filetype in fileTypes:
                 fileTypes[filetype] += 1
            else:
                 fileTypes[filetype] = 1
            
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            #print(f"   {fullpath = }")
            #contents = GET_STRING_FROM_FILE(fullpath)     # use the fullpath!
            #print(f"   {contents[0:42] = }")
        maxType = max(fileTypes, key = fileTypes.get, default = 0)
        if maxType != 0: 
            maxNum = fileTypes[maxType]
    return [maxType, maxNum, fileTypes]


if True:
    path = "/Users/isabelburger/Documents/Previous Classes" # Remember: . means the current directory
    result = filetype_max(path)   # Run!
    print(f"The most common file type is {result[0]} with {result[1]} total files in {path}")  # Yay for f-strings!
    print(f"The breakdown of file types is as follows: {result[2]}")  # Yay for f-strings!




#
# How many files contain my name in the file name? 
#

import os
import os.path

def name_counter(path, name):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and counts the total number of .txt files
    """
    result = list(os.walk(path)) 
    fileList = []
    
    nameCount = 0    # keep count of how many files contain the string 'name' (case insensitive)

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple  
        if '__MACOSX' in currentpath:  continue

        for file in files:       # remember, files is a list of filenames!
            fullpath = currentpath + "/" + file           # construct the full path, or, better: os.path.join(currentpath,file)  
            if file[0] == ".": continue      # skip files that start with dot
            if '.DS_Store' in file: continue         # skip the rest of _this_ loop iteration: back to top
            if name.lower() in file.lower():
                nameCount += 1
                fileList += [file]

    return [nameCount, fileList]  

if True:
    path = "/Users/isabelburger/Documents/Previous Classes" # Remember: . means the current directory
    name = 'CS35_Participant_3'
    result = name_counter(path, name)   # Run!
    print(f"The name {name} appears in file names {result[0]} number of times in {path} in the files: {result[1]}")  # Yay for f-strings!

    path = "/Users/isabelburger/Documents/Previous Classes" # Remember: . means the current directory
    name = 'CS35_Participant_3'
    result = name_counter(path, name)   # Run!
    print(f"The name {name} appears in file names {result[0]} number of times in {path} in the files: {result[1]}")  # Yay for f-strings!



#

import os
import os.path

def fullest_folder(path):
    """ starting from the input, named path
        
        this function "walks" the whole path, including subfolders
        and finds the folder with the most number of files and folders within it 
    """
    result = list(os.walk(path)) 
    
    maxContents = 0    # keep count of how many files contain the string 'name' (case insensitive)

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple
        countContents = len(subfolders) + len(files)
        if '__MACOSX' in currentpath:  continue
        if countContents >= maxContents:
            maxContents = countContents
            max = currentpath  

    return [max, maxContents]

if True:
    path = "/Users/isabelburger/Documents/Previous Classes" # Remember: . means the current directory
    result = fullest_folder(path)   # Run!
    print(f"The fullest folder in is {result[0]} with {result[1]} files or subfolders directly within it.")  # Yay for f-strings!



