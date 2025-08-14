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


# counting the txt files in starting_notebook

import os
import os.path

def count_txt(path):
    '''counting the number of txt files in a folder'''
    result = list(os.walk(path))
    count = 0
    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple 
        for file in files:
            if file.endswith(".txt"):
                count += 1
    return count

print('intro first = ', count_txt("./intro_first"))
print('intro second = ', count_txt("./intro_second"))
print('files_challenge =', count_txt("./files_challenge"))


       
            
       


 


# The maximum depth of directories in the entire folder
import os
import os.path

def count_maxdepth(path):
    '''counting the maximum depth of the directoris among the whole folder'''
    result = os.walk(path)
    depthlist = []

    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple
        countdepth = currentpath.count('/') + 1
        depthlist.append(countdepth)

    return max(depthlist) if depthlist else 0

print('intro_first =', count_maxdepth("./intro_first"))
print('intro_second =', count_maxdepth("./intro_second"))
print('files_challenge =', count_maxdepth("./files_challenge"))


        


# Counting the number of phone numbers containing exactly 10 digits/ 7 digits/ 11 digits

import os
import os.path


def count_digit(path,digit):  #here, the digit is the number of digit you search for
    '''counting the number of phone numbers in the folder that contains the number of digits you desire'''

    counts = []
    for currentpath, subfolders, files in os.walk(path):
        for file in files:                                      #count the number of digits for the phone number in each file
            if file.endswith(".txt"):
                file_path = os.path.join(currentpath,file)
                try:
                    with open(file_path,'r', encoding='utf-8') as f:
                        contents = f.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                # Count the digits in the file
                digit_count = sum(1 for char in contents if char.isdigit())
                counts.append(digit_count)
    count_desired = sum(1 for count in counts if count == digit)
    return count_desired



print('intro_first = ', count_digit(",/intro_first", 10))
print('intro_second =', count_digit("./intro_second", 10))
print('files_challenge = ',count_digit("./files_challenge", 10))       
        
            

    







# Of the exactly-ten-digit phone numbers, how many are in the area code 909 
import os
import os.path

def count_909(path):
    '''counting how many files, with exactly-ten-digit, are there in a folder contain area code 909'''
    count = 0
    digit_list = []
    for currentpath, subfolders, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(currentpath, file)  
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        contents = f.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                digits = ''.join(char for char in contents if char.isdigit())
                if len(digits) == 10 and digits[:3] == '909':
                    count += 1
    return count

print('intro_first = ', count_909(",/intro_first"))
print('intro_second =', count_909("./intro_second"))
print('files_challenge = ',count_909("./files_challenge"))       



# How many people have three "i"'s somewhere in their name 

import os
import os.path

def count_3i(path):
    '''counting the number of names among all the files in the folder that has 3 i's'''
    counts = []
    for currentpath, subfolders, files in os.walk(path):
        for file in files:                                      #count the number of digits for the phone number in each file
            if file.endswith(".txt"):
                file_path = os.path.join(currentpath,file)
                try:
                    with open(file_path,'r', encoding='utf-8') as f:
                        contents = f.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                # Count the digits in the file
                i_count = sum(1 for char in contents if char =='i' or char =='I')
                counts.append(i_count)
    count_desired = sum(1 for count in counts if count == 3)
    return count_desired


print('intro_first = ', count_3i(",/intro_first"))
print('intro_second =', count_3i("./intro_second"))
print('files_challenge = ',count_3i("./files_challenge"))      


# Who has the most "i"s somewhere in their name, including both upper case and lower case i's
import re
import os
import os.path

def get_non_digit_parts(s): #function cited from chatgpt
    """
    Returns a list of all non-digit sequences in the given string.
    """
    # \D+ matches one or more non-digit characters
    return re.findall(r'\D+', s)


def find_most_i(path):
    '''return the name of the person that has the most i's among all the files in the folder'''
    name_dict={}
    for currentpath, subfolders, files in os.walk(path):
        for file in files:                                      #count the number of digits for the phone number in each file
            if file.endswith(".txt"):
                file_path = os.path.join(currentpath,file)
                try:
                    with open(file_path,'r', encoding='utf-8') as f:
                        contents = f.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                lines = contents.splitlines()
                if len(lines) >= 4:
                    continue
                num_i = sum(1 for char in contents if char =='i' or char == 'I')
                person = tuple(get_non_digit_parts(contents))
                name_dict[person] = num_i
    max_i_name = max(name_dict, key = name_dict.get) # line cited from ChatGPT
    return max_i_name

print("intro_first = ", find_most_i("./intro_first"))
print("intro_second",find_most_i("./intro_second"))
print("files_challenge", find_most_i("./files_challenge"))


# How many people have the string Khaby somewhere in their name?

import os
import os.path

def count_Khaby(path):
    '''return the number of names that contain Khaby somewhere among all the files in a folder'''
    count = 0
    for currentpath, subfolders, files in os.walk(path):
        for file in files:                                      #count the number of digits for the phone number in each file
            if file.endswith(".txt"):
                file_path = os.path.join(currentpath,file)
                try:
                    with open(file_path,'r', encoding='utf-8') as f:
                        contents = f.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                person_name = get_non_digit_parts(contents)
                if any("khaby" in part.lower() for part in person_name): # line cited from ChatGPT
                    count += 1
    return count


print("intro_first = ", count_Khaby("./intro_first"))
print("intro_second",count_Khaby("./intro_second"))
print("files_challenge", count_Khaby("./files_challenge"))


S = """0456894586749856
Jenna Luo
"""   

L = S.split("\n")
L[1]


# How many people have the last name we desire ?????

import os
import os.path

def count_last_name(path, name):
    '''return the number of names with last name containing the name we desire'''
    count = 0
    for currentpath, subfolders, files in os.walk(path):
        for file in files:                                      #count the number of digits for the phone number in each file
            if file.endswith(".txt"):
                file_path = os.path.join(currentpath,file)
                try:
                    with open(file_path,'r', encoding='utf-8') as f:
                        all_text = f.read()
                        L = all_text.split("\n")
                        if len(L)<1: continue
                        person_name = L[1]
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                if "," in person_name:                                       # if there's a comma, the part of person_name before the comma is the last name
                    last_name = person_name.split(",",1)[0].strip()          # line cited fron ChatGPT
                else:                                                        # if there's no comma, the part of the person_name after the last space is the last name
                    last_name = person_name.split(" ", 1)[-1].strip()        # line cited from ChatGPT
                if name.lower() in last_name.lower():
                    count += 1
    return count


print("intro_first = ", count_last_name("./intro_first", "Khaby"))
print("intro_second = ", count_last_name("./intro_second", "Khaby"))
print("files_challenge = ", count_last_name("./files_challenge", "Khaby"))

print("intro_first = ", count_last_name("./intro_first", "D'Amelio"))
print("intro_second = ", count_last_name("./intro_second", "D'Amelio"))
print("files_challenge = ", count_last_name("./files_challenge", "D'Amelio")) 

print("intro_first = ", count_last_name("./intro_first", "Luo"))
print("intro_second = ", count_last_name("./intro_second", "Luo"))
print("files_challenge = ", count_last_name("./files_challenge", "Luo")) 

                

                 


# How many people have the first name we desire ?????

import os
import os.path

def count_first_name(path, name):
    '''return the number of names with the first name containing the name we desire'''
    count = 0
    for currentpath, subfolders, files in os.walk(path):
        for file in files:                                      #count the number of digits for the phone number in each file
            if file.endswith(".txt"):
                file_path = os.path.join(currentpath,file)
                try:
                    with open(file_path,'r', encoding='utf-8') as f:
                        all_text = f.read()
                        L = all_text.split("\n")
                        if len(L)<1: continue
                        person_name = L[1]
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                if "," in person_name:                                       # if there's a comma, the part of person_name before the comma is the last name
                    first_name = person_name.split(",",1)[1].strip()          # line cited fron ChatGPT
                else:                                                        # if there's no comma, the part of the person_name after the last space is the last name
                    first_name = person_name.split(" ", 1)[0].strip()        # line cited from ChatGPT
                if name.lower() in first_name.lower():
                    count += 1
    return count
# people have my first name
print("intro_first = ", count_first_name("./intro_first", "Jenna"))
print("intro_second = ", count_first_name("./intro_second", "Jenna"))
print("files_challenge = ", count_first_name("./files_challenge", "Jenna")) 

# people having another first name
print("intro_first = ", count_first_name("./intro_first", "Grothendiek"))
print("intro_second = ", count_first_name("./intro_second", "Grothendiek"))
print("files_challenge = ", count_first_name("./files_challenge", "Grothendiek")) 





# How many people have their first name start with "J" and last name start with "L"?

import os
import os.path

def count_firstlast_letter(path, firstname_letter, lastname_letter):
    '''return the number of names with the first name start with a certain letter and the last name start with a nother letter'''
    count = 0
    for currentpath, subfolders, files in os.walk(path):
        for file in files:                                      #count the number of digits for the phone number in each file
            if file.endswith(".txt"):
                file_path = os.path.join(currentpath,file)
                try:
                    with open(file_path,'r', encoding='utf-8') as f:
                        all_text = f.read()
                        L = all_text.split("\n")
                        if len(L)<1: continue
                        person_name = L[1]
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                if "," in person_name:                                       # if there's a comma, the part of person_name before the comma is the last name
                    first_name = person_name.split(",",1)[1].strip()
                    last_name = person_name.split(",",1)[0].strip()
                    if first_name[0] == firstname_letter and last_name[0] == lastname_letter:
                        count += 1
                elif " " in person_name:                                                        # if there's no comma, the part of the person_name after the last space is the last name
                    first_name = person_name.split(" ", 1)[0].strip()
                    last_name = person_name.split(" ", 1)[-1].strip()
                    if first_name[0] == firstname_letter and last_name[0] ==lastname_letter:
                        count += 1       
    return count

print("intro_first = ", count_firstlast_letter("./intro_first", "J", "L"))
print("intro_second = ", count_firstlast_letter("./intro_second", "J", "L"))
print("files_challenge = ", count_firstlast_letter("./files_challenge", "J", "L")) 





# How many people named Tom have cellphone number start with 909
import os
import os.path

def count_name_909(path, name, code):
    '''return the number of people who have the name and area code we desire'''
    count = 0
    for currentpath, subfolders, files in os.walk(path):
        for file in files:                                      #count the number of digits for the phone number in each file
            if file.endswith(".txt"):
                file_path = os.path.join(currentpath,file)
                try:
                    with open(file_path,'r', encoding='utf-8') as f:
                        all_text = f.read()
                        L = all_text.split("\n")
                        if len(L)<1: continue
                        number = L[0]
                        person_name = L[1]
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                if person_name == name and number[:3] == code:    
                    count += 1                                               
    return count

print("intro_first = ", count_name_909("./intro_first", "Tom","909"))
print("intro_second = ", count_name_909("./intro_second", "Tom", "909"))
print("files_challenge = ", count_name_909("./files_challenge", "Tom","909"))



# counting the total number of files in a folder, and represent it in binary form: 

import os
import os.path


def count_files(path):
    '''count the number of files in a folder'''
    result = list(os.walk(path))
    count = 0
    for folder_tuple in result:
        currentpath, subfolders, files = folder_tuple 
        for file in files:
            count += 1
    return bin(count)

print("intro_first = ", count_files("./intro_first"))
print("intro_second = ", count_files("./intro_second"))
print("files_challenge = ", count_files("./files_challenge"))
        






