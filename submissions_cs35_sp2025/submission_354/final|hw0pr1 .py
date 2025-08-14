#
# python is a popular language for executable expression
#     This is a python cell
#     Try this shortcut: control-return or control-enter runs it:

x = 43
print("x is", x)

x -= 1

# notice that the last expression is the cell's output.
# printing and outputting are different!


#
# Let's fit all of Python into the next five or so cells!
#

L = [7,8,9]

print("L[1:] equals", L[1:])    # slicing!

print(f"L[1:] equals {L[1:]}")  # f strings!

print(f"{L[1:] = }")            # cool = syntax, new for me...

print()

colleges = 'cmc scripps pitzer hmc'
print(colleges)

'c' in colleges          # in is in!


x = 42

print("x =", x)    # "regular" print

print(  f"{x = }" )  # f-string print


"""wrong = 41"""
snack = "Poptarts"

# fs = f"{snack}? I'll take {wrong+1}!"
# print(fs)

for i in range(40,42+1):
    print(f"{snack}? I'll take {i}!")



print(f"{L=}  L.index(9) is", L.index(9))     # calling methods (functions after a ".")
print()
print("colleges.title() returns ", colleges.title())        # ditto!  Let's shout!
print()
print("# of c's in colleges:", colleges.count('c'))
print()
print(colleges.replace('c','C'))


LoW = colleges.split()
print("LoW is", LoW)

LoS = colleges.split('c')
print("LoS is", LoS)


LoW = [ "word", "leisa", "we", "some" ]
''.join(LoW)


#
# f-strings!
#

city = "Chongqing"
weather = "windy"
activity = "shopping"

print(f"hello from {city}, where the weather is {weather}, where we are {activity}...!")
print()

school = "Webb"
sport = "baseball"
print(f"I go to {school}! I like playing {sport}.")

#
# Feel free to create your own quippy example here or in a new cell...
#


# I <3 list comprehensions!
[ 2*x for x in range(20,24) ]


#
# Here is an example that uses list comprehensions to
# count how many of the first 1000 integers are even

LC = [ x for x in range(1000) if x % 2 == 0 ]
print(f"LC = {LC}")
print()

answer = len(LC)
print(f"answer = {answer}")



# remember pwd? It's short for "print working directory"  (that is, your current folder)
# It's a shell command, i.e., to be run at the terminal or command-line (or shell).

# Since this is not python, python needs to know it's a special command:  # does the trick



# Another important shell command:  ls (lists contents)



# the following cell shows cd, changing the current directory.
# We will also try # (to "move up" one level)

# WARNING:  when cd does work, the _whole notebook_ has changed what it considers
#           its "current directory"  You'll see this here:



# And, we need to be able to move "up" or "outward"!
# To do this, use   #     Here, in jupyter:




# we can keep going!   # "goes up" one directory level



# you don't need a new cell each time...



# Really, each notebook cell is a script, sharing state with the others

# Here, our code is guarded by an if True:

if False:
    print(f"It was True!")

# try changing the True to False.
# There's no need to change the printing, because it won't print...

# When submitting notebooks, we ask you to submit notebooks that run...
# However, feel free to keep any+all non-running code! Simply comment it out
# For example, with if False:
# hashtags and """ triple-quoted strings """ also work ...
# ... though they lose the code-structure and syntax highlighting! (As you see! :-)


# reminder: in each cell (script) all lines are run; only the last line is output
y = 58
x = y-7
x = x-1
x = x-8



# BEWARE:  Variables are global through the notebook!
# Here, y will be 58, from the cell above... if you've run that cell!



# a good starting point!

def plus1( N ):
    """ returns a number one larger than its input """
    return N+1

if True:
    print(f"plus1(41) -> {plus1(41)} should be {42}")
    print(f"plus1(9000) -> {plus1(9000)} should be {9001}")
    assert plus1(41) == 42
    assert plus1(9000) == 9001

# print tests succeed and fail "out loud"
# assert tests succeed silently and only fail "out loud" (and crash!)


#
# An infinite loop...
#

print(f"Start!")
i = 0

while True:
    print(i, end=" ", flush=True)
    i = i+1

print(f"End.")



#
# Let's loop!
#

snack = "Poptarts"

for i in range(0,5):
        print(f"{snack}? I'll take {i}!")


# countdown( N )    emphasizes looping and printing (there is no return value!)
#
import time

def countdown( N ):
    """ counts downward from N to 0, printing  """
    for i in range(N,-1,-1):  # starts at N, ends at 0, steps by -1
        print("i ==", i)
        time.sleep(0.2)

    return    # no return value here!

# testing countdown
if True:
    print("Testing countdown(5):")
    countdown(5)  # should print things -- with dramatic pauses!


if True:
    """ a cell for trying out the previous two cells """

    # sign on
    print(f"[[ Start! ]]\n")

    # testing plus1
    result = plus1( 41 )
    print(f"plus1(41) -> {result}")
    print()

    # testing countdown
    print("Testing countdown(4):")
    countdown(4)  # should print things -- with dramatic pauses!

    # sign off
    print("\n[[ Fin. ]]")


def mystery1( s ):
    """ mystery function #1 """
    result = 0

    for i in range(len(s)):
        if s[i] in 'iI':
            result += 1

    return result

result = mystery1("Aliens <3 caffeine")
print(result)


def mystery2( s ):
    """ mystery function #2 """
    result = 0

    for let in s:
        if let in 'iI':
            result += let

    return result

result = mystery2("I like caffeine")
print(result)


def count_eyes( s ):
    """ returns the number of times i or I appears in the input string s """
    result = 0

    for letter in s:
        if letter in 'iI':  #  equivalent:   letter == 'i' or letter == 'I'
            result += 1     #  add one to our overall result (our count)

    return result           #  AFTER the loop, we're done!

print("count_eyes('Yiiikes!') =",count_eyes('Yiiikes!'))
print(f"{count_eyes('Yiiikes!') = }")   # f-strings ⋮)
print()
print(f"{count_eyes('Italy, icily, livelily!') = }")


def just_eyes( s ):
    """ returns a string of only the letters i or I that appear in the input s """
    result = ''

    for letter in s:
        if letter in 'iI':    #  equivalent:   letter == 'i' or letter == 'I'
            result += letter  #  add _the letter_ to our overall result, now a string

    return result             #  AFTER the loop, we're done!

print("just_eyes('Yiiikes!') =",just_eyes('Yiiikes!'))
print(f"{just_eyes('Yiiikes!') = }")   # f-strings ⋮)
print()
print(f"{just_eyes('Italy, icily, livelily!') = }")


def GET_STRING_FROM_FILE(filename=None):
    """ return all of the contents from the file, filename
        will error if the file is not present, then return the empty string ''
    """
    try:
        if filename == None:  filename = "input.txt"
        INPUT_FILENAME = filename
        INPUT_FILE = open(INPUT_FILENAME, "r", encoding='utf-8')    # how to open a file
        DATA = INPUT_FILE.read()                                    # and get all its contents
        INPUT_FILE.close()                                          # close the file (optional)
        #print(DATA)                                                # if we want to see it
        return DATA                                                 # definitely want to return it!
    except FileNotFoundError:                   # wasn't there
        print(f"file not found: {filename}")    # print error
        return ''                               # return empty string ''

# split is a wonderful function, it returns a list.  Try it!
INPUT_STRING = GET_STRING_FROM_FILE("input.txt")

# Let's print only some of this large string, with all of Shakespeare's sonnets:
print(INPUT_STRING[0:200])



INPUT_STRING = GET_STRING_FROM_FILE("input.txt")
INPUT_LIST = INPUT_STRING.split("\n")  # '\n'

for s in INPUT_LIST:                   # Let's test it on each string (line) in our list
    inp = s                            # the input
    out = count_eyes(inp)              # the output
    # print(f"{inp} -> {out}")           # print result
    print(f"{inp:>50s} -> {out:<d}")   # f-strings have formatting options: (> right) (< left)


INPUT_STRING = GET_STRING_FROM_FILE("input.txt")
INPUT_LIST = INPUT_STRING.split("\n")  # '\n'

AllResults = []

for s in INPUT_LIST:                   # Let's test it on each string (line) in our list
    inp = s                            # the input
    out = count_eyes(inp)              # the output
    # print(f"{inp} -> {out}")           # print result
    print(f"{inp:>50s} -> {out:<d}")   # f-strings have formatting options: (> right) (< left)
    AllResults.append( [out,inp] )       # append the output and input as a _sublist_

maximum_line = max(AllResults)          # take the maximum
print("The max and maximum line:", maximum_line)


#
# using python's string library
#

import string

print(string.digits)
print(string.punctuation)
print(string.ascii_lowercase)


# plus1, countdown, count_eyes, and just_eyes are already complete :-)


#
# times42( s ):      	which should print the string s 42 times (on separate lines)
#

def times42( s ):
    ''' Prints the string s exactly 42 times, each on a new line.
    '''
    for _ in range(42):
        print(s)
    pass

print("Test 1: Printing 'Hello'")
times42("Hello")

print("Test 2: Printing 'Python is fun!'")
times42("Python is fun!")


#
# alien( N ):          	which should return the string "aliii...iiien" with exactly N "i"s
#

def alien( N ):
    ''' Returns the string 'aliii...iiien' with exactly N "i" characters.
    '''
    if N < 0:
        return "Error: N must be a non-negative integer."
    
    return "al" + "i" * N + "en"
    pass

print("Test 1:", alien(5))
print("Test 2:", alien(10))


#
# count_digits( s ):   returns the number of digits in the input string s
#

def count_digits( s ):
    ''' Returns the number of digits in the input string s. '''
    return sum(1 for char in s if char.isdigit())
    pass

print("Test 1", count_digits("abc123"))
print("Test 2", count_digits("no"))


#
# just_digits( s ):   returns only the digits in the input string s
#

def just_digits( s ):
    ''' Returns a string containing only the digits from the input string s. '''
    return "".join(char for char in s if char.isdigit())

    pass

print("Test 1:", just_digits("abc123"))
print("Test 2:", just_digits("no"))


#
# just_text( s ):   returns a new version of s, where
#                    everything has been lower-cased, and
#                    all non-letters and non-spaces have been removed
#                    (that is, just keep the "text")
#
#                    Hint: s.lower() returns an all lower-case version of s

def just_text( s ):
    f''' return lowercased, keeping only letters and spaces
    '''
    return "".join(char.lower() for char in s if char.isalpha() or char.isspace())
    pass

print("Test 1:", just_text("Hello, Webb! 123"))
print("Test 2:", just_text("CS is COOL!"))


def reverse_words(s):
    words = s.strip().split()
    
   
    if not words:
        return "Please reenter actual words to reverse"

    reversed_sentence = " ".join(words[::-1])

    return reversed_sentence

#
# The reasoning behind this function is the influence from the earlier functions I had to write for hw0pr1. I originally wanted to just reverse the letters, but thought it would be too easy. I had to think for a while on how I can reverse the words instead of the letters. The process that ended up working was thinking about how the code for reversing the letters work and working from there.
#
print(reverse_words("CS is fun"))
print(reverse_words("Disneyland"))



def falsereality(s):
    return "".join(char.lower() if char.isupper() else char.upper() for char in s)


#
# I wanted to create something that is easy but also complicated at the same time. The process of creating this was just looking at past notes and doing some research online. The process worked great.

print(falsereality("Hello World!"))
print(falsereality("PuiuiniuNININUNINiuninjnijnJNJNINJnjiniNJINIJNjinIJNIJNINJNININIJ"))




