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

interface = "jupyter"
language = "python"
application = "vscode"

print(f"hello from {interface}, running {language}, within {application}...!")
print()

company = "Meta"
print(f"{company} would be proud!")

#
# Feel free to create your own quippy example here or in a new cell...
#

flight = 1028
departure = "ONT"
arrival = "DFW"
dtime = "10:35am"
atime = "12:22pm"

print(f"Flight {flight} departs from {departure} at {dtime} and arrives in {arrival} at {atime}")




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



LC2 = [x*x for x in range(1000) if '4' in str(x*x)]
print(LC2)
answer2 = len(LC2)
print(f"{answer2} of the first 1000 integers contain the digit 4 when squared.")


LC3 = [x*x for x in range(1000) if '4' in str(x*x) or '2' in str(x*x)]
print(LC3)
answer3 = len(LC3)
print(f"{answer3} of the first 1000 integers contain the digit 4 or 2 when squared.")

""" I was interested in this question since it seemed like the majority of numbers would contain at least one '2' or '4', and the result seems reasonable
with that in mind -- over 3/4s of all the squared numbers contain one or the other. I answered this by doing a list comprehension and an if statement to
verify that the values contained either 2 and 4.
"""


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
    result = ''

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
    ''' prints the given string s 42 times on separate lines
    '''
    for i in range(42):
        print(s)

times42("yay")


#
# alien( N ):          	which should return the string "aliii...iiien" with exactly N "i"s
#

def alien( N ):
    ''' returns the word alien with N number of 'i's
    '''
    return "al" + "i"*N + "en"

alien(5)



#
# count_digits( s ):   returns the number of digits in the input string s
#

def remOne(s, e):
    rstring = ''
    for x in s:
        if x != e:
            rstring += x
    return rstring
        

def count_digits( s ):
    '''returns the number of digits 0-9 in the input string s'''
    digitsNotSeen = '0123456789'
    count = 0 
    for i in range(len(s)):
        if s[i] in digitsNotSeen:
            count += 1
            digitsNotSeen = remOne(digitsNotSeen, s[i])
    return count 

count_digits('01asdkfjldsk9999')




#
# just_digits( s ):   returns only the digits in the input string s
#

def just_digits( s ):
    ''' returns a string listing the digits contained in the input string s '''
    digitsSeen = ''
    for i in range(len(s)):
        if s[i] not in digitsSeen:
            digitsSeen += s[i]
    return digitsSeen 
    
just_digits('1233652222')


#
# just_text( s ):   returns a new version of s, where
#                    everything has been lower-cased, and
#                    all non-letters and non-spaces have been removed
#                    (that is, just keep the "text")
#
#                    Hint: s.lower() returns an all lower-case version of s

def just_text( s ):
    ''' returns lowercase string s removing all characters that aren't letters or spaces 
    '''
    rstring2 = ''
    lcs = s.lower()
    text = string.ascii_lowercase + " "
    for x in lcs:
        if x in text:
            rstring2 += x
    return rstring2


just_text("OncE3432 upon019 )(^%$^&^())a TIME...")



#
# your own string-transforming function:  up to you
#    the goal: come up with something uniquely yours ...
#              ... no need to go overboard (more challenges ahead!)
#
# be sure to show off at least two of your own tests, too
#

# Here's my example:

def poptart_score(s):
    """ returns the wordle score vs. "poptart" (exactly 7 letters)
          $ will represent the correct letter in a correct spot
          ~ will represent the correct letter in an incorrect spot
    """
    # let's make sure s has exactly 7 letters
    s = s + "*"*7   # add extra asterisks at the end to have at least 7 letters
    s = s[0:7]      # trim to 7 letters using a slice

    result = ""
    for i in range(len(s)):  # loop over each index i, from 0 to len(s)-1
        if "poptart"[i] == s[i]:
            result += '$'    # correct letter, correct spot
        elif s[i] in "poptart":
            result += '~'    # correct letter, wrong spot
        else:
            result += ' '    # not a correct letter

    return result

#
# be sure to run on a large string
#
INPUT_STRING = """scripps claremontmckenna pitzer mudd pomona sci50l poptart"""
INPUT_LIST = INPUT_STRING.split()
for s in INPUT_LIST:              # Let's test it on each string (word) in our list
    inp = s                       # the function input
    out = poptart_score(s)       # the function output
    inp7 = (s + "*"*7)[:7]          # a seven-character (padded/sliced) input
    print(f"{inp:>25s}  ->  |{inp7}|")
    print( f"{'':>25s}      |{out}|")      # f-strings to show the transformation
    print( f"{'':>25s}      |{'poptart'}|\n")


#
# also be sure to run on at least one file find the results
#     can be only printing, but if there's something to maximize or minimize,
#     it is always fun to see what's "best" or "worst"
#

INPUT_STRING = GET_STRING_FROM_FILE("input.txt")
INPUT_LIST = INPUT_STRING.split("\n")  # '\n'

AllResults = []

for s in INPUT_LIST:              # Let's test it on each string (word) in our list
    inp = s                       # the function input
    out = poptart_score(s)       # the function output
    inp7 = (s + "*"*7)[:7]          # a seven-character (padded/sliced) input
    print(f"{inp[0:20]:>25s}  ->  |{inp7}|")
    print( f"{'':>25s}      |{out}|")      # f-strings to show the transformation
    print( f"{'':>25s}      |{'poptart'}|\n")
    this_score = out.count("$")*4 + out.count("~")*2  # my scoring system!
    AllResults.append( [out,inp] )       # append the output and input as a _sublist_

maximum_line = max(AllResults)          # take the maximum
print("The max and maximum line:", maximum_line)


#
# your own string-transforming function:  up to you
#    encouraged: come up with something uniquely yours (or your team's)
#                ... but no need to go overboard (more problems are ahead!)
#
# be sure to show off a couple of your own tests, too
#

# Here's Kenneth and Charlie's:


def obish(s):
    '''
    Input: a string, s
    Output: the string converted to obish (https://www.instructables.com/How-to-speak-Obish/)
    '''
    vowels = 'aeiouy'
    outputStr = ""
    for i in s:
        if i in vowels:
            outputStr += "ob"
        outputStr += i
    return outputStr

print(obish('bingus'))
print(obish('Kenneth goes to my school.'))
print(obish('brb I have to get my laundry'))
assert obish('bingus') == 'bobingobus'
assert obish('Kenneth goes to my school.') == 'Kobennobeth goboobes tobo moby schoboobol.'
assert obish('brb I have to get my laundry') == 'brb I hobavobe tobo gobet moby lobaobundroby'

#
# note from ZD:  not sure what I'd maximize here, but so be it!
#


#
# your own string-quantifying function:  again, up to you
#   something uniquely yours: a question not-yet-asked :)
#
# be sure to show off at least two of your own tests, too
#

print("your function here! (Be sure to have small-example tests and a large-example test!)")

def calcSpeak(s): 
    letNumDict = {
    "A": "4", "a": "4",
    "E": "3", "e": "3",
    "I": "1", "i": "1",
    "O": "0", "o": "0",
    "L": "1", "l": "1",
    "B": "8", "b": "8",
    "S": "5", "s": "5",
    }
    rstring3 = ""
    for x in s:
        if x in letNumDict.keys():
            rstring3 += letNumDict[x]
        else:
            rstring3 += x
    return rstring3

assert calcSpeak('Table') == 'T4813'
assert calcSpeak('Sea') == '534'


def wordValue(s):
    sum = 0
    for x in s:
        if x in '0123456789':
            sum += int(x)
    return sum

print(wordValue(calcSpeak('Stop')))
print(wordValue(calcSpeak('Yield')))
print(wordValue(calcSpeak('Pedestrian Crossing')))
print(wordValue(calcSpeak('School Zone')))


INPUT_SONNET = GET_STRING_FROM_FILE("allsonnets.txt")
SONNET_WORD_LIST = INPUT_STRING.split("\n")  # '\n'


TheResults = []

for s in SONNET_WORD_LIST:              # Let's test it on each string (word) in our list
    word = s                       # the function input
    score = wordValue(calcSpeak(s))       # the function output
    TheResults.append( [score,word] )       # append the output and input as a _sublist_

max_line = max(TheResults)          # take the maximum
print("The max and maximum line:", max_line)





