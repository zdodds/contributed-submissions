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


# my turn!

animal = "ducks"
people = "CS35_Participant_6"
belief = "spaghetti"

print(f"Hi, I am {people} and I am also a {animal} who is believes in {belief}!")
print()



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



count = 0

for x in range (1000):
  sqr = x**2
  if '4' in str(sqr):
    count += 1

print(count)


# variation - if the lenght of the square is 5...

count = 0

for x in range (1000):
  sqr = x**2
  if len(str(sqr)) == 5:
    #print(x)
    count += 1

print(count)


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
    result = ""

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
        if filename == None:  filename = "allsonnets.txt"
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



INPUT_STRING = GET_STRING_FROM_FILE("allsonnets.txt")
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
    ''' print a string, s, 42 times
    '''
    for i in range (42):
      print(s)
    pass


#
# alien( N ):          	which should return the string "aliii...iiien" with exactly N "i"s
#

def alien( N ):
    ''' return N number of "i"s
    '''
    for i in range (N):
        new = "al" + N*"i" +"en"
    return (new)


#
# count_digits( s ):   returns the number of digits in the input string s
#

def count_digits( s ):
    '''return number of digit of s'''
    count = 0
    for i in s:
      if i.isdigit():
        count += 1
    return count


#
# just_digits( s ):   returns only the digits in the input string s
#

def just_digits( s ):
    ''' return just eh digits of s'''
    new = ''
    for i in s:
      if i.isdigit():
        new += i
    return new


#
# just_text( s ):   returns a new version of s, where
#                    everything has been lower-cased, and
#                    all non-letters and non-spaces have been removed
#                    (that is, just keep the "text")
#
#                    Hint: s.lower() returns an all lower-case version of s

def just_text( s ):
    f''' f-docstrings are allowed, too :)
         be sure to include a docstring here!
    '''
    new = s.lower()
    for i in new:
      if not i.isalpha() and not i.isspace():
        new = new.replace(i,'')
    return new


# my functions:

def lemon(s):
    """ return the amount of the word lemon from string s
    for instance, you need 2 l,e,m,o, and n. to make 2 lemons
    """

    lemon_tree = {'l':0, 'e':0, 'm':0, 'o':0, 'n':0}
    s = just_text(s)
    for i in s:
      if i in lemon_tree:
        lemon_tree[i] += 1

    num_lemon = min(lemon_tree.values())
    return num_lemon

print(lemon("lemon, lemon, and lemons")) #Should return 3
print(lemon("Lemons are tasty, but my mom does not makes the best lime cake")) #should return 2


def college(grades):
    """
    Takes in a list of letter grades.
    - If any grade is below a B-, your offer gets rescinded.
    - If more than half are B/B-/B+, you get a boo.
    - If all are A's, you get a good job.
    """

    B_grades = ["B+", "B", "B-"]
    A_count = 0
    B_count = 0

    for grade in grades:
        if grade in ["C+", "C", "C-", "D", "F"]:
            return "rescind offer, no school for you"
        if grade in B_grades:
            B_count += 1
        if grade == "A":
            A_count += 1

    if A_count == len(grades):
        return "good job"
    if B_count > len(grades) // 2:
        return "boo"

    return "you're good"

print(college(["A", "B", "C", "B", "C"]))  # Should rescind offer
print(college(["A", "B", "B+", "B-", "A"]))  # Should boo
print(college(["A", "A", "A"]))  # Should say good job
print(college(["A", "B", "A"]))  # Should say you're good


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




