import requests

url = "https://www.cs.hmc.edu/~dodds/demo.html"
result = requests.get(url)
print(f"{result = }")


# Let's print the text we just grabbed:
snack_page = result.text
print(snack_page)

text = snack_page         # ok to have many names...


#    0         1         2             # ten's place
#    0123456789012345678901234567      # one's place
s = "abcdefghijklmnopqrstuvwxy&jk"
s.find("j", 15)                            # try 'a', 'j', 'hi', 'hit', and 'z' ! jk!

                                       # s.find("a",15)   # try ("j",15)



end = 0

while True:
    start = snack_page.find('<li class="snack">', end)
    if start == -1: break     # stop if we're done!
    end = start + 42          # 42 characters!
    
    snack_slice = snack_page[ start:end ]
    print(f"{snack_slice = }")

print("\nComplete!")



end = 0

while True:
    start = snack_page.find('<li class="snack">', end)
    if start == -1: break     # stop if we're done!
    end = snack_page.find('</li>', start)  # find the correct ending!
    
    snack_slice = snack_page[ start:end+5 ]
    print(f"{snack_slice = }")

print("\nComplete!")


# we need the length of the search string!
FRONT = len('<li class="snack">')

end = 0

while True:
    start = snack_page.find('<li class="snack">', end)
    if start == -1: break     # stop if we're done!
    end = snack_page.find('</li>', start)  # find the correct ending!
    
    snack_slice = snack_page[ start+FRONT:end ]
    print(f"{snack_slice = }")

print("\nYay!!!")


#
# hw3pr1, part (b)
#

#
# Feel free to use this cell - and additional ones...
#

import requests
import re 

url = "https://tsl.news/"
result = requests.get(url)
print(f"{result = }")

# Let's print the text we just grabbed:
TSL_page = result.text
print(TSL_page)

text = TSL_page         # ok to have many names...

countPom = len(re.findall(r'Pomona', text ))     
countPit = len(re.findall(r'Pitzer', text )) 
countHMC = len(re.findall(r'HMC', text )) + len(re.findall(r'Harvey Mudd', text )) 
countScr = len(re.findall(r'Scripps', text )) 
countCMC = len(re.findall(r'CMC', text )) + len(re.findall(r'Claremont McKenna', text )) 
print(f'{countPom = }')    
print(f'{countPit = }')  
print(f'{countHMC = }')  
print(f'{countScr = }')  
print(f'{countCMC = }')  

countPP = len(re.findall(r'Pomona-Pitzer', text )) + len(re.findall(r'P-P', text))  
countCMS = len(re.findall(r'CMS', text )) + len(re.findall(r'Claremont-Mudd-Scripps', text )) + len(re.findall(r'Claremont Mudd Scripps', text ))
print(f'{countPP = }')  
print(f'{countCMS = }') 




# Let's import the regular expression library (it should be built-in)
import re


# REs are a whole language! 
# Let's see a strategic use, to get our snacks from the snack_page above:
import re

m = re.findall(r'<li class="snack">(.*)</li>', snack_page )      # Yikes!    Common functions: findall, sub, search, match  

print(f"{m = }")                                                 # Wow!!!


# Let's try some smaller examples to build up to the snack_page example:

# fundamental capabilities:  regex matching and substitution  
#
#    the regex:
#      matcher:    replacer:   in this string:
re.sub(r"Harvey",  "Mildred",  "Harvey Mudd")           # the 'r' is for 'raw' strings. They're best for re's.


re.sub(r"car", "cat",  "This car is careful!")          # we'll stick with substitution for now...  uh oh!  space or ,1


re.sub(r"d", "dd", "Harvey Mud")          # try "Mildred Mudd"


# ANCHORS:  Patterns can be anchored:   $ meand the _end_
re.sub(r"d$", "dd", "Mildred Mud" )   # $ signifies (matches) the END of the line


# ANCHORS:  Patterns can be anchored:   ^  means the _start_ 
re.sub(r"M", "^ℳ", "Mildred Mudd" )   # ^ signifies (matches) the START of the line  (unicode M :)


# PLUS  +   means one or more:
re.sub(r"i+", "i", "Isn't the aliiien skiing this weekend? AiiiIIIiiiiIIIeee!" )   # try replacing with "" or "I" or "𝒾" or "ⓘ"


# SquareBrackets  [iI]  mean any from that character group:
re.sub(r"[Ii]+", "i", "Isn't the aliiien skiing this weekend? AiiiIIIiiiiIIIeee!" )   # it can vary within the group!


# SquareBrackets allow ranges, e.g., [a-z]
re.sub(r"[a-z]", "*", "Aha! You've FOUND my secret: 42!")       # use a +,  add A-Z, show \w, for "word" character


# Let's try the range [0-9] and +
re.sub(r" [0-9]+", " 42",  "Aliens <3 pets! They have 45 cats, 6 lemurs, and 789 manatees!")   # DISCUSS!  no +? How to fix?!


re.sub( r"or", "and", "words or phrases" )
re.sub( r"s", "-", "words or phrases" )
re.sub( r"[aeiou]", "-", "words or phrases" )

re.sub( r"$", " [end]", "words or phrases" )
re.sub( r"^", "[start] ", "words or phrases" )

# Challenge! The dot . matches _any_ single character:  
re.sub( r".", "-", "words or phrases" )   # What will this do?

re.sub( r".s", "-S", "words or phrases" )  # And this one?!

re.sub( r".+s", "-S", "words or phrases" )  # And this one?!!


# The star (asterisk) matches ZERO or more times...
re.sub(r"42*", "47", "Favorite #'s:  4 42 422 4222 42222 422222")       # try + {2}  {1,3}   (42)


re.sub(r'a', '-', 'alabama')
re.sub(r'a', '', 'alabama')
re.sub(r'o', '-', 'Google')
re.sub(r'o+', '-', 'Google')
re.sub(r'^o+', '-', 'Google')
re.sub(r'a.a', 'a-a', 'alabama')
re.sub(r'a.*a', 'a-a', 'alabama')
re.sub(r'o*', '-', 'Google')


m = re.findall(r'<li class="snack">(.*)</li>', snack_page )   # parens are a "capture group"   # try w/o it  # try search & sub
                                                   # each set of parents "captures" the text inside it
print(f"{m = }")                                   # it can even be used later, as \1, \2, \3, etc. 


#
# Here is a code cell, with the entire first-draft markdown of the previous cell 
# 
# stored in the Python variable      original_markdown
#

original_markdown = """

# Claremont's Colleges - MARKDOWN version

The Claremont Colleges are a *consortium* of **five** SoCal institutions. <br>
We list them here.

## The 5Cs: a list
+ [Pomona](https://www.pomona.edu/)
+ [CMC](https://www.cmc.edu/)
+ [Pitzer](https://www.pitzer.edu/)
+ [Scripps](https://www.scrippscollege.edu/)
+ [HMC](https://www.hmc.edu/)

The above's an _unordered_ list.  <br>
At the 5Cs, we all agree there's __no__ order!

---

## Today's featured college: [CMC](https://coloradomtn.edu/)

<img src="https://ygzm5vgh89zp-u4384.pressidiumcdn.com/wp-content/uploads/2017/06/GWS_campusview_1000x627.jpg" height=160>

---

### Also featured: &nbsp; Scripps and Pitzer and Mudd and Pomona

<img src="https://i0.wp.com/tsl.news/wp-content/uploads/2018/09/scripps.png?w=1430&ssl=1" height=100px> &nbsp; 
<img src="https://upload.wikimedia.org/wikipedia/commons/f/f9/Brant_Clock_Tower%2C_Pitzer_College%2C_2016_%28cropped%29.jpg" height=100px> &nbsp; 
<img src="https://www.hmc.edu/about/wp-content/uploads/sites/2/2020/02/campus-gv.jpg" height=100px> &nbsp;
<img src="https://upload.wikimedia.org/wikipedia/commons/4/46/Smith_Tower_and_the_San_Gabriel_Mountains.jpg" height=100px>

Are there _other_ schools in Claremont?

### Claremont destinations
+ _Pepo Melo_, a fantastic font of fruit!
+ **Starbucks**, the center of Claremont's "city," not as good as Scripps's _Motley_ 
+ ***Sancho's Tacos***, the village's newest establishment
+ ~~In-and-out CS35_Participant_3~~ (not in Claremont, alas, but close! CMC-supported!)
+ `42`nd Street Bagel, an HMC fave, definitely _well-numbered_
+ Trader Joe's, providing fuel for the walk back to Pitzer _from Trader Joe's_

---

#### Regular Expression Code-of-the-Day 
`import re`               
`pet_statement = re.sub(r'dog', 'cat', 'I <3 dogs')`

#### New Construction of the ~~Day~~ _Decade_!

<img src="https://www.cs.hmc.edu/~dodds/roberts_uc.png" height=150> <br><br>

CMC's **_Roberts Science Center_, also known as _"The Rubiks Cube"_** <br>
Currently under construction, under deadline, and undeterred by SoCal sun, or rain... 

Testing the additional \*asterisk\* vs *italics* feature... and the \_underscore\_ vs _italics_ feature... 

<br><br>


"""


#
# here is a function to write a string to a file (default name: output.html)
#

def write_to_file(contents, filename="output.html"):
    """ writes the string final_contents to the file filename """
    f = open(filename,"w")
    print(contents, file=f)
    print(f"{filename = } written. Try opening it in a browser!")
    f.close()


#
# Let's write our original_markdown to file...
#

write_to_file(original_markdown)


# here is a function to change MARKDOWN to MARKUP
#
import re

def handle_down_to_up(contents):
    """ replace all instances of MARKDOWN with MARKUP """
    new_contents = re.sub(r"MARKDOWN", r"MARKUP", contents)  # simple substitution
    return new_contents

# Let's test this!
if True:
    old_contents = "This is MARKDOWN text"
    new_contents = handle_down_to_up(old_contents) 
    print(new_contents)



# here is a function to handle blank lines (making them <br>)
#
import re

def handle_newlines(contents):
    """ replace all of the just-newline characters \n with HTML newlines <br> """
    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        new_line = re.sub(r"^\s*$", r"<br>", line)  # if a line has only space characters, \s, we make an HTML newline <br>
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents


# Let's test this!
if True:
    old_contents = """
# Title
    
# Another title"""
    new_contents = handle_newlines(old_contents)
    print(new_contents)


# here is a function to handle headers - right now only h1 (top-level)
#
import re

def handle_headers(contents):
    """ replace all of the #, ##, ###, ... ###### headers with <h1>, <h2>, <h3>, ... <h6> """
    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        new_line = re.sub(r"^#{6} (.*)$", r"<h6>\1</h6>", line) # capture the contents and wrap with <h1> and </h1>
        if new_line == line: # Aha! You will be able to handle the other headers here!
            new_line = re.sub(r"^#{5} (.*)$", r"<h5>\1</h5>", line)
        if new_line == line: 
            new_line = re.sub(r"^#{4}(.*)$", r"<h4>\1</h4>", line)
        if new_line == line: 
            new_line = re.sub(r"^#{3} (.*)$", r"<h3>\1</h3>", line)
        if new_line == line:
            new_line = re.sub(r"^#{2} (.*)$", r"<h2>\1</h2>", line)
        if new_line == line:
            new_line = re.sub(r"^# (.*)$", r"<h1>\1</h1>", line)
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

# Let's test this!
if True:
    old_contents = """
# Title
<br>
## Another title
### And ANOTHER!
#### and yet another? 
##### and ANOTHER!
###### and one last one!"""
    new_contents = handle_headers(old_contents)
    print(new_contents)


# here is a function to handle code - using markdown backticks
#
import re

def handle_code(contents):
    """ replace all of the backtick content with <code> </code> """
    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        new_line = re.sub(r"`(.*)`", r"<tt>\1</tt>", line)  # capture the contents and wrap with <code> and </code>
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

# Let's test this!
if True:
    old_contents = """\
This is <tt>42</tt>   
<br> 
Our regex library:  <tt>import re</tt>"""
    new_contents = handle_code(old_contents)
    print(new_contents)


import re

def handle_bold(contents): 
    """ replace all of the bold content (__bold__ or **bold**) with <b> </b> """
    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        new_line = re.sub(r"__(.*)__", r"<b>\1</b>", line)  # capture the contents and wrap with <code> and </code>
        new_line2 = re.sub(r"\*\*(.*)\*\*", r"<b>\1</b>", new_line)
        NewLines.append(new_line2)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

# Let's test this!
if True:
    old_contents = """\
This is **42**   
<br> 
Our regex library:  <tt>import re</tt>
This is __another bold__"""
    new_contents = handle_bold(old_contents)
    print(new_contents)



import re

def handle_italics(contents): 
    """ replace all of the italic content (_bold_ or *bold*) with <i> </i> """
    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        if "img" in line:
            NewLines.append(line)
            continue

        # Find escaped underscores or asterisks (e.g., \_italic\_ or \*italic\*) (helped by ChatGPT here, kept my initial work which was similar but not quite there...)
        escaped_asterisks = re.findall(r'\\\*(.*?)\\\*', line)
        escaped_underscores = re.findall(r'\\_(.*?)\\_', line)

        # First, remove the escaped asterisks and underscores from the line temporarily
        line_without_escaped = re.sub(r'\\\*(.*?)\\\*', r'\1', line)
        line_without_escaped = re.sub(r'\\_(.*?)\\_', r'\1', line_without_escaped)

        # Replace actual italicized text with <i></i> tags
        line_with_asterisks = re.sub(r'(\W|^)\*(.*?)\*(\W|$)', r'\1<i>\2</i>\3', line_without_escaped)
        line_with_underscores = re.sub(r'(\W|^)_(.*?)_(\W|$)', r'\1<i>\2</i>\3', line_with_asterisks)

        # Reinsert the escaped asterisks and underscores back into the line (no backslashes should be added)
        for e in escaped_asterisks:
            line_with_underscores = line_with_underscores.replace(e, r'*' + e + r'*')

        for e in escaped_underscores:
            line_with_underscores = line_with_underscores.replace(e, r'_' + e + r'_')

        # Append the processed line to the final result
        NewLines.append(line_with_underscores)

    new_contents = "\n".join(NewLines)   # Join with \n characters so it's readable by humans
    return new_contents

        
        # returnLine = []

        # for i in range(len(line_with_underscores)):
        #     returnLine += line_with_underscores[i] 
        #     for e in escaped_asterisks:
        #         if line_with_underscores[i:i+len(e)] == e:
        #             if i > 0 and (i+len(e)) < len(line_with_underscores):
        #                 if i-1 != '>':
        #                     returnLine += line_with_underscores[i] + '*' + line_with_underscores[i:i+len(e)] + '*'
        #                     i = i + len(e)
        # return returnLine



        #astOrIt = re.findall(r"(\W|^)\*(.*?)\*(\W|$)", r"\1<i>\2</i>\3", line)  
        #undOrIt = re.findall(r"(\W|^)_(.*?)_(\W|$)", r"\1<i>\2</i>\3", line)
        #for e in astOrIt and not in asteriskList: 
        #re.sub(f'e', )

        #NewLines.append(new_line2)


    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents


# re.findall(r'\bf[a-z]*', 'which foot or hand fell fastest')
# ['foot', 'fell', 'fastest']
# re.findall(r'(\w+)=(\d+)', 'set width=20 and height=10')
# [('width', '20'), ('height', '10')]


# Let's test this!
line = """\
This is *48* and \*48\*
and another *Italics!!* and _another_
but \_this\_ shouldn't _count_...
"""

print("Original line:")
print(line)

# Call the function and print the result
result = handle_italics(line)
print("\nAfter processing:")
print(result)

# astOrIt = re.findall(r"\*(.*?)\*", line)
# print(astOrIt)
# undOrIt = re.findall(r"(_(.*?)_)",  line)
# print(undOrIt)
# asteriskList = re.findall(r'\\\*.*?\\\*', line)
# print(asteriskList)
# underList = re.findall(r'\\_.*?\\_', line)
# print(underList)


    



import re

def handle_strikethrough(contents): 
    """ replace all of the bold content (__bold__ or **bold**) with <b> </b> """
    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        new_line = re.sub(r"~~(.*)~~", r"<s>\1</s>", line)  # capture the contents and wrap with <code> and </code>
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

# Let's test this!
if True:
    old_contents = """\
This is ~~42~~   
<br> 
Our regex library:  <tt>import re</tt>
This is ~~another bold~~"""
    new_contents = handle_strikethrough(old_contents)
    print(new_contents)



def handle_list(contents):
    """ all content preceded by a + character will be surrounded by square brackets as a one element list"""
    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        new_line = re.sub(r"\+ (.*$)", r'[\1]', line)  # capture the contents and wrap with <code> and </code>
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

contents = '+ aslfkjdskfj askdfj '
print(contents)
print(handle_list(contents))


def handle_url(contents):
    """ adjust url formattin from [name](url) to <a href = 'url'>name</a>"""
    NewLines = []
    OldLines = contents.split("\n") 

    for line in OldLines:
        new_line = re.sub(r"\[(.*)\]\((.*)\)", r'<a href="\2">\1</a>', line)  # capture the contents and wrap with <code> and </code>
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

contents = original_markdown
print(contents)
#print(handle_url(contents))
#print(re.sub("\", "fds", contents))
print(handle_url(contents))

# [Google](https://www.google.com) 
# becomes       <a href="https://www.google.com">Google</a>
# You'll have to use TWO capture groups  (.*?)   which become \1 and \2



def handle_color(contents):
    """ change school names to be written in school colors """
    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        if "img" in line:
            NewLines.append(line)
            continue
        new_line = re.sub(r'CMC', r'<span style = "color:Maroon;">CMC</span>', line)  
        new_line1 = re.sub(r'Pomona', r'<span style = "color:Blue;">Pomona</span>', new_line)
        new_line2 = re.sub(r'Pitzer', r'<span style = "color:Orange;">Pitzer</span>', new_line1)
        new_line3 = re.sub(r'Scripps', r'<span style = "color:SeaGreen;">Scripps</span>', new_line2)
        new_line4 = re.sub(r'HMC', r'<span style = "color:GoldenRod;">HMC</span>', new_line3)
        new_line5 = re.sub(r'Mudd', r'<span style = "color:GoldenRod;">Mudd</span>', new_line4)
        NewLines.append(new_line5)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

contents = 'abc CMC abc HMC'
print(contents)
#print(handle_url(contents))
#print(re.sub("\", "fds", contents))
print(handle_color(contents))    

#Color example:  <span style="color:DodgerBlue;">Go Dodgers!</span>
#Bckgd example:  <span style="background:Coral;">Go Dodgers!</span>


#
# overall mardown-to-markup transformer
#

contents_v0 = original_markdown              # here is the input - be sure to run the functions, below:

contents_v1 = handle_down_to_up(contents_v0)   #   blank lines to <br>
contents_v2 = handle_newlines(contents_v1)   #   blank lines to <br>
contents_v3 = handle_headers(contents_v2)    #   # title to <h1>title</h1>  (more needed: ## to <h2>, ... up to <h6>)
contents_v4 = handle_code(contents_v3)       #   `code` to <tt>code</tt>
contents_v5 = handle_bold(contents_v4) # **bold** or __bold__ to <b>bold</b>
contents_v6 = handle_italics(contents_v5) # *italics* or _italics_ to <i>italics</i>
contents_v7 = handle_strikethrough(contents_v6) # ~~strikethrough~~ to <s>strikethrough</s>
contents_v8 = handle_list(contents_v7) # handle lists! 
contents_v9 = handle_url(contents_v8) # handle URLs!
contents_v10 = handle_color(contents_v9) # changes school colors 
final_contents = contents_v10         # here is the output - be sure it's the version you want!

write_to_file(final_contents, "output.html") # now, written to file:  Reload it in your browser!



# we can also print the final output's source - this should show the HTML (so far)
print(final_contents)    
# in addition, _do_ open up output.html in your browser and then View Source to see the same HTML (so far)


