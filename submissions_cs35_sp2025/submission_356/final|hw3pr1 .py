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
s.find("e")                            # try 'a', 'j', 'hi', 'hit', and 'z' ! jk!

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
import requests
from bs4 import BeautifulSoup
import re

#Counting how many times porridge is mentioned in Goldilocks and the Three Bears

url = "https://americanliterature.com/childrens-stories/goldilocks-and-the-three-bears"
result = requests.get(url)
print(f"{result = }")

goldi = result.text

soup= BeautifulSoup(goldi,'html.parser') 
text = soup.get_text()

porridge_count = len(re.findall(r"porridge",text,re.IGNORECASE))

print(f"The word 'porridge' appears {porridge_count} times in Goldilocks and the Three Bears.")



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


re.sub(r"a.*a", "a-a", "alabama")  


re.sub(r"d", "dd", "Harvey Mud")          # try "Mildred Mudd"


# ANCHORS:  Patterns can be anchored:   $ meand the _end_
re.sub(r"d$", "dd", "Mildred Mud" )   # $ signifies (matches) the END of the line


# ANCHORS:  Patterns can be anchored:   ^  means the _start_ 
re.sub(r"^M", "ℳ", "Mildred Mudd" )   # ^ signifies (matches) the START of the line  (unicode M :)


# PLUS  +   means one or more:
re.sub(r"i+", "i", "Isn't the aliiien skiing this weekend? AiiiIIIiiiiIIIeee!" )   # try replacing with "" or "I" or "𝒾" or "ⓘ"


# SquareBrackets  [iI]  mean any from that character group:
re.sub(r"[Ii]+", "i", "Isn't the aliiien skiing this weekend? AiiiIIIiiiiIIIeee!" )   # it can vary within the group!


# SquareBrackets allow ranges, e.g., [a-z]
re.sub(r"[a-z]", "*", "Aha! You've FOUND my secret: 42!")       # use a +,  add A-Z, show \w, for "word" character


# Let's try the range [0-9] and +
re.sub(r"[0-9]+", "42",  "Aliens <3 pets! They have 45 cats, 6 lemurs, and 789 manatees!")   # DISCUSS!  no +? How to fix?!


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


#
# overall mardown-to-markup transformer
#

contents_v0 = original_markdown              # here is the input - be sure to run the functions, below:

contents_v1 = handle_down_to_up(contents_v0)   #   blank lines to <br>
contents_v2 = handle_newlines(contents_v1)   #   blank lines to <br>
contents_v3 = handle_headers(contents_v2)    #   # title to <h1>title</h1>  (more needed: ## to <h2>, ... up to <h6>)
contents_v4 = handle_code(contents_v3)       #   `code` to <tt>code</tt>
contents_v5 = handle_text_stylings(contents_v4)
contents_v6 = handle_lists(contents_v5)
contents_v7 = handle_links(contents_v6)
contents_v8 = feature1(contents_v7)
contents_v9 = feature2(contents_v8)
contents_v10 = feature3(contents_v9)

final_contents = contents_v10                # here is the output - be sure it's the version you want!

write_to_file(final_contents, "output.html") # now, written to file:  Reload it in your browser!


# we can also print the final output's source - this should show the HTML (so far)
print(final_contents)    
# in addition, _do_ open up output.html in your browser and then View Source to see the same HTML (so far)


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
        for i in range(6, 0, -1):  # check for h6 h5 h4 h3 h2 h1 
            pattern = r'^#{' + str(i) + r'}\s+(.*)$' # str(i) is 654321 ^startswith$endswith
            replacement = r'<h' + str(i) + r'>\1</h' + str(i) + r'>' # capture the contents and wrap with <h1> and </h1>
            if re.match(pattern, line):
                line = re.sub(pattern, replacement, line)
                break  # Stop after first match
        NewLines.append(line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

# Let's test this!
if True:
    old_contents = """
# Title
<br>
# H1
## H2
### H3
#### H4
##### H5
###### H6"""
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


# functions to handle word-stylings
# completed with jen lim
#
import re

def handle_text_stylings(contents):
    """ replace tildes with <s></s> for strikethrough
    replace two asterisks **bold** and two underscores __bold__ with <b></b> for bold
    replace asterisks *italic* and underscores _italic_ for italic
    """
    contents = re.sub(r'~~(.*?)~~', r'<s>\1</s>', contents) #strikethrough
    contents = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'<b>\1\2</b>', contents) #bold
    contents = re.sub(r'\*(.*?)\*|_(.*?)_', r'<i>\1\2</i>', contents) #italic

    return contents

# Let's test this!
if True:
    old_contents = """
~~strikethrough~~
**bold** and __bold__
*italic* and _italic_
"""
    new_contents = handle_text_stylings(old_contents)
    print(new_contents)


def handle_lists(contents):
    """ Convert markdown lists to markup lists. """
    contents = re.sub(r'^\+ (.*)', r'<ul><li>\1</li></ul>', contents, flags=re.MULTILINE)
    return contents

# Let's test this!
if True:
    old_contents = """ 
+ list item
+ list item2
"""
    new_contents = handle_lists(old_contents)
    print(new_contents)


def handle_links(contents):
    """ Convert markdown links [text](url) into HTML links <a href='url'>text</a>. """
    contents = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\g<2>">\g<1></a>', contents)
    return contents

# Let's test this!
if True:
    old_contents = """
[Google](https://www.google.com)
[Goldilocks and the Three bears](https://americanliterature.com/childrens-stories/goldilocks-and-the-three-bears)
"""
    new_contents = handle_links(old_contents)
    print(new_contents)


def feature1(contents):
    """ removes backslashes that interupt asterisks and underscores """
    contents = re.sub(r'\\\*', '*', contents) #replace \* with *
    contents = re.sub(r'\\_', '_', contents)
    return contents

# Let's test this!
if True:
    old_contents = """
 This is \*Wow\* and \_Yikes\_.
 This is \_\_Yikes\_\_.
"""
    new_contents = feature1(old_contents)
    print(new_contents)


def feature2(contents):
    """ converts markdown superscript and subscript to markup """
    contents = re.sub(r'@@(.*?)@@', r'<sup>\1</sup>', contents) #superscript
    contents = re.sub(r'%%(.*?)%%', r'<sub>\1</sub>', contents) #subscript
    return contents

# Let's test this!
if True:
    old_contents = """
Y = e@@x@@ (Superscript)
"""
    new_contents = feature2(old_contents)
    print(new_contents)


def feature3(contents):
    """ converts markdown for text colors to markup """
    contents = re.sub(r'\{color:([\w#]+)\|(.*?)\}', r'<span style="color:\1;">\2</span>', contents)
    return contents

# Let's test this!
if True:
    old_contents = """
{color:red|red text}
{color:DodgerBlue|Go Dodgers!}
"""
    new_contents = feature3(old_contents)
    print(new_contents)


