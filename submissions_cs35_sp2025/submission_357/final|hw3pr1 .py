import requests

url = "https://www.cs.hmc.edu/~dodds/demo.html"
result = requests.get(url)
print(f"{result = }")


# Let's print the text we just grabbed:
snack_page = result.text
print(snack_page)

text = snack_page         # ok to have many names...


#hw3pr1a

end = 0

while True:
    start = snack_page.find('<li class="snack">', end)
    if start == -1: 
        break  # Stop if no more snacks are found

    end = snack_page.find('</li>', start)  # Find the closing tag
    if end == -1:
        break  # Prevent errors if no closing tag is found

    snack_content = snack_page[start+len('<li class="snack">'):end]  # Extract content
    print(f"Snack: {snack_content.strip()}")  # Strip to remove spaces/newlines

print("\nComplete!")


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


import requests

url = "https://www.cs.hmc.edu/people/faculty/"
result = requests.get(url)
print(f"{result = }")
snack_page = result.text
print(snack_page)

text = snack_page 




#
# hw3pr1, part (b)
#

#
# Feel free to use this cell - and additional ones...
#

# Listing the names of nobel laureates in physics
import requests

# Fetch the Wikipedia page
url = "https://en.wikipedia.org/wiki/List_of_Nobel_laureates_in_Physics"
result = requests.get(url)

# Extract HTML text
wiki_page = result.text

# Define search term for laureates' names (they are linked inside <td> tags)
search_term = '<th scope="row" data-sort-value="'  # All laureates have Wikipedia links
FRONT = len(search_term)

end = 0

while True:
    start = wiki_page.find(search_term, end)
    if start == -1:
        break  # Stop when no more winners are found

    # Find the first three quotes
    start = wiki_page.find('"', start) + 1  # First quote (before data-sort-value)
    start = wiki_page.find('"', start) + 1  # Second quote (before the actual name)
    start = wiki_page.find('"', start) + 1  # Third quote (beginning of name)

    # Find the fourth quote, which marks the end of the name
    end = wiki_page.find('"', start)  
    if end == -1:
        break  

    # Extract and clean the winner's name
    winner_name = wiki_page[start:end].strip()

    # Print formatted output
    print(f"{winner_name = }")

print("\nYay!!!")







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
        new_line = re.sub(r"^# (.*)$", r"<h1>\1</h1>", line)  # capture the contents and wrap with <h1> and </h1>
                                                              # Aha! You will be able to handle the other headers here!
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

# Let's test this!
if True:
    old_contents = """
# Title
<br>
# Another title"""
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


#
# overall mardown-to-markup transformer
#

contents_v1 = original_markdown              # here is the input - be sure to run the functions, below:

contents_v2 = handle_newlines(contents_v1)   #   blank lines to <br>
contents_v3 = handle_headers(contents_v2)    #   # title to <h1>title</h1>  (more needed: ## to <h2>, ... up to <h6>)
contents_v4 = handle_code(contents_v3)       #   `code` to <tt>code</tt>

final_contents = contents_v1                 # here is the output - be sure it's the version you want!

write_to_file(final_contents, "output.html") # now, written to file:  Reload it in your browser!


# we can also print the final output's source - this should show the HTML (so far)
print(final_contents)    
# in addition, _do_ open up output.html in your browser and then View Source to see the same HTML (so far)


#My hw3pr1 (c)
import re

# [#1] Handle all six levels of headers <h1> through <h6>
def handle_headers(contents):
    """ Convert Markdown headers (#, ##, ###, etc.) into HTML headers """
    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        new_line = re.sub(r"^###### (.*)$", r"<h6>\1</h6>", line)  # Handle <h6>
        new_line = re.sub(r"^##### (.*)$", r"<h5>\1</h5>", new_line)  # Handle <h5>
        new_line = re.sub(r"^#### (.*)$", r"<h4>\1</h4>", new_line)  # Handle <h4>
        new_line = re.sub(r"^### (.*)$", r"<h3>\1</h3>", new_line)  # Handle <h3>
        new_line = re.sub(r"^## (.*)$", r"<h2>\1</h2>", new_line)  # Handle <h2>
        new_line = re.sub(r"^# (.*)$", r"<h1>\1</h1>", new_line)  # Handle <h1>

        NewLines.append(new_line)

    return "\n".join(NewLines)

# [#2] Handle five word-stylings:

# Strikethrough: ~~text~~ → <s>text</s>
def handle_strikethrough(contents):
    return re.sub(r'~~(.*?)~~', r'<s>\1</s>', contents)

# Bold: **bold** or __bold__ → <b>bold</b>
def handle_bold(contents):
    return re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'<b>\1\2</b>', contents)

# Italic: *italic* or _italic_ → <i>italic</i>
def handle_italic(contents):
    return re.sub(r'(?<!\\)(\*|_)([^\1]+?)\1', r'<i>\2</i>', contents)

# Unordered lists: + item → <ul><li>item</li></ul>
def handle_unorderedLists(contents):
    lines = contents.split("\n")
    new_lines = []
    in_list = False

    for line in lines:
        if line.startswith("+ "):
            if not in_list:
                new_lines.append("<ul>")
                in_list = True
            new_lines.append(f"<li>{line[2:]}</li>")
        else:
            if in_list:
                new_lines.append("</ul>")
                in_list = False
            new_lines.append(line)

    if in_list:
        new_lines.append("</ul>")

    return "\n".join(new_lines)

# URLs: [text](link) → <a href="link">text</a>
def handle_url(contents):
    return re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', contents)

# [#3] Three extra stylings:

# Escape sequences: \*word* → *word* instead of <i>word</i>
def handle_backslash(contents):
    """ Convert \*word* to *word* and \_word_ to _word_ instead of escaping them """
    contents = re.sub(r'\\\*', '*', contents)  # Unescape * 
    contents = re.sub(r'\\_', '_', contents)  # Unescape _
    return contents


# Superscript and Subscript: ^sup^ → <sup>sup</sup>, ~sub~ → <sub>sub</sub>
def handle_superscript_subscript(contents):
    contents = re.sub(r'\^([^\^]*)\^', r'<sup>\1</sup>', contents)
    contents = re.sub(r'~([^~]*)~', r'<sub>\1</sub>', contents)
    return contents

# Hide "CS35_Participant_9" by making it invisible
def handle_anna(contents):
    return re.sub(r'\bAnna\b', '<span style="background:white;">CS35_Participant_9</span>', contents, flags=re.IGNORECASE)

# Master function to apply all transformations
def markdown_to_html(markdown_text):
    markdown_text = handle_headers(markdown_text)
    markdown_text = handle_strikethrough(markdown_text)
    markdown_text = handle_bold(markdown_text)
    markdown_text = handle_italic(markdown_text)
    markdown_text = handle_unorderedLists(markdown_text)
    markdown_text = handle_url(markdown_text)
    markdown_text = handle_backslash(markdown_text)
    markdown_text = handle_superscript_subscript(markdown_text)
    markdown_text = handle_anna(markdown_text)
    return markdown_text

# Test Markdown Input
markdown_text = """
# Welcome to My Blog

## About Me
Hi! I'm a student at **The Webb Schools**, and I love **listening to music**, *playing with my cat*, and ~~watching Netflix shows~~.  
I am playing around with **Markdown-to-HTML conversions**—  

### Hello again!  
 
#### Facts  
+ My name is CS35_Participant_9  
+ \*Wow\* 
+ \_\_Bold\_\_
+ I live in South Hutch  

##### Here's a useful link  
Check out [Google](https://google.com) for anything you need!  

###### More things to try out
- Superscript: ^this^
- Subscript: ~that~.
"""

# Convert Markdown to HTML
html_output = markdown_to_html(markdown_text)

# Save output to file
def write_to_file(contents, filename="output.html"):
    with open(filename, "w") as f:
        f.write(contents)
    print(f"File '{filename}' written successfully!")

# Write the output HTML file
write_to_file(html_output)

# Print final HTML
print(html_output)





