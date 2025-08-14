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
# s.find("e")                            # try 'a', 'j', 'hi', 'hit', and 'z' ! jk

s.find("hi", 5)   # try ("j",15)



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
# motivation for this is to see if the frequency of mentions of certain dyes match my expectation of how popular/valued they were in historical contexts

url = "https://en.wikipedia.org/wiki/Natural_dye"

result = requests.get(url)
content = result.text

color_names = [' red', ' blue', ' green', ' yellow', ' purple', ' orange', ' black', ' pink']

color_counts = {color: 0 for color in color_names}

# Count occurrences of each color mentioned in the context of natural dyes
for color in color_names:
    start = 0
    while True:
        start = content.lower().find(color, start)
        if start == -1:
            break
        color_counts[color] += 1
        start += len(color)  # Move past the current match

for color, count in color_counts.items():
    print(f"{color}: {count}")
  
    




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


re.sub(r"d", "dd", "Mildred Mudd")          # try "Mildred Mudd"


# ANCHORS:  Patterns can be anchored:   $ meand the _end_
re.sub(r"d$", "dd", "Mildred Mud" )   # $ signifies (matches) the END of the line


# ANCHORS:  Patterns can be anchored:   ^  means the _start_ 
re.sub(r"^M", "ℳ", "Mildred Mudd" )   # ^ signifies (matches) the START of the line  (unicode M :)


# PLUS  +   means one or more:
re.sub(r"i+", "𝒾", "Isn't the aliiien skiing this weekend? AiiiIIIiiiiIIIeee!" )   # try replacing with "" or "I" or "𝒾" or "ⓘ"


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

# # Challenge! The dot . matches _any_ single character:  
re.sub( r".", "-", "words or phrases" )   # What will this do?

re.sub( r".s", "-S", "words or phrases" )  # And this one?!

re.sub( r".+s", "-S", "words or phrasesl" )  # And this one?!!


# The star (asterisk) matches ZERO or more times...
re.sub(r"42*", "47", "Favorite #'s:  4 42 422 4222 42222 422222")       # try + {2}  {1,3}   (42)


re.sub(r'o*','-', 'Gogle')


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
contents_v4 = handle_code(contents_v3)
contents_v5 = handle_word_mods(contents_v4)       #   `code` to <tt>code</tt>
contents_v6 = handle_unordered_lists(contents_v5)
contents_v7 = handle_urls(contents_v6)
contents_v8 = backgroundPlusCentering(contents_v7)


final_contents = contents_v8                # here is the output - be sure it's the version you want!

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


# here is a function to handle headers -- modifications made with help from Copilot
#
import re

def handle_headers(contents):
    """ replace all of the #, ##, ###, ... ###### headers with <h1>, <h2>, <h3>, ... <h6> """
    NewLines = []
    OldLines = contents.split("\n")

    header_patterns = [
        (r"^###### (.*)$", r"<h6>\1</h6>"),
        (r"^##### (.*)$", r"<h5>\1</h5>"),
        (r"^#### (.*)$", r"<h4>\1</h4>"),
        (r"^### (.*)$", r"<h3>\1</h3>"),
        (r"^## (.*)$", r"<h2>\1</h2>"),
        (r"^# (.*)$", r"<h1>\1</h1>")
    ]

    for line in OldLines:
        new_line = line
        for pattern, replacement in header_patterns:
            new_line = re.sub(pattern, replacement, new_line)
            if new_line != line:  # If a match was found and replaced, break the loop
                break
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)  # join with \n characters so it's readable by humans
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


# made with help from Copilot

import re

def handle_word_mods(markdown_text):
    # Split the text into parts that need to be processed and those that should remain unchanged (image URLs)
    parts = re.split(r'(img src="[^"]*")', markdown_text)

    # Function to convert markdown to html for non-image parts
    def convert_part(part):
        # Convert bold (Markdown: **text** or __text__)
        part = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'<b>\1\2</b>', part)
        
        # Convert italics (Markdown: *text* or _text_)
        part = re.sub(r'\*(.*?)\*|_(.*?)_', r'<i>\1\2</i>', part)
        
        # Convert underline (Markdown: <u>text</u>)
        part = re.sub(r'<u>(.*?)</u>', r'<u>\1</u>', part)
        
        # Convert strikethrough (Markdown: ~~text~~)
        part = re.sub(r'~~(.*?)~~', r'<s>\1</s>', part)
        
        return part
    
    # Process only non-image parts
    html_parts = [convert_part(part) if not part.startswith('img src=') else part for part in parts]
    
    # Join all parts back together
    html_text = ''.join(html_parts)
    
    return html_text




# made with help from Copilot

import re

def handle_unordered_lists(markdown_text):
    # Split the text into lines
    lines = markdown_text.split('\n')
    
    html_lines = []
    in_list = False
    
    for line in lines:
        if line.startswith('+ '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            item = line[2:]
            html_lines.append(f'<li>{item}</li>')
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(line)
    
    if in_list:
        html_lines.append('</ul>')
    
    return '\n'.join(html_lines)






# made with help from Copilot

import markdown

def backgroundPlusCentering(markdown_text):
    # Convert Markdown to HTML
    html_content = markdown.markdown(markdown_text)

    # Add HTML structure with centered content and color-changing background
    html_output = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Centered Content with Color Changing Background</title>
        <style>
            body {{
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                padding: 20px;
                transition: background-color 1s ease;
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
                box-sizing: border-box;
            }}
            .content {{
                width: 100%;
                max-width: 800px;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                text-align: center;
            }}
        </style>
        <script>
            document.addEventListener('DOMContentLoaded', (event) => {{
                const colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#33FFF1'];
                let colorIndex = 0;
                setInterval(() => {{
                    document.body.style.backgroundColor = colors[colorIndex];
                    colorIndex = (colorIndex + 1) % colors.length;
                }}, 2000);
            }});
        </script>
    </head>
    <body>
        <div class="content">
            {html_content}
        </div>
    </body>
    </html>
    """
    return html_output




# made with help from copilot

import re

def handle_urls(markdown_text):
    """Converts markdown-style URLs [urlname](urllink) to HTML hyperlinks <a href="urllink">urlname</a> with hover effect."""
    # Regular expression to match markdown URLs
    url_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')

    # Substitute markdown URLs with HTML hyperlinks
    html_text = re.sub(url_pattern, r'<a href="\2" class="hover-effect">\1</a>', markdown_text)

    # Add HTML structure with hover effect for URLs
    html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Markdown to HTML with Hover Effect</title>
    <style>
        .hover-effect {{
            display: inline-block;
            transition: font-size 0.3s ease;
        }}
        .hover-effect:hover {{
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <div class="content">
        {html_text}
    </div>
</body>
</html>
    """
    return html_output


# string output after all of the above changes 
# (the required ones, plus 1. centering the content, 2. making the background change colors, 3. making the links increase text size if the mouse hovers over them)
# each function that I used AI for notes this in a comment at the top of the block, I used Copilot extensively and found it incredibly helpful for this type of stuff. 


print(contents_v8)



