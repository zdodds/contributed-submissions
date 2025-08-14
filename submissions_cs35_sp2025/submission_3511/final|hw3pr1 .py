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




import requests

url = "https://weather.com/weather/hourbyhour/l/0f0c8c35110bea7cb1c40711cee2d632338ce765cd396db8ed852c231cefa53f"
result = requests.get(url)
print(f"{result = }")

text = result.text
# print(text)





# FRONT = len('<li class="adWrapper">')

end = 0
ads = 0

# Credit to ChatGPT for teaching me using iframe tag to identify ads
# I think the requested HTML and the HTML in reality is different. In my live browser, there are a lot of tags with iframe
while True:
    start = text.find('<iframe', end)
    if start == -1: break     # stop if we're done!
    end = text.find('</iframe>', start)  # find the correct ending!
    
    ad = text[ start:start+42*3 ]
    print(f"iframe{ ad = }")
    ads+=1


# The 2 algorithm below are from my observations of what
end = 0
while True:
    start = text.find('<div', end)
    if start == -1: break     # stop if we're done!
    end = text.find('Advertisement</div>', start)  # find the correct ending!
    
    ad = text[ start:start+42*3 ]
    print(f"properly labeled {ad = }")
    ads+=1

end = 0
while True:
    start = text.find('<a', end)
    if start == -1: break     # stop if we're done!
    end = text.find('sponsored', start)  # find the correct ending!
    
    ad = text[ start:start+42*3  ]
    print(f"sponsored {ad = }")
    ads+=1

# Im sure that is a bunch of different types of ads that this algorithm doesn't account for!
# Funnily enough, I saw an ad for an adblocker.


print(f"{ads = }")




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


re.sub(r'o*','-','Google')


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
"https://upload.w<img src=ikimedia.org/wikipedia/commons/f/f9/Brant_Clock_Tower%2C_Pitzer_College%2C_2016_%28cropped%29.jpg" height=100px> &nbsp; 
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

contents_v1 = handle_down_to_up(contents_v0)   #   changes MARKDOWN to MARKUP
contents_v2 = handle_newlines(contents_v1)   #   blank lines to <br>
contents_v3 = handle_headers(contents_v2)    #   # title to <h1>title</h1>  can accomodate up to 6 #
contents_v4 = handle_code(contents_v3)       #   `code` to <tt>code</tt>
contents_v5 = handle_italic_bold(contents_v4)  # fix italics and bolding
contents_v6 = handle_strike(contents_v5)  #handles strikethrough
contents_v7 = handle_url(contents_v6) #embeds URLs
contents_v8 = handle_ulists(contents_v7) #does unordered lists

# new markdown features
contents_v9 = handle_school(contents_v8) #give each school their corresponding colors
contents_v10 = handle_dash(contents_v9) # changes --- to a big lines
contents_v11 = handle_dodds_love_42(contents_v10) #gives 42 privilige

final_contents = contents_v11                # here is the output - be sure it's the version you want!

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
        # behold! my lazy coding! I'll make a more general code once there is a h7 with html

        new_line = re.sub(r"^# (.*)$", r"<h1>\1</h1>", line)  # capture the contents and wrap with <h1> and </h1>
        new_line = re.sub(r"^## (.*)$", r"<h2>\1</h2>", new_line)
        new_line = re.sub(r"^### (.*)$", r"<h3>\1</h3>", new_line)    
        new_line = re.sub(r"^#### (.*)$", r"<h4>\1</h4>", new_line)    
        new_line = re.sub(r"^##### (.*)$", r"<h5>\1</h5>", new_line)    
        new_line = re.sub(r"^###### (.*)$", r"<h6>\1</h6>", new_line)                                                          # Aha! You will be able to handle the other headers here!
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

# Let's test this!
if True:
    old_contents = """
# Title
## 2nd title
<br>
### Also featured: &nbsp; Scripps and Pitzer and Mudd and Pomona
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


# changes italics and bold

import re

def handle_italic_bold(contents):
    """ 
    replace all of the **[stuff]** with <b>[stuff]</b>
    replace all of the *[stuff]* with <em>[stuff]</em>"""

    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        new_line = re.sub(r"\*\*(.*)\*\*", r"<b>\1</b>", line) # bold must go first cuz it is a more specific one
        
        new_line = re.sub(r"__(.*)__", r"<b>\1</b>", new_line) # bold must go first cuz it is a more specific one

        new_line = re.sub(r"\*(.*)\*", r"<em>\1</em>", new_line)  # capture the contents and wrap with <em> and </em>
        
        if not new_line.startswith("<img src="):
            new_line = re.sub(r"_(.*)_", r"<em>\1</em>", new_line)  # capture the contents and wrap with <em> and </em>
            new_line = re.sub(r"_(.*)_", r"<em>\1</em>", new_line)  # capture the contents and wrap with <em> and </em>

        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents



# Let's test this!
if True:
    old_contents = """
*kaizen* **manufacturing**
_goofy_
__weird__"""
    new_contents = handle_italic_bold(old_contents)
    print(new_contents)



# changes strikethrough

import re

def handle_strike(contents):
    """ 
    replace all of the ~~[stuff]~~ with <s>[stuff]</s>"""

    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        new_line = re.sub(r"~~(.*)~~", r"<s>\1</s>", line) 
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents



# Let's test this!
if True:
    old_contents = """
    
ignorance is ~~bliss~~ strength

"""
    new_contents = handle_strike(old_contents)
    print(new_contents)



# URL

import re

def handle_url(contents):
    """changes a hyperlinked text in markdown to markup"""
    NewLines = []
    OldLines = contents.split("\n")

    for line in OldLines:
        # next three lines from ChatGPT
        pattern = r"\[([^\]]+)\]\((https?:\/\/[^\)]+)\)"
        replacement = r'<a href="\2">\1</a>'
        new_line=  re.sub(pattern, replacement, line)
        NewLines.append(new_line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

# Let's test this!
if True:
    old_contents = """
    
[Not a rickroll](https://music.youtube.com/watch?v=RQxOdTJcw1A)

"""
    new_contents = handle_url(old_contents)
    print(new_contents)


# lists
#
import re

def handle_ulists(contents):
    """replaces an unordered list in markdown to markup"""
    NewLines = []
    OldLines = contents.split("\n")

    # this T/F variable determines where we need to put an <ul> or not
    start=0

    for line in OldLines:

        if line.startswith("+") and start==0:
            start=1
            new_line = re.sub(r"^\+(.*)$", r"<ul>\n<li>\n\1\n</li>", line) 
            NewLines.append(new_line)
        elif line.startswith("+") and start==1:
            new_line = re.sub(r"^\+(.*)$", r"<li>\n\1\n</li>", line) 
            NewLines.append(new_line)
        elif not line.startswith("+") and start==1:
            start = 0
            new_line = re.sub(r"^(.*)$", r"</ul>\n\1", line) 
            NewLines.append(new_line)
        else:
            NewLines.append(line)


    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents

# Let's test this!
if True:
    old_contents = """
things that have a better alternative but i use it anyway
+fountain pens
+wired devices
+fully rigid mountain bike


"""
    new_contents = handle_ulists(old_contents)
    print(new_contents)



# Make 42 coffee colored, bold, italic, and underscored

#
import re


def handle_school(contents):
    """
    give each school their school color
    """
    NewLines = []
    OldLines = contents.split("\n")
    for line in OldLines:
        # the appropreite html syntax credit to chatgpt
        line = re.sub(r"CMC", r'<span style="color: #981A31">CMC</span>', line)  # simple substitution
        # CMC spells out their colors, i had to convert it

        line = re.sub(r"HMC", r'<span style="color: #FDB913">HMC</span>', line)  # simple substitution
        # fun fact! HMC offers their school color in proper hex!

        if not line.startswith("<img src="):
            line = re.sub(r"Pitzer", r'<span style="color: #f7941d">Pitzer</span>', line)  # simple substitution
            # i needed to open a pdf for pitzer >:(

        line = re.sub(r"Scripps", r'<span style="color: #396c4f">Scripps</span>', line)  # simple substitution
        # from https://encycolorpedia.com/ how peculiar

        line = re.sub(r"Pomona", r'<span style="color: #0057b8">Pomona</span>', line)  # simple substitution
        NewLines.append(line)

    new_contents = "\n".join(NewLines)   # join with \n characters so it's readable by humans
    return new_contents




                          
                          
                        

# Let's test this!
if True:
    old_contents = """
    ### Also featured: &nbsp; Scripps and Pitzer and Mudd and Pomona
    
    """
    new_contents = handle_school(old_contents) 
    print(new_contents)


# here is a function to change --- to superlong dash
#
import re

def handle_dash(contents):
    """ change --- to superlong dash """
    new_contents = re.sub(r"---", r"<hr>", contents)  # simple substitution
    return new_contents

# Let's test this!
if True:
    old_contents = "### Also featured: &nbsp; Scripps and Pitzer and Mudd and Pomona"
    new_contents = handle_dash(old_contents) 
    print(new_contents)



# Make 42 coffee colored, bold, italic, and underscored

#
import re


def handle_dodds_love_42(contents):
    """
    Make 42 coffee colored, bold, italic, and underscored
    """
    # the appropreite html syntax credit to chatgpt
    new_contents = re.sub(r"42", r'<span style="color: #6F4E37; font-weight: bold; font-style: italic; text-decoration: underline;">42</span>', contents)  # simple substitution
    return new_contents



                          
                          
                        

# Let's test this!
if True:
    old_contents = "42"
    new_contents = handle_dodds_love_42(old_contents) 
    print(new_contents)





<br>
<br>
<h1>Claremont's Colleges - MARKUP version</h1>
<br>
The Claremont Colleges are a <em>consortium</em> of <b>five</b> SoCal institutions. <br>
We list them here.
<br>
<h2>The 5Cs: a list</h2>
<ul>
<li>
 <a href="https://www.pomona.edu/"><span style="color: #0057b8">Pomona</span></a>
</li>
<li>
 <a href="https://www.cmc.edu/"><span style="color: #981A31">CMC</span></a>
</li>
<li>
 <a href="https://www.pitzer.edu/"><span style="color: #f7941d">Pitzer</span></a>
</li>
<li>
 <a href="https://www.scrippscollege.edu/"><span style="color: #396c4f">Scripps</span></a>
</li>
<li>
 <a href="https://www.hmc.edu/"><span style="color: #FDB913">HMC</span></a>
</li>
</ul>
<br>
The above's an <em>unordered</em> list.  <br>
At the 5Cs, we all agree there's <b>no</b> order!
<br>
<hr>
<br>
<h2>Today's featured college: <a href="https://coloradomtn.edu/"><span style="color: #981A31">CMC</span></a></h2>
<br>
<img src="https://ygzm5vgh89zp-u4384.pressidiumcdn.com/wp-content/uploads/2017/06/GWS_campusview_1000x627.jpg" height=160>
<br>
<hr>
<br>
<h3>Also featured: &nbsp; <span style="color: #396c4f">Scripps</span> and <span style="color: #f7941d">Pitzer</span> and Mudd and <span style="color: #0057b8">Pomona</span></h3>
<br>
<img src="https://i0.wp.com/tsl.news/wp-content/uploads/2018/09/scripps.png?w=1430&ssl=1" height=100px> &nbsp; 
<img src="https://upload.wikimedia.org/wikipedia/commons/f/f9/Brant_Clock_Tower%2C_Pitzer_College%2C_2016_%28cropped%29.jpg" height=100px> &nbsp; 
<img src="https://www.hmc.edu/about/wp-content/uploads/sites/2/2020/02/campus-gv.jpg" height=100px> &nbsp;
<img src="https://upload.wikimedia.org/wikipedia/commons/4/46/Smith_Tower_and_the_San_Gabriel_Mountains.jpg" height=100px>
<br>
Are there <em>other</em> schools in Claremont?
<br>
<h3>Claremont destinations</h3>
<ul>
<li>
 <em>Pepo Melo</em>, a fantastic font of fruit!
</li>
<li>
 <b>Starbucks</b>, the center of Claremont's "city," not as good as <span style="color: #396c4f">Scripps</span>'s <em>Motley</em> 
</li>
<li>
 <b><em>Sancho's Tacos</em></b>, the village's newest establishment
</li>
<li>
 <s>In-and-out CS35_Participant_3</s> (not in Claremont, alas, but close! <span style="color: #981A31">CMC</span>-supported!)
</li>
<li>
 <tt><span style="color: #6F4E37; font-weight: bold; font-style: italic; text-decoration: underline;">42</span></tt>nd Street Bagel, an <span style="color: #FDB913">HMC</span> fave, definitely <em>well-numbered</em>
</li>
<li>
 Trader Joe's, providing fuel for the walk back to <span style="color: #f7941d">Pitzer</span> <em>from Trader Joe's</em>
</li>
</ul>
<br>
<hr>
<br>
<h4>Regular Expression Code-of-the-Day </h4>
<tt>import re</tt>               
<tt>pet_statement = re.sub(r'dog', 'cat', 'I <3 dogs')</tt>
<br>
<h4>New Construction of the <s>Day</s> <em>Decade</em>!</h4>
<br>
<img src="https://www.cs.hmc.edu/~dodds/roberts_uc.png" height=150> <br><br>
<br>
<span style="color: #981A31">CMC</span>'s <b><em>Roberts Science Center<em>, also known as </em>"The Rubiks Cube"</em></b> <br>
Currently under construction, under deadline, and undeterred by SoCal sun, or rain... 
<br>
<br><br>
<br>
<br>
<br>



