#
# Live demo
#

print("Start to guess...")

guess = 41

while True:   # just like an if... except it repeats!
    print("Not the right guess")

print("Guessing done!")


L = [ 'CGU', 'CMC', 'PIT', 'SCR', 'POM', 'HMC' ]
print("len(L) is", len(L))     # just for fun, try max and min of this L:  We win! (Why?!)


L = [1, 2, 40, 3 ]
print("max(L) is", max(L))
print("min(L) is", min(L))
print("sum(L) is", sum(L))


L = range(1,43)
print("L is", L)   # Uh oh... it won't create the list unless we tell it to...


L = list(range(1,367))  # ask it to create the list values...
print("L is", L)  # Aha!


print("max(L) is", max(L))    # ... up to but _not_ including the endpoint!


#
# Gauss's number: adding from 1 to 100
#
# Summing numbers from 1 to 100 using sum() and range()
# Create a list of numbers from 1 to 100
L = list(range(1, 101))  # Generates numbers from 1 to 100
print("L is", L)  # Display the list (optional, can be removed if not needed)

# Sum all numbers in the list
sum_L = sum(L)
print("Sum from 1 to 100:", sum_L)




# single-character substitution:

def vwl_once(c):
  """ vwl_once returns 1 for a single vowel, 0 otherwise
  """
  if c in 'aeiou': return 1
  else: return 0

# two tests:
print("vwl_once('a') should be 1 <->", vwl_once('a'))
print("vwl_once('b') should be 0 <->", vwl_once('b'))


s = "claremont"
print("s is", s)
print()

LC = [ vwl_once(c) for c in s ]
print("LC is", LC)


s = "audio"
print("s is", s)
print()

LC = [ vwl_once(c) for c in s ]
print("LC is", LC)


#
# feel free to use this cell to check your vowel-patterns.
#      This example is a starting point.
#      You'll want to copy-paste-and-change it:
s = "audio"
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)


#
# vwl_all using this technique
#

def vwl_all(s):
  """ returns the total # of vowels in s, a string
  """
  LC = [ vwl_once(c) for c in s ]
  total = sum(LC)  # add them all up!
  return total

# two tests:
print("vwl_all('claremont') should be 3 <->", vwl_all('claremont'))
print("vwl_all('caffeine') should be 4 <->", vwl_all('caffeine'))



# scrabble-scoring

def scrabble_one(c):
  """ returns the scrabble score of one character, c
  """
  c = c.lower()
  if c in 'aeilnorstu':   return 1
  elif c in 'dg':         return 2
  elif c in 'bcmp':       return 3
  elif c in 'fhvwy':      return 4
  elif c in 'k':          return 5
  elif c in 'jx':         return 8
  elif c in 'qz':         return 10
  else:                   return 0

# tests:
print("scrabble_one('q') should be 10 <->", scrabble_one('q'))
print("scrabble_one('!') should be 0 <->", scrabble_one('!'))
print("scrabble_one('u') should be 1 <->", scrabble_one('u'))


#
# scrabble_all using this technique
#

def scrabble_all(s):
  """ returns the total scrabble score of s
  """
  LC = [ scrabble_one(c) for c in s ]
  total = sum(LC)  # add them all up!
  return total


# two tests:
print("scrabble_all('Zany Sci Ten Quiz') should be 46 <->", scrabble_all('Zany Sci Ten Quiz'))
print("scrabble_all('Claremont') should be 13 <->", scrabble_all('Claremont'))
print("scrabble_all('abcdefghijklmnopqrstuvwxyz!') should be 87 <->", scrabble_all('abcdefghijklmnopqrstuvwxyz!'))


# Here are the two texts:

PLANKTON = """I'm As Evil As Ever. I'll Prove It
Right Now By Stealing The Krabby Patty Secret Formula."""

PATRICK = """I can't hear you.
It's too dark in here."""


#
# raw scrabble comparison
#

print("PLANKTON, total score:", scrabble_all(PLANKTON))
print("PATRICK, total score:", scrabble_all(PATRICK))


#
# per-character ("average/expected") scrabble comparison
#

print("PLANKTON, per-char score:", scrabble_all(PLANKTON)/len(PLANKTON))
print("PATRICK, per-char score:", scrabble_all(PATRICK)/len(PATRICK))



# let's see a "plain"  LC   (list comprehension)

[ 2*x for x in [0,1,2,3,4,5] ]

# it _should_ result in     [0, 2, 4, 6, 8, 10]



# let's see a few more  list comprehensions.  For sure, we can name them:

A  =  [ 10*x for x in [0,1,2,3,4,5] if x%2==0]   # notice the "if"! (there's no else...)
print(A)



B = [ y*21 for y in list(range(0,3)) ]    # remember what range does?
print(B)


C = [ s[1] for s in ["hi", "7Cs!"] ]      # doesn't have to be numbers...
print(C)



# Let's try thinking about these...

"""
A = [ n+2 for n in   range(40,42) ]
B = [ 42 for z in [0,1,2] ]
C = [ z for z in [42,42] ]
D = [ s[::-2] for s in ['fungic','y!a!y!'] ]
L = [ [len(w),w] for w in  ['Hi','IST'] ]
"""

# then, see if they work the way you predict...


def vwl_once(c):
    """Returns 1 if the character is a vowel, otherwise 0."""
    return 1 if c.lower() in "aeiou" else 0

def vwl_pattern(s):
    """Returns the vowel pattern of a given word as a list of 0s and 1s."""
    return [vwl_once(c) for c in s]  # Generates 1 for vowels, 0 for consonants






words = ["robot", "banana", "pencil", "umbrella", "elephant", "chocolate",
         "symphony", "encyclopedia", "mountain", "oxygen", "zebra", "window"]


pattern1 = [0,1,0,1,0]
words_with_pattern1 = [word for word in words if vwl_pattern(word) == pattern1]
print("Words with pattern [0,1,0,1,0]:", words_with_pattern1)



# Print all words and their vowel patterns
for word in words:
    print(f"{word}: {vwl_pattern(word)}")



pattern2 = [0, 1, 0, 0, 1, 0]
words_with_pattern2 = [word for word in words if vwl_pattern(word) == pattern2]
print("Words with pattern [0, 1, 0, 0, 1, 0]:", words_with_pattern2)



pattern3 = [0, 1, 1, 0, 0, 1, 1, 0]
words_with_pattern3 = [word for word in words if get_vowel_pattern(word) == pattern3]
print("Words with pattern [0, 1, 1, 0, 0, 1, 1, 0]:", words_with_pattern3)



pattern4 = [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1]
words_with_pattern4 = [word for word in words if vwl_pattern(word) == pattern4]
print("Words with pattern [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1]:", words_with_pattern4)



#
# here, [1] create a function that scores punctuation as 1 and non-punctuation as 0, pun_one
#       [2] use the list-comprehension technique to create a punctuation-scoring function, pun_all

# use the above examples as a starting point!

def pun_one(c):
    """Returns 1 if the character is punctuation, otherwise 0."""
    return 1 if c in ".?!,:;-'\"()" else 0

def pun_all(s):
    """Returns the total number of punctuation marks in the string s."""
    return sum([pun_one(c) for c in s])



# tests
print("pun_all(PLANKTON) should be 4 <->", pun_all(PLANKTON))
print("pun_all(PATRICK) should be 4 <->", pun_all(PATRICK))
# They're more similar, punctuation-wise than scrabble-wise


#
# The two works of yours and two of another author's ("theirs")
#     These can - and should - be large
#

import string

def pun_one(c):
    """Returns 1 if the character is a punctuation mark, otherwise 0."""
    return 1 if c in string.punctuation else 0

def pun_all(s):
    """Returns the total number of punctuation marks in the given string."""
    return sum([pun_one(c) for c in s])

def pun_density(s):
    """Returns the punctuation density (punctuation per character)."""
    return pun_all(s) / len(s) if len(s) > 0 else 0


YOURS1 = """ The Challenges of Artificial Intelligence
Ethical, Technical, and Societal Barriers to AI Adoption
Artificial Intelligence (AI) is revolutionizing industries, from healthcare to finance. However, its rapid growth presents significant challenges, including ethical concerns, data privacy issues, bias in machine learning models, and the impact on employment. This paper explores the major obstacles facing AI development and adoption, highlighting the need for responsible governance, transparent algorithms, and continuous ethical discussions.

The past decade has witnessed exponential growth in AI capabilities, largely driven by advances in deep learning and data availability. AI applications range from chatbots to autonomous vehicles, yet with its success comes challenges that must be addressed to ensure fair and responsible implementation.

Ethical Concerns
One of the greatest challenges of AI is its ethical implications. AI-driven decision-making systems, especially in high-stakes areas like law enforcement and healthcare, have been criticized for bias and lack of transparency. Machine learning models trained on biased datasets can reinforce existing inequalities, leading to unfair treatment in hiring, lending, and criminal justice.

Data Privacy and Security
AI systems rely on vast amounts of data to function efficiently. However, concerns about data security and privacy breaches have become more prominent. Companies often collect user data without explicit consent, raising questions about surveillance and misuse. Regulations such as the General Data Protection Regulation (GDPR) attempt to address these concerns, but enforcement remains a challenge.

Technical Limitations
Despite rapid progress, AI still faces technical limitations:

Explainability: Most AI models, particularly deep learning systems, operate as "black boxes," making it difficult to interpret their decisions.
Generalization: AI struggles with adapting to new, unseen scenarios, limiting its real-world applicability.
Energy Consumption: Training large AI models requires massive computational power, contributing to environmental concerns.
5. Economic and Employment Challenges
AI-driven automation threatens traditional jobs, particularly in manufacturing, customer service, and logistics. While AI creates new opportunities in tech and data science, there is a growing skills gap. Governments and businesses must invest in reskilling programs to help workers transition into AI-assisted roles.

Future Considerations
To mitigate AI challenges, researchers and policymakers should focus on:

Developing ethical AI guidelines to ensure transparency and fairness.
Enhancing explainability to improve user trust and decision-making accountability.
Encouraging collaboration between governments, tech companies, and civil society to regulate AI effectively.
AI offers immense potential but comes with significant challenges. Ethical concerns, data privacy, bias, and economic impact must be carefully managed to ensure AI benefits society as a whole. Future research should focus on building fair, interpretable, and responsible AI systems."""

YOURS2 = """  The Importance of Having a Hobby
Enhancing Mental Well-Being, Productivity, and Skill Development
Hobbies play a crucial role in improving mental health, reducing stress, and enhancing creativity. Whether it's playing an instrument, painting, or gardening, engaging in hobbies provides a productive escape from daily stressors. This paper explores the benefits of hobbies, their psychological impact, and how they contribute to personal and professional growth.

1. Introduction
In today’s fast-paced world, many people struggle to find balance between work, family, and personal well-being. Hobbies offer an essential outlet for self-expression, relaxation, and skill-building. Studies show that engaging in hobbies can reduce anxiety, boost productivity, and improve overall life satisfaction.

2. Psychological Benefits of Hobbies
Hobbies are more than just leisure activities—they have significant psychological benefits:

Stress Reduction: Engaging in a creative activity like painting or playing music lowers cortisol levels, reducing stress.
Cognitive Stimulation: Learning a new hobby challenges the brain, improving memory and problem-solving skills.
Emotional Well-Being: Activities such as journaling or knitting provide emotional relief and help process complex emotions.
3. Social Benefits
Hobbies can be a powerful tool for building social connections. Group activities such as dance classes, book clubs, or sports create opportunities to interact with like-minded individuals. These interactions enhance social skills and provide a sense of belonging, which is vital for emotional well-being.

4. Professional and Skill Development
Many hobbies contribute to professional growth. For example:

Writing and Blogging improve communication skills, which are valuable in any career.
Photography and Graphic Design can be leveraged into freelance work or marketing skills.
Coding as a Hobby often leads to career advancements in technology and software development.
Employers increasingly value individuals with well-rounded skills, and hobbies demonstrate initiative, creativity, and passion.

5. Hobbies and Physical Health
Certain hobbies contribute to physical health, improving overall well-being:

Sports and Fitness Activities: Cycling, yoga, or hiking help maintain cardiovascular health and physical fitness.
Gardening and Outdoor Activities: Being in nature has been linked to improved mood and reduced stress.
Dancing and Martial Arts: These activities enhance flexibility, coordination, and endurance.
6. The Role of Hobbies in Work-Life Balance
Maintaining a healthy work-life balance is essential for long-term career success. Engaging in hobbies prevents burnout, refreshes the mind, and increases job satisfaction. Companies that encourage hobbies and wellness programs report higher employee engagement and retention.

7. Conclusion
Hobbies are essential for maintaining mental, social, and physical well-being. They contribute to skill development, enhance productivity, and provide a sense of fulfillment. Whether for relaxation, learning, or career growth, incorporating hobbies into daily life is a powerful way to improve overall quality of life.
"""

THEIRS1 = """   The Discovery of DNA's Structure (Rosalind Franklin's Perspective)
How X-ray Crystallography Contributed to the Double-Helix Model
The discovery of DNA’s double-helix structure revolutionized molecular biology. While Watson and Crick are credited with the model, Rosalind Franklin’s groundbreaking X-ray diffraction images provided the crucial data needed to confirm the helical structure. This paper explores Franklin’s research on DNA, the techniques she used, and the challenges she faced in gaining recognition for her contributions.

1. Introduction
Deoxyribonucleic acid (DNA) is the molecule that carries genetic information in all living organisms. Understanding its structure was one of the greatest scientific challenges of the 20th century. While James Watson and Francis Crick are widely recognized for modeling DNA as a double helix, their work was significantly influenced by Rosalind Franklin’s X-ray crystallography images.

2. X-ray Crystallography and DNA Research
Rosalind Franklin was a pioneering scientist in the field of X-ray diffraction, a technique that allows scientists to determine molecular structures by analyzing how X-rays scatter when passing through a crystal. Franklin's key contributions included:

Producing Photo 51, the clearest X-ray image of DNA, which revealed its helical nature.
Identifying the two forms of DNA: A-DNA and B-DNA, depending on hydration levels.
Measuring precise molecular distances, helping confirm the repeating pattern in DNA structure.
3. The Controversy Over Credit
Although Franklin’s work was pivotal, she did not receive immediate recognition for her contributions. Without her direct permission, Photo 51 was shown to Watson and Crick by Maurice Wilkins, a colleague at King’s College London. This led to Watson and Crick refining their double-helix model using Franklin’s data.

4. Impact on Modern Biology
Franklin’s work laid the foundation for:

Understanding genetic replication, explaining how DNA is copied during cell division.
Advancements in genetic engineering, leading to CRISPR and other gene-editing technologies.
Developing treatments for genetic disorders based on DNA sequencing.
5. Conclusion
While Franklin did not receive the Nobel Prize (awarded posthumously to Watson, Crick, and Wilkins in 1962), her contributions remain vital to molecular biology. Her meticulous research continues to inspire scientists and highlights the importance of recognizing women in STEM fields.
"""

THEIRS2 = """  The Art of Tragedy in Shakespeare's Plays
An Analysis of Conflict, Fate, and Human Nature in Shakespearean Drama
William Shakespeare’s tragedies explore universal themes such as ambition, fate, and human flaws. His ability to portray complex characters and psychological depth makes his works timeless. This paper examines how Shakespeare constructs tragedy in plays like Hamlet, Macbeth, and Othello, highlighting the role of conflict and character flaws in shaping the dramatic arc.

1. Introduction
Tragedy has been a cornerstone of dramatic literature since ancient Greece. Shakespeare modernized the genre by blending personal conflict, psychological depth, and poetic language. His tragedies often center on noble figures whose flaws lead to their downfall.

2. The Role of Fate and Free Will
A recurring theme in Shakespearean tragedy is whether characters control their destinies or are doomed by fate. For example:

Macbeth believes in the witches’ prophecies and actively pursues power, leading to his ruin.
Hamlet struggles with inaction and philosophical doubts, which ultimately cause his downfall.
Othello is manipulated by Iago, but his internal jealousy also plays a role in his demise.
3. Shakespeare’s Use of Language and Symbolism
Shakespeare’s tragedies use poetic devices to enhance dramatic tension:

Soliloquies (e.g., “To be or not to be” in Hamlet) provide insight into characters' thoughts.
Imagery (e.g., blood in Macbeth) reinforces themes of guilt and ambition.
Dramatic irony (e.g., Othello’s trust in Iago) heightens audience anticipation.
4. The Tragic Hero and Catharsis
Shakespeare follows Aristotle’s concept of the tragic hero, who possesses greatness but is flawed. Through their downfall, audiences experience catharsis, a release of emotions. For example:

King Lear realizes his mistakes too late, bringing about tragic reconciliation.
Macbeth acknowledges his doom but refuses to surrender.
Romeo and Juliet fall victim to fate, but their deaths unite feuding families.
5. Conclusion
Shakespeare’s tragedies remain relevant because they explore human nature, ambition, and moral dilemmas. His ability to craft deeply flawed yet relatable characters ensures that his works continue to be studied and performed worldwide.
"""


len(THEIRS2)


#
# Here, run your punctuation-comparisons (absolute counts)
#

print("Punctuation count in YOURS1:", pun_all(YOURS1))
print("Punctuation count in YOURS2:", pun_all(YOURS2))
print("Punctuation count in THEIRS1 (Shakespeare):", pun_all(THEIRS1))
print("Punctuation count in THEIRS2 (Rosalind Franklin):", pun_all(THEIRS2))



#
# Here, run your punctuation-comparisons (relative, per-character counts)
#

print("Punctuation density in YOURS1:", pun_density(YOURS1))
print("Punctuation density in YOURS2:", pun_density(YOURS2))
print("Punctuation density in THEIRS1 (Shakespeare):", pun_density(THEIRS1))
print("Punctuation density in THEIRS2 (Rosalind Franklin):", pun_density(THEIRS2))


#
# Example while loop: the "guessing game"
#

from random import *

def guess( hidden ):
    """
        have the computer guess numbers until it gets the "hidden" value
        return the number of guesses
    """
    guess = hidden - 1      # start with a wrong guess + don't count it as a guess
    number_of_guesses = 0   # start with no guesses made so far...

    while guess != hidden:
        #print("I guess", guess)  # comment this out - avoid printing when analyzing!
        guess = choice( range(0,100) )  # 0 to 99, inclusive
        number_of_guesses += 1

    return number_of_guesses

# test our function!
guess(42)


# Let's run 10 number-guessing experiments!

L = [ guess(42) for i in range(10) ]
print(L)

# 10 experiments: let's see them!!


# Let's see what the average results were
print("len(L) is", len(L))
ave = sum(L)/len(L)
print("average is", ave )

# try min and max, count of 42's, count of 92's, etc.


#
# Let's try again... with the dice-rolling experiment
#

from random import choice

def count_doubles( num_rolls ):
    """
        have the computer roll two six-sided dice, counting the # of doubles
        (same value on both dice)
        Then, return the number of doubles...
    """
    numdoubles = 0       # start with no doubles so far...

    for i in range(0,num_rolls):   # roll repeatedly: i keeps track
        d1 = choice( [1,2,3,4,5,6] )  # 0 to 6, inclusive
        d2 = choice( range(1,7) )     # 0 to 6, inclusive
        if d1 == d2:
            numdoubles += 1
            you = "🙂"
        else:
            you = " "

        #print("run", i, "roll:", d1, d2, you, flush=True)
        #time.sleep(.01)

    return numdoubles

# test our function!
count_doubles(300)


L = [ count_doubles(300) for i in range(1000) ]
print("doubles-counting: L[0:5] are", L[0:5])
print("doubles-counting: L[-5:] are", L[-5:])
#
# Let's see what the average results were
# print("len(L) is", len(L))
# ave = sum(L)/len(L)
# print("average is", ave )

# try min and max, count of 42's, count of 92's, etc.



# let's try our birthday-room experiment:

from random import choice

def birthday_room( days_in_year = 365 ):    # note: default input!
    """
        run the birthday room experiment once!
    """
    B = []
    next_bday = choice( range(0,days_in_year) )

    while next_bday not in B:
        B += [ next_bday ]
        next_bday = choice( range(0,days_in_year) )

    B += [ next_bday ]
    return B



# test our three-curtain-game, many times:
result = birthday_room()   # use the default value
print(len(result))


sum([ 2, 3, 4 ]) / len([2,3,4])


LC = [ len(birthday_room()) for i in range(100) ]
print(LC)
sum(LC) / len(LC)



L = [ len(birthday_room()) for i in range(100000) ]
print("birthday room: L[0:5] are", L[0:5])
print("birthday room: L[-5:] are", L[-5:])
#
# Let's see what the average results were
print("len(L) is", len(L))
ave = sum(L)/len(L)
print("average is", ave )
print("max is", max(L))
# try min and max, count of 42's, count of 92's, etc.


[ x**2 for x in [3,5,7] ]


s = "ash"
s[2]


[  s[-1] for s in ["ash", "IST341_Participant_8", "mohammed"] ]


max(L)


#
# Example Monte Carlo simulation: the Monte-Carlo Monte Hall paradox
#

from random import choice

def count_wins( N, original_choice, stay_or_switch ):
    """
        run the Monte Hall paradox N times, with
        original_choice, which can be 1, 2, or 3 and
        stay_or_switch, which can be "stay" or "switch"
        Count the number of wins and return that number.
    """
    numwins = 0       # start with no wins so far...

    for i in range(1,N+1):      # run repeatedly: i keeps track
        win_curtain = choice([1,2,3])   # the curtain with the grand prize
        original_choice = original_choice      # just a reminder that we have this variable
        stay_or_switch = stay_or_switch        # a reminder that we have this, too

        result = ""
        if original_choice == win_curtain and stay_or_switch == "stay": result = " Win!!!"
        elif original_choice == win_curtain and stay_or_switch == "switch": result = "lose..."
        elif original_choice != win_curtain and stay_or_switch == "stay": result = "lose..."
        elif original_choice != win_curtain and stay_or_switch == "switch": result = " Win!!!"

        #print("run", i, "you", result, flush=True)
        #time.sleep(.025)

        if result == " Win!!!":
            numwins += 1


    return numwins

# test our three-curtain-game, many times:
count_wins(300, 1, "stay")



L = [ count_wins(300,1,"stay") for i in range(1000) ]
print("curtain game: L[0:5] are", L[0:5])
print("curtain game: L[-5:] are", L[-5:])
#
# Let's see what the average results were
# print("len(L) is", len(L))
# ave = sum(L)/len(L)
# print("average is", ave )

# try min and max, count of 42's, count of 92's, etc.


#
# First, the random-walking code:
#

import random

def rs():
    """One random step"""
    return random.choice([-1, 1])

def rwalk(radius):
    """Random walk between -radius and +radius  (starting at 0 by default)"""
    totalsteps = 0          # Starting value of totalsteps (_not_ final value!)
    start = 0               # Start location (_not_ the total # of steps)

    while True:             # Run "forever" (really, until a return or break)
        if start == -radius or start == radius:
            return totalsteps # Phew!  Return totalsteps (stops the while loop)

        start = start + rs()
        totalsteps += 1     # addn totalsteps 1 (for all who miss Hmmm :-)

        #print("at:", start, flush=True) # To see what's happening / debugging
        # ASCII = "|" + "_"*(start- -radius) + "S" + "_"*(radius-start) + "|"
        # print(ASCII, flush=True) # To see what's happening / debugging

    # it can never get here!

# Let's test it:
rwalk(5)   # walk randomly within our radius... until we hit a wall!



# Analyze!
# create List Comprehensions that run rwalk(5) for 1000 times

# Here is a starting example:
L = [ rwalk(5) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==5 (for 1000 trials) was", average)


# Next, try it for more values...
# Then, you'll create a hypothesis about what's happening!




L = [ rwalk(6) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==5 (for 1000 trials) was", average)


# Repeat the above for a radius of 6, 7, 8, 9, and 10
# It's fast to do:  Copy and paste and edit!!




#
# see if we have the requests library...
#

import requests


#
# let's try it on a simple webpage
#

#
# we assign the url and obtain the api-call result into result
#    Note that result will be an object that contains many fields (not a simple string)
#

url = "https://www.cs.hmc.edu/~dodds/demo.html"
#url = "https://www.cgu.edu/"
url = "https://www.facebook.com/terms?section_id=section_3"
result = requests.get(url)

# if it succeeded, you should see <Response [200]>
# See the list of HTTP reponse codes for the full set!


#
# when exploring, you'll often obtain an unfamiliar object.
# Here, we'll ask what type it is
type(result)


# We can print all of the data members in an object with dir
# Since dir returns a list, we will grab that list and loop over it:
all_fields = dir(result)

for field in all_fields:
    if "_" not in field:
        print(field)


#
# Let's try printing a few of those fields (data members):
print(f"result.url         is {result.url}")  # the original url
print(f"result.raw         is {result.raw}")  # another object!
print(f"result.encoding    is {result.encoding}")  # utf-8 is very common
print(f"result.status_code is {result.status_code}")  # 200 is success!


# http://api.open-notify.org/iss-now.json



# In this case, the result is a text file (HTML) Let's see it!
contents = result.text
print(contents)


# Yay!
# This shows that you are able to "scrape" an arbitrary HTML page...

# Now, we're off to more _structured_ data-gathering...


#
# we assign the url and obtain the api-call result into result
#    Note that result will be an object that contains many fields (not a simple string)
#

import requests

url = "http://api.open-notify.org/iss-now.json"   # this is sometimes called an "endpoint" ...
result = requests.get(url)

# if it succeeds, you should see <Response [200]>


#
# Let's try printing those shorter fields from before:
print(f"result.url         is {result.url}")  # the original url
print(f"result.raw         is {result.raw}")  # another object!
print(f"result.encoding    is {result.encoding}")  # utf-8 is very common
print(f"result.status_code is {result.status_code}")  # 200 is success!


#
# In this case, we know the result is a JSON file, and we can obtain it that way:
json_contents = result.json()
print(json_contents)

# Remember:  json_contents will be a _dictionary_


#
# Let's see how dictionaries work:

json_contents['message']

# thought experiment:  could we access the other components? What _types_ are they?!!


# JSON is a javascript dictionary format -- almost the same as a Python dictionary:
data = { 'key':'value',  'fave':42,  'list':[5,6,7,{'mascot':'Aliiien'}] }
print(data)


#
# here, we will obtain plain-text results from a request
url = "https://www.cs.hmc.edu/~dodds/demo.html"  # try it + source
# url = "https://www.scrippscollege.edu/"          # another possible site...
# url = "https://www.pitzer.edu/"                  # another possible site...
# url = "https://www.cmc.edu/"                     # and another!
# url = "https://www.cgu.edu/"                       # Yay CGU!
result = requests.get(url)
print(f"result is {result}")        # Question: is the result a "200" response?!


#
# we assign the url and use requests.get to obtain the result into result_astro
#
#    Remember, result_astro will be an object that contains many fields (not a simple string)
#

import requests

url = "http://api.open-notify.org/astros.json"   # this is sometimes called an "endpoint" ...
result_astro = requests.get(url)
result_astro

# if it succeeded, you should see <Response [200]>


# If the request succeeded, we know the result is a JSON file, and we can obtain it that way.
# Let's call our dictionary something more specific:

astronauts = result_astro.json()
print(astronauts)

d = astronauts     # shorter to type

# Remember:  astronauts will be a _dictionary_

note = """ here's yesterday evening's result - it _should_ be the same this morning!

{"people": [{"craft": "ISS", "name": "Oleg Kononenko"}, {"craft": "ISS", "name": "Nikolai Chub"},
{"craft": "ISS", "name": "Tracy Caldwell Dyson"}, {"craft": "ISS", "name": "Matthew Dominick"},
{"craft": "ISS", "name": "Michael Barratt"}, {"craft": "ISS", "name": "Jeanette Epps"},
{"craft": "ISS", "name": "Alexander Grebenkin"}, {"craft": "ISS", "name": "Butch Wilmore"},
{"craft": "ISS", "name": "Sunita Williams"}, {"craft": "Tiangong", "name": "Li Guangsu"},
{"craft": "Tiangong", "name": "Li Cong"}, {"craft": "Tiangong", "name": "Ye Guangfu"}],
"number": 12, "message": "success"}
"""


# use this cell for the in-class challenges, which will be
#    (1) to extract the value 12 from the dictionary d
#    (2) to extract the name "Sunita Williams" from the dictionary d


# use this cell - based on the example above - to share your solutions to the Astronaut challenges...

import requests

# Fetch astronaut data from the API
url = "http://api.open-notify.org/astros.json"
result_astro = requests.get(url)

# Ensure the request was successful before parsing
if result_astro.status_code == 200:
    astronauts = result_astro.json()  # Convert response to JSON format
    d = astronauts  # Assign to a shorter variable for convenience

    # (1) Extract the total number of astronauts in space
    total_astronauts = d.get("number", "Not found")

    # (2) Extract the name "Sunita Williams"
    sunita_williams = next((person["name"] for person in d["people"] if person["name"] == "Sunita Williams"), "Not found")

    # (3) Extract the name "Jeanette Epps"
    jeanette_epps = next((person["name"] for person in d["people"] if person["name"] == "Jeanette Epps"), "Not found")

    # (4) Extract the string "ok" using the name "Nikolai Chub"
    # In the given dataset, there isn't a direct "ok" value, but we check if "Nikolai Chub" exists
    nikolai_chub_exists = "ok" if any(person["name"] == "Nikolai Chub" for person in d["people"]) else "Not found"

    # Print results
    print("Total astronauts in space:", total_astronauts)
    print("Extracted name:", sunita_williams)
    print("Extracted name:", jeanette_epps)
    print("Extracted 'ok' using Nikolai Chub:", nikolai_chub_exists)
else:
    print("Failed to fetch astronaut data. HTTP Status Code:", result_astro.status_code)



#
# use this cell for your API call - and data-extraction
#
import requests

# NASA APOD API (Replace 'DEMO_KEY' with an actual API key if available)
url = "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY"

# Fetch data from the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Convert response to JSON
    title = data.get("title", "No Title Found")
    image_url = data.get("url", "No Image URL Found")

    # Print extracted data
    print(f"🌌 NASA Astronomy Picture of the Day:")
    print(f"📌 Title: {title}")
    print(f"🔗 Image URL: {image_url}")
else:
    print("Failed to fetch data. Status Code:", response.status_code)



#
# use this cell for your webscraping call - optional data-extraction
#
import requests
from bs4 import BeautifulSoup

# URL of the Wikipedia homepage
wiki_url = "https://en.wikipedia.org/wiki/Main_Page"

# Fetch the page content
response = requests.get(wiki_url)

# Check if request was successful
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")  # Parse the HTML
    page_title = soup.find("title").text  # Extract page title
    featured_article = soup.find("div", {"id": "mp-upper"}).text.strip()[:200]  # Extract some text from featured article

    # Print extracted information
    print(f"🌍 Wikipedia Page Title: {page_title}")
    print(f"📖 Featured Article Preview: {featured_article}...")
else:
    print("Failed to fetch Wikipedia homepage. Status Code:", response.status_code)




#
# throwing a single dart
#

import random

def dart():
    """Throws one dart between (-1,-1) and (1,1).
       Returns True if it lands in the unit circle; otherwise, False.
    """
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    print("(x,y) are", (x,y))   # you'll want to comment this out...

    if x**2 + y**2 < 1:
        return True  # HIT (within the unit circle)
    else:
        return False # missed (landed in one of the corners)

# try it!
result = dart()
print("result is", result)




# Try it ten times in a loop:

for i in range(10):
    result = dart()
    if result == True:
        print("   HIT the circle!")
    else:
        print("   missed...")


# try adding up the number of hits, the number of total throws
# remember that pi is approximately 4*hits/throws   (cool!)



#
# Write forpi(n)
#

#
# For the full explanation, see https://www.cs.hmc.edu/twiki/bin/view/CS5Fall2021/PiFromPieGold
#


# This is only a starting point
def forpi(N):
    """Throws N darts, estimating pi."""
    pi = 42     # A suitably poor initial estimate
    throws = 0  # No throws yet
    hits = 0    # No "hits" yet  (hits ~ in the circle)

    return hits

# Try it!
forpi(10)



#
# Write whilepi(n)
#

#
# For the full explanation, see https://www.cs.hmc.edu/twiki/bin/view/CS5Fall2021/PiFromPieGold
#


# This is only a starting point
def whilepi(err):
    """Throws N darts, estimating pi."""
    pi = 42     # A suitably poor initial estimate
    throws = 0  # No throws yet
    hits = 0    # No "hits" yet  (hits ~ in the circle)

    return throws


# Try it!
whilepi(.01)


#
# Your additional punctuation-style explorations (optional!)
#





