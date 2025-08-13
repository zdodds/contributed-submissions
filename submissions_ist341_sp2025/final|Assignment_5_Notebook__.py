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

L = list(range(1, 101))
total = sum(L)
print(total)



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





words = ["civic", "sphinx", "ideogram", "aerospace"]

for w in words:
    pattern = [vwl_once(c) for c in w]
    print(w , " ->  ", pattern)


# tests:
print("vwl_all('civic') should be 2 ->", vwl_all('civic'))
print("vwl_all('sphinx') should be 1 ->", vwl_all('sphinx'))
print("vwl_all('ideogram') should be 4 ->", vwl_all('ideogram'))
print("vwl_all('aerospace') should be 5 ->", vwl_all('aerospace'))


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


A = [ n+2 for n in   range(40,42) ]
B = [ 42 for z in [0,1,2] ]
C = [ z for z in [42,42] ]
D = [ s[::-2] for s in ['fungic','y!a!y!'] ]
L = [ [len(w),w] for w in  ['Hi','IST'] ]



# then, see if they work the way you predict...

print(A)
print(B)
print(C)
print(D)
print(L)


def hide_vowels(word):
    """
    Try to guess the words with vowels hidden
    Return the word with vowels replaced by a bullet(●).
    """
    # Replace each vowel (a, e, i, o, u) with "●".
    return ''.join("●" if c in "aeiou" else c for c in word)

def guessing_words(words):
    """For each word in the list, show the word with vowels hidden,
       prompt the user to guess the original word,
       and then tell them if their guess was correct.
    """
    score = 0
    for word in words:
        hidden = hide_vowels(word)

        guess = input("Guess the word:"+ hidden + " \nYour guess: ")

        if guess == word:
            print("Correct!\n")
            score += 1
        else:
            print("Incorrect! The correct word was ", word ,"\n")

    print("Game over! Your final score is " , score,  "out of ", len(words))
words = ["Physician", "Wikipedia", "algorithm"]






# Start the game.
guessing_words(words)




#
# here, [1] create a function that scores punctuation as 1 and non-punctuation as 0, pun_one
#       [2] use the list-comprehension technique to create a punctuation-scoring function, pun_all

# use the above examples as a starting point!


import string

def pun_one(c):
    """
    Returns 1 if character c is a punctuation mark, 0 otherwise.
    """

    pun = "!#$%&'()*+,-./:;<=>?@[\]^_`{|}~"

    return 1 if c in pun else 0

def pun_all(s):
    """
    Returns the total number of punctuation marks in the string s.
    Uses a list comprehension to apply pun_one to each character.
    """
    return sum([pun_one(c) for c in s])



# tests
print("pun_all(PLANKTON) should be 4 <->", pun_all(PLANKTON))
print("pun_all(PATRICK) should be 4 <->", pun_all(PATRICK))
# They're more similar, punctuation-wise than scrabble-wise


#
# The two works of yours and two of another author's ("theirs")
#     These can - and should - be large
#

YOURS1 = """  Abstract
Maintaining student engagement during online video sessions is a critical and challenging aspect of
online education. This paper adopts the Design Science Research (DSR) methodology to explore facial
recognition technology by integrating machine learning and computer vision to develop a real-time system
to monitor and analyze student attention levels and aiming to enhance engagement by introducing
interactive elements that respond to students' attention, thereby improving the overall learning experience.
This innovative solution seeks to address the engagement gap in online educational videos. We found that
maintaining student attention and introducing interactive elements in real-time significantly improves
engagement and the retention of information, thereby contributing positively to the educational process.
Keywords: Engagement – Online Learning – Computer Vision – Machine Learning – Video – adaptive learning –
interactive intervention – Design science research - DSR
1. Introduction
The emergence of the internet can be viewed as the start of the origin of online education. Online education has
seen unprecedented growth in recent years, a trend further accelerated by global events such as the COVID-19
pandemic. In 2022 alone, enrolment in MOOCs (Massive Open Online Courses, available since 2011) jumped to 220
million, a dramatic rise from 40 million the previous year (Shah et al., 2023). These online videos are appealing
primarily due to their benefits; 63% of students appreciate the flexibility they offer, and 30% of students pursue them
in hopes of securing higher-paying jobs (Bay Atlantic University, 2020).
However, while online learning offers numerous advantages, it is also crucial to acknowledge the challenges
presented by online courses, such as potential disengagement, lack of motivation, and even withdrawal. This
disengagement is often most visibly demonstrated through a decline in student attention. A notable decline in attention
is often the first sign of student disengagement, which can significantly undermine the effectiveness of online learning
environments (Sharma et al., 2022). Furthermore, research indicates that engagement with video sessions in Massive
Open Online Courses (MOOCs) averages just six minutes (Geri et al., 2017), underscoring the critical nature of this
issue.
Understanding the factors that influence student engagement becomes pivotal in mitigating the inherent issues of
online courses that contribute to these challenges. Course design, utilized tools, and interactive methods are key
elements that substantially impact the online educational experience and play a crucial role in addressing the
engagement gap common in digital learning environments Gedera et al.,( 2015).
Interactivity in educational content is widely recognized for creating a more engaging and responsive learning
environment. Real-time interaction and feedback, facilitated by learning intervention systems, have been found to
enhance learner engagement (Pérez, 2020). This approach is also identified as a key factor in boosting attention spans
(Cherrett et al., 2009). While many interactive tools have been embedded in online videos, there is a lack of detection
of the precise moment of attention loss to enable real-time interactivity when needed. In a traditional classroom,
teachers may instantly notice any student whose attention has started diminishing and engage the student again,
guaranteeing continual involvement. However, with an online video, this immediate observation and adaptive action
has become difficult. Therefore, integrating real-time, on-demand interaction is crucial for improving attention spans
and enhancing the overall engagement with online educational videos.
The declining attention during online educational videos, coupled with the absence of on-demand interaction, poses
a significant threat to the effectiveness of online learning, potentially compromising academic outcomes and leading
to resource wastage. This paper examines the role of interactive elements in online educational videos, investigating
how they can enhance learning experiences and address the short attention spans often observed in digital learning
contexts.
"""

YOURS2 = """  Introduction
In the animal kingdom, kinship—or the genetic relationship between individuals—plays a
pivotal role in the development of social ties and influencing factors critical for long-term viability.
Kinship is closely linked to the development and stability of social groups, which in turn affect
survival and reproductive success (Arnberg et al. 2015; Madsen et al., 2023). Kinship offers
numerous benefits that enhance the stability, cohesion, and success of social groups formation. For
example, cooperative behaviors among genetically related individuals, such as predator detection
and deterrence, significantly increase group safety by leveraging the principle of safety in numbers
(Shizuka et al., 2014). Kinship also fosters cooperation and reduces intra-group conflict, as
individuals are more likely to prioritize the collective well-being of related group members
(Shizuka et al., 2014). Altruistic behaviors, such as aiding in the care or protection of kin, confer
indirect fitness benefits by ensuring the survival of shared genetic material, even when individuals
do not reproduce directly. Furthermore, kinship groups often exhibit higher levels of trust and
communication, enabling better coordination in critical activities such as foraging, hunting, or
defending territories. Equitable resource sharing is another hallmark of kin-based associations,
with food, shelter, and care being distributed in ways that improve the survival of vulnerable
members, particularly juveniles (Shizuka et al., 2014).
Despite these advantages, it is important to note that social group formation may form
without kinship ties. Groups that are not based on kinship may emerge to access resources, protect
against predators, or cope with environmental challenges. Habitat characteristics and resource
distribution can drive group formation independently of social bonds. Understanding how kinship
interacts with these ecological and social factors is key to explaining the evolution of complex
societies. Group living offers benefits beyond kinship, such as the dilution effect, which reduces
predation risk for individual members of a group as the size increases, regardless of genetic
relationships (Bertram, 1978). These general advantages don't explain why animals consistently
form lasting bonds with specific individuals. These patterns point to other, less understood factors
driving social preferences and group cohesion. Examining how kinship works alongside these
factors is crucial for understanding how complex societies evolved. In the sections that follow, we
attempt to provide context for the current study by beginning with an overview of golden-crowned
sparrows and discuss background research regarding this particular species, kinship ties, and social
group formation.
"""

THEIRS1 = """  by William Shakespeare
                     1
  From fairest creatures we desire increase,
  That thereby beauty's rose might never die,
  But as the riper should by time decease,
  His tender heir might bear his memory:
  But thou contracted to thine own bright eyes,
  Feed'st thy light's flame with self-substantial fuel,
  Making a famine where abundance lies,
  Thy self thy foe, to thy sweet self too cruel:
  Thou that art now the world's fresh ornament,
  And only herald to the gaudy spring,
  Within thine own bud buriest thy content,
  And tender churl mak'st waste in niggarding:
    Pity the world, or else this glutton be,
    To eat the world's due, by the grave and thee.

                     2
  When forty winters shall besiege thy brow,
  And dig deep trenches in thy beauty's field,
  Thy youth's proud livery so gazed on now,
  Will be a tattered weed of small worth held:
  Then being asked, where all thy beauty lies,
  Where all the treasure of thy lusty days;
  To say within thine own deep sunken eyes,
  Were an all-eating shame, and thriftless praise.
  How much more praise deserved thy beauty's use,
  If thou couldst answer 'This fair child of mine
  Shall sum my count, and make my old excuse'
  Proving his beauty by succession thine.
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.

                     3
  Look in thy glass and tell the face thou viewest,
  Now is the time that face should form another,
  Whose fresh repair if now thou not renewest,
  Thou dost beguile the world, unbless some mother.
  For where is she so fair whose uneared womb
  Disdains the tillage of thy husbandry?
  Or who is he so fond will be the tomb,
  Of his self-love to stop posterity?
  Thou art thy mother's glass and she in thee
  Calls back the lovely April of her prime,
  So thou through windows of thine age shalt see,
  Despite of wrinkles this thy golden time.
    But if thou live remembered not to be,
    Die single and thine image dies with thee.

                     4
  Unthrifty loveliness why dost thou spend,
  Upon thy self thy beauty's legacy?
  Nature's bequest gives nothing but doth lend,
  And being frank she lends to those are free:
  Then beauteous niggard why dost thou abuse,
  The bounteous largess given thee to give?
  Profitless usurer why dost thou use
  So great a sum of sums yet canst not live?
  For having traffic with thy self alone,
  Thou of thy self thy sweet self dost deceive,
  Then how when nature calls thee to be gone,
  What acceptable audit canst thou leave?
    Thy unused beauty must be tombed with thee,
    Which used lives th' executor to be.

                     5
  Those hours that with gentle work did frame
  The lovely gaze where every eye doth dwell
  Will play the tyrants to the very same,
  And that unfair which fairly doth excel:
  For never-resting time leads summer on
  To hideous winter and confounds him there,
  Sap checked with frost and lusty leaves quite gone,
  Beauty o'er-snowed and bareness every where:
  Then were not summer's distillation left
  A liquid prisoner pent in walls of glass,
  Beauty's effect with beauty were bereft,
  Nor it nor no remembrance what it was.
    But flowers distilled though they with winter meet,
    Leese but their show, their substance still lives sweet.
"""

THEIRS2 = """  by William Shakespeare
                      30
  When to the sessions of sweet silent thought,
  I summon up remembrance of things past,
  I sigh the lack of many a thing I sought,
  And with old woes new wail my dear time's waste:
  Then can I drown an eye (unused to flow)
  For precious friends hid in death's dateless night,
  And weep afresh love's long since cancelled woe,
  And moan th' expense of many a vanished sight.
  Then can I grieve at grievances foregone,
  And heavily from woe to woe tell o'er
  The sad account of fore-bemoaned moan,
  Which I new pay as if not paid before.
    But if the while I think on thee (dear friend)
    All losses are restored, and sorrows end.

                     31
  Thy bosom is endeared with all hearts,
  Which I by lacking have supposed dead,
  And there reigns love and all love's loving parts,
  And all those friends which I thought buried.
  How many a holy and obsequious tear
  Hath dear religious love stol'n from mine eye,
  As interest of the dead, which now appear,
  But things removed that hidden in thee lie.
  Thou art the grave where buried love doth live,
  Hung with the trophies of my lovers gone,
  Who all their parts of me to thee did give,
  That due of many, now is thine alone.
    Their images I loved, I view in thee,
    And thou (all they) hast all the all of me.

                     32
  If thou survive my well-contented day,
  When that churl death my bones with dust shall cover
  And shalt by fortune once more re-survey
  These poor rude lines of thy deceased lover:
  Compare them with the bett'ring of the time,
  And though they be outstripped by every pen,
  Reserve them for my love, not for their rhyme,
  Exceeded by the height of happier men.
  O then vouchsafe me but this loving thought,
  'Had my friend's Muse grown with this growing age,
  A dearer birth than this his love had brought
  To march in ranks of better equipage:
    But since he died and poets better prove,
    Theirs for their style I'll read, his for his love'.

                     33
  Full many a glorious morning have I seen,
  Flatter the mountain tops with sovereign eye,
  Kissing with golden face the meadows green;
  Gilding pale streams with heavenly alchemy:
  Anon permit the basest clouds to ride,
  With ugly rack on his celestial face,
  And from the forlorn world his visage hide
  Stealing unseen to west with this disgrace:
  Even so my sun one early morn did shine,
  With all triumphant splendour on my brow,
  But out alack, he was but one hour mine,
  The region cloud hath masked him from me now.
    Yet him for this, my love no whit disdaineth,
    Suns of the world may stain, when heaven's sun staineth.

                     34
  Why didst thou promise such a beauteous day,
  And make me travel forth without my cloak,
  To let base clouds o'ertake me in my way,
  Hiding thy brav'ry in their rotten smoke?
  'Tis not enough that through the cloud thou break,
  To dry the rain on my storm-beaten face,
  For no man well of such a salve can speak,
  That heals the wound, and cures not the disgrace:
  Nor can thy shame give physic to my grief,
  Though thou repent, yet I have still the loss,
  Th' offender's sorrow lends but weak relief
  To him that bears the strong offence's cross.
    Ah but those tears are pearl which thy love sheds,
    And they are rich, and ransom all ill deeds.

                     35
  No more be grieved at that which thou hast done,
  Roses have thorns, and silver fountains mud,
  Clouds and eclipses stain both moon and sun,
  And loathsome canker lives in sweetest bud.
  All men make faults, and even I in this,
  Authorizing thy trespass with compare,
  My self corrupting salving thy amiss,
  Excusing thy sins more than thy sins are:
  For to thy sensual fault I bring in sense,
  Thy adverse party is thy advocate,
  And 'gainst my self a lawful plea commence:
  Such civil war is in my love and hate,
    That I an accessary needs must be,
    To that sweet thief which sourly robs from me.

"""


len(THEIRS2)



#
# Here, run your punctuation-comparisons (absolute counts)
#
print("pun_all(YOURS1) ->", pun_all(YOURS1))
print("pun_all(YOURS2) ->", pun_all(YOURS2))
print("pun_all(THEIRS1) ->", pun_all(THEIRS1))
print("pun_all(THEIRS2) ->", pun_all(THEIRS2))


#
# Here, run your punctuation-comparisons (relative, per-character counts)
#
# Compute and print relative punctuation usage (punctuation per character)
len_yours1 = len(YOURS1)
len_yours2 = len(YOURS2)
len_theirs1 = len(THEIRS1)
len_theirs2 = len(THEIRS2)

rel_yours1 = pun_all(YOURS1) / len_yours1
rel_yours2 = pun_all(YOURS2) / len_yours2
rel_theirs1 = pun_all(THEIRS1) / len_theirs1
rel_theirs2 = pun_all(THEIRS2) / len_theirs2

print("\nRelative punctuation usage (punctuation per character):")
print("YOURS1 ->" ,rel_yours1)
print("YOURS2 ->" ,rel_yours2)
print("THEIRS1 ->" ,rel_theirs1)
print("THEIRS2 ->" ,rel_theirs2)





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
print("len(L) is", len(L))
ave = sum(L)/len(L)
print("average is", ave )


# Minimum and maximum values
print("Minimum value in L:", min(L))
print("Maximum value in L:", max(L))

# Count how many times 42 appears
count_42 = L.count(42)
print("Number of 42s in L:", count_42)

# Count how many times 92 appears
count_92 = L.count(92)
print("Number of 92s in L:", count_92)

# Compute and print the average
average = sum(L) / len(L)
print("Average number of wins is:", average)


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




# Repeat the above for a radius of 6, 7, 8, 9, and 10
# It's fast to do:  Copy and paste and edit!!

r_values = [6, 7, 8, 9, 10]


for r in r_values:
    L = [rwalk(r) for _ in range(1000)]
    average = sum(L) / len(L)
    print("For radius= ", r, "the average steps over 1000 trials is" , average)


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
url = "https://www.scrippscollege.edu/"          # another possible site...
url = "https://www.pitzer.edu/"                  # another possible site...
url = "https://www.cmc.edu/"                     # and another!
url = "https://www.cgu.edu/"                       # Yay CGU!
result = requests.get(url)
print(f"result is {result}")        # Question: is the result a "200" response?! Yes


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

number_in_space = d['number']
print("Number of astronauts in space:", number_in_space)



#    (2) to extract the name "Sunita Williams" from the dictionary d
for person in d['people']:
    if person['name'] == "Sunita Williams":
        print("Found Sunita Williams!", person['craft'])
        break



# use this cell - based on the example above - to share your solutions to the Astronaut challenges...

for person in d['people']:
    if person['name'] == "Jeanette Epps":
        print("Found Jeanette Epps!")
        print("She is on craft:", person['craft'])
        break



# Dad Joke API:

import requests

def get_random_dad_joke():

    url = "https://icanhazdadjoke.com/"
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data.get("joke", "No joke found!")
    else:
        return f"Error: {response.status_code}"

dad_joke = get_random_dad_joke()
print("Random Dad Joke:", dad_joke, " 😂😂😂")



# Web Scraping: NASA Astronomy Picture of the Day

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_apod_page():
    """
    Scrapes the NASA APOD page (https://apod.nasa.gov/apod/)
    to retrieve the page title, main image URL (if present),
    and paragraphs of text.
    """
    apod_url = "https://apod.nasa.gov/apod/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        )
    }

    response = requests.get(apod_url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # 1) Get the page <title>
        title_tag = soup.find("title")
        if title_tag:
            print("Page Title:", title_tag.text)
        else:
            print("No <title> found on the page.")

        # 2) Find the main APOD image
        image_tag = soup.find("img")
        if image_tag:
            img_src = image_tag.get("src")
            if img_src:
                full_img_url = urljoin(apod_url, img_src)
                print("Main Image URL:", full_img_url)
            else:
                print("Image tag found, but no 'src' attribute!")
        else:
            print("No <img> tag found on the page.")

        # 3) Extract paragraphs (e.g., explanation text)
        paragraphs = soup.find_all("p")
        if paragraphs:
            print("\nParagraphs from the page:")
            for idx, p in enumerate(paragraphs):
                print(f"Paragraph {idx}:\n{p.get_text(strip=True)}\n")
        else:
            print("No <p> paragraphs found on the page.")
    else:
        print(f"Error fetching APOD page: {response.status_code}")

# Run the scraper
scrape_apod_page()






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





