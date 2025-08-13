#
# Live Demo
#

print ("Start to guess...")

guess = 41

if guess != 42:
  print("Not the right guess")
  guess=input("GuessAgain:")
  guess=int(guess)

print("Guessing done!")


L = [ 'CGU', 'CMC', 'PIT', 'SCR', 'POM', 'HMC' ]
print("len(L) is", len(L))     # just for fun, try max and min of this L:  We win! (Why?!)


L = [1, 2, 40, 3 ]
print("max(L) is", max(L))
print("min(L) is", min(L))
print("sum(L) is", sum(L))


L = range(1,43)
print("L is", L)   # Uh oh... it won't create the list unless we tell it to...


L = list(range(1,35))  # ask it to create the list values...
print("L is", L)  # Aha!


print("max(L) is", max(L))    # ... up to but _not_ including the endpoint!


#
# Gauss's number: adding from 1 to 100
#
L = list(range(1,101))
print("sum(L) is", sum(L))



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
s = "major"
print("s is", s)
print()

LC = [ vwl_once(c) for c in s ]
print("LC is", LC)

d = 'thrift'
LC2 = [ vwl_once(c) for c in d ]
print("LC2 is", LC2)

f = 'analogic'
LC3 = [ vwl_once(c) for c in f ]
print("LC3 is", LC3)


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


#
# here, [1] create a function that scores punctuation as 1 and non-punctuation as 0, pun_one
#       [2] use the list-comprehension technique to create a punctuation-scoring function, pun_all

# use the above examples as a starting point!
def pun_one(x):
  """ pun_one returns 1 for punctuation, 0 otherwise
  """
  if x in '.,;:!?@#$%^&*()-={}[]"\'': return 1
  else: return 0

def pun_all(z):
  """ returns the total # of punctuation in z, a string
  """
  LP = [ pun_one(x) for x in z ]
  total = sum(LP)  # add them all up!
  return total
#




# tests
print("pun_all(PLANKTON) should be 4 <->", pun_all(PLANKTON))
print("pun_all(PATRICK) should be 4 <->", pun_all(PATRICK))
print('I had to include the apostrophes...')
# They're more similar, punctuation-wise than scrabble-wise


#
# The two works of yours and two of another author's ("theirs")
#     These can - and should - be large
print("I hope my undergrad self can out-puncuate Deepseek AI")

YOURS1 = """  Masculinities within John Marston
Red Dead Redemption tells a story from the viewpoint of John Marston. John Marston is an interesting character because of his past and the actions that he has committed. John Marston used to be an outlaw in a gang that raided, kidnapped and murdered and yet the character begins the game being more level-headed, calm and polite - particularly towards women. While people were far less politically correct in the early 1900s in which this game takes place, men and women are treated in a very particular fashion in the game design. While all of the women, namely Bonnie MacFarlane, are capable - more often than not, they portray a damsel-in-distress. It leaves the choice for the player to defend these women or not. John Marston’s character does not portray the very obvious, over-the-top masculinity. He portrays and follows many of the ‘unspoken’ rules of masculinity: John Marston is fairly strong and quiet and really only shares his story with Bonnie. At least in the beginning, John Marston keeps his story a secret and refuses to tell anyone his past. The protagonist also “never lets anyone see him sweat,” seeing as how he was shot in the beginning of the game and shows little signs to the public. These types of masculinities are the ones that John Marston most personify.
	With Red Dead Redemption, there is some focus on masculinity - every choice in a video game is well thought out when it comes to game development. However, mostly because Rockstar Games is skilled at catering to their audience. With a largely emotionless male cowboy, it allows male players to control John Marston and live their power fantasy, masculine dreams. After all, much of the audience for console gamers identify as male. I personally know many gamers that love masculinity in video games and dislike it when their character portrays elements of femininity. Even when Samus takes off her helmet to reveal her gender, gamers were upset because they thought they were playing a male. The writing in the game lends to masculine undertones as well, often reinforcing ideas that “women shouldn’t be able to do this” and “men are meant to do this.” The gameplay, particularly the slow motion shooting, makes it really easy to live a power fantasy and kill many people in an instant, which makes players feel really macho. All of this leads to toxicity when it comes to existing as a male. This is because it disallows males to be influenced by anything other than what they are exposed to in media.

"""
#undergrad writing...
YOURS2 = """  Portal 2: Rules of Play
	The reason Portal (2007) and Portal 2 (2011) are different from a game like Tomb Raider (2013) is because of the progression within the game. Tomb Raider is largely a linear game with branching paths, but with Portal, there is very clearly only one way to progress through the game. There are literal doors that block progression and if the players are unable to figure out the puzzle within the test chambers, the game is unbeatable. There are rules that Portal developers, Valve, put into play that define the game and it’s gameplay elements. Within the first couple minutes of gameplay, the player is presented with a simple puzzle: put the cube on the big red button. Shortly afterwards, the game opens up portals that the player must walk through and this allows the players ample time to analyze what happened while simultaneously displaying the rules of the game. If something goes into a blue portal, it comes out the orange portal and vice versa. Later, it is revealed that momentum plays a part as well. “Speedy-thing goes in, speedy-thing goes out,” as character, GlaDOS, puts it. Most of the game revolves around these core mechanics that are presented to the player in a slow and steady fashion.
These are the rules of the game and players are unable to progress without an understanding of these mechanics, a ludus game design with flat ideological structure. After the rules are established, control is given to the player after the first couple of test chambers so that they may make the portals for themselves. Portal leaves little room for emergent gameplay, as there is typically one way for the average player to beat a level and there aren’t too many systems of rules going on at once. From a gameplay perspective, there is rarely a time in the game where you are challenged by an enemy AI (despite GlaDOS being exactly that, narratively) besides the turrets, which are simple “shoot anything in front of it” objects.
	Physics plays a large role in Portal as players will need to launch themselves and objects into doors, buttons and rooms. Valve designed it into the game by making it one of the primary focuses into the puzzles. Often times you are jumping through a portal at terminal velocity in order to propel out the other portal at the same speed. This gives the player something of a roller-coaster feeling as they soar through the air. Most importantly, it makes the player feel in control of their movements, as they are the reason for figuring out each puzzle. The game never gives a large hint or a helping hand. It does not insult the players’ intelligence - it allows them ample time to fully comprehend the rules of the game.  On the topic of duration, the player mostly never has a portion of the game where they are stressed for time, but time can play an important role during the puzzles, as players will occasionally have to time portal placements in order to build momentum.
Furthermore, the first-person perspective is important in Portal as the emphasis lies in the puzzles. The first-person perspective helps players conceptualize outcomes along with allowing players a more precise control in their actions, including the most important part of the game - placing portals. The perspective allows the gameplay to shine - not detracting from it with role-play or character development (a game that pleases ludologists. Probably.) While the player can see their character if they try hard enough, the character doesn’t speak or develop over the game allowing the player to tie their own experiences with these games. The controls are fairly simple and tight, placing emphasis on mechanics of movements within the confines of the rules in order to add complexity, rather than an initially complex control layout. The only thing that may be fairly jarring for players is the camera jerk when a player goes through a portal through the ceiling. The first person camera spins, sometimes a full 180 degrees, in order to allow the player to always be upright. Portal is first and foremost, a puzzle game that has a set of rules that the player must abide by. These rules are followed, tested and played with in different, increasingly difficult levels. There is no in-game reward other than the satisfaction of figuring it out and progressing, leading to a fair, yet triumphant experience for players.

"""

#write me a 2 page essay on john marston from red dead redemption and how he portrays "unspoken" rules of masculinity
THEIRS1 = """  John Marston and the Unspoken Rules of Masculinity in Red Dead Redemption

John Marston, the protagonist of Rockstar Games' Red Dead Redemption, is a quintessential representation of the rugged, morally complex antihero. Set in the dying days of the American Old West, Marston’s journey is not just one of redemption but also a nuanced exploration of masculinity. Through his character, the game delves into the "unspoken rules" of masculinity—those societal expectations and codes of conduct that define what it means to be a man in a harsh, unforgiving world. Marston embodies these rules in his actions, relationships, and struggles, offering a compelling commentary on the pressures and contradictions of traditional masculinity.

The Stoic Provider
One of the most prominent unspoken rules of masculinity that John Marston embodies is the expectation to be a stoic provider. From the outset, Marston’s motivations are deeply tied to his role as a husband and father. Forced into a mission to hunt down his former gang members by federal agents who threaten his family, Marston’s actions are driven by his commitment to protect and provide for his wife, Abigail, and son, Jack. This aligns with the traditional masculine ideal of the man as the protector and breadwinner, willing to endure hardship and danger for the sake of his family.

However, Marston’s stoicism often masks his vulnerability. He rarely expresses his fears or emotions openly, adhering to the unspoken rule that men must remain strong and unflinching in the face of adversity. This emotional restraint is particularly evident in his interactions with Abigail, where his love and concern are conveyed through actions rather than words. For instance, his relentless pursuit of his former gang members, despite the personal cost, demonstrates his unwavering dedication to securing his family’s future. Yet, this stoicism also isolates him, highlighting the emotional toll of adhering to these rigid masculine norms.

The Code of Honor
Another unspoken rule of masculinity that Marston embodies is the adherence to a personal code of honor. In the lawless world of the Old West, where institutions are corrupt and justice is often arbitrary, Marston’s sense of morality becomes a defining aspect of his character. He operates by a set of principles that prioritize loyalty, fairness, and respect, even as he navigates a morally gray world.

Marston’s code of honor is most evident in his interactions with others. He helps strangers in need, stands up for the oppressed, and shows respect to those who earn it. However, this code is not without its contradictions. As a former outlaw, Marston has a violent past, and his journey is marked by moments of brutality. Yet, these actions are often framed as necessary evils, reflecting the unspoken rule that men must sometimes resort to violence to protect what they hold dear. This duality underscores the complexity of masculinity, where the line between heroism and villainy is often blurred.

The Burden of Independence
Independence is another key aspect of traditional masculinity that Marston embodies. Throughout the game, he is portrayed as a self-reliant figure, capable of surviving in the harsh wilderness and overcoming formidable challenges on his own. This independence is a source of pride for Marston, but it also becomes a burden. The unspoken rule that men must rely on themselves and avoid showing weakness often leaves him isolated and vulnerable.

Marston’s independence is particularly evident in his reluctance to ask for help. Even when faced with overwhelming odds, he prefers to tackle problems on his own, reflecting the societal expectation that men must be self-sufficient. This trait is both a strength and a flaw, as it often leads him into dangerous situations that could have been avoided with collaboration. Ultimately, Marston’s journey suggests that true strength lies not in isolation but in the ability to forge meaningful connections with others.

The Struggle for Redemption
At its core, Red Dead Redemption is a story of redemption, and Marston’s quest to atone for his past sins is deeply tied to his understanding of masculinity. The unspoken rule that men must take responsibility for their actions and seek redemption is a central theme of the game. Marston’s willingness to confront his past and make amends reflects his commitment to this ideal.

However, the game also critiques the notion that redemption is a straightforward or achievable goal. Marston’s efforts to leave his violent past behind are constantly thwarted by the world around him, suggesting that the expectations placed on men to be both strong and morally upright are often unrealistic. His ultimate fate—a tragic and inevitable consequence of his past—underscores the futility of trying to conform to these rigid masculine ideals.

John Marston is a complex and multifaceted character who embodies the unspoken rules of masculinity in a way that is both compelling and critical. Through his stoicism, code of honor, independence, and quest for redemption, Marston reflects the societal expectations placed on men to be strong, self-reliant, and morally upright. Yet, his story also highlights the contradictions and costs of these ideals, offering a nuanced commentary on the pressures of traditional masculinity.

In the end, Marston’s journey is not just one of personal redemption but also a reflection on the broader struggles of men to navigate the expectations placed upon them. His character reminds us that true strength lies not in adhering to rigid norms but in embracing vulnerability, forging connections, and striving for a more authentic and compassionate understanding of what it means to be a man. Through John Marston, Red Dead Redemption challenges players to reconsider the unspoken rules of masculinity and the impact they have on individuals and society as a whole.
"""

THEIRS2 = """  The Difference in Progression Between Portal and Tomb Raider: Rules of Play and Ludus Design Principles

The Portal series and the Tomb Raider franchise are two iconic examples of video games that, while both falling under the broader category of action-adventure, employ vastly different approaches to progression, player engagement, and game design principles. These differences are particularly evident when examining the "rules of play" and the concept of "ludus," a term coined by game theorist Roger Caillois to describe structured, goal-oriented games. By analyzing these two franchises, we can explore how their design philosophies shape player experiences and progression systems.

Rules of Play in Portal
The Portal series, developed by Valve, is a masterclass in minimalist game design. Its rules of play are deceptively simple: players are equipped with a portal gun that creates two connected portals on flat surfaces, allowing them to navigate through space in innovative ways. The game’s progression is built around this single mechanic, which is gradually expanded and complicated through increasingly challenging puzzles.

The rules of play in Portal are tightly defined and consistent. Each level introduces new elements, such as energy balls, light bridges, or tractor beams, but these are always integrated into the core portal mechanic. This creates a sense of mastery as players learn to manipulate the environment within the game’s established rules. The progression is linear but deeply satisfying, as each puzzle builds on the skills and knowledge acquired in previous levels.

The Portal series exemplifies the "ludus" principle, as it is highly structured and goal-oriented. Each level has a clear objective: reach the exit. The game’s design ensures that players are constantly challenged but never overwhelmed, as the rules are introduced incrementally. This approach fosters a sense of accomplishment and intellectual satisfaction, as players must think creatively within the constraints of the game’s mechanics.

Rules of Play in Tomb Raider
In contrast, the Tomb Raider franchise, particularly its modern reboot series developed by Crystal Dynamics, adopts a more open and exploratory approach to progression. The rules of play in Tomb Raider are broader and more varied, encompassing combat, exploration, puzzle-solving, and platforming. The game’s progression is tied to both narrative development and character growth, as players guide Lara Croft through her transformation from a vulnerable survivor to a confident adventurer.

The rules of play in Tomb Raider are less rigid than those in Portal. While there are specific mechanics, such as climbing, shooting, and crafting, the game allows for a greater degree of player agency. Exploration is encouraged, with hidden tombs, collectibles, and side quests providing opportunities for non-linear progression. This creates a more immersive and dynamic experience, as players can choose how to engage with the game world.

However, Tomb Raider still adheres to the "ludus" principle, as it maintains a clear structure and set of goals. The narrative drives the progression, with each chapter advancing the story and introducing new challenges. The game’s design balances freedom with direction, ensuring that players are always working toward a specific objective, whether it’s solving a puzzle, defeating enemies, or uncovering the next piece of the story.

Comparing Progression and Ludus Design
The key difference in progression between Portal and Tomb Raider lies in their approach to player engagement and the application of ludus principles. Portal focuses on intellectual challenge and mastery of a single mechanic, creating a tightly controlled and highly satisfying experience. Its progression is linear and incremental, with each level serving as a stepping stone to greater complexity. This design reinforces the ludus principle by providing clear goals and a structured path to achievement.

Tomb Raider, on the other hand, emphasizes exploration, narrative, and character development. Its progression is more open-ended, allowing players to engage with the game world in diverse ways. While it still adheres to the ludus principle through its structured narrative and objectives, it offers greater flexibility in how those objectives are achieved. This creates a more varied and immersive experience, appealing to players who enjoy both challenge and storytelling.

The Portal series and the Tomb Raider franchise represent two distinct approaches to progression and game design within the action-adventure genre. Portal’s minimalist design and focus on intellectual challenge exemplify the ludus principle through its tightly defined rules of play and linear progression. In contrast, Tomb Raider embraces a broader set of mechanics and a more open-ended structure, balancing ludus with elements of exploration and narrative depth.

Both franchises demonstrate how the rules of play and ludus principles can be applied in different ways to create engaging and memorable experiences. Portal challenges players to think creatively within a constrained framework, while Tomb Raider invites them to explore a rich and dynamic world. Together, they highlight the versatility of video games as a medium and the myriad ways in which designers can craft compelling progression systems. Whether through the precision of Portal or the expansiveness of Tomb Raider, both approaches offer unique insights into the art of game design and the enduring appeal of structured, goal-oriented play.
"""


print(len(THEIRS1))
print(len(YOURS1))
print(len(THEIRS2))
print(len(YOURS2))
print('always has to 1-up me..')


#
# Here, run your punctuation-comparisons (absolute counts)
#
print(pun_all(YOURS1))
print(pun_all(THEIRS1))
print(pun_all(YOURS2))
print(pun_all(THEIRS2))



#
# Here, run your punctuation-comparisons (relative, per-character counts)
#
print(pun_all(YOURS1)/len(YOURS1))
print(pun_all(THEIRS1)/len(THEIRS1))
print(pun_all(YOURS2)/len(YOURS2))
print(pun_all(THEIRS2)/len(THEIRS2))



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
print(result)


LC = [ len(birthday_room()) for i in range(10) ]
print(LC)
sum(LC) /len(LC)



L = [ len(birthday_room()) for i in range(100) ]
print("birthday room: L[0:5] are", L[0:5])
print("birthday room: L[-5:] are", L[-5:])
#
# Let's see what the average results were
print("len(L) is", len(L))
ave = sum(L)/len(L)
print("average is", ave )
print("Max is", max(L))
# try min and max, count of 42's, count of 92's, etc.


[ x**2 for x in [3,5,7]]


[ s[0] for s in ["ash", "IST341_Participant_8", "mohammed"]]


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
Q = [ rwalk(4) for i in range(1000) ]
average2 = sum(Q)/len(Q)
print("The average for radius==4 (for 1000 trials) was", average2)


# Repeat the above for a radius of 6, 7, 8, 9, and 10
# It's fast to do:  Copy and paste and edit!!
# Here is a starting example:
Q = [ rwalk(4) for i in range(1000) ]
average2 = sum(Q)/len(Q)
print("The average for radius==4 (for 1000 trials) was", average2)

W = [ rwalk(6) for i in range(1000) ]
average3 = sum(W)/len(W)
print("The average for radius==6 (for 1000 trials) was", average3)

E = [ rwalk(7) for i in range(1000) ]
average4 = sum(E)/len(E)
print("The average for radius==7 (for 1000 trials) was", average4)

R = [ rwalk(8) for i in range(1000) ]
average5 = sum(R)/len(R)
print("The average for radius==8 (for 1000 trials) was", average5)

T = [ rwalk(9) for i in range(1000) ]
average6 = sum(T)/len(T)
print("The average for radius==9 (for 1000 trials) was", average6)

Y = [ rwalk(10) for i in range(1000) ]
average7 = sum(Y)/len(Y)
print("The average for radius==10 (for 1000 trials) was", average7)


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
TA = d['number']
print(TA)
#    (2) to extract the name "Sunita Williams" from the dictionary d
for astronaut in d["people"]:
    if astronaut["name"] == "Sunita Williams":
        print(astronaut["name"])
        break



# use this cell - based on the example above - to share your solutions to the Astronaut challenges...
for astronaut in d["people"]:
    if astronaut["name"] == "Jeanette Epps":
        print(astronaut["name"])
        break
# Step 1: Locate the dictionary for "Nikolai Chub"
for astronaut in d["people"]:
    if astronaut["name"] == "Nikolai Chub":
        name = astronaut["name"]
        print(name[3] + name[2])
        break


#
# use this cell for your API call - and data-extraction
#import requests

url = "https://catfact.ninja/fact"   # this is sometimes called an "endpoint" ...
result_cf = requests.get(url)
result_cf


#
# use this cell for your webscraping call - optional data-extraction
#

cat_fact = result_cf.text
print(cat_fact)



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





