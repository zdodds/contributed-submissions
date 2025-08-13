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

A = [ n+2 for n in   range(40,42) ]
print(A)
B = [ 42 for z in [0,1,2] ]
print(B)
C = [ z for z in [42,42] ]
print(C)
D = [ s[::-2] for s in ['fungic','y!a!y!'] ]
print(D)
L = [ [len(w),w] for w in  ['Hi','IST'] ]
print(L)

Y = ["Hello" for _ in [1,2,3]]
print(Y)

X = [n**2 - 3 for n in range(5, 8)]
print(X)
# then, see if they work the way you predict...


#
# here, [1] create a function that scores punctuation as 1 and non-punctuation as 0, pun_one
#       [2] use the list-comprehension technique to create a punctuation-scoring function, pun_all

# use the above examples as a starting point!
import string

def pun_one(char):
    return 1 if char in string.punctuation else 0
def pun_all(text):
    return sum([pun_one(c) for c in text])



# tests
print("pun_all(PLANKTON) should be 4 <->", pun_all(PLANKTON))
print("pun_all(PATRICK) should be 4 <->", pun_all(PATRICK))
# They're more similar, punctuation-wise than scrabble-wise


#
# The two works of yours and two of another author's ("theirs")
#     These can - and should - be large
#

YOURS1 = """
FUNDAMENTALS OF PETROLEUM QUALITY MANAGEMENT
1-35. Contaminated, comingled, or dirty fuels can damage expensive engines and cause the failure of criticalcombat missions. Quality management is the responsibility of every element that receives, stores, and issues
Chapter 1
1-8 ATP 4-43 18 April 2022
bulk fuel. Quality management includes two subsets - quality assurance and quality surveillance. Quality assurance determine if the bulk fuel producer or supplier complies with the required specifications detailed in the contract. Quality assurance takes place at the strategic level and is primarily a DLA Energy function. Quality surveillance ensures the on specification fuel provided to the Service is acceptable for the intended use until consumed. Quality surveillance is a Service task performed by qualified personnel using approved petroleum laboratories and test kits across the area of operations.
QUALITY ASSURANCE
1-36. Quality assurance is a planned and systematic pattern of all actions necessary to provide confidencethat adequate technical requirements are established; products and services conform to established technicalrequirements; and satisfactory performance is achieved. For the Government, contract quality assurance is amethod to determine if a supplier of products or services fulfilled its contract obligations pertaining toproducts or services provided. It includes all actions required to ensure the Government is receiving theproper products and services. By common usage, contract quality assurance responsibility is fulfilled whenthe product or service is accepted by the Government and the product no longer belongs to the contractor orthe service is complete.
QUALITY SURVEILLANCE
1-37. Quality surveillance is the aggregate of measures (such as blending, stock rotation, and sampling) usedto determine and maintain the quality of product receipts and Government-owned bulk petroleum productsto the degree necessary to ensure that such products are suitable for their intended use. Quality surveillancetakes place at the strategic, operational and tactical levels. Sediment, water, microbial growth, andcommingled fuel may damage aircraft, ground vehicles, and fuel storage equipment. Contaminated ordeteriorated fuel can cost lives, especially with aircraft.
1-38. A robust, detailed quality surveillance program ensures fuel used in military equipment is clean (clear)and bright and suitable for immediate use for its intended purpose. Quality surveillance applies to allpetroleum products, and is the responsibility of all personnel who handle petroleum. Quality surveillancetests performed depend on where the bulk fuel is in the distribution network. The minimum qualitysurveillance tests required for each fuel type and location are provided in MIL-STD-3004-1A. Army specificquality surveillance guidance can be found within DA PAM 710-2-1 and AR 710-2. In addition to ensuringpetroleum issued is suitable for use, quality surveillance provides insight into how well equipment andproducts are being maintained.
1-39. Daily quality surveillance of petroleum storage and distribution systems is essential to detect leaks,sabotage, damage, pilferage, unintended product comingling, contamination, and deterioration duringstorage. Qualified fuel handlers and petroleum laboratory specialists perform and supervise qualitysurveillance throughout the Army petroleum distribution network. Quality surveillance is the responsibilityof every element in the distribution network that receives, stores, and issues bulk fuel.
1-40. Quality surveillance and sampling of bulk fuel is necessary to ensure that quality products are supplied.Fuel handlers and petroleum laboratory specialists personnel take samples of products for testing as follows:
Upon receipt.
Daily prior to use.
When requested by petroleum offices.
Bulk fuel stored for six months or more. Refer to AR 710-2 and MIL-STD-3004-1A.
The quality of fuel is questioned or it cannot be classified.
A filter separator is first placed in service after the filter-coalescer elements have been changedand within 30 days from the date last sampled from that filter separator.
It is reasonably suspected that an aviation fuel may be contaminated or commingled.
Commercial deliveries of bulk fuel. Refer to DA PAM 710-2-1 and MIL-STD-3004-1A.
1-41.
"""
YOURS2 = """2-2. USTRANSCOM is the functional combatant command responsible for providing and managingstrategic common-user airlift, sealift, and terminal services worldwide. The USTRANSCOM missionincludes planning for and providing air, land, and sea transportation of fuels for DOD during all operationalcontexts.
2-3. Combatant commands, service components, and DLA Energy coordinate with USTRANSCOM tomove petroleum assets outside of the operational area.
2-4. USTRANSCOM’s major subordinate commands include Air Mobility Command as its Air Forcecomponent command, Military Sealift Command as its Navy component command, and the Military SurfaceDeployment and Distribution Command (also known as SDDC) as its Army Service component command.Military Sealift Command operates vessels that sustain our warfighting forces and deliver specializedmaritime services in support of national security objectives. The Military Sealift Command providespetroleum capability through the OPDS. When commercial facilities are not available, are damaged, orinadequate, the OPDS can hook up to commercial and military oceangoing tankers to provide an effectivealternative method for providing high volumes of fuel to the shore.
2-5. The Joint Petroleum Office, USTRANSCOM, represents Commander, USTRANSCOM, on allpetroleum-related issues: Key duties and responsibilities of the USTRANSCOM JPO include —
Prepare plans, policies and procedures for executing petroleum operations related to supportingthe USTRANSCOM strategic mission.
Develop long-range sustainment plans for petroleum support of USTRANSCOM’s inter-theatermission and contingency operations worldwide.
Review long-range plans for positioning of petroleum assets.
Oversee and validate all fuel data reporting by Army Materiel Command and Military SealiftCommand.
Assist combatant commanders on establishing their fuel-related priorities.
Chapter 2
2-2 ATP 4-43 18 April 2022
Coordinate with other JPOs to de-conflict requirements.
DEFENSE LOGISTICS AGENCY
2-6. DLA is the primary DOD logistics provider for supply classes I, II, III, IV, VI, VII and IX. DLAEnergy is a major subordinate command of DLA and is the primary source for class III bulk petroleumsupplies. DLA supports each GCC, often by co-locating a DLA regional command with the sustainmentheadquarters (typically the TSC or ESC). DLA regional commands are the focal point for coordinating DLAactivities throughout the area of responsibility (AOR) and reach back to other DLA elements in CONUS forlogistics solutions.
2-7. In accordance with DODD 5101.8E, DLA Energy —
Serves as the class III (B) executive agent responsible for providing contracted fuel support acrossall services and combatant commands in permissive and semi permissive environments.
Manages the bulk petroleum distribution network from the refinery to the supported andsupporting unit.
Obtains bulk fuel locally within the theater, when possible.
2-8. DLA Energy uses Accountable Property Systemas its automated conduit to manage any location wherethe fuel is owned by DLA Energy. These locations are known as capitalized locations or Defense FuelSupport Points. Defense Fuel Support Points receive, store, and issue capitalized fuel. Capitalization refersto ownership. Capitalized fuel is managed and owned by DLA Energy from refinery to delivery to an Armyunit who is required to reimburse the Defense-Wide Working Capital Fund for the fuel it received. Fuelpurchased from DLA Energy and owned by the Services is non-capitalized.
2-9. The DLA Energy bulk fuels commodity business unit acts as the principal advisor and assistant to theDirector, DLA Energy and the deputy director of operations in directing the accomplishment of missionresponsibilities. These responsibilities are to provide —
Worldwide support of authorized activities in the areas of contracting, distribution, transportation.
Inventory control of: bulk fuels, including jet fuels, distillate fuels, residual fuels, automotivegasoline (for overseas locations only), specified bulk lubricating oils, aircraft engine oils, bulk fueladditives, and crude oil.
2-10. The direct delivery fuels commodity business unit acquires and manages ground, aviation, and shippropulsion fuels delivered directly to the requiring Service from commercial vendors. DLA Energy isresponsible for quality assurance and on-specification delivery of all bulk petroleum products procured anddistributed through the distribution management process. During large-scale combat operations, DLA cannotbe expected to deliver fuel forward of the rear boundary of the corps support area.
"""
THEIRS1 =  """
  FIRST LORD. And grant it.
  HELENA. Thanks, sir; all the rest is mute.
  LAFEU. I had rather be in this choice than throw ames-ace for my
    life.
  HELENA. The honour, sir, that flames in your fair eyes,
    Before I speak, too threat'ningly replies.
    Love make your fortunes twenty times above
    Her that so wishes, and her humble love!
  SECOND LORD. No better, if you please.
  HELENA. My wish receive,
    Which great Love grant; and so I take my leave.
  LAFEU. Do all they deny her? An they were sons of mine I'd have
    them whipt; or I would send them to th' Turk to make eunuchs of.
  HELENA. Be not afraid that I your hand should take;
    I'll never do you wrong for your own sake.
    Blessing upon your vows; and in your bed
    Find fairer fortune, if you ever wed!
  LAFEU. These boys are boys of ice; they'll none have her.
    Sure, they are bastards to the English; the French ne'er got 'em.
  HELENA. You are too young, too happy, and too good,
    To make yourself a son out of my blood.
  FOURTH LORD. Fair one, I think not so.
  LAFEU. There's one grape yet; I am sure thy father drunk wine-but
    if thou be'st not an ass, I am a youth of fourteen; I have known
    thee already.
  HELENA.  [To BERTRAM]  I dare not say I take you; but I give
    Me and my service, ever whilst I live,
    Into your guiding power. This is the man.
  KING. Why, then, young Bertram, take her; she's thy wife.
  BERTRAM. My wife, my liege! I shall beseech your Highness,
    In such a business give me leave to use
    The help of mine own eyes.
  KING. Know'st thou not, Bertram,
    What she has done for me?
  BERTRAM. Yes, my good lord;
    But never hope to know why I should marry her.
  KING. Thou know'st she has rais'd me from my sickly bed.
  BERTRAM. But follows it, my lord, to bring me down
    Must answer for your raising? I know her well:
    She had her breeding at my father's charge.
    A poor physician's daughter my wife! Disdain
    Rather corrupt me ever!
  KING. 'Tis only title thou disdain'st in her, the which
    I can build up. Strange is it that our bloods,
    Of colour, weight, and heat, pour'd all together,
    Would quite confound distinction, yet stand off
    In differences so mighty. If she be
    All that is virtuous-save what thou dislik'st,
    A poor physician's daughter-thou dislik'st
    Of virtue for the name; but do not so.
    From lowest place when virtuous things proceed,
    The place is dignified by the doer's deed;
    Where great additions swell's, and virtue none,
    It is a dropsied honour. Good alone
    Is good without a name. Vileness is so:
    The property by what it is should go,
    Not by the title. She is young, wise, fair;
    In these to nature she's immediate heir;
    And these breed honour. That is honour's scorn
    Which challenges itself as honour's born
    And is not like the sire. Honours thrive
    When rather from our acts we them derive
    Than our fore-goers. The mere word's a slave,
    Debauch'd on every tomb, on every grave
    A lying trophy; and as oft is dumb
    Where dust and damn'd oblivion is the tomb
    Of honour'd bones indeed. What should be said?
    If thou canst like this creature as a maid,
    I can create the rest. Virtue and she
    Is her own dower; honour and wealth from me.
  BERTRAM. I cannot love her, nor will strive to do 't.
  KING. Thou wrong'st thyself, if thou shouldst strive to choose.
  HELENA. That you are well restor'd, my lord, I'm glad.
    Let the rest go.
  KING. My honour's at the stake; which to defeat,
    I must produce my power. Here, take her hand,
    Proud scornful boy, unworthy this good gift,
    That dost in vile misprision shackle up
    My love and her desert; that canst not dream
    We, poising us in her defective scale,
    Shall weigh thee to the beam; that wilt not know
    It is in us to plant thine honour where
    We please to have it grow. Check thy contempt;
    Obey our will, which travails in thy good;
    Believe not thy disdain, but presently
    Do thine own fortunes that obedient right
    Which both thy duty owes and our power claims;
    Or I will throw thee from my care for ever
    Into the staggers and the careless lapse
    Of youth and ignorance; both my revenge and hate
    Loosing upon thee in the name of justice,
    Without all terms of pity. Speak; thine answer.
  BERTRAM. Pardon, my gracious lord; for I submit
    My fancy to your eyes. When I consider
    What great creation and what dole of honour
    Flies where you bid it, I find that she which late
    Was in my nobler thoughts most base is now
    The praised of the King; who, so ennobled,
    Is as 'twere born so.
  KING. Take her by the hand,
    And tell her she is thine; to whom I promise
    A counterpoise, if not to thy estate
    A balance more replete.
  BERTRAM. I take her hand.
  KING. Good fortune and the favour of the King
    Smile upon this contract; whose ceremony
    Shall seem expedient on the now-born brief,
    And be perform'd to-night. The solemn feast
    Shall more attend upon the coming space,
    Expecting absent friends. As thou lov'st her,
    Thy love's to me religious; else, does err.
              Exeunt all but LAFEU and PAROLLES who stay behind,
                                      commenting of this wedding
  LAFEU. Do you hear, monsieur? A word with you.
  PAROLLES. Your pleasure, sir?
  LAFEU. Your lord and master did well to make his recantation.
  PAROLLES. Recantation! My Lord! my master!
  LAFEU. Ay; is it not a language I speak?
  PAROLLES. A most harsh one, and not to be understood without bloody
    succeeding. My master!
  LAFEU. Are you companion to the Count Rousillon?
  PAROLLES. To any count; to all counts; to what is man.
  LAFEU. To what is count's man: count's master is of another style.
  PAROLLES. You are too old, sir; let it satisfy you, you are too
    old.
  LAFEU. I must tell thee, sirrah, I write man; to which title age
    cannot bring thee.
  PAROLLES. What I dare too well do, I dare not do.
  LAFEU. I did think thee, for two ordinaries, to be a pretty wise
    fellow; thou didst make tolerable vent of thy travel; it might
    pass. Yet the scarfs and the bannerets about thee did manifoldly
    dissuade me from believing thee a vessel of too great a burden. I
    have now found thee; when I lose thee again I care not; yet art
    thou good for nothing but taking up; and that thou'rt scarce
    worth.
  PAROLLES. Hadst thou not the privilege of antiquity upon thee-
  LAFEU. Do not plunge thyself too far in anger, lest thou hasten thy
    trial; which if-Lord have mercy on thee for a hen! So, my good
    window of lattice, fare thee well; thy casement I need not open,
    for I look through thee. Give me thy hand.
  PAROLLES. My lord, you give me most egregious indignity.
  LAFEU. Ay, with all my heart; and thou art worthy of it.
  PAROLLES. I have not, my lord, deserv'd it.
  LAFEU. Yes, good faith, ev'ry dram of it; and I will not bate thee
    a scruple.
  PAROLLES. Well, I shall be wiser.
  LAFEU. Ev'n as soon as thou canst, for thou hast to pull at a smack
    o' th' contrary. If ever thou be'st bound in thy scarf and
    beaten, thou shalt find what it is to be proud of thy bondage. I
    have a desire to hold my acquaintance with thee, or rather my
    knowledge, that I may say in the default 'He is a man I know.'
  PAROLLES. My lord, you do me most insupportable vexation.
  LAFEU. I would it were hell pains for thy sake, and my poor doing
    eternal; for doing I am past, as I will by thee, in what motion
    age will give me leave.                                 Exit
  PAROLLES. Well, thou hast a son shall take this disgrace off me:
    scurvy, old, filthy, scurvy lord! Well, I must be patient; there
    is no fettering of authority. I'll beat him, by my life, if I can
    meet him with any convenience, an he were double and double a
    lord. I'll have no more pity of his age than I would have of-
    I'll beat him, and if I could but meet him again.

                         Re-enter LAFEU

  LAFEU. Sirrah, your lord and master's married; there's news for
    you; you have a new mistress.
  PAROLLES. I most unfeignedly beseech your lordship to make some
    reservation of your wrongs. He is my good lord: whom I serve
    above is my master.
  LAFEU. Who? God?
  PAROLLES. Ay, sir.
  LAFEU. The devil it is that's thy master. Why dost thou garter up
    thy arms o' this fashion? Dost make hose of thy sleeves? Do other
    servants so? Thou wert best set thy lower part where thy nose
    stands. By mine honour, if I were but two hours younger, I'd beat
    thee. Methink'st thou art a general offence, and every man should
    beat thee. I think thou wast created for men to breathe
    themselves upon thee.
  PAROLLES. This is hard and undeserved measure, my lord.
  LAFEU. Go to, sir; you were beaten in Italy for picking a kernel
    out of a pomegranate; you are a vagabond, and no true traveller;
    you are more saucy with lords and honourable personages than the
    commission of your birth and virtue gives you heraldry. You are
    not worth another word, else I'd call you knave. I leave you.
 Exit

                           Enter BERTRAM

  PAROLLES. Good, very, good, it is so then. Good, very good; let it
    be conceal'd awhile.
  BERTRAM. Undone, and forfeited to cares for ever!
  PAROLLES. What's the matter, sweetheart?
  BERTRAM. Although before the solemn priest I have sworn,
    I will not bed her.
  PAROLLES. What, what, sweetheart?
  BERTRAM. O my Parolles, they have married me!
    I'll to the Tuscan wars, and never bed her.
  PAROLLES. France is a dog-hole, and it no more merits
    The tread of a man's foot. To th' wars!
  BERTRAM. There's letters from my mother; what th' import is I know
    not yet.
  PAROLLES. Ay, that would be known. To th' wars, my boy, to th'
      wars!
    He wears his honour in a box unseen
    That hugs his kicky-wicky here at home,
    Spending his manly marrow in her arms,
    Which should sustain the bound and high curvet
    Of Mars's fiery steed. To other regions!
    France is a stable; we that dwell in't jades;
    Therefore, to th' war!
  BERTRAM. It shall be so; I'll send her to my house,
    Acquaint my mother with my hate to her,
    And wherefore I am fled; write to the King
    That which I durst not speak. His present gift
    Shall furnish me to those Italian fields
    Where noble fellows strike. War is no strife
    To the dark house and the detested wife.
  PAROLLES. Will this capriccio hold in thee, art sure?
  BERTRAM. Go with me to my chamber and advise me.
    I'll send her straight away. To-morrow
    I'll to the wars, she to her single sorrow.
  PAROLLES. Why, these balls bound; there's noise in it. 'Tis hard:
    A young man married is a man that's marr'd.
    Therefore away, and leave her bravely; go.
    The King has done you wrong; but, hush, 'tis so.      Exeunt>
"""
THEIRS2 = """
ACT II. SCENE 4.
Paris. The KING'S palace

Enter HELENA and CLOWN

  HELENA. My mother greets me kindly; is she well?
  CLOWN. She is not well, but yet she has her health; she's very
    merry, but yet she is not well. But thanks be given, she's very
    well, and wants nothing i' th' world; but yet she is not well.
  HELENA. If she be very well, what does she ail that she's not very
    well?
  CLOWN. Truly, she's very well indeed, but for two things.
  HELENA. What two things?
  CLOWN. One, that she's not in heaven, whither God send her quickly!
    The other, that she's in earth, from whence God send her quickly!

                        Enter PAROLLES

  PAROLLES. Bless you, my fortunate lady!
  HELENA. I hope, sir, I have your good will to have mine own good
    fortunes.
  PAROLLES. You had my prayers to lead them on; and to keep them on,
    have them still. O, my knave, how does my old lady?
  CLOWN. So that you had her wrinkles and I her money, I would she
    did as you say.
  PAROLLES. Why, I say nothing.
  CLOWN. Marry, you are the wiser man; for many a man's tongue shakes
    out his master's undoing. To say nothing, to do nothing, to know
    nothing, and to have nothing, is to be a great part of your
    title, which is within a very little of nothing.
  PAROLLES. Away! th'art a knave.
  CLOWN. You should have said, sir, 'Before a knave th'art a knave';
    that's 'Before me th'art a knave.' This had been truth, sir.
  PAROLLES. Go to, thou art a witty fool; I have found thee.
  CLOWN. Did you find me in yourself, sir, or were you taught to find
    me? The search, sir, was profitable; and much fool may you find
    in you, even to the world's pleasure and the increase of
    laughter.
  PAROLLES. A good knave, i' faith, and well fed.
    Madam, my lord will go away to-night:
    A very serious business calls on him.
    The great prerogative and rite of love,
    Which, as your due, time claims, he does acknowledge;
    But puts it off to a compell'd restraint;
    Whose want, and whose delay, is strew'd with sweets,
    Which they distil now in the curbed time,
    To make the coming hour o'erflow with joy
    And pleasure drown the brim.
  HELENA. What's his else?
  PAROLLES. That you will take your instant leave o' th' King,
    And make this haste as your own good proceeding,
    Strength'ned with what apology you think
    May make it probable need.
  HELENA. What more commands he?
  PAROLLES. That, having this obtain'd, you presently
    Attend his further pleasure.
  HELENA. In everything I wait upon his will.
  PAROLLES. I shall report it so.
  HELENA. I pray you.                              Exit PAROLLES
    Come, sirrah.                                         Exeunt




ACT II. SCENE 5.
Paris. The KING'S palace

Enter LAFEU and BERTRAM

  LAFEU. But I hope your lordship thinks not him a soldier.
  BERTRAM. Yes, my lord, and of very valiant approof.
  LAFEU. You have it from his own deliverance.
  BERTRAM. And by other warranted testimony.
  LAFEU. Then my dial goes not true; I took this lark for a bunting.
  BERTRAM. I do assure you, my lord, he is very great in knowledge,
    and accordingly valiant.
  LAFEU. I have then sinn'd against his experience and transgress'd
    against his valour; and my state that way is dangerous, since I
    cannot yet find in my heart to repent. Here he comes; I pray you
    make us friends; I will pursue the amity

                         Enter PAROLLES

  PAROLLES.  [To BERTRAM]  These things shall be done, sir.
  LAFEU. Pray you, sir, who's his tailor?
  PAROLLES. Sir!
  LAFEU. O, I know him well. Ay, sir; he, sir, 's a good workman, a
    very good tailor.
  BERTRAM.  [Aside to PAROLLES]  Is she gone to the King?
  PAROLLES. She is.
  BERTRAM. Will she away to-night?
  PAROLLES. As you'll have her.
  BERTRAM. I have writ my letters, casketed my treasure,
    Given order for our horses; and to-night,
    When I should take possession of the bride,
    End ere I do begin.
  LAFEU. A good traveller is something at the latter end of a dinner;
    but one that lies three-thirds and uses a known truth to pass a
    thousand nothings with, should be once heard and thrice beaten.
    God save you, Captain.
  BERTRAM. Is there any unkindness between my lord and you, monsieur?
  PAROLLES. I know not how I have deserved to run into my lord's
    displeasure.
  LAFEU. You have made shift to run into 't, boots and spurs and all,
    like him that leapt into the custard; and out of it you'll run
    again, rather than suffer question for your residence.
  BERTRAM. It may be you have mistaken him, my lord.
  LAFEU. And shall do so ever, though I took him at's prayers.
    Fare you well, my lord; and believe this of me: there can be no
    kernal in this light nut; the soul of this man is his clothes;
    trust him not in matter of heavy consequence; I have kept of them
    tame, and know their natures. Farewell, monsieur; I have spoken
    better of you than you have or will to deserve at my hand; but we
    must do good against evil.                              Exit
  PAROLLES. An idle lord, I swear.
  BERTRAM. I think so.
  PAROLLES. Why, do you not know him?
  BERTRAM. Yes, I do know him well; and common speech
    Gives him a worthy pass. Here comes my clog.

                          Enter HELENA

  HELENA. I have, sir, as I was commanded from you,
    Spoke with the King, and have procur'd his leave
    For present parting; only he desires
    Some private speech with you.
  BERTRAM. I shall obey his will.
    You must not marvel, Helen, at my course,
    Which holds not colour with the time, nor does
    The ministration and required office
    On my particular. Prepar'd I was not
    For such a business; therefore am I found
    So much unsettled. This drives me to entreat you
    That presently you take your way for home,
    And rather muse than ask why I entreat you;
    For my respects are better than they seem,
    And my appointments have in them a need
    Greater than shows itself at the first view
    To you that know them not. This to my mother.
                                               [Giving a letter]
    'Twill be two days ere I shall see you; so
    I leave you to your wisdom.
  HELENA. Sir, I can nothing say
    But that I am your most obedient servant.
  BERTRAM. Come, come, no more of that.
  HELENA. And ever shall
    With true observance seek to eke out that
    Wherein toward me my homely stars have fail'd
    To equal my great fortune.
  BERTRAM. Let that go.
    My haste is very great. Farewell; hie home.
  HELENA. Pray, sir, your pardon.
  BERTRAM. Well, what would you say?
  HELENA. I am not worthy of the wealth I owe,
    Nor dare I say 'tis mine, and yet it is;
    But, like a timorous thief, most fain would steal
    What law does vouch mine own.
  BERTRAM. What would you have?
  HELENA. Something; and scarce so much; nothing, indeed.
    I would not tell you what I would, my lord.
    Faith, yes:
    Strangers and foes do sunder and not kiss.
  BERTRAM. I pray you, stay not, but in haste to horse.
  HELENA. I shall not break your bidding, good my lord.
  BERTRAM. Where are my other men, monsieur?
    Farewell!                                        Exit HELENA
    Go thou toward home, where I will never come
    Whilst I can shake my sword or hear the drum.
    Away, and for our flight.
  PAROLLES. Bravely, coragio!                             Exeunt



<<THIS ELECTRONIC VERSION OF THE COMPLETE WORKS OF WILLIAM
SHAKESPEARE IS COPYRIGHT 1990-1993 BY WORLD LIBRARY, INC., AND IS
PROVIDED BY PROJECT GUTENBERG ETEXT OF ILLINOIS BENEDICTINE COLLEGE
WITH PERMISSION.  ELECTRONIC AND MACHINE READABLE COPIES MAY BE
DISTRIBUTED SO LONG AS SUCH COPIES (1) ARE FOR YOUR OR OTHERS
PERSONAL USE ONLY, AND (2) ARE NOT DISTRIBUTED OR USED
COMMERCIALLY.  PROHIBITED COMMERCIAL DISTRIBUTION INCLUDES BY ANY
SERVICE THAT CHARGES FOR DOWNLOAD TIME OR FOR MEMBERSHIP.>>




ACT III. SCENE 1.
Florence. The DUKE's palace

        Flourish. Enter the DUKE OF FLORENCE, attended; two
               FRENCH LORDS, with a TROOP OF SOLDIERS

  DUKE. So that, from point to point, now have you hear
    The fundamental reasons of this war;
    Whose great decision hath much blood let forth
    And more thirsts after.
  FIRST LORD. Holy seems the quarrel
    Upon your Grace's part; black and fearful
    On the opposer.
  DUKE. Therefore we marvel much our cousin France
    Would in so just a business shut his bosom
    Against our borrowing prayers.
  SECOND LORD. Good my lord,
    The reasons of our state I cannot yield,
    But like a common and an outward man
    That the great figure of a council frames
    By self-unable motion; therefore dare not
    Say what I think of it, since I have found
    Myself in my incertain grounds to fail
    As often as I guess'd.
  DUKE. Be it his pleasure.
  FIRST LORD. But I am sure the younger of our nature,
    That surfeit on their ease, will day by day
    Come here for physic.
  DUKE. Welcome shall they be
    And all the honours that can fly from us
    Shall on them settle. You know your places well;
    When better fall, for your avails they fell.
    To-morrow to th' field. Flourish.                     Exeunt




ACT III. SCENE 2.
Rousillon. The COUNT'S palace

Enter COUNTESS and CLOWN

  COUNTESS. It hath happen'd all as I would have had it, save that he
    comes not along with her.
  CLOWN. By my troth, I take my young lord to be a very melancholy
    man.
  COUNTESS. By what observance, I pray you?
  CLOWN. Why, he will look upon his boot and sing; mend the ruff and
    sing; ask questions and sing; pick his teeth and sing. I know a
    man that had this trick of melancholy sold a goodly manor for a
    song.
  COUNTESS. Let me see what he writes, and when he means to come.
                                              [Opening a letter]
  CLOWN. I have no mind to Isbel since I was at court. Our old ling
    and our Isbels o' th' country are nothing like your old ling and
    your Isbels o' th' court. The brains of my Cupid's knock'd out;
    and I begin to love, as an old man loves money, with no stomach.
  COUNTESS. What have we here?
  CLOWN. E'en that you have there.                          Exit
  COUNTESS.  [Reads]  'I have sent you a daughter-in-law; she hath
    recovered the King and undone me. I have wedded her, not bedded
    her; and sworn to make the "not" eternal. You shall hear I am run
    away; know it before the report come. If there be breadth enough
    in the world, I will hold a long distance. My duty to you.
                                           Your unfortunate son,
                                                       BERTRAM.'
    This is not well, rash and unbridled boy,
    To fly the favours of so good a king,
    To pluck his indignation on thy head
    By the misprizing of a maid too virtuous
    For the contempt of empire.

                           Re-enter CLOWN

  CLOWN. O madam, yonder is heavy news within between two soldiers
    and my young lady.
  COUNTESS. What is the -matter?
  CLOWN. Nay, there is some comfort in the news, some comfort; your
    son will not be kill'd so soon as I thought he would.
  COUNTESS. Why should he be kill'd?
  CLOWN. So say I, madam, if he run away, as I hear he does the
    danger is in standing to 't; that's the loss of men, though it be
    the getting of children. Here they come will tell you more. For my
    part, I only hear your son was run away.                Exit

              Enter HELENA and the two FRENCH GENTLEMEN

  SECOND GENTLEMAN. Save you, good madam.
  HELENA. Madam, my lord is gone, for ever gone.
  FIRST GENTLEMAN. Do not say so.
  COUNTESS. Think upon patience. Pray you, gentlemen-
    I have felt so many quirks of joy and grief
    That the first face of neither, on the start,
    Can woman me unto 't. Where is my son, I pray you?
  FIRST GENTLEMAN. Madam, he's gone to serve the Duke of Florence.
    We met him thitherward; for thence we came,
    And, after some dispatch in hand at court,
    Thither we bend again.
  HELENA. Look on this letter, madam; here's my passport.
    [Reads]  'When thou canst get the ring upon my finger, which
    never shall come off, and show me a child begotten of thy body
    that I am father to, then call me husband; but in such a "then" I
    write a "never."
    This is a dreadful sentence.
  COUNTESS. Brought you this letter, gentlemen?
  FIRST GENTLEMAN. Ay, madam;
    And for the contents' sake are sorry for our pains.
  COUNTESS. I prithee, lady, have a better cheer;
    If thou engrossest all the griefs are thine,
    Thou robb'st me of a moiety. He was my son;
    But I do wash his name out of my blood,
    And thou art all my child. Towards Florence is he?
  FIRST GENTLEMAN. Ay, madam.
  COUNTESS. And to be a soldier?
  FIRST GENTLEMAN. Such is his noble purpose; and, believe 't,
    The Duke will lay upon him all the honour
    That good convenience claims.
  COUNTESS. Return you thither?
  SECOND GENTLEMAN. Ay, madam, with the swiftest wing of speed.
  HELENA.  [Reads]  'Till I have no wife, I have nothing in France.'
    'Tis bitter.
  COUNTESS. Find you that there?
  HELENA. Ay, madam.
  SECOND GENTLEMAN. 'Tis but the boldness of his hand haply, which
    his heart was not consenting to.
  COUNTESS. Nothing in France until he have no wife!
    There's nothing here that is too good for him
    But only she; and she deserves a lord
    That twenty such rude boys might tend upon,
    And call her hourly mistress. Who was with him?
  SECOND GENTLEMAN. A servant only, and a gentleman
    Which I have sometime known.
  COUNTESS. Parolles, was it not?
  SECOND GENTLEMAN. Ay, my good lady, he.
  COUNTESS. A very tainted fellow, and full of wickedness.
    My son corrupts a well-derived nature
    With his inducement.
  SECOND GENTLEMAN. Indeed, good lady,
    The fellow has a deal of that too much
    Which holds him much to have.
  COUNTESS. Y'are welcome, gentlemen.
    I will entreat you, when you see my son,
    To tell him that his sword can never win
    The honour that he loses. More I'll entreat you
    Written to bear along.
  FIRST GENTLEMAN. We serve you, madam,
    In that and all your worthiest affairs.
  COUNTESS. Not so, but as we change our courtesies.
    Will you draw near?            Exeunt COUNTESS and GENTLEMEN
  HELENA. 'Till I have no wife, I have nothing in France.'
    Nothing in France until he has no wife!
    Thou shalt have none, Rousillon, none in France
    Then hast thou all again. Poor lord! is't
    That chase thee from thy country, and expose
    Those tender limbs of thine to the event
    Of the non-sparing war? And is it I
    That drive thee from the sportive court, where thou
    Wast shot at with fair eyes, to be the mark
    Of smoky muskets? O you leaden messengers,
    That ride upon the violent speed of fire,
    Fly with false aim; move the still-piecing air,
    That sings with piercing; do not touch my lord.
    Whoever shoots at him, I set him there;
    Whoever charges on his forward breast,
    I am the caitiff that do hold him to't;
    And though I kill him not, I am the cause
    His death was so effected. Better 'twere
    I met the ravin lion when he roar'd
    With sharp constraint of hunger; better 'twere
    That all the miseries which nature owes
    Were mine at once. No; come thou home, Rousillon,
    Whence honour but of danger wins a scar,
    As oft it loses all. I will be gone.
    My being here it is that holds thee hence.
    Shall I stay here to do 't? No, no, although
    The air of paradise did fan the house,
    And angels offic'd all. I will be gone,
    That pitiful rumour may report my flight
    To consolate thine ear. Come, night; end, day.
    For with the dark, poor thief, I'll steal away.         Exit




ACT III. SCENE 3.
Florence. Before the DUKE's palace

Flourish. Enter the DUKE OF FLORENCE, BERTRAM, PAROLLES, SOLDIERS,
drum and trumpets

  DUKE. The General of our Horse thou art; and we,
    Great in our hope, lay our best love and credence
    Upon thy promising fortune.
  BERTRAM. Sir, it is
    A charge too heavy for my strength; but yet
    We'll strive to bear it for your worthy sake
    To th' extreme edge of hazard.
  DUKE. Then go thou forth;
    And Fortune play upon thy prosperous helm,
    As thy auspicious mistress!
  BERTRAM. This very day,
    Great Mars, I put myself into thy file;
    Make me but like my thoughts, and I shall prove
    A lover of thy drum, hater of love.                   Exeunt




ACT III. SCENE 4.
Rousillon. The COUNT'S palace

Enter COUNTESS and STEWARD

  COUNTESS. Alas! and would you take the letter of her?
    Might you not know she would do as she has done
    By sending me a letter? Read it again.
  STEWARD.  [Reads]  'I am Saint Jaques' pilgrim, thither gone.
    Ambitious love hath so in me offended
    That barefoot plod I the cold ground upon,
    With sainted vow my faults to have amended.
    Write, write, that from the bloody course of war
    My dearest master, your dear son, may hie.
    Bless him at home in peace, whilst I from far
    His name with zealous fervour sanctify.
    His taken labours bid him me forgive;
    I, his despiteful Juno, sent him forth
    From courtly friends, with camping foes to live,
    Where death and danger dogs the heels of worth.
    He is too good and fair for death and me;
    Whom I myself embrace to set him free.'
  COUNTESS. Ah, what sharp stings are in her mildest words!
    Rinaldo, you did never lack advice so much
    As letting her pass so; had I spoke with her,
    I could have well diverted her intents,
    Which thus she hath prevented.
  STEWARD. Pardon me, madam;
    If I had given you this at over-night,
    She might have been o'er ta'en; and yet she writes
    Pursuit would be but vain.
  COUNTESS. What angel shall
    Bless this unworthy husband? He cannot thrive,
    Unless her prayers, whom heaven delights to hear
    And loves to grant, reprieve him from the wrath
    Of greatest justice. Write, write, Rinaldo,
    To this unworthy husband of his wife;
    Let every word weigh heavy of her worth
    That he does weigh too light. My greatest grief,
    Though little he do feel it, set down sharply.
    Dispatch the most convenient messenger.
    When haply he shall hear that she is gone
    He will return; and hope I may that she,
    Hearing so much, will speed her foot again,
    Led hither by pure love. Which of them both
    Is dearest to me I have no skill in sense
    To make distinction. Provide this messenger.
    My heart is heavy, and mine age is weak;
    Grief would have tears, and sorrow bids me speak.     Exeunt
"""


len(YOURS1)


import string
import pandas as pd

def punctuation_analysis(text):
    total_punctuation = sum(1 for char in text if char in string.punctuation)
    total_characters = len(text)
    density = total_punctuation / total_characters if total_characters > 0 else 0
    return {"Total Punctuation": total_punctuation, "Punctuation Density": round(density, 4)}



punctuation_data = {
    "YOURS1": punctuation_analysis(YOURS1),
    "YOURS2": punctuation_analysis(YOURS2),
    "THEIRS1": punctuation_analysis(THEIRS1),
    "THEIRS2": punctuation_analysis(THEIRS2)
}

punctuation_df = pd.DataFrame(punctuation_data).T
print(punctuation_df)



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
max = max(L)
min = min(L)
print("min is", min)
print("max is", max)
print("average is", ave )


# try min and max, count of 42's, count of 92's, etc.


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
        #ASCII = "|" + "_"*(start- -radius) + "S" + "_"*(radius-start) + "|"
        #print(ASCII, flush=True) # To see what's happening / debugging

    # it can never get here!

# Let's test it:
rwalk(5)   # walk randomly within our radius... until we hit a wall!


import random

def rwalk(radius):
    """Simulates a 1D random walk until it reaches ±radius and returns the number of steps taken."""
    position = 0
    steps = 0
    while abs(position) < radius:  # Stop when reaching the edge
        position += random.choice([-1, 1])
        steps += 1
    return steps  # Return the number of steps taken to reach the boundary

# Run the simulation 1000 times and store the number of steps taken
L = [rwalk(5) for _ in range(1000)]

# Calculate and print the average steps to reach the edge
average = sum(L) / len(L)
print("The average steps taken to reach radius==5 (for 1000 trials) was", average)

radii = [5, 10, 20, 50, 100,]
num_trials = 1000

results = []

for r in radii:
    L = [rwalk(r) for _ in range(num_trials)]  # Run the simulation for each radius
    avg_steps = sum(L) / len(L)  # Compute average steps
    expected_steps = r**2  # Theoretical expectation

    results.append((r, avg_steps, expected_steps))
    print(f"Radius: {r}, Simulated Average Steps: {avg_steps:.2f}, Expected (r²): {expected_steps}")


def f(s):
  count=0
  for char in s.lower():
    if char in "gcvuaont":
      count+=1
    elif char in "i":
      count+=2
    else:
      count-=1
  return count
  #return s.lower().count("g") + s.lower().count("c") + s.lower().count("u")

print(f("IST341_Participant_8!"))



s= "TUFcgu"
s.upper().count("U")


# Repeat the above for a radius of 6, 7, 8, 9, and 10
# It's fast to do:  Copy and paste and edit!!
radii = [6, 7, 8, 9,10]
num_trials = 1000

results = []

for r in radii:
    L = [rwalk(r) for _ in range(num_trials)]  # Run the simulation for each radius
    avg_steps = sum(L) / len(L)  # Compute average steps
    expected_steps = r**2  # Theoretical expectation

    results.append((r, avg_steps, expected_steps))
    print(f"Radius: {r}, Simulated Average Steps: {avg_steps:.2f}, Expected (r²): {expected_steps}")



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
url = "https://www.cgu.edu/"
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


#url = "http://api.open-notify.org/iss-now.json"
#result = requests.get(url)
#print(f"result.url         is {result.url}")  # the original url
#print(f"result.raw         is {result.raw}")  # another object!
#print(f"result.encoding    is {result.encoding}")  # utf-8 is very common
#print(f"result.status_code is {result.status_code}")  # 200 is success!


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
#url = "https://www.cs.hmc.edu/~dodds/demo.html"  # try it + source
#url = "https://www.scrippscollege.edu/"          # another possible site...
# url = "https://www.pitzer.edu/"                  # another possible site...
# url = "https://www.cmc.edu/"                     # and another!
#url = "https://www.cgu.edu/"                       # Yay CGU!
result = requests.get(url)
print(f"result is {result}")        # Question: is the result a "200" response?!  No not in all , CGU and SCRIPP gave me a 403


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
for person in d["people"]:
    if person["name"] == "Sunita Williams":
        sunita_name = person["name"]
        print(sunita_name)

num_astronauts = d["number"]
print(num_astronauts)


list(d)


# use this cell - based on the example above - to share your solutions to the Astronaut challenges...
for person in d["people"]:
    if person["name"] == "Jeanette Epps":
        sunita_name = person["name"]
        print(sunita_name)

for person in d["people"]:
    if person["name"] == "Oleg Kononenko":
        name = person["name"]
        ok_string = name[0] + name[5]
        print(ok_string)



#
# use this cell for your API call - and data-extraction
#


#
# use this cell for your webscraping call - optional data-extraction
#



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





