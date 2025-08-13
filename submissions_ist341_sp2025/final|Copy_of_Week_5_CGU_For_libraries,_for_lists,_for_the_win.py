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

L = range(1,101)
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


s = "below"
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)


s = "splits"
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)


s = "aboarded" # i cant find a real word
LC = [ vwl_once(c) for c in s ]
print("LC is", LC)


s = "omega"
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
B = [ 42 for z in [0,1,2] ]
C = [ z for z in [42,42] ]
D = [ s[::-2] for s in ['fungic','y!a!y!'] ]
L = [ [len(w),w] for w in  ['Hi','IST'] ]


# then, see if they work the way you predict...
print("A is", A)
print("B is", B)
print("C is", C)
print("D is", D)
print("L is", L)


#
# here, [1] create a function that scores punctuation as 1 and non-punctuation as 0, pun_one
#       [2] use the list-comprehension technique to create a punctuation-scoring function, pun_all

# use the above examples as a starting point!

def pun_one(p):
  """ pun_one returns 1 for a single punctuation, 0 otherwise
  """
  if p in '".,?!:;\'()[]{}—-.../': return 1
  else: return 0

def pun_all(s):
  """ returns the total # of puncuation in s, a string
  """
  LC = [ pun_one(p) for p in s ]
  total = sum(LC)  # add them all up!
  return total


# tests
print("pun_all(PLANKTON) should be 4 <->", pun_all(PLANKTON))
print("pun_all(PATRICK) should be 4 <->", pun_all(PATRICK))
# They're more similar, punctuation-wise than scrabble-wise


#
# The two works of yours and two of another author's ("theirs")
#     These can - and should - be large
#

YOURS1 = """IST341_Participant_7
Predicting Child Support Payment Delinquency Using SAS® Enterprise Miner™ 5.1 Case Study
Introduction (provide an overview of how you plan to structure the case study report, and a quick summary of what your understanding of CRISP-DM is)
This paper focuses on two separate case studies and how they utilize data mining tools and predictive modeling to help predict child support payment delinquency. As we have learned from the CRISP-DM framework, there are six phases that should be implemented when conducting a data mining project. These phases are:
Business Understanding: determine business objectives, assess situation, determine data mining goals, produce project plan
Data Understanding: collect initial data, describe data, explore data, verify data quality
Data Preparation: data set, select data, clean data, construct data, integrate data, format data
Modeling: select modeling technique, generate test design, build model, assess model
Evaluation: evaluate results, review process, determine next steps
Deployment: plan deployment, plan monitoring and maintenance, produce final report, review project
This model is extremely helpful for anyone who is looking to complete a data mining project to reach their business goals. It is important to note that the model and its phases are visualized in a cycle. As noted in the paper, “the arrows indicate the most important and frequent dependencies between the phases, while the outer circle symbolizes the cyclical nature of data mining itself and illustrates that the lessons learned during the data mining process and from the deployed solution can trigger new, often more focused business questions” (Shearer 14). This means that this model emphasizes that in data mining, we can learn new things along the way, so we might have to go back to a previous phase to ensure we are completing our project in an effective and efficient manner.
In each of the case studies, some of the sections described to completing this project aligns with the CRISP-DM framework, while some of the steps taken happen at a different time than described in the framework. Thus, in this report, I will be organizing each of the case studies in a manner that matches the CRISP-DM framework and its order of the phases. I will be using evidence from the text to discuss the issues covered and issues not covered in each phase listed above. After doing this, I will summarize the benefits and weaknesses revealed in each case study. Lastly, I will conclude what I have learned from the case studies and how it aligns with the CRISP-DM framework.


Case Study 1
Business Understanding
Issues covered (including evidence from Case Study Text)
“Our overall objective was to reduce the delinquency in child care payments through the design of intervention policies and programs. Our analytic objective in support of this effort was two-fold:
Use historical data with actual payment information to train and fine tune a predictive model.
Apply this model to data with an unknown target in order to assign a probability of payment” (2).
In this text, it is evident that the business objective and data mining goals are defined. The overall objective is to reduce delinquency in child care payments. The plan is to design intervention policies and programs that can help improve compliance of completing child care payments. The paper also states that historical data will be utilized to train and fine tune a predictive model making it clear that using data mining can help with decision-making and improving enforcement strategies. Another issue that is covered is that key stakeholders and their impact are mentioned. Agencies and policymakers involved with child support enforcement will use the insights gained from the project to help in the reduction of delinquency in child care payments by creating the most efficient and effective policies and programs possible.
“The intent is to identify what factors contribute to an NCP becoming delinquent in making child support payments”
This text shows that the goal is to identify factors contributing to delinquency, which aligns with the business understanding phase as it highlights key business factors, such as root causes to the initial problem that needs to be solved.
“Domain experts subsequently review and analyze significant parameters to identify non-payment trends and develop intervention policies and programs to address the issues uncovered.”
By utilizing domain experts in this data mining project, it ensures that the model is aligned with real-world enforcement practices and not purely based on data insights.


“As new data becomes available, we score the new data in an effort to identify those individuals with a high probability of non-payment. Where possible, child support counselors work with the high-risk NCPs to enroll them in a program, or take other action to avoid nonpayment.”




In this case study, child support counselors are taking action, using data-driven insights to enroll them individuals in programs or follow policies, demonstrating how data mining translates into solving real-world problems.
Issues not covered
Some issues that were not covered in the Business Understanding phase of Case Study 1 are:
Success metrics are undefined. There is no mention of how success and effectiveness will be measured.
Potential risks, challenges, and resource constraints not mentioned. There is no mention of issues such as privacy concerns, bias/fairness. There is also no mention about potential constraints, including data reliability issues or legal constraints.
Data Understanding
Issues covered (including evidence from Case Study Text)
“Data issues to be considered included the following:
How do we combine multiple records?
Which records do we include?
Do we need additional variables?
What is the relevant date range?”
In this text, the issues that are covered are data integration challenges and defining important information, such as relevant date range. The questions listed above highlight the need for understanding data structure and consistency. Moreover, having a clear date range established, historical data in this case, ensures that the dataset represents and aligns with business objectives.
“An NCP was considered delinquent the first time they had a case in which arrears were greater than 100% of their monthly child support obligation; in other words, the point at which the member went 30 days or more without paying the full obligation”
Here, having a definition for when an NCP is considered delinquent, the target variable, makes sure that the dataset captures relevant information regarding delinquency trends.
“The transaction data contained multiple records for each NCP (member). Each row in the original data represented a unique child support order and Member ID combination. A child support order is also referred to as a “case” and is defined by a single Custodial Parent (CP) but may represent multiple dependents”
Here, the dataset is briefly described and how the transactional data represents child support cases, custodial parents, and dependents. This provides insight into the data sources and covers the issue of exploratory data analysis.
Issues not covered
Issues that were not covered in the Data Understanding section of Case Study 1 are:
There was no discussion of if there were any biases presented in the data. For example, if there was bias towards certain payer demographics or specific enforcement actions.
There was also no justification for data inclusion and exclusion. There was no information about why records, features, and time periods were used or not.
The issue of sensitive data was not brought and how they were ensuring compliance to handling such data.
Lastly, it is not clear that the data is entirely valid or reliable.

Data Preparation
Issues covered (including evidence from Case Study Text)
“As part of the data preparation, these transactional records were combined at the member level”
The text states that transactional records were combined at the member level in the data preparation phase. Doing this shows that the case study covers the issue of ensuring proper structuring of the dataset for the predictive model.
“Additional metrics were created from the transaction (case) level data:
Frequent Mover flag
Job Jumper flag
Various counters (number of employers, number of children born out-of-wedlock)
Indicator flags (home phone, work phone, e-mail address, employment status)”
There were new variables/features created, which helps the predictive model perform better in the Modeling phase.
“Once the data mining database was created at the member level and target correlations examined, additional data exploration and data preparation were required to address the following:
Missing Values
Filter Outliers, establish reasonable caps
Transformation into grouped variables
Skewed variable distribution”
This ensures that data quality issues are handled.
Issues not covered
While data quality issues are handled, it is not explained how this will be done.
We are unsure if the data is balanced for fair predictions
Other issues not covered are feature selection validation, normalization, justification for feature choices
Issues covered (including evidence from Case Study Text)
“This combination of increased dimension and complexity results in decreased generalization-- which is counter to our goal. In training the model, it is important to look out for three types of variables that do not provide value-add:
Proxy variable is a variable whose effect mirrors that of another variable already included in the model
Extraneous input is a variable that provides no additional predictive value to the model
Variables with known data quality issues are omitted from the modeling process”
Ensuring the model is able to generalize and decrease its complexity, the first issue that's covered is by removing redundant and unimportant variables that can negatively affect the model’s performance.
“Because we had a target variable, we initially began applying supervised modeling techniques such as decision trees and regression. We soon discovered that the data collected did not support these. As previously mentioned, this client was in the initial stages of integrating data mining processes into their decision support systems. As such, they were also in the initial stages of both collecting and cleansing data. One of the main benefits of mining at this stage is recognizing what data is available, necessary, and useful. Because most of the data available to us was demographic in nature, we speculated that if we clustered the data prior to any predictive modeling attempts, we might increase the robustness of our model”
The issue covered here is choosing a modeling technique. Supervised modeling techniques were initially picked and limitations were revealed. Thus, a new modeling technique was introduced, which will be discussed in the next point.
“We decided to explore unsupervised modeling methods, which do not use the target variable. We obtained mild improvements by clustering the data prior to any predictive modeling attempts. “
By using a different modeling technique, unsupervised learning techniques, there were some improvements in the modeling process.
Other issues covered in this section were cluster analysis, which helped identify important patterns. The supervised and unsupervised methods were combined to help with the accuracy of the model.
Moreover, "using lift charts, misclassification rates, and averaged squared error as our assessment criteria, we determined that decision trees provided the best fit for this data with respect to performance" shows that reasoning behind using decision trees and describes the model. Another issue covered was the assessment of the model. The text states that "the following parameters were deemed significant in our final models: Maximum number of consecutive months paid, Total monthly obligation, Job Jumper flag, Total number of months ever paid, Count of cases for each NCP where initial arrears existed."
Issues not covered
There was no reasoning behind why other models besides decision trees were not considered.
Hyperparameter tuning and model optimization
Detecting and mitigating biases in predictive modeling

Issues covered (including evidence from Case Study Text)
"Using lift charts, misclassification rates, and averaged squared error as our assessment criteria, we determined that decision trees provided the best fit for this data with respect to performance."
By using performance metrics, the issue that is being covered is assessing the model’s accuracy.
"Child support agencies generally have limited funds to invest toward identifying potential delinquent payers. The fact that a decision tree will identify more non-payers up front is desirable. Trees also consistently provided slightly lower misclassification rates"
The model is being evaluated on how suitable it is for child support agencies. Here, we see that decision trees are the most practical and cost-effective model for implementation.
Issues not covered
Robustness and Generalization of the Model
No mention of how to model will perform long-term and what will be done to keep track of this.
There was no mention on how the model can affect those involved (ethical issues).

Issues covered (including evidence from Case Study Text)
"Over the long term, this state will move towards deployment as follows: Apply the model to new data as it becomes available, Determine a threshold probability for potential delinquency, Identify NCPs who are above the threshold."
The case mentions their approach to deployment and that it will be used long-term.
“Determine which policy/program will be most effective, Engage the NCP in a delinquency prevention program."
The model will be utilized and integrated with policy and decision-making.
"In the future they plan to further explore the use of predictive modeling techniques, especially as they are able to add data about the agency’s interactions with the NCPs."
There is mention that there is a commitment to data improvement over time.
Issues Not Covered
No mention of how the model’s performance will be monitored and updated
No discussion on ethical and legal considerations in deployment
No mention of scalability and resource allocation
No discussion of how false positives and false negatives will be handled.
No mention of how stakeholders will be involved.

Summary Comments
The business objective and the goals of the data mining project were clearly outlined. It was also clear that they would be utilizing data mining techniques.
The case study aligns with the CRISP-DM framework for the most part. This is helpful in meeting the goals of the project.
The combined use of supervised and unsupervised learning techniques for clustering helped with prediction accuracy.
Adding and modifying features also improved the predictive model.
Decision Trees were used as it was the most cost-effective and practical model for this case.
Using lift charts, misclassification rates, and averaged square area helped in the evaluation phase, which ensured the most effective approach was selected.
Failures or weaknesses
Although there was a brief discussion on model selection, there was no comparison made with other model options.
There was no discussion on handling or detecting bias, which can negatively impact certain groups of people.
There is no plan on how the model will perform over time and how this will be monitored.
There is no discussion on how this project can impact people socially and legally, especially with false positives.


Case Study 2
Business Understanding
Issues Covered
“The state wanted to use data mining to resolve issues including: What cases have the highest probability of paying continuously? For cases in which the children are supported by welfare, how can they increase the probability of finding and requiring the NCPs to pay? What child support enforcement activities are the most productive?”
Here, the case study defines and outlines the business objectives of the data mining project.
“Initially, the most important data issue for the problem definition for State B was whether to model at the case level or at the NCP level. This state can only take legal actions on an NCP by case. However, because we were ultimately trying to model the behavior of an individual and not a case, we decided to compile the data at the NCP level.”
In this text, the situation of the case is assessed and risks/constraints are identified. Doing this helps with deciding which model will be most effective.
“We quickly determined that defining who was delinquent was not a simple task... an NCP could be classified as ‘not delinquent’ only if they had paid the full amount for six months.”
Delinquency in this project is defined as the target variable.
“This strict definition resulted in the ‘non-delinquency’ classification being quite a rare event and could possibly confound model effects looking for differences between delinquent and non-delinquent payers.”
In this study, it is recognized that potential challenges in the data understanding and preparation phases may arise.
“For this reason, we constructed an additional target variable in our model for testing. Our second target variable was an interval variable that contained the number of months paid in full by the NCP in the time period T1 to T6.”
The study ensures that additional target variables are defined, which helps the model consider different cases of delinquency.
Issues Not Covered
There is no mention on how cost-effective predictive modeling would be compared to other strategies.
There is no discussion on potential biases in the data and how they will handle ethical considerations.
Compared to Case Study 1, there is no discussion about stakeholders, policies, and their impact on the project.
The study heavily focuses on predictive modeling, but does not discuss alternative strategies on improving compliance.
Similarly to the lack of mention of bias detection, there is no discussion on how false positives and false negatives can impact certain groups involved.
Data Understanding
Issues Covered
A comprehensive list of data sources, such as tables, and variables.
"In contrast to State A, we had more information on the behavior of the NCP and their interactions with the child support collection agency and legal system."
Data based on behavior and interactions is included, which is key information in enhancing the predictive model’s accuracy.
"For many variables, we were able to calculate not only the current value, but also whether any changes had occurred in the month previous to T7."
Here, we see that the dataset being used tracks changes over time, which is helpful in uncovering evolving patterns in delinquency.
Issues Not Covered
The quality of data is not assessed/considered. For example, missing data and inconsistencies are not discussed.
No reasoning behind why specific data is included and excluded.
No discussion of potential biases in the data being collected.
No mention on how sensitive information is being handled
No discussion on compliance considerations regarding privacy.
Data Preparation
Issues Covered
"As outlined in Case Study 1, data cleansing was also necessary. For example, NCPs with more than three employers on file were given a count value = 33 in the system, which would unnecessarily skew results."
There is a focus on data cleaning and handling errors. This helps with accuracy and precision in the predictions.
The study also incorporated multiple data sources into one dataset. This data preparation step is crucial in helping the model perform better.
Issues Not Covered
There was no discussion on how missing/incomplete data would be handled.
No mention of the standardization and normalization of variables, which can negatively affect the model’s performance.
Similar to Case Study 1, there is no mention on how data would be balanced for fair representation.
No reasoning behind why variables are selected/transformed.
Modeling Phase
Issues Covered
“Non-delinquent payers as defined by the binary target were very rare, so we over-sampled our data to create a modeling data set with approximately 1/3 non-delinquent payers and 2/3 delinquent payers.”
The issue of class imbalance is acknowledged, so oversampling techniques are utilized to avoid biases in the model.
“Based on our model development using the binary target versus the interval target, we found significant improvement in model accuracy using the target variable that describes the number of months paid consecutively in the six-month period.”
To improve the accuracy of the predictions, alternative target variable definitions are tested and refined.
“State B sought to gain additional data points. The state enforcement agency was able to purchase a financial score from a major credit reporting agency for a number of their cases. Also, they were able to create an indicator variable on whether the NCP is receiving any benefits at the federal level.”
The project uses additional data sources as a way to revise the model and improve its accuracy.
The type of model used for the project is also mentioned, which is a logistic regression model.
Issues Not Covered
No justification on why logistic regression was picked over other models
There is no mention of hyperparameter tuning for model optimization
There is no discussion on how biases in the model will be handled
No discussion on maintenance of the model over long periods of time.
Evaluation Phase
Issues Covered
“Beyond using validation and test data sets to validate the model, State B wanted to implement a field test of the results.”
The model is tested in real-world situations, which helps validate the model’s performance.
 “Rather than scoring a new set of cases and sending top scoring cases to the field, they tested the model on the delinquent cases in the model. They took the cases with the highest predicted probability to NOT be delinquent, but who were actually delinquent as defined by the target variable.”
By looking at and testing false negatives, it helps identify where the model went wrong and people who were incorrectly classified.
 “In all, they sent 218 cases to three offices in the field to investigate. In 60% of those cases, the field offices were able to work the case resulting in some next action such as filing new legal actions or issuing an automatic withholding.”
The model’s success rate was measured and acknowledged.
Issues Not Covered
There is no discussion on the details of performance metrics
There is no discussion on if the model is cost-effective.
Deployment Phase
Issues Covered
“The results of this effort were successful enough that the agency felt confident in two things: purchasing financial scores was worth the monetary investment and the model was appropriate for continued use.”
The model was implemented in a real-world scenario showing that the model is practical.
“The agency could first use the Sequence Analysis Node in SAS Enterprise Miner to determine all implemented legal action paths.”
 This discusses and considers future enhancements and adjustments to the model.
Issues Not Covered
There is no mention of how the model will be monitored and maintained.
No discussion on how stakeholders will be involved.

Summary Comments
The project aligns with the CRISP-DM framework.
Class imbalance is addressed by using oversampling techniques.
Additional data sources integrated further improving predictive accuracy.
Refinement to target variable definitions helped with model performance by reducing misclassification errors.
Model was used in a real-world scenario validating its practicality.
Mention of future enhancements and improvements to model
Failures/weaknesses
No reasoning behind choosing a logistic regression model over other models.
No discussion on handling and detecting biases
There was no plan on how the model will be monitored and maintained over time.
No discussion on how false positives and false negatives will be handled and how it can impact certain individuals.
Resource allocation was unclear.
No mention on how stakeholders will be involved.


Conclusion (what have you learned from these two cases)
After reading through the case studies and aligning them with the CRISP-DM framework, it is clear that data mining projects must be handled differently. By using predictive modeling for real-world situations, such as child support enforcement, it is important that models should be continuously refined, monitored, and maintained. While each project covered many important issues regarding each phase, there were several that were not mentioned. These projects must implement and consider things like privacy and bias and fairness concerns.


"""

YOURS2 = """IST341_Participant_7
Huiru Hu
Jiadi Liu
Nitipon (Tony) Moonwichit

Marvel Universe Social Network Analysis and Report

Introduction and Research Question(s)
The Marvel Universe is one of the most interconnected fictional worlds ever created, encompassing over 6,000 characters across comics, films, television, and other media. Its complex narrative structure, characterized by a web of alliances, rivalries, and collaborations, provides a unique opportunity to study network dynamics within a fictional framework. While it may initially appear as mere entertainment, the Marvel Universe functions as a dynamic social network where characters are the nodes, and their interactions form the edges. This intricate structure is not just a storytelling mechanism but also a reflection of real-world social systems. Analyzing this network enables researchers to uncover patterns of influence, hierarchy, and community, much like those observed in human societies.This study focuses on two pivotal research questions:

Who are the most influential and central characters in the Marvel Universe based on centrality measures?
What clusters of superheroes emerge using community detection algorithms?

The first research question seeks to identify key characters who serve as connectors or hubs within the network. Centrality measures such as degree, betweenness, and eigenvector centrality offer insights into the importance of specific nodes in a network. Characters like Spider-Man and Iron Man, for example, are known to interact with diverse groups across the Marvel Universe, making them central figures in terms of influence and connectivity. Identifying these central characters allows us to understand which superheroes act as bridges between different communities, enabling collaboration and driving the overarching narrative. These insights can reveal the storytelling strategies employed by Marvel’s writers, who often design central characters to play pivotal roles in uniting disparate groups during major crossover events or conflicts.
The second research question delves into the natural clustering of superheroes within the Marvel Universe. Using community detection algorithms, we can uncover groups or clusters of characters that form based on shared traits, missions, or narrative arcs. These clusters often represent well-known superhero teams such as the Avengers, X-Men, or Guardians of the Galaxy, each characterized by dense internal connections and distinct roles within the broader network. For example, the X-Men cluster frequently highlights themes of inclusion and fighting for civil rights, while the Avengers cluster focuses on leadership, teamwork, and the challenges of managing powerful personalities. Beyond established teams, community detection can reveal unexpected alliances or subgroups, providing insights into how Marvel’s creators structure relationships to enhance narrative complexity and depth.
The rationale for this research lies in the parallel between fictional superhero networks and real-world social systems. Just as human social networks exhibit certain patterns—such as small-world phenomena, preferential attachment, and high transitivity—these patterns are likely mirrored in the Marvel Universe. For instance, small-world networks are characterized by short path lengths, allowing for efficient communication and interaction among nodes. Similarly, preferential attachment explains why certain nodes, such as popular characters, attract more connections over time, reinforcing their centrality in the network. High transitivity, or the tendency for nodes to form tightly connected clusters, mirrors the cohesive nature of superhero teams. By analyzing these features within the Marvel Universe, we can gain insights into the storytelling mechanisms that make these narratives so compelling and relatable to audiences.
Furthermore, this research contributes to a broader understanding of how fictional networks are crafted to resonate with real-world social dynamics. Marvel’s writers often use interconnected narratives to create a sense of continuity and interdependence, emphasizing themes of teamwork, leadership, and the balance of individual and collective goals. These themes are central to human experiences, making the Marvel Universe a mirror of the societies we live in. By exploring how characters interact and influence one another, this study not only enhances our appreciation of the Marvel Universe but also illustrates the relevance of network theory in analyzing both fictional and real-world systems.
Ultimately, the analysis of the Marvel Universe as a superhero network bridges the gap between entertainment and academic inquiry. It underscores the value of applying social network theory to understand the dynamics of fictional worlds and their reflection of societal structures. By answering the proposed research questions, this study aims to illuminate the roles of central characters, the formation of superhero clusters, and the broader implications of these dynamics for storytelling and cultural representation. In doing so, it demonstrates how the Marvel Universe serves as a fascinating microcosm for exploring the complexity of human and superhuman relationships.

Popularity and Cultural Impact of Comics
Comics, especially superhero narratives, have captivated audiences for decades due to their ability to combine action, moral dilemmas, and relatable themes. They provide escapism while tackling universal issues such as justice, resilience, and societal conflict. Scholars like McCloud (1993) emphasize the unique power of comics to blend visual and textual storytelling, creating immersive experiences that resonate emotionally with readers. Superheroes themselves serve as cultural icons, embodying the ideals, struggles, and aspirations of the eras in which they were created. From Captain America's patriotic symbolism during World War II to the inclusive narratives surrounding heroes like Black Panther and Ms. Marvel, these characters have provided a platform for societal commentary and the representation of diverse perspectives. Moreover, the appeal of superhero narratives lies in their emphasis on community and connection—heroes rarely work in isolation but are part of larger networks of allies and adversaries. Also, superhero stories often extend beyond entertainment, offering nuanced critiques of social issues. The X-Men, for example, are frequently interpreted as allegories for marginalized communities, exploring themes of acceptance and prejudice. Similarly, Avengers storylines highlight themes of teamwork and leadership, while inter-team rivalries reflect real-world political and ideological conflicts. These elements make the Marvel Universe not only a fictional narrative but also a lens through which societal dynamics are examined.

Social Network Theory and its Application to Marvel
Social network theory offers a robust framework for analyzing the complex relationships within the Marvel Universe. At its core, the theory posits that individual actors (nodes) are interconnected through relationships (edges), forming networks that influence behaviors, roles, and outcomes. Within the superhero network, centrality measures such as degree, betweenness, and eigenvector centrality help identify influential characters who act as connectors or hubs, linking disparate groups and facilitating collaboration.
Furthermore, community detection algorithms, grounded in modularity optimization, uncover clusters of superheroes—groups of characters with dense internal connections but fewer ties to the rest of the network. These clusters often reflect narrative dynamics, such as teams like the Avengers or X-Men, and provide insights into the storytelling strategies that highlight collaboration and conflict.
The study of the Marvel superhero network through these lenses allows us to explore how fictional worlds emulate real-world social structures. Just as individuals in human social networks play distinct roles—leaders, brokers, or peripheral actors—superheroes fulfill similar functions within their universe. This analysis not only deepens our understanding of the Marvel Universe but also illustrates the power of social network theory in unraveling the complexity of human (and superhuman) relationships.

Description of the Dataset
Obtaining from Kaggle (https://www.kaggle.com/datasets/csanhueza/the-marvel-universe-social-network), this is how the data set looks like:

The data set is a 574,467 by 2 table which contains the network of heroes which appear together in the comics. This structure suggests that the dataset could be used to construct a social network, where each hero serves as a node, and the interactions between them form edges connecting these nodes. This format is particularly suitable for studying various network-based attributes, such as five-number summary, key influences, or community detection.

Network Visualization(s)
We found that, after creating a social network using the data set, our network has four connected components where one of them is particularly large and the other three are negligible as they only have 18 vertices combined. The following is the largest connected component of the network and another one below it is the same network with community detection.

Social Network Analysis of the Marvel Universe
As mentioned before, the Marvel Universe is a fictional realm made up of hundreds of characters, storylines, and relationships. The network is filled with characters that interact with each other through different movies, comics, and series, which we can explore through social network analysis tools in R. In our social network analysis, we computed the five-number summary, centrality measures, and community detection. We can utilize these tools to answer our questions about who the most influential superheroes are and what community detection can tell us about the Marvel Universe.
The first thing we did was import the necessary data to analyze the Marvel Universe. Then, we needed to convert the CSV to a graph, which allows us to do the necessary computations.
# import data
hero_network <- read.csv("hero-network.csv")
# convert network to graph
hero_graph <- graph_from_data_frame(hero_network, directed = FALSE)

Five-number Summary
After importing and converting the data, we focused on writing code to compile the five-number summary. To be specific, the five-number summary consists of the network size, density, number of components, diameter, and the clustering coefficient. This gives us information about the overall structural properties of the Marvel Universe.
The network size is the total number of nodes in the network.
# network size
network_size <- vcount(hero_graph)

The density is the proportion of edges to all possible edges in the network. This information tells us how interconnected the network is.
# density
density <- edge_density(hero_graph)

The next property we looked at was the number of components, which tell us the total number of disconnected subgraphs.
# components
components <- components(hero_graph)$no

The diameter of the largest component in the marvel universe which is the longest shortest path between any two heroes in the largest component.
# diameter
largest_component <- induced_subgraph(hero_graph, which(components(hero_graph)$membership == which.max(components(hero_graph)$csize)))

diameter <- diameter(largest_component, directed = FALSE)

Here, we measured the clustering coefficient, which is the fraction of closed triplets over all connected triplets, both open and closed. In other words, how likely it is that two heroes connected to one hero also have a relationship with each other.
clustering_coefficient <- transitivity(hero_graph)

Here, we compiled all of the computations together, so the results are returned in an organized list.
# five number summary
list(
  network_size = network_size,
  density = density,
  components = components,
  diameter = diameter,
  clustering_coefficient = clustering_coefficient
)
## $network_size
## [1] 6426
##
## $density
## [1] 0.02782795
##
## $components
## [1] 4
##
## $diameter
## [1] 5
##
## $clustering_coefficient
## [1] 0.1945397

Centrality Measures and Correlation Matrix
The second part of the social network analysis focuses on calculating the following centrality measures: degree, betweenness, closeness, and eigenvector. The degree centrality is the number of edges connected to a hero. Then, we looked at betweenness centrality, which measures how often a hero lies on the shortest paths between other heroes. We also computed closeness centrality to measure how close a hero is to all other heroes in the marvel universe. Lastly, we computed the eigenvector centrality, which measures a hero’s influence based on the importance of its neighbors.
# centrality calculations
# Degree Centrality
degree_cent <- degree(hero_graph)

# Betweenness Centrality
betweenness_cent <- betweenness(hero_graph, normalized = TRUE)

# Closeness Centrality
closeness_cent <- closeness(hero_graph, normalized = TRUE)

# Eigenvector Centrality
eigenvector_cent <- eigen_centrality(hero_graph)$vector

After writing the code to calculate each centrality measure, we created a dataframe of each of them and ordered the heroes based on centrality measures to easily identify the most influential and central nodes in the Marvel Universe.
# create dataframe of centralities
centrality_df <- data.frame(
  Node = names(degree_cent),
  Degree = degree_cent,
  Betweenness = betweenness_cent,
  Closeness = closeness_cent,
  Eigenvector = eigenvector_cent
)
# order heroes by importance based on degree
degree_df <- centrality_df[order(-centrality_df$Degree), ]
head(degree_df, 5)
##                                      Node Degree Betweenness Closeness
## CAPTAIN AMERICA           CAPTAIN AMERICA  16499  0.09400364 0.5853280
## SPIDER-MAN/PETER PAR SPIDER-MAN/PETER PAR  13717  0.12981558 0.5757031
## IRON MAN/TONY STARK  IRON MAN/TONY STARK   11817  0.05176861 0.5630053
## THOR/DR. DONALD BLAK THOR/DR. DONALD BLAK  11427  0.03736845 0.5524228
## THING/BENJAMIN J. GR THING/BENJAMIN J. GR  10681  0.03308810 0.5593191
##                      Eigenvector
## CAPTAIN AMERICA        1.0000000
## SPIDER-MAN/PETER PAR   0.4428942
## IRON MAN/TONY STARK    0.7726545
## THOR/DR. DONALD BLAK   0.6848890
## THING/BENJAMIN J. GR   0.7991839
# order heroes by importance based on betweenness
betweenness_df <- centrality_df[order(-centrality_df$Betweenness), ]
head(betweenness_df, 5)
##                                      Node Degree Betweenness Closeness
## SPIDER-MAN/PETER PAR SPIDER-MAN/PETER PAR  13717  0.12981558 0.5757031
## CAPTAIN AMERICA           CAPTAIN AMERICA  16499  0.09400364 0.5853280
## IRON MAN/TONY STARK  IRON MAN/TONY STARK   11817  0.05176861 0.5630053
## WOLVERINE/LOGAN          WOLVERINE/LOGAN   10353  0.05156761 0.5564530
## DR. STRANGE/STEPHEN  DR. STRANGE/STEPHEN    5576  0.04107434 0.5387656
##                      Eigenvector
## SPIDER-MAN/PETER PAR   0.4428942
## CAPTAIN AMERICA        1.0000000
## IRON MAN/TONY STARK    0.7726545
## WOLVERINE/LOGAN        0.4507776
## DR. STRANGE/STEPHEN    0.2247140
# order heroes by importance based on closeness
closeness_df <- centrality_df[order(-centrality_df$Closeness), ]
head(closeness_df, 5)
##                          Node Degree  Betweenness Closeness Eigenvector
## LUDLUM, ROSS     LUDLUM, ROSS     13 1.615214e-08         1           0
## ORWELL                 ORWELL     13 1.615214e-08         1           0
## ASHER, MICHAEL ASHER, MICHAEL     13 1.615214e-08         1           0
## FAGIN                   FAGIN     13 1.615214e-08         1           0
## HOFFMAN               HOFFMAN     13 1.615214e-08         1           0
# order heroes by importance based on eigenvector
eigen_df <- centrality_df[order(-centrality_df$Eigenvector), ]
head(eigen_df, 5)
##                                      Node Degree Betweenness Closeness
## CAPTAIN AMERICA           CAPTAIN AMERICA  16499  0.09400364 0.5853280
## THING/BENJAMIN J. GR THING/BENJAMIN J. GR  10681  0.03308810 0.5593191
## HUMAN TORCH/JOHNNY S HUMAN TORCH/JOHNNY S  10237  0.02616542 0.5564047
## IRON MAN/TONY STARK  IRON MAN/TONY STARK   11817  0.05176861 0.5630053
## MR. FANTASTIC/REED R MR. FANTASTIC/REED R   9775  0.02743416 0.5576153
##                      Eigenvector
## CAPTAIN AMERICA        1.0000000
## THING/BENJAMIN J. GR   0.7991839
## HUMAN TORCH/JOHNNY S   0.7850654
## IRON MAN/TONY STARK    0.7726545
## MR. FANTASTIC/REED R   0.7715652

Based on the centrality measures, we also created a correlation matrix where each cell represents the correlation between two centrality measures. The numbers in each cell reveal the relationships between centrality measures. Doing this helps us identify relationships by understanding which centrality measures are aligned and which are distinct.
# correlation matrix
cor_matrix <- cor(centrality_df[, -1])
cor_matrix
##                Degree Betweenness Closeness Eigenvector
## Degree      1.0000000   0.7959734 0.4207373   0.9549524
## Betweenness 0.7959734   1.0000000 0.2413997   0.7220559
## Closeness   0.4207373   0.2413997 1.0000000   0.3597889
## Eigenvector 0.9549524   0.7220559 0.3597889   1.0000000

Community Detection Algorithm: Walktrap
The last section of our social network analysis is community detection to identify clusters of heroes in the network. We used the Walktrap algorithm, which is based on the principle that random walks on the graph are more likely to stay within densely connected communities than to move between them. This algorithm uses short random walks for community detection, provides multiple community levels, is ideal for strongly cohesive clusters, and considers modularity for partitioning.
walktrap_comm <- cluster_walktrap(hero_graph)
walktrap_num_comm <- length(unique(membership(walktrap_comm)))
walktrap_mod <- modularity(walktrap_comm)
walktrap_num_comm
## [1] 273
walktrap_mod
## [1] 0.437035


Results and Implications for Theory and Practice in the Marvel Superhero Network
The Marvel Universe superhero network provides a unique lens for exploring network theory, centrality, and community dynamics. Through network analysis, we have uncovered key structural properties, identified influential characters, and analyzed hero group formation. This essay discusses our findings and their implications for theoretical understanding and practical applications in real-world leadership, collaboration, and community-building.

Network Overview
The Marvel Universe comprises 6,426 unique heroes, forming a moderately sparse network with a density of 2.78%. The network's diameter is 5, reflecting a small-world effect where any hero can connect to another in at most five steps. The clustering coefficient of 0.1945 indicates limited local clustering, suggesting that heroes connected to the same central figure are less likely to collaborate directly. Despite this, the network remains cohesive, with only four disconnected components.

Centrality Analysis
Key centrality measures reveal the most influential heroes:
Spider-Man: High degree and eigenvector centrality position him as a global connector and influencer.
Iron Man: High betweenness centrality highlights his role as a bridge linking disparate groups.
Captain America: High closeness centrality underscores his efficiency in reaching other nodes.
Wolverine: High team-up frequencies make him a critical connector across teams.
These characters are pivotal in maintaining network cohesion and driving intergroup collaborations.

Community Detection
Using the Walktrap algorithm, we identified 273 communities, each representing groups of heroes with frequent interactions. Examples include:
Avengers: Led by Iron Man and Captain America.
X-Men: Centered around Wolverine, Cyclops, and Storm.
Fantastic Four: Featuring Mr. Fantastic, Invisible Woman, Human Torch, and The Thing.
A modularity score of 0.437 indicates a reasonably well-defined community structure, though crossovers blur boundaries between groups, reflecting Marvel's dynamic storytelling.

Implications for Theory

Leadership and Centrality
The high centrality values of characters like Captain America and Iron Man demonstrate that leadership traits—such as the ability to inspire and unite—are integral to network influence. This supports theories emphasizing the role of connectors and bridges in sustaining network cohesion and facilitating communication.

Community Formation and Modularity
The detection of 273 communities aligns with theories of modular networks, where tightly connected subgroups coexist within larger systems. Marvel's communities reflect real-world dynamics, such as shared goals, geographic proximity, and common traits or experiences. This supports the homophily principle, which posits that individuals with similar characteristics or goals are more likely to form connections.

Small-World Effect
The small diameter and reliance on central nodes confirm the small-world phenomenon, where information or influence can spread efficiently across even sparse networks. This theoretical concept is vital for understanding social systems, emphasizing the importance of global connectivity over dense local clustering.

Implications for Practice
Leadership Development
Understanding the traits of central heroes can inform leadership training:
Encouraging leaders to act as connectors by bridging isolated groups.
Fostering traits like inspiration and collaboration, seen in Captain America's ability to unite diverse teams.
Organizations can cultivate leaders who not only excel in direct interactions but also influence and connect broader systems.
Team Building
Marvel's community dynamics offer insights into group formation:
Teams formed around shared goals or geographic proximity are likely to perform well.
Crossovers between groups reflect the importance of inter-team communication in fostering innovation and resilience.
Practically, this can guide workplace collaboration strategies, such as forming project teams with complementary skill sets or encouraging inter-departmental networking.
Marketing and Storytelling
Marvel's network structure provides a blueprint for engaging audiences:
Characters like Spider-Man and Wolverine, who frequently collaborate, can anchor cross-team storylines or marketing campaigns.
Community overlaps suggest opportunities for bundled products or events appealing to fans of multiple teams.
Digital and Organizational Networks
Insights from Marvel's network can inform digital platforms and organizational structures:
Social media recommendation algorithms can mimic the small-world effect to connect users efficiently.
Organizations can design structures that balance modularity (team specialization) with global connectivity (inter-department collaboration).

The analysis of the Marvel superhero network bridges theory and practice, offering insights into leadership, group dynamics, and efficient connectivity. Heroes like Captain America and Spider-Man exemplify key roles as connectors and influencers, reflecting traits valued in real-world leaders. The modular community structure mirrors how human groups form around shared traits or goals, emphasizing universal principles of collaboration. These findings not only deepen our understanding of network science but also provide actionable strategies for enhancing leadership, storytelling, and organizational efficiency in various fields.
"""

THEIRS1 = """

     To Sherlock Holmes she is always the woman. I have seldom heard him
     mention her under any other name. In his eyes she eclipses and
     predominates the whole of her sex. It was not that he felt any
     emotion akin to love for Irene Adler. All emotions, and that one
     particularly, were abhorrent to his cold, precise but admirably
     balanced mind. He was, I take it, the most perfect reasoning and
     observing machine that the world has seen, but as a lover he would
     have placed himself in a false position. He never spoke of the softer
     passions, save with a gibe and a sneer. They were admirable things
     for the observer--excellent for drawing the veil from men's motives
     and actions. But for the trained reasoner to admit such intrusions
     into his own delicate and finely adjusted temperament was to
     introduce a distracting factor which might throw a doubt upon all his
     mental results. Grit in a sensitive instrument, or a crack in one of
     his own high-power lenses, would not be more disturbing than a strong
     emotion in a nature such as his. And yet there was but one woman to
     him, and that woman was the late Irene Adler, of dubious and
     questionable memory.

     I had seen little of Holmes lately. My marriage had drifted us away
     from each other. My own complete happiness, and the home-centred
     interests which rise up around the man who first finds himself master
     of his own establishment, were sufficient to absorb all my attention,
     while Holmes, who loathed every form of society with his whole
     Bohemian soul, remained in our lodgings in Baker Street, buried among
     his old books, and alternating from week to week between cocaine and
     ambition, the drowsiness of the drug, and the fierce energy of his
     own keen nature. He was still, as ever, deeply attracted by the study
     of crime, and occupied his immense faculties and extraordinary powers
     of observation in following out those clues, and clearing up those
     mysteries which had been abandoned as hopeless by the official
     police. From time to time I heard some vague account of his doings:
     of his summons to Odessa in the case of the Trepoff murder, of his
     clearing up of the singular tragedy of the Atkinson brothers at
     Trincomalee, and finally of the mission which he had accomplished so
     delicately and successfully for the reigning family of Holland.
     Beyond these signs of his activity, however, which I merely shared
     with all the readers of the daily press, I knew little of my former
     friend and companion.

     One night--it was on the twentieth of March, 1888--I was returning
     from a journey to a patient (for I had now returned to civil
     practice), when my way led me through Baker Street. As I passed the
     well-remembered door, which must always be associated in my mind with
     my wooing, and with the dark incidents of the Study in Scarlet, I was
     seized with a keen desire to see Holmes again, and to know how he was
     employing his extraordinary powers. His rooms were brilliantly lit,
     and, even as I looked up, I saw his tall, spare figure pass twice in
     a dark silhouette against the blind. He was pacing the room swiftly,
     eagerly, with his head sunk upon his chest and his hands clasped
     behind him. To me, who knew his every mood and habit, his attitude
     and manner told their own story. He was at work again. He had risen
     out of his drug-created dreams and was hot upon the scent of some new
     problem. I rang the bell and was shown up to the chamber which had
     formerly been in part my own.

     His manner was not effusive. It seldom was; but he was glad, I think,
     to see me. With hardly a word spoken, but with a kindly eye, he waved
     me to an armchair, threw across his case of cigars, and indicated a
     spirit case and a gasogene in the corner. Then he stood before the
     fire and looked me over in his singular introspective fashion.

     "Wedlock suits you," he remarked. "I think, Watson, that you have put
     on seven and a half pounds since I saw you."

     "Seven!" I answered.

     "Indeed, I should have thought a little more. Just a trifle more, I
     fancy, Watson. And in practice again, I observe. You did not tell me
     that you intended to go into harness."

     "Then, how do you know?"

     "I see it, I deduce it. How do I know that you have been getting
     yourself very wet lately, and that you have a most clumsy and
     careless servant girl?"

     "My dear Holmes," said I, "this is too much. You would certainly have
     been burned, had you lived a few centuries ago. It is true that I had
     a country walk on Thursday and came home in a dreadful mess, but as I
     have changed my clothes I can't imagine how you deduce it. As to Mary
     Jane, she is incorrigible, and my wife has given her notice, but
     there, again, I fail to see how you work it out."

     He chuckled to himself and rubbed his long, nervous hands together.

     "It is simplicity itself," said he; "my eyes tell me that on the
     inside of your left shoe, just where the firelight strikes it, the
     leather is scored by six almost parallel cuts. Obviously they have
     been caused by someone who has very carelessly scraped round the
     edges of the sole in order to remove crusted mud from it. Hence, you
     see, my double deduction that you had been out in vile weather, and
     that you had a particularly malignant boot-slitting specimen of the
     London slavey. As to your practice, if a gentleman walks into my
     rooms smelling of iodoform, with a black mark of nitrate of silver
     upon his right forefinger, and a bulge on the right side of his
     top-hat to show where he has secreted his stethoscope, I must be
     dull, indeed, if I do not pronounce him to be an active member of the
     medical profession."

     I could not help laughing at the ease with which he explained his
     process of deduction. "When I hear you give your reasons," I
     remarked, "the thing always appears to me to be so ridiculously
     simple that I could easily do it myself, though at each successive
     instance of your reasoning I am baffled until you explain your
     process. And yet I believe that my eyes are as good as yours."

     "Quite so," he answered, lighting a cigarette, and throwing himself
     down into an armchair. "You see, but you do not observe. The
     distinction is clear. For example, you have frequently seen the steps
     which lead up from the hall to this room."

     "Frequently."

     "How often?"

     "Well, some hundreds of times."

     "Then how many are there?"

     "How many? I don't know."

     "Quite so! You have not observed. And yet you have seen. That is just
     my point. Now, I know that there are seventeen steps, because I have
     both seen and observed. By-the-way, since you are interested in these
     little problems, and since you are good enough to chronicle one or
     two of my trifling experiences, you may be interested in this." He
     threw over a sheet of thick, pink-tinted note-paper which had been
     lying open upon the table. "It came by the last post," said he. "Read
     it aloud."

     The note was undated, and without either signature or address.

     "There will call upon you to-night, at a quarter to eight o'clock,"
     it said, "a gentleman who desires to consult you upon a matter of the
     very deepest moment. Your recent services to one of the royal houses
     of Europe have shown that you are one who may safely be trusted with
     matters which are of an importance which can hardly be exaggerated.
     This account of you we have from all quarters received. Be in your
     chamber then at that hour, and do not take it amiss if your visitor
     wear a mask."

     "This is indeed a mystery," I remarked. "What do you imagine that it
     means?"

     "I have no data yet. It is a capital mistake to theorize before one
     has data. Insensibly one begins to twist facts to suit theories,
     instead of theories to suit facts. But the note itself. What do you
     deduce from it?"

     I carefully examined the writing, and the paper upon which it was
     written.

     "The man who wrote it was presumably well to do," I remarked,
     endeavouring to imitate my companion's processes. "Such paper could
     not be bought under half a crown a packet. It is peculiarly strong
     and stiff."

     "Peculiar--that is the very word," said Holmes. "It is not an English
     paper at all. Hold it up to the light."

     I did so, and saw a large "E" with a small "g," a "P," and a large
     "G" with a small "t" woven into the texture of the paper.

     "What do you make of that?" asked Holmes.

     "The name of the maker, no doubt; or his monogram, rather."

     "Not at all. The 'G' with the small 't' stands for 'Gesellschaft,'
     which is the German for 'Company.' It is a customary contraction like
     our 'Co.' 'P,' of course, stands for 'Papier.' Now for the 'Eg.' Let
     us glance at our Continental Gazetteer." He took down a heavy brown
     volume from his shelves. "Eglow, Eglonitz--here we are, Egria. It is
     in a German-speaking country--in Bohemia, not far from Carlsbad.
     'Remarkable as being the scene of the death of Wallenstein, and for
     its numerous glass-factories and paper-mills.' Ha, ha, my boy, what
     do you make of that?" His eyes sparkled, and he sent up a great blue
     triumphant cloud from his cigarette.

     "The paper was made in Bohemia," I said.

     "Precisely. And the man who wrote the note is a German. Do you note
     the peculiar construction of the sentence--'This account of you we
     have from all quarters received.' A Frenchman or Russian could not
     have written that. It is the German who is so uncourteous to his
     verbs. It only remains, therefore, to discover what is wanted by this
     German who writes upon Bohemian paper and prefers wearing a mask to
     showing his face. And here he comes, if I am not mistaken, to resolve
     all our doubts."

     As he spoke there was the sharp sound of horses' hoofs and grating
     wheels against the curb, followed by a sharp pull at the bell. Holmes
     whistled.

     "A pair, by the sound," said he. "Yes," he continued, glancing out of
     the window. "A nice little brougham and a pair of beauties. A hundred
     and fifty guineas apiece. There's money in this case, Watson, if
     there is nothing else."

     "I think that I had better go, Holmes."

     "Not a bit, Doctor. Stay where you are. I am lost without my Boswell.
     And this promises to be interesting. It would be a pity to miss it."

     "But your client--"

     "Never mind him. I may want your help, and so may he. Here he comes.
     Sit down in that armchair, Doctor, and give us your best attention."

     A slow and heavy step, which had been heard upon the stairs and in
     the passage, paused immediately outside the door. Then there was a
     loud and authoritative tap.

     "Come in!" said Holmes.

     A man entered who could hardly have been less than six feet six
     inches in height, with the chest and limbs of a Hercules. His dress
     was rich with a richness which would, in England, be looked upon as
     akin to bad taste. Heavy bands of astrakhan were slashed across the
     sleeves and fronts of his double-breasted coat, while the deep blue
     cloak which was thrown over his shoulders was lined with
     flame-coloured silk and secured at the neck with a brooch which
     consisted of a single flaming beryl. Boots which extended halfway up
     his calves, and which were trimmed at the tops with rich brown fur,
     completed the impression of barbaric opulence which was suggested by
     his whole appearance. He carried a broad-brimmed hat in his hand,
     while he wore across the upper part of his face, extending down past
     the cheekbones, a black vizard mask, which he had apparently adjusted
     that very moment, for his hand was still raised to it as he entered.
     From the lower part of the face he appeared to be a man of strong
     character, with a thick, hanging lip, and a long, straight chin
     suggestive of resolution pushed to the length of obstinacy.

     "You had my note?" he asked with a deep harsh voice and a strongly
     marked German accent. "I told you that I would call." He looked from
     one to the other of us, as if uncertain which to address.

     "Pray take a seat," said Holmes. "This is my friend and colleague,
     Dr. Watson, who is occasionally good enough to help me in my cases.
     Whom have I the honour to address?"

     "You may address me as the Count Von Kramm, a Bohemian nobleman. I
     understand that this gentleman, your friend, is a man of honour and
     discretion, whom I may trust with a matter of the most extreme
     importance. If not, I should much prefer to communicate with you
     alone."

     I rose to go, but Holmes caught me by the wrist and pushed me back
     into my chair. "It is both, or none," said he. "You may say before
     this gentleman anything which you may say to me."

     The Count shrugged his broad shoulders. "Then I must begin," said he,
     "by binding you both to absolute secrecy for two years; at the end of
     that time the matter will be of no importance. At present it is not
     too much to say that it is of such weight it may have an influence
     upon European history."

     "I promise," said Holmes.

     "And I."

     "You will excuse this mask," continued our strange visitor. "The
     august person who employs me wishes his agent to be unknown to you,
     and I may confess at once that the title by which I have just called
     myself is not exactly my own."

     "I was aware of it," said Holmes dryly.

     "The circumstances are of great delicacy, and every precaution has to
     be taken to quench what might grow to be an immense scandal and
     seriously compromise one of the reigning families of Europe. To speak
     plainly, the matter implicates the great House of Ormstein,
     hereditary kings of Bohemia."

     "I was also aware of that," murmured Holmes, settling himself down in
     his armchair and closing his eyes.

     Our visitor glanced with some apparent surprise at the languid,
     lounging figure of the man who had been no doubt depicted to him as
     the most incisive reasoner and most energetic agent in Europe. Holmes
     slowly reopened his eyes and looked impatiently at his gigantic
     client.

     "If your Majesty would condescend to state your case," he remarked,
     "I should be better able to advise you."

     The man sprang from his chair and paced up and down the room in
     uncontrollable agitation. Then, with a gesture of desperation, he
     tore the mask from his face and hurled it upon the ground. "You are
     right," he cried; "I am the King. Why should I attempt to conceal
     it?"

     "Why, indeed?" murmured Holmes. "Your Majesty had not spoken before I
     was aware that I was addressing Wilhelm Gottsreich Sigismond von
     Ormstein, Grand Duke of Cassel-Felstein, and hereditary King of
     Bohemia."

     "But you can understand," said our strange visitor, sitting down once
     more and passing his hand over his high white forehead, "you can
     understand that I am not accustomed to doing such business in my own
     person. Yet the matter was so delicate that I could not confide it to
     an agent without putting myself in his power. I have come incognito
     from Prague for the purpose of consulting you."

     "Then, pray consult," said Holmes, shutting his eyes once more.

     "The facts are briefly these: Some five years ago, during a lengthy
     visit to Warsaw, I made the acquaintance of the well-known
     adventuress, Irene Adler. The name is no doubt familiar to you."

     "Kindly look her up in my index, Doctor," murmured Holmes without
     opening his eyes. For many years he had adopted a system of docketing
     all paragraphs concerning men and things, so that it was difficult to
     name a subject or a person on which he could not at once furnish
     information. In this case I found her biography sandwiched in between
     that of a Hebrew rabbi and that of a staff-commander who had written
     a monograph upon the deep-sea fishes.

     "Let me see!" said Holmes. "Hum! Born in New Jersey in the year 1858.
     Contralto--hum! La Scala, hum! Prima donna Imperial Opera of
     Warsaw--yes! Retired from operatic stage--ha! Living in London--quite
     so! Your Majesty, as I understand, became entangled with this young
     person, wrote her some compromising letters, and is now desirous of
     getting those letters back."

     "Precisely so. But how--"

     "Was there a secret marriage?"

     "None."

     "No legal papers or certificates?"

     "None."

     "Then I fail to follow your Majesty. If this young person should
     produce her letters for blackmailing or other purposes, how is she to
     prove their authenticity?"

     "There is the writing."

     "Pooh, pooh! Forgery."

     "My private note-paper."

     "Stolen."

     "My own seal."

     "Imitated."

     "My photograph."

     "Bought."

     "We were both in the photograph."

     "Oh, dear! That is very bad! Your Majesty has indeed committed an
     indiscretion."

     "I was mad--insane."

     "You have compromised yourself seriously."

     "I was only Crown Prince then. I was young. I am but thirty now."

     "It must be recovered."

     "We have tried and failed."

     "Your Majesty must pay. It must be bought."

     "She will not sell."

     "Stolen, then."

     "Five attempts have been made. Twice burglars in my pay ransacked her
     house. Once we diverted her luggage when she travelled. Twice she has
     been waylaid. There has been no result."

     "No sign of it?"

     "Absolutely none."

     Holmes laughed. "It is quite a pretty little problem," said he.

     "But a very serious one to me," returned the King reproachfully.

     "Very, indeed. And what does she propose to do with the photograph?"

     "To ruin me."

     "But how?"

     "I am about to be married."

     "So I have heard."

     "To Clotilde Lothman von Saxe-Meningen, second daughter of the King
     of Scandinavia. You may know the strict principles of her family. She
     is herself the very soul of delicacy. A shadow of a doubt as to my
     conduct would bring the matter to an end."

     "And Irene Adler?"

     "Threatens to send them the photograph. And she will do it. I know
     that she will do it. You do not know her, but she has a soul of
     steel. She has the face of the most beautiful of women, and the mind
     of the most resolute of men. Rather than I should marry another
     woman, there are no lengths to which she would not go--none."

     "You are sure that she has not sent it yet?"

     "I am sure."

     "And why?"

     "Because she has said that she would send it on the day when the
     betrothal was publicly proclaimed. That will be next Monday."

     "Oh, then we have three days yet," said Holmes with a yawn. "That is
     very fortunate, as I have one or two matters of importance to look
     into just at present. Your Majesty will, of course, stay in London
     for the present?"

     "Certainly. You will find me at the Langham under the name of the
     Count Von Kramm."

     "Then I shall drop you a line to let you know how we progress."

     "Pray do so. I shall be all anxiety."

     "Then, as to money?"

     "You have carte blanche."

     "Absolutely?"

     "I tell you that I would give one of the provinces of my kingdom to
     have that photograph."

     "And for present expenses?"

     The King took a heavy chamois leather bag from under his cloak and
     laid it on the table.

     "There are three hundred pounds in gold and seven hundred in notes,"
     he said.

     Holmes scribbled a receipt upon a sheet of his note-book and handed
     it to him.

     "And Mademoiselle's address?" he asked.

     "Is Briony Lodge, Serpentine Avenue, St. John's Wood."

     Holmes took a note of it. "One other question," said he. "Was the
     photograph a cabinet?"

     "It was."

     "Then, good-night, your Majesty, and I trust that we shall soon have
     some good news for you. And good-night, Watson," he added, as the
     wheels of the royal brougham rolled down the street. "If you will be
     good enough to call to-morrow afternoon at three o'clock I should
     like to chat this little matter over with you."





          CHAPTER II



     At three o'clock precisely I was at Baker Street, but Holmes had not
     yet returned. The landlady informed me that he had left the house
     shortly after eight o'clock in the morning. I sat down beside the
     fire, however, with the intention of awaiting him, however long he
     might be. I was already deeply interested in his inquiry, for, though
     it was surrounded by none of the grim and strange features which were
     associated with the two crimes which I have already recorded, still,
     the nature of the case and the exalted station of his client gave it
     a character of its own. Indeed, apart from the nature of the
     investigation which my friend had on hand, there was something in his
     masterly grasp of a situation, and his keen, incisive reasoning,
     which made it a pleasure to me to study his system of work, and to
     follow the quick, subtle methods by which he disentangled the most
     inextricable mysteries. So accustomed was I to his invariable success
     that the very possibility of his failing had ceased to enter into my
     head.

     It was close upon four before the door opened, and a drunken-looking
     groom, ill-kempt and side-whiskered, with an inflamed face and
     disreputable clothes, walked into the room. Accustomed as I was to my
     friend's amazing powers in the use of disguises, I had to look three
     times before I was certain that it was indeed he. With a nod he
     vanished into the bedroom, whence he emerged in five minutes
     tweed-suited and respectable, as of old. Putting his hands into his
     pockets, he stretched out his legs in front of the fire and laughed
     heartily for some minutes.

     "Well, really!" he cried, and then he choked and laughed again until
     he was obliged to lie back, limp and helpless, in the chair.

     "What is it?"

     "It's quite too funny. I am sure you could never guess how I employed
     my morning, or what I ended by doing."

     "I can't imagine. I suppose that you have been watching the habits,
     and perhaps the house, of Miss Irene Adler."

     "Quite so; but the sequel was rather unusual. I will tell you,
     however. I left the house a little after eight o'clock this morning
     in the character of a groom out of work. There is a wonderful
     sympathy and freemasonry among horsey men. Be one of them, and you
     will know all that there is to know. I soon found Briony Lodge. It is
     a bijou villa, with a garden at the back, but built out in front
     right up to the road, two stories. Chubb lock to the door. Large
     sitting-room on the right side, well furnished, with long windows
     almost to the floor, and those preposterous English window fasteners
     which a child could open. Behind there was nothing remarkable, save
     that the passage window could be reached from the top of the
     coach-house. I walked round it and examined it closely from every
     point of view, but without noting anything else of interest.

     "I then lounged down the street and found, as I expected, that there
     was a mews in a lane which runs down by one wall of the garden. I
     lent the ostlers a hand in rubbing down their horses, and received in
     exchange twopence, a glass of half and half, two fills of shag
     tobacco, and as much information as I could desire about Miss Adler,
     to say nothing of half a dozen other people in the neighbourhood in
     whom I was not in the least interested, but whose biographies I was
     compelled to listen to."

     "And what of Irene Adler?" I asked.

     "Oh, she has turned all the men's heads down in that part. She is the
     daintiest thing under a bonnet on this planet. So say the
     Serpentine-mews, to a man. She lives quietly, sings at concerts,
     drives out at five every day, and returns at seven sharp for dinner.
     Seldom goes out at other times, except when she sings. Has only one
     male visitor, but a good deal of him. He is dark, handsome, and
     dashing, never calls less than once a day, and often twice. He is a
     Mr. Godfrey Norton, of the Inner Temple. See the advantages of a
     cabman as a confidant. They had driven him home a dozen times from
     Serpentine-mews, and knew all about him. When I had listened to all
     they had to tell, I began to walk up and down near Briony Lodge once
     more, and to think over my plan of campaign.

     "This Godfrey Norton was evidently an important factor in the matter.
     He was a lawyer. That sounded ominous. What was the relation between
     them, and what the object of his repeated visits? Was she his client,
     his friend, or his mistress? If the former, she had probably
     transferred the photograph to his keeping. If the latter, it was less
     likely. On the issue of this question depended whether I should
     continue my work at Briony Lodge, or turn my attention to the
     gentleman's chambers in the Temple. It was a delicate point, and it
     widened the field of my inquiry. I fear that I bore you with these
     details, but I have to let you see my little difficulties, if you are
     to understand the situation."

     "I am following you closely," I answered.

     "I was still balancing the matter in my mind when a hansom cab drove
     up to Briony Lodge, and a gentleman sprang out. He was a remarkably
     handsome man, dark, aquiline, and moustached--evidently the man of
     whom I had heard. He appeared to be in a great hurry, shouted to the
     cabman to wait, and brushed past the maid who opened the door with
     the air of a man who was thoroughly at home.

     "He was in the house about half an hour, and I could catch glimpses
     of him in the windows of the sitting-room, pacing up and down,
     talking excitedly, and waving his arms. Of her I could see nothing.
     Presently he emerged, looking even more flurried than before. As he
     stepped up to the cab, he pulled a gold watch from his pocket and
     looked at it earnestly, 'Drive like the devil,' he shouted, 'first to
     Gross & Hankey's in Regent Street, and then to the Church of St.
     Monica in the Edgeware Road. Half a guinea if you do it in twenty
     minutes!'

     "Away they went, and I was just wondering whether I should not do
     well to follow them when up the lane came a neat little landau, the
     coachman with his coat only half-buttoned, and his tie under his ear,
     while all the tags of his harness were sticking out of the buckles.
     It hadn't pulled up before she shot out of the hall door and into it.
     I only caught a glimpse of her at the moment, but she was a lovely
     woman, with a face that a man might die for.

     "'The Church of St. Monica, John,' she cried, 'and half a sovereign
     if you reach it in twenty minutes.'

     "This was quite too good to lose, Watson. I was just balancing
     whether I should run for it, or whether I should perch behind her
     landau when a cab came through the street. The driver looked twice at
     such a shabby fare, but I jumped in before he could object. 'The
     Church of St. Monica,' said I, 'and half a sovereign if you reach it
     in twenty minutes.' It was twenty-five minutes to twelve, and of
     course it was clear enough what was in the wind.

     "My cabby drove fast. I don't think I ever drove faster, but the
     others were there before us. The cab and the landau with their
     steaming horses were in front of the door when I arrived. I paid the
     man and hurried into the church. There was not a soul there save the
     two whom I had followed and a surpliced clergyman, who seemed to be
     expostulating with them. They were all three standing in a knot in
     front of the altar. I lounged up the side aisle like any other idler
     who has dropped into a church. Suddenly, to my surprise, the three at
     the altar faced round to me, and Godfrey Norton came running as hard
     as he could towards me.

     "'Thank God,' he cried. 'You'll do. Come! Come!'

     "'What then?' I asked.

     "'Come, man, come, only three minutes, or it won't be legal.'

     "I was half-dragged up to the altar, and before I knew where I was I
     found myself mumbling responses which were whispered in my ear, and
     vouching for things of which I knew nothing, and generally assisting
     in the secure tying up of Irene Adler, spinster, to Godfrey Norton,
     bachelor. It was all done in an instant, and there was the gentleman
     thanking me on the one side and the lady on the other, while the
     clergyman beamed on me in front. It was the most preposterous
     position in which I ever found myself in my life, and it was the
     thought of it that started me laughing just now. It seems that there
     had been some informality about their license, that the clergyman
     absolutely refused to marry them without a witness of some sort, and
     that my lucky appearance saved the bridegroom from having to sally
     out into the streets in search of a best man. The bride gave me a
     sovereign, and I mean to wear it on my watch-chain in memory of the
     occasion."

     "This is a very unexpected turn of affairs," said I; "and what then?"

     "Well, I found my plans very seriously menaced. It looked as if the
     pair might take an immediate departure, and so necessitate very
     prompt and energetic measures on my part. At the church door,
     however, they separated, he driving back to the Temple, and she to
     her own house. 'I shall drive out in the park at five as usual,' she
     said as she left him. I heard no more. They drove away in different
     directions, and I went off to make my own arrangements."

     "Which are?"

     "Some cold beef and a glass of beer," he answered, ringing the bell.
     "I have been too busy to think of food, and I am likely to be busier
     still this evening. By the way, Doctor, I shall want your
     co-operation."

     "I shall be delighted."

     "You don't mind breaking the law?"

     "Not in the least."

     "Nor running a chance of arrest?"

     "Not in a good cause."

     "Oh, the cause is excellent!"

     "Then I am your man."

     "I was sure that I might rely on you."

     "But what is it you wish?"

     "When Mrs. Turner has brought in the tray I will make it clear to
     you. Now," he said as he turned hungrily on the simple fare that our
     landlady had provided, "I must discuss it while I eat, for I have not
     much time. It is nearly five now. In two hours we must be on the
     scene of action. Miss Irene, or Madame, rather, returns from her
     drive at seven. We must be at Briony Lodge to meet her."

     "And what then?"

     "You must leave that to me. I have already arranged what is to occur.
     There is only one point on which I must insist. You must not
     interfere, come what may. You understand?"

     "I am to be neutral?"

     "To do nothing whatever. There will probably be some small
     unpleasantness. Do not join in it. It will end in my being conveyed
     into the house. Four or five minutes afterwards the sitting-room
     window will open. You are to station yourself close to that open
     window."

     "Yes."

     "You are to watch me, for I will be visible to you."

     "Yes."

     "And when I raise my hand--so--you will throw into the room what I
     give you to throw, and will, at the same time, raise the cry of fire.
     You quite follow me?"

     "Entirely."

     "It is nothing very formidable," he said, taking a long cigar-shaped
     roll from his pocket. "It is an ordinary plumber's smoke-rocket,
     fitted with a cap at either end to make it self-lighting. Your task
     is confined to that. When you raise your cry of fire, it will be
     taken up by quite a number of people. You may then walk to the end of
     the street, and I will rejoin you in ten minutes. I hope that I have
     made myself clear?"

     "I am to remain neutral, to get near the window, to watch you, and at
     the signal to throw in this object, then to raise the cry of fire,
     and to wait you at the corner of the street."

     "Precisely."

     "Then you may entirely rely on me."

     "That is excellent. I think, perhaps, it is almost time that I
     prepare for the new role I have to play."

     He disappeared into his bedroom and returned in a few minutes in the
     character of an amiable and simple-minded Nonconformist clergyman.
     His broad black hat, his baggy trousers, his white tie, his
     sympathetic smile, and general look of peering and benevolent
     curiosity were such as Mr. John Hare alone could have equalled. It
     was not merely that Holmes changed his costume. His expression, his
     manner, his very soul seemed to vary with every fresh part that he
     assumed. The stage lost a fine actor, even as science lost an acute
     reasoner, when he became a specialist in crime.

     It was a quarter past six when we left Baker Street, and it still
     wanted ten minutes to the hour when we found ourselves in Serpentine
     Avenue. It was already dusk, and the lamps were just being lighted as
     we paced up and down in front of Briony Lodge, waiting for the coming
     of its occupant. The house was just such as I had pictured it from
     Sherlock Holmes' succinct description, but the locality appeared to
     be less private than I expected. On the contrary, for a small street
     in a quiet neighbourhood, it was remarkably animated. There was a
     group of shabbily dressed men smoking and laughing in a corner, a
     scissors-grinder with his wheel, two guardsmen who were flirting with
     a nurse-girl, and several well-dressed young men who were lounging up
     and down with cigars in their mouths.

     "You see," remarked Holmes, as we paced to and fro in front of the
     house, "this marriage rather simplifies matters. The photograph
     becomes a double-edged weapon now. The chances are that she would be
     as averse to its being seen by Mr. Godfrey Norton, as our client is
     to its coming to the eyes of his princess. Now the question is--Where
     are we to find the photograph?"

     "Where, indeed?"

     "It is most unlikely that she carries it about with her. It is
     cabinet size. Too large for easy concealment about a woman's dress.
     She knows that the King is capable of having her waylaid and
     searched. Two attempts of the sort have already been made. We may
     take it, then, that she does not carry it about with her."

     "Where, then?"

     "Her banker or her lawyer. There is that double possibility. But I am
     inclined to think neither. Women are naturally secretive, and they
     like to do their own secreting. Why should she hand it over to anyone
     else? She could trust her own guardianship, but she could not tell
     what indirect or political influence might be brought to bear upon a
     business man. Besides, remember that she had resolved to use it
     within a few days. It must be where she can lay her hands upon it. It
     must be in her own house."

     "But it has twice been burgled."

     "Pshaw! They did not know how to look."

     "But how will you look?"

     "I will not look."

     "What then?"

     "I will get her to show me."

     "But she will refuse."

     "She will not be able to. But I hear the rumble of wheels. It is her
     carriage. Now carry out my orders to the letter."

     As he spoke the gleam of the side-lights of a carriage came round the
     curve of the avenue. It was a smart little landau which rattled up to
     the door of Briony Lodge. As it pulled up, one of the loafing men at
     the corner dashed forward to open the door in the hope of earning a
     copper, but was elbowed away by another loafer, who had rushed up
     with the same intention. A fierce quarrel broke out, which was
     increased by the two guardsmen, who took sides with one of the
     loungers, and by the scissors-grinder, who was equally hot upon the
     other side. A blow was struck, and in an instant the lady, who had
     stepped from her carriage, was the centre of a little knot of flushed
     and struggling men, who struck savagely at each other with their
     fists and sticks. Holmes dashed into the crowd to protect the lady;
     but just as he reached her he gave a cry and dropped to the ground,
     with the blood running freely down his face. At his fall the
     guardsmen took to their heels in one direction and the loungers in
     the other, while a number of better-dressed people, who had watched
     the scuffle without taking part in it, crowded in to help the lady
     and to attend to the injured man. Irene Adler, as I will still call
     her, had hurried up the steps; but she stood at the top with her
     superb figure outlined against the lights of the hall, looking back
     into the street.

     "Is the poor gentleman much hurt?" she asked.

     "He is dead," cried several voices.

     "No, no, there's life in him!" shouted another. "But he'll be gone
     before you can get him to hospital."

     "He's a brave fellow," said a woman. "They would have had the lady's
     purse and watch if it hadn't been for him. They were a gang, and a
     rough one, too. Ah, he's breathing now."

     "He can't lie in the street. May we bring him in, marm?"

     "Surely. Bring him into the sitting-room. There is a comfortable
     sofa. This way, please!"

     Slowly and solemnly he was borne into Briony Lodge and laid out in
     the principal room, while I still observed the proceedings from my
     post by the window. The lamps had been lit, but the blinds had not
     been drawn, so that I could see Holmes as he lay upon the couch. I do
     not know whether he was seized with compunction at that moment for
     the part he was playing, but I know that I never felt more heartily
     ashamed of myself in my life than when I saw the beautiful creature
     against whom I was conspiring, or the grace and kindliness with which
     she waited upon the injured man. And yet it would be the blackest
     treachery to Holmes to draw back now from the part which he had
     intrusted to me. I hardened my heart, and took the smoke-rocket from
     under my ulster. After all, I thought, we are not injuring her. We
     are but preventing her from injuring another.

     Holmes had sat up upon the couch, and I saw him motion like a man who
     is in need of air. A maid rushed across and threw open the window. At
     the same instant I saw him raise his hand and at the signal I tossed
     my rocket into the room with a cry of "Fire!" The word was no sooner
     out of my mouth than the whole crowd of spectators, well dressed and
     ill--gentlemen, ostlers, and servant-maids--joined in a general
     shriek of "Fire!" Thick clouds of smoke curled through the room and
     out at the open window. I caught a glimpse of rushing figures, and a
     moment later the voice of Holmes from within assuring them that it
     was a false alarm. Slipping through the shouting crowd I made my way
     to the corner of the street, and in ten minutes was rejoiced to find
     my friend's arm in mine, and to get away from the scene of uproar. He
     walked swiftly and in silence for some few minutes until we had
     turned down one of the quiet streets which lead towards the Edgeware
     Road.

     "You did it very nicely, Doctor," he remarked. "Nothing could have
     been better. It is all right."

     "You have the photograph?"

     "I know where it is."

     "And how did you find out?"

     "She showed me, as I told you she would."

     "I am still in the dark."

     "I do not wish to make a mystery," said he, laughing. "The matter was
     perfectly simple. You, of course, saw that everyone in the street was
     an accomplice. They were all engaged for the evening."

     "I guessed as much."

     "Then, when the row broke out, I had a little moist red paint in the
     palm of my hand. I rushed forward, fell down, clapped my hand to my
     face, and became a piteous spectacle. It is an old trick."

     "That also I could fathom."

     "Then they carried me in. She was bound to have me in. What else
     could she do? And into her sitting-room, which was the very room
     which I suspected. It lay between that and her bedroom, and I was
     determined to see which. They laid me on a couch, I motioned for air,
     they were compelled to open the window, and you had your chance."

     "How did that help you?"

     "It was all-important. When a woman thinks that her house is on fire,
     her instinct is at once to rush to the thing which she values most.
     It is a perfectly overpowering impulse, and I have more than once
     taken advantage of it. In the case of the Darlington substitution
     scandal it was of use to me, and also in the Arnsworth Castle
     business. A married woman grabs at her baby; an unmarried one reaches
     for her jewel-box. Now it was clear to me that our lady of to-day had
     nothing in the house more precious to her than what we are in quest
     of. She would rush to secure it. The alarm of fire was admirably
     done. The smoke and shouting were enough to shake nerves of steel.
     She responded beautifully. The photograph is in a recess behind a
     sliding panel just above the right bell-pull. She was there in an
     instant, and I caught a glimpse of it as she half-drew it out. When I
     cried out that it was a false alarm, she replaced it, glanced at the
     rocket, rushed from the room, and I have not seen her since. I rose,
     and, making my excuses, escaped from the house. I hesitated whether
     to attempt to secure the photograph at once; but the coachman had
     come in, and as he was watching me narrowly it seemed safer to wait.
     A little over-precipitance may ruin all."

     "And now?" I asked.

     "Our quest is practically finished. I shall call with the King
     to-morrow, and with you, if you care to come with us. We will be
     shown into the sitting-room to wait for the lady, but it is probable
     that when she comes she may find neither us nor the photograph. It
     might be a satisfaction to his Majesty to regain it with his own
     hands."

     "And when will you call?"

     "At eight in the morning. She will not be up, so that we shall have a
     clear field. Besides, we must be prompt, for this marriage may mean a
     complete change in her life and habits. I must wire to the King
     without delay."

     We had reached Baker Street and had stopped at the door. He was
     searching his pockets for the key when someone passing said:

     "Good-night, Mister Sherlock Holmes."

     There were several people on the pavement at the time, but the
     greeting appeared to come from a slim youth in an ulster who had
     hurried by.

     "I've heard that voice before," said Holmes, staring down the dimly
     lit street. "Now, I wonder who the deuce that could have been."





          CHAPTER III



     I slept at Baker Street that night, and we were engaged upon our
     toast and coffee in the morning when the King of Bohemia rushed into
     the room.

     "You have really got it!" he cried, grasping Sherlock Holmes by
     either shoulder and looking eagerly into his face.

     "Not yet."

     "But you have hopes?"

     "I have hopes."

     "Then, come. I am all impatience to be gone."

     "We must have a cab."

     "No, my brougham is waiting."

     "Then that will simplify matters." We descended and started off once
     more for Briony Lodge.

     "Irene Adler is married," remarked Holmes.

     "Married! When?"

     "Yesterday."

     "But to whom?"

     "To an English lawyer named Norton."

     "But she could not love him."

     "I am in hopes that she does."

     "And why in hopes?"

     "Because it would spare your Majesty all fear of future annoyance. If
     the lady loves her husband, she does not love your Majesty. If she
     does not love your Majesty, there is no reason why she should
     interfere with your Majesty's plan."

     "It is true. And yet--Well! I wish she had been of my own station!
     What a queen she would have made!" He relapsed into a moody silence,
     which was not broken until we drew up in Serpentine Avenue.

     The door of Briony Lodge was open, and an elderly woman stood upon
     the steps. She watched us with a sardonic eye as we stepped from the
     brougham.

     "Mr. Sherlock Holmes, I believe?" said she.

     "I am Mr. Holmes," answered my companion, looking at her with a
     questioning and rather startled gaze.

     "Indeed! My mistress told me that you were likely to call. She left
     this morning with her husband by the 5.15 train from Charing Cross
     for the Continent."

     "What!" Sherlock Holmes staggered back, white with chagrin and
     surprise. "Do you mean that she has left England?"

     "Never to return."

     "And the papers?" asked the King hoarsely. "All is lost."

     "We shall see." He pushed past the servant and rushed into the
     drawing-room, followed by the King and myself. The furniture was
     scattered about in every direction, with dismantled shelves and open
     drawers, as if the lady had hurriedly ransacked them before her
     flight. Holmes rushed at the bell-pull, tore back a small sliding
     shutter, and, plunging in his hand, pulled out a photograph and a
     letter. The photograph was of Irene Adler herself in evening dress,
     the letter was superscribed to "Sherlock Holmes, Esq. To be left till
     called for." My friend tore it open and we all three read it
     together. It was dated at midnight of the preceding night and ran in
     this way:

     "My dear Mr. Sherlock Holmes:
     "You really did it very well. You took me in completely. Until after
     the alarm of fire, I had not a suspicion. But then, when I found how
     I had betrayed myself, I began to think. I had been warned against
     you months ago. I had been told that if the King employed an agent it
     would certainly be you. And your address had been given me. Yet, with
     all this, you made me reveal what you wanted to know. Even after I
     became suspicious, I found it hard to think evil of such a dear, kind
     old clergyman. But, you know, I have been trained as an actress
     myself. Male costume is nothing new to me. I often take advantage of
     the freedom which it gives. I sent John, the coachman, to watch you,
     ran up stairs, got into my walking-clothes, as I call them, and came
     down just as you departed.
     "Well, I followed you to your door, and so made sure that I was
     really an object of interest to the celebrated Mr. Sherlock Holmes.
     Then I, rather imprudently, wished you good-night, and started for
     the Temple to see my husband.
     "We both thought the best resource was flight, when pursued by so
     formidable an antagonist; so you will find the nest empty when you
     call to-morrow. As to the photograph, your client may rest in peace.
     I love and am loved by a better man than he. The King may do what he
     will without hindrance from one whom he has cruelly wronged. I keep
     it only to safeguard myself, and to preserve a weapon which will
     always secure me from any steps which he might take in the future. I
     leave a photograph which he might care to possess; and I remain, dear
     Mr. Sherlock Holmes,
     "Very truly yours,
     "Irene Norton, nÃ©e Adler."

     "What a woman--oh, what a woman!" cried the King of Bohemia, when we
     had all three read this epistle. "Did I not tell you how quick and
     resolute she was? Would she not have made an admirable queen? Is it
     not a pity that she was not on my level?"

     "From what I have seen of the lady she seems indeed to be on a very
     different level to your Majesty," said Holmes coldly. "I am sorry
     that I have not been able to bring your Majesty's business to a more
     successful conclusion."

     "On the contrary, my dear sir," cried the King; "nothing could be
     more successful. I know that her word is inviolate. The photograph is
     now as safe as if it were in the fire."

     "I am glad to hear your Majesty say so."

     "I am immensely indebted to you. Pray tell me in what way I can
     reward you. This ring--" He slipped an emerald snake ring from his
     finger and held it out upon the palm of his hand.

     "Your Majesty has something which I should value even more highly,"
     said Holmes.

     "You have but to name it."

     "This photograph!"

     The King stared at him in amazement.

     "Irene's photograph!" he cried. "Certainly, if you wish it."

     "I thank your Majesty. Then there is no more to be done in the
     matter. I have the honour to wish you a very good-morning." He bowed,
     and, turning away without observing the hand which the King had
     stretched out to him, he set off in my company for his chambers.

     And that was how a great scandal threatened to affect the kingdom of
     Bohemia, and how the best plans of Mr. Sherlock Holmes were beaten by
     a woman's wit. He used to make merry over the cleverness of women,
     but I have not heard him do it of late. And when he speaks of Irene
     Adler, or when he refers to her photograph, it is always under the
     honourable title of the woman.
"""

THEIRS2 = """



                               A CASE OF IDENTITY

                               Arthur Conan Doyle



     "My dear fellow," said Sherlock Holmes as we sat on either side of
     the fire in his lodgings at Baker Street, "life is infinitely
     stranger than anything which the mind of man could invent. We would
     not dare to conceive the things which are really mere commonplaces of
     existence. If we could fly out of that window hand in hand, hover
     over this great city, gently remove the roofs, and peep in at the
     queer things which are going on, the strange coincidences, the
     plannings, the cross-purposes, the wonderful chains of events,
     working through generations, and leading to the most outrÃ© results,
     it would make all fiction with its conventionalities and foreseen
     conclusions most stale and unprofitable."

     "And yet I am not convinced of it," I answered. "The cases which come
     to light in the papers are, as a rule, bald enough, and vulgar
     enough. We have in our police reports realism pushed to its extreme
     limits, and yet the result is, it must be confessed, neither
     fascinating nor artistic."

     "A certain selection and discretion must be used in producing a
     realistic effect," remarked Holmes. "This is wanting in the police
     report, where more stress is laid, perhaps, upon the platitudes of
     the magistrate than upon the details, which to an observer contain
     the vital essence of the whole matter. Depend upon it, there is
     nothing so unnatural as the commonplace."

     I smiled and shook my head. "I can quite understand your thinking
     so." I said. "Of course, in your position of unofficial adviser and
     helper to everybody who is absolutely puzzled, throughout three
     continents, you are brought in contact with all that is strange and
     bizarre. But here"--I picked up the morning paper from the
     ground--"let us put it to a practical test. Here is the first heading
     upon which I come. 'A husband's cruelty to his wife.' There is half a
     column of print, but I know without reading it that it is all
     perfectly familiar to me. There is, of course, the other woman, the
     drink, the push, the blow, the bruise, the sympathetic sister or
     landlady. The crudest of writers could invent nothing more crude."

     "Indeed, your example is an unfortunate one for your argument," said
     Holmes, taking the paper and glancing his eye down it. "This is the
     Dundas separation case, and, as it happens, I was engaged in clearing
     up some small points in connection with it. The husband was a
     teetotaler, there was no other woman, and the conduct complained of
     was that he had drifted into the habit of winding up every meal by
     taking out his false teeth and hurling them at his wife, which, you
     will allow, is not an action likely to occur to the imagination of
     the average story-teller. Take a pinch of snuff, Doctor, and
     acknowledge that I have scored over you in your example."

     He held out his snuffbox of old gold, with a great amethyst in the
     centre of the lid. Its splendour was in such contrast to his homely
     ways and simple life that I could not help commenting upon it.

     "Ah," said he, "I forgot that I had not seen you for some weeks. It
     is a little souvenir from the King of Bohemia in return for my
     assistance in the case of the Irene Adler papers."

     "And the ring?" I asked, glancing at a remarkable brilliant which
     sparkled upon his finger.

     "It was from the reigning family of Holland, though the matter in
     which I served them was of such delicacy that I cannot confide it
     even to you, who have been good enough to chronicle one or two of my
     little problems."

     "And have you any on hand just now?" I asked with interest.

     "Some ten or twelve, but none which present any feature of interest.
     They are important, you understand, without being interesting.
     Indeed, I have found that it is usually in unimportant matters that
     there is a field for the observation, and for the quick analysis of
     cause and effect which gives the charm to an investigation. The
     larger crimes are apt to be the simpler, for the bigger the crime the
     more obvious, as a rule, is the motive. In these cases, save for one
     rather intricate matter which has been referred to me from
     Marseilles, there is nothing which presents any features of interest.
     It is possible, however, that I may have something better before very
     many minutes are over, for this is one of my clients, or I am much
     mistaken."

     He had risen from his chair and was standing between the parted
     blinds gazing down into the dull neutral-tinted London street.
     Looking over his shoulder, I saw that on the pavement opposite there
     stood a large woman with a heavy fur boa round her neck, and a large
     curling red feather in a broad-brimmed hat which was tilted in a
     coquettish Duchess of Devonshire fashion over her ear. From under
     this great panoply she peeped up in a nervous, hesitating fashion at
     our windows, while her body oscillated backward and forward, and her
     fingers fidgeted with her glove buttons. Suddenly, with a plunge, as
     of the swimmer who leaves the bank, she hurried across the road, and
     we heard the sharp clang of the bell.

     "I have seen those symptoms before," said Holmes, throwing his
     cigarette into the fire. "Oscillation upon the pavement always means
     an affaire de coeur. She would like advice, but is not sure that the
     matter is not too delicate for communication. And yet even here we
     may discriminate. When a woman has been seriously wronged by a man
     she no longer oscillates, and the usual symptom is a broken bell
     wire. Here we may take it that there is a love matter, but that the
     maiden is not so much angry as perplexed, or grieved. But here she
     comes in person to resolve our doubts."

     As he spoke there was a tap at the door, and the boy in buttons
     entered to announce Miss Mary Sutherland, while the lady herself
     loomed behind his small black figure like a full-sailed merchant-man
     behind a tiny pilot boat. Sherlock Holmes welcomed her with the easy
     courtesy for which he was remarkable, and, having closed the door and
     bowed her into an armchair, he looked her over in the minute and yet
     abstracted fashion which was peculiar to him.

     "Do you not find," he said, "that with your short sight it is a
     little trying to do so much typewriting?"

     "I did at first," she answered, "but now I know where the letters are
     without looking." Then, suddenly realising the full purport of his
     words, she gave a violent start and looked up, with fear and
     astonishment upon her broad, good-humoured face. "You've heard about
     me, Mr. Holmes," she cried, "else how could you know all that?"

     "Never mind," said Holmes, laughing; "it is my business to know
     things. Perhaps I have trained myself to see what others overlook. If
     not, why should you come to consult me?"

     "I came to you, sir, because I heard of you from Mrs. Etherege, whose
     husband you found so easy when the police and everyone had given him
     up for dead. Oh, Mr. Holmes, I wish you would do as much for me. I'm
     not rich, but still I have a hundred a year in my own right, besides
     the little that I make by the machine, and I would give it all to
     know what has become of Mr. Hosmer Angel."

     "Why did you come away to consult me in such a hurry?" asked Sherlock
     Holmes, with his finger-tips together and his eyes to the ceiling.

     Again a startled look came over the somewhat vacuous face of Miss
     Mary Sutherland. "Yes, I did bang out of the house," she said, "for
     it made me angry to see the easy way in which Mr. Windibank--that is,
     my father--took it all. He would not go to the police, and he would
     not go to you, and so at last, as he would do nothing and kept on
     saying that there was no harm done, it made me mad, and I just on
     with my things and came right away to you."

     "Your father," said Holmes, "your stepfather, surely, since the name
     is different."

     "Yes, my stepfather. I call him father, though it sounds funny, too,
     for he is only five years and two months older than myself."

     "And your mother is alive?"

     "Oh, yes, mother is alive and well. I wasn't best pleased, Mr.
     Holmes, when she married again so soon after father's death, and a
     man who was nearly fifteen years younger than herself. Father was a
     plumber in the Tottenham Court Road, and he left a tidy business
     behind him, which mother carried on with Mr. Hardy, the foreman; but
     when Mr. Windibank came he made her sell the business, for he was
     very superior, being a traveller in wines. They got Â£4700 for the
     goodwill and interest, which wasn't near as much as father could have
     got if he had been alive."

     I had expected to see Sherlock Holmes impatient under this rambling
     and inconsequential narrative, but, on the contrary, he had listened
     with the greatest concentration of attention.

     "Your own little income," he asked, "does it come out of the
     business?"

     "Oh, no, sir. It is quite separate and was left me by my uncle Ned in
     Auckland. It is in New Zealand stock, paying 4Â½ per cent. Two
     thousand five hundred pounds was the amount, but I can only touch the
     interest."

     "You interest me extremely," said Holmes. "And since you draw so
     large a sum as a hundred a year, with what you earn into the bargain,
     you no doubt travel a little and indulge yourself in every way. I
     believe that a single lady can get on very nicely upon an income of
     about Â£60."

     "I could do with much less than that, Mr. Holmes, but you understand
     that as long as I live at home I don't wish to be a burden to them,
     and so they have the use of the money just while I am staying with
     them. Of course, that is only just for the time. Mr. Windibank draws
     my interest every quarter and pays it over to mother, and I find that
     I can do pretty well with what I earn at typewriting. It brings me
     twopence a sheet, and I can often do from fifteen to twenty sheets in
     a day."

     "You have made your position very clear to me," said Holmes. "This is
     my friend, Dr. Watson, before whom you can speak as freely as before
     myself. Kindly tell us now all about your connection with Mr. Hosmer
     Angel."

     A flush stole over Miss Sutherland's face, and she picked nervously
     at the fringe of her jacket. "I met him first at the gasfitters'
     ball," she said. "They used to send father tickets when he was alive,
     and then afterwards they remembered us, and sent them to mother. Mr.
     Windibank did not wish us to go. He never did wish us to go anywhere.
     He would get quite mad if I wanted so much as to join a Sunday-school
     treat. But this time I was set on going, and I would go; for what
     right had he to prevent? He said the folk were not fit for us to
     know, when all father's friends were to be there. And he said that I
     had nothing fit to wear, when I had my purple plush that I had never
     so much as taken out of the drawer. At last, when nothing else would
     do, he went off to France upon the business of the firm, but we went,
     mother and I, with Mr. Hardy, who used to be our foreman, and it was
     there I met Mr. Hosmer Angel."

     "I suppose," said Holmes, "that when Mr. Windibank came back from
     France he was very annoyed at your having gone to the ball."

     "Oh, well, he was very good about it. He laughed, I remember, and
     shrugged his shoulders, and said there was no use denying anything to
     a woman, for she would have her way."

     "I see. Then at the gasfitters' ball you met, as I understand, a
     gentleman called Mr. Hosmer Angel."

     "Yes, sir. I met him that night, and he called next day to ask if we
     had got home all safe, and after that we met him--that is to say, Mr.
     Holmes, I met him twice for walks, but after that father came back
     again, and Mr. Hosmer Angel could not come to the house any more."

     "No?"

     "Well, you know father didn't like anything of the sort. He wouldn't
     have any visitors if he could help it, and he used to say that a
     woman should be happy in her own family circle. But then, as I used
     to say to mother, a woman wants her own circle to begin with, and I
     had not got mine yet."

     "But how about Mr. Hosmer Angel? Did he make no attempt to see you?"

     "Well, father was going off to France again in a week, and Hosmer
     wrote and said that it would be safer and better not to see each
     other until he had gone. We could write in the meantime, and he used
     to write every day. I took the letters in in the morning, so there
     was no need for father to know."

     "Were you engaged to the gentleman at this time?"

     "Oh, yes, Mr. Holmes. We were engaged after the first walk that we
     took. Hosmer--Mr. Angel--was a cashier in an office in Leadenhall
     Street--and--"

     "What office?"

     "That's the worst of it, Mr. Holmes, I don't know."

     "Where did he live, then?"

     "He slept on the premises."

     "And you don't know his address?"

     "No--except that it was Leadenhall Street."

     "Where did you address your letters, then?"

     "To the Leadenhall Street Post Office, to be left till called for. He
     said that if they were sent to the office he would be chaffed by all
     the other clerks about having letters from a lady, so I offered to
     typewrite them, like he did his, but he wouldn't have that, for he
     said that when I wrote them they seemed to come from me, but when
     they were typewritten he always felt that the machine had come
     between us. That will just show you how fond he was of me, Mr.
     Holmes, and the little things that he would think of."

     "It was most suggestive," said Holmes. "It has long been an axiom of
     mine that the little things are infinitely the most important.  Can
     you remember any other little things about Mr. Hosmer Angel?"

     "He was a very shy man, Mr. Holmes. He would rather walk with me in
     the evening than in the daylight, for he said that he hated to be
     conspicuous. Very retiring and gentlemanly he was. Even his voice was
     gentle. He'd had the quinsy and swollen glands when he was young, he
     told me, and it had left him with a weak throat, and a hesitating,
     whispering fashion of speech. He was always well dressed, very neat
     and plain, but his eyes were weak, just as mine are, and he wore
     tinted glasses against the glare."

     "Well, and what happened when Mr. Windibank, your stepfather,
     returned to France?"

     "Mr. Hosmer Angel came to the house again and proposed that we should
     marry before father came back. He was in dreadful earnest and made me
     swear, with my hands on the Testament, that whatever happened I would
     always be true to him. Mother said he was quite right to make me
     swear, and that it was a sign of his passion. Mother was all in his
     favour from the first and was even fonder of him than I was. Then,
     when they talked of marrying within the week, I began to ask about
     father; but they both said never to mind about father, but just to
     tell him afterwards, and mother said she would make it all right with
     him. I didn't quite like that, Mr. Holmes. It seemed funny that I
     should ask his leave, as he was only a few years older than me; but I
     didn't want to do anything on the sly, so I wrote to father at
     Bordeaux, where the company has its French offices, but the letter
     came back to me on the very morning of the wedding."

     "It missed him, then?"

     "Yes, sir; for he had started to England just before it arrived."

     "Ha! that was unfortunate. Your wedding was arranged, then, for the
     Friday. Was it to be in church?"

     "Yes, sir, but very quietly. It was to be at St. Saviour's, near
     King's Cross, and we were to have breakfast afterwards at the St.
     Pancras Hotel. Hosmer came for us in a hansom, but as there were two
     of us he put us both into it and stepped himself into a four-wheeler,
     which happened to be the only other cab in the street. We got to the
     church first, and when the four-wheeler drove up we waited for him to
     step out, but he never did, and when the cabman got down from the box
     and looked there was no one there! The cabman said that he could not
     imagine what had become of him, for he had seen him get in with his
     own eyes. That was last Friday, Mr. Holmes, and I have never seen or
     heard anything since then to throw any light upon what became of
     him."

     "It seems to me that you have been very shamefully treated," said
     Holmes.

     "Oh, no, sir! He was too good and kind to leave me so. Why, all the
     morning he was saying to me that, whatever happened, I was to be
     true; and that even if something quite unforeseen occurred to
     separate us, I was always to remember that I was pledged to him, and
     that he would claim his pledge sooner or later. It seemed strange
     talk for a wedding-morning, but what has happened since gives a
     meaning to it."

     "Most certainly it does. Your own opinion is, then, that some
     unforeseen catastrophe has occurred to him?"

     "Yes, sir. I believe that he foresaw some danger, or else he would
     not have talked so. And then I think that what he foresaw happened."

     "But you have no notion as to what it could have been?"

     "None."

     "One more question. How did your mother take the matter?"

     "She was angry, and said that I was never to speak of the matter
     again."

     "And your father? Did you tell him?"

     "Yes; and he seemed to think, with me, that something had happened,
     and that I should hear of Hosmer again. As he said, what interest
     could anyone have in bringing me to the doors of the church, and then
     leaving me? Now, if he had borrowed my money, or if he had married me
     and got my money settled on him, there might be some reason, but
     Hosmer was very independent about money and never would look at a
     shilling of mine. And yet, what could have happened? And why could he
     not write? Oh, it drives me half-mad to think of it, and I can't
     sleep a wink at night." She pulled a little handkerchief out of her
     muff and began to sob heavily into it.

     "I shall glance into the case for you," said Holmes, rising, "and I
     have no doubt that we shall reach some definite result. Let the
     weight of the matter rest upon me now, and do not let your mind dwell
     upon it further. Above all, try to let Mr. Hosmer Angel vanish from
     your memory, as he has done from your life."

     "Then you don't think I'll see him again?"

     "I fear not."

     "Then what has happened to him?"

     "You will leave that question in my hands. I should like an accurate
     description of him and any letters of his which you can spare."

     "I advertised for him in last Saturday's Chronicle," said she. "Here
     is the slip and here are four letters from him."

     "Thank you. And your address?"

     "No. 31 Lyon Place, Camberwell."

     "Mr. Angel's address you never had, I understand. Where is your
     father's place of business?"

     "He travels for Westhouse & Marbank, the great claret importers of
     Fenchurch Street."

     "Thank you. You have made your statement very clearly. You will leave
     the papers here, and remember the advice which I have given you. Let
     the whole incident be a sealed book, and do not allow it to affect
     your life."

     "You are very kind, Mr. Holmes, but I cannot do that. I shall be true
     to Hosmer. He shall find me ready when he comes back."

     For all the preposterous hat and the vacuous face, there was
     something noble in the simple faith of our visitor which compelled
     our respect. She laid her little bundle of papers upon the table and
     went her way, with a promise to come again whenever she might be
     summoned.

     Sherlock Holmes sat silent for a few minutes with his fingertips
     still pressed together, his legs stretched out in front of him, and
     his gaze directed upward to the ceiling. Then he took down from the
     rack the old and oily clay #e, which was to him as a counsellor,
     and, having lit it, he leaned back in his chair, with the thick blue
     cloud-wreaths spinning up from him, and a look of infinite languor in
     his face.

     "Quite an interesting study, that maiden," he observed. "I found her
     more interesting than her little problem, which, by the way, is
     rather a trite one. You will find parallel cases, if you consult my
     index, in Andover in '77, and there was something of the sort at The
     Hague last year. Old as is the idea, however, there were one or two
     details which were new to me. But the maiden herself was most
     instructive."

     "You appeared to read a good deal upon her which was quite invisible
     to me," I remarked.

     "Not invisible but unnoticed, Watson. You did not know where to look,
     and so you missed all that was important. I can never bring you to
     realise the importance of sleeves, the suggestiveness of thumb-nails,
     or the great issues that may hang from a boot-lace. Now, what did you
     gather from that woman's appearance? Describe it."

     "Well, she had a slate-coloured, broad-brimmed straw hat, with a
     feather of a brickish red. Her jacket was black, with black beads
     sewn upon it, and a fringe of little black jet ornaments. Her dress
     was brown, rather darker than coffee colour, with a little purple
     plush at the neck and sleeves. Her gloves were greyish and were worn
     through at the right forefinger. Her boots I didn't observe. She had
     small round, hanging gold earrings, and a general air of being fairly
     well-to-do in a vulgar, comfortable, easy-going way."

     Sherlock Holmes clapped his hands softly together and chuckled.

     "'Pon my word, Watson, you are coming along wonderfully. You have
     really done very well indeed. It is true that you have missed
     everything of importance, but you have hit upon the method, and you
     have a quick eye for colour. Never trust to general impressions, my
     boy, but concentrate yourself upon details. My first glance is always
     at a woman's sleeve. In a man it is perhaps better first to take the
     knee of the trouser. As you observe, this woman had plush upon her
     sleeves, which is a most useful material for showing traces. The
     double line a little above the wrist, where the typewritist presses
     against the table, was beautifully defined. The sewing-machine, of
     the hand type, leaves a similar mark, but only on the left arm, and
     on the side of it farthest from the thumb, instead of being right
     across the broadest part, as this was. I then glanced at her face,
     and, observing the dint of a pince-nez at either side of her nose, I
     ventured a remark upon short sight and typewriting, which seemed to
     surprise her."

     "It surprised me."

     "But, surely, it was obvious. I was then much surprised and
     interested on glancing down to observe that, though the boots which
     she was wearing were not unlike each other, they were really odd
     ones; the one having a slightly decorated toe-cap, and the other a
     plain one. One was buttoned only in the two lower buttons out of
     five, and the other at the first, third, and fifth. Now, when you see
     that a young lady, otherwise neatly dressed, has come away from home
     with odd boots, half-buttoned, it is no great deduction to say that
     she came away in a hurry."

     "And what else?" I asked, keenly interested, as I always was, by my
     friend's incisive reasoning.

     "I noted, in passing, that she had written a note before leaving home
     but after being fully dressed. You observed that her right glove was
     torn at the forefinger, but you did not apparently see that both
     glove and finger were stained with violet ink. She had written in a
     hurry and dipped her pen too deep. It must have been this morning, or
     the mark would not remain clear upon the finger. All this is amusing,
     though rather elementary, but I must go back to business, Watson.
     Would you mind reading me the advertised description of Mr. Hosmer
     Angel?"

     I held the little printed slip to the light.

     "Missing," it said, "on the morning of the fourteenth, a gentleman
     named Hosmer Angel. About five ft. seven in. in height; strongly
     built, sallow complexion, black hair, a little bald in the centre,
     bushy, black side-whiskers and moustache; tinted glasses, slight
     infirmity of speech. Was dressed, when last seen, in black frock-coat
     faced with silk, black waistcoat, gold Albert chain, and grey Harris
     tweed trousers, with brown gaiters over elastic-sided boots. Known to
     have been employed in an office in Leadenhall Street. Anybody
     bringing--"

     "That will do," said Holmes. "As to the letters," he continued,
     glancing over them, "they are very commonplace. Absolutely no clue in
     them to Mr. Angel, save that he quotes Balzac once. There is one
     remarkable point, however, which will no doubt strike you."

     "They are typewritten," I remarked.

     "Not only that, but the signature is typewritten. Look at the neat
     little 'Hosmer Angel' at the bottom. There is a date, you see, but no
     superscription except Leadenhall Street, which is rather vague. The
     point about the signature is very suggestive--in fact, we may call it
     conclusive."

     "Of what?"

     "My dear fellow, is it possible you do not see how strongly it bears
     upon the case?"

     "I cannot say that I do unless it were that he wished to be able to
     deny his signature if an action for breach of promise were
     instituted."

     "No, that was not the point. However, I shall write two letters,
     which should settle the matter. One is to a firm in the City, the
     other is to the young lady's stepfather, Mr. Windibank, asking him
     whether he could meet us here at six o'clock tomorrow evening. It is
     just as well that we should do business with the male relatives. And
     now, Doctor, we can do nothing until the answers to those letters
     come, so we may put our little problem upon the shelf for the
     interim."

     I had had so many reasons to believe in my friend's subtle powers of
     reasoning and extraordinary energy in action that I felt that he must
     have some solid grounds for the assured and easy demeanour with which
     he treated the singular mystery which he had been called upon to
     fathom. Once only had I known him to fail, in the case of the King of
     Bohemia and of the Irene Adler photograph; but when I looked back to
     the weird business of the Sign of Four, and the extraordinary
     circumstances connected with the Study in Scarlet, I felt that it
     would be a strange tangle indeed which he could not unravel.

     I left him then, still puffing at his black clay #e, with the
     conviction that when I came again on the next evening I would find
     that he held in his hands all the clues which would lead up to the
     identity of the disappearing bridegroom of Miss Mary Sutherland.

     A professional case of great gravity was engaging my own attention at
     the time, and the whole of next day I was busy at the bedside of the
     sufferer. It was not until close upon six o'clock that I found myself
     free and was able to spring into a hansom and drive to Baker Street,
     half afraid that I might be too late to assist at the dÃ©nouement of
     the little mystery. I found Sherlock Holmes alone, however, half
     asleep, with his long, thin form curled up in the recesses of his
     armchair. A formidable array of bottles and test-tubes, with the
     pungent cleanly smell of hydrochloric acid, told me that he had spent
     his day in the chemical work which was so dear to him.

     "Well, have you solved it?" I asked as I entered.

     "Yes. It was the bisulphate of baryta."

     "No, no, the mystery!" I cried.

     "Oh, that! I thought of the salt that I have been working upon. There
     was never any mystery in the matter, though, as I said yesterday,
     some of the details are of interest. The only drawback is that there
     is no law, I fear, that can touch the scoundrel."

     "Who was he, then, and what was his object in deserting Miss
     Sutherland?"

     The question was hardly out of my mouth, and Holmes had not yet
     opened his lips to reply, when we heard a heavy footfall in the
     passage and a tap at the door.

     "This is the girl's stepfather, Mr. James Windibank," said Holmes.
     "He has written to me to say that he would be here at six. Come in!"

     The man who entered was a sturdy, middle-sized fellow, some thirty
     years of age, clean-shaven, and sallow-skinned, with a bland,
     insinuating manner, and a pair of wonderfully sharp and penetrating
     grey eyes. He shot a questioning glance at each of us, placed his
     shiny top-hat upon the sideboard, and with a slight bow sidled down
     into the nearest chair.

     "Good-evening, Mr. James Windibank," said Holmes. "I think that this
     typewritten letter is from you, in which you made an appointment with
     me for six o'clock?"

     "Yes, sir. I am afraid that I am a little late, but I am not quite my
     own master, you know. I am sorry that Miss Sutherland has troubled
     you about this little matter, for I think it is far better not to
     wash linen of the sort in public. It was quite against my wishes that
     she came, but she is a very excitable, impulsive girl, as you may
     have noticed, and she is not easily controlled when she has made up
     her mind on a point. Of course, I did not mind you so much, as you
     are not connected with the official police, but it is not pleasant to
     have a family misfortune like this noised abroad. Besides, it is a
     useless expense, for how could you possibly find this Hosmer Angel?"

     "On the contrary," said Holmes quietly; "I have every reason to
     believe that I will succeed in discovering Mr. Hosmer Angel."

     Mr. Windibank gave a violent start and dropped his gloves. "I am
     delighted to hear it," he said.

     "It is a curious thing," remarked Holmes, "that a typewriter has
     really quite as much individuality as a man's handwriting. Unless
     they are quite new, no two of them write exactly alike. Some letters
     get more worn than others, and some wear only on one side. Now, you
     remark in this note of yours, Mr. Windibank, that in every case there
     is some little slurring over of the 'e,' and a slight defect in the
     tail of the 'r.' There are fourteen other characteristics, but those
     are the more obvious."

     "We do all our correspondence with this machine at the office, and no
     doubt it is a little worn," our visitor answered, glancing keenly at
     Holmes with his bright little eyes.

     "And now I will show you what is really a very interesting study, Mr.
     Windibank," Holmes continued. "I think of writing another little
     monograph some of these days on the typewriter and its relation to
     crime. It is a subject to which I have devoted some little attention.
     I have here four letters which purport to come from the missing man.
     They are all typewritten. In each case, not only are the 'e's'
     slurred and the 'r's' tailless, but you will observe, if you care to
     use my magnifying lens, that the fourteen other characteristics to
     which I have alluded are there as well."

     Mr. Windibank sprang out of his chair and picked up his hat. "I
     cannot waste time over this sort of fantastic talk, Mr. Holmes," he
     said. "If you can catch the man, catch him, and let me know when you
     have done it."

     "Certainly," said Holmes, stepping over and turning the key in the
     door. "I let you know, then, that I have caught him!"

     "What! where?" shouted Mr. Windibank, turning white to his lips and
     glancing about him like a rat in a trap.

     "Oh, it won't do--really it won't," said Holmes suavely. "There is no
     possible getting out of it, Mr. Windibank. It is quite too
     transparent, and it was a very bad compliment when you said that it
     was impossible for me to solve so simple a question. That's right!
     Sit down and let us talk it over."

     Our visitor collapsed into a chair, with a ghastly face and a glitter
     of moisture on his brow. "It--it's not actionable," he stammered.

     "I am very much afraid that it is not. But between ourselves,
     Windibank, it was as cruel and selfish and heartless a trick in a
     petty way as ever came before me. Now, let me just run over the
     course of events, and you will contradict me if I go wrong."

     The man sat huddled up in his chair, with his head sunk upon his
     breast, like one who is utterly crushed. Holmes stuck his feet up on
     the corner of the mantelpiece and, leaning back with his hands in his
     pockets, began talking, rather to himself, as it seemed, than to us.

     "The man married a woman very much older than himself for her money,"
     said he, "and he enjoyed the use of the money of the daughter as long
     as she lived with them. It was a considerable sum, for people in
     their position, and the loss of it would have made a serious
     difference. It was worth an effort to preserve it. The daughter was
     of a good, amiable disposition, but affectionate and warm-hearted in
     her ways, so that it was evident that with her fair personal
     advantages, and her little income, she would not be allowed to remain
     single long. Now her marriage would mean, of course, the loss of a
     hundred a year, so what does her stepfather do to prevent it? He
     takes the obvious course of keeping her at home and forbidding her to
     seek the company of people of her own age. But soon he found that
     that would not answer forever. She became restive, insisted upon her
     rights, and finally announced her positive intention of going to a
     certain ball. What does her clever stepfather do then? He conceives
     an idea more creditable to his head than to his heart. With the
     connivance and assistance of his wife he disguised himself, covered
     those keen eyes with tinted glasses, masked the face with a moustache
     and a pair of bushy whiskers, sunk that clear voice into an
     insinuating whisper, and doubly secure on account of the girl's short
     sight, he appears as Mr. Hosmer Angel, and keeps off other lovers by
     making love himself."

     "It was only a joke at first," groaned our visitor. "We never thought
     that she would have been so carried away."

     "Very likely not. However that may be, the young lady was very
     decidedly carried away, and, having quite made up her mind that her
     stepfather was in France, the suspicion of treachery never for an
     instant entered her mind. She was flattered by the gentleman's
     attentions, and the effect was increased by the loudly expressed
     admiration of her mother. Then Mr. Angel began to call, for it was
     obvious that the matter should be pushed as far as it would go if a
     real effect were to be produced. There were meetings, and an
     engagement, which would finally secure the girl's affections from
     turning towards anyone else. But the deception could not be kept up
     forever. These pretended journeys to France were rather cumbrous. The
     thing to do was clearly to bring the business to an end in such a
     dramatic manner that it would leave a permanent impression upon the
     young lady's mind and prevent her from looking upon any other suitor
     for some time to come. Hence those vows of fidelity exacted upon a
     Testament, and hence also the allusions to a possibility of something
     happening on the very morning of the wedding. James Windibank wished
     Miss Sutherland to be so bound to Hosmer Angel, and so uncertain as
     to his fate, that for ten years to come, at any rate, she would not
     listen to another man. As far as the church door he brought her, and
     then, as he could go no farther, he conveniently vanished away by the
     old trick of stepping in at one door of a four-wheeler and out at the
     other. I think that was the chain of events, Mr. Windibank!"

     Our visitor had recovered something of his assurance while Holmes had
     been talking, and he rose from his chair now with a cold sneer upon
     his pale face.

     "It may be so, or it may not, Mr. Holmes," said he, "but if you are
     so very sharp you ought to be sharp enough to know that it is you who
     are breaking the law now, and not me. I have done nothing actionable
     from the first, but as long as you keep that door locked you lay
     yourself open to an action for assault and illegal constraint."

     "The law cannot, as you say, touch you," said Holmes, unlocking and
     throwing open the door, "yet there never was a man who deserved
     punishment more. If the young lady has a brother or a friend, he
     ought to lay a whip across your shoulders. By Jove!" he continued,
     flushing up at the sight of the bitter sneer upon the man's face, "it
     is not part of my duties to my client, but here's a hunting crop
     handy, and I think I shall just treat myself to--" He took two swift
     steps to the whip, but before he could grasp it there was a wild
     clatter of steps upon the stairs, the heavy hall door banged, and
     from the window we could see Mr. James Windibank running at the top
     of his speed down the road.

     "There's a cold-blooded scoundrel!" said Holmes, laughing, as he
     threw himself down into his chair once more. "That fellow will rise
     from crime to crime until he does something very bad, and ends on a
     gallows. The case has, in some respects, been not entirely devoid of
     interest."

     "I cannot now entirely see all the steps of your reasoning," I
     remarked.

     "Well, of course it was obvious from the first that this Mr. Hosmer
     Angel must have some strong object for his curious conduct, and it
     was equally clear that the only man who really profited by the
     incident, as far as we could see, was the stepfather. Then the fact
     that the two men were never together, but that the one always
     appeared when the other was away, was suggestive. So were the tinted
     spectacles and the curious voice, which both hinted at a disguise, as
     did the bushy whiskers. My suspicions were all confirmed by his
     peculiar action in typewriting his signature, which, of course,
     inferred that his handwriting was so familiar to her that she would
     recognise even the smallest sample of it. You see all these isolated
     facts, together with many minor ones, all pointed in the same
     direction."

     "And how did you verify them?"

     "Having once spotted my man, it was easy to get corroboration. I knew
     the firm for which this man worked. Having taken the printed
     description, I eliminated everything from it which could be the
     result of a disguise--the whiskers, the glasses, the voice, and I
     sent it to the firm, with a request that they would inform me whether
     it answered to the description of any of their travellers. I had
     already noticed the peculiarities of the typewriter, and I wrote to
     the man himself at his business address asking him if he would come
     here. As I expected, his reply was typewritten and revealed the same
     trivial but characteristic defects. The same post brought me a letter
     from Westhouse & Marbank, of Fenchurch Street, to say that the
     description tallied in every respect with that of their employee,
     James Windibank. VoilÃ  tout!"

     "And Miss Sutherland?"

     "If I tell her she will not believe me. You may remember the old
     Persian saying, 'There is danger for him who taketh the tiger cub,
     and danger also for whoso snatches a delusion from a woman.' There is
     as much sense in Hafiz as in Horace, and as much knowledge of the
     world."
"""


len(THEIRS2)


#
# Here, run your punctuation-comparisons (absolute counts)
#
print(pun_all(YOURS1))
print(pun_all(YOURS2))
print(pun_all(THEIRS1))
print(pun_all(THEIRS2))


#
# Here, run your punctuation-comparisons (relative, per-character counts)
#
print(pun_all(YOURS1)/len(YOURS1))
print(pun_all(YOURS2)/len(YOURS2))
print(pun_all(THEIRS1)/len(THEIRS1))
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




# Repeat the above for a radius of 6, 7, 8, 9, and 10
# It's fast to do:  Copy and paste and edit!!
L = [ rwalk(6) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==6 (for 1000 trials) was", average)



L = [ rwalk(7) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==7 (for 1000 trials) was", average)



L = [ rwalk(8) for i in range(1000) ]

# then, find the average of those 1000 experiments for rwalk(5)
average = sum(L)/len(L)
print("The average for radius==8 (for 1000 trials) was", average)



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

print(d['number'])
print(d['people'][8]['name'])


# use this cell - based on the example above - to share your solutions to the Astronaut challenges...
print(d['people'][5]['name'])
print(d['people'][1]['name'][3] + d['people'][1]['name'][2])


#
# use this cell for your API call - and data-extraction
#
import requests

url = "https://jsonplaceholder.typicode.com/users"
result = requests.get(url)

d = result.json()
print(d)
print()
last_user = d[-1]
print(f"Name: {last_user['name']}")
print(f"Address: {last_user['address']}")
print(f"Email: {last_user['email']}")


#
# use this cell for your webscraping call - optional data-extraction
#
import requests

url = "http://books.toscrape.com/"
result = requests.get(url)

print(f"result is {result}")

print(result.text)

webpage = result.text

start = webpage.find("<h1>") + len("<h1>")  # Move start position to after "<h1>"
end = webpage.find("</h1>", start)  # Find closing </h1> tag
heading = webpage[start:end]  # Extract text between <h1> and </h1>

print(f"Extracted Heading: {heading}")



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





