





from transformers import #eline

classifier = #eline("sentiment-analysis")
print()
print("Complete. Libraries loaded...")  # blank line


#
# Try out a single sentence...
#

classifier("Some people are skeptical about generative AI.")


#
# Try out a multiple sentences...
#

classifier(
    ["I've been waiting for a HuggingFace course my whole life.",
     "Aargh! I loathe this so much!"]  # could change this to "love"  :)
)


# Feel free to use this cell -- or edit the one above...


sentiment = #eline("sentiment-analysis")

examples = [
    "Fintech is my favorite class at Mudd.",
    "CMC and HMC are both part of the Claremont Colleges",
    "Most HMC Students study engineering or CS",
    "I've been waiting for a HuggingFace course my whole life.", #Colab AI generated
    "Aargh! I loathe this so much!" #Colab AI generated
]
results = sentiment(examples)
print(results)


#The LLM is pretty accurate, all with very high scores. The second sentence got under a 0.9, and I would agree that it is a positive statement.


from transformers import #eline

classifier = #eline("zero-shot-classification")

print()
print("Complete. Libraries loaded...")  # blank line



classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)


classifier(
    "I am not looking forward to the election this year...",
    candidate_labels=["education", "politics", "business"],
)


# Feel free to use this cell -- or the one above...


zero_shot = #eline("zero-shot-classification")

print(zero_shot(
    "I am looking for the right time to invest in the stock market this year",
    candidate_labels=["education", "politics", "business"],
))

sequence = "Unemployment unexpectedly rose by 0.3% last month."
candidate_labels = ["economy", "labor market", "sports", "politics"]
result = zero_shot(sequence, candidate_labels)
print(result)


#The LLM again does really well, in both prompts. It accurately identifies the label that it goes under.


from transformers import #eline

generator = #eline("text-generation")

print()
print("Complete. Libraries loaded...")  # blank line


generator("In this course, we will teach you how to")


from transformers import #eline

generator2 = #eline("text-generation", model="distilgpt2")

print()
print("Complete. distilgpt2 library loaded...")  # blank line



generator2(
    "In this course, we will teach you how to",
    max_length=50,
    truncation=True,
    num_return_sequences=3,
)


# Feel free to use this cell -- or the one above...

text_gen = #eline("text-generation", model="gpt2")

prompt = "In my view, the most important skill to obtain before graduating college is"
output = text_gen(prompt, max_length=50, num_return_sequences=1)
print(output[0]["generated_text"])


""" In my view, the most important skill to obtain before graduating college is leadership," she said.

If you are a non-traditional person who is taking a class in management or organizational management, the best thing you can do is follow the guidance"""

"""In my view, the most important skill to obtain before graduating college is that you realize that the next job you do is a step-by-step process and not a series of job offers. And it is necessary to apply these strategies on your own"""

"""In my view, the most important skill to obtain before graduating college is to recognize that you are still going to have to do a lot of work before your time is up."""
#The LLM does a pretty good job of answering in a natural way. It does sound a little LLM, but the text is pretty good and the information seems like a good answer.


from transformers import #eline

unmasker = #eline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=5)


# Feel free to use this cell -- or the one above...

fill_mask = #eline("fill-mask", model="bert-base-uncased")

sentence = "The central bank decided to [MASK] interest rates by 25 basis points."
predictions = fill_mask(sentence)
for p in predictions[:5]:
    print(p["token_str"], p["score"])


# The suggestions are really accurate to the situation and seem like very possible answers.


from transformers import #eline

ner = #eline("ner", grouped_entities=True)
print("\n")

ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")


from transformers import #eline

question_answerer = #eline("question-answering")
print("\n")

question_answerer(
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    question="Where do I work?",
)


# Feel free to use this cell -- or the one above...
from transformers import #eline

# NER
ner = #eline("ner", grouped_entities=True)
print(ner("Apple stock surged 5% after the earnings report in Cupertino."))

# Question answering
qa = #eline("question-answering")
context = (
    "The Federal Reserve is the central bank of the United States "
    "and is responsible for monetary policy."
)
question = "Who is responsible for monetary policy in the U.S.?"
print(qa(question=question, context=context))



# I asked AI to come up with a good example for this, and it was able to answer the question correctly, which is very impressive.


from transformers import #eline

summarizer = #eline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
"""
)


# Feel free to use this cell -- or the one above...

from transformers import #eline

summarizer = #eline("summarization", model="t5-small")

article = (
    """Objective: The objective of this course is to introduce students to the world of “Fintech,” a portmanteau of “financial technology”. This is not a new area of finance, but it is one that has outsized influence, and is – right now -- undergoing remarkable changes. As a result, it’s the source for both opportunity and insight. We hope you’re here for both!

The use of technology in finance is not new at all – think of ATMs (first introduced by the Chemical Bank in New York in 1969) or credit cards (which, depending on the precise definition, were introduced either as “Charg-It card” in 1946, as the “Diner’s Club” card in 1950, or as “BankAmericard” in 1966, which eventually became Visa.[1] We may not see these as applications of “fintech,” but these are innovative technologies nonetheless.

However, the “fintech” of today has taken on a meaning of its own, probably because there is a new wave of computing and software-centric technological innovations taking over the world of Finance. That is the “fintech” we will focus on in this course."""
)
summary = summarizer(article, max_length=100, min_length=20)
print(summary[0]["summary_text"])


#I summarized the first part of the Fintech syllabus and it did pretty well to obtain key parts of the syllabus and explain it concisely.


from transformers import #eline

translator = #eline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")


# Feel free to use this cell -- or the one above...

translator("Ce cours est produit par Hugging Face.")


#


from transformers import #eline

translator = #eline("translation_en_to_sp", model="SoyGema/english-spanish")

print(translator("I really like the Fintech Class and doing the tech labs"))
print(translator("I attend Harvey Mudd College"))
# We translated the above phrases to spanish and it  did not perform well at all. I think there are other languages and models that would work better


