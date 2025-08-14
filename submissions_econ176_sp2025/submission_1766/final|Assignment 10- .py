





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


test_sentences = [
    "I'm feeling okay about how the project turned out.",
    "This class has been incredibly rewarding and insightful.",
    "I'm not sure if this data makes any sense at all.",
    "The results were fine, though not as great as expected.",
    "Everything about this experience was just average.",
    "It’s hard to tell if this is a win or a loss."
]

results = classifier(test_sentences)

for i, (sentence, result) in enumerate(zip(test_sentences, results)):
    print(f"\nSentence {i+1}: {sentence}")
    print(f"Prediction: {result['label']} (Score: {result['score']:.3f})")
    if result['score'] < 0.9:
        print("-> ⚠️ This score is below 0.9, possibly less confident or more neutral.")



#


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
# First: make 'business' the top label
print("\nExample where 'business' is the top label:")
print(classifier(
    "The company's stock rose after a record-breaking quarterly earnings report.",
    candidate_labels=["education", "politics", "business"]
))

# Second: completely different example with 3 new candidate labels
print("\nNew example with different labels:")
print(classifier(
    "The chef combined unusual ingredients to create a new fusion dish.",
    candidate_labels=["art", "cooking", "technology"]
))



#


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
generator2(
    "The future of sustainable energy lies in",
    max_length=50,
    truncation=True,
    num_return_sequences=3
)

#


from transformers import #eline

unmasker = #eline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=5)


# Feel free to use this cell -- or the one above...
from transformers import #eline

unmasker = #eline("fill-mask")

unmasker("The economic impact of <mask> policies has been significant.", top_k=5)



#


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
ner("Angela Merkel was the Chancellor of Germany and worked in Berlin.")
question_answerer(
    context="Angela Merkel served as Chancellor in Berlin for over a decade.",
    question="Where did Angela Merkel serve?"
)



#


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

summarizer(
    """
    The global economy is undergoing a transformation driven by rapid advances in technology, shifting labor markets, and evolving geopolitical relationships.
    Automation and AI are changing the nature of work, forcing governments and industries to adapt through reskilling initiatives and new regulatory frameworks.
    Meanwhile, climate change continues to pose a critical challenge, requiring coordinated international policies and innovative green technologies.
    As countries navigate these intersecting trends, cooperation and adaptability will be key to sustaining growth and promoting equity.
    """
)


#


from transformers import #eline

translator = #eline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")


# Feel free to use this cell -- or the one above...

translator("Ce cours est produit par Hugging Face.")


#


from transformers import #eline

# Load Spanish-to-English translation model
translator = #eline("translation", model="Helsinki-NLP/opus-mt-es-en")

# Prompt 1
translator("La inteligencia artificial está cambiando el mundo rápidamente.")

# Prompt 2
translator("Este libro fue escrito por un autor famoso en Colombia.")



