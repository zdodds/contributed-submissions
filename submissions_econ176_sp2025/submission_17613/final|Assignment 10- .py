





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
     "Aargh! I love this so much!"]  # could change this to "love"  :)
)


# Feel free to use this cell -- or edit the one above...

classifier(
    ["I am skeptical about the future of AI.",
     "I am optimistic about the future of AI.",
     "Today is a sunny day.",
     "What should I eat today?",
     "What day of the week is it?",
     "I am dreading going to the park which is my favorite place.",
     "I am excited to go to the park which is my least favorite place."])


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
classifier(
    "I have a meeting tomorrow with a client.",
    candidate_labels=["education", "politics", "business"],
)


classifier(
    "I want to travel this summer.",
    candidate_labels=["Tokyo", "London", "Hawaii"],
)


classifier(
    "I like to eat chocolate mochi xiaolongbao.",
    candidate_labels=["Food", "Dessert", "Sweet"],
)


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
    "In this course, we will teach you how to",
    max_length=50,
    truncation=True,
    num_return_sequences=3,
)


generator2(
    "In this course, we will teach you how to",
    max_length=50,
    truncation=True,
    num_return_sequences=3,
)


generator2(
    "In this course, we will teach you how to",
    max_length=50,
    truncation=True,
    num_return_sequences=3,
)


generator2(
    "I am hungry, I will go eat",
    max_length=25,
    truncation=True,
    num_return_sequences=3,
)


generator2(
    "I am hungry, I will go eat",
    max_length=25,
    truncation=True,
    num_return_sequences=3,
)


generator2(
    "I am hungry, I will go eat",
    max_length=25,
    truncation=True,
    num_return_sequences=3,
)


from transformers import #eline

unmasker = #eline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=5)


# Feel free to use this cell -- or the one above...

unmasker = #eline("fill-mask")
unmasker("I will eat <mask> for lunch today.", top_k=5)


from transformers import #eline

ner = #eline("ner", grouped_entities=True)
print("\n")

ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")


# My example

ner = #eline("ner", grouped_entities=True)
print("\n")

ner("She is a professor at Harvey Mudd College in California.")


from transformers import #eline

question_answerer = #eline("question-answering")
print("\n")

question_answerer(
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    question="Where do I work?",
)


# My example

question_answerer = #eline("question-answering")
print("\n")

question_answerer(
    context="My favorite ice cream flavors are chocolate and banana, but I like chocolate more.",
    question="What is my one favorite ice cream flavor?",
)


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

summarizer = #eline("summarization")
summarizer(
    """
    OpenAI said late Wednesday that it hired Fidji Simo, the chief executive of Instacart,
    to take on a new role running the artificial intelligence company’s business and operations teams.

    In a blog post, Sam Altman, OpenAI’s chief executive, said he would remain in charge as the head of
    the company. But Ms. Simo’s appointment as chief executive of applications would free him up to focus
    on other parts of the organization, including research, computing and safety systems, he said.

    “We have become a global product company serving hundreds of millions of users worldwide and growing
    very quickly,” Mr. Altman said in the blog post. He added that OpenAI had also become an “infrastructure
    company” that delivered artificial intelligence tools at scale.
"""
)


from transformers import #eline

translator = #eline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")


# Feel free to use this cell -- or the one above...

translator("Ce cours est produit par Hugging Face.")


#


translator = #eline("translation", model="Helsinki-NLP/opus-mt-zh-en")
translator("我们试试这个翻译的怎么样")


translator = #eline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-es")
translator("The sky is blue like the ocean like the sky.")


