





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


my_sentences = [
    "The stock market is experiencing a lot of volatility lately.",
    "This new technology seems promising for future development.",
    "I'm unsure about the outcome of this project.",
    "The customer service was neither good nor bad.",
    "It's just an average day.",
    "The report was factually correct but lacked depth."
]
classifier(my_sentences)

# Agree with all classifications, impressive!


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
    "The company announced record profits in the last quarter.",
    candidate_labels=["education", "politics", "business"],
)

#




new_labels = ["sports", "technology", "food", "travel"]

# Example for sports
classifier(
    "The home team won the championship game in a thrilling overtime victory.",
    candidate_labels=new_labels,
)




# Example for technology
classifier(
    "The latest smartphone features a revolutionary new camera system and AI capabilities.",
    candidate_labels=new_labels,
)




# Example for food
classifier(
    "This restaurant is famous for its authentic pasta dishes and delightful desserts.",
    candidate_labels=new_labels,
)




# Example for travel
classifier(
    "We're planning a backpacking trip through Southeast Asia next summer.",
    candidate_labels=new_labels,
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

new_prompt = "The future of finance will likely involve"
generator2(
    new_prompt,
    max_length=50,
    truncation=True,
    num_return_sequences=3,
)

#

# The generated text seems to be consistently on theme but it doesn't make much sense.


from transformers import #eline

unmasker = #eline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=5)


# Feel free to use this cell -- or the one above...

unmasker("Artificial intelligence is rapidly transforming the <mask> industry.", top_k=5)

# All of these mask-fill suggustions make sense. It is impressive this simple model has the world knowledge to generate believable results.


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

my_ner_sentence = "[REDACTED] attends Claremont McKenna College, which is located in California, and is studying Financial Technology."
ner(my_ner_sentence)


#


my_context = "The European Central Bank, headquartered in Frankfurt, Germany, recently adjusted its monetary policy in response to rising inflation."
my_question = "Where is the European Central Bank located?"
question_answerer(
    context=my_context,
    question=my_question,
)


# Looks like these models completed all the tasks I indicated correctly


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

my_text_to_summarize = """
Understanding how citizens form opinions in an era dominated by partisan media is crucial, as reliance on sources like Fox News and MSNBC can shape divergent perceptions of reality. This thesis moves beyond static channel choice to investigate the relationship between exposure to dynamic, time-varying characteristics of partisan cable news content and individual policy attitudes. It specifically asks how the volume (salience) and ideological slant (framing) of recent news coverage associate with opinions on key policy issues. To address this, the study integrates individual-level data from the Cooperative Election Study (CES) from 2020-2023 (approx. 68,000 observations across seven policy issues) with a high-frequency dataset of Fox News and MSNBC transcripts. Leveraging large language models, every relevant news segment broadcast during this period was classified for topic and ideological stance (liberal/conservative). These classifications were used to construct daily time series of content volume and slant for each channel and policy topic. For each CES respondent, 7-day rolling aggregates of media content ending the day before their interview were calculated and interacted with their self-reported channel viewership status. Ordinary Least Squares regression models were estimated to predict standardized policy attitudes (0=Conservative, 1=Liberal), incorporating these media exposure interaction terms alongside controls for demographics, ideology, party identification, and county and year fixed effects. The analysis reveals a robust association between the ideological slant of recent media exposure and policy attitudes. Controlling for content volume and other factors, exposure to more liberal-slanted coverage in the preceding week was strongly and statistically significantly associated with holding more liberal views across nearly all policy domains for viewers of both channels (most p<0.001). For instance, a hypothetical shift from perfectly balanced to exclusively liberal slant (+1 change) in Fox News' coverage of assault weapons over a week was associated with a 0.146 (SE=0.012) increase in support for a ban among Fox viewers. The effects associated with normalized slant generally dominated those related to content volume, which were smaller and less consistent. A simpler model examining only the net directional tone (liberal minus conservative segments) also showed highly significant associations in the expected direction (e.g., each additional net liberal Fox segment on CO2 regulation associated with a 0.068 point increase in support, SE=0.004, p<0.001). These findings provide quantitative support for the hypothesis that how partisan news outlets frame issues is strongly correlated with audience opinion, distinct from mere channel selection or the overall amount of coverage. This demonstrates the potential role of specific, time-varying media narratives in reinforcing and potentially driving policy attitude polarization, showing the value of integrating granular, AI-assisted content analysis with large-scale survey data to understand media influence.
"""
summarizer(my_text_to_summarize)

# This seems like an ok summary of my thesis abstract!


from transformers import #eline

translator = #eline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")


# Feel free to use this cell -- or the one above...

translator("Ce cours est produit par Hugging Face.")


#


from transformers import #eline

translator_en_es = #eline("translation", model="Helsinki-NLP/opus-mt-en-es")
print("English to Spanish translator loaded.")


# Prompt 1
translation1 = translator_en_es("Hello, how are you today?")
print(translation1)

# Prompt 2
translation2 = translator_en_es("This is an interesting exercise in machine learning.")
print(translation2)


# Translation worked! Great notebook, thanks!


