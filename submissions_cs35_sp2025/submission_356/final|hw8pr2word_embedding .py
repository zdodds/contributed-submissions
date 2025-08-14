#
# hw8pr2.py ~ cs35 ~ Word Embeddings: Computing with word _meanings_
#

# our usual libraries
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns


# Most important for hw8 is gensim, the word-embedding (word2vec) library
# See if you already have it installed:
import gensim

# if not, try         
# # install gensim


## install gensim


# The word-embeddings are in the large file word2vec_model.txt

# Make sure that file is here:



#
# This function, read_word2vec_model, wraps the creation of a gensim model, let's say, m
#
#      To use it, run the line   m = read_word2vec_model()
#

from gensim.models import KeyedVectors

def read_word2vec_model(filename = "word2vec_model.txt"):  
    """ a function that reads a word2vec model from the file
        "word2vec_model.txt" and returns a model object that
        we will usually name m or model...
    """
    try:
        print("Starting to load the model in ", filename, "...")
        model = KeyedVectors.load_word2vec_format(filename, binary=False)
        print("Model loaded.\n")
    except FileNotFoundError as e:
        print(f"  [WARNING]    The file {filename} was not found.     [WARNING]  ")
        return None   # returning a placeholder, not a model

    # let's print some attributes
    print("The model built is", model, "\n")
    print("The vocabulary has", model.vectors.shape[0], "words")   # The vocabulary has 43981 words
    print("Each word is a vector of size", model.vector_size)  # 300
    print("\nTry m.get_vector('python') to see a the vector for 'python'!")
    print("m is dictionary-like, e.g., try 'python' in m\n")
    model.fill_norms()  # freezes the model, m, as-is (no more training)
    # we weren't going to train more, so no worries (in week7, at least)
    return model



# 
# best to run this only once... or once in a while, as needed
#

m = read_word2vec_model()


print(f"m is {m}")   # let's see how it prints...


'python' in m


m.get_vector('python')


'queen' in m


m.get_vector('queen')


if 'poptart' in m:
    print("That word is in m")
else:
    print("That word is NOT in m")


if 'cookies' in m:
    print("That word is in m")
else:
    print("That word is NOT in m")


#
# So, we can check the "meaning" of 'king', 'queen', 'snake', and 'python':
#
m.get_vector('snake')   # m.get_vector('queen')  m.get_vector('snake')   m.get_vector('king')

# which are not very useful ... until we compare them to other meanings:


# Let's see the built-in similarity method
m.similarity('python','snake')   # should be .6606292...

#


# First, a couple of variable-assignment statements
# These might start to feel disturbingly meta ...
python = m.get_vector('python')
snake = m.get_vector('snake')
language = m.get_vector('language')
code = m.get_vector('code')
queen = m.get_vector('queen')


import numpy as np
print(f"{np.linalg.norm(python) = }")  # this is the length of the vector - always 1, watch out for rounding

# the dot product is available in the numpy library
print(f"{np.dot(python, snake) = }")

# This is exactly the built-in similarity:


# we can use np to find the angle, in degrees, between the two vectors :-)
deg = np.degrees(np.arccos(0.66063))  # dot is cosine; converting from radians to degrees
print(f"...which is {deg:7.2f} degrees")

# for unit vectors, "dot product" is the same as the "cosine similarity"  
#     which is the cos of the angle between the two vectors


# Let's again see the built-in similarity method:
m.similarity('python','snake')   # should be the same .6606292...


m.distance( 'python', 'snake' )   # The distance is 1 minus the similarity


m.distance( 'python', 'coffee' )   # let's see...


m.similarity('python','coffee')


# with similarity, the biases of the datset can show through: let's check "programmer" vs "woman" and "man"
#
simw = m.similarity("programmer","woman")   # president  programmer
print(f"similarity w 'woman':  {simw:7.3f}")

simm = m.similarity("programmer","man")     # president  programmer
print(f"similarity w 'man':    {simm:7.3f}")

# notice that the values provide a starting-point to _quantify_ the bias in the dataset
# quantifying dataset bias is currently a very active area of research
# it would also be possible to compare both of these with 
print()
simprs = m.similarity("programmer","person")    # try it!
print(f"similarity w 'person': {simprs:7.3f}")


#
# the dataset will reflect the biases of the training data / source-texts  (6B tokens from Google News)
#
# for task#2, as you explore possibilities, see if there is a way for the similarity-scores to quantify the biases present...
# 
# key constraint: lots of tokens are missing...


# Let's compare multiple similarities:

python_snake = m.similarity('python','snake')
python_coffee = m.similarity('python','coffee')
snake_coffee = m.similarity('snake','coffee')

print(f"python_snake  similarity: {python_snake}")   # try :4.2f after the variable for formatting
print(f"python_coffee similarity: {python_coffee}")  # 4 characters wide, 2 places after the decimal point
print(f"snake_coffee  similarity: {snake_coffee}")


#
# Let's define an "odd-one-out" from any collection of words, 
# simply by considering all possible similarities (and adding them up for each word)

"""
here, for example:

python_snake  similarity: .66
python_coffee similarity: .02
snake_coffee  similarity: .08

So, summing the similarities for each word separately:
  python:  .66 + .02 == .68
  coffee:  .08 + .02 == .10
  snake:   .66 + .08 == .74

+++ In this case, "coffee" is the odd one out  (intuitive, in some ways)


# What do you think about python, serpent, snake?
# or python, serpent, snake, code?

"""



# notice that the split function makes creating lists-of-words a bit easier
initial_words = "snake serpent python code ai ml programming".split()
initial_words


#
# here is a _single_ keyword, with similarities computed against every word w in initial_words
key = 'python'

LoS = []
LoW = []
for w in initial_words:
    if w in m:  # m is the word, w present in the vocabulary?
        similarity = m.similarity(key,w)
        print(f"similarity between {key} and {w}: {similarity:6.2f}", )
        LoS.append( similarity )
        LoW.append( w )
    else:
        print(f"  __  {w}  __ was not in the vocabulary", )   # not every word will be present

print(f"LoS is {LoS}")
print(f"LoW is {LoW}")



#
# here is a signature line for odd_one_out (a starting point)
#

def odd_one_out( LoW, m ):
    """ 
        odd_one_out should take in LoW, a list-of-words
        odd_one_out also takes in m, a gensim word-embedding model (of type KeyedVectors)
        and it should return the word, w, in LoW that is _least_ like all the others

        The idea:  run a pairwise-comparison of all words!
        Then find the sum of the dis-similarities (and return the one with the largest sum)
    """

    similarity_sums = {}

    for word in LoW:
        if word not in m:
            continue

        total_similarity = 0
        for other in LoW:
            if other != word and other in m:
                total_similarity += m.similarity(word, other)

        similarity_sums[word] = total_similarity

    if not similarity_sums:
        print("None of the words were in the model.")
        return None

    odd_word = min(similarity_sums, key=similarity_sums.get)
    return odd_word


# example 1
LoW = ['apple', 'banana', 'grapes', 'cat']
odd_one_out(LoW, m)

#
# Successful, as the model was able to pick out cat, which is the odd one out of all the different types of fruit.
#


# example 2
LoW = ['red', 'blue', 'purple', 'black','green']
odd_one_out(LoW, m)

#
# Other, as the group of words were not more similar than other in any way, they are all just different colors.
#


# example 3
LoW = ['redvelvet', 'vanilla', 'chocolate', 'mushrooms','peanutbutter']
odd_one_out(LoW, m)

#
# Successful, as the model was able to pick out mushrooms, which is the odd one out of all the typical ice cream flavors.
#


#
# Create and run three examples - of at least 4 words each - for your odd_one_out function.
# For example,
#        LoW = "apple banana cat pear".split()
#
# Also, note if you would describe them as successful, unsuccessful, or "other" !


#
# This is an alternative view of the data -- within a projection of word-embedding space itself
# This is in 2d. A 3d version is here: https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/index.html
#
# Let's see the word vectors in two dimensions!
#
def visualize_wordvecs(wordlist, model):
    """ example of finding an outlier with word2vec and graphically """
    # 
    # Are all of the works in the model?
    #
    for w in wordlist:
        if w not in model:
            print("Aargh - the model does not contain", w)
            print("Stopping...")
            return
    #
    # Next, we use PCA, Principal Components Analysis, to toss out 298 dimensions!
    # and create a scatterplot of the words...
    #
    # Intuitive description of PCA:   https://setosa.io/ev/principal-component-analysis/
    #
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy

    pca = PCA(n_components=2)             # 2 dimensions
    pca_model = pca.fit(model.vectors)    # all 43,981 words with 300 numbers each!
    LoM = [model.get_vector(w) for w in wordlist]   # list of models for each word w
    
    word_vectors = numpy.vstack(LoM)     # vstack creates a vertical column from a list
    transformed_words = pca_model.transform(word_vectors)  # transform to our 2d space

    # scatterplot
    plt.subplots(figsize=(15,10))  # (18, 12)
    plt.scatter(transformed_words[:,0],transformed_words[:,1])
    
    # This is matplotlib's code for _annotating_ graphs (yay!)
    for i, word in enumerate(wordlist):
        plt.annotate(word, (transformed_words[i,0], transformed_words[i,1]), size='large')
        # it's possible to be more sophisticated, but this is ok for now

    plt.show()
    return


#
# Example of calling visualize_wordvecs...
#
#LoW = "breakfast lunch dinner coffee snake senate".split()     #  cereal python, one two three four five twelve
LoW = "one two three four".split() 
#LoW = "breakfast lunch dinner".split()  
visualize_wordvecs(LoW, m)    


# Starting point for visualizing 2d similarity via a heat map

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

# adaptation of the previous similarity-based code
key = 'python'      # this is a _single_ word - the task is to loop over the list, perhaps named Keys
LoS = []
LoW = []
for w in initial_words:
    if w in m:  # is the word, w present in the vocabulary?
        similarity = m.similarity(key,w)
        print(f"similarity between {key} and {w}: {similarity:6.2f}", )
        LoS.append( similarity )
        LoW.append( w )
    else:
        print(f"  __  {w}  __ was not in the vocabulary", )   # not every word will be present

print(f"LoS is {LoS}")
print(f"LoW is {LoW}")

my_data_list = [ LoS ]
my_dataframe = pd.DataFrame(my_data_list, columns=LoW)


# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(15,10))  # (18, 12)
sns.heatmap(data=my_dataframe, annot=True, fmt="4.2f", linewidths=2, yticklabels=["python"], square=True, cmap="Purples", cbar=False, ax=ax)

ylocs, ylabels = plt.yticks()
plt.setp(ylabels, rotation=0, fontsize=15)
xlocs, xlabels = plt.xticks()
plt.setp(xlabels, rotation=70, fontsize=15)
"Result:"

# The goal is to output a square heatmap with all of the similarities plotted...


#
# Let's take a look at some additional "geometry" of word-meanings (cool!)
#

m.most_similar(positive='python', topn=10)  # negative='snake'


#
# With this most_similar method, we can "subtract" vectors, too:
#

m.most_similar(positive='python', negative='snake', topn=10) 


#
# Here, see if you can determine the analogy that is being computed using word embeddings:
# 

m.most_similar(positive=['king','woman'], negative=['man'], topn=10) 


# 
# This problem is about building and testing analogies...
# 
# This function has a hard-coded set of words, i.e., 'woman', 'king', and 'man'
# Your tasks:
#      + add inputs to the function 
#
def test_most_similar(m):
    """ example of most_similar """
    print("Testing most_similar on the king - man + woman example...")
    #results = m.most_similar(positive=['woman', 'king'], negative=['man'], topn=10) # topn == # of results
    results = m.most_similar(positive=['Italy', 'Paris'], negative=['France'], topn=10) # topn == # of results
    return results

hard_coded_results = test_most_similar(m)
hard_coded_results


#
# here is a starting point for generate_analogy:

def generate_analogy(w1, w2, w3, m):
    """
    Solves the analogy: w1 is to w2 as w3 is to ??
    Returns the best guess word from the model m.
    """
    for w in [w1, w2, w3]:
        if w not in m:
            print(f"'{w}' is not in the model.")
            return None

    result = m.most_similar(positive=[w2, w3], negative=[w1], topn=1)

    # Return the best match (word only)
    return result[0][0]


#
# be sure to test -- and show off some of your own that work (and some that don't)
#

generate_analogy("man", "king", "woman", m)  
# generate_analogy("Germany", "Berlin", "France", m)


generate_analogy("walk", "walking", "swim", m)


generate_analogy("sushi", "Japan", "taco", m) # fail?


#
# your check_analogy function
#
def check_analogy(w1, w2, w3, w4, m):
  
    for w in [w1, w2, w3, w4]:
        if w not in m:
            print(f"'{w}' not in the model.")
            return 0

    try:
        results = m.most_similar(positive=[w2, w3], negative=[w1], topn=100)
    except KeyError:
        return 0

    for rank, (word, _) in enumerate(results):
        if word == w4:
            return 100 - rank  # Higher score for higher ranking

    return 0  # Not found in top 100



print("Top quartile:", check_analogy("man", "king", "woman", "queen", m))
print("Second quartile:", check_analogy("music", "guitar", "painting", "canvas", m))
print("Third quartile:", check_analogy("fruit", "flower", "stump", "tree", m))
print("Bottom quartile:", check_analogy( "fall", "roast", "winter", "soup", m ))



# completed with jen lim


