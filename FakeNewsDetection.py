"""
Fake News Detection Deep Learning Project
The dataset was taken from kaggle - https://www.kaggle.com/c/fake-news/data

Solution Methodology

Step 1: 
After importing the dataset, the first step is to preprocess the data using different techniques like removing stop words using NLTK.

Step 2:
-    Creating GloVe (global vectors for word representation).
     The algorithm mapping words into a meaningful space where the distance between words is related to semantic similarity
-    Creating word to index dictionary containing the words in the given text
-    Building the word embedding matrix


Step 3: 
Training the classifier with Tensorflow, Keras to build a classifier and predict fake and true news

@author: hagar chen
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from tensorflow.keras.models import Model
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import regularizers

from wordcloud import WordCloud
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Dropout, Dense, Input,Embedding

#%%Import the dataset from the csv file
train = pd.read_csv (r"C:\Users\roeec\OneDrive\Desktop\Hagar Project\binaryModel\train.csv")
#train = pd.read_csv (".../input/train.csv")
# Data structure
print(train.head())
print(train.shape)
print(train.isnull().sum())
train.dropna(inplace = True)
print('\n',train.isnull().sum())

#%% Extract the prediction and show the distribution
y_train = train['label']
import seaborn as sns
sns.countplot(train['label']).set(title='Labels Count')

X_train = train['text']
X_train = X_train.str.lower() #returns the lowercase string
X_train = X_train.tolist() #convert to a list

#plotting a word cloud to show the most frequently occurring words
myString = ' '.join(map(str,X_train))
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(myString)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
#%% Preprocess the data using different techniques like tokenising, stemming, removing stop words
def remove_stopwords(data):
    """
    remove stopwords from list of strings python.
    
    Arguments:
    data -- list of strings (with stopwords)
        
    Returns:
    output_array -- list of strings (without stopwords)
    """
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stopwords:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array
#remove stop words
filtered_sentence = []
stopwords=set(stopwords.words('english'))
x_train_sw=remove_stopwords(X_train)



#%% convert words to indexes
def read_glove_vector(glove_vec):
    """
    read the content of the GloVe Vector file
    Arguments:
    glove_vec -- GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
    
    Returns: 
    word_to_vec_map -- a dictionary that maps the words to their respective word embeddings
    """
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map


# word_to_vec_map: dictionary mapping words to their GloVe vector representation
word_to_vec_map = read_glove_vector("C:/Users/roeec/OneDrive/Desktop/Hagar Project/glove.6B.50d.txt")

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train_sw)
words_to_index = tokenizer.word_index


# =============================================================================
# maximum length of one review we will use for our training, 
# I have kept the value to be 50, we will pad this later so all the inputs have the same length
# =============================================================================

maxLen = 1000

vocab_len = len(words_to_index) #The number of words in the training set
embed_vector_len = word_to_vec_map['moon'].shape[0] #The length of the mapping vector

emb_matrix = np.zeros((vocab_len, embed_vector_len))

for word, index in words_to_index.items():
  embedding_vector = word_to_vec_map.get(word)
  if embedding_vector is not None:
    emb_matrix[index-1, :] = embedding_vector
    
#%% split the data into training and test set
X_train, X_test,Y_train, Y_test = train_test_split(x_train_sw ,y_train, test_size=0.2, random_state = 45)

#  Tokenizing and paddinf input data and converty output data to categorical.
X_train_indices = tokenizer.texts_to_sequences(X_train)
X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')

X_test_indices = tokenizer.texts_to_sequences(X_test)
X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')

val_dataset = (X_test_indices,Y_test)
#%%  Building the predictive model
def pretrained_embedding_layer(word_to_vec_map, word_to_index,maxLen):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary
    maxLen --  maximum number of words read from the database in one chunk  

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(words_to_index) + 1              # adding 1 to fit Keras embedding (requirement)
    any_word = list(word_to_vec_map.keys())[0]
    embed_vector_len = word_to_vec_map[any_word].shape[0]    # define dimensionality of your GloVe word vectors (= 50)
      
    # Initialize the embedding matrix as a numpy array of zeros.
    emb_matrix = np.zeros((vocab_len,embed_vector_len))
    
    # Set each row "idx" of the embedding matrix to be the word vector representation of the idx'th word of the vocabulary
    for word, index in words_to_index.items():
      embedding_vector = word_to_vec_map.get(word)
      if embedding_vector is not None:
        emb_matrix[index, :] = embedding_vector

    # Define Keras embedding layer with the correct input and output sizes
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen, weights = [emb_matrix], trainable=True)
        
    return embedding_layer


def  FakeNews_Model(input_shape,word_to_vec_map,words_to_index,maxLen):
    """
    model :  Building the predictive model to detect fake news
    Arguments: 
    input_shape -- input dimension
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary
    maxLen --  maximum number of words read from the database in one chunk  
    
    Returns: model -- the predictive model
    """    
 
    X_indices = Input(shape = input_shape)
    # Create the embedding layer pretrained with GloVe Vectors 
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, words_to_index,maxLen)
    embeddings = embedding_layer(X_indices)
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.3)(X)
    X = LSTM(128)(X)
    X = Dense(1, activation='sigmoid',kernel_regularizer=regularizers.L2(l2=0.01))(X)
    model = Model(inputs=X_indices, outputs=X)
    
    return model
#%% Building the model
model = FakeNews_Model((maxLen,),word_to_vec_map,words_to_index,maxLen)
model.summary()

#%% Training the model
pretrainedNetwork = 1 #load pretrained network
if pretrainedNetwork:
    # load the model from disk
    model = tf.keras.models.load_model('saved_model/my_model')
else:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    adam = keras.optimizers.Adam(learning_rate = lr_schedule)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_indices, Y_train, batch_size=32 ,epochs=15, shuffle=True,validation_data = val_dataset)

#%% Evaluate the performance
loss, acc = model.evaluate(X_test_indices, Y_test)
print("accuracy: {:5.2f}%".format(100 * acc))
y_proba = model.predict(X_test_indices)
y_classes =np.where(y_proba > 0.5, 1,0)
cm = confusion_matrix( y_classes,Y_test)
cm_display = ConfusionMatrixDisplay(cm).plot()
#%% save the model to disk
filepath = 'trainedModel'
tf.keras.models.save_model(model,
    filepath,
    overwrite=True,
    include_optimizer=True)