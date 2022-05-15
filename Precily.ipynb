{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Paragraph Similarity"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Importing the dataset and libraries "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "id": "A5SJ38hfKSTd",
    "outputId": "6d60f9f4-a36d-4ee7-d473-7cbe8540f037"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text1</th>\n",
       "      <th>text2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>broadband challenges tv viewing the number of ...</td>\n",
       "      <td>gardener wins double in glasgow britain s jaso...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>rap boss arrested over drug find rap mogul mar...</td>\n",
       "      <td>amnesty chief laments war failure the lack of ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>player burn-out worries robinson england coach...</td>\n",
       "      <td>hanks greeted at wintry premiere hollywood sta...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>hearts of oak 3-2 cotonsport hearts of oak set...</td>\n",
       "      <td>redford s vision of sundance despite sporting ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>sir paul rocks super bowl crowds sir paul mcca...</td>\n",
       "      <td>mauresmo opens with victory in la amelie maure...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               text1  \\\n",
       "0  broadband challenges tv viewing the number of ...   \n",
       "1  rap boss arrested over drug find rap mogul mar...   \n",
       "2  player burn-out worries robinson england coach...   \n",
       "3  hearts of oak 3-2 cotonsport hearts of oak set...   \n",
       "4  sir paul rocks super bowl crowds sir paul mcca...   \n",
       "\n",
       "                                               text2  \n",
       "0  gardener wins double in glasgow britain s jaso...  \n",
       "1  amnesty chief laments war failure the lack of ...  \n",
       "2  hanks greeted at wintry premiere hollywood sta...  \n",
       "3  redford s vision of sundance despite sporting ...  \n",
       "4  mauresmo opens with victory in la amelie maure...  "
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "data = pd.read_csv('Precily_Text_Similarity.csv')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "NSJGQ6IHNGuK",
    "outputId": "7d4c4722-037c-4691-9239-e4144aad5f42"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3000"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Working"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we are using bag of words model\n",
    "In order to determine the similarity between two text paragraphs,each of them is converted into a vector of words and their counts. The matrix formed is known as Document Term Matrix (DTM) where each row is a document and each column represents the term/token in the document.\n",
    "While creating the DTM, some pre-processing is done, i.e, all words are converted to lower case and punctuation is removed.This matrix is generated using count vectorizer.\n",
    "Then we calculate cosine similarity between the two paragraphs.\n",
    "The less is the value of cosine distance more is the similarity."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### DTM for 1st row of paragraphs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   100  100m  12  120  15  1500m  16  19  1998  200  ...  win  winning  wins  \\\n",
      "0    2     0   2    1   0      0   0   1     0    0  ...    0        0     0   \n",
      "1    0     1   0    0   1      2   1   0     1    1  ...    3        1     1   \n",
      "\n",
      "   with  women  won  world  year  years  yet  \n",
      "0     7      0    0      0     6      1    0  \n",
      "1    10      4    6      3     1      0    1  \n",
      "\n",
      "[2 rows x 414 columns]\n",
      "Similarity between these two paragraphs is  0.6852556785505336\n"
     ]
    }
   ],
   "source": [
    "dtm= count_vectorizer.fit_transform([data['text1'][0],data['text2'][0]])\n",
    "print(pd.DataFrame(data=dtm.toarray(), columns=count_vectorizer.get_feature_names()))\n",
    "val=1-cosine(dtm[0].toarray(),dtm[1].toarray())\n",
    "print(\"Similarity between these two paragraphs is \",val)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Calculating similarity and storing in the dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 435
    },
    "id": "y3vmc5TQLlSQ",
    "outputId": "9c2a5a60-8af0-4874-b4d9-2931f5b42507"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text1</th>\n",
       "      <th>text2</th>\n",
       "      <th>similarity count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>broadband challenges tv viewing the number of ...</td>\n",
       "      <td>gardener wins double in glasgow britain s jaso...</td>\n",
       "      <td>0.685256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>rap boss arrested over drug find rap mogul mar...</td>\n",
       "      <td>amnesty chief laments war failure the lack of ...</td>\n",
       "      <td>0.431582</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>player burn-out worries robinson england coach...</td>\n",
       "      <td>hanks greeted at wintry premiere hollywood sta...</td>\n",
       "      <td>0.564167</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>hearts of oak 3-2 cotonsport hearts of oak set...</td>\n",
       "      <td>redford s vision of sundance despite sporting ...</td>\n",
       "      <td>0.665098</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>sir paul rocks super bowl crowds sir paul mcca...</td>\n",
       "      <td>mauresmo opens with victory in la amelie maure...</td>\n",
       "      <td>0.556794</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               text1  \\\n",
       "0  broadband challenges tv viewing the number of ...   \n",
       "1  rap boss arrested over drug find rap mogul mar...   \n",
       "2  player burn-out worries robinson england coach...   \n",
       "3  hearts of oak 3-2 cotonsport hearts of oak set...   \n",
       "4  sir paul rocks super bowl crowds sir paul mcca...   \n",
       "\n",
       "                                               text2  similarity count  \n",
       "0  gardener wins double in glasgow britain s jaso...          0.685256  \n",
       "1  amnesty chief laments war failure the lack of ...          0.431582  \n",
       "2  hanks greeted at wintry premiere hollywood sta...          0.564167  \n",
       "3  redford s vision of sundance despite sporting ...          0.665098  \n",
       "4  mauresmo opens with victory in la amelie maure...          0.556794  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "count_vectorizer = CountVectorizer()\n",
    "from scipy.spatial.distance import cosine\n",
    "\n",
    "sim=[]\n",
    "for ans in range(len(data)):\n",
    "    dtm= count_vectorizer.fit_transform([data['text1'][ans],data['text2'][ans]])\n",
    "    pd.DataFrame(data=dtm.toarray(), columns=count_vectorizer.get_feature_names())\n",
    "    sim.append(1-cosine(dtm[0].toarray(),dtm[1].toarray()))\n",
    "    \n",
    "data['similarity count']=sim\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 328
    },
    "id": "4K1Nul6tW5FS",
    "outputId": "c39cf682-dc3d-4bee-fdeb-1a630c69d4e1"
   },
   "outputs": [],
   "source": [
    "data1 = pd.read_csv('Precily_Text_Similarity.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "PD_qiKvyTynD"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text1</th>\n",
       "      <th>text2</th>\n",
       "      <th>TIDF similarity count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>broadband challenges tv viewing the number of ...</td>\n",
       "      <td>gardener wins double in glasgow britain s jaso...</td>\n",
       "      <td>0.551968</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>rap boss arrested over drug find rap mogul mar...</td>\n",
       "      <td>amnesty chief laments war failure the lack of ...</td>\n",
       "      <td>0.312518</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>player burn-out worries robinson england coach...</td>\n",
       "      <td>hanks greeted at wintry premiere hollywood sta...</td>\n",
       "      <td>0.408212</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>hearts of oak 3-2 cotonsport hearts of oak set...</td>\n",
       "      <td>redford s vision of sundance despite sporting ...</td>\n",
       "      <td>0.530670</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>sir paul rocks super bowl crowds sir paul mcca...</td>\n",
       "      <td>mauresmo opens with victory in la amelie maure...</td>\n",
       "      <td>0.424369</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               text1  \\\n",
       "0  broadband challenges tv viewing the number of ...   \n",
       "1  rap boss arrested over drug find rap mogul mar...   \n",
       "2  player burn-out worries robinson england coach...   \n",
       "3  hearts of oak 3-2 cotonsport hearts of oak set...   \n",
       "4  sir paul rocks super bowl crowds sir paul mcca...   \n",
       "\n",
       "                                               text2  TIDF similarity count  \n",
       "0  gardener wins double in glasgow britain s jaso...               0.551968  \n",
       "1  amnesty chief laments war failure the lack of ...               0.312518  \n",
       "2  hanks greeted at wintry premiere hollywood sta...               0.408212  \n",
       "3  redford s vision of sundance despite sporting ...               0.530670  \n",
       "4  mauresmo opens with victory in la amelie maure...               0.424369  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "# import pandas as pd\n",
    "# sent1 = \"India is a republic country. We are proud Indians.\"\n",
    "# sent2 = \"The current Prime Minister of India is Shri. Narendra Modi.\"\n",
    "tfidf_vectorizer = TfidfVectorizer()\n",
    "sim1=[]\n",
    "for ans in range(len(data)):\n",
    "    tfidf_vectors= tfidf_vectorizer.fit_transform([data['text1'][ans],data['text2'][ans]])\n",
    "    pd.DataFrame(data=tfidf_vectors.toarray(),columns=tfidf_vectorizer.get_feature_names())\n",
    "    sim1.append(1-cosine(tfidf_vectors[0].toarray(),tfidf_vectors[1].toarray()))\n",
    "\n",
    "data1['TIDF similarity count']=sim1\n",
    "data1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "        and       any        as        be       can  classification   columns  \\\n",
      "0  0.288591  0.135202  0.000000  0.192394  0.270404        0.096197  0.000000   \n",
      "1  0.281481  0.000000  0.197806  0.070370  0.000000        0.070370  0.098903   \n",
      "\n",
      "      could     count  countvectorizer  ...        to     twice    unique  \\\n",
      "0  0.135202  0.000000         0.000000  ...  0.192394  0.000000  0.000000   \n",
      "1  0.000000  0.197806         0.098903  ...  0.070370  0.098903  0.197806   \n",
      "\n",
      "       used  valuable        we      were     which      word     words  \n",
      "0  0.000000  0.135202  0.096197  0.000000  0.135202  0.096197  0.192394  \n",
      "1  0.098903  0.000000  0.140741  0.098903  0.000000  0.140741  0.211111  \n",
      "\n",
      "[2 rows x 73 columns]\n",
      "0.3587795765041317\n"
     ]
    }
   ],
   "source": [
    "s1= \"Max_df stands for maximum document frequency. Similar to min_df, we can ignore words which occur frequently. These words could be like the word ‘the’ that occur in every document and does not provide and valuable information to our text classification or any other machine learning model and can be safely ignored.\"\n",
    "#s2= \"Max_df stands for maximum document frequency. Similar to min_df, we can ignore words which occur frequently. These words could be like the word ‘the’ that occur in every document and does not provide and valuable information to our text classification or any other machine learning model and can be safely ignored.\"\n",
    "s2= \"We have 8 unique words in the text and hence 8 different columns each representing a unique word in the matrix. The row represents the word count. Since the words ‘is’ and ‘my’ were repeated twice we have the count for those particular words as 2 and 1 for the rest.Countvectorizer makes it easy for text data to be used directly in machine learning and deep learning models such as text classification.\"\n",
    "#s2=\"Divya Jayendra Chaudhari\"\n",
    "# dtm= count_vectorizer.fit_transform([s1,s2])\n",
    "# print(pd.DataFrame(data=dtm.toarray(), columns=count_vectorizer.get_feature_names()))\n",
    "# print(1-cosine(dtm[0].toarray(),dtm[1].toarray()))\n",
    "\n",
    "tfidf_vectors= tfidf_vectorizer.fit_transform([s1,s2])\n",
    "print(pd.DataFrame(data=tfidf_vectors.toarray(),columns=tfidf_vectorizer.get_feature_names()))\n",
    "print(1-cosine(tfidf_vectors[0].toarray(),tfidf_vectors[1].toarray()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'model' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[1;32mIn [1]\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mjoblib\u001b[39;00m\n\u001b[1;32m----> 3\u001b[0m joblib\u001b[38;5;241m.\u001b[39mdump(\u001b[43mmodel\u001b[49m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPrecily.joblib\u001b[39m\u001b[38;5;124m'\u001b[39m)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'model' is not defined"
     ]
    }
   ],
   "source": [
    "\n",
    "model = tfidf_vectors\n",
    "\n",
    "import joblib\n",
    "\n",
    "joblib.dump(model, 'Precily.joblib')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "Precily.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
