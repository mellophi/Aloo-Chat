import os
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# NLTK
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


def make_doc_list(root='teext'):
    data_folder = os.path.join(os.getcwd(), root)
    filepaths = []
    docs = []

    for file in os.listdir(root):
        filepaths.append(os.path.join(data_folder, file))
    
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            docs.append(f.read())
    
    return docs

# filepath = "/home/ayon/Aloo-Chat/teext/TOI_001_Need_to_improve_crop_productivity_to_meet_demand_of_not_only_India_but_world_Tomar.txt"

topic_model = BERTopic()
docs = make_doc_list()

topic, probs = topic_model.fit_transform(docs)

print(topic_model.get_topic_info())

# docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
# print(docs[0])

