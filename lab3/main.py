from gensim import corpora
import requests
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import textwrap
import string

class c:
    OKBLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    END = '\033[0m'

url = "http://www.gutenberg.org/files/11/11-0.txt"
response = requests.get(url)
response.encoding = 'utf-8'
book_text = response.text

tokenizer = nltk.WordPunctTokenizer()
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.add("im")
stop_words.add("oh")
stop_words.add("em")

lemmatizer = WordNetLemmatizer()

def preprocess_document(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

processed_text = preprocess_document(book_text)
text = processed_text

positions = [match.start() for match in re.finditer(r'\bend project\b', text)]

if len(positions) >= 0:
    text = text[:positions[0]]
else:
    print("END not found")

positions = [match.start() for match in re.finditer(r'\bchapter\b', text)]
if len(positions) >= 2:
    text = text[positions[12]:]
else:
    print("CHAPTER I not found")

chapters = re.split(r'\bchapter [ivx]+\b', text)
chapters = [re.sub(r'^\s*\.+\s*', '', chapter) for chapter in chapters if chapter.strip()]

processed_chapters = [preprocess_document(chapter) for chapter in chapters]

for idx in range(len(processed_chapters)):
    processed_chapters[idx] = processed_chapters[idx].translate(str.maketrans('', '', string.punctuation))

tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=20)
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_chapters)

top_words = {}
feature_names = tfidf_vectorizer.get_feature_names_out()
for idx, row in enumerate(tfidf_matrix):
    top_indices = row.toarray()[0].argsort()[-20:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_words[f"Chapter {idx+1}"] = top_features

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(chapters)

id2word = corpora.Dictionary([ch.split() for ch in chapters])
corpus = [id2word.doc2bow(ch.split()) for ch in chapters]

num_topics = 12

lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(tfidf_matrix)

for topic_idx, topic in enumerate(lda_model.components_):
    print(f"{c.YELLOW}Chapter #{topic_idx + 1}:{c.END}")
    print(f"{c.PURPLE}LDA:{c.END}", end=" ")
    print(*[tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-20:]])
    print(f"{c.CYAN}TF-IDF:{c.END}", end=" ")
    print(*top_words[f"Chapter {topic_idx + 1}"])
    print()
