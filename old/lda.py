from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from config import Config
import glob

no_features = 1000
file_paths = list(glob.glob(Config.TEXT_DATA_PATH))
print(len(file_paths))

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(input="filename", max_df=0.95, min_df=2, max_features=no_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(file_paths)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print(tfidf_feature_names)

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(input="filename", max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(file_paths)
tf_feature_names = tf_vectorizer.get_feature_names()

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


if __name__ == "__main__":
    no_top_words = 10
    display_topics(nmf, tfidf_feature_names, no_top_words)
    display_topics(lda, tf_feature_names, no_top_words)