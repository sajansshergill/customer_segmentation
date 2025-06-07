from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_recommender(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['job_description'])
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix

def recommend_jobs(df, sim_matrix, job_index, top_n=5):
    similarity_scores = list(enumerate(sim_matrix[job_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_jobs = [i[0] for i in similarity_scores[1:top_n+1]]
    return df.iloc[similar_jobs][['job_title', 'organization', 'location']]