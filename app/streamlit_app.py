import sys
import os
from turtle import st
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.etl import load_and_clean_data
from scripts.clustering import apply_gmm_clustering
from scripts.recommender import train_recommender, recommend_jobs

# Load data
st.title("Job Segmentation & Recommendation App")
st.write("Segment jobs and get similar job recommendations based on job description.")

# Load and clean
df = load_and_clean_data("/Users/sajanshergill/Downloads/customer_segmentation/data/monster_com-job_sample.csv")

# Apply clustering
df = apply_gmm_clustering(df)

# Train recommender
sim_matrix = train_recommender(df)

# Sidebar selection
job_index = st.sidebar.slider("Select a Job Index", 0, len(df)-1, 100)
st.subheader("Selected Job")
st.write(df.iloc[job_index][['job_title', 'organization', 'location', 'cluster_gmm']])

# Recommend
st.subheader("💼 Similar Job Recommendations")
recommended = recommend_jobs(df, sim_matrix, job_index, top_n=5)
st.dataframe(recommended)

# Optional: Show cluster insights
st.subheader("📊 Cluster Insights")
st.bar_chart(df['cluster_gmm'].value_counts())
