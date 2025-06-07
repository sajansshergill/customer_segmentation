# 📊 Customer Segmentation and Job Recommendation App


This project demonstrates how to segment job postings and build a personalized recommender system using behavioral and textual data. It uses real-world job listing data and applies unsupervised learning to uncover patterns and personas.

----


## 📊 Live Demo (GIF)

![Dashboard Demo](assets/demo.gif)

---

## Demo Video



---


# 🚀 Features
- ETL Pipeline: Clean and preprocess job data (sector, job type, salary, state).

- Clustering: Segment jobs using Gaussian Mixture Models (GMM).

- Content-Based Recommender: Recommend similar jobs based on TF-IDF vectorization of job descriptions.

- Streamlit App: Interactive interface to explore job clusters and recommendations.

----

# 🧠 Tech Stack

- Python 3.10+

- pandas, numpy

- scikit-learn

- streamlit

---

# 📁 Project Structure

customer_segmentation/
├── app/
│   └── streamlit_app.py            # Streamlit UI logic
├── data/
│   └── monster_com-job_sample.csv # Dataset
├── scripts/
│   ├── etl.py                      # Data loading & feature engineering
│   ├── clustering.py               # GMM clustering logic
│   └── recommender.py              # TF-IDF + recommender logic
├── README.md


---

# 📌 Use Case

This app mimics behavior seen on job boards like Lensa, where users can:

- Explore job clusters (e.g., "Tech Generalists" or "Finance Niche Experts")

- Get real-time job recommendations based on text similarity
