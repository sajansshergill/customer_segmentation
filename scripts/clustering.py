from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture

def apply_gmm_clustering(df, n_clusters=5):
    le_sector = LabelEncoder()
    le_type = LabelEncoder()

    df['sector_encoded'] = le_sector.fit_transform(df['sector'])
    df['job_type_encoded'] = le_type.fit_transform(df['job_type'])

    features = df[['sector_encoded', 'job_type_encoded', 'description_length', 'median_salary', 'state_popularity']].fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    df['cluster_gmm'] = gmm.fit_predict(scaled_features)

    return df