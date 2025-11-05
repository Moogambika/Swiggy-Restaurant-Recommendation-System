import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.preprocessing import StandardScaler

CLEAN = "cleaned_data.csv"
ENCODED = "encoded_data.csv"
KMEANS_FILE = "kmeans.pkl"
ENCODER_FILE = "encoder.pkl"

def load_data():
    df_clean = pd.read_csv(CLEAN)
    df_enc = pd.read_csv(ENCODED)
    return df_clean, df_enc

def train_kmeans(df_enc, n_clusters=12, random_state=42):
    # Drop orig_index before clustering
    X = df_enc.drop(columns=['orig_index']).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(Xs)
    # Save scaler + kmeans for later use (we can combine or save scaler separately)
    joblib.dump({'kmeans':kmeans, 'scaler':scaler}, KMEANS_FILE)
    print(f"KMeans trained and saved to {KMEANS_FILE}")
    return kmeans, scaler

def kmeans_recommend(query_vector, top_n=5):
    # query_vector: 1D array of same feature columns as encoded (no orig_index)
    meta = joblib.load(KMEANS_FILE)
    kmeans = meta['kmeans']
    scaler = meta['scaler']
    # transform
    q = scaler.transform(query_vector.reshape(1, -1))
    cluster = kmeans.predict(q)[0]
    # find items in same cluster
    df_enc = pd.read_csv(ENCODED)
    X = df_enc.drop(columns=['orig_index']).values
    Xs = scaler.transform(X)
    labels = kmeans.labels_
    idxs = np.where(labels == cluster)[0]
    # compute distance to centroid to rank
    centroid = kmeans.cluster_centers_[cluster]
    dists = np.linalg.norm(Xs[idxs] - centroid, axis=1)
    rank_order = idxs[np.argsort(dists)]
    # return top_n with orig_index
    out = df_enc.iloc[rank_order[:top_n]][['orig_index']].copy()
    return out['orig_index'].tolist()

def cosine_recommend(query_vector, top_n=5):
    df_enc = pd.read_csv(ENCODED)
    X = df_enc.drop(columns=['orig_index']).values
    sims = cosine_similarity(query_vector.reshape(1,-1), X).flatten()
    top_idx = np.argsort(-sims)[:top_n]
    return df_enc.iloc[top_idx]['orig_index'].tolist(), sims[top_idx].tolist()

# Example helper: build a query vector from user choices
def build_query_vector(user_pref, encoder):
    # user_pref: dict e.g. {'city':'Chennai','cuisine':['South Indian','Biryani'], 'rating':4.0, 'cost':250}
    # encoder is fitted OneHotEncoder (joblib.load('encoder.pkl'))
    cat_features = encoder.feature_names_in_.tolist()  # e.g., ['city','cuisine']
    # build categorical row with single selection per field (or multi-hot if multiple cuisines)
    # For simplicity, we'll construct a DataFrame with columns matching encoder.get_feature_names_out
    enc_cols = encoder.get_feature_names_out().tolist()
    cat_vec = np.zeros(len(enc_cols))
    # set city (single)
    for i, col in enumerate(enc_cols):
        parts = col.split('_', 1)
        if len(parts) < 2: continue
        prefix, val = parts[0], parts[1]
        if prefix in user_pref:
            # user_pref[prefix] can be single or list
            pref_val = user_pref[prefix]
            if isinstance(pref_val, list):
                if val in pref_val: cat_vec[i] = 1
            else:
                if val == str(pref_val): cat_vec[i] = 1

    # numeric features ordering must match encoded file order; read encoded_data columns
    df_enc = pd.read_csv(ENCODED)
    enc_columns = df_enc.drop(columns=['orig_index']).columns.tolist()
    # reconstruct full vector in same column order
    numeric_part = []
    for c in ['rating','rating_count','cost']:
        if c in enc_columns:
            numeric_part.append(user_pref.get(c, df_enc[c].median()))
    # numeric_part length let's assume 3 or less
    # Build final vector
    final = []
    # df_enc columns are like ['rating','rating_count','cost','city_...','cuisine_...']
    # find split index
    cat_start = len([c for c in enc_columns if c in ['rating','rating_count','cost']])
    for c in enc_columns:
        if c in ['rating','rating_count','cost']:
            final.append(user_pref.get(c, df_enc[c].median()))
        else:
            # find position in enc_cols
            try:
                idx = enc_cols.index(c)
                final.append(int(cat_vec[idx]))
            except ValueError:
                final.append(0)
    return np.array(final)

if __name__ == "__main__":
    # Example training run
    df_clean, df_enc = load_data()
    train_kmeans(df_enc, n_clusters=12)
