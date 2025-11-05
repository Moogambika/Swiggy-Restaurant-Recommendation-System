import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO

# =======================================
# Page Configuration
# =======================================
st.set_page_config(
    page_title="Swiggy Restaurant Recommendation System",
    page_icon="🍴",
    layout="wide",
)

# =======================================
# Custom Styling
# =======================================
st.markdown("""
<style>
.stApp { background-color: white !important; }
header[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
.logo-title-container { display:flex; align-items:center; gap:10px; margin-bottom:10px; margin-left:2px; }
.logo-title-container img { width:150px; height:auto; margin-left:30px; }
.logo-title-container h1 { margin:0; padding:0; white-space:nowrap; color:#FC8019 !important; margin-left:-40px; font-family:'Trebuchet MS', sans-serif; }
.welcome-text { margin-left:220px; margin-top:2px; margin-bottom:20px; }
.welcome-text h3 { font-size:25px; color:#FC8019 !important; font-family:'Trebuchet MS', sans-serif; }
.welcome-text p { margin:8px 0 0 120px; font-size:18px; color:#FC8019 !important; font-family:'Trebuchet MS', sans-serif; }
.main h3 { margin-top:20px !important; margin-bottom:8px !important; color:#262730 !important; }
.main p { margin-top:3px !important; margin-bottom:3px !important; color:#262730 !important; line-height:1.4; }
hr { border:1px solid #ddd; margin-top:15px !important; margin-bottom:15px !important; }
.main .block-container { color:#262730 !important; }
.main a { color:#1f77b4 !important; }
[data-testid="stImage"] { display:inline-block; }
</style>
""", unsafe_allow_html=True)

# =======================================
# File Paths
# =======================================
CLEAN = "cleaned_data.csv"
ENCODED = "encoded_data.csv"
ENCODER_FILE = "encoder.pkl"

# =======================================
# Load Data and Encoder
# =======================================
@st.cache_data
def load_files():
    df_clean = pd.read_csv(CLEAN)
    df_enc = pd.read_csv(ENCODED)
    encoder = joblib.load(ENCODER_FILE)
    return df_clean, df_enc, encoder

# =======================================
# Build Query Vector
# =======================================
def build_query_vector_from_inputs(enc, df_enc, city, cuisines, rating, cost):
    enc_cols = enc.get_feature_names_out().tolist()
    cat_vec = np.zeros(len(enc_cols))

    for i, col in enumerate(enc_cols):
        prefix, val = col.split('_', 1)
        if prefix == 'city' and val == city:
            cat_vec[i] = 1
        if prefix == 'cuisine' and val in cuisines:
            cat_vec[i] = 1

    enc_columns = df_enc.drop(columns=['orig_index'], errors='ignore').columns.tolist()
    final = []
    for c in enc_columns:
        if c in ['rating', 'rating_count', 'cost']:
            if c == 'rating':
                final.append(rating)
            elif c == 'rating_count':
                final.append(df_enc['rating_count'].median() if 'rating_count' in df_enc.columns else 0)
            else:
                final.append(cost)
        else:
            try:
                idx = enc_cols.index(c)
                final.append(int(cat_vec[idx]))
            except ValueError:
                final.append(0)
    return np.array(final)

# =======================================
# Weighted Cosine Similarity
# =======================================
def compute_weighted_similarity(qvec, df_enc):
    X_df = df_enc.drop(columns=['orig_index'], errors='ignore').fillna(0)
    weights = np.ones(X_df.shape[1])
    for i, col in enumerate(X_df.columns):
        if col.startswith("cuisine_"): weights[i] = 5.0
        elif col.startswith("city_"): weights[i] = 3.0
        elif col == "rating": weights[i] = 4.0
        elif col == "cost": weights[i] = 1.0
        elif col == "rating_count": weights[i] = 1.0
    X_weighted = X_df.values * weights
    qvec_weighted = qvec * weights
    sims = cosine_similarity(qvec_weighted.reshape(1, -1), X_weighted).flatten()
    return sims

# =======================================
# Display Results
# =======================================
def show_results(orig_indices, df_clean, scores=None):
    st.markdown('<h2 style="color: #262730;">🍽️ Recommended Restaurants</h2>', unsafe_allow_html=True)

    if len(orig_indices) == 0:
        st.markdown('<p style="color:#000000;">⚠️ No restaurants found with the selected criteria.</p>', unsafe_allow_html=True)
        return

    for i, idx in enumerate(orig_indices):
        row = df_clean.iloc[int(idx)]
        name = row.get("name", "")
        city = row.get("city", "")
        cuisine = row.get("cuisine", "")
        rating = row.get("rating", "")
        cost = row.get("cost", "")
        address = row.get("address", "")
        link = row.get("link", "")
        score = scores[i] if scores is not None else None

        st.markdown(f'<h3 style="color:#000000; font-weight:600;">{i+1}. {name} — {cuisine}</h3>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#000000;">🏙️ City: {city} | ⭐ Rating: {rating} | 💰 Cost: ₹{cost}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#000000;">📍 Address: {address}</p>', unsafe_allow_html=True)
        if link and str(link).lower() != 'nan':
            st.markdown(f'<p><a href="{link}" target="_blank" style="color:#1f77b4;">🔗 View on Swiggy / Menu</a></p>', unsafe_allow_html=True)
        if score is not None:
            st.markdown(f'<p style="color:#000000;">Similarity Score: {score:.3f}</p>', unsafe_allow_html=True)
        st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# =======================================
# Main App
# =======================================
def main():
    # Logo and Title
    try:
        logo = Image.open("swiggy_logo.png")
        buffered = BytesIO()
        logo.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(f"""
        <div class="logo-title-container">
            <img src="data:image/png;base64,{img_str}" alt="Swiggy Logo">
            <h1>Swiggy Restaurant Recommendation System</h1>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("<h1 style='color:#FC8019;'>🍴 Swiggy Restaurant Recommendation System</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-text">
        <h3>Welcome to the Swiggy Restaurant Recommendation System! 🍽️</h3>
        <p>Delivering happiness, one recommendation at a time. ❤️</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Load Data
    df_clean, df_enc, encoder = load_files()

    # Sidebar Inputs
    st.sidebar.header("🔍 Your Preferences")
    cities = sorted(df_clean['city'].dropna().unique().tolist())
    city = st.sidebar.selectbox("Select City", options=cities)

    raw_cuisines = {c.strip() for c in ','.join(df_clean['cuisine'].dropna().astype(str)).split(',')}
    all_cuisines = sorted([c for c in raw_cuisines if c and not any(x in c.upper() for x in ['AM','PM','TO','CLOSED'])])
    cuisines = st.sidebar.multiselect("Select Cuisines (1–3)", options=all_cuisines, default=all_cuisines[:1])

    rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
    cost = st.sidebar.slider("Approx Price for Two (₹)", int(df_clean['cost'].min()), int(df_clean['cost'].max()), int(df_clean['cost'].median()))
    top_n = st.sidebar.number_input("Number of Results", min_value=1, max_value=20, value=5)

    # Button Click
    if st.sidebar.button("🔎 Get Recommendations"):
        status = st.empty()
        status.markdown("<p style='color:#070738;font-weight:bold; font-size:18px;'>⏳ Finding restaurants...</p>", unsafe_allow_html=True)

        qvec = build_query_vector_from_inputs(encoder, df_enc, city, cuisines, rating, cost)

        # Filter by city
        city_mask = df_clean['city'].str.lower().str.strip() == city.lower().strip()
        df_city = df_clean[city_mask]
        df_enc_city = df_enc.loc[df_city.index]

        # Further filter by cuisine
        if cuisines:
            cuisine_mask = df_city['cuisine'].apply(lambda x: any(c.lower() in str(x).lower() for c in cuisines))
            df_city = df_city[cuisine_mask]
            df_enc_city = df_enc_city.loc[df_city.index]

        # If nothing found, fallback gracefully
        if df_city.empty:
            st.markdown(f'<p style="color:#000000; font-size:18px;">⚠️ No restaurants found in {city} for {", ".join(cuisines)}. Showing similar restaurants instead.</p>', unsafe_allow_html=True)
            df_city = df_clean[city_mask]
            df_enc_city = df_enc.loc[df_city.index]

        # Compute similarity
        sims = compute_weighted_similarity(qvec, df_enc_city)
        top_indices = df_city.index[np.argsort(sims)[::-1][:top_n]]
        top_scores = np.sort(sims)[::-1][:top_n]

        show_results(top_indices, df_clean, scores=top_scores)
        status.empty()

# =======================================
# Run App
# =======================================
if __name__ == "__main__":
    main()
