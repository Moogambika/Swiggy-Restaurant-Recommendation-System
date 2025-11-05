🍴 Swiggy Restaurant Recommendation System

A machine learning-based restaurant recommendation system that helps users discover restaurants similar to their preferences based on city, cuisine, rating and cost — built using Python, Pandas, Scikit-learn and Streamlit.

🚀 Features

Cleans and preprocesses Swiggy restaurant data.

Encodes categorical features using OneHotEncoder.

Uses Cosine Similarity(Unsupervised Learning) to find similar restaurants.

Beautiful Streamlit web app with Swiggy-style UI.

Includes EDA (Data Visualization) for insights like top cuisines, average costs, and rating trends.

🧠 Tech Stack

Python 3.13

Pandas, NumPy, Scikit-learn

Matplotlib, Seaborn

Streamlit

Joblib

📂 Project Files

File	Description

data_cleaning.ipynb	Cleans raw dataset (handles missing values, formats, duplicates).

preprocess_and_encode.ipynb	Encodes city & cuisine features using OneHotEncoder.

eda.ipynb	Visual data analysis (top cuisines, cost trends, rating distribution).

recommendation.py	Core cosine similarity logic for generating recommendations.

app.py	Streamlit web app with UI for user interaction and result display.

cleaned_data.csv	Cleaned restaurant dataset.

encoded_data.csv	Encoded dataset for model input.

encoder.pkl	Saved encoder for transforming new inputs.


💻 How It Works

User selects city, cuisine, minimum rating, and approximate cost.

The app encodes these inputs and compares them to all restaurants using cosine similarity.

The most similar restaurants are shown with names, ratings, costs, and links.

📊 Example EDA Insights

🍕 Top 10 Cuisines — most popular food types.

💸 Average Cost per City — affordability comparison.

⭐ Rating Distribution — quality overview.

▶️ Run the App
pip install -r requirements.txt
streamlit run app.py


Then open: http://localhost:8501

🌟 Future Enhancements

Use TF-IDF / BERT for text-based restaurant features.

Add sentiment analysis on reviews.

Deploy on Streamlit Cloud or AWS EC2.

👩‍💻 Author 
Moogambika Govindaraj
Data Science Enthusiast

Acknowledgements:
I sincerely thank my internal mentors and my external mentor for guiding me throughout this project. Your insights, encouragement and support have been invaluable in helping me understand and successfully complete this work. I am especially grateful to the GUVI internal mentors for giving me this opportunity and for their continuous guidance.

Moogambika Govindaraj
AI & ML Developer Intern
