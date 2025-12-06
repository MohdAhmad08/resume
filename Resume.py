import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample training data (keywords used for learning)
data = [
    "Python Machine Learning SQL Tableau Pandas NumPy Excel",
    "Java HTML CSS JavaScript WordPress PHP",
    "Data Science Power BI SQL Python ML Analytics Excel",
]

# Labels --> 1 means matching profile, 0 means not matching
labels = [1, 0, 1]

# Train TF-IDF model
vectorizer = TfidfVectorizer()
vectorizer.fit(data)

# Save model
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("vectorizer.pkl saved")
