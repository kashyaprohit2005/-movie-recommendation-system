import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def train_and_save():
    print("1. Loading dataset...")
    file_paths = ['datasets/reviews.txt', 'reviews.txt']
    target_path = None
    for path in file_paths:
        if os.path.exists(path):
            target_path = path
            break

    reviews_list, status_list = [], []

    if target_path:
        with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '\t' in line:
                    parts = line.split('\t', 1)
                    # Format: label<TAB>review text
                    status_list.append(parts[0])
                    reviews_list.append(parts[1] if len(parts) > 1 else parts[0])
                elif ',' in line:
                    parts = line.rsplit(',', 1)
                    reviews_list.append(parts[0])
                    status_list.append(parts[1] if len(parts) > 1 else 'good')
                else:
                    reviews_list.append(line)
                    status_list.append('good')

    fallback_reviews = [
        ("This movie was fantastic and brilliant, loved the acting!", 1),
        ("Amazing plot, wonderful cinematography, truly a masterpiece.", 1),
        ("Great direction, super cast and memorable soundtrack.", 1),
        ("Terrible movie, absolute waste of time and boring script.", 0),
        ("Horrible acting, predictable story and worst direction ever.", 0),
        ("Disappointing and dull film, do not recommend watching.", 0),
        ("Loved every minute of it, highly recommended!", 1),
        ("Poor visual effects, bad dialogue and very slow pacing.", 0),
        ("Outstanding performance by the entire cast, exciting thriller.", 1),
        ("Garbage film, completely meaningless and pathetic plot.", 0),
    ]

    for text, label in fallback_reviews:
        reviews_list.append(text)
        status_list.append(str(label))

    status_map = {'1': 1, '0': 0, 'positive': 1, 'negative': 0, 'good': 1, 'bad': 0, 1: 1, 0: 0}
    y_clean = [status_map.get(str(s).lower().strip(), 1) for s in status_list]

    print(f"Total review entries processed: {len(reviews_list)}")
    print("2. Vectorizing text data...")
    cv = CountVectorizer(max_features=5000, lowercase=True)
    X_vector = cv.fit_transform(reviews_list)

    print("3. Training MultinomialNB model...")
    clf = MultinomialNB()
    clf.fit(X_vector, y_clean)

    print("4. Saving pickle files...")
    with open('tranform.pkl', 'wb') as f:
        pickle.dump(cv, f)
    with open('nlp_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("Done! 'nlp_model.pkl' and 'tranform.pkl' updated.")
    return clf, cv


if __name__ == '__main__':
    train_and_save()
