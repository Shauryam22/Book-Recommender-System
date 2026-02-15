from flask import Flask, render_template, request
import pickle
import numpy as np
import random

# Load preprocessed data
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-S'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['Avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

def smart_recommend(book_name):
    """Improved recommend function that matches partial book names."""
    # Preprocess input
    book_name_words = [w.lower() for w in book_name.split()]

    # Split each title into words
    book_words = [title.split() for title in pt.index]

    # Count word matches for each title
    counts = []
    for title_words in book_words:
        count = sum(1 for w in title_words for k in book_name_words if w.lower() == k)
        counts.append(count)

    # Choose book with max overlap
    best_match_index = np.argmax(counts)
    best_match_title = pt.index[best_match_index]

    # Get top 5 similar items
    similar_items = sorted(
        list(enumerate(similarity_scores[best_match_index])),
        key=lambda x: x[1], reverse=True
    )[1:20]

    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        item = [
            temp_df['Book-Title'].values[0],
            temp_df['Book-Author'].values[0],
            temp_df['Image-URL-S'].values[0]
        ]
        data.append(item)

    return best_match_title, data


@app.route('/recommend_books', methods=['POST'])
def recommend_books():
    user_input = request.form.get('user_input')

    try:
        best_match, data = smart_recommend(user_input)
        return render_template('recommend.html', data=data, selected_book=best_match)
    except Exception as e:
        return render_template('recommend.html', error=str(e))


from flask import jsonify


@app.route('/suggest')
def suggest():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])

    # Get partial matches
    matches = [title for title in pt.index if query in title.lower()]

    # If no matches, try individual word overlap (like smart_recommend)
    if not matches:
        query_words = query.split()
        for title in pt.index:
            if any(word in title.lower() for word in query_words):
                matches.append(title)

    # Limit to top 8 suggestions
    matches = matches[:4]

    return jsonify(matches)

if __name__ == '__main__':
    app.run(debug=True)
