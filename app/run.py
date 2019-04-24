import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# Define function tokenize to normalize, tokenize and lemmatize text string
def tokenize(text):
    """Normalize, tokenize and lemmatize text string
    
    Args:
    text: string, String containing message for processing
       
    Returns:
    clean_tokens: list, List containing normalized and lemmatized word tokens
    """

    # Replace URL links in text string with string 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Substitute characters in text string which match regular expression r'[^a-zA-Z0-9]'
    # with single whitespace
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # Get word tokens from text string
    tokens = word_tokenize(text)
    
    # Instantiate WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Get stop words in 'English' language
    stop_words = stopwords.words("english")

    # Clean tokens
    clean_tokens = []
    for tok in tokens:
        # convert token to lowercase as stop words are in lowercase
        tok_low = tok.lower() 
        if tok_low not in stop_words:
            # Lemmatize token and remove the leading and trailing spaces from lemmatized token
            clean_tok = lemmatizer.lemmatize(tok_low).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


# Add custom Estimator
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if pos_tags:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return float(True)
        return float(False)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# Define function to calculate the multi-label f-score
def multi_label_fscore(y_true, y_pred, beta=1):
    """Calculate individual weighted average fbeta score of each category and
    geometric mean of weighted average fbeta score of each category
    
    Args:
    y_true: dataframe, dataframe containing true labels i.e. Y_test
    y_pred: ndarray, ndarray containing predicted labels i.e. Y_pred
    beta: numeric, beta value
       
    Returns:
    f_score_gmean: float, geometric mean of fbeta score for each category
    """
    
    b = beta
    f_score_dict = {}
    score_list = []
    
    # Create dataframe y_pred_df from ndarray y_pred 
    y_pred_df = pd.DataFrame(y_pred, columns=y_true.columns)

    for column in y_true.columns:
        score = round(fbeta_score(y_true[column], y_pred_df[column], beta, average='weighted'),4)
        score_list.append(score)
    f_score_dict['category'] = y_true.columns.tolist()
    f_score_dict['f_score'] = score_list

   
    f_score_df = pd.DataFrame.from_dict(f_score_dict)

    f_score_gmean = gmean(f_score_df['f_score'])

    return f_score_gmean


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = [df[(df['genre_news']== 1) & (df['genre_social']== 0)]['message'].count(),
                    df[(df['genre_news']== 0) & (df['genre_social']== 0)]['message'].count(),
                    df[(df['genre_news']== 0) & (df['genre_social']== 0)]['message'].count()]
    # Assigning genre names directly as genre feature was encoded as dummy variable in ETL pipeline
    # genre 'direct' count is retrieved by df[(df['genre_news']== 0) & (df['genre_social']== 0)]['message'].count()
    genre_names = [news, social, direct]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()