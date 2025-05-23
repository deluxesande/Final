from time import time
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import datetime
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import altair as alt
import base64
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import tweepy
from textblob import TextBlob  # For sentiment analysis
import re

# Read credentials from db
def load_users_and_passwords(filenames=["auth/users.txt", "auth/password.txt"]):
    users = []
    passwords = []

    with open(filenames[0], "r", encoding="utf-8") as file:
        users.extend([line.strip() for line in file.readlines()])

    with open(filenames[1], "r", encoding="utf-8") as file:
        passwords.extend([line.strip() for line in file.readlines()])

    return users, passwords

# Define user credentials
# names = ["Lucy"]
# usernames = ["Lucy"]
# passwords = ["password1", "password2"]  # Replace with your actual passwords

# Define credentials stored in db
usernames, passwords = load_users_and_passwords()
names = usernames

# Hash the passwords
# hashed_passwords = stauth.Hasher(passwords).generate()
# hashed_passwords = stauth.Hasher().hash(passwords)

# This one worked
hashed_passwords = [stauth.Hasher().hash(password) for password in passwords]

# Create the credentials dictionary
# this only adds the first user
# credentials = {
#     "usernames": {
#         usernames[0]: {
#             "name": names[0],
#             "password": hashed_passwords[0]
#         }
#     }
# }

# Adding all credentials loaded
credentials = {
    "usernames": {}
}

for key, username in enumerate(usernames):
    new_user = {username: {"name": names[key],"password": hashed_passwords[key]}}
    credentials["usernames"].update(new_user)

# Save the credentials to a pickle file
file_path = "hashed_pw.pkl"
with open(file_path, "wb") as file:
    pickle.dump(credentials, file)

print(f"Credentials saved to {file_path}")

# Hide menu and footer
hide_menu = """
<style>
#MainMenu {
    visibility:visible;
}
footer {
    visibility:visible;
}
footer:after{
    content:'Copyright © 2022: Indispensables';
    display:block;
    position:relative;
    color:red;
    padding:5px;
    top:3px;
}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# Function to set background
@st.cache_data  # Updated from st.cache
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            border-radius: 10px;
            ; 
            z-index: -1; 
        }}

        .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5); /* Black overlay with 50% opacity */
        z-index: -1;
    }}
        </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Load dataset
d1 = pd.read_csv('df1.csv', low_memory=False)  # Added low_memory=False to handle DtypeWarning
d2 = pd.read_csv('df2.csv', low_memory=False)
d3 = pd.read_csv('df3.csv', low_memory=False)
d4 = pd.read_csv('df4.csv', low_memory=False)
d5 = pd.read_csv('df5.csv', low_memory=False)
df = pd.concat([d1, d2, d3, d4, d5], axis=0, ignore_index=True)
df.dropna(subset=['tweet_clean'], inplace=True)
df['time'] = pd.to_datetime(df['time']).dt.normalize()

# Read tweets from a text file
def load_tweets_from_file(filenames=["fetched_tweets_1.txt", "fetched_tweets_2.txt"]):
    tweets = []
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as file:
            tweets.extend([line.strip() for line in file.readlines()])
    return tweets

# Define current date
current_date = datetime.datetime.now().date()

# Set background
set_bg("background.png")

# Sidebar navigation
with st.sidebar:
    navigation = option_menu(None, ["Home", "Politics Today", "Presidential Election Prediction", "Register"],
                             icons=['house-fill', "book-half", "check-circle-fill", "person-plus-fill"], default_index=1)
# Initialize navigation state
if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Home"
# Home Section
if navigation == "Home":
    st.markdown("<h2 style='text-align: center; color: white;'>The Indispensables Election Analysis</h2>", unsafe_allow_html=True)
    st.markdown("*****************")
    st.subheader("About Elections")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        Elections are held in Kenya every fifth year on the second Tuesday in August. The election cycle is upon us,
        and at Indispensables, we aim to keep you up to date with the latest trends, changing popularities of political 
        coalitions, and political figures as we head toward the election.
        """)
        st.markdown("""
                    <h4 style='font-weight: bold; color: white;  border-radius: 5px;'>
                    """, unsafe_allow_html=True)
    with col2:
        st.image("b.png")

# Politics Today Section
if navigation == "Politics Today":
    st.markdown("""
    <h4 style='font-weight: bold; color: black; background-color: #f0f0f0;  border-radius: 5px;'>
    Elections are approaching. Here, we seek to show you trending topics, the changing popularities
    of political parties and politicians.
    </h4>
    """, unsafe_allow_html=True)
    navigate2 = option_menu(
        menu_title=None,
        options=["Trending Topics", "Political Parties", "Political Figures"],
        icons=["activity", "flag-fill", "people-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#fff", },
            "nav-link": {"font-size": "12px", "text-align": "left", "margin": "5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "red"},
        },
    )

    # Trending Topics
    if navigate2 == "Trending Topics":
        st.markdown("""
        <h3 style='font-weight: bold; color: black; background-color: #f0f0f0;  border-radius: 5px;'>
        Dates of Interest
        </h3>
        """, unsafe_allow_html=True)
        start = st.date_input(label='Start: ', value=current_date, key='#start',
                              help="The start date time", on_change=lambda: None)
        end = st.date_input(label='End: ', value=current_date + datetime.timedelta(days=30), key='#end',
                            help="The end date time", on_change=lambda: None)

        if start >= current_date and start < end:
            st.success(f'Start date: `{start}`\nEnd date: `{end}`')
            st.markdown(f"""
            <div style='background-color:"light gray"; padding: 10px; border-radius: 5px;'>
                <strong style='border-color:"white"; padding: 2px 5px; border-radius: 3px;'>Start date:</strong> `{start}`<br>
                <strong style='border-color: white; padding: 2px 5px; border-radius: 3px;'>End date:</strong> `{end}`
            </div>
            """, unsafe_allow_html=True)

            # Fetch live tweets using Twitter API
            def fetch_live_tweets(start_date, end_date, keywords):
                client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAADGh0QEAAAAAAnkLkU0ZyQSvAVQatuhbD5Wne4I%3DK58LTMvTbQUKb76c1WgjOJuqY4YpkUCl6qRkCmQTTRhVivF2U4');           query = f"({' OR '.join(keywords)}) lang:en -is:retweet"
                tweets = []
                try:
                    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, max_results=100).flatten(limit=1000):
                        tweets.append(tweet.text)
                except Exception as e:
                    st.error(f"Error fetching tweets: {e}")
                return tweets

            keywords = ["election", "president", "Kenya"]
            # live_tweets = fetch_live_tweets(start, end, keywords)

            # Replace live_tweets with tweets loaded from the file
            live_tweets = load_tweets_from_file()

            live_df = pd.DataFrame(live_tweets, columns=['tweet_clean'])

            # Preprocess the tweets
            def preprocess_tweets(tweets):
                processed_tweets = []
                for tweet in tweets:
                    # Remove URLs
                    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
                    # Remove special characters and numbers
                    tweet = re.sub(r'\W+', ' ', tweet)
                    # Convert to lowercase
                    tweet = tweet.lower()
                    processed_tweets.append(tweet)
                return processed_tweets

            live_df['tweet_clean'] = preprocess_tweets(live_df['tweet_clean'])

            # Check if tweet_clean is empty
            if live_df['tweet_clean'].empty:
                st.error("No tweets were fetched or all tweets are empty. Please try again with different keywords or date ranges.")
            else:
                try:
                    # Apply TF-IDF vectorization
                    tf = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', lowercase=False)
                    tfidf_matrix = tf.fit_transform(live_df['tweet_clean'])
                    total_words = tfidf_matrix.sum(axis=0)
                    freq = [(word, total_words[0, idx]) for word, idx in tf.vocabulary_.items()]
                    freq = sorted(freq, key=lambda x: x[1], reverse=True)
                    bigram = pd.DataFrame(freq, columns=['bigram', 'count'])
                    bigram['count'] = ((bigram['count'] / bigram['count'].sum()) * 100).round(2)

                    # Plot trending topics
                    source = bigram.head(20)
                    bar_chart = alt.Chart(source).mark_bar().encode(
                        y='count',
                        x='bigram',
                        color='bigram'
                    )
                    text = bar_chart.mark_text(align='left', baseline='middle', dx=3).encode(text='count')
                    plt = (bar_chart + text).properties(height=600)
                    st.altair_chart(plt, use_container_width=True)
                except ValueError as e:
                    st.error(f"Error processing tweets: {e}")

        else:
            st.error('Error: End date must fall after start date.')

# Presidential Election Prediction Section
if navigation == "Presidential Election Prediction":
    file_path = Path(__file__).parent / "hashed_pw.pkl"
    if file_path.exists():
        try:
            with file_path.open("rb") as file:
                credentials = pickle.load(file)
        except (pickle.UnpicklingError, EOFError):
            st.error("Error: The credentials file is corrupted. Please recreate the file.")
            credentials = {}
    else:
        st.error("Error: Credentials file not found. Please recreate it.")
        credentials = {}

    authenticator = stauth.Authenticate(credentials, "Elections_Predictor", "abcdef", cookie_expiry_days=30)
    # name, authentication_status, username = authenticator.login(label="Login Section", location="main")
    # name, authentication_status, username = authenticator.login("Login Section", location="main")

    result = authenticator.login()

    name, authentication_status, username = ["", False, ""]  # Default values

    if st.session_state.get('authentication_status'):
        authenticator.logout("Logout", "main")
        st.title(f"Welcome *{st.session_state.get("name")}*")

        st.write("Predict the presidential aspirant most likely to win the forthcoming elections.")

        start = st.date_input(label='Start: ', value=current_date, key='#start', help="The start date time")
        end = st.date_input(label='End: ', value=current_date + datetime.timedelta(days=30), key='#end', help="The end date time")

        if start <= current_date and start < end:
            st.success(f'Start date: `{start}`\nEnd date: `{end}`')

            # Fetch live tweets
            keywords = ["Elections", "President", "Kenya"]
            # live_tweets = fetch_live_tweets(start, end, keywords)
            live_tweets = load_tweets_from_file()

            # Analyze sentiment
            def analyze_sentiment(tweets):
                sentiments = []
                for tweet in tweets:
                    analysis = TextBlob(tweet)
                    polarity = analysis.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)
                    sentiments.append(polarity)
                return sentiments

            sentiments = analyze_sentiment(live_tweets)
            live_df = pd.DataFrame({'tweet_clean': live_tweets, 'sentiment': sentiments})

            # Clean the data: Remove rows with zero or invalid sentiment values
            live_df = live_df[live_df['sentiment'] != 0]  # Remove rows where 'sentiment' is 0
            live_df = live_df.dropna()  # Remove rows with missing values (if any)

            # Display sentiment analysis results
            st.write("Sentiment Analysis Results:")
            st.write(live_df)

            # Predict using a trained model
            from sklearn.ensemble import RandomForestClassifier

            X = df[['Polarity']]  # Replace 'other_features' with relevant features
            y = df['Expressions']  # Target variable

            if len(X) == 0 or len(y) == 0:
                st.error("No data available for training!")

            model = RandomForestClassifier()
            try:
                model.fit(X, y)
            except Exception as e:
                st.error(f"Error training model: {e}")
                predictions = []

            # Clean the data: Remove rows with zero or invalid sentiment values
            live_df = live_df[live_df['sentiment'] != 0]  # Remove rows where 'sentiment' is 0
            live_df = live_df.dropna()  # Remove rows with missing values (if any)


            live_X = live_df[['sentiment']].rename(columns={'sentiment': 'Polarity'})  # Rename 'sentiment' to 'Polarity'
            predictions = model.predict(live_X)

            # Display predictions
            st.write("Predicted Election Outcomes:")
            st.write(predictions)

            # Predict the possible winner
            if len(predictions) > 0:
                # Count the frequency of each predicted outcome
                prediction_counts = pd.Series(predictions).value_counts()

                st.write(prediction_counts)

                # Determine the most frequent outcome
                possible_winner = prediction_counts.idxmax()
                st.write("Possible Winner Based on Predictions:")
                st.write(f"The predicted winner is: **{possible_winner}** with {prediction_counts[possible_winner]} mentions.")

        else:
            st.error('Error: End date must fall after start date.')

    elif st.session_state.get('authentication_status') is False:
        st.error('Username/password is incorrect')
    elif st.session_state.get('authentication_status') is None:
        st.warning('Please enter your username and password')

# Register Section
if navigation == "Register":
    st.subheader("Register a New User")
    
    # Input fields for username and password
    new_username = st.text_input("Enter a username")
    new_password = st.text_input("Enter a password", type="password")
    
    if st.button("Register"):
        if new_username and new_password:
            # Save the username and password to a text file
            with open("auth/users.txt", "a") as file:
                file.write(f"\n{new_username}")
            with open("auth/password.txt", "a") as file:
                file.write(f"\n{new_password}")
            st.success("User registered successfully!")
        else:
            st.error("Please fill in both username and password.")