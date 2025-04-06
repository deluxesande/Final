# API_KEY="AAAAAAAAAAAAAAAAAAAAAFa30QEAAAAAV4q3wfKyToTgJdrxSU1gbVw94Bs%3DOw6cBVB2tHr5fUCkTR9PJOxgfKJwxBhHLu5bW2CTlA03MZO30W"
# API_KEY="AAAAAAAAAAAAAAAAAAAAADGh0QEAAAAAAnkLkU0ZyQSvAVQatuhbD5Wne4I%3DK58LTMvTbQUKb76c1WgjOJuqY4YpkUCl6qRkCmQTTRhVivF2U4"
API_KEY="AAAAAAAAAAAAAAAAAAAAAJi30QEAAAAAZyASb5pXFi%2BhTONf44ECutusDPo%3Dsri2yPoNzxiw4miNmy57l2wHPqT1JEeNvXZVK3eTsnOdlN6uNw"

import tweepy
import time

client = tweepy.Client(bearer_token=API_KEY)  # Free tier
# query = "python lang:en -is:retweet"
keywords = ["election", "president", "Kenya"]
query = f"({' OR '.join(keywords)}) lang:en -is:retweet"

# Method to store tweets in a file
def store_tweets(tweets, filename="fetched_tweets.txt"):
    with open(filename, "a", encoding="utf-8") as file:
        for tweet in tweets:
            file.write(tweet.text + "\n")

# Fetch tweets in batches with a delay
for _ in range(5):  # Example: Fetch 5 batches of tweets
    tweets = client.search_recent_tweets(query=query, max_results=10)
    if tweets.data:
        store_tweets(tweets.data)  # Store fetched tweets in a file
        for tweet in tweets.data:
            print(tweet.text)
    else:
        print("No tweets found.")
    time.sleep(5)  # Add a 5-second delay between requests