import re

from tweet import Tweet
from sequence import Sequence
from load_emojis import load_emojis

emojis = load_emojis()

def strip_tweet(content):  
  # remove URLs, usernames, hashtags (http://stackoverflow.com/a/13896637/4314737)
  content =re.sub(r"(?:\@|#|https?\://)\S+", "", content)
  # remove whitespaces
  return re.sub(r"\s+", "", content)

for tweet in Tweet.objects[:50000]:
  lines = tweet.content.splitlines()
  for line in lines:
    line = strip_tweet(line)
    # only save lines which have length > 2 and end with a emoji
    if len(line) > 2 and line[-1:] in emojis:
      print(line)
      Sequence.objects(content=line).update_one(content=line, upsert=True)

  print("-----------------")