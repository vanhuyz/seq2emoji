from mongoengine import * 

connect('seq2emoji')

class Tweet(Document):
  content = StringField(required=True)