from mongoengine import * 

connect('seq2emoji')

class Sequence(Document):
  content = StringField(required=True, unique=True)