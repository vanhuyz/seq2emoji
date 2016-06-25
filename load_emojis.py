def load_emojis():
  with open('400_emoji_list.txt', 'r') as emoji_file:
    emojis= emoji_file.read().splitlines()
  return emojis