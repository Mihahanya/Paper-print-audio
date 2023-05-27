
import os

os.system('python create_layout.py')
print('---')
os.system('python audio_to_picture.py')
print('---')
os.system('python pic_to_audio.py')

print('---\nfinally')

