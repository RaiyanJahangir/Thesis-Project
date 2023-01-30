# Import text to speech conversion library
from gtts import gTTS
# Text2Speech generation
tts = gTTS('প্র্যাকটিস মুড অন হয়েছে', lang='bn')
# Save converted audio as mp3 format
tts.save('voice_folder/practice_ready.mp3')