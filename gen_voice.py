# Import text to speech conversion library
from gtts import gTTS
# Text2Speech generation
a='40% সঠিক হয়েছে'
tts = gTTS(a, lang='bn')
# Save converted audio as mp3 format
tts.save('voice_folder/test_result.mp3')