import speech_recognition as sr
from os import path

AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), str("prueba4.wav")) #Obtenemos el audio a convertir debe ser .wav
print(AUDIO_FILE,'Este es ')
r = sr.Recognizer()

with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # Leemos completamente el archivo de audio
# recognize speech usando Google Speech Recognition
try:         # Llamamos al metodo de reconocimiento por google y le pasamos el audio
    salida = "Google dice: " + r.recognize_google(audio) # Guardamos la salida en una variable
except sr.UnknownValueError: # Definimos excepciones que se puedan presentar
    salida = ("Google Speech Recognition no pudo entender el audio")
except sr.RequestError as e:
    salida = ("no se pueden usar los servicios de Google Speech Recognition; {0}".format(e))

print(salida)

WIT_AI_KEY = "COBIS4RA5JUJEIFUS3QTDWHSOY3L45YR"  # Wit.ai keys are 32-character uppercase alphanumeric strings


try:
            # Llamamos al metodo de reconocimiento por wit y le pasamos el audio, y la key
    salida2 =  ("Wit.ai dice: " + r.recognize_wit(audio, key=WIT_AI_KEY)) # Guardamos la salida en una variable
except sr.UnknownValueError: # Definimos excepciones que se puedan presentar
    salida2 = ("Wit.ai no pudo entender el audio")
except sr.RequestError as e:
    salida2 = ("no se pueden usar los servicios de Wit.ai; {0}".format(e))

print (salida2)



