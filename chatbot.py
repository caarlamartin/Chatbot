# Bibliotecas que vamos a usar
import random   # Para operaciones aleatorias
import json     # Para manejar el archivo intents.json
import pickle       # Para guardar y cargar objetos como vocabularios y etiquetas
import numpy as np  # Para trabajar con arreglos y operaciones matemáticas

# Librerías para procesamiento del lenguaje natural (PLN)
import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

# Inicializamos el lematizador para procesar las palabras
lemmatizer = WordNetLemmatizer()

# Importamos los archivos generados en el código de entrenamiento
# `intents.json`: Contiene patrones, respuestas y etiquetas
# `words.pkl`: Contiene el vocabulario procesado
# `classes.pkl`: Contiene las categorías/etiquetas que el modelo ha aprendido
# `chatbot_model.h5`: Es el modelo de red neuronal entrenado
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Función para preprocesar una oración
# Toma una oración como entrada, la tokeniza en palabras individuales y las lematiza
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Función para crear una bolsa de palabras (bag of words)
# Convierte las palabras de la oración en un vector binario de unos y ceros, según si están presentes en el vocabulario
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)    # Preprocesamos la oración
    bag = [0]*len(words)    # Creamos un vector de ceros del tamaño del vocabulario
    for w in sentence_words:    # Iteramos sobre cada palabra de la oración
        for i, word in enumerate(words):    # Comparamos con las palabras del vocabulario
            if word == w:   # Si la palabra está en el vocabulario, marcamos su posición con un 1
                bag[i]=1
    print(bag)
    return np.array(bag)

# Función para predecir la clase de una oración
# Usa el modelo entrenado para predecir la categoría de la oración de entrada
def predict_class(sentence):
    bow = bag_of_words(sentence)    # Convertimos la oración en una bolsa de palabras
    res = model.predict(np.array([bow]))[0] # Realizamos la predicción con el modelo
    max_index = np.where(res ==np.max(res))[0][0]   # Buscamos la categoría con la mayor probabilidad
    category = classes[max_index]   # Asignamos la categoría correspondiente
    return category

# Función para obtener una respuesta del chatbot
# Basada en la etiqueta predicha, selecciona una respuesta aleatoria de intents.json
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']   # Obtenemos todas las intenciones del archivo JSON
    result = ""
    for i in list_of_intents:   # Buscamos la intención correspondiente al tag predicho
        if i["tag"]==tag:
            result = random.choice(i['responses'])  # Seleccionamos una respuesta aleatoria de las disponibles
            break
    return result

# Función principal para generar una respuesta
# Recibe un mensaje del usuario, predice su clase y genera la respuesta correspondiente
def respuesta(message):
    ints = predict_class(message)   # Predice la intención de la entrada del usuario
    res = get_response(ints, intents)   # Obtiene una respuesta basada en la intención predicha
    return res

# Bucle infinito para interactuar con el chatbot
# El programa sigue ejecutándose hasta que se detenga manualmente
while True:
    message = input()   # Entrda del usuario
    print(respuesta(message))   # Imprime la respuesta generada
