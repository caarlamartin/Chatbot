# Bibliotecas que vamos a usar
import random   # Para operaciones aleatorias
import json     # Para manejar el archivo intents.json
import pickle       # Para guardar y cargar objetos como vocabularios y etiquetas
import numpy as np  # Para trabajar con arreglos y operaciones matemáticas

# Librerías para procesamiento del lenguaje natural (PLN)
import nltk
from nltk.stem import WordNetLemmatizer #Para pasar las palabras a su forma raíz

# Librerías para crear la red neuronal
from keras.models import Sequential     # Para construir un modelo de red neuronal secuecial
from keras.layers import Dense, Activation, Dropout     # Capas de la red neuronal
from keras.optimizers import SGD        # Optimizador para ajustar los pesos del modelo

# Inicializamos el lematizador para procesar las palabras
lemmatizer = WordNetLemmatizer()

# Importa el archivo intents.json creado previamente que contiene patrones, etiquetas y respuestas
intents = json.loads(open('intents.json').read())

# Descargamos recursos de NLTK necesarios para tokenización y lematización
nltk.download('punkt')  # Tokenizador de oraciones
nltk.download('wordnet')    # Diccionario para lematización
nltk.download('omw-1.4')    # Complemento para WordNet

# Inicializamos listas para almacenar palabras únicas, categorías y documentos etiquetados
words = []  # Lista de palabras únicas extraídas de los patrones
classes = []    # Listas de categorías o etiquetas (tags)
documents = []  # Paras de (lista de palabras del patrón, etiqueta asociada)
ignore_letters = ['?', '!', '¿', '.', ',']  # Caracteres que queremos ignorar

# Procesamos los datos del archivo intents.json
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizamos cada patrón en una lista de palabras
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list) # Añadimos las palabras a la lista global
        documents.append((word_list, intent["tag"]))    #Asociamos palabras con su categoría
        if intent["tag"] not in classes:
            classes.append(intent["tag"])   # Añade etiquetas únicas

# Lematizamos y eliminamos duplicados de la lista de palabras
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))  # Eliminamos duplicados y ordenados

# Guardamos las palabras y las categorías en archivos para uso posterior
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Pasa la información a unos y ceros según las palabras presentes en cada categoría para hacer el entrenamiento
training = []   # Lista que contendrá las características y etiquetas
output_empty = [0]*len(classes) # Lista vacía para las etiquetas
for document in documents:
    bag = []    # vector binario para el patrón actual
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    # Creamos una vector de salida para la etiqueta correspondiente
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1  # Marcamos la etiqueta como 1
    training.append([bag, output_row])  # Añadimos las características y etiquetas al conjunto de entrenamiento
# Mezclamos los datos de entrenamiento para evitar sesgos
random.shuffle(training)
print(len(training)) 
# Extraemos características, x, y etiquetas, y, de los datos de entrenamiento
train_x=[]  
train_y=[]
for i in training:
    train_x.append(i[0])
    train_y.append(i[1])
# Convertimos en arrays la clasificación
train_x = np.array(train_x) 
train_y = np.array(train_y)

#Creamos una red neuronal multicapa para la clasificación
model = Sequential()
# Capa de entrada con 18 neuronas y activación ReLu
model.add(Dense(128, input_shape=(len(train_x[0]),), name="inp_layer", activation='relu'))
# Primera capa de Dropout para prevenir sobreajustes
model.add(Dropout(0.5, name="hidden_layer1"))
# Segunda capa densa con 64 neuronas y activación ReLu
model.add(Dense(64, name="hidden_layer2", activation='relu'))
# Segunda capa de Dropout
model.add(Dropout(0.5, name="hidden_layer3"))
# Capa de salida con tantas neuronas como categorías y activación Softmax para clasificación multiclase
model.add(Dense(len(train_y[0]), name="output_layer", activation='softmax'))

#Creamos el optimizador y lo compilamos
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#Entrenamos el modelo y lo guardamos
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save("chatbot_model.h5")
