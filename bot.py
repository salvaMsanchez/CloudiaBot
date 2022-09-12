#--------------------IMPORTACIONES--------------------

# para manejar la API de Telegram
import telebot
from telebot.types import ReplyKeyboardMarkup # crear botones
from telebot.types import ForceReply # citar un mensaje
from telebot.types import ReplyKeyboardRemove # eliminar botonera
from telebot.types import InlineKeyboardMarkup # crear botonera inline
from telebot.types import InlineKeyboardButton # definir botones inline

# NLP
import nltk
nltk.download('punkt')

# Derivación regresiva para textos en español
from nltk.stem import SnowballStemmer
spanish_stemmer = SnowballStemmer('spanish')

# Tensorflow
import numpy as np
import tensorflow as tf
import tflearn
import random as rd

# Web Scraping
from bs4 import BeautifulSoup
import requests

# Gestión de archivos y sistema
import pickle
import os
from os.path import exists
import json
import csv
import time

# Creación de gráficos
from matplotlib import pyplot as plt

# LATEX
from pylatex import Document, PageStyle, Head, MiniPage, Section, Subsection, Subsubsection, Tabular, Math, TikZ, Axis, \
    StandAloneGraphic, Plot, Figure, Matrix, Alignat, MultiRow, MultiColumn, Hyperref, Package, SubFigure, \
    LargeText, LineBreak, MediumText, NewPage, Tabu, Itemize, Command
from pylatex.utils import escape_latex, NoEscape, italic, bold
import datetime

# Variables
from config import TOKEN

# Funciones
from helper import normalize

#!echo "Done!"

#--------------------FUNCIONES GENERALES--------------------

def clean_up_sentence(sentence):
  # tokenización
  sentence_words = nltk.word_tokenize(sentence, "spanish")
  sentence_words = [word.lower() for word in sentence_words if word.isalpha()] # Eliminar los signos de puntuación
  # derivar regresivamente cada palabra 
  sentence_words = [spanish_stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words

# devuelve el array de palabras de la bolsa: 0 o 1 para cada palabra de la bolsa que existe en la frase --> Función para la elaboración de la bolsa de palabras
def bow(sentence, words, show_details=False):
  # tokenización
  sentence_words = clean_up_sentence(sentence)
  # bolsa de palabras
  bag = [0]*len(words)  
  for s in sentence_words:
      for i,w in enumerate(words):
          if w == s: 
              bag[i] = 1
              if show_details:
                  print("found in bag: %s" % w)

  return(np.array(bag))

#ERROR_THRESHOLD = 0.25 # comentar esta línea y la del if ERROR para que no se criben las predicciones y modificamos los corchetes de sitio
# función de clasificación a través del modelo
def classify(sentence):
  # generar probabilidades a partir del modelo
  results = model.predict([bow(sentence, words)])[0]
  # filter out predictions below a threshold
  results = [[i,r] for i,r in enumerate(results)] #if r>ERROR_THRESHOLD]
  # ordenar de mayor a menor según la probabilidad
  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in results:
      return_list.append((etiquetas[r[0]], r[1]))
  # devuelve una tupla de intención y probabilidad
  return return_list

def func(pct, allvalues): 
  absolute = int(pct / 100. * np.sum(allvalues)) 
  return "{:.1f}%".format(pct, absolute)

def crear_imagen(filename, grupos, len_grupos, porcentajes_limpios):
  # ajustar tamaño del gráfico circular según la entrada de grupos de grados para que no cambie mucho el tamaño en la salida
  if len_grupos <= 5:
    tamano = 3
  else:
    tamano = 15

  wp = { 'linewidth' : 1, 'edgecolor' : "green" } 

  fig, ax = plt.subplots(figsize = (15, tamano)) 
  wedges, texts, autotexts = ax.pie(porcentajes_limpios,  
                                  autopct = lambda pct: func(pct, porcentajes_limpios), 
                                  labels = grupos, 
                                  shadow = False, 
                                  startangle = 90, 
                                  wedgeprops = wp, 
                                  textprops = dict(color ="black"))  

  plt.setp(autotexts, size = 8, weight ="bold") 
  ax.set_title("RECOMENDACIÓN GRADOS")

  plt.savefig(filename, dpi=300, bbox_inches='tight')

def hyperlink(url,text):
  text = escape_latex(text)
  return NoEscape(r'\href{' + url + '}{' + text + '}')

def latex(img_filename_mayor, grados_reales_menor, enlaces_reales_menor, grados_reales_mayor, enlaces_reales_mayor, porcentajes_mayor, ciencias, letras, nombre):
  geometry_options = {
    "head": "35pt",
    "margin": "0.5in",
    "bottom": "0.6in",
    "includeheadfoot": True
  }
  doc = Document(geometry_options=geometry_options)
  doc.packages.append(Package('hyperref'))

  # Generar el estilo de la primera página
  first_page = PageStyle("firstpage")

  # Imagen de cabecera
  with first_page.create(Head("L")) as header_left:
    with header_left.create(MiniPage(width=NoEscape(r"0.49\textwidth"), pos='c')) as logo_wrapper:
      logo_file = './Cloudia_Bot/images/imagotipo.png'
      logo_wrapper.append(StandAloneGraphic(image_options="width=70px", filename=logo_file))

  # Añadir estilo del documento
  with first_page.create(Head("R")) as right_header:
    with right_header.create(MiniPage(width=NoEscape(r"0.49\textwidth"), pos='c', align='r')) as satle_wrapper:
      satle_wrapper.append(LargeText(bold("CloudiaBot")))
      satle_wrapper.append(LineBreak())
      satle_wrapper.append(MediumText(bold("Predicción y recomendación de grados universitarios")))
      satle_wrapper.append(LineBreak())

  doc.preamble.append(first_page)
  doc.change_document_style("firstpage")
  # Fin del estilo de la primera página

  # Añadir información
  with doc.create(Tabu("X[r]")) as first_page_table:    
    branch = MiniPage(width=NoEscape(r"0.49\textwidth"), pos='t!', align='r')
    branch.append(MediumText(bold(f"Nombre: ")))
    branch.append(MediumText(f"{nombre.title()}"))
    branch.append(LineBreak())
    x = datetime.datetime.now()
    fecha = x.strftime("%d/%m/%Y")
    branch.append(MediumText(bold(f"Fecha: ")))
    branch.append(MediumText(f"{fecha}"))

  first_page_table.add_row([branch])
  first_page_table.add_empty_row()

  with doc.create(Section("Introducción")):
    doc.append('Informe en el que se detalla la información producida por una Inteligencia Articial para la predicción académica de alumnos y alumnas de Bachillerato.\n\n')
    doc.append('Señalar que estos resultados son generados por una ') 
    doc.append(bold('Inteligencia Artificial '))
    doc.append('y que no son determinantes para el futuro de una persona, sino que representan una herramienta más de descubrimiento personal y académico.')
    doc.append('\n\nAdemás, todo el proceso realizado para la conformación de este informe se ha hecho ')
    doc.append(bold('respetando la privacidad de los datos aportados, '))
    doc.append("los cuáles se emplearán únicamente para fines de investigación universitaria y mejora del asistente conversacional.")
    doc.append("\n\nPor último, se indica que todo este proyecto se enmarca dentro de la filosofía de ")
    doc.append(italic('software '))
    doc.append("libre, ajustándose a las licencias ")
    doc.append(hyperlink("https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt", "MIT "))
    doc.append("y ")
    doc.append(hyperlink("https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode", "CC BY-NC-SA"))
  with doc.create(Section("Predicción de éxito académico")):
    with doc.create(Subsection('¿Ciencias o Letras?')):
      with doc.create(Itemize()) as itemize:
        itemize.add_item("La probabilidad de acabar con éxito un grado relacionado con el ámbito científico y numérico es del ")
        doc.append(bold(f"{ciencias}%"))
        itemize.add_item("La probabilidad de acabar con éxito un grado relacionado con el ámbito humanístico es del ")
        doc.append(bold(f"{letras}%"))
  # GRÁFICOS
  with doc.create(Section("Recomendación de grados universitarios")):
    with doc.create(Subsection("Grados más adecuados según el perfil")):
      with doc.create(Figure(position='h!')) as grafico:
        grafico.add_image(img_filename_mayor, width='335px')
        grafico.add_caption("Gráfico con los grados que más encajan con el usuario según la Inteligencia Artificial")
      
    # LISTA DE GRADOS DE MAYOR PROB CON SU ENLACE WEB
    with doc.create(Subsection('Lista de los grados recomendados y su enlace web')):
      with doc.create(Tabular('|c|c')) as table:
        table.add_hline()
        table.add_row((MultiColumn(2, align='|c|', data='Recomendación IA'),))
        table.add_hline()
        table.add_row(("Grado", "Enlace web"))
        table.add_hline()

        len_listas = len(grados_reales_mayor) 
        
        for x in range(len_listas):
          grado = None
          enlace = None
          grado = grados_reales_mayor[x]
          enlace = hyperlink(enlaces_reales_mayor[x], "web")
          table.add_row((grado, enlace))
          table.add_hline()

    doc.append(NewPage())

    # LISTA DE GRADOS DE MENOR PROB CON SU ENLACE WEB
    with doc.create(Subsection('Grados de menor representación')):
      doc.append('Aquí se listan los grados que han obtenido una probabilidad más baja en la predicción; sin embargo, representan carreras universitarias que se deberían consultar y tener en cuenta.\n')
      doc.append(LineBreak())
      with doc.create(Tabular('|c|c')) as table:
        table.add_hline()
        table.add_row((MultiColumn(2, align='|c|', data='Recomendación IA'),))
        table.add_hline()
        table.add_row(("Grado", "Enlace web"))
        table.add_hline()

        len_listas = len(grados_reales_menor) 
        
        for x in range(len_listas):
          grado = None
          enlace = None
          grado = grados_reales_menor[x]
          enlace = hyperlink(enlaces_reales_menor[x], "web")
          table.add_row((grado, enlace))
          table.add_hline()

  doc.generate_pdf('Informe_IA', clean_tex=True)

#--------------------TOKEN--------------------

# creamos el bot
#TOKEN = None
#with open("token.txt") as f:
#  TOKEN = f.read().strip()
bot = telebot.TeleBot(TOKEN)

#--------------------Módulo 1: saludo y preguntas iniciales--------------------

# detección o creación de csv
file_exists = exists("usuarios.csv")
if file_exists == False:
  with open('usuarios.csv', 'a') as f:  
    writer = csv.writer(f)
    writer.writerow(["id", "nombre", "pronombre", "universidad", "grado", "otros_grados", "no_universidad", "motivo"])

usuarios = {}
pronombre_indice = {}

@bot.message_handler(commands=["start", "comenzar"])
def start(message):
  """Saluda, introduce qué hace como chatbot y pregunta el nombre al usuario"""
  markup = ReplyKeyboardRemove()
  bot.send_message(message.chat.id, "Soy Cloudia \U0001F916, ¡encantadx de conocerte!", reply_markup=markup)
  time.sleep(1)
  bot.send_message(message.chat.id, "Aquí podrás conocer la predicción de futuro académica que hace mi Inteligencia Artificial en relación a una serie de datos que me aportarás a lo largo de nuestra conversación")
  time.sleep(2)
  bot.send_message(message.chat.id, "Al final, se generará un informe con los porcentajes de éxito académico tanto en Ciencias como en Humanidades y las recomendaciones de los grados universitarios donde podrías encajar")
  time.sleep(2.5)
  bot.send_message(message.chat.id, "Además, puedo enlazarte con los planes de estudio de los grados que más te interesen de la Universidad de Granada")
  time.sleep(2)
  # preguntamos
  bot.send_message(message.chat.id, "¡Ups! ¡Pero qué mala educación la mía! \U0001F635 Se me olvidaban dos preguntas muy importantes", parse_mode="markdown")
  time.sleep(1.5)
  msg = bot.send_message(message.chat.id, "¿Cuál es tu *nombre*?", parse_mode="markdown")
  bot.register_next_step_handler(msg, nombre)

def nombre(message):
  """Pregunta el pronombre al usuario"""
  usuarios[message.chat.id] = {}
  usuarios[message.chat.id]["nombre"] = normalize(message.text).lower()
  nombre = usuarios[message.chat.id]["nombre"]
  time.sleep(0.5)
  bot.send_message(message.chat.id, f"¡Hola, {nombre.title()}! \U0001F44B", parse_mode="markdown")
  time.sleep(1.5)
  # definimos los botones
  markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
  markup_teclado.add("Él", "Ella", "Elle")
  # preguntamos
  msg = bot.send_message(message.chat.id, "¿Con qué *pronombre* quieres que te trate?", reply_markup=markup_teclado, parse_mode="markdown")
  bot.register_next_step_handler(msg, universidad)

def universidad(message):
  """Pregunta asistencia a la universidad"""
  usuarios[message.chat.id]["pronombre"] = normalize(message.text).lower()
  if normalize(message.text.lower()) == "el":
    pronombre_indice["numero"] = 0
    bot.send_message(message.chat.id, "¡Genial! ¡Lo tendré en cuenta! \U0001F600")
    time.sleep(1)
    bot.send_message(message.chat.id, "Ahora permíteme que te haga unas preguntas de inicio muy rápidas")
    #name = message.from_user.first_name
    nombre = usuarios[message.chat.id]["nombre"]
    time.sleep(1.5)
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, f"¿Tienes pensado ir a la universidad, {nombre.title()}?", reply_markup=markup)
    bot.register_next_step_handler(msg, si_no_universidad)
  elif normalize(message.text.lower()) == "ella":
    pronombre_indice["numero"] = 1
    bot.send_message(message.chat.id, "¡Genial! ¡Lo tendré en cuenta! \U0001F600")
    time.sleep(1)
    bot.send_message(message.chat.id, "Ahora permíteme que te haga unas preguntas de inicio muy rápidas")
    #name = message.from_user.first_name
    nombre = nombre = usuarios[message.chat.id]["nombre"]
    time.sleep(1.5)
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, f"¿Tienes pensado ir a la universidad, {nombre.title()}?", reply_markup=markup)
    bot.register_next_step_handler(msg, si_no_universidad)
  elif normalize(message.text.lower()) == "elle":
    pronombre_indice["numero"] = 2
    bot.send_message(message.chat.id, "¡Genial! ¡Lo tendré en cuenta! \U0001F600")
    time.sleep(1)
    bot.send_message(message.chat.id, "Ahora permíteme que te haga unas preguntas de inicio muy rápidas")
    #name = message.from_user.first_name
    nombre = nombre = usuarios[message.chat.id]["nombre"]
    time.sleep(1.5)
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, f"¿Tienes pensado ir a la universidad, {nombre.title()}?", reply_markup=markup)
    bot.register_next_step_handler(msg, si_no_universidad)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Él'*, *'Ella'* o *'Elle'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Él", "Ella", "Elle")
    msg = bot.send_message(message.chat.id, "¿Con qué *pronombre* quieres que te trate?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, universidad)

def si_no_universidad(message):
  """Pregunta qué carrera escogerá o qué hará si no va a la universidad"""
  usuarios[message.chat.id]["universidad"] = normalize(message.text).lower()
  usuarios[message.chat.id]["no_universidad"] = "va a la universidad" 
  if normalize(message.text.lower()) == "si":
    msg = bot.send_message(message.chat.id, "¿Cuál es la carrera universitaria a la que tienes ahora mismo pensado acceder como primera opción?")
    bot.register_next_step_handler(msg, si_universidad)
  elif normalize(message.text.lower()) == "no":
    msg = bot.send_message(message.chat.id, "Si no quieres ir a la universidad, ¿qué tienes pensado hacer?")
    bot.register_next_step_handler(msg, no_universidad)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Tienes pensado ir a la universidad?", reply_markup=markup)
    bot.register_next_step_handler(msg, si_no_universidad)

def si_universidad(message):
  """Pregunta si tiene otros grados pensados a parte de su primera opción"""
  usuarios[message.chat.id]["grado"] = normalize(message.text).lower()
  if not message.text.startswith("/"):
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Tienes otros grados pensados a los que te gustaría acceder?", reply_markup=markup)
    bot.register_next_step_handler(msg, otros_grados_universidad)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \n*Escribe* el nombre de la carrera", parse_mode="markdown")
    msg = bot.send_message(message.chat.id, "¿Cuál es la carrera universitaria a la que tienes ahora mismo pensado acceder como primera opción?")
    bot.register_next_step_handler(msg, si_no_universidad)

def otros_grados_universidad(message):
  """Pregunta qué otros grados tiene pensados o el motivo de no tener otros grados como segunda opción"""
  lista_opciones_1 = ["Lo tengo claro desde pequeño", "Lo tengo claro desde pequeña", "Lo tengo claro desde pequeñe"]
  if normalize(message.text.lower()) == "si":
    msg = bot.send_message(message.chat.id, "¿Cuáles?")
    bot.register_next_step_handler(msg, emocion_universidad)
  elif normalize(message.text.lower()) == "no":
    usuarios[message.chat.id]["otros_grados"] = "no"
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón o escribe en el teclado", row_width=1, resize_keyboard=True)
    markup.add(lista_opciones_1[pronombre_indice["numero"]], "No me interesan otros grados", "No conozco otros grados lo suficiente", "Otros motivos")
    # preguntamos
    bot.send_message(message.chat.id, "¿Cuál es el motivo de decirme solo un grado?")
    msg = bot.send_message(message.chat.id, "Pulsa un botón para elegir una opción o escribe en el teclado otro motivo que ahí no aparezca", reply_markup=markup)
    bot.register_next_step_handler(msg, guardar_datos)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Tienes otros grados pensados a los que te gustaría acceder?", reply_markup=markup)
    bot.register_next_step_handler(msg, otros_grados_universidad)

def emocion_universidad(message):
  """Pregunta el motivo de tener otros grados como segunda opción"""
  usuarios[message.chat.id]["otros_grados"] = normalize(message.text).lower() 
  if not message.text.startswith("/"):
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", row_width=1, resize_keyboard=True)
    markup.add("Conozco qué ofrecen estos grados, pero no tengo clara mi decisión", "No conozco lo suficiente los grados y estos son los que siempre he pensado hacer", "Otros motivos")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Cuál es el motivo de tener en mente varios grados a elegir?", reply_markup=markup)
    bot.register_next_step_handler(msg, guardar_datos)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \n*Escribe* los nombres de las carreras", parse_mode="markdown")
    msg = bot.send_message(message.chat.id, "¿Tienes otros grados pensados a los que te gustaría acceder?")
    bot.register_next_step_handler(msg, emocion_universidad)

def no_universidad(message):
  print(pronombre_indice)
  """Pregunta el motivo de no querer asistir a la universidad"""
  lista_opciones_1 = ["No me siento informado", "No me siento informada", "No me siento informade"]
  lista_opciones_2 = ["No me siento preparado", "No me siento preparada", "No me siento preparade"]
  usuarios[message.chat.id]["grado"] = "no"
  usuarios[message.chat.id]["otros_grados"] = "no"
  usuarios[message.chat.id]["no_universidad"] = normalize(message.text).lower()
  if not message.text.startswith("/"):
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón o escribe en el teclado", row_width=1, resize_keyboard=True)
    markup.add("No quiero ir a la universidad", lista_opciones_1[pronombre_indice["numero"]], lista_opciones_2[pronombre_indice["numero"]], "Otros motivos")
    # preguntamos
    bot.send_message(message.chat.id, "¿Cuál es el motivo de no querer ir a la universidad?")
    msg = bot.send_message(message.chat.id, "Pulsa un botón para elegir una opción o escribe en el teclado otro motivo que ahí no aparezca", reply_markup=markup)
    bot.register_next_step_handler(msg, guardar_datos)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \n*Escribe* qué quieres hacer", parse_mode="markdown")
    msg = bot.send_message(message.chat.id, "Si no quieres ir a la universidad, ¿qué tienes pensado hacer?")
    bot.register_next_step_handler(msg, no_universidad)

def guardar_datos(message):
  """Guarda la información recopilada"""
  usuarios[message.chat.id]["motivo"] = normalize(message.text).lower()
  print(usuarios)
  # exportar datos a csv
  with open('usuarios.csv', 'a') as f:  
    writer = csv.writer(f)
    for id, dic in usuarios.items():
      writer.writerow([id, dic["nombre"], dic["pronombre"], dic["universidad"], dic["grado"], dic["otros_grados"], dic["no_universidad"], dic["motivo"]])
  markup = ReplyKeyboardRemove()
  bot.send_message(message.chat.id, "¡Genial! Muchas gracias por tus respuestas", reply_markup=markup)
  time.sleep(1.5)
  bot.send_message(message.chat.id, "Dame un segundo para trabajar sobre ellas")
  time.sleep(1)
  # stickers files id
  oso = "CAACAgIAAxkBAAJAxmLjJZwqXq71KNf5QA2sLaiBc8JoAAIFAQAC9wLID9HldLdGJaShKQQ"
  cocodrilo = "CAACAgIAAxkBAAJAwmLjJZI6x4V_z2Btl9Vdh7h3HyWMAAIgCQACGELuCOGKIKihOgrZKQQ"
  lista_sti = [oso, cocodrilo]
  numero_random = rd.randint(0, 1)
  bot.send_chat_action(message.chat.id, "typing")
  bot.send_sticker(message.chat.id, lista_sti[numero_random])
  time.sleep(7)
  msg = bot.send_message(message.chat.id, "¡Ya estoy! Ahora... ¡Sigamos con la predicción académica! \U0001F4AB")
  time.sleep(2)
  # pasamos al siguiente módulo: predicción de éxito académico
  start_prediccion(msg)

  #--------------------Módulo 2: predicción éxito académico--------------------

  #----------Preguntas----------

respuestas = {}
prediccion_ciencias_letras = {}

@bot.message_handler(commands=["prediccion1"])
def start_prediccion(message):
  """Bienvenida al módulo de predicción académica en torno a las ciencias y las letras"""
  markup = ReplyKeyboardRemove()
  bot.send_message(message.chat.id, "Déjame que vaya sacando mi bola de cristal...", reply_markup=markup)
  time.sleep(1)
  bola_1 = "CAACAgIAAxkBAAJBrGL5FXvqDx2_QR-_U03cT-BWBMyiAAJeEgACh9XwS8hbRrJNn5s8KQQ"
  bola_2 = "CAACAgIAAxkBAAJA3WLlvJnnuK-3neaq1waVUPBrN3jkAAI9EwACVl9JS-nYaAzAlApkKQQ"
  bola_3 = "CAACAgIAAxkBAAJBqGL5FV2n9rzBmDgj_si2HlWBxD5EAALkAAPOGnYL34mTpRR8z4spBA"
  lista_sti = [bola_1, bola_2, bola_3]
  numero_random = rd.randint(0, 1)
  bot.send_sticker(message.chat.id, lista_sti[numero_random])
  time.sleep(2.5)
  msg = bot.send_message(message.chat.id, "Para realizar la predicción académica necesito que me respondas a una serie de preguntas muy breves")
  time.sleep(2.5)
  sexo(msg)

def sexo(message):
  # definimos los botones
  markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
  markup_teclado.add("Femenino", "Masculino")
  # preguntamos
  msg = bot.send_message(message.chat.id, "¿Cuál es tu sexo biológico?", reply_markup=markup_teclado)
  bot.register_next_step_handler(msg, direccion)

def direccion(message):
  respuestas[message.chat.id] = {}
  if normalize(message.text.lower()) == "femenino" or normalize(message.text.lower()) == "masculino":
    if normalize(message.text.lower()) == "femenino":
      respuestas[message.chat.id]["sex"] = 1
    elif normalize(message.text.lower()) == "masculino":
      respuestas[message.chat.id]["sex"] = 0
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Rural", "Urbana")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Vives en una zona rural (pueblo) o urbana (ciudad)?", reply_markup=markup)
    bot.register_next_step_handler(msg, tamano_familia)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Femenino'* o *'Masculino'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Femenino", "Masculino")
    msg = bot.send_message(message.chat.id, "¿Cuál es tu sexo biológico?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, direccion)

def tamano_familia(message):
  if normalize(message.text.lower()) == "rural" or normalize(message.text.lower()) == "urbana":
    if normalize(message.text.lower()) == "rural":
      respuestas[message.chat.id]["address"] = 0
    elif normalize(message.text.lower()) == "urbana":
      respuestas[message.chat.id]["address"] = 1
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Menos o igual a tres miembros", "Más de tres miembros")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Cuántos miembros convivís en casa?", reply_markup=markup)
    bot.register_next_step_handler(msg, edu_materno)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Rural'* o *'Urbana'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Rural", "Urbana")
    msg = bot.send_message(message.chat.id, "¿Vives en una zona rural (pueblo) o urbana (ciudad)?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, tamano_familia)

def edu_materno(message):
  if normalize(message.text.lower()) == "menos o igual a tres miembros" or normalize(message.text.lower()) == "mas de tres miembros":
    if normalize(message.text.lower()) == "menos o igual a tres miembros":
      respuestas[message.chat.id]["famsize"] = 0
    elif normalize(message.text.lower()) == "mas de tres miembros":
      respuestas[message.chat.id]["famsize"] = 1
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Nada", "Educación Primaria", "ESO", "Bachillerato", "Grado universitario")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Qué nivel académico tiene la que consideras tu figura maternal?", reply_markup=markup)
    bot.register_next_step_handler(msg, edu_paterno)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Menos o igual a tres miembros'* o *'Más de tres miembros'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Menos o igual a tres miembros", "Más de tres miembros")
    msg = bot.send_message(message.chat.id, "¿Cuántos miembros convivís en casa?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, edu_materno)

def edu_paterno(message):
  if normalize(message.text.lower()) == "nada" or normalize(message.text.lower()) == "educacion primaria" or normalize(message.text.lower()) == "eso" or normalize(message.text.lower()) == "bachillerato" or normalize(message.text.lower()) == "grado universitario":
    if normalize(message.text.lower()) == "nada":
      respuestas[message.chat.id]["Medu"] = 0
    elif normalize(message.text.lower()) == "educacion primaria":
      respuestas[message.chat.id]["Medu"] = 1
    elif normalize(message.text.lower()) == "eso":
      respuestas[message.chat.id]["Medu"] = 2
    elif normalize(message.text.lower()) == "bachillerato":
      respuestas[message.chat.id]["Medu"] = 3
    elif normalize(message.text.lower()) == "grado universitario":
      respuestas[message.chat.id]["Medu"] = 4
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Nada", "Educación Primaria", "ESO", "Bachillerato", "Grado universitario")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Qué nivel académico tiene la que consideras tu figura paternal?", reply_markup=markup)
    bot.register_next_step_handler(msg, tiempo_estudio_mates)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Nada'*, *'Educación Primaria'*, *'ESO'*, *'Bachillerato'* o *'Grado universitario'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Nada", "Educación Primaria", "ESO", "Bachillerato", "Grado universitario")
    msg = bot.send_message(message.chat.id, "¿Qué nivel académico tiene la que consideras tu figura maternal?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, edu_paterno)

def tiempo_estudio_mates(message):
  if normalize(message.text.lower()) == "nada" or normalize(message.text.lower()) == "educacion primaria" or normalize(message.text.lower()) == "eso" or normalize(message.text.lower()) == "bachillerato" or normalize(message.text.lower()) == "grado universitario":
    if normalize(message.text.lower()) == "nada":
      respuestas[message.chat.id]["Fedu"] = 0
    elif normalize(message.text.lower()) == "educacion primaria":
      respuestas[message.chat.id]["Fedu"] = 1
    elif normalize(message.text.lower()) == "eso":
      respuestas[message.chat.id]["Fedu"] = 2
    elif normalize(message.text.lower()) == "bachillerato":
      respuestas[message.chat.id]["Fedu"] = 3
    elif normalize(message.text.lower()) == "grado universitario":
      respuestas[message.chat.id]["Fedu"] = 4
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Menos de dos horas", "Entre dos y cinco horas", "Entre cinco y diez horas", "Más de diez horas")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Cuántas horas estudias *Matemáticas* a la semana?", reply_markup=markup, parse_mode="markdown")
    bot.register_next_step_handler(msg, tiempo_estudio_lengua)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Nada'*, *'Educación Primaria'*, *'ESO'*, *'Bachillerato'* o *'Grado universitario'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Nada", "Educación Primaria", "ESO", "Bachillerato", "Grado universitario")
    msg = bot.send_message(message.chat.id, "¿Qué nivel académico tiene la que consideras tu figura paternal?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, tiempo_estudio_mates)

def tiempo_estudio_lengua(message):
  if normalize(message.text.lower()) == "menos de dos horas" or normalize(message.text.lower()) == "entre dos y cinco horas" or normalize(message.text.lower()) == "entre cinco y diez horas" or normalize(message.text.lower()) == "mas de diez horas" or normalize(message.text.lower()) == "grado universitario":
    if normalize(message.text.lower()) == "menos de dos horas":
      respuestas[message.chat.id]["studytime_mates"] = 1
    elif normalize(message.text.lower()) == "entre dos y cinco horas":
      respuestas[message.chat.id]["studytime_mates"] = 2
    elif normalize(message.text.lower()) == "entre cinco y diez horas":
      respuestas[message.chat.id]["studytime_mates"] = 3
    elif normalize(message.text.lower()) == "mas de diez horas":
      respuestas[message.chat.id]["studytime_mates"] = 4
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Menos de dos horas", "Entre dos y cinco horas", "Entre cinco y diez horas", "Más de diez horas")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Y *Lengua*?", reply_markup=markup, parse_mode="markdown")
    bot.register_next_step_handler(msg, apoyo_familia_mates)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Menos de dos horas'*, *'Entre dos y cinco horas'*, *'Entre cinco y diez horas'* o *'Más de diez horas'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Menos de dos horas", "Entre dos y cinco horas", "Entre cinco y diez horas", "Más de diez horas")
    msg = bot.send_message(message.chat.id, "¿Cuántas horas estudias *Matemáticas* a la semana?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, tiempo_estudio_lengua)

def apoyo_familia_mates(message):
  if normalize(message.text.lower()) == "menos de dos horas" or normalize(message.text.lower()) == "entre dos y cinco horas" or normalize(message.text.lower()) == "entre cinco y diez horas" or normalize(message.text.lower()) == "mas de diez horas" or normalize(message.text.lower()) == "grado universitario":
    if normalize(message.text.lower()) == "menos de dos horas":
      respuestas[message.chat.id]["studytime_lengua"] = 1
    elif normalize(message.text.lower()) == "entre dos y cinco horas":
      respuestas[message.chat.id]["studytime_lengua"] = 2
    elif normalize(message.text.lower()) == "entre cinco y diez horas":
      respuestas[message.chat.id]["studytime_lengua"] = 3
    elif normalize(message.text.lower()) == "mas de diez horas":
      respuestas[message.chat.id]["studytime_lengua"] = 4
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Tienes a alguna persona en casa que te puede ayudar con *Matemáticas*?", reply_markup=markup, parse_mode="markdown")
    bot.register_next_step_handler(msg, apoyo_familia_lengua)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Menos de dos horas'*, *'Entre dos y cinco horas'*, *'Entre cinco y diez horas'* o *'Más de diez horas'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Menos de dos horas", "Entre dos y cinco horas", "Entre cinco y diez horas", "Más de diez horas")
    msg = bot.send_message(message.chat.id, "¿Cuántas horas estudias *Lengua* a la semana?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, apoyo_familia_mates)

def apoyo_familia_lengua(message):
  if normalize(message.text.lower()) == "si" or normalize(message.text.lower()) == "no":
    if normalize(message.text.lower()) == "si":
      respuestas[message.chat.id]["famsup_mates"] = 1
    elif normalize(message.text.lower()) == "no":
      respuestas[message.chat.id]["famsup_mates"] = 0
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Y con *Lengua*?", reply_markup=markup, parse_mode="markdown")
    bot.register_next_step_handler(msg, academia_mates)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Tienes a alguna persona en casa que te puede ayudar con *Matemáticas*?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, apoyo_familia_lengua)

def academia_mates(message):
  if normalize(message.text.lower()) == "si" or normalize(message.text.lower()) == "no":
    if normalize(message.text.lower()) == "si":
      respuestas[message.chat.id]["famsup_lengua"] = 1
    elif normalize(message.text.lower()) == "no":
      respuestas[message.chat.id]["famsup_lengua"] = 0
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Acudes a clases de *Matemáticas* por las tardes, ya sean particulares o en una academia?", reply_markup=markup, parse_mode="markdown")
    bot.register_next_step_handler(msg, academia_lengua)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Tienes a alguna persona en casa que te puede ayudar con *Lengua*?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, academia_mates)

def academia_lengua(message):
  if normalize(message.text.lower()) == "si" or normalize(message.text.lower()) == "no":
    if normalize(message.text.lower()) == "si":
      respuestas[message.chat.id]["paid_mates"] = 1
    elif normalize(message.text.lower()) == "no":
      respuestas[message.chat.id]["paid_mates"] = 0
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Y de *Lengua*?", reply_markup=markup, parse_mode="markdown")
    bot.register_next_step_handler(msg, extraescolares)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Acudes a clases de *Matemáticas* por las tardes, ya sean particulares o en una academia?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, academia_lengua)

def extraescolares(message):
  if normalize(message.text.lower()) == "si" or normalize(message.text.lower()) == "no":
    if normalize(message.text.lower()) == "si":
      respuestas[message.chat.id]["paid_lengua"] = 1
    elif normalize(message.text.lower()) == "no":
      respuestas[message.chat.id]["paid_lengua"] = 0
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Tienes actividades extraescolares (baile, baloncesto, etc.) dos o más días a la semana?", reply_markup=markup)
    bot.register_next_step_handler(msg, internet)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Acudes a clases de *Lengua* por las tardes, ya sean particulares o en una academia?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, extraescolares)

def internet(message):
  if normalize(message.text.lower()) == "si" or normalize(message.text.lower()) == "no":
    if normalize(message.text.lower()) == "si":
      respuestas[message.chat.id]["activities"] = 1
    elif normalize(message.text.lower()) == "no":
      respuestas[message.chat.id]["activities"] = 0
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Tienes acceso a internet en tu casa?", reply_markup=markup)
    bot.register_next_step_handler(msg, primer_trimestre_matematicas)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Tienes actividades extraescolares (baile, baloncesto, etc.) dos o más días a la semana?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, internet)

def primer_trimestre_matematicas(message):
  if normalize(message.text.lower()) == "si" or normalize(message.text.lower()) == "no":
    if normalize(message.text.lower()) == "si":
      respuestas[message.chat.id]["internet"] = 1
    elif normalize(message.text.lower()) == "no":
      respuestas[message.chat.id]["internet"] = 0
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Aprobaste *Matemáticas* en el primer trimestre (último primer trimestre cursado)?", reply_markup=markup, parse_mode="markdown")
    bot.register_next_step_handler(msg, primer_trimestre_lengua)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Tienes acceso a internet en tu casa?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, primer_trimestre_matematicas)

def primer_trimestre_lengua(message):
  if normalize(message.text.lower()) == "si" or normalize(message.text.lower()) == "no":
    if normalize(message.text.lower()) == "si":
      respuestas[message.chat.id]["G1_mates"] = 1
    elif normalize(message.text.lower()) == "no":
      respuestas[message.chat.id]["G1_mates"] = 0
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Y *Lengua*?", reply_markup=markup, parse_mode="markdown")
    bot.register_next_step_handler(msg, segundo_trimestre_matematicas)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Aprobaste *Matemáticas* en el primer trimestre (último primer trimestre cursado)?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, primer_trimestre_lengua)

def segundo_trimestre_matematicas(message):
  if normalize(message.text.lower()) == "si" or normalize(message.text.lower()) == "no":
    if normalize(message.text.lower()) == "si":
      respuestas[message.chat.id]["G1_lengua"] = 1
    elif normalize(message.text.lower()) == "no":
      respuestas[message.chat.id]["G1_lengua"] = 0
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Aprobaste *Matemáticas* en el segundo trimestre (último segundo trimestre cursado)?", reply_markup=markup, parse_mode="markdown")
    bot.register_next_step_handler(msg, segundo_trimestre_lengua)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Aprobaste *Lengua* en el primer trimestre (último primer trimestre cursado)?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, segundo_trimestre_matematicas)

def segundo_trimestre_lengua(message):
  if normalize(message.text.lower()) == "si" or normalize(message.text.lower()) == "no":
    if normalize(message.text.lower()) == "si":
      respuestas[message.chat.id]["G2_mates"] = 1
    elif normalize(message.text.lower()) == "no":
      respuestas[message.chat.id]["G2_mates"] = 0
    # definimos los botones
    markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
    markup.add("Sí", "No")
    # preguntamos
    msg = bot.send_message(message.chat.id, "¿Y *Lengua*?", reply_markup=markup, parse_mode="markdown")
    bot.register_next_step_handler(msg, datos)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Aprobaste *Matemáticas* en el segundo trimestre (último segundo trimestre cursado)?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, segundo_trimestre_lengua)

#----------Procesamiento de los datos----------

def datos(message):
  if normalize(message.text.lower()) == "si" or normalize(message.text.lower()) == "no":
    if normalize(message.text.lower()) == "si":
      respuestas[message.chat.id]["G2_lengua"] = 1
    elif normalize(message.text.lower()) == "no":
      respuestas[message.chat.id]["G2_lengua"] = 0
    sexo = respuestas[message.chat.id]["sex"]
    direccion = respuestas[message.chat.id]["address"]
    tamano_familia = respuestas[message.chat.id]["famsize"]
    edu_materno = respuestas[message.chat.id]["Medu"]
    edu_paterno = respuestas[message.chat.id]["Fedu"]
    tiempo_estudio_mates = respuestas[message.chat.id]["studytime_mates"]
    tiempo_estudio_lengua = respuestas[message.chat.id]["studytime_lengua"]
    apoyo_familia_mates = respuestas[message.chat.id]["famsup_mates"]
    apoyo_familia_lengua = respuestas[message.chat.id]["famsup_lengua"]
    academia_mates = respuestas[message.chat.id]["paid_mates"]
    academia_lengua = respuestas[message.chat.id]["paid_lengua"]
    extraescolares = respuestas[message.chat.id]["activities"]
    internet = respuestas[message.chat.id]["internet"]
    primer_trimestre_matematicas = respuestas[message.chat.id]["G1_mates"]
    segundo_trimestre_matematicas = respuestas[message.chat.id]["G2_mates"]
    primer_trimestre_lengua = respuestas[message.chat.id]["G1_lengua"]
    segundo_trimestre_lengua = respuestas[message.chat.id]["G2_lengua"]

    matematicas = np.array([[sexo, direccion, tamano_familia, edu_materno, edu_paterno, tiempo_estudio_mates, apoyo_familia_mates, academia_mates, extraescolares, internet, primer_trimestre_matematicas, segundo_trimestre_matematicas]])
    lengua = np.array([[sexo, direccion, tamano_familia, edu_materno, edu_paterno, tiempo_estudio_lengua, apoyo_familia_lengua, academia_lengua, extraescolares, internet, primer_trimestre_lengua, segundo_trimestre_lengua]])
    
    # predicción matemáticas
    model_matematicas = tf.keras.models.load_model('./Cloudia_Bot/model.school_prediction_mat')
    resultado_prediccion_matematicas = model_matematicas.predict(matematicas)[0][0] # acceder al float del numpy.array
    # predicción lengua
    model_lengua = tf.keras.models.load_model('./Cloudia_Bot/model.school_prediction_len')
    resultado_prediccion_lengua = model_lengua.predict(lengua)[0][0] # acceder al float del numpy.array
    

    prediccion_ciencias_letras["matematicas"] = round(resultado_prediccion_matematicas * 100, 4)
    prediccion_ciencias_letras["lengua"] = round(resultado_prediccion_lengua * 100, 4)

    print(prediccion_ciencias_letras)

    markup = ReplyKeyboardRemove()
    bot.send_message(message.chat.id, "¡Perfecto! Muchas gracias por tus respuestas", reply_markup=markup)
    time.sleep(1.5)
    bot.send_message(message.chat.id, "Dame otro segundo para que mi Inteligencia Artificial las procese matemáticamente")
    time.sleep(1)
    # stickers files id
    yoda = "CAACAgIAAxkBAAJAsmLjI66pRVgsrb7pFLR3C_BbaY1HAAJ4AgACVp29Cvy6CLWRfRwMKQQ"
    patricio = "CAACAgIAAxkBAAJBlGL5FEjzi2c_yPzqF6c5p7J3akZ-AALgAAOWn4wOUr49XZjLh8cpBA"
    lista_sti = [yoda, patricio]
    numero_random = rd.randint(0, 1)
    bot.send_sticker(message.chat.id, lista_sti[numero_random])
    time.sleep(7)
    msg = bot.send_message(message.chat.id, "¡Ya estoy! ¡Vayamos con el último paso! \U0001F6B6")
    time.sleep(4)
    # pasamos al siguiente módulo: predicción de éxito académico
    start_prediccion_grados(msg)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Aprobaste *Lengua* en el segundo trimestre (último segundo trimestre cursado)?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, datos)

#--------------------Módulo 3: predicción de grados--------------------

#----------Importación del modelo ya entrenado----------

# restaurar toda nuestra estructura de datos
import pickle
data = pickle.load(open("./Cloudia_Bot/net_dnn/training_data", "rb")) #rb --> lectura binaria
words = data["words"]
etiquetas = data["classes"]
train_x = data["train_x"]
train_y = data["train_y"]

# importar nuestro archivo de intenciones
import json
with open("./Cloudia_Bot/net_dnn/intenciones.json") as json_data:
  intents = json.load(json_data)

# restablecer los datos del gráfico
tf.compat.v1.reset_default_graph()

# Construir la red neuronal
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Definir el modelo y configurar el tensorboard
model = tflearn.DNN(net, tensorboard_dir='./Cloudia_Bot/net_dnn/tflearn_logs')

# load our saved model
model.load("./Cloudia_Bot/net_dnn/model.tflearn_prediction")

#----------Desarrollo del bot en telegram----------

entradas = {}
cuenta_imagen = {"numero": 1}
email_dict = {}

# detección o creación de csv
file_exists = exists("entradas_grados.csv")
if file_exists == False:
  with open('entradas_grados.csv', 'a') as f:  
    writer = csv.writer(f)
    writer.writerow(["entrada_1", "entrada_2"])

@bot.message_handler(commands=["prediccion2"])
def start_prediccion_grados(message):
  """Expone de qué va esta predicción y muestra un ejemplo de entrada"""
  markup = ReplyKeyboardRemove()
  bot.send_message(message.chat.id, "Para realizar esta última predicción, necesito que escribas un mensaje donde con oraciones muy *simples* y *afirmativas* hables sobre tus asignaturas favoritas, qué se te da bien, qué te gusta, etc.", reply_markup=markup, parse_mode="markdown")
  time.sleep(3)
  bot.send_message(message.chat.id, "Aquí te dejo un ejemplo:")
  time.sleep(2)
  msg = bot.send_message(message.chat.id, "Mi asignatura favorita es matemáticas, aunque también se me da bien la química y la física. Dentro de ellas me gusta la parte de ecuaciones y química orgánica. Me gusta viajar. Soy una persona colaborativa, que ayuda a los demás y que le encanta cantar")
  bot.register_next_step_handler(msg, seguir)

def seguir(message):
  entradas[message.chat.id] = {}
  entradas[message.chat.id]["primera"] = normalize(message.text)
  print(entradas)
  # definimos los botones
  markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
  markup_teclado.add("Sí", "No")
  # preguntamos
  msg = bot.send_message(message.chat.id, "¿Quieres añadir algo más?", reply_markup=markup_teclado)
  bot.register_next_step_handler(msg, respuesta)

def respuesta(message):
  if normalize(message.text.lower()) == "si":
    msg = bot.send_message(message.chat.id, "¡Genial! Ya puedes escribir")
    bot.register_next_step_handler(msg, final)
  elif normalize(message.text.lower()) == "no":
    bot.send_message(message.chat.id, "¡Genial!")
    time.sleep(1.5)
    bot.send_message(message.chat.id, "¡El trabajo por tu parte está hecho!")
    time.sleep(1.5)
    msg = bot.send_message(message.chat.id, "Ahora déjame un poco de tiempo para que mi Inteligencia Artificial pueda trabajar sobre lo que me has dicho")
    time.sleep(2)
    mono = "CAACAgEAAxkBAAJAumLjJDDtghIY6CpE_akxYixnpsLXAAIeAQACOA6CEUZYaNdphl79KQQ"
    spidercerdo = "CAACAgIAAxkBAAJBmGL5FHOssNZunVmLwERSLatiefPtAALUFQACldQRSoISm4Rl51_gKQQ"
    lista_sti = [mono, spidercerdo]
    numero_random = rd.randint(0, 1)
    bot.send_sticker(message.chat.id, lista_sti[numero_random])
    time.sleep(5)
    final(message)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    msg = bot.send_message(message.chat.id, "¿Quieres añadir algo más a lo que has escrito?")
    bot.register_next_step_handler(msg, respuesta)

def final(message):
  entradas[message.chat.id]["segunda"] = normalize(message.text)
  # exportar datos a csv
  with open('entradas_grados.csv', 'a') as f:  
    writer = csv.writer(f)
    for id, dic in usuarios.items():
      writer.writerow([entradas[message.chat.id]["primera"], entradas[message.chat.id]["segunda"]])
  # SI EL USUARIO DICE QUE NO AÑADE NADA MÁS A SU REDACCIÓN
  if entradas[message.chat.id]["segunda"] == "No" or entradas[message.chat.id]["segunda"] == "no":
    print("NO HA DICHO NADA MÁS")
    # TRATAMIENTO DE LA ENTRADA PARA SEPARARLA EN FRASES A TRAVÉS DE LA COMA Y LA I GRIEGA
    entrada = entradas[message.chat.id]["primera"]
    if entrada[-1] == ".":
      entrada = entrada[:-1]
    frases = entrada.split('.')
    frases_sin_espacios = []
    for frase in frases:
      frase = frase.strip()
      frases_sin_espacios.append(frase)
    frases_nuevas = []
    for frase in frases_sin_espacios:
      if "," in frase:
        frase_nueva = frase.split(",")
        for x in range(len(frase_nueva)):
          frases_nuevas.append(frase_nueva[x].strip())
      else:
        frases_nuevas.append(frase)
    
    lista_frase = []
    frases_y_sust = []
    lista_frases_y = []
    frases_resto = []
    frases_y = []
    frases = []

    for indice, frase in enumerate(frases_nuevas):
      lista_frase.append(frase.split())
    
    for indice_frase, frase in enumerate(lista_frase):
      for indice_palabra, palabra in enumerate(frase):
        if palabra == "y":
          lista_frase[indice_frase][indice_palabra] = "#"
          concatenado = " ".join(lista_frase[indice_frase])
          
    for indice_frase, frase in enumerate(lista_frase):
      concatenado = " ".join(lista_frase[indice_frase])
      frases_y_sust.append(concatenado)

    for indice, frase in enumerate(frases_y_sust):
      if "#" in frase:
        frase1 = frase.split("#")
        lista_frases_y.append(frase1)
      else:
        frases_resto.append(frase)

    for frase in lista_frases_y:
      indice = len(frase)
      for x in range(indice):
        frases_y.append(frase[x])

    for frase in frases_resto:
      frase = frase.strip()
      frases.append(frase)

    for frase in frases_y:
      frase = frase.strip()
      frases.append(frase.strip())

    # CLASIFICACIÓN FRASES CON EL MODELO ENTRENADO, SACAR PORCENTAJES Y CLASIFICACIÓN SEGÚN IMPORTANCIA PORCENTUAL
    mayor_prob = {}
    menor_prob = {}
    
    for frase in frases:
      clasif = classify(frase)
      for x in range(7):
        prob = clasif[x][1] * 100
        if 2 >= prob >= 0:
          grado = clasif[x][0]
          try:
            if menor_prob[grado] != 0:
              menor_prob[grado] = menor_prob[grado] + 1
          except:
            menor_prob[grado] = 1
        elif prob > 2:
          grado = clasif[x][0]
          try:
            if mayor_prob[grado] != 0:
              mayor_prob[grado] = mayor_prob[grado] + 1
          except:
            mayor_prob[grado] = 1  

    mayor_prob_porcentajes = {}
    menor_prob_porcentajes = {}
    total = 0

    for nombre, valor in mayor_prob.items():
      total = total + valor
    for nombre, valor in mayor_prob.items():
      porcentaje = str(round((valor * 100) / total, 2)) + "%"
      mayor_prob_porcentajes[nombre] = porcentaje

    for nombre, valor in menor_prob.items():
      total = total + valor
    for nombre, valor in menor_prob.items():
      porcentaje = str(round((valor * 100) / total, 2)) + "%"
      menor_prob_porcentajes[nombre] = porcentaje

    # JSON GRADOS
    with open("./Cloudia_Bot/net_dnn/grados.json") as grados_json:
      grados = json.load(grados_json)
    
    # EXTRACCIÓN DE NOMBRES DE GRADOS Y ENLACES A PARTIR DE LA INFORMACIÓN QUE TENEMOS
    # Y EL JSON
    grados_codificados_mayor = []
    grados_codificados_menor = []
    porcentajes_mayor = []
    porcentajes_menor = []

    grados_reales_mayor = []
    enlaces_reales_mayor = []
    grados_reales_menor = []
    enlaces_reales_menor = []

    grupos_mayor_prob = []
    grupos_menor_prob = []

    # Separar porcentajes
    for x, y in mayor_prob_porcentajes.items():
      grados_codificados_mayor.append(x)
      porcentajes_mayor.append(y)

    for x, y in menor_prob_porcentajes.items():
      grados_codificados_menor.append(x)
      porcentajes_menor.append(y)

    for n in range(31):
      for x, y in grados.items():
        if y[n]["tag"] in grados_codificados_mayor:
          grados_mayor_prob = y[n]["nombre"]
          grupos_mayor_prob.append(grados_mayor_prob)
          enlaces_mayor_prob = y[n]["enlace"]
          for a in range(0, len(grados_mayor_prob)):
            grados_reales_mayor.append(grados_mayor_prob[a])
            enlaces_reales_mayor.append(enlaces_mayor_prob[a])
        if y[n]["tag"] in grados_codificados_menor:
          grados_menor_prob = y[n]["nombre"]
          grupos_menor_prob.append(grados_menor_prob)
          enlaces_menor_prob = y[n]["enlace"]
          for a in range(0, len(grados_menor_prob)):
            grados_reales_menor.append(grados_menor_prob[a])
            enlaces_reales_menor.append(enlaces_menor_prob[a])

    print(enlaces_reales_mayor)
    print(enlaces_reales_menor)

    # UNIÓN DE GRUPOS DE GRADOS CON SUS PORCENTAJES
    grupos_mayor_prob_limp = []
    grupos_menor_prob_limp = []
    for grupo in grupos_mayor_prob:
      porcentaje = None
      for guardar_porcentaje in porcentajes_mayor:
        porcentaje = guardar_porcentaje
      grupo = ", ".join(grupo)
      grupos_mayor_prob_limp.append(grupo)

    for grupo in grupos_menor_prob:
      porcentaje = None
      for guardar_porcentaje in porcentajes_menor:
        porcentaje = guardar_porcentaje
      
      grupo = ", ".join(grupo)
      grupos_menor_prob_limp.append(grupo)

    # PORCENTAJES LIMPIOS (FLOAT)
    porcentajes_limpios_mayor = []
    porcentajes_limpios_menor = []
    for x in porcentajes_mayor:
      x = x.replace("%", "")
      porcentajes_limpios_mayor.append(float(x))
    for x in porcentajes_menor:
      x = x.replace("%", "")
      porcentajes_limpios_menor.append(float(x))

    # MENSAJE DE FINALIZACIÓN DE LA PREDICCIÓN
    bot.send_message(message.chat.id, "¡Maravilloso! Ya he terminado")
    time.sleep(1.5)
    bot.send_message(message.chat.id, "Ya solo quedaría ponerme mis gafas y escribir el informe con todos los resultados")
    time.sleep(2)
    bot.send_message(message.chat.id, "Dame un poco de tiempo y lo terminaré en un segundo. Ahora te aviso")
    time.sleep(1.5)
    perro = "CAACAgIAAxkBAAJAvmLjJWtvklGYD0zFLJI7dboloENWAAKBAQACK15TC14KbD5sAAF4tCkE"
    pez = "CAACAgIAAxkBAAJAymLjJbQHOxuhAiO_kAX7SJ9B1mWzAAJsAgACVp29CioZXpzE0N6fKQQ"
    lista_sti = [perro, pez]
    numero_random = rd.randint(0, 1)
    bot.send_sticker(message.chat.id, lista_sti[numero_random])

    # PREPARACIÓN DE LAS LISTAS DE GRADOS QUE SE ENVIARÁN PARA REALIZAR EL GRÁFICO Y LA LISTA DEL INFORME
    # limpieza de aquellos grados que se repiten en los de menor probabilidad con los de mayor probabilidad
    grupos_menor_prob_limp_sin = []
    for grupo in grupos_menor_prob_limp:
      if grupo not in grupos_mayor_prob_limp:
        grupos_menor_prob_limp_sin.append(grupo)
    #---
    grados_reales_menor_limp_sin = []
    for grado in grados_reales_menor:
      if grado not in grados_reales_mayor:
        grados_reales_menor_limp_sin.append(grado)
    #---
    enlaces_reales_menor_limp_sin = []
    for enlace in enlaces_reales_menor:
      if enlace not in enlaces_reales_mayor:
        enlaces_reales_menor_limp_sin.append(enlace)

    # AÑADIR IMAGEN
    numero_imagen_str = str(cuenta_imagen["numero"])
    len_grupos_mayor = len(grupos_mayor_prob_limp)
    crear_imagen(f"gradoMayor{numero_imagen_str}.png", grupos_mayor_prob_limp, len_grupos_mayor, porcentajes_limpios_mayor)
    print(grupos_mayor_prob_limp)
    print(grupos_menor_prob_limp)
    cuenta_imagen["numero"] += 1
    time.sleep(7)

    # MÓDULO LATEX
    latex(f"gradoMayor{numero_imagen_str}.png", grados_reales_menor_limp_sin, enlaces_reales_menor_limp_sin, grados_reales_mayor, enlaces_reales_mayor, porcentajes_mayor, prediccion_ciencias_letras["matematicas"], prediccion_ciencias_letras["lengua"], usuarios[message.chat.id]["nombre"])
    bot.send_message(message.chat.id, "¡Tus resultados ya están listos! \U0001F4DC")
    time.sleep(2)
    bot.send_message(message.chat.id, "¡Ya se están enviando! \U0001F680")
    doc = open('./Informe_IA.pdf', 'rb')
    bot.send_document(message.chat.id, doc)
    time.sleep(2)
    bot.send_message(message.chat.id, "¡Muchísimas gracias por confiar en mí! Espero que te haya servido de ayuda")
    time.sleep(1.5)
    bot.send_message(message.chat.id, "El objetivo es que descubras nuevos grados y consigas mucha más información para que puedas tomar la mejor decisión para ti")
    time.sleep(2.5)
    bot.send_message(message.chat.id, "También te recomiendo que le eches un vistazo a todos los grados que ofrece la UGR. ¡Pulsa en /planesEstudio para verlos!")
    time.sleep(2)
    bot.send_message(message.chat.id, "Por último, te invito a rellenar este [formulario](https://forms.gle/94zbKg9LGBFU8qeV9) que te llevará tres minutos y que ayudaría muchísimo a mi creador para mejorar mis funciones y para su investigación universitaria. \n\nGracias \U00002764", parse_mode="markdown")
    # eliminamos los datos en pos de ahorrar espacio consumido
    del usuarios[message.chat.id]
    del entradas[message.chat.id]

  # SI EL USUARIO DICE QUE AÑADE ALGO MÁS A SU REDACCIÓN
  else:
    bot.send_message(message.chat.id, "¡Genial!")
    time.sleep(1.5)
    bot.send_message(message.chat.id, "¡El trabajo por tu parte está hecho!")
    time.sleep(1.5)
    msg = bot.send_message(message.chat.id, "Ahora déjame un poco de tiempo para que mi Inteligencia Artificial pueda trabajar sobre lo que me has dicho")
    time.sleep(2)
    mono = "CAACAgEAAxkBAAJAumLjJDDtghIY6CpE_akxYixnpsLXAAIeAQACOA6CEUZYaNdphl79KQQ"
    spidercerdo = "CAACAgIAAxkBAAJBmGL5FHOssNZunVmLwERSLatiefPtAALUFQACldQRSoISm4Rl51_gKQQ"
    lista_sti = [mono, spidercerdo]
    numero_random = rd.randint(0, 1)
    bot.send_sticker(message.chat.id, lista_sti[numero_random])
    time.sleep(5)
    # UNIÓN DE LAS DOS ENTRADAS, TRATAMIENTO DE LA MISMA PARA SEPARARLA EN FRASES A TRAVÉS DE LA COMA Y LA I GRIEGA
    entrada1 = entradas[message.chat.id]["primera"]
    if entrada1[-1] == ".":
      entrada1 = entrada1[:-1]
    #print(entrada1)
    entrada2 = entradas[message.chat.id]["segunda"]
    if entrada2[-1] == ".":
      entrada2 = entrada2[:-1]
    #print(entrada2)
    entrada = entrada1 + ". " + entrada2
    #print(entrada)
    frases = entrada.split('.')
    frases_sin_espacios = []
    for frase in frases:
      frase = frase.strip()
      frases_sin_espacios.append(frase)
    frases_nuevas = []
    for frase in frases_sin_espacios:
      if "," in frase:
        frase_nueva = frase.split(",")
        for x in range(len(frase_nueva)):
          frases_nuevas.append(frase_nueva[x].strip())
      else:
        frases_nuevas.append(frase)

    lista_frase = []
    frases_y_sust = []
    lista_frases_y = []
    frases_resto = []
    frases_y = []
    frases = []

    for indice, frase in enumerate(frases_nuevas):
      lista_frase.append(frase.split())

    for indice_frase, frase in enumerate(lista_frase):
      for indice_palabra, palabra in enumerate(frase):
        if palabra == "y":
          lista_frase[indice_frase][indice_palabra] = "#"
          concatenado = " ".join(lista_frase[indice_frase])
          
    for indice_frase, frase in enumerate(lista_frase):
      concatenado = " ".join(lista_frase[indice_frase])
      frases_y_sust.append(concatenado)

    for indice, frase in enumerate(frases_y_sust):
      if "#" in frase:
        frase1 = frase.split("#")
        lista_frases_y.append(frase1)
      else:
        frases_resto.append(frase)

    for frase in lista_frases_y:
      indice = len(frase)
      for x in range(indice):
        frases_y.append(frase[x])

    for frase in frases_resto:
      frase = frase.strip()
      frases.append(frase)

    for frase in frases_y:
      frase = frase.strip()
      frases.append(frase)


    # CLASIFICACIÓN FRASES CON EL MODELO ENTRENADO, SACAR PORCENTAJES Y CLASIFICACIÓN SEGÚN IMPORTANCIA PORCENTUAL
    mayor_prob = {}
    menor_prob = {}
    
    for frase in frases:
      clasif = classify(frase)
      for x in range(7):
        prob = clasif[x][1] * 100
        if 2 >= prob >= 0:
          grado = clasif[x][0]
          try:
            if menor_prob[grado] != 0:
              menor_prob[grado] = menor_prob[grado] + 1
          except:
            menor_prob[grado] = 1
        elif prob > 2:
          grado = clasif[x][0]
          try:
            if mayor_prob[grado] != 0:
              mayor_prob[grado] = mayor_prob[grado] + 1
          except:
            mayor_prob[grado] = 1  

    mayor_prob_porcentajes = {}
    menor_prob_porcentajes = {}
    total = 0

    for nombre, valor in mayor_prob.items():
      total = total + valor
    for nombre, valor in mayor_prob.items():
      porcentaje = str(round((valor * 100) / total, 2)) + "%"
      mayor_prob_porcentajes[nombre] = porcentaje

    for nombre, valor in menor_prob.items():
      total = total + valor
    for nombre, valor in menor_prob.items():
      porcentaje = str(round((valor * 100) / total, 2)) + "%"
      menor_prob_porcentajes[nombre] = porcentaje
    
    # JSON GRADOS
    with open("./Cloudia_Bot/net_dnn/grados.json") as grados_json:
      grados = json.load(grados_json)
    
    # EXTRACCIÓN DE NOMBRES DE GRADOS Y ENLACES A PARTIR DE LA INFORMACIÓN QUE TENEMOS
    # Y EL JSON
    grados_codificados_mayor = []
    grados_codificados_menor = []
    porcentajes_mayor = []
    porcentajes_menor = []

    grados_reales_mayor = []
    enlaces_reales_mayor = []
    grados_reales_menor = []
    enlaces_reales_menor = []

    grupos_mayor_prob = []
    grupos_menor_prob = []

    # Separar porcentajes
    for x, y in mayor_prob_porcentajes.items():
      grados_codificados_mayor.append(x)
      porcentajes_mayor.append(y)

    for x, y in menor_prob_porcentajes.items():
      grados_codificados_menor.append(x)
      porcentajes_menor.append(y)

    for n in range(31):
      for x, y in grados.items():
        if y[n]["tag"] in grados_codificados_mayor:
          grados_mayor_prob = y[n]["nombre"]
          grupos_mayor_prob.append(grados_mayor_prob)
          enlaces_mayor_prob = y[n]["enlace"]
          for a in range(0, len(grados_mayor_prob)):
            grados_reales_mayor.append(grados_mayor_prob[a])
            enlaces_reales_mayor.append(enlaces_mayor_prob[a])
        if y[n]["tag"] in grados_codificados_menor:
          grados_menor_prob = y[n]["nombre"]
          grupos_menor_prob.append(grados_menor_prob)
          enlaces_menor_prob = y[n]["enlace"]
          for a in range(0, len(grados_menor_prob)):
            grados_reales_menor.append(grados_menor_prob[a])
            enlaces_reales_menor.append(enlaces_menor_prob[a])

    # limpieza de posibles grados repetidos
    grados_reales_mayor = list(dict.fromkeys(grados_reales_mayor))
    grados_reales_menor = list(dict.fromkeys(grados_reales_menor))

    # UNIÓN DE GRUPOS DE GRADOS CON SUS PORCENTAJES
    grupos_mayor_prob_limp = []
    grupos_menor_prob_limp = []
    for grupo in grupos_mayor_prob:
      porcentaje = None
      for guardar_porcentaje in porcentajes_mayor:
        porcentaje = guardar_porcentaje
      grupo = ", ".join(grupo)
      grupos_mayor_prob_limp.append(grupo)

    for grupo in grupos_menor_prob:
      porcentaje = None
      for guardar_porcentaje in porcentajes_menor:
        porcentaje = guardar_porcentaje
      
      grupo = ", ".join(grupo)
      grupos_menor_prob_limp.append(grupo)

    # PORCENTAJES LIMPIOS (FLOAT)
    porcentajes_limpios_mayor = []
    porcentajes_limpios_menor = []
    for x in porcentajes_mayor:
      x = x.replace("%", "")
      porcentajes_limpios_mayor.append(float(x))
    for x in porcentajes_menor:
      x = x.replace("%", "")
      porcentajes_limpios_menor.append(float(x))

    # MENSAJE DE FINALIZACIÓN DE LA PREDICCIÓN
    bot.send_message(message.chat.id, "¡Maravilloso! Ya he terminado")
    time.sleep(1.5)
    bot.send_message(message.chat.id, "Ya solo quedaría ponerme mis gafas y escribir el informe con todos los resultados")
    time.sleep(2)
    bot.send_message(message.chat.id, "Dame un poco de tiempo y lo terminaré en un segundo. Ahora te aviso")
    time.sleep(1.5)
    perro = "CAACAgIAAxkBAAJAvmLjJWtvklGYD0zFLJI7dboloENWAAKBAQACK15TC14KbD5sAAF4tCkE"
    pez = "CAACAgIAAxkBAAJAymLjJbQHOxuhAiO_kAX7SJ9B1mWzAAJsAgACVp29CioZXpzE0N6fKQQ"
    lista_sti = [perro, pez]
    numero_random = rd.randint(0, 1)
    bot.send_sticker(message.chat.id, lista_sti[numero_random])

    # PREPARACIÓN DE LAS LISTAS DE GRADOS QUE SE ENVIARÁN PARA REALIZAR EL GRÁFICO Y LA LISTA DEL INFORME
    # limpieza de aquellos grados que se repiten en los de menor probabilidad con los de mayor probabilidad
    grupos_menor_prob_limp_sin = []
    for grupo in grupos_menor_prob_limp:
      if grupo not in grupos_mayor_prob_limp:
        grupos_menor_prob_limp_sin.append(grupo)
    #---
    grados_reales_menor_limp_sin = []
    for grado in grados_reales_menor:
      if grado not in grados_reales_mayor:
        grados_reales_menor_limp_sin.append(grado)
    #---
    enlaces_reales_menor_limp_sin = []
    for enlace in enlaces_reales_menor:
      if enlace not in enlaces_reales_mayor:
        enlaces_reales_menor_limp_sin.append(enlace)

    # AÑADIR IMAGEN
    numero_imagen_str = str(cuenta_imagen["numero"])
    len_grupos_mayor = len(grupos_mayor_prob_limp)
    crear_imagen(f"gradoMayor{numero_imagen_str}.png", grupos_mayor_prob_limp, len_grupos_mayor, porcentajes_limpios_mayor)
    #crear_imagen(f"gradoMenor{numero_imagen_str}.png", grupos_menor_prob_limp_sin, len_grupos_menor, porcentajes_limpios_menor)
    cuenta_imagen["numero"] += 1
    time.sleep(7)

    # MÓDULO LATEX
    latex(f"gradoMayor{numero_imagen_str}.png", grados_reales_menor_limp_sin, enlaces_reales_menor_limp_sin, grados_reales_mayor, enlaces_reales_mayor, porcentajes_mayor, prediccion_ciencias_letras["matematicas"], prediccion_ciencias_letras["lengua"], usuarios[message.chat.id]["nombre"])
    time.sleep(2)
    bot.send_message(message.chat.id, "¡Tus resultados ya están listos! \U0001F4DC")
    time.sleep(1.5)
    bot.send_message(message.chat.id, "¡Ya se están enviando! \U0001F680")
    doc = open('./Informe_IA.pdf', 'rb')
    bot.send_document(message.chat.id, doc)
    time.sleep(2)
    bot.send_message(message.chat.id, "¡Muchísimas gracias por confiar en mí! Espero que te haya servido de ayuda")
    time.sleep(1.5)
    bot.send_message(message.chat.id, "El objetivo es que descubras nuevos grados y consigas mucha más información para que puedas tomar la mejor decisión para ti")
    time.sleep(2.5)
    bot.send_message(message.chat.id, "También te recomiendo que le eches un vistazo a todos los grados que ofrece la UGR. ¡Pulsa aquí /planesEstudio para verlos!")
    time.sleep(2)
    bot.send_message(message.chat.id, "Por último, te invito a rellenar este [formulario](https://forms.gle/94zbKg9LGBFU8qeV9) que te llevará tres minutos y que ayudaría muchísimo a mi creador para mejorar mis funciones y para su investigación universitaria", parse_mode="markdown")
    # eliminamos los datos en pos de ahorrar espacio consumido
    del usuarios[message.chat.id]
    del entradas[message.chat.id]

#--------------------Módulo 4: plan de estudios--------------------

#----------Web Scraping----------

URL = "https://www.ugr.es/estudiantes/grados"
html_text = requests.get(URL).text

soup = BeautifulSoup(html_text, "lxml")

titulos = soup.find_all("h2", class_ ="sidebar")
titulo_artes_humanidades = titulos[0].text
titulo_ciencias = titulos[1].text
titulo_ciencias_salud = titulos[2].text
titulo_ciencias_sociales_juridicas = titulos[3].text
titulo_ingenieria_arquitectura = titulos[4].text

listas = soup.find_all("ul")
artes_humanidades = listas[14] # Artes y Humnanidades
ciencias = listas[15] # Ciencias
ciencias_salud = listas[16] # Ciencias de la salud
ciencias_sociales_juridicas = listas[17] # Ciencias sociales y jurídicas
ingenieria_arquitectura = listas[18] # Ingeniería y arquitectura

dic_titulos = {}
ramas = []

for x in range(6):
  titulo = titulos[x].text
  ramas.append(titulo)
  dic_titulos[titulo] = None

n = 0
for x in range(14, 20):
  lista_grados = listas[x]
  dic_titulos[ramas[n]] = lista_grados
  n = n + 1

#----------Desarrollo----------

# CONSTANTES 
N_RES_PAG = 5 # número de resultados a mostrar en cada página
MAX_ANCHO_ROW = 8 # máximo botones por fila
DIR = {"busquedas": "busquedas"}
for key in DIR:
  try:
    os.mkdir(key)
  except:
    pass

@bot.callback_query_handler(func=lambda x:True)
def respuesta_botones_inline(call):
  """Gestiona las acciones de los botones callback_data"""
  cid = call.from_user.id
  mid = call.message.id
  if call.data == "cerrar":
    bot.delete_message(cid, mid)
    retorno(call.message)
    return
  datos = pickle.load(open(f"{DIR['busquedas']}{cid}_{mid}", "rb"))
  if call.data == "anterior":
    # si ya estamos en la primera página
    if datos["pag"] == 0:
      bot.answer_callback_query(call.id, "Ya estás en la primera página")
    else:
      datos["pag"] -= 1 # retrocedemos una página
      pickle.dump(datos, open(f"{DIR['busquedas']}{cid}_{mid}", "wb"))
      mostrar_pagina(datos["lista"], cid, datos["pag"], mid)
    return
  elif call.data == "posterior":
    # si ya estamos en la última página
    if datos["pag"] * N_RES_PAG + N_RES_PAG >= len(datos["lista"]):
      bot.answer_callback_query(call.id, "Ya estás en la última página")
    else:
      datos["pag"] += 1 # avanzamos una página
      pickle.dump(datos, open(f"{DIR['busquedas']}{cid}_{mid}", "wb"))
      mostrar_pagina(datos["lista"], cid, datos["pag"], mid)
    return

@bot.message_handler(commands=["planesEstudio"])
def start(message):
  """Bienvenida al módulo, explicación y recomendación"""
  markup = ReplyKeyboardRemove()
  bot.send_message(message.chat.id, "Genial! Consultemos los planes de estudios. Una vez elegida la rama, podrás navegar por un menú donde podrás avanzar de página \U000027a1, retroceder \U00002b05 y salir \U0000274c", reply_markup=markup)
  time.sleep(2)
  bot.send_message(message.chat.id, "¡OJO! \U0001F441 Una vez que entres en un grado. Te recomiendo que consultes las *guías docentes* de cada asignatura (enlace donde pone 'ver guía') para saber más sobre su desarrollo: objetivos de la asignatura, conocimientos previos que se recomiendan, cómo aprobar, y mucho más", parse_mode="markdown")
  time.sleep(1.5)
  msg = bot.send_message(message.chat.id, "Representa una información muy valiosa \U0001F31F para saber qué te encontrarás a lo largo de cada asignatura", parse_mode="markdown")
  time.sleep(2.5)
  rama_estudios(msg)

def rama_estudios(message):
  # definimos los botones
  markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón")
  markup_teclado.add(f"{titulo_artes_humanidades}", f"{titulo_ciencias}", f"{titulo_ciencias_salud}", f"{titulo_ciencias_sociales_juridicas}", f"{titulo_ingenieria_arquitectura}")#, f"{titulo_dobles_titulaciones}")
  # preguntamos
  msg = bot.send_message(message.chat.id, f"¿Qué rama de estudios quieres consultar?", reply_markup=markup_teclado)
  bot.register_next_step_handler(msg, scraping)

def scraping(message):
  """Web Scraping"""

  if normalize(message.text.lower()) == "artes y humanidades" or normalize(message.text.lower()) == "ciencias" or normalize(message.text.lower()) == "ciencias de la salud" or normalize(message.text.lower()) == "ciencias sociales y juridicas" or normalize(message.text.lower()) == "ingenieria y arquitectura":

    lista_titulos = []
  
    for titulo in dic_titulos:
      lista_titulos.append(titulo.lower())

      if normalize(message.text.lower()) == normalize(titulo.lower()):

        lista = []
        elementos = dic_titulos[titulo].find_all("a")
        for elemento in elementos:
          texto = elemento.text
          texto_separado = texto.split("en ")
          grado = texto_separado[1]
          enlace_fuente = elemento.get("href")
          enlace = f"https://www.ugr.es{enlace_fuente}"
          
          lista.append([grado, enlace])

        mostrar_pagina(lista, message.chat.id)

  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones dadas", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add(f"{titulo_artes_humanidades}", f"{titulo_ciencias}", f"{titulo_ciencias_salud}", f"{titulo_ciencias_sociales_juridicas}", f"{titulo_ingenieria_arquitectura}")
    msg = bot.send_message(message.chat.id, "¿Qué rama de estudios quieres consultar?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, scraping)

def mostrar_pagina(lista, cid, pag=0, mid=None):
  """Muestra el menú de grados universitarios con su enlace según la rama de conocimiento escogida por el usuario"""

  markup = InlineKeyboardMarkup(row_width=MAX_ANCHO_ROW) # número de botones en cada fila (3 por defecto)
  b_anterior = InlineKeyboardButton("\U00002b05", callback_data="anterior")
  b_cerrar = InlineKeyboardButton("\U0000274c", callback_data="cerrar")
  b_posterior = InlineKeyboardButton("\U000027a1", callback_data="posterior")
  inicio = pag*N_RES_PAG # nº resultado inicio página en curso
  fin = inicio + N_RES_PAG # nº resultado fin página en curso
  # controlamos que el último resultado de la página no supere el total de resultados
  if fin > len(lista):
    fin = len(lista)
  mensaje = f"<i>Resultados {inicio + 1} - {fin} de {len(lista)}</i>\n\n"
  n = 1
  botones = [] # botones de índice en la página
  for item in lista[inicio:fin]:
    botones.append(InlineKeyboardButton(str(n), url=item[1]))
    mensaje += f"[<b>{n}</b>] {item[0]}\n"
    n +=1
  markup.add(*botones)
  markup.row(b_anterior, b_cerrar, b_posterior)
  if mid:
    bot.edit_message_text(mensaje, cid, mid, reply_markup=markup, parse_mode="html", disable_web_page_preview=True)
  else:
    res = bot.send_message(cid, mensaje, reply_markup=markup, parse_mode="html", disable_web_page_preview=True)
    mid = res.message_id
    datos = {"pag": 0, "lista": lista}
    pickle.dump(datos, open(f"{DIR['busquedas']}{cid}_{mid}", "wb"))

def retorno(message):
  # definimos los botones
  markup = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón", resize_keyboard=True)
  markup.add("Sí", "No")
  # preguntamos
  msg = bot.send_message(message.chat.id, f"¿Quieres volver a consultar los planes de estudio?", reply_markup=markup)
  bot.register_next_step_handler(msg, despedida)

def despedida(message):
  if normalize(message.text.lower()) == "si":
    msg = bot.send_message(message.chat.id, "Genial!")
    rama_estudios(msg)
  elif normalize(message.text.lower()) == "no":
    markup = ReplyKeyboardRemove()
    bot.send_message(message.chat.id, "¡Perfecto, muchas gracias!")
    time.sleep(2)
    bot.send_message(message.chat.id, "Por si no lo has hecho ya, te invito a rellenar este [formulario](https://forms.gle/94zbKg9LGBFU8qeV9) que te llevará tres minutos y que ayudaría muchísimo a mi creador a mejorar mis funciones y su investigación universitaria. \n\nGracias \U00002764", parse_mode="markdown", reply_markup=markup)
  else:
    bot.send_message(message.chat.id, "Perdón, no te he entendido \U0001F974 \nPulsa un *botón* para elegir una de las opciones o escribe *'Sí'* o *'No'*", parse_mode="markdown")
    # definimos los botones
    markup_teclado = ReplyKeyboardMarkup(one_time_keyboard=True, input_field_placeholder="Pulsa un botón para elegir una opción", resize_keyboard=True)
    markup_teclado.add("Sí", "No")
    msg = bot.send_message(message.chat.id, "¿Quieres volver a consultar los planes de estudio?", reply_markup=markup_teclado, parse_mode="markdown")
    bot.register_next_step_handler(msg, despedida)

#--------------------INICIAR BOT--------------------

bot_vivo = True

# iniciamos el bot
if bot_vivo == True:
  # configuración de los comandos de uso en el bot  
  bot.set_my_commands([telebot.types.BotCommand("/start", "Empezar de cero la conversación si el bot no responde")])
  print("Bot vivo")
  bot.infinity_polling()
  print("Bot muerto")  