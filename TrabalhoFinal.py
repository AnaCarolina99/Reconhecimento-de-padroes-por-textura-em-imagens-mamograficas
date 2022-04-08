"""
Membros do Grupo:
Ana Carolina Medeiros Gonçalves
Ana Luiza Pacheco Leite
Caio Igor Vasconcelos Nazareth
"""
import shutil
import tkinter as tk
import cv2
import glob
import warnings
from tkinter import *
from tkinter import Scrollbar
from tkinter import ttk
from tkinter import filedialog as dlg
from tkinter import filedialog
import PIL
from PIL import Image,ImageTk, ImageOps
import PIL.ImageTk
import numpy as np
import os.path
from numpy import asarray
import random
from random import sample
from random import seed
from random import randint
import time
import skimage as sk
from skimage.feature import greycomatrix, greycoprops
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_uint
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.cluster import entropy
from sklearn.metrics import confusion_matrix
from matplotlib import image
import matplotlib.pyplot as plt2
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,models,metrics,callbacks
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import load_model
from tensorflow.keras.layers import Activation, Conv2D, Flatten, Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import itertools
import os
import pandas as pd
import seaborn as sn
import tensorflow as tf

warnings.simplefilter(action='ignore',category=FutureWarning)

treinou = False
leuArquivos = False
carregouImagem = False
equalizou = False

#criar interface
my_window = Tk()
my_window.title("Processamento de imagens")
my_window.geometry("1000x1000")

my_menu = Menu(my_window)
my_window.config(menu=my_menu)

#criar canvas
my_canvas = Canvas(my_window, width=1366,height=720,background='white')

recortar = False
retangulos = []
areas = []
click_Ret=0

def abrirImagem():
    global my_image, recortar, retangulos, areas, classificarSel, fname5
    global copy
    global fname, fname3
    filename = filedialog.askopenfilename(initialdir='Downloads\PI', title="Arquivos",filetypes=(("png files","*.png"),("all files","*.*")))
    my_canvas.delete("all")
    recortar = False
    classificarSel = False
    retangulos = []
    areas = []
    #abrir image na interface
    fname = filename
    fname3 = filename
    fname5 = filename
    my_image = Image.open(filename)
    save = "save.png"
    imgem = my_image.save(save)
    copy = ImageTk.PhotoImage(my_image)
    global carregouImagem
    carregouImagem = True
    #criar canvas
    my_canvas.config(width=copy.width(), height=copy.height())
    my_canvas.pack(expand=True)
    my_canvas.my_image = copy
    my_canvas.create_image(0,0,image=copy, anchor=tk.NW)
    my_canvas.place(x=0,y=0)

lista_imagensB1=[]
lista_imagensB2=[]
lista_imagensB3=[]
lista_imagensB4=[]

lista_imagensSave1=[]
lista_imagensSave2=[]
lista_imagensSave3=[]
lista_imagensSave4=[]

lista_imagensArq1=[]
lista_imagensArq2=[]
lista_imagensArq3=[]
lista_imagensArq4=[]

def hasElemento(rand, lista):
    i = 0
    while i < len(lista):
        if rand == lista[i]:
            return True
        i+=1
    return False

def treinamento():

    global test_generator, train_generator
    
    if leuArquivos:
        tempoInicio = time.time()
        dir1Teste = 'Teste/1'
        dir2Teste = 'Teste/2'
        dir3Teste = 'Teste/3'
        dir4Teste = 'Teste/4'

        treinado = 'Treinados/'

        dir1Trei = 'Treinamento/1'
        dir2Trei = 'Treinamento/2'
        dir3Trei = 'Treinamento/3'
        dir4Trei = 'Treinamento/4'

        dir1Vali = 'Validacao/1'
        dir2Vali = 'Validacao/2'
        dir3Vali = 'Validacao/3'
        dir4Vali = 'Validacao/4'
        
        for file in os.scandir(dir1Teste):
            os.remove(file.path)
        
        for file in os.scandir(dir2Teste):
            os.remove(file.path)
        
        for file in os.scandir(dir3Teste):
            os.remove(file.path)
        
        for file in os.scandir(dir4Teste):
            os.remove(file.path)
        
        for file in os.scandir(dir1Trei):
            os.remove(file.path)

        for file in os.scandir(dir2Trei):
            os.remove(file.path)

        for file in os.scandir(dir3Trei):
            os.remove(file.path)

        for file in os.scandir(dir4Trei):
            os.remove(file.path)

        for file in os.scandir(dir1Vali):
            os.remove(file.path)

        for file in os.scandir(dir2Vali):
            os.remove(file.path)

        for file in os.scandir(dir3Vali):
            os.remove(file.path)
        
        for file in os.scandir(dir4Vali):
            os.remove(file.path)
        
        for file in os.scandir(treinado):
            os.remove(file.path)
        
        listaTr1 = []
        listaTrRand1 = []
        num = len(lista_imagensB1) * 0.75
        numValid = num * 0.5
        while num > 0:
            rand = random.randint(0, 99)
            if not(hasElemento(rand, listaTrRand1)):
                listaTrRand1.append(rand)
                listaTr1.append(lista_imagensB1[rand])
                lista_imagensSave1[rand].save('Treinamento/1/' + lista_imagensArq1[rand])
                if numValid > 1:
                    lista_imagensSave1[rand].save('Validacao/1/' + lista_imagensArq1[rand])
                    numValid -= 1
                num -= 1

        listaTe1 = []
        num = len(lista_imagensB1) * 0.25
        numValid = num * 0.5
        while num > 0:
            rand = random.randint(0, 99)
            if not(hasElemento(rand, listaTrRand1)):
                listaTrRand1.append(rand)
                listaTe1.append(lista_imagensB1[rand])
                lista_imagensSave1[rand].save('Teste/1/' + lista_imagensArq1[rand])
                if numValid > 0:
                    lista_imagensSave1[rand].save('Validacao/1/' + lista_imagensArq1[rand])
                    numValid -= 1
                num-=1

        listaTr2 = []
        listaTrRand2 = []
        num = len(lista_imagensB2) * 0.75
        numValid = num * 0.5
        while num > 0:
            rand = random.randint(0, 99)
            if not(hasElemento(rand, listaTrRand2)):
                listaTrRand2.append(rand)
                listaTr2.append(lista_imagensB2[rand])
                lista_imagensSave2[rand].save('Treinamento/2/' + lista_imagensArq2[rand])
                if numValid > 1:
                    lista_imagensSave2[rand].save('Validacao/2/' + lista_imagensArq2[rand])
                    numValid -= 1
                num-=1

        listaTe2 = []
        num = len(lista_imagensB2) * 0.25
        numValid = num * 0.5
        while num > 0:
            rand = random.randint(0, 99)
            if not(hasElemento(rand, listaTrRand2)):
                listaTrRand2.append(rand)
                listaTe2.append(lista_imagensB2[rand])
                lista_imagensSave2[rand].save('Teste/2/' + lista_imagensArq2[rand])
                if numValid > 0:
                    lista_imagensSave2[rand].save('Validacao/2/' + lista_imagensArq2[rand])
                    numValid -= 1
                num-=1

        listaTr3 = []
        listaTrRand3 = []
        num = len(lista_imagensB3) * 0.75
        numValid = num * 0.5
        while num > 0:
            rand = random.randint(0, 99)
            if not(hasElemento(rand, listaTrRand3)):
                listaTrRand3.append(rand)
                listaTr3.append(lista_imagensB3[rand])
                lista_imagensSave3[rand].save('Treinamento/3/' + lista_imagensArq3[rand])
                if numValid > 1:
                    lista_imagensSave3[rand].save('Validacao/3/' + lista_imagensArq3[rand])
                    numValid -= 1
                num-=1

        listaTe3 = []
        num = len(lista_imagensB3) * 0.25
        numValid = num * 0.5
        while num > 0:
            rand = random.randint(0, 99)
            if not(hasElemento(rand, listaTrRand3)):
                listaTrRand3.append(rand)
                listaTe3.append(lista_imagensB3[rand])
                lista_imagensSave3[rand].save('Teste/3/' + lista_imagensArq3[rand])
                if numValid > 0:
                    lista_imagensSave3[rand].save('Validacao/3/' + lista_imagensArq3[rand])
                    numValid -= 1
                num-=1

        listaTr4 = []
        listaTrRand4 = []
        num = len(lista_imagensB4) * 0.75
        numValid = num * 0.5
        while num > 0:
            rand = random.randint(0, 99)
            if not(hasElemento(rand, listaTrRand4)):
                listaTrRand4.append(rand)
                listaTr4.append(lista_imagensB4[rand])
                lista_imagensSave4[rand].save('Treinamento/4/' + lista_imagensArq4[rand])
                if numValid > 1:
                    lista_imagensSave4[rand].save('Validacao/4/' + lista_imagensArq4[rand])
                    numValid -= 1
                num-=1

        listaTe4 = []
        num = len(lista_imagensB4) * 0.25
        numValid = num * 0.5
        while num > 0:
            rand = random.randint(0, 99)
            if not(hasElemento(rand, listaTrRand4)):
                listaTrRand4.append(rand)
                listaTe4.append(lista_imagensB4[rand])
                lista_imagensSave4[rand].save('Teste/4/' + lista_imagensArq4[rand])
                if numValid > 0:
                    lista_imagensSave4[rand].save('Validacao/4/' + lista_imagensArq4[rand])
                    numValid -= 1
                num-=1
        
        train_path = 'Treinamento/'
        valid_path = 'Validacao/'
        test_path = 'Teste/'

        train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=train_path,target_size=(224,224), classes=['1','2','3','4'], batch_size=5)
        valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=valid_path,target_size=(224,224), classes=['1','2','3','4'], batch_size=5)
        test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=test_path,target_size=(224,224), classes=['1','2','3','4'], batch_size=5, shuffle=False)
        
        assert train_batches.n == 300
        assert valid_batches.n == 200
        assert test_batches.n == 100
        assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 4

        x,y = test_batches.next()
        x.shape

        base_model = ResNet50(include_top=False, weights='imagenet')
        x = base_model.output
        x= GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(train_batches.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_batches,
                                epochs=3)

        valid_batches.class_indices
        {'1':0,'2':1,'3':2,'4':3}

        dir_path = 'Teste/'
        global img
        for dir in [1,2,3,4]:
            for i in glob.glob(f"{dir_path}/{dir}/*"):
                img = image.load_img(i)
                X = image.img_to_array(img)
                X = np.expand_dims(X, axis = 0)
                images = np.vstack([X])
                val = model.predict(images)

        model.save('Treinados/modelo.h5')

        test_loss, test_acc = model.evaluate(test_batches,verbose=2)

        nb_samples = len(test_batches)
        y_prob=[]
        y_act=[]
        predicted_class = []
        actual_class = []
        test_batches.reset()
        for _ in range(nb_samples):
            X_test,Y_test = test_batches.next()
            y_prob.append(model.predict(X_test))
            y_act.append(Y_test)

        maiorProb1 = -1
        maiorAct1 = -1
        maiorProb2 = -1
        maiorAct2 = -1
        maiorProb3 = -1
        maiorAct3 = -1
        maiorProb4 = -1
        maiorAct4 = -1
        indexI1 = 0
        indexI2 = 0
        indexI3 = 0
        indexI4 = 0
        indexJ1 = 0
        indexJ2 = 0
        indexJ3 = 0
        indexJ4 = 0
        indiceI = 0
        while indiceI < len(y_prob):
            indiceJ = 0
            while indiceJ < len(y_prob[indiceI]):
                if maiorProb1 < 0 or maiorProb1 < y_prob[indiceI][indiceJ][0]:
                    maiorProb1 = y_prob[indiceI][indiceJ][0]
                    indexI1 = indiceI
                    indexJ1 = indiceJ
                if maiorProb2 < 0 or maiorProb2 < y_prob[indiceI][indiceJ][1]:
                    maiorProb2 = y_prob[indiceI][indiceJ][1]
                    indexI2 = indiceI
                    indexJ2 = indiceJ
                if maiorProb3 < 0 or maiorProb3 < y_prob[indiceI][indiceJ][2]:
                    maiorProb3 = y_prob[indiceI][indiceJ][2]
                    indexI3 = indiceI
                    indexJ3 = indiceJ
                if maiorProb4 < 0 or maiorProb4 < y_prob[indiceI][indiceJ][3]:
                    maiorProb4 = y_prob[indiceI][indiceJ][3]
                    indexI4 = indiceI
                    indexJ4 = indiceJ
                indiceJ+=1
            indiceI+=1

        global matriz_y_prob
        texto = ""
        matriz_y_prob = []
        vetAux = []
        vetAux.append(maiorProb1)
        vetAux.append(y_prob[indexI1][indexJ1][1])
        vetAux.append(y_prob[indexI1][indexJ1][2])
        vetAux.append(y_prob[indexI1][indexJ1][3])

        cont = 0
        while cont < len(vetAux):
            texto = texto + str(vetAux[cont]) + "\t"
            cont += 1
        texto = texto + "\n"

        matriz_y_prob.append(vetAux)

        vetAux = []
        vetAux.append(y_prob[indexI2][indexJ2][0])
        vetAux.append(maiorProb2)
        vetAux.append(y_prob[indexI2][indexJ2][2])
        vetAux.append(y_prob[indexI2][indexJ2][3])

        cont = 0
        while cont < len(vetAux):
            texto = texto + str(vetAux[cont]) + "\t"
            cont += 1
        texto = texto + "\n"

        matriz_y_prob.append(vetAux)

        vetAux = []
        vetAux.append(y_prob[indexI3][indexJ3][0])
        vetAux.append(y_prob[indexI3][indexJ3][1])
        vetAux.append(maiorProb3)
        vetAux.append(y_prob[indexI3][indexJ3][3])

        cont = 0
        while cont < len(vetAux):
            texto = texto + str(vetAux[cont]) + "\t"
            cont += 1
        texto = texto + "\n"

        matriz_y_prob.append(vetAux)

        vetAux = []
        vetAux.append(y_prob[indexI4][indexJ4][0])
        vetAux.append(y_prob[indexI4][indexJ4][1])
        vetAux.append(y_prob[indexI4][indexJ4][2])
        vetAux.append(maiorProb4)

        cont = 0
        while cont < len(vetAux):
            texto = texto + str(vetAux[cont]) + "\t"
            cont += 1
        texto = texto + "\n"

        matriz_y_prob.append(vetAux)
        with open('Treinados/matrizConfusao.txt', 'w') as ppd:
            ppd.write(texto)
        global treinou
        treinou = True
        tempo = time.time() - tempoInicio
        tempo = (int)(tempo)
        minutos = tempo//60
        segundos = tempo - (minutos * 60)
        newWindow = Toplevel(my_window)
        newWindow.title("Tempo de execução")
        newWindow.geometry("500x100")
        Label(newWindow, 
         text = "Treinamento finalizado em " + str(minutos) + " minutos e " + str(segundos) + " segundos").pack()
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("300x100")
        Label(newWindow, 
         text = "É necessário ler os diretórios de entrada\nantes de executar esta função").pack()

def classificar():
    global fname5
    tempoInicio = time.time()
    verificaModelo = os.path.exists('Treinados/modelo.h5')
    if (treinou or verificaModelo) and carregouImagem:
        model = tf.keras.models.load_model('Treinados/modelo.h5')
        img = image.load_img(fname5)
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis = 0)
        images = np.vstack([X])
        val = model.predict(images)
        index = np.argmax(val[0])
        newWindow = Toplevel(my_window)
        newWindow.title("Classificação")
        newWindow.geometry("450x100")
        strResp = ""
        if val[0][index] == val[0][0]:
            strResp = "BIRADS 1"
        elif val[0][index] == val[0][1]:
            strResp = "BIRADS 2"
        elif val[0][index] == val[0][2]:
            strResp = "BIRADS 3"
        elif val[0][index] == val[0][3]:
            strResp = "BIRADS 4"
        tempo = time.time() - tempoInicio
        Label(newWindow, 
            text = "Esta imagem é: " + strResp + "\nClassificação realizada em " + str(tempo) + " segundos").pack()
    elif carregouImagem:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário treinar a rede antes\nde executar esta função").pack()
    elif treinou or verificaModelo:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("350x100")
        Label(newWindow, 
         text = "É necessário treinar a rede e carregar uma imagem\nantes de executar esta função").pack()


def matrizConfusao():
    global matriz_y_prob
    verificaMatriz = os.path.exists('Treinados/matrizConfusao.txt')
    if treinou:
        newWindow = Tk()
        newWindow.title('Matriz de confusão')
        newWindow.geometry("550x250")
        tv = ttk.Treeview(newWindow)
        tv['columns'] = ("BIRADS 1","BIRADS 2","BIRADS 3","BIRADS 4")

        tv.column("#0", width=0, stretch=NO)
        tv.column("BIRADS 1", anchor=CENTER, width=120, minwidth=120)
        tv.column("BIRADS 2", anchor=CENTER, width=120, minwidth=120)
        tv.column("BIRADS 3", anchor=CENTER, width=120, minwidth=120)
        tv.column("BIRADS 4", anchor=CENTER, width=120, minwidth=120)

        tv.heading("#0", text="", anchor=CENTER)
        tv.heading("BIRADS 1", text="BIRADS 1", anchor=CENTER)
        tv.heading("BIRADS 2", text="BIRADS 2", anchor=CENTER)
        tv.heading("BIRADS 3", text="BIRADS 3", anchor=CENTER)
        tv.heading("BIRADS 4", text="BIRADS 4", anchor=CENTER)
        
        i = 0
        while i < len(matriz_y_prob):
            tv.insert(parent='', index='end',iid=i, text="", values=(matriz_y_prob[i][0], matriz_y_prob[i][1],matriz_y_prob[i][2],matriz_y_prob[i][3]))
            i += 1
        tv.pack(pady=20)
        newWindow.mainloop()
    elif verificaMatriz:
        with open('Treinados/matrizConfusao.txt', 'r') as arquivo:
            dados = arquivo.read()
        linha = dados.split("\n")
        cont = 0
        matriz_y_prob = []
        while cont < len(linha)-1:
            coluna = linha[cont].split("\t")
            contCol = 0
            vetAux = []
            while contCol < len(coluna)-1:
                vetAux.append(coluna[contCol])
                contCol += 1
            matriz_y_prob.append(vetAux)
            cont += 1
        
        newWindow = Tk()
        newWindow.title('Matriz de confusão')
        newWindow.geometry("550x250")
        tv = ttk.Treeview(newWindow)
        tv['columns'] = ("BIRADS 1","BIRADS 2","BIRADS 3","BIRADS 4")

        tv.column("#0", width=0, stretch=NO)
        tv.column("BIRADS 1", anchor=CENTER, width=120, minwidth=120)
        tv.column("BIRADS 2", anchor=CENTER, width=120, minwidth=120)
        tv.column("BIRADS 3", anchor=CENTER, width=120, minwidth=120)
        tv.column("BIRADS 4", anchor=CENTER, width=120, minwidth=120)

        tv.heading("#0", text="", anchor=CENTER)
        tv.heading("BIRADS 1", text="BIRADS 1", anchor=CENTER)
        tv.heading("BIRADS 2", text="BIRADS 2", anchor=CENTER)
        tv.heading("BIRADS 3", text="BIRADS 3", anchor=CENTER)
        tv.heading("BIRADS 4", text="BIRADS 4", anchor=CENTER)
        
        i = 0
        while i < len(matriz_y_prob):
            tv.insert(parent='', index='end',iid=i, text="", values=(matriz_y_prob[i][0], matriz_y_prob[i][1],matriz_y_prob[i][2],matriz_y_prob[i][3]))
            i += 1
        tv.pack(pady=20)
        newWindow.mainloop()
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário treinar a rede antes\nde executar esta função").pack()

def lerDiretorio():
    global lista_imagensB1, lista_imagensB2, lista_imagensB3, lista_imagensB4

    cont = 0

    path1 = "imagens/1/*.*"
    path2 = "imagens/2/*.*"
    path3 = "imagens/3/*.*"
    path4 = "imagens/4/*.*"

    imagem = 0

    for arq in glob.glob(path1):
        b1 = cv2.imread(arq)
        lista_imagensB1.append(b1)
        img = Image.open(arq)
        lista_imagensSave1.append(img)
        lista_imagensArq1.append(str(imagem) +'.png')
        imagem+=1
        cont = cont + 1

    for arq in glob.glob(path2):
        b2 = cv2.imread(arq)
        img = Image.open(arq)
        lista_imagensSave2.append(img)
        lista_imagensArq2.append(str(imagem) + '.png')
        imagem+=1
        lista_imagensB2.append(b2)
        cont = cont + 1

    for arq in glob.glob(path3):
        b3 = cv2.imread(arq)
        img = Image.open(arq)
        lista_imagensSave3.append(img)
        lista_imagensArq3.append(str(imagem) + '.png')
        imagem+=1
        lista_imagensB3.append(b3)
        cont = cont + 1

    for arq in glob.glob(path4):
        b4 = cv2.imread(arq)
        img = Image.open(arq)
        lista_imagensSave4.append(img)
        lista_imagensArq4.append(str(imagem) + '.png')
        imagem+=1
        lista_imagensB4.append(b4)
        cont = cont + 1
    global leuArquivos
    leuArquivos = True


def default():
    global my_image, recortar, retangulos, areas, classificarSel
    global copy, fname5
    global fname3
    if carregouImagem:
        my_canvas.delete("all")
        recortar = False
        classificarSel = False
        retangulos = []
        areas = []
        my_image = Image.open(fname3)
        save = "save.png"
        fname5 = fname3
        imgem = my_image.save(save)
        copy = ImageTk.PhotoImage(my_image)
        my_canvas.config(width=copy.width(), height=copy.height())
        my_canvas.pack(expand=True)
        my_canvas.my_image = copy
        my_canvas.create_image(0,0,image=copy, anchor=tk.NW)
        my_canvas.place(x=0,y=0)
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def zoomOUT():
    global my_image, recortar, retangulos, areas, classificarSel, fname5
    global copy
    if carregouImagem:
        my_canvas.delete("all")
        recortar = False
        classificarSel = False
        retangulos = []
        areas2=[]
        areas3=[]
        copy = ImageTk.PhotoImage(my_image)
        width = copy.width()*0.75
        height = copy.height()*0.75
        my_image = my_image.resize((int(width),int (height)), Image.ANTIALIAS)
        save = "save.png"
        fname5 = "save.png"
        imgem = my_image.save(save)
        copy = ImageTk.PhotoImage(my_image)
        my_canvas.config(width=copy.width(), height=copy.height())
        my_canvas.pack(expand=True)
        my_canvas.my_image = copy
        my_canvas.create_image(0,0,image=copy, anchor=tk.NW)
        my_canvas.place(x=0,y=0)
        if(len(areas) > 0):
            for i in range(0,len(areas)):
                for j in range(0,len(areas[i])):
                    areas2.append(int(areas[i][j] * 0.75))
                areas3.append(areas2)
                retangulos.append(my_canvas.create_rectangle(areas2,outline='blue',width=5))
                areas2=[]
            areas=areas3
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def zoomIN():
    global my_image, recortar, retangulos, areas, classificarSel, fname5
    global copy
    if carregouImagem:
        my_canvas.delete("all")
        recortar = False
        classificarSel = False
        retangulos = []
        areas2=[]
        areas3=[]
        copy = ImageTk.PhotoImage(my_image)
        width = copy.width()*1.25
        height = copy.height()*1.25
        my_image = my_image.resize((int(width),int (height)), Image.ANTIALIAS)
        save = "save.png"
        fname5 = "save.png"
        imgem = my_image.save(save)
        copy = ImageTk.PhotoImage(my_image)
        my_canvas.config(width=copy.width(), height=copy.height())
        my_canvas.pack(expand=True)
        my_canvas.my_image = copy
        my_canvas.create_image(0,0,image=copy, anchor=tk.NW)
        my_canvas.place(x=0,y=0)
        if(len(areas) > 0):
            for i in range(0,len(areas)):
                for j in range(0,len(areas[i])):
                    areas2.append(int(areas[i][j] * 1.25))
                areas3.append(areas2)
                retangulos.append(my_canvas.create_rectangle(areas2,outline='blue',width=5))
                areas2=[]
            areas=areas3
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def calcular0():
    global fname, fname5
    tempoInicio = time.time()
    if carregouImagem:
        image = imread(fname)
        image_gray = rgb2gray(image)
        resultado = np.zeros(6)
        newWindow = Toplevel(my_window)
        newWindow.title("Características da imagem")
        newWindow.geometry("450x500")
        i = 0
        while i < len(resultado)-1:
            matrizResultante   = greycomatrix(image_gray,[2**i],[0])
            resultado[0] = greycoprops(matrizResultante, 'homogeneity')
            resultado[1] = entropy(image_gray)
            resultado[2] = greycoprops(matrizResultante, 'energy')
            resultado[3] = greycoprops(matrizResultante, 'contrast')
            resultado[4] = greycoprops(matrizResultante, 'dissimilarity')
            resultado[5] = greycoprops(matrizResultante, 'correlation')
            Label(newWindow, 
                text = "Matriz de co-ocorrência com raio " + str(2**i) +
                "\nHomogeneidade: " + str("{:.4f}".format(resultado[0])) +
                "\t\tEntropia: " + str("{:.4f}".format(resultado[1])) +
                "\nEnergia: " + str("{:.4f}".format(resultado[2])) +
                "\t\tContraste: " + str("{:.4f}".format(resultado[3])) +
                "\nDissimilaridade: " + str("{:.4f}".format(resultado[4])) +
                "\t\tCorrelação: " + str("{:.4f}".format(resultado[5])) + "\n"
                ).pack()
            i+=1
        tempo = time.time() - tempoInicio
        Label(newWindow,text ="Tempo: "  + str(tempo) + " segundos").pack()
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def calcular45():
    global fname, fname5
    tempoInicio = time.time()
    if carregouImagem:
        image = imread(fname)
        image_gray = rgb2gray(image)
        resultado = np.zeros(6)
        newWindow = Toplevel(my_window) 
        newWindow.title("Características da imagem")
        newWindow.geometry("450x500")
        i = 0
        while i < len(resultado)-1:
            matrizResultante = greycomatrix(image_gray,[2**i],[45])
            resultado[0] = greycoprops(matrizResultante, 'homogeneity')
            resultado[1] = entropy(image_gray)
            resultado[2] = greycoprops(matrizResultante, 'energy')
            resultado[3] = greycoprops(matrizResultante, 'contrast')
            resultado[4] = greycoprops(matrizResultante, 'dissimilarity')
            resultado[5] = greycoprops(matrizResultante, 'correlation')
            Label(newWindow, 
                text = "Matriz de co-ocorrência com raio " + str(2**i) +
                "\nHomogeneidade: " + str("{:.4f}".format(resultado[0])) +
                "\t\tEntropia: " + str("{:.4f}".format(resultado[1])) +
                "\nEnergia: " + str("{:.4f}".format(resultado[2])) +
                "\t\tContraste: " + str("{:.4f}".format(resultado[3])) +
                "\nDissimilaridade: " + str("{:.4f}".format(resultado[4])) +
                "\t\tCorrelação: " + str("{:.4f}".format(resultado[5])) + "\n"
                ).pack()
            i += 1
        tempo = time.time() - tempoInicio
        Label(newWindow,text ="Tempo: "  + str(tempo) + " segundos").pack()
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def calcular90():
    global fname, fname5
    tempoInicio = time.time()
    if carregouImagem:
        image = imread(fname)
        image_gray = rgb2gray(image)
        resultado = np.zeros(6)
        newWindow = Toplevel(my_window) 
        newWindow.title("Características da imagem")
        newWindow.geometry("450x500") 
        i = 0
        while i < len(resultado)-1:
            matrizResultante = greycomatrix(image_gray,[2**i],[90])
            resultado[0] = greycoprops(matrizResultante, 'homogeneity')
            resultado[1] = entropy(image_gray)
            resultado[2] = greycoprops(matrizResultante, 'energy')
            resultado[3] = greycoprops(matrizResultante, 'contrast')
            resultado[4] = greycoprops(matrizResultante, 'dissimilarity')
            resultado[5] = greycoprops(matrizResultante, 'correlation')
            Label(newWindow, 
                text = "Matriz de co-ocorrência com raio " + str(2**i) +
                "\nHomogeneidade: " + str("{:.4f}".format(resultado[0])) +
                "\t\tEntropia: " + str("{:.4f}".format(resultado[1])) +
                "\nEnergia: " + str("{:.4f}".format(resultado[2])) +
                "\t\tContraste: " + str("{:.4f}".format(resultado[3])) +
                "\nDissimilaridade: " + str("{:.4f}".format(resultado[4])) +
                "\t\tCorrelação: " + str("{:.4f}".format(resultado[5])) + "\n"
                ).pack()
            i += 1
        tempo = time.time() - tempoInicio
        Label(newWindow,text ="Tempo: "  + str(tempo) + " segundos").pack()
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def calcular135():
    global fname, fname5
    tempoInicio = time.time()
    if carregouImagem:
        image = imread(fname)
        image_gray = rgb2gray(image)
        resultado = np.zeros(6)
        newWindow = Toplevel(my_window) 
        newWindow.title("Características da imagem") 
        newWindow.geometry("450x500") 
        i = 0
        while i < len(resultado)-1:
            matrizResultante = greycomatrix(image_gray,[2**i],[135])
            resultado[0] = greycoprops(matrizResultante, 'homogeneity')
            resultado[1] = entropy(image_gray)
            resultado[2] = greycoprops(matrizResultante, 'energy')
            resultado[3] = greycoprops(matrizResultante, 'contrast')
            resultado[4] = greycoprops(matrizResultante, 'dissimilarity')
            resultado[5] = greycoprops(matrizResultante, 'correlation')
            Label(newWindow, 
                text = "Matriz de co-ocorrência com raio " + str(2**i) +
                "\nHomogeneidade: " + str("{:.4f}".format(resultado[0])) +
                "\t\tEntropia: " + str("{:.4f}".format(resultado[1])) +
                "\nEnergia: " + str("{:.4f}".format(resultado[2])) +
                "\t\tContraste: " + str("{:.4f}".format(resultado[3])) +
                "\nDissimilaridade: " + str("{:.4f}".format(resultado[4])) +
                "\t\tCorrelação: " + str("{:.4f}".format(resultado[5])) + "\n"
                ).pack()
            i += 1
        tempo = time.time() - tempoInicio
        Label(newWindow,text ="Tempo: "  + str(tempo) + " segundos").pack()
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def calcularSel0():
    global fname, classificarSel, fname5
    tempoInicio = time.time()
    if classificarSel:
        if carregouImagem:
            imgCanvas = Image.open(fname)
            cropped_img = imgCanvas.crop(areas[-1])
            crop_img = cropped_img.save("selecao.png")
            image = imread("selecao.png")
            image_gray = rgb2gray(image)
            resultado = np.zeros(6)
            newWindow = Toplevel(my_window) 
            newWindow.title("Características da região selecionada") 
            newWindow.geometry("450x500") 
            i = 0
            while i < len(resultado)-1:
                matrizResultante = greycomatrix(image_gray,[2**i],[0])
                resultado[0] = greycoprops(matrizResultante, 'homogeneity')
                resultado[1] = entropy(image_gray)
                resultado[2] = greycoprops(matrizResultante, 'energy')
                resultado[3] = greycoprops(matrizResultante, 'contrast')
                resultado[4] = greycoprops(matrizResultante, 'dissimilarity')
                resultado[5] = greycoprops(matrizResultante, 'correlation')
                Label(newWindow, 
                    text = "Matriz de co-ocorrência com raio " + str(2**i) +
                    "\nHomogeneidade: " + str("{:.4f}".format(resultado[0])) +
                    "\t\tEntropia: " + str("{:.4f}".format(resultado[1])) +
                    "\nEnergia: " + str("{:.4f}".format(resultado[2])) +
                    "\t\tContraste: " + str("{:.4f}".format(resultado[3])) +
                    "\nDissimilaridade: " + str("{:.4f}".format(resultado[4])) +
                    "\t\tCorrelação: " + str("{:.4f}".format(resultado[5])) + "\n"
                    ).pack()
                i += 1
            tempo = time.time() - tempoInicio
            Label(newWindow,text ="Tempo: "  + str(tempo) + " segundos").pack()
        else:
            newWindow = Toplevel(my_window)
            newWindow.title("Erro")
            newWindow.geometry("250x100")
            Label(newWindow, 
            text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def calcularSel45():
    global fname, classificarSel, fname5
    tempoInicio = time.time()
    if classificarSel:
        if carregouImagem:
            imgCanvas = Image.open(fname)
            cropped_img = imgCanvas.crop(areas[-1])
            crop_img = cropped_img.save("selecao.png")
            image = imread("selecao.png")
            image_gray = rgb2gray(image)
            resultado = np.zeros(6)
            newWindow = Toplevel(my_window) 
            newWindow.title("Características da região selecionada") 
            newWindow.geometry("450x500")
            i = 0
            while i < len(resultado)-1:
                matrizResultante = greycomatrix(image_gray,[2**i],[45])
                resultado[0] = greycoprops(matrizResultante, 'homogeneity')
                resultado[1] = entropy(image_gray)
                resultado[2] = greycoprops(matrizResultante, 'energy')
                resultado[3] = greycoprops(matrizResultante, 'contrast')
                resultado[4] = greycoprops(matrizResultante, 'dissimilarity')
                resultado[5] = greycoprops(matrizResultante, 'correlation')
                Label(newWindow, 
                    text = "Matriz de co-ocorrência com raio " + str(2**i) +
                    "\nHomogeneidade: " + str("{:.4f}".format(resultado[0])) +
                    "\t\tEntropia: " + str("{:.4f}".format(resultado[1])) +
                    "\nEnergia: " + str("{:.4f}".format(resultado[2])) +
                    "\t\tContraste: " + str("{:.4f}".format(resultado[3])) +
                    "\nDissimilaridade: " + str("{:.4f}".format(resultado[4])) +
                    "\t\tCorrelação: " + str("{:.4f}".format(resultado[5])) + "\n"
                    ).pack()
                i += 1
            tempo = time.time() - tempoInicio
            Label(newWindow,text ="Tempo: "  + str(tempo) + " segundos").pack()
        else:
            newWindow = Toplevel(my_window)
            newWindow.title("Erro")
            newWindow.geometry("250x100")
            Label(newWindow, 
            text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def calcularSel90():
    global fname, classificarSel, fname5
    tempoInicio = time.time()
    if classificarSel:
        if carregouImagem:
            tempoInicio = time.time()
            imgCanvas = Image.open(fname)
            cropped_img = imgCanvas.crop(areas[-1])
            crop_img = cropped_img.save("selecao.png")
            image = imread("selecao.png")
            image_gray = rgb2gray(image)
            resultado = np.zeros(6)
            newWindow = Toplevel(my_window) 
            newWindow.title("Características da região selecionada") 
            newWindow.geometry("450x500") 
            i = 0
            while i < len(resultado)-1:
                matrizResultante = greycomatrix(image_gray,[2**i],[90])
                resultado[0] = greycoprops(matrizResultante, 'homogeneity')
                resultado[1] = entropy(image_gray)
                resultado[2] = greycoprops(matrizResultante, 'energy')
                resultado[3] = greycoprops(matrizResultante, 'contrast')
                resultado[4] = greycoprops(matrizResultante, 'dissimilarity')
                resultado[5] = greycoprops(matrizResultante, 'correlation')
                Label(newWindow, 
                    text = "Matriz de co-ocorrência com raio " + str(2**i) +
                    "\nHomogeneidade: " + str("{:.4f}".format(resultado[0])) +
                    "\t\tEntropia: " + str("{:.4f}".format(resultado[1])) +
                    "\nEnergia: " + str("{:.4f}".format(resultado[2])) +
                    "\t\tContraste: " + str("{:.4f}".format(resultado[3])) +
                    "\nDissimilaridade: " + str("{:.4f}".format(resultado[4])) +
                    "\t\tCorrelação: " + str("{:.4f}".format(resultado[5])) + "\n"
                    ).pack()
                i+=1
            tempo = time.time() - tempoInicio
            Label(newWindow,text ="Tempo: "  + str(tempo) + " segundos").pack()
        else:
            newWindow = Toplevel(my_window)
            newWindow.title("Erro")
            newWindow.geometry("250x100")
            Label(newWindow, 
            text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def calcularSel135():
    global fname, classificarSel, fname5
    tempoInicio = time.time()
    if classificarSel:
        if carregouImagem:
            imgCanvas = Image.open(fname)
            cropped_img = imgCanvas.crop(areas[-1])
            crop_img = cropped_img.save("selecao.png")
            image = imread("selecao.png")
            image_gray = rgb2gray(image)
            resultado = np.zeros(6)
            newWindow = Toplevel(my_window) 
            newWindow.title("Características da região selecionada") 
            newWindow.geometry("450x500") 
            i = 0
            while i < len(resultado)-1:
                matrizResultante = greycomatrix(image_gray,[2**i],[135])
                resultado[0] = greycoprops(matrizResultante, 'homogeneity')
                resultado[1] = entropy(image_gray)
                resultado[2] = greycoprops(matrizResultante, 'energy')
                resultado[3] = greycoprops(matrizResultante, 'contrast')
                resultado[4] = greycoprops(matrizResultante, 'dissimilarity')
                resultado[5] = greycoprops(matrizResultante, 'correlation')
                Label(newWindow, 
                    text = "Matriz de co-ocorrência com raio " + str(2**i) +
                    "\nHomogeneidade: " + str("{:.4f}".format(resultado[0])) +
                    "\t\tEntropia: " + str("{:.4f}".format(resultado[1])) +
                    "\nEnergia: " + str("{:.4f}".format(resultado[2])) +
                    "\t\tContraste: " + str("{:.4f}".format(resultado[3])) +
                    "\nDissimilaridade: " + str("{:.4f}".format(resultado[4])) +
                    "\t\tCorrelação: " + str("{:.4f}".format(resultado[5])) + "\n"
                    ).pack()
                i += 1
            tempo = time.time() - tempoInicio
            Label(newWindow,text ="Tempo: "  + str(tempo) + " segundos").pack()
        else:
            newWindow = Toplevel(my_window)
            newWindow.title("Erro")
            newWindow.geometry("250x100")
            Label(newWindow, 
            text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def criarRetangulo(event):
    global classificarSel, click_Ret,xRet,yRet,my_image,copy,area,recortar, retangulos, areas
    if click_Ret == 0:
        xRet=event.x
        yRet=event.y
        click_Ret=1
    else:
        x1=event.x
        y1=event.y
        retangulos.append(my_canvas.create_rectangle(xRet,yRet,x1,y1,outline='blue',width=5))
        area = (xRet, yRet, x1, y1)
        areas.append(area)
        recortar = True
        classificarSel = True
        click_Ret=0

def regiao128(event):
    global classificarSel, xRet,yRet,my_image,copy, x, y, area, recortar, retangulos, areas
    x= event.x
    y= event.y
    if x < 64:
        if y < 64:
            retangulos.append(my_canvas.create_rectangle(0,0,128,128,outline='blue',width=5))
            area = (0,0,128,128)
            areas.append(area)
        elif y > (my_canvas.my_image.height()-64):
            yRet = my_canvas.my_image.height()-128
            retangulos.append(my_canvas.create_rectangle(0,yRet,128,yRet+128,outline='blue',width=5))
            area = (0,yRet,128,yRet+128)
            areas.append(area)
        else:
            retangulos.append(my_canvas.create_rectangle(0,y-64,128,y+64,outline='blue',width=5))
            area = (0,y-64,128,y+64)
            areas.append(area)
    elif x > (my_canvas.my_image.width()-64):
        if event.y < 64:
            xRet = my_canvas.my_image.width()-128
            retangulos.append(my_canvas.create_rectangle(xRet,0,(xRet+128),128,outline='blue',width=5))
            area = (xRet,0,(xRet+128),128)
            areas.append(area)
        elif y > (my_canvas.my_image.height()-64):
            xRet = my_canvas.my_image.width()-128
            yRet = my_canvas.my_image.height()-128
            retangulos.append(my_canvas.create_rectangle(xRet,yRet,(xRet+128),(yRet+128),outline='blue',width=5))
            area = (xRet,yRet,(xRet+128),(yRet+128))
            areas.append(area)
        else:
            retangulos.append(my_canvas.create_rectangle(xRet,y-64,(xRet+128),y+64,outline='blue',width=5))
            area = (xRet,y-64,(xRet+128),y+64)
            areas.append(area)
    else:
        if event.y < 64:
            xRet = my_canvas.my_image.width()-128
            retangulos.append(my_canvas.create_rectangle(x-64,0,x+64,128,outline='blue',width=5))
            area = (x-64,0,x+64,128)
            areas.append(area)
        elif y > (my_canvas.my_image.height()-64):
            xRet = my_canvas.my_image.width()-128
            yRet = my_canvas.my_image.height()-128
            retangulos.append(my_canvas.create_rectangle(x-64,yRet,x+64,(yRet+128),outline='blue',width=5))
            area = (x-64,yRet,x+64,(yRet+128))
            areas.append(area)
        else:
            retangulos.append(my_canvas.create_rectangle(x-64,y-64,x+64,y+64,outline='blue',width=5))
            area = (x-64,y-64,x+64,y+64)
            areas.append(area)
    recortar = True
    classificarSel = True

def recorte():
	global area, recortar, retangulos, areas, my_image, fname, fname5
	if recortar:
		fname = "save.png"
		fname5 = "save.png"
		img = Image.open(fname)
		cropped_img = img.crop(areas[-1])
		crop_img = cropped_img.save(fname)
		my_image = Image.open(fname)
		copy = ImageTk.PhotoImage(cropped_img)
		my_canvas.delete("all")
		my_canvas.config(width=copy.width(), height=copy.height())
		my_canvas.pack(expand=True)
		my_canvas.my_image = copy
		my_canvas.create_image(0,0,image=copy, anchor=tk.NW)
		my_canvas.place(x=0,y=0)
	retangulos =[]
	areas = []
	recortar = False

def classificarSelecao():
    global classificarSel, retangulos, areas, my_image, fname5
    tempoInicio = time.time()
    if classificarSel:
        verificaModelo = os.path.exists('Treinados/modelo.h5')
        if (treinou or verificaModelo) and carregouImagem:
            model = tf.keras.models.load_model('Treinados/modelo.h5')
            imgCanvas = Image.open(fname5)
            cropped_img = imgCanvas.crop(areas[-1])
            crop_img = cropped_img.save("selecao.png")
            img = image.load_img("selecao.png")
            X = image.img_to_array(img)
            X = np.expand_dims(X, axis = 0)
            images = np.vstack([X])
            val = model.predict(images)
            index = np.argmax(val[0])
            newWindow = Toplevel(my_window)
            newWindow.title("Classificação")
            newWindow.geometry("450x100")
            strResp = ""
            if val[0][index] == val[0][0]:
                strResp = "BIRADS 1"
            elif val[0][index] == val[0][1]:
                strResp = "BIRADS 2"
            elif val[0][index] == val[0][2]:
                strResp = "BIRADS 3"
            elif val[0][index] == val[0][3]:
                strResp = "BIRADS 4"
            tempo = time.time() - tempoInicio
            Label(newWindow, 
            text = "Esta imagem é: " + strResp + "\nClassificação realizada em " + str(tempo) + " segundos").pack()
        elif carregouImagem:
            newWindow = Toplevel(my_window)
            newWindow.title("Erro")
            newWindow.geometry("250x100")
            Label(newWindow, 
            text = "É necessário treinar a rede antes\nde executar esta função").pack()
        elif treinou or verificaModelo:
            newWindow = Toplevel(my_window)
            newWindow.title("Erro")
            newWindow.geometry("250x100")
            Label(newWindow, 
            text = "É necessário carregar uma imagem\nantes de executar esta função").pack()
        else:
            newWindow = Toplevel(my_window)
            newWindow.title("Erro")
            newWindow.geometry("350x100")
            Label(newWindow, 
            text = "É necessário treinar a rede e carregar uma imagem\nantes de executar esta função").pack()

def apagaUltimoRetangulo():
	global retangulos, areas, recortar, classificarSel
	if len(retangulos) > 0:
		my_canvas.delete(retangulos[-1])
		retangulos.pop()
		areas.pop()
	if len(areas) == 0:
		recortar = False
		classificarSel = False

def quant256():
    global my_image, fname5
    if carregouImagem:
        my_image = Image.open(fname5)
        copy = my_image
        #Deixar imagem em grayscape(preto e branco)
        copy = ImageOps.grayscale(copy)
        #converter imagem para numpy
        numpydata = asarray(copy)
        #criacao da imagem quantizada
        r = 1
        numpydata = np.uint8(numpydata / r) * r
        # converter a imagem para pil novamente 
        copy = Image.fromarray(numpydata)
        # converter a imagem para tkinter
        copy2 = ImageTk.PhotoImage(copy)
        img = ImageTk.getimage(copy2)
        my_image = img
        fname5 = "save.png"
        save = "save.png"
        imgem = my_image.save(save)
        #criação canvas
        my_canvas.config(width=copy2.width(), height=copy2.height())
        my_canvas.pack(expand=True)
        my_canvas.my_image = copy2
        my_canvas.create_image(0,0,image=copy2, anchor=tk.NW)
        my_canvas.place(x=0,y=0)
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
            text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def quant32():
    global my_image, fname5
    if carregouImagem:
        my_image = Image.open(fname5)
        copy = my_image
        copy = ImageOps.grayscale(copy)
        numpydata = asarray(copy)
        r = 8
        numpydata = np.uint8(numpydata / r) * r
        copy = Image.fromarray(numpydata)
        copy2 = ImageTk.PhotoImage(copy)
        img = ImageTk.getimage(copy2)
        my_image = img
        save = "save.png"
        fname5 = "save.png"
        imgem = my_image.save(save)
        my_canvas.config(width=copy2.width(), height=copy2.height())
        my_canvas.pack(expand=True)
        my_canvas.my_image = copy2
        my_canvas.create_image(0,0,image=copy2, anchor=tk.NW)
        my_canvas.place(x=0,y=0)
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
            text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def quant16():
    global my_image, fname5
    if carregouImagem:
        my_image = Image.open(fname5)
        copy = my_image
        copy = ImageOps.grayscale(copy)
        numpydata = asarray(copy)
        r = 16
        numpydata = np.uint8(numpydata / r) * r
        copy = Image.fromarray(numpydata)
        copy2 = ImageTk.PhotoImage(copy)
        img = ImageTk.getimage(copy2)
        my_image = img
        save = "save.png"
        fname5 = "save.png"
        imgem = my_image.save(save)
        my_canvas.config(width=copy2.width(), height=copy2.height())
        my_canvas.pack(expand=True)
        my_canvas.my_image = copy2
        my_canvas.create_image(0,0,image=copy2, anchor=tk.NW)
        my_canvas.place(x=0,y=0)
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
            text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def resolu32():
    global my_image, fname5
    global copy
    if carregouImagem:
        my_image = Image.open(fname5)
        copy = my_image
        width, height = copy.size
        copy = copy.resize((32,32))
        copy = copy.resize((width,height),Image.NEAREST)
        copy = ImageTk.PhotoImage(copy)
        img = ImageTk.getimage(copy)
        my_image = img
        save = "save.png"
        fname5 = "save.png"
        imgem = my_image.save(save)
        #criar canvas
        my_canvas.config(width=copy.width(), height=copy.height())
        my_canvas.pack(expand=True)
        my_canvas.my_image = copy
        my_canvas.create_image(0,0,image=copy, anchor=tk.NW)
        my_canvas.place(x=0,y=0)
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def resolu64():
    global my_image, fname5
    global copy
    if carregouImagem:
        my_image = Image.open(fname5)
        copy = my_image
        width, height = copy.size
        copy = copy.resize((64,64))
        copy = copy.resize((width,height),Image.NEAREST)
        copy = ImageTk.PhotoImage(copy)
        img = ImageTk.getimage(copy)
        my_image = img
        save = "save.png"
        fname5 = "save.png"
        imgem = my_image.save(save)
        #criar canvas
        my_canvas.config(width=copy.width(), height=copy.height())
        my_canvas.pack(expand=True)
        my_canvas.my_image = copy
        my_canvas.create_image(0,0,image=copy, anchor=tk.NW)
        my_canvas.place(x=0,y=0)
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

#Função resposável por chamar a função "criarRetangulo" e habilitar a função de cliques na tela
def printRetangulo():
    if carregouImagem:
        my_canvas.bind('<Button-1>', criarRetangulo)
        my_canvas.grid(row=1,column=1)
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def printRegiao128():
    if carregouImagem:
	    my_canvas.bind('<Button-1>', regiao128)
	    my_canvas.grid(row=1,column=1)
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def histograma():
    global my_image, fname5
    global fname
    global fname2
    if carregouImagem:
        img = cv2.imread(fname5)
        my_image = Image.open(fname5)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        fname2 = img_output
        img_canvas = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_output))
        my_canvas.pack(expand=False)
        my_canvas.my_image = img_canvas
        img = ImageTk.getimage(img_canvas)
        save = "save.png"
        fname5 = "save.png"
        imgem = img.save(save)
        my_canvas.create_image(0,0,image=img_canvas, anchor=tk.NW)
        my_canvas.place(x=0,y=0)
        equalizou = True
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário carregar uma imagem\nantes de executar esta função").pack()

def grafico():
    global fname2
    if equalizou:
        gray_img = cv2.imread(fname5)
        hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
        plt.hist(gray_img.ravel(),256,[0,256])
        plt.title('Histograma de Tons de Cinza Antes da Equalização')
        plt.show()
        gray_img2 = asarray(fname2)
        hist = cv2.calcHist([gray_img2],[0],None,[256],[0,256])
        plt.hist(gray_img2.ravel(),256,[0,256])
        plt.title('Histograma de Tons de Cinza Depois da Equalização')
        plt.show()
    else:
        newWindow = Toplevel(my_window)
        newWindow.title("Erro")
        newWindow.geometry("250x100")
        Label(newWindow, 
         text = "É necessário equalizar a imagem\nantes de executar esta função").pack()

    


#Menu para o file e sair do programa
file_menu = Menu(my_menu)
my_menu.add_cascade(label="Arquivo", menu=file_menu)
file_menu.add_command(label="Carregar imagem",command=abrirImagem)
file_menu.add_separator()
file_menu.add_command(label="Default", command= default)
file_menu.add_separator()
file_menu.add_command(label="ler Diretorios",command=lerDiretorio)
file_menu.add_separator()
file_menu.add_command(label="Exit",command=my_window.quit)



#Menu para edit
edit_menu = Menu(my_menu)
my_menu.add_cascade(label="Zoom", menu=edit_menu)
edit_menu.add_command(label="Zoom in", command= zoomIN )
edit_menu.add_separator()
edit_menu.add_command(label="Zoom out", command= zoomOUT)

edit_option = Menu(my_menu)
my_menu.add_cascade(label="Regiões", menu=edit_option)
edit_option.add_command(label="Regiao 128", command=printRegiao128)
edit_option.add_separator()
edit_option.add_command(label="Região tamanho variável", command=printRetangulo)
edit_option.add_separator()
edit_option.add_command(label="Recortar região", command=recorte)
edit_option.add_separator()
edit_option.add_command(label="Desfazer", command=apagaUltimoRetangulo)
edit_option.add_separator()
edit_option.add_command(label="Equalização", command=histograma)
edit_option.add_command(label="Histograma Equalização", command=grafico)

edit_option = Menu(my_menu)
my_menu.add_cascade(label="Quantização", menu=edit_option)
edit_option.add_command(label="256", command=quant256)
edit_option.add_separator()
edit_option.add_command(label="32", command=quant32)
edit_option.add_separator()
edit_option.add_command(label="16", command=quant16)

edit_option = Menu(my_menu)
my_menu.add_cascade(label="Resolução", menu=edit_option)
edit_option.add_command(label="64x64", command=resolu64)
edit_option.add_separator()
edit_option.add_command(label="32x32", command=resolu32)

edit_option = Menu(my_menu)
my_menu.add_cascade(label="Matrizes de co-ocorrência da imagem", menu=edit_option)
edit_option.add_command(label="0 grau", command=calcular0)
edit_option.add_separator()
edit_option.add_command(label="45 graus", command=calcular45)
edit_option.add_separator()
edit_option.add_command(label="90 graus", command=calcular90)
edit_option.add_separator()
edit_option.add_command(label="135 graus", command=calcular135)

edit_option = Menu(my_menu)
my_menu.add_cascade(label="Matrizes de co-ocorrência da região selecionada", menu=edit_option)
edit_option.add_command(label="0 grau", command=calcularSel0)
edit_option.add_separator()
edit_option.add_command(label="45 graus", command=calcularSel45)
edit_option.add_separator()
edit_option.add_command(label="90 graus", command=calcularSel90)
edit_option.add_separator()
edit_option.add_command(label="135 graus", command=calcularSel135)

edit_option = Menu(my_menu)
my_menu.add_cascade(label="Rede Neural", menu=edit_option)
edit_option.add_command(label="Treinar", command=treinamento)
edit_option.add_separator()
edit_option.add_command(label="Classificar", command=classificar)
edit_option.add_separator()
edit_option.add_command(label="Classificar Seleção", command=classificarSelecao)
edit_option.add_separator()
edit_option.add_command(label="Matriz Confusão", command=matrizConfusao)


#finalizar interface
my_window.mainloop()