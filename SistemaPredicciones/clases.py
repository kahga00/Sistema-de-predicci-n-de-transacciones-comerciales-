#Manipulacion y tratamiento de datos
from tokenize import PlainToken
import numpy as np
import pandas as pd
import tkinter as tk
#Visualización de Dato
import matplotlib.pyplot as plt
import seaborn as sb

#Normalizar datos
from locale import normalize
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

 #Biblioteca de Distancia Euclidiana
from scipy.spatial import distance


class Graficacion:
    def __init__(self, datos):
        self.datos = datos

    def imprimirGrafico(self):
        plt.figure(figsize=(22,6))
        sb.lineplot(x=self.datos['fechaHora'], y=self.datos['trxs'])
        plt.title('Transacciones por 5 minutos')
        plt.show()


class NormalizarDatos:
    def __init__(self, datos):
         self.datos = datos

    def normalizar(self):
        trx = self.datos['trxs']
        d = pd.DataFrame(trx)

        tran = d.to_numpy()
        scaler = preprocessing.MinMaxScaler()
        normalizedlist=scaler.fit_transform(tran)
    
       
        return normalizedlist


class DistEuclidiana:
    def __init__(self, datos, longMuestra):
         self.datos=datos
         n1 = NormalizarDatos(self.datos)
         self.normalized_trx = n1.normalizar()
         self.longIterador =longMuestra+1
         self.longMuestra = longMuestra
         self.distancias=[]
         self.f = open("archivo.txt", "w")





    def distancia(self):
        #Definir valores
        trx = self.datos['trxs']
        d = pd.DataFrame(trx)

        tran = d.to_numpy()
        indice=len(self.normalized_trx)-self.longMuestra
        muestra1=self.normalized_trx[indice:indice+self.longMuestra]
        muestra1 =np.concatenate(muestra1)
        #Crear archivo y agregar Datos Normalizados
        self.f = open("archivo.txt", "a")
        
    
        muestra2 =[]
        iterador=len(self.normalized_trx)-self.longIterador
        longResta =1
        distancias=[]
        cumplen=[]

        #Inicio de for
        for i in range(self.longMuestra+1,len(self.normalized_trx),longResta):
        
            muestra2=self.normalized_trx[iterador:iterador+self.longIterador-longResta]
            muestra2 =np.concatenate(muestra2)
                #Agregar muestra 2
            self.f = open("archivo.txt", "a")
            self.f.write("Muestra1: ")
            self.f = open("archivo.txt", "a")
            self.f.write(str(muestra1))
            self.f = open("archivo.txt", "a")
            self.f.write("        Iterador:")
            self.f = open("archivo.txt", "a")
            self.f.write(str(iterador))
            self.f = open("archivo.txt", "a")
            self.f.write("        Muestra 2:")
            self.f = open("archivo.txt", "a")
            
            self.f.write(str(muestra2))
    

            #Operacion euclidiana
            dist = distance.euclidean(muestra1,muestra2)
            distancia2 = 1- dist
            self.f.write("       Distancia: ")
            self.f = open("archivo.txt", "a")
            self.f.write(str(distancia2))
            if distancia2 >= 0.9:
                self.f.write("       Si ")

                cumplen.append(tran[iterador+1])
                self.f.write("\n")

            else:
                self.f.write("       No ")
                self.f.write("\n")
            
            #Agrega resultado a un array
            distancias.append(dist)
            #Se recorre el segundo valor, 1 punto
            self.longIterador=self.longIterador+1
            longResta = longResta+1
            iterador=len(self.normalized_trx)-self.longIterador
            self.distancias=distancias
           
        cumplen = np.concatenate(cumplen)
        self.f.write(" Datos que cumplen: ")
        self.f = open("archivo.txt", "a")
        self.f.write(str(cumplen))
        suma = 0
        
        for valor in cumplen:
            suma = suma + valor
        longCumplen = len(cumplen)
        promedio = suma /longCumplen
        self.f.write("\n")
        self.f.write("Predicción: ")
        self.f = open("archivo.txt", "a")
        self.f.write(str(promedio))
        return distancias

    def leerArchivo(self):
        self.f = open("archivo.txt", "r")
        print(self.f.read())

