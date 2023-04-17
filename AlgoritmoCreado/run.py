
from clases import DistEuclidiana

#Biblioteca
import pandas as pd


datos= pd.read_csv("prueba3.csv", sep=',')
#d1 = Graficacion(datos)
#d1.imprimirGrafico()

longMuestra=4
dist = DistEuclidiana(datos,longMuestra)
distanciaEuclidiana = dist.distancia()

#lEER ARCHIVO

dist.leerArchivo()


