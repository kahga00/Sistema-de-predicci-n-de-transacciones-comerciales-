import pandas
from clases import *
#Biblioteca
import pandas as pd

class run():
    def __init__(self):
        super().__init__()
        self.entrenamiento_p()
        self.Mostrar()

    def entrenamiento_p(self):
        self.longMuestra=4
        self.dist = DistEuclidiana(self.datos,self.longMuestra)
        self.distanciaEuclidiana = self.dist.distancia()

    def Mostrar(self):
        #lEER ARCHIVO
        self.dist.leerArchivo()

if __name__=='__main__':
    Variable = run()
    Variable.mainloop()
