#Manipulacion y tratamiento de datos
from tokenize import PlainToken
import numpy as np
import pandas as pd
#Visualización de Dato
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow 
#Normalizar datos
from locale import normalize
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

import collections
import sys
import tkinter as tk
from tkinter import ttk,messagebox, Menu, scrolledtext, filedialog
from tkinter.filedialog import askopenfile,asksaveasfilename
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler
from matplotlib.figure import Figure
from matplotlib import style
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.animation as animation
from math import *
pd.options.mode.chained_assignment = None  # default='warn'

 #Biblioteca de Distancia Euclidiana
from scipy.spatial import distance

from clases import *
from run import *

class GUIone(tk.Tk):
    def __init__(self):
        super().__init__()
        ###Edicion de ventana
        #iconodeapp
        #Nombre de la ventana
        self.title('Sistema de prediccion')
        # Modificamos el tamaño de la ventana (pixeles)
        self.geometry('1250x720')
        self.config(bg="#264653")

        self.rowconfigure(0, minsize=1, weight=10)
        self.rowconfigure(1, minsize=1, weight=10)
        self.rowconfigure(2, minsize=1, weight=10)
        self.rowconfigure(3, minsize=1, weight=10)
        self.columnconfigure(0, minsize=1, weight=10)
        self.columnconfigure(1, minsize=1, weight=10)
        self.columnconfigure(2, minsize=1, weight=10)

        self.fila1 = tk.Frame(self, bg='#264653')
        self.fila2 = tk.Frame(self, bg='#264653')
        self.fila3 = tk.Frame(self, bg='#264653')
        self.fila4 = tk.Frame(self, bg='#264653')
        self.columna1 = tk.Frame(self, bg='#264653')
        self.columna2 = tk.Frame(self, bg='#264653')
        self.columna3 = tk.Frame(self, bg='#264653')

        self.fila1.grid(row=0, column=0, sticky='WENS', padx=5, pady=5)
        self.fila2.grid(row=1, column=0, sticky='WENS', padx=5, pady=5)
        self.fila3.grid(row=2, column=0, sticky='WENS', padx=5, pady=5)
        self.fila4.grid(row=3, column=0, sticky='WENS', padx=5, pady=5)
        self.columna1.grid(row=0, column=0, sticky='WENS', padx=5, pady=5)
        self.columna2.grid(row=0, column=1, sticky='WENS', padx=5, pady=5)
        self.columna3.grid(row=0, column=2, sticky='WENS', padx=5, pady=5)

        self.campo_csv = tk.Text(self, wrap=tk.WORD)
        self.campo_csv2 = tk.Text(self, wrap=tk.WORD)
        self.texto1 = tk.Text(self, wrap=tk.WORD)
        self.p = tk.Text(self, wrap=tk.WORD)
        self.archivocsv = None
        self.archivocsv2 = None
        self.p = None
        self.archivo_abierto = False
        self.archivo_abierto2 = False
        self._Tabuladores()
        self._menu_principal()
        #para las graficas de muestra




    ###########tabulador
    def _Tabuladores(self):
        color_pestaña = ttk.Style()
        configuracion = {"TNotebook.Tab": {"configure": {"padding": [15, 5],"background": "#fca311","relief" : ("FLAT"),"bd":("50")},
                                           "map": {"background": [("selected", "#14213d"),("active", "#14213d")],
                                                   "foreground": [("selected", "#ffffff"),("active", "#ffffff")],
                                                   }}}
        color_pestaña.theme_create("estilo", parent="alt", settings=configuracion)
        color_pestaña.theme_use("estilo")

        # tab control
        self.control_tabulador = ttk.Notebook(self)
        # primer ventana tab1
        self.control_tabulador.grid(row=0, column=0, padx=40, pady=40, sticky='WENS',columnspan=3,rowspan=4)
        # agregamos tab1 al tab control
        tabulador1 = ttk.LabelFrame(self.control_tabulador, text='Metodo MLP:"Multi-Layered Perceptron"')
        self.control_tabulador.add(tabulador1, text='Supervisado')
        self._componentestab1(tabulador1)
        #ventana tab2
        tabulador2 = ttk.Labelframe(self.control_tabulador, text='csv - resultados')
        self.control_tabulador.add(tabulador2, text='Datos_Supervisado')
        self._componentestab2(tabulador2)
        #ventana tab3
        tabulador3 = ttk.Labelframe(self.control_tabulador, text='Metodo propuesto')
        self.control_tabulador.add(tabulador3, text='Agrupamiento')
        self._componentestab3(tabulador3)
        #ventana tab4
        tabulador4 = ttk.Labelframe(self.control_tabulador, text='csv - resultados')
        self.control_tabulador.add(tabulador4, text='Datos_Agrupamiento')
        self._componentestab4(tabulador4)
        #probablemente aqui se mande a llamar a las graficas



    def _componentestab1(self, tabulador):
        self._diseño_interfaz(tabulador)

        self.boton1 = tk.Button(tabulador, text='iniciar entrenamiento', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                           cursor='hand2',command=self._evento_entrenamiento)
        self.boton1.grid(row=0, column=0, padx=40, pady=40, sticky='WENS')

        self.boton2 = tk.Button(tabulador, text='Cargar datos', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                           cursor='hand2',command=self._evento_carga)
        self.boton2.grid(row=1, column=0, padx=40, pady=40, sticky='WENS')

        self.boton3 = tk.Button(tabulador, text='Generar datos de muestra', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                           cursor='hand2',command=self._evento_muestra)
        self.boton3.grid(row=2, column=0, padx=40, pady=40, sticky='WENS')

        progreso = ttk.Progressbar(tabulador, orient='horizontal', length=550)
        progreso.grid(row=4, column=1, padx=10, pady=10)
        def ejecutar_barra():
            if self.boton1:
                def iniciar():
                    self._evento_entrenamiento.start()
                def detener():
                    self._evento_entrenamiento.after(progreso.stop)

    def _componentestab2(self, tabulador):
        self.campo_csv = scrolledtext.ScrolledText(tabulador, width=50, height=15, wrap=tk.WORD)
        self.campo_csv.grid(row=1, column=1)
        self.boton4 = tk.Button(tabulador, text='Guardar', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                           cursor='hand2')##############command=self._guardar)
        self.boton4.grid(row=1, column=0, padx=40, pady=40, sticky='WENS')




    def _componentestab3(self, tabulador):
        self._diseño_interfaz(tabulador)
        ###################botones
        # Creamos un boton (widget), el objeto padre es ventana
        self.boton99 = tk.Button(tabulador, text='iniciar entrenamiento', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                                cursor='hand2',command=self._evento_entrenamiento_propuesto)
        self.boton99.grid(row=0, column=0, padx=40, pady=40, sticky='WENS')

        self.boton98 = tk.Button(tabulador, text='Cargar datos', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                                cursor='hand2',command=self._evento_carga_propuesto)
        self.boton98.grid(row=1, column=0, padx=40, pady=40, sticky='WENS')

        self.boton97 = tk.Button(tabulador, text='Generar datos de muestra', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                                cursor='hand2',command=self._evento_muestra_propuesto)
        self.boton97.grid(row=2, column=0, padx=40, pady=40, sticky='WENS')

        #progreso = ttk.Progressbar(tabulador, orient='horizontal', length=550)
        #progreso.grid(row=4, column=1, padx=10, pady=10)
        self.campo_csv2 = scrolledtext.ScrolledText(tabulador, width=50, height=25, wrap=tk.WORD)
        self.campo_csv2.grid(row=1, column=1, columnspan=5, rowspan=4)

    def _componentestab4(self, tabulador):
        self.texto1 = scrolledtext.ScrolledText(tabulador, width=160, height=30, wrap=tk.WORD)
        self.texto1.grid(row=1, column=1,columnspan=5,rowspan=4)

        self.boton96 = tk.Button(tabulador, text='Guardar', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                                cursor='hand2')  ##############command=self._guardar)
        self.boton96.grid(row=1, column=0, padx=40, pady=40, sticky='WENS')


    def _diseño_interfaz(self, tabulador):
        tabulador.rowconfigure(0, minsize=1, weight=1)
        tabulador.rowconfigure(1, minsize=1, weight=1)
        tabulador.rowconfigure(2, minsize=1, weight=1)
        tabulador.rowconfigure(3, minsize=1, weight=20)
        tabulador.columnconfigure(0, minsize=1, weight=1)
        tabulador.columnconfigure(1, minsize=1, weight=1)
        tabulador.columnconfigure(2, minsize=1, weight=10)

        tabulador.fila1 = tk.Frame(tabulador, bg='#edf2f4')
        tabulador.fila2 = tk.Frame(tabulador, bg='#edf2f4')
        tabulador.fila3 = tk.Frame(tabulador, bg='#edf2f4')
        tabulador.fila4 = tk.Frame(tabulador, bg='#14213d')
        tabulador.columna1 = tk.Frame(tabulador, bg='#edf2f4')
        tabulador.columna2 = tk.Frame(tabulador, bg='#edf2f4')
        tabulador.columna3 = tk.Frame(tabulador, bg='#edf2f4')

        tabulador.fila1.grid(row=0, column=0, sticky='WENS', padx=5, pady=5, rowspan=4)
        tabulador.fila2.grid(row=1, column=0, sticky='WENS', padx=5, pady=5, rowspan=4)
        tabulador.fila3.grid(row=2, column=0, sticky='WENS', padx=5, pady=5, rowspan=4)
        tabulador.fila4.grid(row=3, column=0, sticky='WENS', padx=5, pady=5, rowspan=4)
        tabulador.columna1.grid(row=0, column=0, sticky='WENS', padx=5, pady=5, columnspan=3)
        tabulador.columna2.grid(row=0, column=1, sticky='WENS', padx=5, pady=5, columnspan=3)
        tabulador.columna3.grid(row=0, column=2, sticky='WENS', padx=5, pady=5, columnspan=3)





#### metodos o acciones que suceden al momento de presionar un boton
    #Llamado de boton1
    def _evento_entrenamiento(self):
        self.df_comp = pd.read_csv(self.archivo_abierto, parse_dates=[0], header=None, index_col=0, names=['fechaHora', 'trxs'],
                              dayfirst=True)
        self.df = self.df_comp.copy()
        self.df = self.df.drop(self.df[1152:1440].index, axis=0)
        # Se crean visualizaciones para ver si los datos se encuentran correctamente insertados
        # asignación de fecha final e inicial
        self.start_date = self.df.index.min()
        self.end_date = self.df.index.max()
        self.Frecuencia = self.df.resample('5T').mean()  # Ajuste a 5 minutos
        ####algoritmo de conversion a aprendizaje supervisado
        steps = 288
        def series_to_supervised(data, n_in=2, n_out=2, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            cols, names = list(), list()
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
            # put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg

        # load dataset
        self.values = self.df.values
        # ensure all data is float
        self.values = self.values.astype('float32')
        # normalize features
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.values = self.values.reshape(-1, 1)  # esto lo hacemos porque tenemos 1 sola dimension
        self.scaled = self.scaler.fit_transform(self.values)
        # frame as supervised learning
        reframed = series_to_supervised(self.scaled, steps, 1)
        reframed.head()

        ####################################
        # split into train and test sets
        self.values = reframed.values
        n_train_days = (1152) - (288 + steps)
        self.train = self.values[:n_train_days, :]
        self.test = self.values[n_train_days:, :]
        # split into input and outputs
        x_train, y_train = self.train[:, :-1], self.train[:, -1]
        self.x_val, self.y_val = self.test[:, :-1], self.test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        self.x_val = self.x_val.reshape((self.x_val.shape[0], 1, self.x_val.shape[1]))

        ########################################
        def crear_modeloFF():
            self.model = Sequential()
            self.model.add(Dense(steps, input_shape=(1, steps), activation='tanh'))
            self.model.add(Flatten())
            self.model.add(Dense(1, activation='tanh'))
            self.model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=["mse"])
            self.model.summary()
            return self.model

        ###########################################
        EPOCHS = 500

        self.model = crear_modeloFF()

        self.history = self.model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(self.x_val, self.y_val), batch_size=steps)

        ########################################
        self.results = self.model.predict(self.x_val)

        ##################################
        self.comparation = pd.DataFrame(np.array([self.y_val, [x[0] for x in self.results]])).transpose()
        self.comparation.columns = ['real', 'prediccion']
        self.inverted = self.scaler.inverse_transform(self.comparation.values)

        self.comp2 = pd.DataFrame(self.inverted)
        self.comp2.columns = ['real', 'prediccion']
        self.comp2['diferencia'] = self.comp2['real'] - self.comp2['prediccion']
        ################
        self.Days = self.df[self.start_date:self.end_date]
        self.Days
        #############################
        self.values = self.Days.values
        self.values = self.values.astype('float32')
        # normalize features
        self.values = self.values.reshape(-1, 1)  # esto lo hacemos porque tenemos 1 sola dimension
        self.scaled = self.scaler.fit_transform(self.values)
        self.reframed = series_to_supervised(self.scaled, steps, 1)
        self.reframed.drop(self.reframed.columns[[4]], axis=1, inplace=True)
        self.reframed.head(7)  # aquicambie28112022tenia7
        ###########################
        self.values = self.reframed.values
        x_test = self.values[6:, :]
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        x_test

        ###################################
        def agregarNuevoValor(x_test, nuevoValor):
            for i in range(x_test.shape[2] - 1):
                x_test[0][0][i] = x_test[0][0][i + 1]
            x_test[0][0][x_test.shape[2] - 1] = nuevoValor
            return x_test

        ##########################################
        results = []
        for i in range(288):
            parcial = self.model.predict(x_test)
            results.append(parcial[0])
            print(x_test)
            x_test = agregarNuevoValor(x_test, parcial[0])
        ###################################################
        self.adimen = [x for x in results]
        self.inverted = self.scaler.inverse_transform(self.adimen)
        self.inverted
        ##############################################
        self.DayNew = pd.DataFrame(self.inverted)
        self.DayNew.columns = ['pronostico']
        self.DayNew.to_csv('quintodia.csv')
        ##################################################
        i = 0
        for fila in self.DayNew.pronostico:
            i = i + 1
            self.Days.loc[str(i)] = fila
        self.Days.tail(288)
        ###################################################
        # plt.plot(Frecuencia[start_date:end_date].values)

        #############################################
        self.df2 = self.df_comp.copy()
        self.df2 = self.df2.drop(self.df[0:1152].index, axis=0)
        self.muestrainicial = self.df2.values
        self.quintofinal = self.Days.values

        self.boton1.config(text='Reiniciar todo el proceso')

    def _evento_carga(self):
        self.archivo_abierto = askopenfile(title="abrir", mode='r',initialdir="C:/Users/user/Downloads",filetypes=[("Archivo csv","*.csv")])
        self.campo_csv.delete(1.0,tk.END)

        mensaje = 'No se cargo correctamente el csv o no selecciono nada. \n \n vuelva a intentarlo.'
        if not self.archivo_abierto:
            self.messagebox.showerror('Alerta de entrada', mensaje)
            return

        with open(self.archivo_abierto.name,'r') as self.archivocsv:
            texto = self.archivocsv.read()
            self.campo_csv.insert(1.0, texto)
            self.title(f'*Archivo completo del csv - {self.archivocsv.name}')

        self.boton2.config(text='Cargar otro csv')


    def _evento_muestra(self):
        self.results = self.model.predict(self.x_val)
        plt.scatter(range(len(self.y_val)), self.y_val, c='g')
        plt.scatter(range(len(self.results)), self.results, c='r')
        plt.title('Validación')
        plt.show()

        ################################
        plt.plot(self.history.history['loss'])
        plt.title('loss')
        plt.plot(self.history.history['val_loss'])
        plt.title('validate loss')
        plt.legend(['loss', 'val_loss'])
        plt.show()
        print('genera una grafica final')
        #################################
        self.comp2['real'].plot(color="blue")
        self.comp2['prediccion'].plot(color="green")
        plt.legend(['real', 'predicción'])
        plt.show()
        self.comp2.columns = ['real', 'prediccion', 'diferencia']
        self.comp2.to_csv('pronostico.csv')
        self.ultimodia = self.Days[:]
        plt.plot(self.ultimodia.values)
        self.muestrainicial = self.df_comp[:]
        plt.plot(self.muestrainicial.values)
        plt.legend(['ultimodia', 'muestrainicial'])
        plt.show()

    # Llamado de boton99
    def _evento_entrenamiento_propuesto(self):
        self.datos = pd.read_csv(self.archivo_abierto2, sep=',')
        run.entrenamiento_p(self)
        self.boton98.config(text='Reiniciar entrenamiento')


    def _evento_carga_propuesto(self):
        self.archivo_abierto2 = askopenfile(title="abrir", mode='r', initialdir="C:/Users/user/Downloads",
                                           filetypes=[("Archivo csv", "*.csv")])
        self.campo_csv2.delete(1.0, tk.END)

        mensaje = 'No se cargo correctamente el csv o no selecciono nada. \n \n vuelva a intentarlo.'
        if not self.archivo_abierto2:
            self.messagebox.showerror('Alerta de entrada', mensaje)
            return

        with open(self.archivo_abierto2.name, 'r') as self.archivocsv2:
            texto = self.archivocsv2.read()
            self.campo_csv2.insert(1.0, texto)
            self.title(f'*Archivo completo del csv - {self.archivocsv2.name}')
        self.boton98.config(text='Cargar otro archivo de datos')

    def _evento_muestra_propuesto(self):
        run.Mostrar(self)
        self.f = open("archivo.txt", "r")
        self.mensaje = self.f.read()
        self.texto1.insert(1.0, self.mensaje)


    def _enviar(self):
        # Modificamos el texto del label
        self.etiqueta1.config(text=self.entrada_var1.get())

    def _salir(self):

        self.messagebox.showinfo('No se guardaran cambios ni el csv a menos que los descargue', mensaje1 + 'informativo')

        #messagebox.showerror('Mensaje error', mensaje1 + 'error')
        #messagebox.showwarning('Problema', mensaje1 + 'alerta')
        self.quit()
        self.destroy()
        sys.exit()
        self._menu_principal(self)

    def _menu_principal(self):
        # Configuracion de menu
        self._menu_principal=Menu()
        #tearoff = false para evitar separacion en el menu de la interfaz
        self.submenu_archivo = Menu(self._menu_principal, tearoff='false')
        #nueva opcion
        self.submenu_archivo.add_command(label='Cargar csv')
        #Nueva opcion
        self.submenu_archivo.add_separator()
        self.submenu_archivo.add_command(label='Salir',command=self._salir)
        #agregamos las opciones al menu
        self._menu_principal.add_cascade(menu=self.submenu_archivo, label='Opciones')
        #Mostramos el menu
        self.config(menu=self._menu_principal)



if __name__=='__main__':
    GUIone = GUIone()
    GUIone.mainloop()