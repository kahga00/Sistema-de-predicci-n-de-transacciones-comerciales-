# GUI - Graphical User Interface
# Tkinter - TK Interface
# Importamos el módulo de tkinter
import sys
import tkinter as tk
from tkinter import ttk,messagebox, Menu, scrolledtext, filedialog
from tkinter.filedialog import askopenfile,asksaveasfilename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)

class GUIone(tk.Tk):
    def __init__(self):
        super().__init__()
        ###Edicion de ventana
        #iconodeapp
        self.iconbitmap('grafico.ico')
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
        self.archivocsv = None
        self.archivo_abierto = False
        self._Tabuladores()
        self._menu_principal()





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

    def _componentestab1(self, tabulador):
        self._diseño_interfaz(tabulador)

        #self.etiqueta1 = tk.Label(tabulador, text='Aquí se mostrará un plot')
        #self.etiqueta1.grid(row=0, column=1)
        ###################botones
        # Creamos un boton (widget), el objeto padre es ventana
        self.boton1 = tk.Button(tabulador, text='iniciar entrenamiento', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                           cursor='hand2',command=self._evento_entrenamiento)
        self.boton1.grid(row=0, column=0, padx=40, pady=40, sticky='WENS')

        self.boton2 = tk.Button(tabulador, text='Cargar datos', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                           cursor='hand2',command=self._evento_carga)
        self.boton2.grid(row=1, column=0, padx=40, pady=40, sticky='WENS')

        self.boton3 = tk.Button(tabulador, text='Generar datos de muestra', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                           cursor='hand2',command=self._evento_muestra)
        self.boton3.grid(row=2, column=0, padx=40, pady=40, sticky='WENS')

        self.df_comp = tk.Frame(tabulador)
        self.df_comp.grid(row=0, column=1)

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
        self.boton4 = tk.Button(tabulador, text='Guardar csv', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                           cursor='hand2')##############command=self._guardar)
        self.boton4.grid(row=1, column=0, padx=40, pady=40, sticky='WENS')




    def _componentestab3(self, tabulador):
        self._diseño_interfaz(tabulador)

        self.etiqueta2 = tk.Label(tabulador, text='Aquí se mostrará un plot')
        self.etiqueta2.grid(row=0, column=1)
        ###################botones
        # Creamos un boton (widget), el objeto padre es ventana
        self.boton99 = tk.Button(tabulador, text='iniciar entrenamiento', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                                cursor='hand2',command=self._evento_entrenamiento_propuesto)
        self.boton99.grid(row=0, column=0, padx=40, pady=40, sticky='WENS')

        self.boton98 = tk.Button(tabulador, text='Cargar datos', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                                cursor='hand2',command=self._evento_carga)
        self.boton98.grid(row=1, column=0, padx=40, pady=40, sticky='WENS')

        self.boton97 = tk.Button(tabulador, text='Generar datos de muestra', fg='#ffffff', relief=tk.RAISED, bg='#14213d',
                                cursor='hand2',command=self._evento_muestra)
        self.boton97.grid(row=2, column=0, padx=40, pady=40, sticky='WENS')

        progreso = ttk.Progressbar(tabulador, orient='horizontal', length=550)
        progreso.grid(row=4, column=1, padx=10, pady=10)

        def ejecutar_barra():
            if self.boton1:
                def iniciar():
                    self._evento_entrenamiento.start()

                def detener():
                    self._evento_entrenamiento.after(progreso.stop)

    def _componentestab4(self, tabulador):
        contenido = 'Contenido del csv'
        #scroll
        scroll = scrolledtext.ScrolledText(tabulador, width=50, height=15, wrap=tk.WORD)
        scroll.insert(tk.INSERT, contenido)
        #mostrar scroll
        scroll.grid(row=0, column=0)


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

        tabulador.fila1.grid(row=0, column=0, sticky='WENS', padx=5, pady=5)
        tabulador.fila2.grid(row=1, column=0, sticky='WENS', padx=5, pady=5)
        tabulador.fila3.grid(row=2, column=0, sticky='WENS', padx=5, pady=5)
        tabulador.fila4.grid(row=3, column=0, sticky='WENS', padx=5, pady=5)
        tabulador.columna1.grid(row=0, column=0, sticky='WENS', padx=5, pady=5)
        tabulador.columna2.grid(row=0, column=1, sticky='WENS', padx=5, pady=5)
        tabulador.columna3.grid(row=0, column=2, sticky='WENS', padx=5, pady=5)



#### metodos o acciones que suceden al momento de presionar un boton
    #Llamado de boton1
    def _evento_entrenamiento(self):
        self.boton1.config(text='Reiniciar todo el proceso')


    # Llamado de boton99
    def _evento_entrenamiento_propuesto(self):
            self.boton99.config(text='Reiniciar todo el proceso')

    def _evento_carga(self):
        self.boton2.config(text='Cargar otro csv')
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

        self.df_comp = pd.read_csv(self.archivo_abierto, parse_dates=[0], header=None, index_col=0, squeeze=True, names=['fechaHora', 'trxs'], dayfirst=True)
        self.df_comp.plot()
        #df = df_comp.copy()
        #df = df.drop(df[1152:1440].index, axis=0)
        #start_date = df.index.min()
        #end_date = df.index.max()
        #Frecuencia = df.resample('5T').mean()  # Ajuste a 5 minutos
        #plt.plot(Frecuencia[start_date:end_date].values)
        plt.show()
        #df.plot()
        #plt.show()

    #def _guardar(self):

    def _evento_muestra(self):
        print('genera una grafica final')



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

# Iniciamos la ventana (esta línea la ejecutamos al final) Si la ejecutamos antes, no se muestran los cambios anteriores

if __name__=='__main__':
    GUIone = GUIone()
    GUIone.mainloop()