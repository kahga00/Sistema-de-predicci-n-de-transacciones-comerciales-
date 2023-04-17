import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler

#Se cargan los datos deñ csv


df_comp = pd.read_csv('prueba4.csv', parse_dates=[0], header=None,index_col=0, squeeze=True,names=['fechaHora','trxs'], dayfirst = True)
df_comp.plot()
df = df_comp.copy()
df= df.drop(df[1152:1440].index,axis=0)
#Se crean visualizaciones para ver si los datos se encuentran correctamente insertados
print("Primeros datos del csv")
print(df.head())
print("Fecha minima y maxima del csv")
print(df.index.min())
print(df.index.max())
#asignación de fecha final e inicial
start_date = df.index.min()
end_date = df.index.max()
Frecuencia =df.resample('5T').mean() #Ajuste a 5 minutos
plt.plot(Frecuencia[start_date:end_date].values)
plt.show()
df.plot()
plt.show()
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
values = df.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
values = values.reshape(-1, 1)  # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, steps, 1)
reframed.head()

####################################
# split into train and test sets
values = reframed.values
n_train_days = (1152) - (288+steps)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

########################################
def crear_modeloFF():
    model = Sequential()
    model.add(Dense(steps, input_shape=(1,steps),activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])
    model.summary()
    return model

###########################################
EPOCHS=500

model = crear_modeloFF()

history=model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=steps)


########################################
results=model.predict(x_val)
print( len(results) )
plt.scatter(range(len(y_val)),y_val,c='g')
plt.scatter(range(len(results)),results,c='r')
plt.title('Validación')
plt.show()


################################
plt.plot(history.history['loss'])
plt.title('loss')
plt.plot(history.history['val_loss'])
plt.title('validate loss')
plt.legend(['loss','val_loss'])
plt.show()


##################################
comparation = pd.DataFrame(np.array([y_val, [x[0] for x in results]])).transpose()
comparation.columns = ['real', 'prediccion']
print( comparation )
inverted = scaler.inverse_transform(comparation.values)

comp2 = pd.DataFrame(inverted)
comp2.columns = ['real', 'prediccion']
comp2['diferencia'] = comp2['real'] - comp2['prediccion']
print(comp2)
################
print(comp2.describe())
#######################
comp2['real'].plot(color = "blue")
comp2['prediccion'].plot(color = "green")
plt.legend(['real','predicción'])

comp2.columns = ['real','prediccion','diferencia']
comp2.to_csv('pronostico.csv')
##################################
Days = df[start_date:end_date]
Days
#############################
values = Days.values
values = values.astype('float32')
print(values)
# normalize features
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, steps, 1)
reframed.drop(reframed.columns[[4]], axis=1, inplace=True)
reframed.head(7)#aquicambie28112022tenia7
###########################
values = reframed.values
x_test = values[6:, :]#aquicambie28112022n2tenia6
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_test.shape)
x_test
###################################
def agregarNuevoValor(x_test,nuevoValor):
    for i in range(x_test.shape[2]-1):
        x_test[0][0][i] = x_test[0][0][i+1]
    x_test[0][0][x_test.shape[2]-1]=nuevoValor
    return x_test
##########################################
results=[]
for i in range(288):
    parcial=model.predict(x_test)
    results.append(parcial[0])
    print(x_test)
    x_test=agregarNuevoValor(x_test,parcial[0])
###################################################
adimen = [x for x in results]
print(adimen)
inverted = scaler.inverse_transform(adimen)
inverted
##############################################
DayNew = pd.DataFrame(inverted)
DayNew.columns = ['pronostico']
plt.show()
DayNew.to_csv('quintodia.csv')
##################################################
i=0
for fila in DayNew.pronostico:
    i=i+1
    Days.loc[str(i)] = fila
    print(fila)
Days.tail(288)
###################################################
#plt.plot(Frecuencia[start_date:end_date].values)
ultimodia= Days[:]
plt.plot(ultimodia.values)
muestrainicial= df_comp[:]
plt.plot(muestrainicial.values)
plt.legend(['ultimodia','muestrainicial'])
plt.show()
#############################################
df2 = df_comp.copy()
df2= df2.drop(df[0:1152].index,axis=0)
muestrainicial= df2.values
print(df2.describe())
quintofinal= Days.values

diferenciatotal= pd.DataFrame
diferenciatotal.columns = ['finalfinal']
diferenciatotal['finalfinal'] = muestrainicial - quintofinal
diferenciatotal.columns = ['finalfinal']
diferenciatotal.to_csv('ultimoarchivo.csv')
print(diferenciatotal)

n=0
if diferenciatotal.values >= 500 or diferenciatotal.values <= -500:
    n = n + 1
if n<=15:
    print("no hay alarmas")
else:
    print("El sistema generara una alarma para este cliente")




