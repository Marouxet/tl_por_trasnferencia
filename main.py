# Importamos librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##### INPUTS ########
grupo = 'test'
archivo_input_x1 = 'H-X1 09.txt' 
archivo_input_x2 = 'H-X2 09.txt' 
archivo_output = 'H-XT 09.txt' 

x1 = 0.125 
x2 = 0.24 

vel_sonido = 343 

####################


# Defino carpeta donde se guardan los archivos
carpeta = f"data/{grupo}/"

# Leemos archivos
in1 =pd.read_csv(carpeta+archivo_input_x1, sep='\t',skiprows=1)
in2 =pd.read_csv(carpeta+archivo_input_x2, sep='\t',skiprows=1)
out =pd.read_csv(carpeta+archivo_output, sep='\t',skiprows=1)

# Cambio nombre de columnas, pongo frecuencia como index del DataFrame
in1.columns = ['frecuencia','in1_magnitud','in1_fase','in1_coherencia']
in2.columns = ['frecuencia','in2_magnitud','in2_fase','in2_coherencia']
out.columns = ['frecuencia','out_magnitud','out_fase','out_coherencia']


# Establezco la columna frecuencia como index (nombre de fila) y uno todo en una sola variable
in1.set_index('frecuencia', inplace = True)
in2.set_index('frecuencia', inplace = True)
out.set_index('frecuencia', inplace = True)

# Uno todos los dataframes en uno solo

data = pd.merge(
    in1,
    in2,
    left_index=True,
    right_index=True
    )

data = pd.merge(
    data,
    out,
    left_index=True,
    right_index=True
    )

# Pongo la frecuencia como una columna
data.reset_index(inplace=True)

# Agrego data sobre kx1 y kx2
data['kX1'] = 2*np.pi*data['frecuencia']*x1/vel_sonido
data['kX2'] = 2*np.pi*data['frecuencia']*x2/vel_sonido


# Agrego columna r
data['term1'] = data['in2_magnitud']*np.exp(1j*data['in2_fase']-data['kX1'])
data['term2'] = data['in1_magnitud']*np.exp(1j*data['in1_fase']-data['kX2'])
data['term3'] = data['in1_magnitud']*np.exp(1j*data['in1_fase']+data['kX2'])
data['term4'] = data['in2_magnitud']*np.exp(1j*data['in2_fase']+data['kX1'])
data['r'] = (data['term1'] - data['term2']) / (data['term3'] - data['term4']) 
data['abs(r)'] = np.abs(data['r'])
data['fase(r)'] = np.angle(data['r'])


# calculo Tau
data['tau_term1'] = data['out_magnitud'].apply(lambda x:10**(x/10)) / data['in1_magnitud'].apply(lambda x:10**(x/10))
data['tau_term2'] = 1 + data['abs(r)']**2 + 2*data['abs(r)'] * np.cos(2*data['kX1'] + data['fase(r)'])
data['tau'] = data['tau_term1'] * data['tau_term2']

# calculo de TL
data['TL'] = -10*np.log10(data['tau'])

# EXPORT
nombre_archivo = '/'.join(carpeta.split('/')[:-1]) + '/TL.csv'
output = data[['frecuencia','TL']].to_csv(nombre_archivo)

# PLOT
fig,ax = plt.subplots(3,1,figsize = [10,8])

ax[0].semilogx(
    data['frecuencia'],
    data['TL'],
 
)
ax[0].set_xlabel('Frecuencia (Hz)')
ax[0].set_ylabel('Transmission Loss (dB)')
ax[0].grid()


ax[1].semilogx(
    data['frecuencia'],
    data['out_magnitud'],
    label = 'Magnitud a la salida [dBFS]'
 
)
ax[1].semilogx(
    data['frecuencia'],
    data['in1_magnitud'],
    label = 'Magnitud a la entrada punto 1 [dBFS]'
)

ax[1].semilogx(
    data['frecuencia'],
    data['in2_magnitud'],
    label = 'Magnitud a la entrada punto 2 [dBFS]'
 
)
ax[1].set_xlabel('Frecuencia (Hz)')
ax[1].set_ylabel('Niveles relativos a la entrada y la salida]')
ax[1].grid()
ax[1].legend()

ax[2].semilogx(
    data['frecuencia'],
    data['out_coherencia'],
    label = 'Coherencia a la salida'
 
)
ax[2].semilogx(
    data['frecuencia'],
    data['in1_coherencia'],
    label = 'Coherencia a la entrada punto 1'
)

ax[2].semilogx(
    data['frecuencia'],
    data['in2_coherencia'],
    label = 'Coherencia a la entrada punto 2'
 
)
ax[2].set_xlabel('Frecuencia (Hz)')
ax[2].set_ylabel('Coherencias')
ax[2].grid()
ax[2].legend()

plt.show()
