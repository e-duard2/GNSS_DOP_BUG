import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
from termcolor import colored
from satelite import Satelite
from satelite import LLA_to_ECEF
import DOP
from tqdm import tqdm


R_Tierra = 6378e3
h_orbita = 20e6
h_satelite = h_orbita + R_Tierra
inclinacion = 55
longitud_del_nodo_ascendente = 0
offset = 60
velocidad_angular = 2*np.pi/(12*3600)

num_grupos = 6  # Número total de grupos
satelites_por_grupo = 4  # Número de satélites por grupo

constelacion = np.empty((num_grupos * satelites_por_grupo,), dtype=object)

#posición real del usuario EETAC
user_lat=41.27551963562848
user_lon = 1.9872230495391976
user_alt = 2

ru = LLA_to_ECEF(user_lat,user_lon,user_alt)
#posión falsa del usuario ETSID
valencia_coords = [39.4699, -0.3763]
r_u_fake = LLA_to_ECEF(valencia_coords[0],valencia_coords[1],2)


for grupo in range(num_grupos):
    for numero in range(1, satelites_por_grupo + 1):
        index = grupo * satelites_por_grupo + (numero - 1)
        identificador = f"{(grupo+1)*numero}"
        satelite = Satelite(identificador=identificador, inclinacion=inclinacion, longitud_del_nodo_ascendente= grupo * offset, r=h_satelite, offset=numero * 90 )
        constelacion[index] = satelite


# Simulación del movimiento orbital
tiempo= np.arange(0,12*3600,20*60)

# Crear un vector de objetos datetime
datetime_vector = []

# Fecha y hora actual del sistema
start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


for t in tiempo:
    for i in range(len(constelacion)):
        constelacion[i].update_position(velocidad_angular*t)
    delta = timedelta(seconds=float(t))
    datetime_vector.append(start_date + delta)

"""MÍNIMOS CUADRADOS para ESTIMAR la POSICIÓN y el “BIAS” a partir de MEDIDAS PERFECTAS de PSEUDODISTANCIA"""

#estimación DOP senñal sin ruidio y con bias = 3s
ru_estado = DOP.estimador(constelacion,0,ru,r_u_fake)
ru_estimada = ru_estado[:3]
print(colored("""\nMÍNIMOS CUADRADOS para ESTIMAR la POSICIÓN y el “BIAS” a partir de MEDIDAS PERFECTAS de PSEUDODISTANCIA""",color='cyan',attrs=['bold','underline']))
print('posición verdaderea del usuario X,Y,Z',ru)
print('posición estimada del usuario X,Y,Z',ru_estimada)
print(colored(f'Bias {ru_estado[3]}s','yellow'))
print(colored(f'''error realtivo % [X,Y,Z] {str(np.abs((ru_estimada-ru))/ru*100)}''','green'))

print(colored(f"""Lat/lon alt estimada: {satelite.ecef_to_wgs84(ru_estimada[0],ru_estimada[1],ru_estimada[2])}""",color='magenta'))

"""ESTIMACIÓN de la POSICIÓN y del “BIAS” por MÍNIMOS CUADRADOS, a partir de MEDIDAS RUIDOSAS de PSEUDODISTANCIA, y OBTENCIÓN de la DOP"""

P = DOP.matrizP(constelacion,0,ru,r_u_fake)
print('matriz P para el primer instante:')
print(P)

F = DOP.matrizF(ru_estimada[0],ru_estimada[1],ru_estimada[2])
print('Matriz F:')
print(F)
print('Matriz F traspuesta:')
print(F.T)
Penu = (F.T@P)@F
print(Penu)
dop = DOP.DOP(constelacion,0,ru,r_u_fake,ru_estimada)
print(dop)

GDOP = [0]*len(tiempo)
PDOP = [0]*len(tiempo)
HDOP = [0]*len(tiempo)
VDOP = [0]*len(tiempo)
TDOP = [0]*len(tiempo)
for position_index in tqdm(range(len(tiempo)), desc="Calculando"):
    ru_estado = DOP.estimador_ruido_blanco_gausiano(constelacion,0,ru,r_u_fake)
    ru_estimada = ru_estado[:3]
    dop = DOP.DOP(constelacion,position_index,ru,r_u_fake,ru_estimada)
    GDOP[int(position_index)]=dop.GDOP
    PDOP[int(position_index)]=dop.PDOP
    HDOP[int(position_index)]=dop.HDOP
    VDOP[int(position_index)]=dop.VDOP
    TDOP[int(position_index)]=dop.TDOP



#This code breaks the execution unexpectedly 

plt.plot(datetime_vector, GDOP)
plt.plot(datetime_vector, PDOP)
plt.plot(datetime_vector, HDOP, '--')
plt.plot(datetime_vector, VDOP, '--')
plt.plot(datetime_vector, TDOP)

# Mostrar el gráfico
plt.show()

"""
Traceback (most recent call last):
  File "ruta_al_archivo/practica2.py", line 109, in <module>
    plt.plot(datetime_vector, GDOP)
  File "ruta_a_matplotlib/pyplot.py", line 3590, in plot
    return gca().plot(
           ^^^^^^^^^^^
  File "ruta_a_mplot3d/axes3d.py", line 1622, in plot
    self.auto_scale_xyz(xs, ys, zs, had_data)
  File "ruta_a_mplot3d/axes3d.py", line 613, in auto_scale_xyz
    self.xy_dataLim.update_from_data_xy(
  File "ruta_a_transforms.py", line 951, in update_from_data_xy
    path = Path(xy)
           ^^^^^^^^
  File "ruta_a_path.py", line 129, in __init__
    vertices = _to_unmasked_float_array(vertices)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "ruta_a_cbook.py", line 1345, in _to_unmasked_float_array
    return np.asarray(x, float)
           ^^^^^^^^^^^^^^^^^^^^
TypeError: float() argument must be a string or a real number, not 'datetime.datetime'

"""
