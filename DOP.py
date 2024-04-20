import numpy as np
from satelite import Satelite
from satelite import LLA_to_ECEF
from termcolor import colored
from tqdm import tqdm
from pyproj import Proj, transform
import warnings



def estimador(lista_satellite,position_index,user_position_array,user_position_array_FAKE,bias=3):
    c = 3e8
    i = 0
    
    #determinar la visibilidad de los satelites:
    lista_satelites_visibles = []
    for satellite in lista_satellite:
        satellite.satellite_visiblility(user_position_array)
        if  satellite.visibility[position_index] == True:
            lista_satelites_visibles.append(satellite)

    pseudorangos_verdaderos = np.ndarray((len(lista_satelites_visibles),))

    for i in range(len(lista_satelites_visibles)):
        satelite = lista_satelites_visibles[i]
        pseudorango = satelite.pseuedorango(position_index = position_index,ru = user_position_array,b=0)
        pseudorangos_verdaderos[i] = pseudorango
        
    vector_posicion_estimada = np.array(user_position_array_FAKE)
    vector_estado_acutal = np.concatenate((user_position_array_FAKE, np.array([bias])))

    for _ in range(200):

        A = []
        vector_residuales = np.empty_like(lista_satelites_visibles)
        for i in range(len(lista_satelites_visibles)):
            satelite = lista_satelites_visibles[i] #Obtener el satelite
            pseudorango = satelite.pseuedorango(position_index = position_index,ru = vector_posicion_estimada,b=vector_estado_acutal[3]) #calcular el pseudorango con las posiciones y bias estimados
            residual = pseudorangos_verdaderos[i]-pseudorango #obtener el residual
            vector_residuales[i] = residual #añadir el residual al vector
            #preparar para el calculo de la fila n de la matriz A:
            Xs = satelite.X[position_index]
            Ys = satelite.Y[position_index]
            Zs = satelite.Z[position_index]
            pseudorangoV = pseudorangos_verdaderos[i]
            Xu = vector_posicion_estimada[0]
            Yu = vector_posicion_estimada[1]
            Zu = vector_posicion_estimada[2]
            Bu = vector_estado_acutal [3]
            #calculo de la fila A:
            filaA = [-(Xs-Xu)/(pseudorangoV-Bu*c),-(Ys-Yu)/(pseudorangoV-Bu*c),-(Zs-Zu)/(pseudorangoV-Bu*c),c]
            A.append(filaA)
        A = np.array(A)
        vector_residuales_traspuesto = vector_residuales.reshape(-1,1) #transponer el vector de una sola fila a n filas y 1 columna
        incrementos_estimador =  (np.linalg.inv(A.T @ A) @ A.T)@vector_residuales_traspuesto
        vector_estado_acutal = vector_estado_acutal.reshape(-1,1)
        vector_estado_acutal = vector_estado_acutal + incrementos_estimador
        vector_estado_acutal = vector_estado_acutal.reshape(-1)
        vector_posicion_estimada = vector_estado_acutal[:3]
        if np.linalg.norm(user_position_array-vector_posicion_estimada) <= 1e-6:
            break
    return vector_estado_acutal


def estimador_ruido_blanco_gausiano(lista_satellite,position_index,user_position_array,user_position_array_FAKE,bias=3,ganancia_ruido = 80):
    c = 3e8
    i = 0
    
    #determinar la visibilidad de los satelites:
    lista_satelites_visibles = []
    for satellite in lista_satellite:
        satellite.satellite_visiblility(user_position_array)
        if  satellite.visibility[position_index] == True:
            lista_satelites_visibles.append(satellite)

    pseudorangos_verdaderos = np.ndarray((len(lista_satelites_visibles),))

    for i in range(len(lista_satelites_visibles)):
        satelite = lista_satelites_visibles[i]
        pseudorango = satelite.pseuedorango(position_index = position_index,ru = user_position_array,b=0) + ganancia_ruido *np.random.randn()
        pseudorangos_verdaderos[i] = pseudorango
        
    vector_posicion_estimada = np.array(user_position_array_FAKE)
    vector_estado_acutal = np.concatenate((user_position_array_FAKE, np.array([bias])))

    for _ in range(50):

        A = []
        vector_residuales = np.empty_like(lista_satelites_visibles)
        for i in range(len(lista_satelites_visibles)):
            satelite = lista_satelites_visibles[i] #Obtener el satelite
            pseudorango = satelite.pseuedorango(position_index = position_index,ru = vector_posicion_estimada,b=vector_estado_acutal[3]) #calcular el pseudorango con las posiciones y bias estimados
            residual = pseudorangos_verdaderos[i]-pseudorango #obtener el residual
            vector_residuales[i] = residual #añadir el residual al vector
            #preparar para el calculo de la fila n de la matriz A:
            Xs = satelite.X[position_index]
            Ys = satelite.Y[position_index]
            Zs = satelite.Z[position_index]
            pseudorangoV = pseudorangos_verdaderos[i]
            Xu = vector_posicion_estimada[0]
            Yu = vector_posicion_estimada[1]
            Zu = vector_posicion_estimada[2]
            Bu = vector_estado_acutal [3]
            #calculo de la fila A:
            filaA = [-(Xs-Xu)/(pseudorangoV-Bu*c),-(Ys-Yu)/(pseudorangoV-Bu*c),-(Zs-Zu)/(pseudorangoV-Bu*c),c]
            A.append(filaA)
        A = np.array(A)
        vector_residuales_traspuesto = vector_residuales.reshape(-1,1) #transponer el vector de una sola fila a n filas y 1 columna
        incrementos_estimador =  (np.linalg.inv(A.T @ A) @ A.T)@vector_residuales_traspuesto
        vector_estado_acutal = vector_estado_acutal.reshape(-1,1)
        vector_estado_acutal = vector_estado_acutal + incrementos_estimador
        vector_estado_acutal = vector_estado_acutal.reshape(-1)
        vector_posicion_estimada = vector_estado_acutal[:3]
        if np.linalg.norm(user_position_array-vector_posicion_estimada) <= 1e-6:
            break
    return vector_estado_acutal


def matrizP (lista_satellite,position_index,user_position_array,user_position_array_FAKE,bias=3,ganancia_ruido = 80,iteracionesP = 25):

    c = 3e8

    x = np.zeros_like(np.zeros([iteracionesP]))
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    b = np.zeros_like(x)

    for i in (range(iteracionesP)):
        ru_estado = estimador_ruido_blanco_gausiano(lista_satellite,position_index,user_position_array,user_position_array_FAKE,bias,ganancia_ruido)
        x[i] = ru_estado[0]
        y[i] = ru_estado[1]
        z[i] = ru_estado[2]
        b[i] = ru_estado[3]

    mx = np.sum(x)/iteracionesP
    my = np.sum(y)/iteracionesP
    mz = np.sum(z)/iteracionesP
    mcb = c*np.sum(b)/iteracionesP
    varianza_x = np.sum((x - mx)**2) / (iteracionesP - 1)
    varianza_y = np.sum((y - my)**2) / (iteracionesP - 1)
    varianza_z = np.sum((z - mz)**2) / (iteracionesP - 1)
    varianza_b = np.sum((b - mcb)**2) / (iteracionesP - 1)
    cov_xy = np.sum((x - mx) * (y - my)) / (iteracionesP - 1)
    cov_xz = np.sum((x - mx) * (z - mz)) / (iteracionesP - 1)
    cov_xb = np.sum((x - mx) * (b - mcb)) / (iteracionesP - 1)
    cov_yz = np.sum((y - my) * (z - mz)) / (iteracionesP - 1)
    cov_yb = np.sum((y - my) * (b - mcb)) / (iteracionesP - 1)
    cov_zb = np.sum((z - mz) * (b - mcb)) / (iteracionesP - 1)

    P = np.array([[varianza_x, cov_xy, cov_xz],
                  [cov_xy, varianza_y, cov_yz],
                  [cov_xz, cov_yz, varianza_z],
                  ])

    return P


def matrizF(x,y,z):

    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    # Ignorar las advertencias FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Convertir coordenadas ECEF a WGS84
    lon, lat, alt = transform(ecef, Proj(proj='latlong', ellps='WGS84', datum='WGS84'), x, y, z, radians=False)
    warnings.resetwarnings()
    # Convertir la latitud y longitud de grados a radianes
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Calcular los vectores este, norte y up
    e = np.array([
        -np.sin(lon_rad),
        np.cos(lon_rad),
        0
    ])

    n = np.array([
        -np.sin(lat_rad) * np.cos(lon_rad),
        -np.sin(lat_rad) * np.sin(lon_rad),
        np.cos(lat_rad)
    ])

    u = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])

    # Organizar los vectores en una matriz ortogonal (F)
    F = np.vstack((e, n, u))

    return F

class DOP:

    def __init__(self,lista_satellite,position_index,user_position_array,user_position_array_FAKE,user_position_estimated,bias=3,ganancia_ruido = 80) -> None:
        
        self.Pxyz = matrizP(lista_satellite,position_index,user_position_array,user_position_array_FAKE,bias=3,ganancia_ruido = 80,iteracionesP = 25)
        self.F = matrizF(user_position_estimated[0],user_position_estimated[1],user_position_estimated[2])
        self.Penu = self.F.T @ self.Pxyz @ self.F
        self.GDOP = self.calculate_GDOP()
        self.PDOP = self.calculate_PDOP()
        self.HDOP = self.calculate_HDOP()
        self.VDOP = self.calculate_VDOP()
        self.TDOP = self.calculate_TDOP()

    def calculate_GDOP(self):
        return np.sqrt(np.trace(self.Penu))

    def calculate_PDOP(self):
        return np.sqrt(np.sum(np.diag(self.Penu)[:3]))

    def calculate_HDOP(self):
        return np.sqrt(np.sum(np.diag(self.Penu)[:2]))

    def calculate_VDOP(self):
        return np.sqrt(self.Penu[2, 2])

    def calculate_TDOP(self):
        return np.sqrt(self.Penu[2, 2])
    def __str__(self):
        return f"GDOP: {self.GDOP}\nPDOP: {self.PDOP}\nHDOP: {self.HDOP}\nVDOP: {self.VDOP}\nTDOP: {self.TDOP}"
