import numpy as np
from pyproj import Proj, transform
import warnings


class Satelite:
    visibility = []
    def __init__(self, identificador, inclinacion, longitud_del_nodo_ascendente, r,offset):
        self.identificador = identificador 
        self.inclinacion = inclinacion * np.pi/180
        self.longitud_del_nodo_ascendente = longitud_del_nodo_ascendente * np.pi/180
        self.r = r
        self.offset = offset * np.pi/180
        self.X = []
        self.Y = []
        self.Z = []
        

    def __str__(self) ->str:
            
            return f"Satelite {self.identificador}:\n"\
            f"Inclinación: {np.degrees(self.inclinacion)} grados\n"\
            f"Longitud del Nodo Ascendente: {np.degrees(self.longitud_del_nodo_ascendente)} grados\n"\
            f"Radio orbital: {self.r} km\n"\
            f"Coordenadas:\n"\
            f"  X: {self.X}\n"\
            f"  Y: {self.Y}\n"\
            f"  Z: {self.Z}"
    def update_position(self, u):

        self.X.append(self.r*np.cos(u + self.offset) *np.cos(self.longitud_del_nodo_ascendente) - self.r *np.sin(u+self.offset)*np.cos(self.inclinacion)*np.cos(self.longitud_del_nodo_ascendente))
        self.Y.append(self.r*np.cos(u + self.offset) *np.sin(self.longitud_del_nodo_ascendente) + self.r *np.sin(u+self.offset)*np.cos(self.inclinacion)*np.cos(self.longitud_del_nodo_ascendente))
        self.Z.append( self.r*np.sin(u + self.offset) *np.sin(self.inclinacion))
    
    def ecef_to_wgs84(self, x, y, z):
        # Definir el sistema de coordenadas ECEF
        ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        # Ignorar las advertencias FutureWarning
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Convertir coordenadas ECEF a WGS84
        lon, lat, alt = transform(ecef, Proj(proj='latlong', ellps='WGS84', datum='WGS84'), x, y, z, radians=False)
        warnings.resetwarnings()
        return lat, lon, alt
        """x = np.array([x]).reshape(np.array([x]).shape[-1], 1)
        y = np.array([y]).reshape(np.array([y]).shape[-1], 1)
        z = np.array([z]).reshape(np.array([z]).shape[-1], 1)
        a=6378137
        a_sq=a**2
        e = 8.181919084261345e-2
        e_sq = 6.69437999014e-3

        f = 1/298.257223563
        b = a*(1-f)

        # calculations:
        r = np.sqrt(x**2 + y**2)
        ep_sq  = (a**2-b**2)/b**2
        ee = (a**2-b**2)
        f = (54*b**2)*(z**2)
        g = r**2 + (1 - e_sq)*(z**2) - e_sq*ee*2
        c = (e_sq**2)*f*r**2/(g**3)
        s = (1 + c + np.sqrt(c**2 + 2*c))**(1/3.)
        p = f/(3.*(g**2)*(s + (1./s) + 1)**2)
        q = np.sqrt(1 + 2*p*e_sq**2)
        r_0 = -(p*e_sq*r)/(1+q) + np.sqrt(0.5*(a**2)*(1+(1./q)) - p*(z**2)*(1-e_sq)/(q*(1+q)) - 0.5*p*(r**2))
        u = np.sqrt((r - e_sq*r_0)**2 + z**2)
        v = np.sqrt((r - e_sq*r_0)**2 + (1 - e_sq)*z**2)
        z_0 = (b**2)*z/(a*v)
        h = u*(1 - b**2/(a*v))
        phi = np.arctan((z + ep_sq*z_0)/r)
        lambd = np.arctan2(y, x)"""


        return phi*180/np.pi, lambd*180/np.pi, h

    def get_lat_lon_for_all_points(self):
        # Crear una lista para almacenar las coordenadas geográficas
        lat_list = []
        lon_list = []

        # Iterar sobre las coordenadas cartesianas y convertirlas
        for i in range(len(self.X)):
            lat, lon, alt = self.ecef_to_wgs84(self.X[i], self.Y[i], self.Z[i])
            lat_list.append(lat)
            lon_list.append(lon)

        return lat_list,lon_list
    

    def satellite_visiblility(self, ru, min_angle_deg = 0):
        
        """
        Determina si un satélite es visible desde un punto de observación en la Tierra.

        Parámetros:
        - ru (array-like): Vector de posición del punto de observación en la Tierra (NumPy array).
        
        - min_angle_deg (float, opcional): El ángulo mínimo de visibilidad en grados.
        
        Retorna:
        - bool: True si el satélite es visible, False si no lo es.

        La función calcula el ángulo entre el punto de observación y el satélite.
        Si el ángulo está dentro del rango permitido (entre -90° + min_angle_deg y 90° - min_angle_deg),
        se considera que el satélite es visible desde el punto de observación.

        Ejemplo de uso:
        ru = np.array([0, 0, 0])  # Vector de posición del punto de observación
        rsat = np.array([1000, 1000, 1000])  # Vector de posición del satélite
        es_visible = satellite_visiblility(ru, rsat, min_angle_deg=5)
        print("¿El satélite es visible?", es_visible)
        """
        self.visibility = []
        for i in range(len(self.X)):
            rsat = np.array([self.X[i],self.Y[i],self.Z[i]])
            angulo_vis = np.arccos((ru.dot(rsat-ru))/(np.linalg.norm(ru)*np.linalg.norm(rsat-ru)))

            if angulo_vis >= -np.pi/2 + np.deg2rad(min_angle_deg) and angulo_vis <= np.pi/2 - np.deg2rad(min_angle_deg):
                self.visibility.append(True)
            else:
                self.visibility.append(False)
    
    def pseuedorango(self,position_index,ru,b=0):
        rsat = np.array([self.X[position_index],self.Y[position_index],self.Z[position_index]])
        vector = rsat-ru[:3]
        c =3e8
        pseudorango = np.linalg.norm(vector) + c*b
        return pseudorango

         

def LLA_to_ECEF(lat, lon, alt):
    ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = transform(lla, ecef, lon, lat, alt, radians=False)

    return np.array([x, y, z])

if __name__ == '__main__':
    lat=41.27551963562848
    lon = 1.9872230495391976
    alt = 2
    print(LLA_to_ECEF(lat,lon,alt))

