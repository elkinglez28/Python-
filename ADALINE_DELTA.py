# -*- coding: utf-8 -*-
"""
@author: Elkin Ramirez
Nombre: ADALINE Regla Delta
"""
# Importar bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ADALINE():
    # Constructor
    def __init__(self,d,xi,n_muestras,wi,fac_ap,epochs,precision,w_ajustado):
        self.d = d
        self.xi = xi
        self.n_muestras = n_muestras
        self.wi = wi
        self.fac_ap = fac_ap
        self.epochs = epochs
        self.precision = precision
        self.y = 0 # salida de la red
        self.w_ajustado = w_ajustado
    
    def Entrenamiento(self):
        E =1 # Error de salida
        E_ac = 0 # error actual
        Error_prev = 0 # error anterior
        Ew = 0 # Error cuadratico medio
        E_red = [] # error de la red
        E_total = 0 # error total
        
        while (np.abs(E) > self.precision):
            Error_prev = Ew
            for i in range(self.n_muestras):
                self.y = sum(self.xi[i,:] * self.wi) # calculo de la salida de la red
                E_ac = (self.d[i] - self.y) # calculo del error
                self.wi = self.wi + (self.fac_ap * E_ac * self.xi[i,:]) # ajustar los pesos
                
                E_total = E_total + ((E_ac)**2)
            
            # Calcular el error cuadratico medio
            Ew = ((1/self.n_muestras) * (E_total))
            E = (Ew - Error_prev) # Error de la red
            E_red.append(np.abs(E))
            self.epochs += 1
        return self.wi, self.epochs, E_red
    
    def F_operacion(self):
        salida = []
        for j in range(self.n_muestras):
            self.y = sum(self.xi[j,:] * self.w_ajustado)
            salida.append(self.y)
        return salida

# Ciclo principal
if __name__ == "__main__":
    # leer la tabla de excel
    tabla = pd.ExcelFile("adaline_Tdata.xlsx")
    v_datos = tabla.parse("Hoja1")
    # Comvertir los datos de la tabla en una matriz
    v_datos = np.array(v_datos)
    # datos de entrada xi
    xi = v_datos[:,0:3]
    # Valores deseados
    d = v_datos[:,3]
    # numero de muestras
    n_muestras = len(d)
    # establecer el vector de pesos w
    wi = np.array([3.12,2.00,1.86])
    # factor de aprendizaje
    fac_ap = 0.3
    # epocas
    epochs = 0
    precision = 0.000000001
    w_ajustado = []
    # Inicializar la Red ADALINE
    red = ADALINE(d,xi,n_muestras,wi,fac_ap,epochs,precision,w_ajustado)
    w_ajustado, epocas, error = red.Entrenamiento()
    # Grafica
    plt.ylabel('Error',Fontsize = 12)
    plt.xlabel('Épocas',Fontsize = 12)
    plt.title("ADALINE, Regla Delta")
    x = np.arange(epocas)
    plt.plot(x,error,'m->',label="Error cuadratico")
    plt.legend(loc='upper right')
    plt.show()
    print("Pesos ajustados",w_ajustado)
    #xi2 = v_datos[:,4:7]
    #d2 = v_datos[:,7]
    red = ADALINE(d,xi,n_muestras,wi,fac_ap,epochs,precision,w_ajustado)
    salidas = red.F_operacion()
    print("Salidas de la red: ", salidas)
