import matplotlib.pyplot as plt
import numpy as np
from src.robin.decision_model.terms import EnumeratedFS


def dibuja_cto_enumerado(x, y, eti_leyenda:str):
    b, = plt.plot(x,y)
    # Coloca los valores del eje Y y leyendas 
    plt.yticks(np.arange(0,1.1,0.1))
    plt.legend([b], [eti_leyenda], loc='best')
    # Muestra la gráfica
    plt.show()


def dibuja_varios_CDEnumerado(ctosEnumerados:list, 
                              ctoUnion:EnumeratedFS=None, 
                              mostrarUnion:bool=True, enOtraGrafica:bool=False,
                              muestraCentroide:bool=False):
    l = []
    leyendas = []
    for cto in ctosEnumerados:
        x, y = cto.get_x_values(), cto.get_y_values()
        b, = plt.plot(x, y)
        l.append(b) 
        leyendas.append( cto.get_name() )
        # Coloca los valores del eje Y y leyendas 
    if mostrarUnion and not enOtraGrafica and ctoUnion!=None:
        b, = plt.plot(ctoUnion.get_x_values(),ctoUnion.get_y_values())
        l.append(b)
        leyendas.append(ctoUnion.get_name())
        if muestraCentroide:
            ctoUnion.genera_grafica(graphic_show=False, centroid_show=muestraCentroide)
    plt.legend(l, leyendas, loc='best')
    plt.yticks(np.arange(0,1.1,0.1))
    # Muestra la gráfica
    plt.show()
    if mostrarUnion and enOtraGrafica:
        ctoUnion.genera_grafica(centroid_show=muestraCentroide)
        # dibuja_cto_enumerado(ctoUnion.get_x_valores(),ctoUnion.get_y_valores(), 'Union')
