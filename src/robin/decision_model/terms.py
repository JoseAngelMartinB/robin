""" Clase para manejar los terminos, especialmente en DocString"""

from typing import Callable
import matplotlib.pyplot as plt
import numpy as np


class Term():
    """
    Clase para manejar los términos
    """

    def __init__(self) -> None:
        pass


class RealT(Term):
    """
    Clase para manejar términos tipo Real
    """

    def __init__(self, real: float) -> None:
        super().__init__()
        self.real = real

    # GETTERS
    def get_real(self) -> float:
        """
        Devuelve el valor real
        """
        return self.real

    def __str__(self) -> str:
        """
        Devuelve el termino real como cadena
        """
        return str(self.get_real())


class CategoryT(Term):
    """
    Clase para manejar términos tipo Categoría
    """

    def __init__(self, category: str) -> None:
        super().__init__()
        self.category = category

    # GETTERS
    def get_category(self) -> str:
        """
        Devuelve el valor real
        """
        return self.category


################################
# COMIENZA LA CLASE MembershipFS Y EnumerateFS
################################
def triangular(in_val, values:tuple) -> float:
    """ Función que calcula una función de pertenencia triangular

        Args:
                in_val (float): El valor float para calcular su pertenencia.
                values (tuple): 3 valores float que definen la función de pertenencia

        Returns:
                float: valor de pertenencia de val al conjunto representado por valores.
    """
    if in_val<values[0]:
        return 0.0
    if in_val>=values[0] and in_val<values[1]:
        return (in_val-values[0])/(values[1]-values[0])
    if in_val>=values[1] and in_val<=values[2]:
        return (values[2]-in_val)/(values[2]-values[1])
    if in_val>values[2]:
        return 0.0


def trapezoidal(in_val, values:tuple) -> float:
    """ Función que calcula una función de pertenencia trapezoidal

        Args:
                in_val (float): El valor float para calcular su pertenencia.
                values (tuple): 4 valores float que definen la función de pertenencia

        Returns:
                float: valor de pertenencia de val al conjunto representado por valores.
    """
    if in_val<values[0]:
        return 0.0
    if in_val>=values[0] and in_val<values[1]:
        return (in_val-values[0])/(values[1]-values[0])
    if in_val>=values[1] and in_val<=values[2]:
        return 1
    if in_val>=values[2] and in_val<=values[3]:
        return (values[3]-in_val)/(values[3]-values[2])
    if in_val>values[3]:
        return 0


class MembershipFS(Term):
    """ Clase para modelar una conjunto difuso 

        Esta clase modela un conjunto difuso con cualquier tipo de función de 
        pertenencia. Se debe pasar la función de pertenencia en la creación 
        del objeto

        Attributes:
                nombre (str): str que contiene el nombre del conjunto.
                valores (tuple[float]): tuple de floats que definen el conjunto difuso.
                func_pert (Callable): función que calcula la pertenencia de los valores.  
    """
    def __init__(self, name:str, function: Callable[[tuple],float], values:tuple):
        self.name = name
        self.values = values
        self.function = function


    # GETTERS
    def get_name(self) -> str:
        """Devuelve el nombre del MembershipFS.

        Args:
            None

        Returns:
            str.
        """
        return self.name


    def get_values(self) -> tuple[float]:
        """Devuelve los valores del MembershipFS.

        Args:
            None

        Returns:
            tuple of a indeterminate number of floats.
        """
        return self.values


    def get_membership(self) -> Callable:
        """Devuelve la función python utilizada como función de pertenencia.

        Args:
            None

        Returns:
            Callable.
        """
        return self.function


    #SETTERS
    def set_name(self, name:str) -> None:
        """Asigna el nombre del MembershipFS.

        Args:
            name: str.

        Returns:
            None.
        """
        self.name = name


    def set_values(self, values:tuple[float]) -> None:
        """Asigna los valores a utilizar por la función de pertenencia.

        Args:
            values: tuple of an indeterminate number of float.

        Returns:
            None.
        """
        self.values = values

    def set_func_pert(self, m_function:Callable[[tuple],float]) -> None:
        """Asigna la función python a utilizar como función de pertenencia.

        Args:
            m_function: función a utilizar, debe utilizar como parámetros
            una tupla de float y un valor float.

        Returns:
            None.
        """
        self.function = m_function

    # MÉTODOS PROPIOS
    def __str__(self) -> str:
        """Devuelve los atributos de esta clase como str.

         Args:
            None
            
        Returns:
                str.
        """
        return str(self.name) + ': ' + str(self.values)


    def membership_grade(self, in_val:float) -> float:
        """Calcula la pertenencia de in_val al conjunto difuso que modela esta clase 
        según la función en self.function.

        Args:
                in_val (float): El valor float para calcular su pertenencia.

        Returns:
                float.
        """
        return self.function( in_val, self.values )


class EnumeratedFS():
    """ Clase para modelar una conjunto difuso con enumeración de elementos

        Modela un conunto difuso con la función enumerada.

        Attributes:
                x_valores (list[float]): valor del que se tiene la pertenencia.
                y_valores (list[float]): valor de pertenencia.
                func_pert (Callable): función que calcula la pertenencia de los valores.  
    """
    def __init__(self, name:str, x_values: tuple[float], y_values:tuple[float]):
        self.name = name
        self.x_values = x_values
        self.y_values = y_values


    # GETTERS
    def get_name(self) -> str:
        """Devuelve el nombre del EnumeratedFS.

        Args:
            None

        Returns:
            str.
        """
        return self.name


    def get_x_values(self) -> tuple:
        """Devuelve los valores de X del MembershipFS.

        Args:
            None

        Returns:
            tuple of a indeterminate number of floats.
        """
        return self.x_values


    def get_y_values(self) -> tuple:
        """Devuelve los valores de Y de MembershipFS.

        Args:
            None

        Returns:
            tuple of a indeterminate number of floats.
        """
        return self.y_values

    # FUNCIONES PROPIAS
    def genera_grafica(self, legend_label:str='', graphic_show:bool=True,
                    centroid_show:bool=True) -> None:
        """Devuelve la gráfica del EnumeratedFS.

        Args:
            legend_label(str): la leyenda a mostrar
            graphic_show(bool): si se muestra la gráfica
            centroid_show(bool): en caso de mostrar la gráfica si se desea mostrar 
                la ubicación del centroide 

        Returns:
            None.
        """
        line, = plt.plot(self.get_x_values(), self.get_y_values())
        x_line2d = [line]
        list_legends = []
        if legend_label!= '':
            list_legends = [legend_label]
        else:
            list_legends = [self.get_name()]
        if centroid_show:
            cent = self.centroide()
            line, = plt.plot([cent,cent],[0,1])
            x_line2d.append(line)
            list_legends.append( 'defuzz val = ' + str( round(cent,2) ) )
        # Coloca los valores del eje Y y leyendas
        plt.yticks(np.arange(0,1.1,0.1))
        plt.legend(x_line2d, list_legends, loc='best')
        # Muestra la gráfica
        if graphic_show:
            plt.show()

    def centroide(self) -> float:
        """Devuelve el centroide de este EnuemratedFS.

        Args:
            None

        Returns:
            float.
        """
        nume = sum( [a*b for a,b in zip(self.get_x_values(),self.get_y_values())] )
        deno = sum(self.get_y_values())
        return nume/deno


# función que una varios conjuntos enumerados en uno
def enumerated_fs_union(enumerated_sets:list, name:str) -> EnumeratedFS:
    """Devuelve un EnumeratedFS que es la unión/agregación de todos los
        EnumeratedFS en ctosEnuemr.
    
    Args:
        EnumeratedSets(list)
        name(str)

    Returns:
        EnumeratedFS.
    """
    x_union = []
    y_union = []
    for cto in enumerated_sets:
        comp_x, comp_y = cto.get_x_values(), cto.get_y_values()
        if not y_union: # Equivalente a "y_union==[]"
            x_union = comp_x
            y_union = comp_y
        else:
            x_union = comp_x
            y_union = [ max(e,e1) for e,e1 in zip(y_union,comp_y) ]
    return EnumeratedFS(name, x_union, y_union)
