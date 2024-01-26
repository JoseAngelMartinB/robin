""" Clase padre de todos los tipos de variables"""
from src.robin.decision_model.terms import MembershipFS

class Variable():
    """ Clase para modelar una variable de cualquier tipo. Es la clase padre de todas.

        Tendrá las componentes nombre.

        Attributes:
            name (str): str que contiene el nombre de la variable.
            support (tuple[float]): lista de 2 float que definen el dominio de la 
            variable como un valor inferior del dominio y el superior como un 
            intervalo cerrado.
            term (float): valor real del término.  
    """
    def __init__(self, name:str, position: int):
        self.name = name
        self.position = position


    # GETTERS
    def get_name(self) -> str:
        """Devuelve el nombre de la variable lingüística.

        Args:
            None
        
        Returns:
            str.
        """
        return self.name


    def get_position(self) -> float:
        """ Devuelve la posicion en el lugar de entrada de la var. linguística.
            
        Args:
            None
        
        Returns:
            float.
        """
        return self.position

    # SETTERS
    # Faltan porque no tengo claro si es interesante permitir modificar los CDs
    # que forman parte de la VL


    # FUNCIONES PROPIAS
    def __str__(self) -> str:
        """Devuelve lesta clase como str.

        Args:
            None

        Returns:
            str.
        """
        return str(self.name) + '\n'

class RealV(Variable):
    """ Clase para modelar una variable real 

        Esta clase modela una variable real como float.

        Attributes:
            nombre (str): str que contiene el nombre de la variable (heredada de Variable).
            dominio (tuple[float]): lista de 2 float que definen el dominio de la 
            variable como un valor inferior del dominio y el superior como un 
            intervalo cerrado (heredada de Variable).
            CDs (str): lista con los conjuntos difusos que forman los
            términos primarios.  
    """
    def __init__(self, name:str, position: int, domain:list[float]):
        Variable.__init__(self,name, position)
        self.domain = domain

    # GETTERS
    def get_lower_domain(self) -> float:
        """Devuelve el mínimo valor del dominio de la var. linguística.

        Args:
            None
        
        Returns:
            float.
        """
        return self.domain[0]

    def get_upper_domain(self) -> float:
        """ Devuelve el máximo valor del dominio de la var. linguística.
            
        Args:
            None
        
        Returns:
            float.
        """
        return self.domain[1]

    # SETTERS
    # Faltan porque no tengo claro si es interesante permitir modificar los CDs
    # que forman parte de la VL


    # FUNCIONES PROPIAS
    def __str__(self) -> str:
        """Devuelve lesta clase como str.

        Args:
            None

        Returns:
            str.
        """
        return str(self.name) + ' ' + str(self.get_lower_domain()) + ' ' \
                    + str(self.get_upper_domain())  + '\n'

class CategoryV(Variable):
    """ Clase para modelar una variable categórica 

        Esta clase modela una variable categórica con categorías modeladas 
        como cadenas.

        Attributes:
            nombre (str): str que contiene el nombre de la variable (heredada de Variable).
            dominio (tuple[float]): lista de 2 float que definen el dominio de la 
            variable como un valor inferior del dominio y el superior como un 
            intervalo cerrado (heredada de Variable).
            CDs (str): lista con los conjuntos difusos que forman los
            términos primarios.  
    """
    def __init__(self, name:str, position: int, categories:list[str]):
        Variable.__init__(self,name, position)
        self.categories = categories

    # GETTERS
    def get_categories(self) -> float:
        """Devuelve el termino primario de nombre "name".

        Args:
            None.

        Returns:
            float.
        """
        return self.categories

    def is_in_categories(self, category:str) -> bool:
        """Devuelve el termino primario de nombre "name".

        Args:
            None.

        Returns:
            float.
        """
        return category in self.get_categories()

    # SETTERS
    # Faltan porque no tengo claro si es interesante permitir modificar los CDs
    # que forman parte de la VL


    # FUNCIONES PROPIAS
    def __str__(self) -> str:
        """Devuelve lesta clase como str.

        Args:
            None

        Returns:
            str.
        """
        return str(self.name) + ' ' + str(self.get_categories()) + '\n'

class LinguisticV(Variable):
    """ Clase para modelar una variable lingüística 

        Esta clase modela una variable lingüística reducida. No modela los 5 campos
        de la definición, pero sí es funcional para los problemas a tratar.

        Attributes:
            nombre (str): str que contiene el nombre de la variable.
            dominio (tuple[float]): lista de 2 float que definen el dominio de la 
            variable como un valor inferior del dominio y el superior como un 
            intervalo cerrado.
            CDs (list[CDifuso]): lista con los conjuntos difusos que forman los
            términos primarios.  
    """
    def __init__(self, name:str, position: int, support:list[float], values:dict):
        Variable.__init__(self,name, position)
        self.support = support
        self.values = values


    # GETTERS
    def get_lower_support(self) -> float:
        """Devuelve el mínimo valor del dominio de la var. linguística.

        Args:
            None
        
        Returns:
            float.
        """
        return self.support[0]

    def get_upper_support(self) -> float:
        """ Devuelve el máximo valor del dominio de la var. linguística.
            
        Args:
            None
        
        Returns:
            float.
        """
        return self.support[1]
    

    def get_values(self) -> float:
        """Devuelve el termino primario de nombre "name".

        Args:
            None.

        Returns:
            float.
        """
        return self.values

    def get_fuzzy_set(self, name:str) -> MembershipFS:
        """Devuelve el termino primario de nombre "name".

        Args:
            name (str): nombre del FS a devolver.

        Returns:
            MembershipFS.
        """
        return self.values[name]

    # SETTERS
    # Faltan porque no tengo claro si es interesante permitir modificar los CDs
    # que forman parte de la VL


    # FUNCIONES PROPIAS
    def __str__(self) -> str:
        """Devuelve lesta clase como str.

        Args:
            None

        Returns:
            str.
        """
        chain = ''
        for elem in self.values:
            chain += '\n' + str(elem)
        return str(self.name) + ' ' + str(self.support) + ' ' + chain + '\n'
