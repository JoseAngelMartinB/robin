import numpy as np

from typing import Callable

from src.robin.decision_model.propositions import PDC, FuzzyAP
from src.robin.decision_model.terms import EnumeratedFS


class RD():
    """ Clase para modelar una regla difusa. 

    Se modela con dos componentes de tipo PDC (antecedente) y de tipo 
    PDA (consecuente)

        Attributes:
            antecedente (PDC): proposición difusa compuesta que representa el
            antecedente.
            consecuente (list[Callable]): proposición difusa compuesta que representa 
            el consecuente. 
    """
    def __init__(self, name:str, antecedent:PDC):
        self.name = name
        self.antecedent = antecedent

    # GETTERS
    def get_name(self) -> str:
        """Devuelve el nombre de la regla.

        Args:
            None

        Returns:
            str.
        """
        return self.name


    def get_antecedent(self) -> PDC:
        """Devuelve el antecedente.

        Args:
            None
            
        Returns:
            PDC.
        """
        return self.antecedent



class MandaniRule(RD):
    """ Clase para modelar una regla difusa tipo Mandani. 

    Se modela con dos componentes de tipo PDC (antecedente) y de tipo 
    PDA (consecuente)

        Attributes:
            antecedente (PDC): proposición difusa compuesta que representa el
            antecedente.
            consecuente (list[Callable]): proposición difusa compuesta que representa 
            el consecuente. 
    """
    def __init__(self, name:str, antecedent:PDC, consequent:FuzzyAP):
        RD.__init__(self, name, antecedent)
        self.consequent = consequent


    def get_consequent(self) -> FuzzyAP:
        """Devuelve el consecuente.

        Args:
            None
            
        Returns:
            PDA.
        """
        return self.consequent


    def membership_grade(self, values:list[float]) -> EnumeratedFS:
        """Devuelve la pertenencia a la regla, es decir, utiliza A y B, para resolver
        A=>B, con el operador de implicación de Mandani: 
                mu_R(x,y)=min(mu_A(X),mu_B(y)).

        Attributes:
            valores (list[float]): lista de valores para cada proposición

        Returns:
            EnumeratedFS.
        """
        mu_antecedent = self.get_antecedent().membership_grade(values)
        if mu_antecedent>0.0:
            mini = self.get_consequent().get_variable().get_lower_support()
            maxi = self.get_consequent().get_variable().get_upper_support()
            output_grades = [ (round(val,2),min(mu_antecedent,self.get_consequent().get_term().membership_grade(val))) for val in np.arange(mini,maxi,1,) ]
            x_values, y_values = zip(*output_grades)
            return EnumeratedFS('Out ' + self.get_name(), x_values, y_values)
        else:
            return EnumeratedFS('Out ', [], []) # PASAR A CDEnumerado VACÍO


    def __str__(self) -> str:
        """Devuelve los atributos de esta clase como str.

        Args:
            None
        
        Returns:
            str.
        """
        return 'IF' + str(self.get_antecedent()) + 'THEN ' + str(self.get_consequent())


class TSKRule(RD):
    """ Clase para modelar una regla difusa tipo TSK. 

    Se modela con dos componentes de tipo PDC (antecedente) y de tipo 
    PDA (consecuente)

        Attributes:
            antecedente (PDC): proposición difusa compuesta que representa el
            antecedente.
            consecuente (list[float]): proposición difusa compuesta que representa 
            el consecuente. 
    """
    def __init__(self, name:str, antecedent:PDC, consequent:Callable):
        RD.__init__(self, name, antecedent)
        self.consequent = consequent


    def get_consequent(self) -> Callable:
        """Devuelve el consecuente.

        Args:
            None
            
        Returns:
            Callable.
        """
        return self.consequent


    def membership_grade(self, values:list[float]) -> float:
        """Devuelve la pertenencia a la regla, es decir, utiliza A y B, para resolver
        A=>B, con el operador de implicación de Mandani: 
                mu_R(x,y)=min(mu_A(X),mu_B(y)).

        Attributes:
            valores (list[float]): lista de valores para cada proposición

        Returns:
            float.
        """
        return self.get_antecedent().membership_grade(values)


    def __str__(self) -> str:
        """Devuelve los atributos de esta clase como str.

        Args:
            None
        
        Returns:
            str.
        """
        return self.get_name() + ': IF' + str(self.get_antecedent()) + 'THEN ' + str(self.get_consequent())
