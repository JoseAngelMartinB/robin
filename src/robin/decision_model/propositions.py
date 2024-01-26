"""
proposiciones herencia
"""
from typing import Callable

from src.robin.decision_model.variables import Variable, RealV, CategoryV, LinguisticV
from src.robin.decision_model.terms import Term, RealT, CategoryT, MembershipFS

class AtomicProposition():
    """ Clase para modelar una proposición atómica

        Esta clase modela las proposiciones atómicas, por ejemplo:
        V es T

        Attributes:
            antecedent(LinguisticV): Variable Lingüística de la PDA.
            CL (CL): término primario de la PDA.  
    """
    def __init__(self, variable:Variable, term:Term):
        self.variable = variable
        self.term = term

    # GETTERS
    def get_variable(self) -> Variable:
        """Devuelve la variable.

        Args:
            None

        Returns:
            Variable.
        """
        return self.variable

    def get_term(self) -> Term:
        """Devuelve el término primario.

        Args:
            None

        Returns:
            Term.
        """
        return self.term


    # GETTERS
    def set_variable(self, variable:Variable) -> None:
        """Devuelve la variable.

        Args:
            variable (Variable): variable a asignar.

        Returns:
            None
        """
        self.variable = variable

    def set_term(self, term:Term) -> None:
        """Devuelve el término independiente.

        Args:
            term(MembershipFS): conjunto (término) a asignar.

        Returns:
            None
        """
        self.term = term


class CompoundProposition():
    """ Clase para modelar una proposición compuesta. 

    Se modela con dos listas: (1) la primera tiene como componentes objetos AP 
    y/o CP, (2) La segunda tiene conectivas implementadas como funciones (Callable)

        Attributes:
            proposiciones (list): Lista con las proposiciones que componen la PDC.
            conectivas (list[Callable]): conectivas entre las PDAs que componen el 
            atributo "proposiciones". Tendrá una longitud un elemento inferior al
            atributo "proposiciones". 
    """
    def __init__(self, propos:list, functions:dict ):
        # Creación de las proposiciones a partir de "propos"
        self.proposiciones = []
        # Creación de las conectivas a partir de "propos" y funciones
        self.conectivas_texto = []
        self.conectivas_funcion = []


    # GETTERS
    def get_proposicion(self, pos:int) -> AtomicProposition:
        """Devuelve la variable de la proposición.

        Args:
            pos(int): posición de la proposición a recuperar.
        
        Returns:
            PDA.
        """
        return self.proposiciones[pos]

    def get_num_proposiciones(self) -> int:
        """Devuelve el número de proposiciones.

        Args:
            None
        
        Returns:
            int.
        """
        return len(self.proposiciones)

    def get_conectiva_texto(self, pos:int) -> str:
        """Devuelve la t_norma/t_conorma como texto entre las proposiciones difusas
        en las posiciones pos-1 y pos+1.

        Args:
            pos(int): posición de la conectiva a devolver

        Returns:
            str.
        """
        return self.conectivas_texto[pos]

    def get_conectiva_funcion(self, pos:int) -> Callable:
        """Devuelve el t_norma/t_conorma a utilizar entre las proposiciones difusas
        en las posiciones pos-1 y pos+1.

        Args:
            pos(int): posición de la función a devolver

        Returns:
            Callable.
        """
        return self.conectivas_funcion[pos]


class RealAP(AtomicProposition):
    """ Clase para modelar una proposición atómica real

        Esta clase modela las proposiciones reales atómicas, por ejemplo:
        T es 12

        Attributes:
            variable(RealV): Variable real de la RealAP.
            term (RealT): término primario de la RealAP.  
    """
    def __init__(self, variable:RealV, term:RealT):
        AtomicProposition.__init__(self, variable, term)


    # GETTERS


    # FUNCIONES PROPIAS
    def membership_grade(self, value:float) -> float:
        """Devuelve el valor de pertenencia de "value" al termino primario 
        (valor real).

        Args:
            value(float): valor a calcular su pertenencia.

        Returns:
            float: valor de pertenencia
        """
        if round(value,8) == round( float(self.get_term().get_real()), 8):
            return 1.0
        return 0.0


    def __str__(self) -> str:
        """Devuelve los atributos de esta clase como str.

        Args:
            None.

        Returns:
            str.
        """
        return str(self.get_variable().get_name()) + ' es ' + str(self.get_term())


class CategoricalAP(AtomicProposition):
    """ Clase para modelar una proposición atómica categorica

        Esta clase modela las proposiciones reales categoricas, por ejemplo:
        Tipo es 'Citroen'

        Attributes:
            variable(RealV): Variable real de la RealAP.
            term (RealT): término primario de la RealAP.  
    """
    def __init__(self, variable:CategoryV, term:CategoryT):
        AtomicProposition.__init__(self, variable, term)


    # GETTERS


    # FUNCIONES PROPIAS
    def membership_grade(self, value:str) -> float:
        """Devuelve el valor de pertenencia de "value" al termino primario 
        (valor real).

        Args:
            value(float): valor a calcular su pertenencia.

        Returns:
            float: valor de pertenencia
        """
        if self.get_variable().is_in_categories(value) and self.get_term().get_category()==value:
            return 1.0
        return 0.0


    def __str__(self) -> str:
        """Devuelve los atributos de esta clase como str.

        Args:
            None.

        Returns:
            str.
        """
        return str(self.get_variable().get_name()) + ' es ' + str(self.get_term())


class FuzzyAP(AtomicProposition):
    """ Clase para modelar una proposición difusa

        Esta clase modela las proposiciones difusas atómicas, por ejemplo:
        T es Fría

        Attributes:
            antecedent(LinguisticV): Variable Lingüística de la PDA.
            CL (CL): término primario de la PDA.  
    """
    def __init__(self, variable:LinguisticV, term:MembershipFS):
        AtomicProposition.__init__(self, variable, term)


    # GETTERS


    # FUNCIONES PROPIAS
    def membership_grade(self, value:float) -> float:
        """Devuelve el valor de pertenencia de "value" al termino primario 
        (conjunto difuso).

        Args:
            value(float): valor a calcular su pertenencia.

        Returns:
            float: valor de pertenencia
        """
        return self.get_term().membership_grade(value)


    def __str__(self) -> str:
        """Devuelve los atributos de esta clase como str.

        Args:
            None.

        Returns:
            str.
        """
        return str(self.get_variable().get_name()) + ' es ' + str(self.get_term().get_name())

class PDC(CompoundProposition):
    """ Clase para modelar una proposición difusa compuesta. 

    Se modela con dos listas: (1) la primera tiene como componentes objetos PDA 
    y/o PDC, (2) La segunda tiene conectivas implementadas como funciones (Callable)

        Attributes:
            proposiciones (list): Lista con las proposiciones que componen la PDC.
            conectivas (list[Callable]): conectivas entre las PDAs que componen el 
            atributo "proposiciones". Tendrá una longitud un elemento inferior al
            atributo "proposiciones". 
    """
    def __init__(self, propos:list, functions:dict ):
        # Creación de las proposiciones a partir de "propos"
        self.proposiciones = []
        for fuz_propo in propos[0::2]:
            if len(fuz_propo)>2:
                self.proposiciones.append( PDC(fuz_propo, functions) )
            else:
                if fuz_propo[0]['type']=='real':
                    self.proposiciones.append( RealAP( # Crea PDA
                        RealV(fuz_propo[0]['name'],
                                  fuz_propo[0]['position'],
                                  fuz_propo[0]['support']), # Primer parámetro un VL
                        RealT( float(fuz_propo[1]) )  # Segundo parámetro un CD
                        )
                    )
                if fuz_propo[0]['type']=='categorical':
                    self.proposiciones.append( CategoricalAP( # Crea PDA
                        CategoryV(fuz_propo[0]['name'],
                                  fuz_propo[0]['position'],
                                  fuz_propo[0]['labels']), # Primer parámetro un VL
                        CategoryT( fuz_propo[1] )  # Segundo parámetro un CD
                        )
                    )
                if fuz_propo[0]['type']=='fuzzy':
                    self.proposiciones.append( FuzzyAP( # Crea PDA
                        LinguisticV(fuz_propo[0]['name'],
                                    fuz_propo[0]['position'],
                                    fuz_propo[0]['support'],
                                    fuz_propo[0]['sets']), # Primer parámetro un VL
                        fuz_propo[0]['sets'][fuz_propo[1]] )  # Segundo parámetro un CD
                    )
        # Creación de las conectivas a partir de "propos" y funciones
        self.conectivas_texto = [ conec for conec in propos[1::2] ]
        self.conectivas_funcion = [ functions[conec] for conec in propos[1::2] ]


    # GETTERS

    # SETTERS


    # FUNCIONES PROPIAS
    def membership_grade(self, valores:list[float]) -> float:
        """Devuelve el grado de pertenencia del PDC.

        Args:
            valores(list[float]): los valores que se pasarán a cada una de las proposiciones 

        Returns:
            float.
        """
        result = []
        for propo in self.proposiciones:
            if str(type(propo))[-5:-2]=='PDC':
                pos = propo.get_proposicion(0).get_variable().get_position()
                #val = [valores[pos]]*propo.get_num_proposiciones()
                result.append( propo.membership_grade(valores) )
                #print('ENTRO EN PDC:',pos,propo.get_variable().get_name())
            else:
                pos = propo.get_variable().get_position()
                result.append( propo.membership_grade(valores[pos]) )
        res = result[0]
        for r,conec in zip(result[1:],self.conectivas_funcion):
            res = conec(res,r)
        return res
        #self.get_proposicion(0).get_variable().get_name()
        #result = self.get_proposicion(0).membership_grade(valores[0])
        #for propo,conec,val in zip(self.proposiciones[1:],self.conectivas_funcion,valores[1:]):
        #    if str(type(propo))[-5:-2]=='PDC':
        #        val = [val]*propo.get_num_proposiciones()
        #    result = conec(result,propo.membership_grade(val))
        #return result

    def __str__(self) -> str:
        """Devuelve la clase como cadena.

        Args:
            None

        Returns:
            str.
        """
        cad = ''
        for pos,elem in enumerate(self.proposiciones):
            try:
                cad += ' (' + str(elem) + ') ' + self.conectivas_texto[pos]
            except IndexError:
                cad += ' (' + str(elem) + ') '
        return cad
