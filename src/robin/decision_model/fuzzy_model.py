""" Importación de módulos necesarios"""

from src.robin.decision_model.fuzzy_rule import RD, MandaniRule, TSKRule
from src.robin.decision_model.terms import EnumeratedFS, enumerated_fs_union


class FuzzyModel():
    """ Clase para modelar un modelo difuso (sistema de control). 

    Se modela con una lista que contiene objetos de tipo RD

        Attributes:
            RDs (list[RDs]): lista con las reglas modeladas con objetos de tipo RD. 
    """
    def __init__(self, rules:list[RD] ):
        self.rules = rules

    # GETTERS
    def get_rule(self, pos:int) -> RD:
        """Devuelve la regla en la posición pos.

        Args:
            pos(int): posición de la regla a recuperar.

        Returns:
            RD.
        """
        return self.rules[pos]

    def get_rules(self) -> list[RD]:
        """Devuelve la lista que contiene las reglas de todo el modelo.

        Returns:
            list[RD]
        """
        return self.rules

    # SETTERS
    def set_rules(self, regla:RD, pos:int) -> None:
        """Cambia la regla en la posición "pos".

        Args:
            regla (RD): nueva regla a colocar en "pos"
            pos (int): posición que se cambia la regla.

        Returns:
            None
        """
        self.rules[pos] = regla

    # FUNCIONES PROPIAS
    def add_rule(self, rule:RD) -> None:
        """Añade la regla pasada como parámetro al modelo.

        Args:
            regla (RD): regla a añadir

        Returns:
            None
        """
        self.rules.append(rule)

    def __str__(self) -> str:
        """Devuelve los atributos de esta clase como str.

        Returns:
            str.
        """
        cad = ''
        for regla in self.rules:
            cad += str(regla) + '\n'
        return cad


class MandaniFuzzyModel(FuzzyModel):
    """ Clase para modelar un modelo difuso (sistema de control). 

    Se modela con una lista que contiene objetos de tipo RD

        Attributes:
            RDs (list[RDs]): lista con las reglas modeladas con objetos de tipo RD. 
    """
    def __init__(self, rules:list[MandaniRule] ):
        FuzzyModel.__init__(self, rules)

    def rule_mandani_inference(self, inputs:list[list[float]],
                                 set_name:str='Union') -> tuple[EnumeratedFS,list[float]]:
        """Realiza la inferencia tipo Mandani para una regla.

        Args:
            input (list[list[float]]): estructura de lista que contienen los valores de entrada 
                para realizar los cálculos
            set_name (str): nombre del conjunto difuso obtenido

        Returns:
            tuple[EnumeratedFS,list[float]]
        """
        enumerated_fs_list = [] # contendrá los CDEnumerados resultados de las reglas
        for rule in self.get_rules():
            # Calcular la pertenencia a la regla actual
            enumerated_output_set = rule.membership_grade(inputs)
            if enumerated_output_set.get_x_values()!=[]: # la añade al modelo si no es vacía
                enumerated_fs_list.append( enumerated_output_set )
        # Se realiza la unión de las salidas de las reglas en el CDEnumerado union
        # obteniendo un conjunto enumerado (primer valor devuelto) y los conjuntos
        # enumerados obtenidos como salida de cada regla
        return enumerated_fs_union(enumerated_fs_list, set_name), enumerated_fs_list

    def model_mandani_inference(self, in_values:list[list]) -> list[dict]:
        """Realiza la inferencia tipo Mandani.

        Args:
            in_values (list[list]): lista que contiene una lista para cada 
                entrada a testear

        Returns:
            tuple(EnumeratedFS,list[float]
        """
        to_return = []
        for value in in_values:
            enumerated_fuzzy_set, enumerated_fs_list = self.rule_mandani_inference(value)
            to_return.append( {'Input': value,
                    'FSEnumeratedUnion': enumerated_fuzzy_set,
                    'actives_rules': [elem.get_name()[-2:] for elem in enumerated_fs_list],
                    'EnumeratedFS': enumerated_fs_list})
        return to_return


class TSKFuzzyModel(FuzzyModel):
    """ Clase para modelar un modelo difuso de tipo TSK. 

    Se modela con una lista que contiene objetos de tipo RD

        Attributes:
            RDs (list[RDs]): lista con las reglas modeladas con objetos de tipo RD. 
    """
    def __init__(self, rules:list[TSKRule] ):
        FuzzyModel.__init__(self, rules)


    def model_tsk_inference(self, inputs:list[list[float]]) -> list[float]:
        """Realiza la inferencia tipo TSK para el modelo ante una entrada.

        Args:
            input (list[list[float]]): estructura de lista que contienen los valores de entrada 
                para realizar los cálculos sobre cada una de las variables
            set_name (str): nombre del conjunto difuso obtenido

        Returns:
            tuple[EnumeratedFS,list[float]]
        """
        rules_membership_value_list = [] # contendrá los CDEnumerados resultados de las reglas
        consequents_values_list = []
        names_list = []
        for rule in self.get_rules():
            # Calcular la pertenencia a la regla actual
            membership_value = rule.membership_grade(inputs)
            if membership_value>0.0: # la añade al modelo si no es vacía
                rules_membership_value_list.append( membership_value )
                consequents_values_list.append( rule.get_consequent()(*inputs) )
                names_list.append( rule.get_name() )
        # Se realiza la unión de las salidas de las reglas en el CDEnumerado union
        # obteniendo un conjunto enumerado (primer valor devuelto) y los conjuntos
        # enumerados obtenidos como salida de cada regla
        returns_values = [ membership*consequent
                        for membership,consequent in zip(rules_membership_value_list,consequents_values_list)]
        return sum(returns_values)/sum(rules_membership_value_list), rules_membership_value_list, consequents_values_list, names_list

    def loop_tsk_inference(self, in_values:list[list]) -> list[dict]:
        """Realiza la inferencia tipo Mandani sobre esta clase modelo utilizando una 
        secuencia de entrada como una lista.

        Args:
            in_values (list[list]): lista que contiene una lista para cada 
                entrada a testear, es la secuencia de entrada

        Returns:
            list[dict]
        """
        to_return = []
        for value in in_values:
            out_value, rules_membership_value_list, consequents_values_list, names_list = self.model_tsk_inference(value)
            to_return.append( {'Input': value,
                    'actives_rules': names_list,
                    'consequents_values_list': consequents_values_list,
                    'rule_membership_value_list': rules_membership_value_list,
                    'Output': out_value} )
        return to_return


class AcumulativeTSKFuzzyModel(FuzzyModel):
    """ Clase para modelar un modelo difuso de tipo TSK ACUMULATIVO. 

    Se modela con una lista que contiene objetos de tipo RD

        Attributes:
            RDs (list[RDs]): lista con las reglas modeladas con objetos de tipo RD. 
    """
    def __init__(self, rules:list[TSKRule] ):
        FuzzyModel.__init__(self, rules)


    def model_tsk_inference(self, inputs:list[list[float]]) -> list[float]:
        """Realiza la inferencia tipo TSK para el modelo ante una entrada UTILIZANDO EL MODELO
        ACUMULATIVO DEL PAPER.

        Args:
            input (list[list[float]]): estructura de lista que contienen los valores de entrada 
                para realizar los cálculos sobre cada una de las variables
            set_name (str): nombre del conjunto difuso obtenido

        Returns:
            tuple[EnumeratedFS,list[float]]
        """
        rules_membership_value_list = [] # contendrá los CDEnumerados resultados de las reglas
        consequents_values_list = []
        names_list = []
        for rule in self.get_rules():
            # Calcular la pertenencia a la regla actual
            membership_value = rule.membership_grade(inputs)
            #print('MSRULE:', rule.get_name(), membership_value)
            if membership_value>0.0: # la añade al modelo si no es vacía
                rules_membership_value_list.append( membership_value ) 
                consequents_values_list.append( rule.get_consequent()() )
                names_list.append( rule.get_name() )
        # Se realiza la unión de las salidas de las reglas en el CDEnumerado union
        # obteniendo un conjunto enumerado (primer valor devuelto) y los conjuntos
        # enumerados obtenidos como salida de cada regla
        returns_values = [ membership*consequent
                        for membership,consequent in zip(rules_membership_value_list,consequents_values_list)]
        return sum(returns_values), rules_membership_value_list, consequents_values_list, names_list


    def loop_tsk_inference(self, in_values:list[list]) -> list[dict]:
        """Realiza la inferencia tipo Mandani sobre esta clase modelo utilizando una 
        secuencia de entrada como una lista.

        Args:
            in_values (list[list]): lista que contiene una lista para cada 
                entrada a testear, es la secuencia de entrada

        Returns:
            list[dict]
        """
        to_return = []
        for value in in_values:
            out_value, rules_membership_value_list, consequents_values_list, names_list = self.model_tsk_inference(value)
            to_return.append( {'Input': value,
                    'actives_rules': names_list,
                    'consequents_values_list': consequents_values_list,
                    'rule_membership_value_list': rules_membership_value_list,
                    'Output': out_value} )
        return to_return
