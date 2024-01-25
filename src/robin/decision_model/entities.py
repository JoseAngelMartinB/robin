"""Entities for the decision_model module."""

from typing import Callable, List, Mapping, Tuple, Union


class Variable:
    """
    Class to model a variable of any type.

    Attributes:
        name (str): str containing the name of the variable.
    """

    def __init__(self, name: str):
        self.name = name


class RealT:
    """
    Class to handle Real type terms.

    Attributes:
        real (float): float containing the value of the term.
    """

    def __init__(self, real: float) -> None:
        self.real = real


class CategoryT:
    """
    Class to handle Category type terms.
    """

    def __init__(self, category: str) -> None:
        self.category = category


class CategoryV(Variable):
    """
    Class to model a categorical variable.

    Attributes:
        name (str): str containing the name of the variable.
        categories (List[str]): list of str containing the categories of the variable.
    """

    def __init__(self, name: str, categories: List[str]):
        Variable.__init__(self, name)
        self.categories = categories

    def is_in_categories(self, category: str) -> bool:
        """
        Returns True if the category is in the categories of the variable.

        Args:
            category(str): category to check if it is in the variable.

        Returns:
            bool.
        """
        return category in self.categories

    def __str__(self) -> str:
        """
        Returns the attributes of this class as str.

        Returns:
            str.
        """
        return str(self.name) + ' ' + str(self.categories) + '\n'


class AtomicProposition:
    """
    Class to model an atomic proposition.

    Attributes:
        variable (Variable): Variable object.
        term (Term): Term object.
    """

    def __init__(self, variable: Variable, term: Union[RealT, CategoryT]):
        self.variable = variable
        self.term = term


class CompoundProposition:
    """
    Class to model a compound proposition.

    Attributes:
        antecedent (AtomicProposition): AtomicProposition object.
        CL (str): Primary term of the proposition.
    """

    def __init__(self, propos: List, functions: Mapping):
        self.propositions = []
        self.text_connectives = []
        self.function_connectives = []

    # GETTERS
    def get_proposition(self, pos: int) -> AtomicProposition:
        """
        Returns the atomic proposition in the position pos.

        Args:
            pos(int): position of the atomic proposition to return

        Returns:
            AtomicProposition.
        """
        return self.propositions[pos]

    def get_num_propositions(self) -> int:
        """
        Returns the number of propositions in the compound proposition.

        Returns:
            int.
        """
        return len(self.propositions)

    def get_connective_text(self, pos: int) -> str:
        """
        Returns the t_norm/t_conorm as text between the fuzzy propositions in the
        positions pos-1 and pos+1.

        Args:
            pos(int): position of the connective to return

        Returns:
            str.
        """
        return self.text_connectives[pos]

    def get_connective_function(self, pos: int) -> Callable:
        """
        Returns the t_norm/t_conorm to use between the fuzzy propositions in the
        positions pos-1 and pos+1.

        Args:
            pos(int): position of the function to return

        Returns:
            Callable.
        """
        return self.function_connectives[pos]


class FuzzyAP(AtomicProposition):
    """
    Class to model a fuzzy atomic proposition.

    Attributes:
        variable (Variable): Variable object.
        term (Term): Term object.
    """

    def __init__(self, variable: LinguisticV, term: MembershipFS):
        AtomicProposition.__init__(self, variable, term)

    def __str__(self) -> str:
        """
        Returns the attributes of this class as str.

        Returns:
            str.
        """
        return str(self.variable.name + ' is ' + str(self.term))


class PDC(CompoundProposition):
    """
    Class to model a fuzzy compound proposition.

    Attributes:
        propositions (List): List with the propositions that compose the PDC.
        connectives (List[Callable]): connectives between the PDAs that compose the
        attribute "propositions". It will have a length one element lower than the
        attribute "propositions".
    """

    def __init__(self, propos: List, functions: Mapping):
        super().__init__(propos, functions)
        self.propositions = []
        for fuz_propo in propos[0::2]:
            if len(fuz_propo) > 2:
                self.propositions.append(PDC(fuz_propo, functions))
            else:
                self.propositions.append(FuzzyAP(
                    LinguisticV(fuz_propo[0]['name'],
                                fuz_propo[0]['position'],
                                fuz_propo[0]['support'],
                                fuz_propo[0]['sets']),
                    fuz_propo[0]['sets'][fuz_propo[1]])
                )

        self.conectivas_texto = [conec for conec in propos[1::2]]
        self.conectivas_funcion = [functions[conec] for conec in propos[1::2]]

    def membership_grade(self, values: list[float]) -> float:
        """
        Returns the membership grade of the PDC.

        Args:
            values(list[float]): the values that will be passed to each of the propositions
        """
        result = []
        for propo in self.propositions:
            if str(type(propo))[-9:-2] == 'FuzzyAP':
                pos = propo.variable.name
                result.append(propo.membership_grade(values[pos]))
                # print('ENTRO EN FuzzyAP:',pos,propo.get_variable().get_name())
            if str(type(propo))[-5:-2] == 'PDC':
                pos = propo.get_proposition(0).variable.get_position()
                # val = [valores[pos]]*propo.get_num_proposiciones()
                result.append(propo.membership_grade(values))
                # print('ENTRO EN PDC:',pos,propo.get_variable().get_name())
        res = result[0]
        for r, conec in zip(result[1:], self.conectivas_funcion):
            res = conec(res, r)
        return res

    def __str__(self) -> str:
        """
        Returns the attributes of this class as str.

        Returns:
            str.
        """
        cad = ''
        for pos, elem in enumerate(self.propositions):
            try:
                cad += ' (' + str(elem) + ') ' + self.conectivas_texto[pos]
            except IndexError:
                cad += ' (' + str(elem) + ') '
        return cad
