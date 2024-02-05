from json import load
from typing import List, Mapping

from src.robin.decision_model.terms import trapezoidal, MembershipFS
from src.robin.decision_model.propositions import PDC
from src.robin.decision_model.fuzzy_rule import TSKRule
from src.robin.decision_model.others import funciones


def read_json_variables(file_name, var_names):
    with open(file_name) as file:
        output = {}
        data = load(file)
        for i, var in enumerate(data['variables']):
            d = {}
            d['name'] = var['name']
            d['type'] = var['type']
            if d["type"] == "categorical":
                d['labels'] = var['labels'][0:]
            elif d["type"] == "fuzzy":
                d['position'] = var_names.index(d['name'])
                d['support'] = var['support']
                sets = {}
                for name_set in var['sets']:
                    sets[name_set] = MembershipFS(name_set, trapezoidal, var[name_set])
                d['sets'] = sets
            else:
                d['support'] = var['support']
            output[d['name']] = d
            # print( output[d['name']] )
        return output


def troceaRule(rule):
    # Localizar nombre de la regla y la propia regla
    r = rule.split('::: IF ')
    name = r[0]
    rule = r[1]
    # Cortar antecedente y consecuente
    r = rule.split(' THEN ')
    antecedent = r[0]
    consequent = r[1][0:-1]
    return name, antecedent, float(consequent)


def positions2Propositions(pos, first, last, seq, variables):
    '''
    Función que dado un antecedente devuelve la
    '''
    sol = []
    i = 0
    while i < len(pos):
        if pos[i][1] == first and pos[i + 1][1] == last:
            cad = seq[pos[i][0] + 1:pos[i + 1][0]]
            c = cad.split(' ')
            sol.append([variables[c[0]], c[-1]])
            i += 1
        else:
            sol.append(pos[i][1])
        i += 1
    return sol


def generaLista(antecedent, symbols, first, last, variables):
    # Calcula las posiciones de los simbolos ['(', ')','&','|']
    pos = [(i, s) for i, s in enumerate(antecedent) if s in symbols]
    # Genera la lista de entrada al programa
    lista = positions2Propositions(pos, '(', ')', antecedent, variables)
    # Convierte la lista al formato de salida
    sol = []
    i = 0
    while i < len(lista):
        if lista[i] != first:
            sol.append(lista[i])  # procesar hasta last
        else:
            s = []
            i += 1
            while lista[i] != last:
                s.append(lista[i])
                i += 1
            sol.append(s)
        i += 1
    return sol


def read_rules(file_name: str, variables):
    rules = []
    with open(file_name, 'r') as fichero:
        for rule in fichero.readlines():
            # print(rule, end='')
            name, antecedent, consequent = troceaRule(rule)
            lista = generaLista(antecedent, ['(', ')', '&', '|'], '(', ')', variables)
            # print('lista:',lista)
            # print('PROPOSITIONS', lista)
            # print('Name:', name)
            # print('Antecedent:', antecedent)
            # print('Consequent:', consequent, type(consequent), float(consequent))
            # print('Lista Entrada:', str(lista))
            # print('')

            r = TSKRule(
                name,
                PDC(lista, funciones),  # antedente
                lambda cons=consequent: cons
                # consecuente es una función en función de las entradas, por ejemplo: x*0.3+y*0.5
            )
            rules.append(r)
            # print('LA REGLA:', rules[-1], rules[-1].get_consequent()())
    return rules


def get_rules_from_dict(data: Mapping[str, str],
                        variables: Mapping[str, MembershipFS]
                        ) -> List[TSKRule]:
    """
    Generates a list of TSKRule objects from a dictionary of rules.

    Args:
        data: A dictionary of rules.
        variables: A dictionary of variables.

    Returns:
        A list of TSKRule objects.
    """
    rules = []
    for rule_id in data:
        name = rule_id
        rule = data[rule_id]
        antecedent, consequent = rule.split(' THEN ')
        consequent = float(consequent)
        lista = generaLista(antecedent, ['(', ')', '&', '|'], '(', ')', variables)

        r = TSKRule(name,
                    PDC(lista, funciones),  # antedente
                    lambda cons=consequent: cons)
        rules.append(r)
    return rules


def get_variables_from_dict(data: List[Mapping]) -> Mapping[str, MembershipFS]:
    """
    Generates a dictionary of variables from a list of variables.

    Args:
        data: A list of variables.

    Returns:
        A dictionary of variables.
    """
    output = {}
    var_names = [var['name'] for var in data]
    for i, var in enumerate(data):
        d = {}
        d['name'] = var['name']
        d['type'] = var['type']
        if d["type"] == "categorical":
            d['labels'] = var['labels'][0:]
        elif d["type"] == "fuzzy":
            d['position'] = var_names.index(d['name'])
            d['support'] = var['support']
            sets = {}
            for name_set in var['sets']:
                sets[name_set] = MembershipFS(name_set, trapezoidal, var[name_set])
            d['sets'] = sets
        else:
            d['support'] = var['support']
        output[d['name']] = d
    return output
