import random
import csv


def genera_armas(weapons_names, n_armas_tipo, domi_tipo):
    sal = {} # recoge el resultado
    # devuelve un diccionario con el nombre del arma como clave y una lista con los valores de cada una
    # de las variables que definen el arma
    for weapon, n_armas, dominio in zip(weapons_names, n_armas_tipo, domi_tipo):
        for n in range(n_armas):  # Crea los valores de n armas
            sal[weapon+'_N'+str(n)] = [round(random.uniform(*lista), 2) for lista in dominio]
    #del sal['NAME']
    return sal


def show_services(variables_names,weapons,file_name=None, output_file=False):
    s = 'service_id'

    for e in variables_names:
        s += ',' + e 
    s += '\n'
    for e in weapons:
        s += e
        for e1 in weapons[e]:
            s += ',' + str(e1)
        s += '\n'
    if (output_file):
        file = open(file_name, "w")
        file.write(s)
        file.close()


def read_supply(file_name):
    weapons = {}
    with open(file_name, newline='') as File:  
        reader = csv.reader(File)
        for row in reader:
            if row[0] != 'service_id':
                weapons[row[0]] = [round(float(e), 2) for e in row[1:]]
            else:
                var_names = row[1:]
    # Devuelve los nombres de las variables y el diccionario de los servicios
    return var_names, weapons
