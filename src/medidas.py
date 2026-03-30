import numpy as np
import pandas as pd



def media_evolve(lista_datos: list) -> float:
    return round(sum(lista_datos) / len(lista_datos), 2)

def mediana_evolve(lista_datos: list) -> float:
    sorted_lista = sorted(lista_datos)
    n = len(lista_datos)
    if n % 2 == 0:
        return (sorted_lista[n//2 - 1] + sorted_lista[n//2]) / 2
    else:
        return sorted_lista[n//2]

def percentil_evolve(lista_datos: list, percentil: int) -> float:
    sorted_lista = sorted(lista_datos)
    n = len(lista_datos)
    k = round((percentil / 100) * n, 0)
    return sorted_lista[int(k) - 1]

def varianza_evolve(lista_datos: list) -> float:
    media = media_evolve(lista_datos)
    return round(sum((x - media) ** 2 for x in lista_datos) / len(lista_datos), 2)

def desviacion_evolve(lista_datos: list) -> float:
    return round(varianza_evolve(lista_datos) ** 0.5, 2)

def IQR_evolve(lista_datos: list) -> float:
    q75 = percentil_evolve(lista_datos, 75)
    q25 = percentil_evolve(lista_datos, 25)
    return round(q75 - q25, 2)




if __name__ == "__main__":
    
    np.random.seed(42)
    edad = list(np.random.randint(20, 60, 100))
    salario =  list(np.random.normal(45000, 15000, 100))
    experiencia = list(np.random.randint(0, 30, 100))


    np.random.seed(42)
    df = pd.DataFrame({
        'edad': np.random.randint(20, 60, 100),
        'salario': np.random.normal(45000, 15000, 100),
        'experiencia': np.random.randint(0, 30, 100)
    })

    print("resultado pandas")
    print("--------------------------------")
    print(df.describe())

    print("resultado funciones")
    print("--------------------------------")

    print(media_evolve(edad))
    print(mediana_evolve(edad))
    print(percentil_evolve(edad, 50))
    print(varianza_evolve(edad))
    print(desviacion_evolve(edad))
    print(IQR_evolve(edad))

    print(media_evolve(salario))
    print(mediana_evolve(salario))
    print(percentil_evolve(salario, 50))
    print(varianza_evolve(salario))
    print(desviacion_evolve(salario))
    print(IQR_evolve(salario))

    print(media_evolve(experiencia))
    print(mediana_evolve(experiencia))
    print(percentil_evolve(experiencia, 50))
    print(varianza_evolve(experiencia))
    print(desviacion_evolve(experiencia))
    print(IQR_evolve(experiencia))