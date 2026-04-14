import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
BASE_DIR   = 'practica_final_[gavilan_muñoz_pau]'
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'Video_Games.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = 'Global_Sales'


# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────

def cargar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset desde un fichero CSV.

    Parámetros:
        ruta (str): Ruta al fichero CSV.

    Retorna:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    return pd.read_csv(ruta)


# ─────────────────────────────────────────────
# 2.1 PREPROCESAMIENTO
# ─────────────────────────────────────────────

def preprocesar(df: pd.DataFrame):
    """
    Aplica el preprocesamiento completo del dataset:
    - Elimina columnas que no aportan información al modelo.
    - Elimina filas con nulos en columnas relevantes.
    - Codifica variables categóricas con get_dummies (OneHot).
    - Escala variables numéricas con StandardScaler.
    - Divide en conjuntos Train (80%) y Test (20%).

    Justificación:
    - 'Name' y 'Developer' tienen demasiadas categorías únicas → no aportan
      capacidad predictiva generalizableble y generan ruido.
    - 'User_Score' viene como string con valores 'tbd' → se descarta.
    - 'Critic_Score', 'Critic_Count', 'User_Count' tienen >50% nulos → se
      descartan para no perder demasiadas filas.
    - Se usa get_dummies (OneHotEncoding) para variables categóricas de baja
      cardinalidad (Platform, Genre, Rating) porque la regresión lineal no
      admite relaciones ordinales entre categorías.
    - Se aplica StandardScaler solo a las numéricas para que los coeficientes
      sean comparables entre sí.

    Parámetros:
        df (pd.DataFrame): Dataset original.

    Retorna:
        tuple: X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("=" * 60)
    print("2.1 PREPROCESAMIENTO")
    print("=" * 60)

    # Columnas a eliminar: no aportan info útil al modelo
    cols_eliminar = ['Name', 'Developer', 'User_Score',
                     'Critic_Score', 'Critic_Count', 'User_Count']
    df_clean = df.drop(columns=cols_eliminar, errors='ignore')

    # Eliminar filas con nulos en columnas relevantes
    df_clean = df_clean.dropna(subset=['Year_of_Release', 'Genre',
                                       'Publisher', 'Rating', TARGET])
    print(f"\nFilas tras limpiar nulos: {df_clean.shape[0]}")

    # Eliminar Publisher (demasiadas categorías únicas → ruido)
    df_clean = df_clean.drop(columns=['Publisher'], errors='ignore')

    # Separar features y target
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]

    # Codificar variables categóricas con OneHot (get_dummies)
    cols_cat = X.select_dtypes(include=['object', 'str']).columns.tolist()
    print(f"Variables categóricas codificadas (OneHot): {cols_cat}")
    X = pd.get_dummies(X, columns=cols_cat, drop_first=True)

    # Escalar variables numéricas
    cols_num = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X[cols_num] = scaler.fit_transform(X[cols_num])

    feature_names = X.columns.tolist()
    print(f"Total de features tras preprocesamiento: {len(feature_names)}")

    # Split Train 80% / Test 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

    return X_train, X_test, y_train, y_test, feature_names, scaler


# ─────────────────────────────────────────────
# 2.2 MODELO A — REGRESIÓN LINEAL
# ─────────────────────────────────────────────

def entrenar_regresion_lineal(X_train, y_train) -> LinearRegression:

    print("\n" + "=" * 60)
    print("2.2 MODELO A — REGRESIÓN LINEAL")
    print("=" * 60)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    print("Modelo entrenado correctamente.")
    return modelo


def evaluar_modelo(modelo: LinearRegression, X_test, y_test,
                   feature_names: list) -> dict:

    y_pred = modelo.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n  MAE  (Error Absoluto Medio):      {mae:.4f}")
    print(f"  RMSE (Raíz Error Cuadrático Med): {rmse:.4f}")
    print(f"  R²   (Coef. Determinación):       {r2:.4f}")

    # Top 10 variables más influyentes por valor absoluto del coeficiente
    coefs = pd.Series(np.abs(modelo.coef_), index=feature_names)
    top10 = coefs.sort_values(ascending=False).head(10)
    print("\n  Top 10 variables más influyentes (|coeficiente|):")
    print(top10.to_string())

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2,
            'y_pred': y_pred, 'y_test': y_test}


def guardar_metricas(metricas: dict) -> None:

    out_path = os.path.join(OUTPUT_DIR, 'ej2_metricas_regresion.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("EJERCICIO 2 — MÉTRICAS REGRESIÓN LINEAL\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"MAE  (Error Absoluto Medio):       {metricas['MAE']:.4f}\n")
        f.write(f"RMSE (Raíz Error Cuadrático Medio):{metricas['RMSE']:.4f}\n")
        f.write(f"R²   (Coef. de Determinación):     {metricas['R2']:.4f}\n\n")
        f.write("Interpretación:\n")
        f.write(f"  - El modelo explica el {metricas['R2']*100:.2f}% de la varianza de Global_Sales.\n")
        f.write(f"  - En promedio, el error de predicción es de {metricas['MAE']:.4f} millones de unidades.\n")


# ─────────────────────────────────────────────
# GRÁFICO DE RESIDUOS
# ─────────────────────────────────────────────

def plot_residuos(metricas: dict) -> None:

    y_pred = metricas['y_pred']
    y_test = metricas['y_test']
    residuos = np.array(y_test) - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuos, alpha=0.4, color='steelblue',
               edgecolors='black', linewidths=0.3, s=20)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5,
               label='Residuo = 0')
    ax.set_title('Gráfico de Residuos — Regresión Lineal', fontsize=14)
    ax.set_xlabel('Valores Predichos (Global_Sales)')
    ax.set_ylabel('Residuos (Real - Predicho)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'ej2_residuos.png')
    plt.savefig(out_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    df = cargar_datos(DATA_PATH)

    X_train, X_test, y_train, y_test, feature_names, scaler = preprocesar(df)

    modelo = entrenar_regresion_lineal(X_train, y_train)

    metricas = evaluar_modelo(modelo, X_test, y_test, feature_names)

    guardar_metricas(metricas)

    plot_residuos(metricas)