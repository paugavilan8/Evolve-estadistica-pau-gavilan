import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE_DIR   = 'practica_final_[gavilan_muñoz_pau]'
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def regresion_lineal_multiple(X_train, y_train, X_test):
    """
    Ajusta un modelo de Regresión Lineal Múltiple usando la solución
    analítica OLS: β = (XᵀX)⁻¹ Xᵀy

    Parámetros
    ----------
    X_train : np.ndarray (n_train, p) — Features de entrenamiento
    y_train : np.ndarray (n_train,)   — Target de entrenamiento
    X_test  : np.ndarray (n_test, p)  — Features de test

    Retorna
    -------
    coefs  : np.ndarray (p+1,) — Coeficientes [β₀, β₁, ..., βₚ]
    y_pred : np.ndarray (n_test,) — Predicciones sobre X_test
    """
    X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    coefs = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
    X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    y_pred = X_test_b @ coefs
    return coefs, y_pred


def calcular_mae(y_real, y_pred):
    """
    Calcula el Mean Absolute Error: MAE = (1/n) * Σ |y_real - y_pred|

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del MAE
    """
    return np.mean(np.abs(y_real - y_pred))


def calcular_rmse(y_real, y_pred):
    """
    Calcula el Root Mean Squared Error: RMSE = sqrt((1/n) * Σ (y_real - y_pred)²)

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del RMSE
    """
    return np.sqrt(np.mean((y_real - y_pred) ** 2))


def calcular_r2(y_real, y_pred):
    """
    Calcula el coeficiente de determinación R² = 1 - SS_res / SS_tot

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del R²
    """
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - (ss_res / ss_tot)


def graficar_real_vs_predicho(y_real, y_pred,
                               ruta_salida=None):
    """
    Scatter plot de Valores Reales vs. Valores Predichos con línea y=x.

    Parámetros
    ----------
    y_real      : np.ndarray — Valores reales del test set
    y_pred      : np.ndarray — Predicciones del modelo
    ruta_salida : str        — Ruta donde guardar la imagen
    """
    if ruta_salida is None:
        ruta_salida = os.path.join(OUTPUT_DIR, 'ej3_predicciones.png')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_real, y_pred, alpha=0.5, color='steelblue',
               edgecolors='black', linewidths=0.3, s=30, label='Predicciones')
    lim_min = min(y_real.min(), y_pred.min())
    lim_max = max(y_real.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color='red', linestyle='--', linewidth=1.5, label='Predicción perfecta')
    ax.set_title('Valores Reales vs. Valores Predichos', fontsize=14)
    ax.set_xlabel('Valores Reales')
    ax.set_ylabel('Valores Predichos')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    SEMILLA = 42
    rng = np.random.default_rng(SEMILLA)

    n_muestras = 200
    n_features = 3

    X = rng.standard_normal((n_muestras, n_features))
    coefs_reales = np.array([5.0, 2.0, -1.0, 0.5])
    ruido = rng.normal(0, 1.5, n_muestras)
    y = coefs_reales[0] + X @ coefs_reales[1:] + ruido

    corte = int(0.8 * n_muestras)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    coefs, y_pred = regresion_lineal_multiple(X_train, y_train, X_test)

    mae  = calcular_mae(y_test, y_pred)
    rmse = calcular_rmse(y_test, y_pred)
    r2   = calcular_r2(y_test, y_pred)

    print("=" * 50)
    print("RESULTADOS — Regresión Lineal Múltiple (NumPy)")
    print("=" * 50)
    print(f"\nCoeficientes reales:    {coefs_reales}")
    print(f"Coeficientes ajustados: {np.round(coefs, 4)}")
    print(f"\nMétricas sobre test set:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    ruta_coef = os.path.join(OUTPUT_DIR, 'ej3_coeficientes.txt')
    with open(ruta_coef, "w", encoding='utf-8') as f:
        f.write("Regresión Lineal Múltiple — Coeficientes ajustados\n")
        f.write("=" * 50 + "\n")
        nombres = ["Intercepto (β₀)"] + [f"β{i+1} (feature {i+1})" for i in range(n_features)]
        for nombre, valor in zip(nombres, coefs):
            f.write(f"  {nombre}: {valor:.6f}\n")
        f.write("\nCoeficientes reales de referencia:\n")
        for nombre, valor in zip(nombres, coefs_reales):
            f.write(f"  {nombre}: {valor:.6f}\n")
    print(f"[Guardado] {ruta_coef}")

    ruta_met = os.path.join(OUTPUT_DIR, 'ej3_metricas.txt')
    with open(ruta_met, "w", encoding='utf-8') as f:
        f.write("Regresión Lineal Múltiple — Métricas de evaluación\n")
        f.write("=" * 50 + "\n")
        f.write(f"  MAE  : {mae:.6f}\n")
        f.write(f"  RMSE : {rmse:.6f}\n")
        f.write(f"  R²   : {r2:.6f}\n")

    graficar_real_vs_predicho(y_test, y_pred)