import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

BASE_DIR   = 'practica_final_[gavilan_muñoz_pau]'
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generar_serie_temporal(semilla=42):
    """
    Genera una serie temporal sintética con componentes conocidos.
    NO MODIFICAR ESTE BLOQUE.

    Parámetros
    ----------
    semilla : int — Semilla aleatoria (NO modificar)

    Retorna
    -------
    serie : pd.Series con índice DatetimeIndex diario (2018-01-01 → 2023-12-31)
    """
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(fechas)
    t = np.arange(n)

    tendencia      = 0.05 * t + 50
    estacionalidad = (15 * np.sin(2 * np.pi * t / 365.25)
                      + 6 * np.cos(4 * np.pi * t / 365.25))
    ciclo          = 8 * np.sin(2 * np.pi * t / 1461)
    ruido          = rng.normal(loc=0, scale=3.5, size=n)

    valores = tendencia + estacionalidad + ciclo + ruido
    return pd.Series(valores, index=fechas, name="valor")


def visualizar_serie(serie):
    """
    Genera y guarda el gráfico de la serie temporal completa.
    Salida: output/ej4_serie_original.png

    Parámetros
    ----------
    serie : pd.Series — La serie temporal a visualizar
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(serie.index, serie.values, color='steelblue', linewidth=0.8)
    ax.set_title('Serie Temporal Sintética — 6 años de datos diarios', fontsize=13)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Valor')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, 'ej4_serie_original.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()


def descomponer_serie(serie):
    """
    Descompone la serie con seasonal_decompose (model='additive', period=365)
    y guarda los 4 subgráficos.
    Salida: output/ej4_descomposicion.png

    Parámetros
    ----------
    serie : pd.Series — La serie temporal

    Retorna
    -------
    resultado : DecomposeResult
    """
    resultado = seasonal_decompose(serie, model='additive', period=365)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(4, 1, hspace=0.5)
    componentes = [
        (resultado.observed,  'Observada',      'steelblue'),
        (resultado.trend,     'Tendencia',      'darkorange'),
        (resultado.seasonal,  'Estacionalidad', 'mediumseagreen'),
        (resultado.resid,     'Residuo',        'tomato'),
    ]
    for i, (comp, titulo, color) in enumerate(componentes):
        ax = fig.add_subplot(gs[i])
        ax.plot(comp.index, comp.values, color=color, linewidth=0.7)
        ax.set_title(titulo, fontsize=11)
        ax.set_ylabel('Valor')
        ax.grid(True, linestyle='--', alpha=0.4)
    fig.suptitle('Descomposición Aditiva de la Serie Temporal (period=365)',
                 fontsize=13, y=1.01)
    ruta = os.path.join(OUTPUT_DIR, 'ej4_descomposicion.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    return resultado


def analizar_residuo(residuo):
    """
    Analiza el residuo: estadísticos, test ADF, test Jarque-Bera,
    gráficos ACF/PACF e histograma con curva normal.

    Parámetros
    ----------
    residuo : pd.Series — Componente residuo de la descomposición
    """
    residuo_limpio = residuo.dropna()

    media     = residuo_limpio.mean()
    std       = residuo_limpio.std()
    asimetria = residuo_limpio.skew()
    curtosis  = residuo_limpio.kurt()

    print(f"\n  Media:     {media:.4f}")
    print(f"  Std:       {std:.4f}")
    print(f"  Asimetría: {asimetria:.4f}")
    print(f"  Curtosis:  {curtosis:.4f}")

    resultado_adf = adfuller(residuo_limpio)
    p_adf    = resultado_adf[1]
    adf_stat = resultado_adf[0]
    print(f"\n  Test ADF: estadístico={adf_stat:.4f}, p={p_adf:.6f}")
    print("  → Residuo ESTACIONARIO " if p_adf < 0.05 else "  → NO estacionario ")

    jb_stat, jb_p = jarque_bera(residuo_limpio)
    print(f"\n  Test Jarque-Bera: estadístico={jb_stat:.4f}, p={jb_p:.6f}")
    print("  → No se rechaza normalidad " if jb_p > 0.05 else "  → Se rechaza normalidad ")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(residuo_limpio, lags=60, ax=axes[0], color='steelblue')
    axes[0].set_title('ACF del Residuo', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.4)
    plot_pacf(residuo_limpio, lags=60, ax=axes[1], color='steelblue', method='ywm')
    axes[1].set_title('PACF del Residuo', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    ruta_acf = os.path.join(OUTPUT_DIR, 'ej4_acf_pacf.png')
    plt.savefig(ruta_acf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Guardado] {ruta_acf}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuo_limpio, bins=60, density=True, color='steelblue',
            edgecolor='black', alpha=0.7, label='Residuo')
    x_range = np.linspace(residuo_limpio.min(), residuo_limpio.max(), 300)
    ax.plot(x_range, stats.norm.pdf(x_range, loc=media, scale=std),
            color='red', linewidth=2,
            label=f'Normal teórica (μ={media:.2f}, σ={std:.2f})')
    ax.set_title('Histograma del Residuo con Curva Normal Teórica', fontsize=13)
    ax.set_xlabel('Residuo')
    ax.set_ylabel('Densidad')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    ruta_hist = os.path.join(OUTPUT_DIR, 'ej4_histograma_ruido.png')
    plt.savefig(ruta_hist, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Guardado] {ruta_hist}")

    ruta_txt = os.path.join(OUTPUT_DIR, 'ej4_analisis.txt')
    with open(ruta_txt, 'w', encoding='utf-8') as f:
        f.write("EJERCICIO 4 — ANÁLISIS DEL RESIDUO\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Media:     {media:.4f}\n")
        f.write(f"Std:       {std:.4f}\n")
        f.write(f"Asimetría: {asimetria:.4f}\n")
        f.write(f"Curtosis:  {curtosis:.4f}\n\n")
        f.write(f"Test ADF:\n  Estadístico: {adf_stat:.4f}\n  p-valor: {p_adf:.6f}\n")
        f.write("  → Residuo ESTACIONARIO\n" if p_adf < 0.05 else "  → NO estacionario\n")
        f.write(f"\nTest Jarque-Bera (normalidad):\n  Estadístico: {jb_stat:.4f}\n  p-valor: {jb_p:.6f}\n")
        f.write("  → No se rechaza normalidad\n" if jb_p > 0.05 else "  → Se rechaza normalidad\n")
    print(f"[Guardado] {ruta_txt}")


if __name__ == "__main__":

    print("=" * 55)
    print("EJERCICIO 4 — Análisis de Series Temporales")
    print("=" * 55)

    SEMILLA = 42
    serie = generar_serie_temporal(semilla=SEMILLA)

    print(f"\nSerie generada:")
    print(f"  Periodo: {serie.index[0].date()} → {serie.index[-1].date()}")
    print(f"  Observaciones: {len(serie)}")
    print(f"  Media: {serie.mean():.2f} | Std: {serie.std():.2f}")
    print(f"  Min / Max: {serie.min():.2f} / {serie.max():.2f}")

    print("\n[1/3] Visualizando la serie original...")
    visualizar_serie(serie)

    print("[2/3] Descomponiendo la serie...")
    resultado = descomponer_serie(serie)

    print("[3/3] Analizando el residuo...")
    if resultado is not None:
        analizar_residuo(resultado.resid)

    print("\nSalidas esperadas en output/:")
    salidas = ['ej4_serie_original.png', 'ej4_descomposicion.png',
               'ej4_acf_pacf.png', 'ej4_histograma_ruido.png', 'ej4_analisis.txt']
    for s in salidas:
        existe = os.path.exists(os.path.join(OUTPUT_DIR, s))
        print(f"  [{'✓' if existe else '✗'}] output/{s}")