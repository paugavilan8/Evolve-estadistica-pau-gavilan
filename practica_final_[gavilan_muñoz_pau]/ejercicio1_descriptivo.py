import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

np.random.seed(42)

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────
BASE_DIR = 'practica_final_[gavilan_muñoz_pau]'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Video_Games.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variable objetivo
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
    df = pd.read_csv(ruta)
    return df


# ─────────────────────────────────────────────
# A) RESUMEN ESTRUCTURAL
# ─────────────────────────────────────────────

def resumen_estructural(df: pd.DataFrame) -> None:
    """
    Imprime el resumen estructural del dataset: dimensiones, tamaño en
    memoria, tipos de dato y porcentaje de valores nulos por columna.

    Parámetros:
        df (pd.DataFrame): Dataset cargado.

    Retorna:
        None
    """
    print("=" * 60)
    print("A) RESUMEN ESTRUCTURAL")
    print("=" * 60)
    print(f"\nDimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print("\nTipos de dato por columna:")
    print(df.dtypes)

    nulos = df.isnull().sum()
    pct_nulos = (nulos / len(df) * 100).round(2)
    resumen_nulos = pd.DataFrame({'Nulos': nulos, 'Porcentaje (%)': pct_nulos})
    resumen_nulos = resumen_nulos[resumen_nulos['Nulos'] > 0]
    print("\nValores nulos por columna:")
    print(resumen_nulos)
    print("""
Decisión de tratamiento de nulos:
  - Year_of_Release: se eliminarán filas sin año para análisis temporal.
  - Critic_Score, User_Score, Critic_Count, User_Count: columnas con muchos
    nulos; se usarán solo donde se requieran (dropna local), sin imputar.
  - Developer, Rating: se mantienen como NaN para análisis categórico.
    """)


# ─────────────────────────────────────────────
# B) ESTADÍSTICOS DESCRIPTIVOS NUMÉRICOS
# ─────────────────────────────────────────────

def estadisticos_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula y muestra estadísticos descriptivos de todas las variables
    numéricas: media, mediana, moda, desviación típica, varianza, mínimo,
    máximo, cuartiles, IQR, skewness y curtosis.
    Guarda el resumen en output/ej1_descriptivo.csv.

    Parámetros:
        df (pd.DataFrame): Dataset cargado.

    Retorna:
        pd.DataFrame: Tabla de estadísticos descriptivos.
    """
    print("\n" + "=" * 60)
    print("B) ESTADÍSTICOS DESCRIPTIVOS NUMÉRICOS")
    print("=" * 60)

    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    resumen = df[cols_num].describe().T
    resumen['variance'] = df[cols_num].var()
    resumen['mode'] = df[cols_num].mode().iloc[0]
    resumen['skewness'] = df[cols_num].skew()
    resumen['kurtosis'] = df[cols_num].kurt()
    resumen['IQR'] = resumen['75%'] - resumen['25%']
    print(resumen.to_string())

    iqr_target = df[TARGET].quantile(0.75) - df[TARGET].quantile(0.25)
    print(f"\n  → IQR de {TARGET}: {iqr_target:.4f}")
    print(f"  → Skewness de {TARGET}: {df[TARGET].skew():.4f}")
    print(f"  → Curtosis de {TARGET}: {df[TARGET].kurt():.4f}")

    out_path = os.path.join(OUTPUT_DIR, 'ej1_descriptivo.csv')
    resumen.to_csv(out_path)
    return resumen


# ─────────────────────────────────────────────
# C) DISTRIBUCIONES — Histogramas, Boxplots y Outliers
# ─────────────────────────────────────────────

def plot_histogramas(df: pd.DataFrame) -> None:
    """
    Genera y guarda histogramas de todas las variables numéricas.
    Salida: output/ej1_histogramas.png

    Parámetros:
        df (pd.DataFrame): Dataset cargado.

    Retorna:
        None
    """
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(cols_num)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols_num):
        axes[i].hist(df[col].dropna(), bins=40, color='steelblue',
                     edgecolor='black', alpha=0.8)
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel('Valor')
        axes[i].set_ylabel('Frecuencia')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Histogramas de Variables Numéricas', fontsize=15, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'ej1_histogramas.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_boxplots(df: pd.DataFrame) -> None:
    """
    Genera y guarda boxplots de la variable objetivo segmentados por cada
    variable categórica con <= 30 categorías únicas.
    Salida: output/ej1_boxplots.png

    Parámetros:
        df (pd.DataFrame): Dataset cargado.

    Retorna:
        None
    """
    cols_cat = df.select_dtypes(include=['object', 'str']).columns.tolist()
    cols_cat_validas = [c for c in cols_cat if df[c].nunique() <= 30]

    n = len(cols_cat_validas)
    fig, axes = plt.subplots(n, 1, figsize=(14, n * 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_cat_validas):
        order = (
            df.groupby(col)[TARGET]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        sns.boxplot(data=df, x=col, y=TARGET, order=order, ax=ax,
            color='steelblue')
        ax.set_title(f'{TARGET} por {col}', fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel(f'{TARGET} (millones)')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Boxplots de Global_Sales por Variable Categórica', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'ej1_boxplots.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def detectar_outliers_iqr(df: pd.DataFrame) -> None:
    """
    Detecta outliers en la variable objetivo usando el método IQR
    (límites: Q1 - 1.5·IQR y Q3 + 1.5·IQR) y guarda el informe en
    output/ej1_outliers.txt.

    Justificación del método: la distribución de Global_Sales es muy
    asimétrica (skewness ≈ 17), por lo que IQR es más robusto que Z-score.

    Parámetros:
        df (pd.DataFrame): Dataset cargado.

    Retorna:
        None
    """
    print("\n" + "=" * 60)
    print("C) DETECCIÓN DE OUTLIERS — Método IQR")
    print("   (Justificación: distribución muy asimétrica → IQR más robusto)")
    print("=" * 60)

    Q1 = df[TARGET].quantile(0.25)
    Q3 = df[TARGET].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR

    outliers = df[(df[TARGET] < limite_inf) | (df[TARGET] > limite_sup)]
    print(f"\n  Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
    print(f"  Límite inferior: {limite_inf:.3f}")
    print(f"  Límite superior: {limite_sup:.3f}")
    print(f"  Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"\n  Top 10 outliers por {TARGET}:")
    print(outliers.nlargest(10, TARGET)[['Name', 'Platform', TARGET]].to_string(index=False))

    out_path = os.path.join(OUTPUT_DIR, 'ej1_outliers.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("EJERCICIO 1 — DETECCIÓN DE OUTLIERS (Método IQR)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Variable objetivo: {TARGET}\n")
        f.write(f"Método: IQR (robusto ante distribuciones asimétricas)\n\n")
        f.write(f"Q1:               {Q1:.3f}\n")
        f.write(f"Q3:               {Q3:.3f}\n")
        f.write(f"IQR:              {IQR:.3f}\n")
        f.write(f"Límite inferior:  {limite_inf:.3f}\n")
        f.write(f"Límite superior:  {limite_sup:.3f}\n\n")
        f.write(f"Total outliers:   {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)\n\n")
        f.write("Top 20 outliers por Global_Sales:\n")
        f.write(outliers.nlargest(20, TARGET)[['Name', 'Platform', TARGET]].to_string(index=False))


# ─────────────────────────────────────────────
# D) VARIABLES CATEGÓRICAS
# ─────────────────────────────────────────────

def analisis_categoricas(df: pd.DataFrame) -> None:
    """
    Calcula frecuencias absolutas y relativas de cada variable categórica,
    detecta desbalance (categoría dominante > 50%) y genera gráficos de barras.
    Salida: output/ej1_categoricas.png

    Parámetros:
        df (pd.DataFrame): Dataset cargado.

    Retorna:
        None
    """
    print("\n" + "=" * 60)
    print("D) VARIABLES CATEGÓRICAS")
    print("=" * 60)

    cols_cat = df.select_dtypes(include=['object', 'str']).columns.tolist()
    cols_mostrar = [c for c in cols_cat if df[c].nunique() <= 30]

    for col in cols_cat:
        freq_abs = df[col].value_counts()
        freq_rel = df[col].value_counts(normalize=True) * 100
        tabla = pd.DataFrame({'Frecuencia': freq_abs, 'Porcentaje (%)': freq_rel.round(2)})
        print(f"\n--- {col} ({df[col].nunique()} categorías) ---")
        print(tabla.head(10).to_string())
        top_pct = freq_rel.iloc[0]
        if top_pct > 50:
            print(f"  ⚠ DESBALANCE: '{freq_abs.index[0]}' domina con {top_pct:.1f}% de los datos.")

    n = len(cols_mostrar)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 5))
    axes = axes.flatten()

    for i, col in enumerate(cols_mostrar):
        freq = df[col].value_counts().head(15)
        axes[i].bar(freq.index, freq.values, color='coral', edgecolor='black')
        axes[i].set_title(f'Frecuencia: {col}', fontsize=11)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distribución de Variables Categóricas', fontsize=15)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'ej1_categoricas.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# E) CORRELACIONES
# ─────────────────────────────────────────────

def analisis_correlaciones(df: pd.DataFrame) -> None:
    """
    Calcula y visualiza la matriz de correlaciones de Pearson de las variables
    numéricas. Identifica las 3 variables con mayor correlación con el target
    y detecta posible multicolinealidad (|r| > 0.9).
    Salida: output/ej1_heatmap_correlacion.png

    Parámetros:
        df (pd.DataFrame): Dataset cargado.

    Retorna:
        None
    """
    print("\n" + "=" * 60)
    print("E) CORRELACIONES")
    print("=" * 60)

    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    df_num = df[cols_num].dropna()
    corr_matrix = df_num.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax, annot_kws={'size': 8})
    ax.set_title('Matriz de Correlaciones de Pearson', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'ej1_heatmap_correlacion.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Guardado] {out_path}")

    corr_target = corr_matrix[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print(f"\nTop 3 variables con mayor correlación con '{TARGET}':")
    print(corr_target.head(3).to_string())

    print("\nDetección de multicolinealidad (|r| > 0.9):")
    encontrado = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.9:
                print(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: r = {r:.4f}")
                encontrado = True
    if not encontrado:
        print("  No se detectaron pares con |r| > 0.9")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    df = cargar_datos(DATA_PATH)

    resumen_estructural(df)
    estadisticos_numericos(df)
    plot_histogramas(df)
    plot_boxplots(df)
    detectar_outliers_iqr(df)
    analisis_categoricas(df)
    analisis_correlaciones(df)