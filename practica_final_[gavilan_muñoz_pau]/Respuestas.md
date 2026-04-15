# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

El dataset elegido es **Video Game Sales with Ratings** de Kaggle. Contiene 16.719
filas y 16 columnas con información sobre videojuegos: plataforma, género, publisher,
ventas por región y puntuaciones de críticos y usuarios. La variable objetivo es
`Global_Sales` (ventas globales en millones de unidades), una variable numérica
continua con sentido claro para regresión.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset proviene de Kaggle (Video Game Sales with Ratings). La variable objetivo
> es `Global_Sales`. Tiene sentido hacer regresión porque es una variable numérica
> continua que representa las ventas globales de cada juego en millones de unidades,
> y queremos predecir cuánto venderá un juego según sus características.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Todas las variables de ventas tienen distribuciones muy asimétricas hacia la derecha
> (skewness de `Global_Sales` ≈ 17,38), con la mayoría de juegos vendiendo menos de
> 0,5M y muy pocos superando los 10M. Sí hay outliers, especialmente en `Global_Sales`
> (ej: Wii Sports con 82,53M). Se detectaron usando el método IQR por ser más robusto
> ante asimetría. Se han mantenido en el dataset ya que son datos reales válidos, solo
> documentados.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Las tres variables con mayor correlación con `Global_Sales` son:
> - `NA_Sales` (r = 0.96)
> - `EU_Sales` (r = 0.94)
> - `Other_Sales` (r = 0.80)
>
> Es esperable ya que `Global_Sales` es la suma de todas las ventas regionales.
> Se detectó multicolinealidad entre ellas con |r| > 0,9.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Sí, varias columnas tienen nulos. Los más relevantes son `Critic_Score` (51,33%),
> `User_Count` (54,60%), `User_Score` (40,10%) y `Developer` (39,61%). El tratamiento
> aplicado fue: eliminación de filas sin `Year_of_Release` para el análisis temporal,
> y uso local con `dropna()` para columnas con muchos nulos, sin imputar, para no
> distorsionar el análisis general.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

Se eliminaron columnas con alta cardinalidad o muchos nulos (`Name`, `Developer`,
`Publisher`, `User_Score`, `Critic_Score`). Las variables categóricas `Platform`,
`Genre` y `Rating` se codificaron con `get_dummies`. Las numéricas se escalaron con
`StandardScaler`. Split 80/20 con `random_state=42`.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal 
sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> - **MAE = 0.6979** millones de unidades de error promedio
> - **RMSE = 2.4603**
> - **R² = 0.0240** — el modelo explica solo el 2.40% de la varianza
>
> El modelo no funciona bien. El R² de 0.024 indica que las variables
> disponibles (plataforma, género, año, rating) apenas tienen capacidad
> predictiva sobre las ventas globales. Esto tiene sentido: el éxito
> comercial de un videojuego depende principalmente de factores no presentes
> en el dataset como el marketing, la franquicia, o el momento de lanzamiento.
> Además, la distribución de Global_Sales es muy asimétrica con outliers
> extremos (Wii Sports: 82M), lo que penaliza el RMSE y dificulta el ajuste
> lineal. El gráfico de residuos confirma heteroscedasticidad clara — los
> residuos aumentan con el valor predicho, violando los supuestos de la
> regresión lineal.

**Comparativa y mejoras concretas:**

> El bajo R²=0.024 confirma que el modelo lineal sobre estos datos tiene poca
> capacidad predictiva. Las mejoras más directas serían:
> - **Transformación logarítmica del target**: aplicar log(Global_Sales + 1)
>   reduciría la asimetría extrema (skewness ≈ 17) y el impacto de outliers
>   como Wii Sports (82M), haciendo los datos más compatibles con regresión lineal.
> - **Eliminación de outliers en entrenamiento**: filtrar juegos con
>   Global_Sales > límite IQR (1.085M) para que el modelo no se distorsione
>   por los extremos, aunque se perdería información de los grandes éxitos.
> - **Modelos no lineales**: Random Forest o Gradient Boosting capturarían
>   mejor las interacciones no lineales entre plataforma, género y año,
>   probablemente con R² significativamente mayor.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

Se implementó la solución analítica OLS usando solo NumPy, sobre datos sintéticos
generados con semilla 42. Los datos siguen la relación lineal
y = 5 + 2·x₁ - 1·x₂ + 0.5·x₃ + ruido(σ=1.5).

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> La fórmula calcula los coeficientes que minimizan el error cuadrático entre los
> valores reales y los predichos. `Xᵀy` proyecta el target sobre las features, y
> `(XᵀX)⁻¹` normaliza esa proyección para obtener los coeficientes óptimos.
> La columna de unos es necesaria para estimar el intercepto β₀. Sin ella, el modelo
> estaría forzado a pasar por el origen, lo que raramente es correcto.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real| Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       | 5.0383         |
| β₁        | 2.0       | 1.8938         |
| β₂        |-1.0       |-1.0475         |
| β₃        | 0.5       | 0.4650         |

> Los coeficientes ajustados son muy próximos a los reales. La pequeña desviación
> se debe al ruido gaussiano añadido durante la generación de datos (σ=1.5).

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> - **MAE = 1.2231**
> - **RMSE = 1.5748**
> - **R² = 0.7248** 
>
> Sí, los valores se aproximan correctamente a los de referencia. El R² ≈ 0,80 indica
> que el modelo explica el 80% de la varianza, y el 20% restante corresponde al ruido
> no predictible.

**Pregunta 3.4** — Compara los resultados con la regresión del Ejercicio 2 y explica qué ha sucedido.

---

## Ejercicio 4 — Series Temporales

La serie sintética cubre 6 años de datos diarios (2.192 observaciones, 2018–2023)
generada con `seed=42`. Combina tendencia lineal creciente (parte de 50), 
estacionalidad anual con dos armónicos, ciclo de ~4 años y ruido gaussiano (σ=3.5).
Se aplicó descomposición aditiva con `period=365`.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> Sí, presenta una tendencia **lineal creciente** generada con `0.05 * t + 50`.
> Parte de un valor inicial ≈ 50 y alcanza ≈ 160 al final de los 6 años, lo que
> supone un incremento acumulado de ≈ +110 unidades. Se aprecia claramente en la
> componente de tendencia de la descomposición, que sube de forma continua
> y uniforme a lo largo de todo el período.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Sí, estacionalidad anual con **periodo de 365 días**. La amplitud es de
> aproximadamente **±21 unidades** (combinación de seno y coseno: `15·sin + 6·cos`). El patrón se repite cada año con la misma forma,
> visible en el gráfico de estacionalidad de la descomposición.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> Sí, hay un ciclo de largo plazo de ≈ **4 años** (~1461 días) y amplitud ±8 unidades.
> Se diferencia de la tendencia en que es oscilatorio y reversible — sube y baja —
> mientras que la tendencia es unidireccional y acumulativa. En la serie original
> se aprecian las ondulaciones de largo plazo superpuestas sobre la subida general.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> - **Media = 0.1271**
> - **Std = 3.2220**
> - **Asimetría = -0.0509**
> - **Curtosis = -0.0610**
> - **Test ADF: p-valor = 0.000000**
> - **Test Jarque-Bera: p-valor = 0.5766**
>
> El residuo sí se ajusta a un ruido ideal gaussiano: media ≈ 0, desviación típica
> próxima al σ=3.5 original, completamente estacionario según ADF (p≈0), y
> distribución normal confirmada por Jarque-Bera (p=0.58 > 0.05). Los gráficos
> ACF y PACF muestran todos los coeficientes dentro de las bandas de confianza,
> confirmando ausencia de autocorrelación — ruido blanco ideal.

---

*Fin del documento de respuestas*