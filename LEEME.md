# EcoSegrega — estudio factorial de clasificación de residuos con PyTorch (CPU o GPU)

**Pregunta de investigación:** en la clasificación automática de residuos sólidos
según la NTP 900.058:2019, **¿qué influye más en el desempeño: la arquitectura de
la red o incluir fotos tomadas en el contexto peruano? ¿Y se potencian entre sí?**

Para responderla, EcoSegrega no entrena "un modelo" sino un **experimento**:

- **Diseño factorial 2 × 2** (cap. 22 del libro *Machine Learning Experimental*).
  - Factor A, arquitectura: `mobilenet_v3_large` frente a `efficientnet_b0`.
  - Factor B, fotos propias en el entrenamiento: `sin` frente a `con`.
- **Validación cruzada agrupada y estratificada de K = 5 pliegues** (cap. 10), con
  los mismos pliegues para los 4 tratamientos (diseño aparejado, cap. 17).
- **Análisis estadístico** (cap. 18): ANOVA factorial con el pliegue como bloque,
  ANOVA de una vía con Tukey HSD, y Friedman con Wilcoxon-Holm como ruta no paramétrica.
- **Un modelo final** con el tratamiento ganador, evaluado **una sola vez** en una
  prueba apartada desde el inicio (cap. 9.5).
- **Salida:** la clase del residuo y el **color de tacho**. El esquema por defecto
  es el no municipal (7 colores), pensado para la universidad o una empresa.

Se entrena con datos de **Roboflow Universe** y con **fotos propias**, en
**CPU o en GPU NVIDIA** (`--equipo cpu` / `--equipo gpu`).

---

## 1. Instalación (Windows)

Necesita Python 3.10, 3.11 o 3.12 (python.org, marque "Add Python to PATH"). No
use Conda ni instale el CUDA Toolkit (cap. 1.2–1.3).

1. Abra **cmd** en la carpeta del proyecto.
2. Ejecute `instalar_gpu.bat` (o `instalar_cpu.bat` si su PC no tiene tarjeta NVIDIA).

**Qué versión de CUDA elegir.** Escriba `nvidia-smi` en cmd y mire `CUDA Version`.
Ese número es el **máximo** que soporta el controlador (cap. 1.4.2):

| `nvidia-smi` dice | Orden de instalación |
|---|---|
| 12.6 – 12.9 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` (lo que hace el .bat) |
| 13.0 o más | puede usar `.../whl/cu130` |
| no hay tarjeta NVIDIA | `.../whl/cpu` |

Nunca use `pip install torch` a secas: desde PyTorch 2.11 trae CUDA 13 y con un
controlador más antiguo entrena en CPU sin avisar.

**Verifique siempre:** `python etapas\e0_entorno.py`. Mide la velocidad de la CPU
y de la GPU en su equipo y **estima cuántas horas tomará el estudio** en cada una.

## 2. Datos

### Roboflow
Busque "garbage classification" en [universe.roboflow.com](https://universe.roboflow.com).
Sirven datasets de **clasificación** (formato *Folder Structure*) y de
**detección** (*YOLOv8* o *COCO*). En los de detección, cada caja se recorta como
una imagen. Puede bajarlos a mano y descomprimirlos en
`datos\originales\roboflow\<nombre>\`, o con el script:

```bat
set ROBOFLOW_API_KEY=su_clave
python etapas\e1_roboflow.py --workspace wiegon --proyecto garbage-classification-3-ssg5n --version 1 --formato yolov8 --licencia "CC BY 4.0"
```

Las etiquetas en inglés (`PLASTIC`, `Glass`, `biodegradable`…) se traducen con
`equivalencias` en `config\ntp_900058.yaml`. Las que falten, la etapa 2 se las lista.

### Fotos propias (el factor B del estudio)
```
datos\originales\propias\plastico\botella_san_luis_01\  1.jpg 2.jpg 3.jpg
datos\originales\propias\vidrio\botella_pilsen_01\      ...
datos\originales\propias\peligroso\pilas_aa_01\         ...
```
Use **una subcarpeta por objeto**: las fotos del mismo objeto siempre van juntas
al mismo pliegue. Si no, el modelo "reconoce la botella" en vez de aprender el
material (fuga por grupo, caps. 8.5.2 y 9). Apunte a **15 objetos o más por clase**,
con 3 a 10 fotos cada uno, en distintos fondos, luces y ángulos.

¿Todavía no tiene fotos propias? Corra el estudio con
`--config config\estudio_aumento.yaml`. Ahí el factor B pasa a ser *aumento de
datos básico frente a fuerte*.

## 3. El estudio, etapa por etapa

| Etapa | Orden | Qué hace |
|---|---|---|
| 0 | `python etapas\e0_entorno.py` | Python, venv, PyTorch, GPU, medición CPU vs GPU (cap. 1) |
| 1 | `python etapas\e1_roboflow.py …` | Descarga con procedencia y licencia (cap. 7) |
| 2 | `python etapas\e2_inventario.py` | Junta fuentes, traduce etiquetas, recorta, valida, deduplica (caps. 7–8) |
| 3 | `python etapas\e3_pliegues.py` | Aparta la **prueba** y arma **K pliegues** agrupados y estratificados (caps. 9–10) |
| 4 | `python etapas\e4_estudio_factorial.py --equipo gpu` | 4 tratamientos × K pliegues = 20 entrenamientos. **Se puede cortar y retomar.** |
| 5 | `python etapas\e5_analisis_estadistico.py` | Supuestos, ANOVA factorial con bloques, Tukey, Friedman, gráfico de interacción |
| 6 | `python etapas\e6_modelo_final.py --equipo gpu` | Reentrena al ganador con todo y evalúa **una vez** en la prueba |
| 7 | `python etapas\e7_clasificar.py carpeta_fotos\` | Galería HTML con el tacho de cada foto |
| 8 | `python etapas\e8_mapas_gradcampp.py` | Grad-CAM++ de aciertos **y fallos** + prueba de aleatorización (cap. 21) |

O todo junto: `correr_estudio.bat gpu`.

**Antes del estudio largo, haga un ensayo rápido** que solo comprueba que todo corre:
```bat
python etapas\e3_pliegues.py --config config\ensayo_cpu.yaml
python etapas\e4_estudio_factorial.py --config config\ensayo_cpu.yaml --equipo gpu
python etapas\e5_analisis_estadistico.py --config config\ensayo_cpu.yaml
```

### CPU o GPU
Se elige con `--equipo gpu`, `--equipo cpu` o `equipo: auto` en el YAML. Con
`gpu`, si PyTorch no ve la tarjeta, el programa **se detiene** con las
instrucciones para arreglarlo: nunca entrena en CPU a escondidas (cap. 1.4).
Cada fila de `resultados\estudio\resultados_pliegues.csv` guarda el equipo usado,
los segundos y las imágenes por segundo. Así puede reportar el costo de cada
tratamiento (cap. 16.4). En CPU el estudio completo puede tardar muchas horas: use
GPU, Google Colab o reduzca `k` y `epocas`.

## 4. Qué entrega cada etapa

- `datos\derivados\resumen_inventario.txt` y `apartados.csv`: qué entró, qué se
  apartó y por qué (nada se borra, cap. 7.2.4).
- `datos\derivados\informe_pliegues.txt`: tamaños, cero solapamiento de grupos,
  clases y fuentes por pliegue.
- `resultados\estudio\resultados_pliegues.csv`: la **matriz del experimento**, con
  una fila por tratamiento y pliegue (F1 macro, exactitud, F1 en fotos propias,
  épocas, tiempo).
- `resultados\estudio\informe_estadistico.md` y `figuras_estudio.png`: tablas
  ANOVA, Tukey, Friedman, gráfico de interacción y un **párrafo redactado** en
  formato de tesis (cap. 18.3.1), con la limitación que hay que declarar.
- `resultados\final\informe_prueba.md`: F1 macro con IC 95 % (bootstrap por
  grupo), tabla por clase, resultados por fuente (Roboflow frente a fotos propias),
  exactitud de **tacho** en los dos esquemas, matrices de confusión y cobertura del
  umbral "revisar".
- `resultados\final\modelo_final.pt`: el modelo con su receta de preprocesamiento,
  temperatura y umbral.
- `resultados\bitacora.jsonl`: una línea por entrenamiento, con parámetros,
  métricas, versión de torch/CUDA, GPU, commit de git y huella de los datos (cap. 13).

## 5. Decisiones metodológicas (para el capítulo de método)

| Decisión | Por qué (libro) |
|---|---|
| Unidad de generalización = objeto fotografiado o foto original de Roboflow (las copias `*.rf.*` y los casi-duplicados se unen) | caps. 8.5, 9.1 |
| Prueba apartada al inicio y consultada una sola vez, con registro en `consultas_prueba.log` | caps. 8.7, 9.5 |
| Los mismos K pliegues y la misma semilla por pliegue para todos los tratamientos | caps. 10.5, 16.2 |
| Factor B "sin": las fotos propias se quitan solo del entrenamiento; la validación es idéntica | cap. 16.2 (igualar antes de comparar) |
| Ajuste parcial con tasas discriminativas y descongelado gradual; BatchNorm, dropout y StochasticDepth de lo congelado en `eval()` | cap. 6.4 |
| AdamW, calentamiento + coseno, recorte de gradiente, precisión mixta solo en GPU, pesos de clase | caps. 4, 12.2 |
| Preprocesamiento de evaluación = receta exacta de los pesos preentrenados | cap. 6.4.1 |
| Modelo final con épocas fijas (mediana de la mejor época en CV), sin conjunto que mirar | cap. 5.3 |
| Calibración (temperatura) y umbral con predicciones **fuera de pliegue**, no con la prueba | cap. 12.3.1 |
| F1 macro como respuesta, con tabla por clase y soporte | cap. 15.3 |
| Bloques por pliegue, η² parcial, ruta paramétrica y no paramétrica | caps. 17.6, 18.5, 22.2 |
| Grad-CAM++ con prueba de aleatorización y figuras de fallos | cap. 21 |

## 6. Código de colores (NTP 900.058:2019)

| Clase | No municipal (por defecto) | Municipal |
|---|---|---|
| Plástico | Blanco | Verde |
| Papel y cartón | Azul | Verde |
| Vidrio | Verde | Verde |
| Metal | Amarillo | Verde |
| Orgánico | Marrón | Marrón |
| Peligroso | Rojo | Rojo |
| No aprovechable | Negro | Negro |

El esquema se cambia en `decision.esquema` del YAML, o con `--esquema municipal`
en la etapa 7. Los colores se editan en `config\ntp_900058.yaml`: confírmelos con
su docente, porque algunas instituciones usan variantes.

## 7. Panel interactivo local

Para abrir el panel interactivo en el navegador, ejecute `abrir_panel.bat`.
Usa por defecto el modelo `resultados_aumento\final\modelo_final.pt`; puede
cambiar la ruta desde el panel.

El panel incluye clasificación de fotos, un modo de cámara en vivo, guía de
separación por clase y tacho, métricas del estudio y descarga de resultados.
En **Cámara en vivo**, pulse **START** y acepte el permiso de cámara del
navegador; el modelo muestra su predicción sobre el video. Puede limitar el
análisis a 1, 2 o 4 cuadros por segundo para ajustar la carga del equipo. La
cámara y las fotos se procesan localmente, no se graban ni se guardan en el
proyecto. Las predicciones de fotos se pueden descargar en CSV.

El panel escucha solo en la computadora local; no lo exponga a Internet. Para
instalar dependencias manualmente, active `venv` y ejecute
`pip install -r requisitos.txt`.

## 8. Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `Se pidió la GPU pero PyTorch no la ve` | Reinstale torch con el `--index-url` de la sección 1 |
| `CUDA out of memory` | `lote: 16` en el YAML; cierre el navegador |
| `WinError 1455` o falta RAM | `hilos_carga: 0` o `1` |
| Pérdida `nan` | `tasa: 3.0e-4` o `precision_mixta: false` |
| `etiqueta_sin_equivalencia:xxx` | Agregue `xxx: clase` en `equivalencias` |
| La etapa 4 se cortó | Vuelva a lanzarla: retoma lo que faltaba |
| Cambió fotos después de la etapa 3 | `e3_pliegues.py --forzar` y borre `resultados\estudio\` (las comparaciones deben usar los mismos pliegues) |

Pruebas sin GPU ni torch: `python pruebas\test_rapido.py`.
