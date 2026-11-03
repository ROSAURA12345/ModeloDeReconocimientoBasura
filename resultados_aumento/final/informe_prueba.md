# Evaluación final en la prueba apartada — mobilenet_v3_large+basico

- Red `mobilenet_v3_large` (pesos `MobileNet_V3_Large_Weights.IMAGENET1K_V2`), factor B = aumento:basico, 10 épocas fijas, entrenada con 6954 imágenes en 22.8 min
- Prueba: 1228 imágenes de 1228 grupos; nunca usada antes de esta etapa

| Métrica | Valor | IC 95 % (bootstrap por grupo) |
|---|---|---|
| F1 macro | 0.9762 | [0.9666, 0.9854] |
| Exactitud | 0.9796 | [0.9715, 0.9870] |
| Exactitud balanceada | 0.9788 | |
| Exactitud de tacho (no municipal, 7 colores) | 0.9796 | |
| Exactitud de tacho (municipal, 4 colores) | 0.9927 | |
| Cobertura del umbral 0.30 | 100.0 % | |
| Exactitud en casos 'seguros' | 0.9796 | |
| ECE sin calibrar → calibrado | 0.1308 → 0.0068 | |

## Por clase

| clase | precision | sensibilidad | f1 | soporte |
|---|---|---|---|---|
| plastico | 0.9615 | 0.969 | 0.9653 | 129 |
| papel_carton | 0.9931 | 0.9897 | 0.9914 | 290 |
| vidrio | 0.9965 | 0.9697 | 0.9829 | 297 |
| metal | 0.9231 | 0.9391 | 0.931 | 115 |
| organico | 0.9932 | 0.9932 | 0.9932 | 148 |
| peligroso | 0.9592 | 1.0 | 0.9792 | 141 |
| no_aprovechable | 0.9907 | 0.9907 | 0.9907 | 108 |

## Por fuente (¿funciona con fotos tomadas en Perú?)

| fuente | imagenes | exactitud | f1_macro |
|---|---|---|---|
| roboflow:garbage-detection-l4ilj-v1 | 1228 | 0.9796 | 0.9762 |

Media en validación cruzada para este tratamiento: 0.9745 (F1 macro). Esa cifra es algo optimista: la parada temprana eligió la época mirando el mismo pliegue que se reporta. La cifra que se puede prometer es la de esta prueba (cap. 5.3 y 10.4). Si la prueba sale mucho peor, sospeche además sobreajuste a los pliegues o diferencia de dominio (cap. 5.4).

![](confusion_clases.png)

![](confusion_tachos.png)
