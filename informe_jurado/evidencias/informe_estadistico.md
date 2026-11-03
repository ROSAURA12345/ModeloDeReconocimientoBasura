# Informe estadístico — residuos-factorial-aumento

Respuesta: **f1_macro** · 4 tratamientos · K = 5 pliegues completos · factor A = arquitectura, factor B = aumento.

## 1. Descriptivos y costo

| tratamiento | count | mean | std | min | max |
|---|---|---|---|---|---|
| mobilenet_v3_large+basico | 5 | 0.9745 | 0.0023 | 0.972 | 0.9766 |
| efficientnet_b0+basico | 5 | 0.9712 | 0.0041 | 0.9654 | 0.9751 |
| mobilenet_v3_large+fuerte | 5 | 0.9699 | 0.0022 | 0.9674 | 0.9724 |
| efficientnet_b0+fuerte | 5 | 0.9668 | 0.0036 | 0.9641 | 0.9727 |

| tratamiento | duracion_s | img_s | mejor_epoca | entrenables |
|---|---|---|---|---|
| efficientnet_b0+basico | 2444.7 | 30.5 | 9.2 | 3950785.0 |
| efficientnet_b0+fuerte | 2246.3 | 32.9 | 8.6 | 3950785.0 |
| mobilenet_v3_large+basico | 1445.8 | 50.0 | 9.0 | 4117799.0 |
| mobilenet_v3_large+fuerte | 1556.7 | 46.0 | 8.4 | 4117799.0 |

## 2. Supuestos

- Shapiro-Wilk efficientnet_b0+basico: W = 0.914, p = 0,493
- Shapiro-Wilk efficientnet_b0+fuerte: W = 0.808, p = 0,094
- Shapiro-Wilk mobilenet_v3_large+basico: W = 0.791, p = 0,068
- Shapiro-Wilk mobilenet_v3_large+fuerte: W = 0.909, p = 0,462
- Levene: W = 0.454, p = 0,718

Con K = 5 las pruebas de supuestos tienen poca potencia: que no rechacen no prueba que se cumplan. Por eso se reporta también la ruta no paramétrica.

## 3. ANOVA factorial con bloques (pliegue)

| termino | SC | gl | CM | F | p | eta2_parcial |
|---|---|---|---|---|---|---|
| A | 5.1e-05 | 1 | 5.1e-05 | 6.399 | 0.02643 | 0.348 |
| B | 0.000104 | 1 | 0.000104 | 12.958 | 0.003647 | 0.519 |
| AxB | 0.0 | 1 | 0.0 | 0.003 | 0.9592 | 0.0 |
| bloque (pliegue) | 6.3e-05 | 4 | 1.6e-05 | 1.966 | 0.164 | 0.396 |
| error | 9.6e-05 | 12 | 8e-06 | nan |  | nan |

- Efecto principal de A (efficientnet_b0 → mobilenet_v3_large): +0.0032
- Efecto principal de B (basico → fuerte): -0.0046
- Interacción (ganancia de B con mobilenet_v3_large menos ganancia de B con efficientnet_b0): -0.0001


## 4. ANOVA de una vía sobre los tratamientos y Tukey HSD

F(3, 16) = 5.198, p = 0,011, η² = 0.494

| grupo_1 | grupo_2 | dif_medias(2-1) | ic_inf | ic_sup | p_ajust | rechaza_H0 |
|---|---|---|---|---|---|---|
| efficientnet_b0+basico | efficientnet_b0+fuerte | -0.0045 | -0.0102 | 0.0012 | 0.1517 | False |
| efficientnet_b0+basico | mobilenet_v3_large+basico | 0.0033 | -0.0024 | 0.009 | 0.3863 | False |
| efficientnet_b0+basico | mobilenet_v3_large+fuerte | -0.0014 | -0.0071 | 0.0043 | 0.9035 | False |
| efficientnet_b0+fuerte | mobilenet_v3_large+basico | 0.0078 | 0.0021 | 0.0135 | 0.006412 | True |
| efficientnet_b0+fuerte | mobilenet_v3_large+fuerte | 0.0031 | -0.0026 | 0.0088 | 0.4209 | False |
| mobilenet_v3_large+basico | mobilenet_v3_large+fuerte | -0.0046 | -0.0103 | 0.0011 | 0.1354 | False |

Los intervalos que no contienen el cero son diferencias significativas (cap. 18.4).

## 5. Ruta no paramétrica (medidas repetidas)

Friedman: χ² = 12.120, p = 0,007

| par | dif_mediana | p | p_holm |
|---|---|---|---|
| efficientnet_b0+basico vs efficientnet_b0+fuerte | 0.0049 | 0.0625 | 0.375 |
| efficientnet_b0+basico vs mobilenet_v3_large+basico | -0.001 | 0.0625 | 0.375 |
| efficientnet_b0+basico vs mobilenet_v3_large+fuerte | 0.0013 | 0.625 | 0.625 |
| efficientnet_b0+fuerte vs mobilenet_v3_large+basico | -0.0079 | 0.0625 | 0.375 |
| efficientnet_b0+fuerte vs mobilenet_v3_large+fuerte | -0.0048 | 0.3125 | 0.625 |
| mobilenet_v3_large+basico vs mobilenet_v3_large+fuerte | 0.0034 | 0.0625 | 0.375 |

Con K = 5 el menor p posible de Wilcoxon es 0.0625: esta ruta casi no puede rechazar. Considere K = 10 (cap. 10.2.1).

## 6. Redacción sugerida (formato del cap. 18.3.1)

> Se evaluó un diseño factorial 2 × 2 (arquitectura × aumento) con validación cruzada agrupada y estratificada de 5 pliegues, empleando el pliegue como bloque. El efecto de arquitectura sobre el f1_macro fue F(1, 12) = 6.40, p = 0,026, η²p = 0.348; el de aumento, F(1, 12) = 12.96, p = 0,004, η²p = 0.519; y la interacción, F(1, 12) = 0.00, p = 0,959. El tratamiento con mayor media fue mobilenet_v3_large+basico (0.9745 ± 0.0023).

Limitación a declarar (cap. 17.6): los pliegues comparten datos de entrenamiento, por lo que los valores p pueden ser algo optimistas. Además, la parada temprana elige la época con el mismo pliegue que se reporta, de modo que las medias absolutas están sesgadas al alza (el sesgo es igual para todos los tratamientos, así que la comparación sigue siendo justa). Las conclusiones se apoyan también en los tamaños del efecto y en la evaluación final sobre la prueba apartada (etapa 6).

Supuesto de normalidad: sin evidencia en contra.

![](figuras_estudio.png)
