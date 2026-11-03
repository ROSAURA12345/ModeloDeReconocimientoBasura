"""Métricas multiclase (cap. 15), calibración (cap. 12) y tablas. Sin torch.

Lo que se informa siempre: la tabla por clase CON su soporte, el F1 macro (el
único promedio que no esconde una clase rara) y la matriz de confusión. La
exactitud sola nunca.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support)


def resumen(y, p, n_clases: int) -> dict:
    etiquetas = list(range(n_clases))
    return {
        "exactitud": float(accuracy_score(y, p)),
        "exactitud_balanceada": float(balanced_accuracy_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "f1_macro": float(f1_score(y, p, labels=etiquetas, average="macro", zero_division=0)),
        "f1_ponderado": float(f1_score(y, p, labels=etiquetas, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, p)) if len(set(y)) > 1 else float("nan"),
    }


def f1_macro_fn(n_clases: int):
    etiquetas = list(range(n_clases))
    return lambda y, p: f1_score(y, p, labels=etiquetas, average="macro", zero_division=0)


def tabla_por_clase(y, p, clases: list[str]) -> pd.DataFrame:
    pr, rc, f1, sop = precision_recall_fscore_support(
        y, p, labels=list(range(len(clases))), zero_division=0)
    return pd.DataFrame({"clase": clases, "precision": pr, "sensibilidad": rc,
                         "f1": f1, "soporte": sop}).round(4)


def matriz(y, p, n_clases: int) -> np.ndarray:
    return confusion_matrix(y, p, labels=list(range(n_clases)))


def ece(probs: np.ndarray, y: np.ndarray, cajas: int = 10) -> float:
    """Error de calibración esperado sobre la confianza máxima (10 cajas de igual anchura)."""
    conf = probs.max(1)
    acierto = (probs.argmax(1) == y).astype(float)
    bordes = np.linspace(0, 1, cajas + 1)
    total = 0.0
    for a, b in zip(bordes[:-1], bordes[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs(acierto[m].mean() - conf[m].mean())
    return float(total)


def umbral_confianza(probs: np.ndarray, y: np.ndarray, precision_objetivo: float) -> float:
    """Menor confianza u tal que, entre las predicciones con conf >= u, la exactitud
    alcanza la precisión objetivo. Se elige en VALIDACIÓN, nunca en prueba (cap. 12.3.1).
    Debajo de u, el sistema dice 'no estoy seguro' en vez de mandar al tacho equivocado."""
    conf = probs.max(1)
    acierto = probs.argmax(1) == y
    orden = np.argsort(-conf)
    acum = np.cumsum(acierto[orden]) / np.arange(1, len(orden) + 1)
    validos = np.nonzero(acum >= precision_objetivo)[0]
    if len(validos) == 0:
        return float(conf.max()) if len(conf) else 1.0
    return float(conf[orden][validos.max()])


def a_markdown(df: pd.DataFrame) -> str:
    """Tabla Markdown sin depender de 'tabulate'."""
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    filas = ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def figura_confusion(m: np.ndarray, etiquetas: list[str], destino, titulo: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filas = m.sum(1, keepdims=True)
    norm = np.divide(m, filas, out=np.zeros_like(m, dtype=float), where=filas > 0)
    fig, ax = plt.subplots(figsize=(1.1 * len(etiquetas) + 3, 1.0 * len(etiquetas) + 2.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            ax.text(j, i, f"{m[i, j]}\n{100 * norm[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=40, ha="right")
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(destino, dpi=140)
    plt.close(fig)
