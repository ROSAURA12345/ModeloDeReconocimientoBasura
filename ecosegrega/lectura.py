"""Lectura de imágenes para PyTorch: Dataset, transformaciones y DataLoader (cap. 2.6).

Dos recetas de aumento (básico / fuerte) definidas en el YAML; el aumento solo se
aplica en entrenamiento. La evaluación usa exactamente la receta de los pesos
preentrenados: tamaño, interpolación, media y desviación (cap. 6.4.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .fotos import abrir_rgb
from .hardware import generador, semilla_worker
from .rutas import ruta


def recetas(receta: dict, tam: int, aumento: dict | None):
    """receta = {media, desv, redimension, interpolacion} tomada de los pesos."""
    modo = T.InterpolationMode(receta["interpolacion"])
    lado = int(round(receta["redimension"] * tam / 224))
    norm = T.Normalize(receta["media"], receta["desv"])
    evaluacion = T.Compose([T.Resize(lado, interpolation=modo), T.CenterCrop(tam), T.ToTensor(), norm])
    if aumento is None:
        return evaluacion, evaluacion
    pasos = [T.RandomResizedCrop(tam, scale=(float(aumento.get("recorte_min", 0.8)), 1.0), interpolation=modo)]
    if aumento.get("volteo", True):
        pasos.append(T.RandomHorizontalFlip())
    if float(aumento.get("rotacion", 0)):
        pasos.append(T.RandomRotation(float(aumento["rotacion"])))
    c = float(aumento.get("color", 0))
    if c:
        pasos.append(T.ColorJitter(c, c, c, min(0.05, c / 4)))
    pasos += [T.ToTensor(), norm]
    b = float(aumento.get("borrado", 0))
    if b:
        pasos.append(T.RandomErasing(p=b, scale=(0.02, 0.15)))
    return T.Compose(pasos), evaluacion


class Fotos(Dataset):
    def __init__(self, tabla: pd.DataFrame, clases: list[str], transformacion):
        self.rutas = [str(ruta(r)) for r in tabla["ruta"]]
        indice = {c: i for i, c in enumerate(clases)}
        self.y = np.array([indice[c] for c in tabla["clase"]], dtype=np.int64)
        self.t = transformacion

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        return self.t(abrir_rgb(self.rutas[i])), int(self.y[i])


def cargador(ds: Dataset, lote: int, entrenar: bool, equipo: torch.device, hilos: int, semilla: int):
    return DataLoader(ds, batch_size=lote, shuffle=entrenar, drop_last=entrenar and len(ds) > lote,
                      num_workers=hilos, pin_memory=equipo.type == "cuda",
                      persistent_workers=entrenar and hilos > 0,
                      worker_init_fn=semilla_worker, generator=generador(semilla))


def pesos_clase(y: np.ndarray, n: int) -> torch.Tensor:
    """Inverso de la frecuencia (cap. 12.2), calculado SOLO con el entrenamiento del pliegue."""
    c = np.bincount(y, minlength=n).astype(float)
    w = np.where(c > 0, c.sum() / (n * np.maximum(c, 1)), 0.0)
    return torch.tensor(w / w[c > 0].mean(), dtype=torch.float32)
