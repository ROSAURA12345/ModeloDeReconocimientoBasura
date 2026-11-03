"""Clasificador de fotos nuevas con el modelo final."""
import torch

from . import hardware, lectura
from .norma import Catalogo
from .redes import Red


class Clasificador:
    def __init__(self, modelo, equipo="auto", norma="config/ntp_900058.yaml"):
        self.equipo = hardware.elegir_equipo(equipo)
        self.red, self.ck = Red.cargar(modelo, self.equipo)
        self.clases = self.ck["clases"]
        self.T = float(self.ck.get("temperatura", 1.0))
        self.umbral = float(self.ck.get("umbral_confianza", 0.0))
        _, self.t = lectura.recetas(self.red.receta, int(self.ck["tam_imagen"]), None)
        self.cat = Catalogo.desde_yaml(norma)

    @torch.no_grad()
    def __call__(self, imagen, esquema="no_municipal"):
        x = self.t(imagen).unsqueeze(0).to(self.equipo)
        probs = (self.red.modelo(x).float() / self.T).softmax(1)[0].cpu()
        orden = probs.argsort(descending=True)
        clase = self.clases[int(orden[0])]
        tacho = self.ck["tachos"][clase][esquema]
        conf = float(probs[orden[0]])
        return {
            "clase": clase,
            "nombre": self.cat.nombre(clase),
            "confianza": conf,
            "seguro": conf >= self.umbral,
            "tacho": tacho,
            "color": self.cat.color_hex(tacho),
            "top3": [(self.clases[int(i)], float(probs[i])) for i in orden[:3]],
        }
