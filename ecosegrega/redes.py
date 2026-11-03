"""Redes preentrenadas, congelado por etapas y Grad-CAM++ (caps. 6 y 21).

La clase `Red` envuelve un modelo de torchvision y sabe:
  * qué módulos son su cabeza, sus 4 etapas y su "tronco" de entrada;
  * congelar/liberar etapas (cabeza | parcial | completo) y descongelar gradual;
  * armar los grupos de parámetros con tasas discriminativas y sin decaimiento
    en sesgos/normalización;
  * mantener en eval() la normalización y el dropout de los bloques congelados;
  * guardarse y cargarse con la receta de preprocesamiento de sus pesos.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import StochasticDepth

CATALOGO_REDES = {
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "IMAGENET1K_V2"),
    "efficientnet_b0":    ("efficientnet_b0",    "EfficientNet_B0_Weights",    "IMAGENET1K_V1"),
    "resnet18":           ("resnet18",           "ResNet18_Weights",           "IMAGENET1K_V1"),
    "resnet50":           ("resnet50",           "ResNet50_Weights",           "IMAGENET1K_V2"),
}
_NORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _trozos(lista, k):
    n = len(lista)
    c = [round(i * n / k) for i in range(k + 1)]
    return [lista[c[i]:c[i + 1]] for i in range(k) if c[i + 1] > c[i]]


class Red:
    def __init__(self, nombre: str, n_clases: int, preentrenada: bool = True, dropout: float = 0.2):
        if nombre not in CATALOGO_REDES:
            raise SystemExit(f"Red '{nombre}' no disponible. Opciones: {list(CATALOGO_REDES)}")
        fn, cls_pesos, version = CATALOGO_REDES[nombre]
        pesos = getattr(models, cls_pesos)[version] if preentrenada else None
        m = getattr(models, fn)(weights=pesos)
        self.nombre = nombre
        self.pesos = f"{cls_pesos}.{version}" if pesos is not None else "aleatorios"
        self.receta = {"media": [0.485, 0.456, 0.406], "desv": [0.229, 0.224, 0.225],
                       "redimension": 256, "interpolacion": "bilinear"}
        if pesos is not None:
            tr = pesos.transforms()
            rs = tr.resize_size
            self.receta = {"media": [float(v) for v in tr.mean], "desv": [float(v) for v in tr.std],
                           "redimension": int(rs[0] if isinstance(rs, (list, tuple)) else rs),
                           "interpolacion": str(getattr(tr.interpolation, "value", tr.interpolation))}
        if nombre.startswith("resnet"):
            m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, n_clases))
            self.etapas = [[m.layer1], [m.layer2], [m.layer3], [m.layer4]]
            self.cabeza, self.capa_mapa = m.fc, m.layer4
        elif nombre.startswith("mobilenet"):
            m.classifier[2] = nn.Dropout(dropout)
            m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_clases)
            self.etapas = _trozos(list(m.features.children()), 4)
            self.cabeza, self.capa_mapa = m.classifier, m.features[-1]
        else:
            m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, n_clases))
            self.etapas = _trozos(list(m.features.children()), 4)
            self.cabeza, self.capa_mapa = m.classifier, m.features[-1]
        self.modelo = m
        self.objetivo: list = []
        self.estrategia = "cabeza"

    # ---------------- congelado (cap. 6.2 - 6.4) ----------------
    @staticmethod
    def _p(mods):
        return [p for mod in mods for p in mod.parameters()]

    def tronco(self):
        ids = {id(p) for e in self.etapas for p in self._p(e)} | {id(p) for p in self.cabeza.parameters()}
        return [p for p in self.modelo.parameters() if id(p) not in ids]

    def plan(self, estrategia: str, etapas_liberadas: int) -> None:
        desde_salida = list(reversed(self.etapas))
        if estrategia == "cabeza":
            self.objetivo = []
        elif estrategia == "parcial":
            self.objetivo = desde_salida[:max(0, int(etapas_liberadas))]
        elif estrategia == "completo":
            self.objetivo = desde_salida
        else:
            raise SystemExit("estrategia debe ser cabeza, parcial o completo")
        self.estrategia = estrategia

    def liberar(self, n_etapas: int) -> int:
        """Congela todo, libera la cabeza + las n primeras etapas del objetivo. Devuelve entrenables."""
        for p in self.modelo.parameters():
            p.requires_grad_(False)
        for p in self.cabeza.parameters():
            p.requires_grad_(True)
        for etapa in self.objetivo[:n_etapas]:
            for p in self._p(etapa):
                p.requires_grad_(True)
        if self.estrategia == "completo" and n_etapas >= len(self.objetivo):
            for p in self.tronco():
                p.requires_grad_(True)
        return sum(p.numel() for p in self.modelo.parameters() if p.requires_grad)

    def grupos(self, tasa: float, factor: float, decaimiento: float) -> list[dict]:
        bloques = [("cabeza", list(self.cabeza.parameters()), tasa)]
        bloques += [(f"etapa-{k}", self._p(e), tasa / factor ** k) for k, e in enumerate(self.objetivo, 1)]
        if self.estrategia == "completo" and self.tronco():
            bloques.append(("tronco", self.tronco(), tasa / factor ** (len(self.objetivo) + 1)))
        out = []
        for nombre, ps, lr in bloques:
            for suf, sub, wd in (("", [p for p in ps if p.ndim > 1], decaimiento),
                                 ("/sin_wd", [p for p in ps if p.ndim <= 1], 0.0)):
                if sub:
                    out.append({"params": sub, "lr": lr, "lr_base": lr, "weight_decay": wd,
                                "nombre": nombre + suf})
        return out

    def modo_entrenamiento(self) -> None:
        """train() + apagar normalización/dropout/StochasticDepth de lo congelado (cap. 6.4.1)."""
        self.modelo.train()
        for m in self.modelo.modules():
            ps = list(m.parameters())
            if isinstance(m, _NORMS) and ps and not any(p.requires_grad for p in ps):
                m.eval()
            if ps and not any(p.requires_grad for p in ps):
                for sub in m.modules():
                    if isinstance(sub, (nn.Dropout, StochasticDepth)):
                        sub.eval()

    # ---------------- persistencia ----------------
    def guardar(self, destino, clases, extra: dict) -> None:
        torch.save({"estado": self.modelo.state_dict(), "red": self.nombre, "clases": list(clases),
                    "pesos_preentrenados": self.pesos, "receta": self.receta, **extra}, destino)

    @classmethod
    def cargar(cls, archivo, equipo: torch.device):
        try:
            ck = torch.load(archivo, map_location="cpu", weights_only=True)
        except TypeError:
            ck = torch.load(archivo, map_location="cpu")
        red = cls(ck["red"], len(ck["clases"]), preentrenada=False)
        red.modelo.load_state_dict(ck["estado"])
        red.receta = ck["receta"]
        red.pesos = ck["pesos_preentrenados"]
        red.modelo.to(equipo).eval()
        return red, ck


# ---------------- Grad-CAM++ (cap. 21.1, tabla 21.1) ----------------
def gradcam_pp(modelo, capa, x: torch.Tensor, clase: int | None = None):
    """Grad-CAM++: pesos por posición a partir de gradientes de orden 2 y 3 (aproximados
    con la exponencial de la puntuación). Reparte mejor que Grad-CAM cuando hay varios
    objetos de la misma clase. Devuelve (mapa HxW en [0,1], clase, probabilidad)."""
    guardado = {}

    def gancho(_m, _e, salida):
        guardado["A"] = salida
        salida.register_hook(lambda g: guardado.__setitem__("G", g))

    h = capa.register_forward_hook(gancho)
    try:
        modelo.eval()
        modelo.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        z = modelo(x)
        prob = z.softmax(1)
        if clase is None:
            clase = int(prob.argmax(1))
        z[0, clase].backward()
        A, G = guardado["A"].detach(), guardado["G"].detach()
        g2, g3 = G ** 2, G ** 3
        suma_A = A.sum(dim=(2, 3), keepdim=True)
        denom = 2 * g2 + suma_A * g3
        alfa = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        w = (alfa * F.relu(G)).sum(dim=(2, 3), keepdim=True)
        mapa = F.relu((w * A).sum(1, keepdim=True))
        mapa = F.interpolate(mapa, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        mapa = mapa - mapa.min()
        if float(mapa.max()) > 0:
            mapa = mapa / mapa.max()
        return mapa.cpu().numpy(), clase, float(prob[0, clase])
    finally:
        h.remove()


def superponer(imagen, mapa: np.ndarray, alfa: float = 0.45):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    from PIL import Image

    m = Image.fromarray((mapa * 255).astype(np.uint8)).resize(imagen.size, Image.Resampling.BILINEAR)
    color = colormaps["inferno"](np.asarray(m) / 255.0)[..., :3]
    base = np.asarray(imagen.convert("RGB"), dtype=np.float32) / 255.0
    return Image.fromarray(((1 - alfa) * base + alfa * color).clip(0, 1).__mul__(255).astype(np.uint8))
