"""El entrenador: el bucle canónico del cap. 2.7 con las técnicas del cap. 4.

Uso:
    ent = Entrenador(cfg, clases, equipo, "efficientnet_b0", aumento)
    res = ent.ajustar(df_entrenamiento, df_validacion)    # validación cruzada: con parada temprana
    res = ent.ajustar(df_todo, None, epocas_fijas=7)      # modelo final: épocas fijas, sin mirar prueba
    logits, y = ent.predecir(df)

Cuida, en cada paso: zero_grad, train()/eval(), .item() al acumular, datos y pesos
de clase en el mismo dispositivo, pérdida dividida al acumular, desescalar antes
de recortar con precisión mixta, y restaurar el MEJOR estado al terminar.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lectura
from .redes import Red


class _SinEscala:
    def scale(self, x):
        return x

    def unscale_(self, opt):
        pass

    def step(self, opt):
        opt.step()

    def update(self):
        pass


def factor_lr(paso: int, calentamiento: int, total: int) -> float:
    """Calentamiento lineal + coseno (cap. 4.4.2)."""
    if paso < calentamiento:
        return (paso + 1) / max(1, calentamiento)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (paso - calentamiento) / max(1, total - calentamiento))))


def f1_macro(y, p, n):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(range(n)), average="macro", zero_division=0))


class Entrenador:
    def __init__(self, cfg: dict, clases: list[str], equipo: torch.device, red: str, aumento: dict):
        self.cfg, self.ec = cfg, cfg["entrenamiento"]
        self.clases, self.n = clases, len(clases)
        self.equipo = equipo
        self.red = Red(red, self.n, preentrenada=True)
        self.red.plan(self.ec["estrategia"], int(self.ec.get("etapas_liberadas", 1)))
        self.t_tr, self.t_ev = lectura.recetas(self.red.receta, int(self.ec["tam_imagen"]), aumento)
        self.amp = bool(self.ec.get("precision_mixta", True)) and equipo.type == "cuda"
        self.semilla = int(cfg["semilla"])

    # ------------------------------------------------------------------
    def _cargador(self, df, entrenar):
        t = self.t_tr if entrenar else self.t_ev
        hilos = int(self.ec.get("hilos_carga", 2))
        lote = int(self.ec["lote"]) * (1 if entrenar else 2)
        return lectura.cargador(lectura.Fotos(df, self.clases, t), lote, entrenar, self.equipo, hilos,
                                self.semilla)

    @torch.no_grad()
    def _logits(self, dl, amp):
        self.red.modelo.eval()
        Z, Y = [], []
        for x, y in dl:
            with torch.autocast(device_type=self.equipo.type, dtype=torch.float16, enabled=amp):
                z = self.red.modelo(x.to(self.equipo, non_blocking=True))
            Z.append(z.float().cpu())
            Y.append(y)
        return torch.cat(Z).numpy(), torch.cat(Y).numpy()

    def predecir(self, df: pd.DataFrame):
        dl = self._cargador(df, False)
        z, y = self._logits(dl, self.amp)
        if not np.isfinite(z).all() and self.amp:   # GTX 16xx: a veces fp16 da NaN
            print("[AVISO] NaN/inf en fp16; se repite la predicción en fp32.")
            z, y = self._logits(dl, False)
        if not np.isfinite(z).all():
            raise SystemExit("[ERROR] El modelo produce NaN/inf: baje la tasa o desactive precision_mixta.")
        return z, y

    # ------------------------------------------------------------------
    def ajustar(self, df_tr: pd.DataFrame, df_va: pd.DataFrame | None, epocas_fijas: int | None = None,
                al_terminar_epoca=None) -> dict:
        ec, red, eq = self.ec, self.red, self.equipo
        epocas = int(epocas_fijas or ec["epocas"])
        gradual_k = 2 if len(red.objetivo) else 0
        liberar = (lambda e: min(len(red.objetivo), e // gradual_k)) if gradual_k else (lambda e: 0)
        red.modelo.to(eq)
        red.liberar(liberar(0))
        opt = torch.optim.AdamW(red.grupos(float(ec["tasa"]), float(ec.get("factor_discriminativo", 2.6)),
                                           float(ec["decaimiento"])))
        dl_tr = self._cargador(df_tr, True)
        ds_y = dl_tr.dataset.y
        pesos = lectura.pesos_clase(ds_y, self.n).to(eq) if ec.get("pesos_clase", True) else None
        criterio = nn.CrossEntropyLoss(weight=pesos, label_smoothing=float(ec.get("suavizado_etiquetas", 0.0)))
        esc = torch.amp.GradScaler("cuda") if self.amp else _SinEscala()
        total = epocas * len(dl_tr)
        cal = int(float(ec.get("calentamiento_epocas", 1)) * len(dl_tr))
        paso = 0
        for g in opt.param_groups:
            g["lr"] = g["lr_base"] * factor_lr(0, cal, total)

        dl_va = self._cargador(df_va, False) if df_va is not None and len(df_va) else None
        mejor, mejor_estado, mejor_epoca, espera = -1.0, None, 0, 0
        historial = []
        liberadas = liberar(0)
        t0 = time.perf_counter()
        for epoca in range(1, epocas + 1):
            nuevas = liberar(epoca - 1)
            if nuevas != liberadas:
                liberadas = nuevas
                red.liberar(liberadas)
            red.modo_entrenamiento()
            suma, aciertos, n_vistos, normas = 0.0, 0, 0, []
            te = time.perf_counter()
            for x, y in dl_tr:
                x, y = x.to(eq, non_blocking=True), y.to(eq, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=eq.type, dtype=torch.float16, enabled=self.amp):
                    z = red.modelo(x)
                    perdida = criterio(z, y)
                esc.scale(perdida).backward()
                esc.unscale_(opt)
                activos = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
                normas.append(float(torch.nn.utils.clip_grad_norm_(
                    activos, float(ec.get("recorte_gradiente", 1.0)) or float("inf"))))
                esc.step(opt)
                esc.update()
                paso += 1
                for g in opt.param_groups:
                    g["lr"] = g["lr_base"] * factor_lr(paso, cal, total)
                suma += perdida.item() * y.size(0)
                aciertos += (z.argmax(1) == y).sum().item()
                n_vistos += y.size(0)
            seg = time.perf_counter() - te
            fila = {"epoca": epoca, "perdida_tr": suma / max(1, n_vistos), "exactitud_tr": aciertos / max(1, n_vistos),
                    "norma_grad": float(np.nanmean([v for v in normas if math.isfinite(v)] or [np.nan])),
                    "segundos": seg, "img_s": n_vistos / max(seg, 1e-9), "lr_cabeza": opt.param_groups[0]["lr"],
                    "etapas_liberadas": liberadas}
            if not math.isfinite(fila["perdida_tr"]):
                raise SystemExit("[ERROR] Pérdida NaN: baje la tasa o desactive precision_mixta.")
            if dl_va is not None:
                zv, yv = self._logits(dl_va, self.amp)
                if not np.isfinite(zv).all():
                    zv, yv = self._logits(dl_va, False)
                with torch.no_grad():
                    fila["perdida_va"] = float(criterio(torch.from_numpy(zv).to(eq), torch.from_numpy(yv).to(eq)))
                fila["exactitud_va"] = float((zv.argmax(1) == yv).mean())
                fila["f1_macro_va"] = f1_macro(yv, zv.argmax(1), self.n)
                if fila["f1_macro_va"] > mejor + 1e-4:             # parada temprana (cap. 5.3)
                    mejor, mejor_epoca, espera = fila["f1_macro_va"], epoca, 0
                    mejor_estado = {k: v.detach().cpu().clone() for k, v in red.modelo.state_dict().items()}
                else:
                    espera += 1
            historial.append(fila)
            if al_terminar_epoca:
                al_terminar_epoca(fila)
            if dl_va is not None and espera >= int(ec.get("paciencia", 4)):
                break
        if mejor_estado is not None:
            red.modelo.load_state_dict(mejor_estado)             # el MEJOR, no el último
        return {"historial": pd.DataFrame(historial), "mejor_epoca": mejor_epoca or len(historial),
                "duracion_s": time.perf_counter() - t0,
                "entrenables": sum(p.numel() for p in red.modelo.parameters() if p.requires_grad),
                "total_parametros": sum(p.numel() for p in red.modelo.parameters())}


def ajustar_temperatura(logits: np.ndarray, y: np.ndarray) -> float:
    z, t = torch.tensor(logits, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def cierre():
        opt.zero_grad()
        perdida = F.cross_entropy(z / log_t.exp(), t)
        perdida.backward()
        return perdida

    opt.step(cierre)
    T = float(log_t.detach().exp().clamp(0.05, 20.0))
    return T if math.isfinite(T) else 1.0


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
