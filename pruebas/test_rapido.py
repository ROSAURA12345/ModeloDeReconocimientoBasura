"""Pruebas de las piezas que no necesitan PyTorch.

    python pruebas/test_rapido.py        (o:  pytest pruebas/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from ecosegrega import contrastes as C  # noqa: E402
from ecosegrega import pliegues as P  # noqa: E402
from ecosegrega.galeria import galeria  # noqa: E402
from ecosegrega.inventario import raiz_original, submuestrear  # noqa: E402
from ecosegrega.norma import Catalogo  # noqa: E402


def datos_sinteticos(n_grupos=240, semilla=0):
    rng = np.random.default_rng(semilla)
    filas = []
    for g in range(n_grupos):
        clase = rng.choice(["a", "b", "c", "d"], p=[0.4, 0.3, 0.2, 0.1])
        fuente = rng.choice(["roboflow:x", "propias"], p=[0.7, 0.3])
        for k in range(rng.integers(1, 8)):
            filas.append({"ruta": f"{g}_{k}.jpg", "grupo": f"g{g}", "clase": clase, "fuente": fuente})
    return pd.DataFrame(filas)


def test_pliegues():
    df = datos_sinteticos()
    out = P.construir(df, 0.15, 5, ("clase", "fuente"), 1)
    P.verificar(out)
    assert set(out.bloque) == {"prueba", "1", "2", "3", "4", "5"}
    frac = (out.bloque == "prueba").mean()
    assert 0.10 < frac < 0.20, frac
    for i in range(1, 6):
        tr, va = P.pliegue(out, i)
        assert "prueba" not in set(tr.bloque) | set(va.bloque)
        assert not set(tr.grupo) & set(va.grupo)
        assert len(tr) + len(va) == (out.bloque != "prueba").sum()
    assert P.construir(df, 0.15, 5, ("clase", "fuente"), 1)["bloque"].equals(out["bloque"])


def test_submuestra_por_grupo():
    df = datos_sinteticos()
    sub = submuestrear(df, 5, 0)
    assert sub.groupby("clase")["grupo"].nunique().max() <= 5
    completos = df[df.grupo.isin(sub.grupo)].shape[0]
    assert completos == len(sub)        # nunca parte un grupo


def test_contrastes_factorial():
    rng = np.random.default_rng(1)
    filas = []
    medias = {("x", "sin"): 0.80, ("x", "con"): 0.85, ("y", "sin"): 0.80, ("y", "con"): 0.85}
    bloque = rng.normal(0, 0.02, 6)
    for (a, b), mu in medias.items():
        for k in range(6):
            filas.append({"tratamiento": f"{a}+{b}", "a": a, "b": b, "pliegue": k + 1,
                          "y": mu + bloque[k] + rng.normal(0, 0.003)})
    largo = pd.DataFrame(filas)
    t = C.factorial_bloques(largo).set_index("termino")
    assert t.loc["B", "p"] < 0.001          # B tiene efecto
    assert t.loc["A", "p"] > 0.01           # A no
    assert t.loc["error", "gl"] == 15       # (ab-1)(r-1) = 3*5
    ef = C.efectos_2x2(largo)
    assert abs(ef["efecto_B"] - 0.05) < 0.01 and abs(ef["interaccion"]) < 0.01
    tk = C.tukey(largo)
    assert len(tk) == 6
    una = C.anova_una_via(largo)
    assert 0 <= una["eta2"] <= 1
    npar = C.friedman_wilcoxon(largo)
    assert np.isfinite(npar["friedman"][0])
    assert np.allclose(C.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_norma_no_municipal():
    cat = Catalogo.desde_yaml(RAIZ / "config/ntp_900058.yaml")
    assert cat.tacho("plastico", "no_municipal") == "blanco"
    assert cat.tacho("papel_carton", "no_municipal") == "azul"
    assert cat.tacho("metal", "no_municipal") == "amarillo"
    assert cat.mapear("GLASS") == "vidrio"
    assert raiz_original("a_jpg.rf.123.jpg") == "a_jpg"


def test_galeria(tmp=None):
    from PIL import Image
    from ecosegrega.galeria import miniatura
    destino = Path(tmp or RAIZ / "resultados") / "_prueba_galeria.html"
    destino.parent.mkdir(exist_ok=True)
    f = {"archivo": "x.jpg", "mini": miniatura(Image.new("RGB", (300, 200), (10, 120, 200))), "nombre": "Plástico",
         "confianza": 0.93, "seguro": True, "tacho": "blanco", "color": "#FAFAFA", "top3": [("plastico", 0.93)]}
    galeria([f, {**f, "seguro": False}], destino, "no_municipal", 0.6)
    texto = destino.read_text(encoding="utf-8")
    assert "Tacho BLANCO" in texto and "Revisar" in texto
    destino.unlink()


if __name__ == "__main__":
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print(f"OK  {nombre}")
    print("Todas las pruebas pasaron.")
