"""Prueba apartada + K pliegues agrupados y estratificados (caps. 9 y 10).

Un único archivo, pliegues.csv, asigna cada imagen a un bloque:
    "prueba"      -> se aparta UNA vez y no se usa hasta la etapa 6
    "1" ... "K"   -> pliegues de validación cruzada
Reglas que se hacen cumplir con aserciones (cap. 8.6, 10.6):
  * un grupo (objeto / foto original) está en un solo bloque;
  * todos los tratamientos usan los MISMOS pliegues (diseño aparejado, cap. 10.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tabla_de_grupos(df: pd.DataFrame, estratos) -> pd.DataFrame:
    t = df.groupby("grupo").agg(n=("ruta", "size"),
                                **{c: (c, lambda s: s.mode().iloc[0]) for c in estratos}).reset_index()
    t["estrato"] = t[list(estratos)].astype(str).agg("|".join, axis=1)
    return t


def asignar(tabla: pd.DataFrame, proporciones, etiquetas, semilla: int) -> dict:
    """Reparte grupos entre bloques buscando, dentro de cada estrato, la proporción
    de IMÁGENES pedida. Determinista dada la semilla."""
    p = np.asarray(proporciones, dtype=float)
    p = p / p.sum()
    rng = np.random.default_rng(semilla)
    destino = {}
    for _, sub in sorted(tabla.groupby("estrato"), key=lambda kv: kv[0]):
        sub = sub.iloc[rng.permutation(len(sub))].sort_values("n", ascending=False, kind="stable")
        objetivo = p * sub["n"].sum()
        llevado = np.zeros(len(p))
        # Desfase rotatorio por estrato: con estratos chicos, los primeros pliegues no
        # se quedan siempre con todo
        orden_desempate = rng.permutation(len(p))
        for _, g in sub.iterrows():
            deficit = objetivo - llevado
            mejor = np.flatnonzero(np.isclose(deficit, deficit.max()))
            k = min(mejor, key=lambda i: list(orden_desempate).index(i))
            destino[g["grupo"]] = etiquetas[k]
            llevado[k] += g["n"]
    return destino


def construir(df: pd.DataFrame, p_prueba: float, k: int, estratos=("clase", "fuente"),
              semilla: int = 2026) -> pd.DataFrame:
    estratos = tuple(estratos)
    t = tabla_de_grupos(df, estratos)
    # 1) Prueba apartada
    d1 = asignar(t, [1 - p_prueba, p_prueba], ["resto", "prueba"], semilla)
    out = df.copy()
    out["bloque"] = out["grupo"].map(d1)
    # 2) K pliegues sobre el resto
    resto = out[out["bloque"] == "resto"]
    t2 = tabla_de_grupos(resto, estratos)
    d2 = asignar(t2, [1.0] * k, [str(i) for i in range(1, k + 1)], semilla + 1)
    mascara = out["bloque"] == "resto"
    out.loc[mascara, "bloque"] = out.loc[mascara, "grupo"].map(d2)
    verificar(out)
    return out


def verificar(df: pd.DataFrame) -> None:
    """Ningún grupo en dos bloques. Si pasa, se detiene todo (cap. 8.6)."""
    if df["bloque"].isna().any():
        raise AssertionError("Hay imágenes sin bloque asignado")
    por_grupo = df.groupby("grupo")["bloque"].nunique()
    malos = por_grupo[por_grupo > 1]
    if len(malos):
        raise AssertionError(f"FUGA POR GRUPO: {len(malos)} grupos en más de un bloque, p. ej. "
                             f"{list(malos.index[:5])}")


def pliegue(df: pd.DataFrame, i: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entrenamiento, validación) del pliegue i. La prueba nunca aparece aquí."""
    cv = df[df["bloque"] != "prueba"]
    va = cv[cv["bloque"] == str(i)]
    tr = cv[cv["bloque"] != str(i)]
    assert not (set(tr["grupo"]) & set(va["grupo"])), f"Pliegue {i}: fuga por grupo"
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def informe(df: pd.DataFrame, minimo_prueba: int = 30) -> str:
    L = ["=" * 74, "INFORME DE PRUEBA APARTADA Y PLIEGUES", "=" * 74]
    bloques = sorted(df["bloque"].unique(), key=lambda b: (b == "prueba", b))
    L.append("\n[1] Tamaños por bloque")
    for b in bloques:
        d = df[df.bloque == b]
        L.append(f"    {b:>7s}: {len(d):6d} imágenes ({100 * len(d) / len(df):5.1f} %) | "
                 f"{d['grupo'].nunique():5d} grupos")
    L.append("\n[2] Solapamiento de grupos entre bloques: "
             + ("OK, cero" if df.groupby("grupo")["bloque"].nunique().max() == 1 else "FUGA"))
    L.append("\n[3] % de imágenes de cada clase por bloque")
    t = pd.crosstab(df["clase"], df["bloque"], normalize="columns").mul(100)[bloques]
    t["total"] = df["clase"].value_counts(normalize=True).mul(100)
    L.append(t.round(1).to_string())
    L.append("\n[4] Imágenes por fuente y bloque")
    L.append(pd.crosstab(df["fuente"], df["bloque"])[bloques].to_string())
    L.append(f"\n[5] Grupos por clase en PRUEBA (conviene >= {minimo_prueba}, cap. 9.5)")
    gp = df[df.bloque == "prueba"].groupby("clase")["grupo"].nunique()
    for c in sorted(df["clase"].unique()):
        n = int(gp.get(c, 0))
        L.append(f"    {c:16s} {n:5d}" + ("" if n >= minimo_prueba else "   <- pocos: IC ancho"))
    L.append("=" * 74)
    return "\n".join(L)
