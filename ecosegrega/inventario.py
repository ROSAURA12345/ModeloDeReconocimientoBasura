"""Inventario de datos (caps. 7 y 8): Roboflow + fotos propias en una sola tabla.

El inventario es UNA tabla con una fila por imagen utilizable y estas columnas:

    ruta, clase, etiqueta_origen, fuente, grupo, split_origen, ancho, alto,
    dhash, sha256

La columna clave es `grupo`: la unidad de generalización (cap. 9). Todas las
imágenes de un mismo grupo irán juntas a train, a val o a test. Se construye así:

* Roboflow: Roboflow nombra las aumentaciones como  IMG_123_jpg.rf.<hash>.jpg.
  Todo lo anterior a ".rf." identifica la foto original, así que las tres o cinco
  copias aumentadas de una foto comparten grupo. En datasets de detección, todos
  los recortes de la misma foto comparten grupo.
* Fotos propias: propias/<clase>/<objeto>/foto.jpg  ->  grupo = el objeto.
  Diez fotos de la misma botella son UNA observación muestreada diez veces.
* Además se buscan casi-duplicados con dHash entre TODAS las fuentes y se
  fusionan sus grupos (la misma foto puede estar en dos datasets de Roboflow).

Nada se borra: lo que no pasa la validación va a rechazados.csv con su motivo.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from .norma import Catalogo, normalizar_etiqueta
from .rutas import relativa, ruta
from .fotos import (HEIC_OK, abrir_rgb, dhash, distancias_hamming_bajas,
                       es_imagen, motivos_rechazo, sha256)

SPLITS = {"train": "train", "valid": "valid", "val": "valid", "validation": "valid", "test": "test"}


def raiz_original(nombre_archivo: str) -> str:
    """'IMG_20_jpg.rf.9a8b7c.jpg' -> 'IMG_20_jpg'.  'foto.jpg' -> 'foto'."""
    nombre = Path(nombre_archivo).name
    if ".rf." in nombre:
        return nombre.split(".rf.")[0]
    return Path(nombre).stem


# -----------------------------------------------------------------------------
# Lectores de los formatos de exportación de Roboflow
# -----------------------------------------------------------------------------
def _split_de(p: Path, base: Path) -> str:
    for parte in p.relative_to(base).parts:
        if parte.lower() in SPLITS:
            return SPLITS[parte.lower()]
    return ""


def detectar_formato(carpeta: Path) -> str:
    if any(carpeta.rglob("data.yaml")):
        return "yolo"
    if any(carpeta.rglob("_annotations.coco.json")):
        return "coco"
    if any(carpeta.rglob("_classes.csv")):
        return "csv"
    return "carpetas"


def _leer_carpetas(carpeta: Path, nombre_ds: str):
    """Clasificación 'Folder Structure': [split/]<clase>/imagen.jpg"""
    for img in sorted(carpeta.rglob("*")):
        if not es_imagen(img):
            continue
        etiqueta = img.parent.name
        if etiqueta.lower() in SPLITS:  # imagen suelta dentro de train/ sin clase
            continue
        yield {"origen": img, "caja": None, "etiqueta_origen": etiqueta,
               "fuente": f"roboflow:{nombre_ds}",
               "grupo": f"rf:{nombre_ds}:{raiz_original(img.name)}",
               "split_origen": _split_de(img, carpeta)}


def _leer_csv(carpeta: Path, nombre_ds: str):
    """Clasificación 'Multi-Label CSV' de Roboflow: _classes.csv con columnas one-hot."""
    for archivo in sorted(carpeta.rglob("_classes.csv")):
        df = pd.read_csv(archivo)
        df.columns = [c.strip() for c in df.columns]
        col_archivo = df.columns[0]
        clases = list(df.columns[1:])
        for _, fila in df.iterrows():
            img = archivo.parent / str(fila[col_archivo]).strip()
            activas = [c for c in clases if int(fila[c]) == 1]
            registro = {"origen": img, "caja": None, "fuente": f"roboflow:{nombre_ds}",
                        "grupo": f"rf:{nombre_ds}:{raiz_original(img.name)}",
                        "split_origen": _split_de(img, carpeta)}
            if len(activas) != 1:
                registro["etiqueta_origen"] = "|".join(activas)
                registro["motivo"] = "multietiqueta_o_sin_etiqueta"
            else:
                registro["etiqueta_origen"] = activas[0]
            yield registro


def _leer_yolo(carpeta: Path, nombre_ds: str):
    """Detección/segmentación en formato YOLO: cada caja se convierte en un recorte."""
    data_yaml = next(carpeta.rglob("data.yaml"))
    with open(data_yaml, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    nombres = meta["names"]
    if isinstance(nombres, dict):
        nombres = [nombres[k] for k in sorted(nombres, key=int)]
    for etiquetas in sorted(carpeta.rglob("labels")):
        if not etiquetas.is_dir():
            continue
        imagenes = etiquetas.parent / "images"
        if not imagenes.is_dir():
            continue
        for img in sorted(imagenes.iterdir()):
            if not es_imagen(img):
                continue
            txt = etiquetas / (img.stem + ".txt")
            if not txt.exists():
                continue
            for k, linea in enumerate(txt.read_text(encoding="utf-8").splitlines()):
                v = linea.split()
                if len(v) < 5:
                    continue
                cls, nums = int(float(v[0])), [float(x) for x in v[1:]]
                if len(nums) == 4:                      # cx cy w h normalizados
                    cx, cy, w, h = nums
                    caja = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                else:                                   # polígono de segmentación
                    xs, ys = nums[0::2], nums[1::2]
                    caja = (min(xs), min(ys), max(xs), max(ys))
                yield {"origen": img, "caja": ("norm", caja), "indice_caja": k,
                       "etiqueta_origen": nombres[cls] if cls < len(nombres) else f"id_{cls}",
                       "fuente": f"roboflow:{nombre_ds}",
                       "grupo": f"rf:{nombre_ds}:{raiz_original(img.name)}",
                       "split_origen": _split_de(img, carpeta)}


def _leer_coco(carpeta: Path, nombre_ds: str):
    for archivo in sorted(carpeta.rglob("_annotations.coco.json")):
        with open(archivo, encoding="utf-8") as f:
            coco = json.load(f)
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        imgs = {i["id"]: i for i in coco.get("images", [])}
        contador = Counter()
        for a in coco.get("annotations", []):
            info = imgs.get(a["image_id"])
            if info is None or "bbox" not in a:
                continue
            x, y, w, h = a["bbox"]
            img = archivo.parent / info["file_name"]
            k = contador[img.name]
            contador[img.name] += 1
            yield {"origen": img, "caja": ("px", (x, y, x + w, y + h)), "indice_caja": k,
                   "etiqueta_origen": cats.get(a["category_id"], str(a["category_id"])),
                   "fuente": f"roboflow:{nombre_ds}",
                   "grupo": f"rf:{nombre_ds}:{raiz_original(img.name)}",
                   "split_origen": _split_de(img, carpeta)}


LECTORES = {"carpetas": _leer_carpetas, "csv": _leer_csv, "yolo": _leer_yolo, "coco": _leer_coco}


def explorar_roboflow(raiz: Path):
    """Recorre datos/crudo/roboflow/<dataset>/ y detecta el formato de cada uno."""
    if not raiz.exists():
        return
    for ds in sorted(p for p in raiz.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        formato = detectar_formato(ds)
        print(f"  [roboflow] {ds.name}: formato '{formato}'")
        yield from LECTORES[formato](ds, ds.name)


def explorar_propias(raiz: Path):
    """propias/<clase>/<objeto>/foto.jpg   (recomendado)
       propias/<clase>/foto.jpg            (aceptado: cada foto es su propio grupo)
    Las carpetas que empiezan con '_' se ignoran (p. ej. _por_etiquetar)."""
    if not raiz.exists():
        return
    sueltas = Counter()
    for dir_clase in sorted(p for p in raiz.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        for img in sorted(dir_clase.rglob("*")):
            if not es_imagen(img):
                continue
            partes = img.relative_to(dir_clase).parts
            if any(p.startswith("_") for p in partes[:-1]):
                continue
            if len(partes) >= 2:
                objeto = partes[0]
            else:
                objeto = Path(partes[0]).stem
                sueltas[dir_clase.name] += 1
            yield {"origen": img, "caja": None, "etiqueta_origen": dir_clase.name,
                   "fuente": "propias", "grupo": f"propia:{dir_clase.name}/{objeto}",
                   "split_origen": ""}
    for clase, n in sueltas.items():
        print(f"  [propias] AVISO: {n} fotos sueltas en '{clase}/' (sin carpeta de objeto). "
              "Si varias son del MISMO objeto, póngalas en una subcarpeta: si no, "
              "la partición no podrá mantenerlas juntas (fuga por grupo, cap. 9).")


# -----------------------------------------------------------------------------
# Procesamiento de cada registro
# -----------------------------------------------------------------------------
def _recortar(im: Image.Image, caja, margen: float) -> Image.Image:
    tipo, (x0, y0, x1, y1) = caja
    W, H = im.size
    if tipo == "norm":
        x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
    mx, my = (x1 - x0) * margen, (y1 - y0) * margen
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(W, int(x1 + mx)), min(H, int(y1 + my))
    return im.crop((x0, y0, x1, y1))


class _Cache:
    """Evita recalcular huellas de imágenes que no cambiaron (ruta, tamaño, mtime)."""

    def __init__(self, archivo: Path):
        self.archivo = archivo
        self.d = {}
        if archivo.exists():
            with open(archivo, encoding="utf-8") as f:
                for fila in csv.DictReader(f):
                    if fila.get("dhash16") is not None:   # caches de versiones viejas se ignoran
                        self.d[fila["clave"]] = fila

    @staticmethod
    def clave(p: Path) -> str:
        s = p.stat()
        return f"{relativa(p)}|{s.st_size}|{int(s.st_mtime)}"

    def guardar(self):
        campos = ["clave", "ancho", "alto", "dhash", "dhash16", "sha256", "motivo"]
        with open(self.archivo, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(self.d.values())


def construir(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dcfg = cfg["datos"]
    catalogo = Catalogo.desde_yaml(dcfg["norma"])
    reglas = {k: dcfg[k] for k in ("lado_min", "desviacion_min", "aspecto_max") if k in dcfg}
    margen = float(dcfg.get("margen_recorte", 0.1))
    dir_recortes = ruta(dcfg["recortes"])
    dir_recortes.mkdir(parents=True, exist_ok=True)
    cache = _Cache(ruta(dcfg["inventario"]).parent / "cache_huellas.csv")

    print("Explorando fuentes...")
    registros = list(explorar_roboflow(ruta(dcfg["roboflow"]))) + list(explorar_propias(ruta(dcfg["propias"])))
    print(f"Registros encontrados: {len(registros)}")

    validos, rechazados = [], []
    desconocidas = Counter()
    for n, r in enumerate(registros, 1):
        if n % 500 == 0:
            print(f"  procesadas {n}/{len(registros)}")
        origen = Path(r["origen"])
        base = {"etiqueta_origen": r["etiqueta_origen"], "fuente": r["fuente"],
                "grupo": r["grupo"], "split_origen": r["split_origen"]}

        def rechazar(motivo, ruta_txt=None):
            rechazados.append({"ruta": ruta_txt or relativa(origen), **base, "motivo": motivo})

        if "motivo" in r:
            rechazar(r["motivo"])
            continue
        clase = catalogo.mapear(r["etiqueta_origen"])
        if Catalogo.es_desconocida(clase):
            desconocidas[normalizar_etiqueta(r["etiqueta_origen"])] += 1
            rechazar(f"etiqueta_sin_equivalencia:{normalizar_etiqueta(r['etiqueta_origen'])}")
            continue
        if clase is None:
            rechazar(f"etiqueta_descartada:{normalizar_etiqueta(r['etiqueta_origen'])}")
            continue
        if not origen.exists():
            rechazar("archivo_no_existe")
            continue
        if origen.suffix.lower() in {".heic", ".heif"} and not HEIC_OK:
            rechazar("heic_sin_soporte(pip install pillow-heif)")
            continue

        # --- imagen final: el archivo original o el recorte de la caja ---
        if r["caja"] is not None:
            destino = dir_recortes / r["fuente"].split(":", 1)[-1] / (r["split_origen"] or "sin_split") / \
                f"{origen.stem}__c{r['indice_caja']}.jpg"
            if not destino.exists():
                try:
                    im = abrir_rgb(origen, orientar=False)  # las cajas se refieren a los píxeles tal cual
                    rec = _recortar(im, r["caja"], margen)
                except Exception as e:  # noqa: BLE001
                    rechazar(f"ilegible:{type(e).__name__}", f"{relativa(origen)}#c{r['indice_caja']}")
                    continue
                if min(rec.size) < reglas.get("lado_min", 48):
                    rechazar(f"recorte_pequeno_{rec.size[0]}x{rec.size[1]}",
                             f"{relativa(origen)}#c{r['indice_caja']}")
                    continue
                destino.parent.mkdir(parents=True, exist_ok=True)
                rec.save(destino, quality=95)
            archivo = destino
        else:
            archivo = origen

        clave = _Cache.clave(archivo)
        info = cache.d.get(clave)
        if info is None:
            try:
                im = abrir_rgb(archivo)
                motivos = motivos_rechazo(im, reglas)
                info = {"clave": clave, "ancho": im.size[0], "alto": im.size[1],
                        "dhash": f"{dhash(im):016x}", "dhash16": f"{dhash(im, 16):064x}",
                        "sha256": sha256(archivo),
                        "motivo": ";".join(motivos)}
            except Exception as e:  # noqa: BLE001
                info = {"clave": clave, "ancho": 0, "alto": 0, "dhash": "", "dhash16": "", "sha256": "",
                        "motivo": f"ilegible:{type(e).__name__}"}
            cache.d[clave] = info
        if info["motivo"]:
            rechazar(info["motivo"], relativa(archivo))
            continue
        validos.append({"ruta": relativa(archivo), "clase": clase, **base,
                        "ancho": int(info["ancho"]), "alto": int(info["alto"]),
                        "dhash": info["dhash"], "dhash16": info["dhash16"], "sha256": info["sha256"]})
    cache.guardar()

    man = pd.DataFrame(validos)
    rech = pd.DataFrame(rechazados, columns=["ruta", "etiqueta_origen", "fuente", "grupo",
                                             "split_origen", "motivo"])
    if man.empty:
        return man, rech, {"desconocidas": dict(desconocidas)}

    man, rech, info_dup = deduplicar(man, rech, int(dcfg.get("hamming_max", 4)))
    info_dup["desconocidas"] = dict(desconocidas.most_common())
    return man, rech, info_dup


# -----------------------------------------------------------------------------
# Duplicados (cap. 8.5.3) y fusión de grupos
# -----------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, elementos):
        self.p = {e: e for e in elementos}

    def raiz(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def unir(self, a, b):
        ra, rb = self.raiz(a), self.raiz(b)
        if ra != rb:
            # Raíz determinista: la menor alfabéticamente
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


def deduplicar(man: pd.DataFrame, rech: pd.DataFrame, hamming_max: int):
    info = {}
    # 1) Duplicados idénticos (mismo contenido byte a byte)
    man = man.reset_index(drop=True)
    clases_por_sha = man.groupby("sha256")["clase"].nunique()
    conflicto = man["sha256"].map(clases_por_sha) > 1
    nuevos_rech = []
    for _, f in man[conflicto].iterrows():
        nuevos_rech.append({**{k: f[k] for k in ["ruta", "etiqueta_origen", "fuente", "grupo", "split_origen"]},
                            "motivo": "duplicado_con_etiquetas_distintas"})
    man = man[~conflicto]
    repetido = man.duplicated("sha256", keep="first")
    primera = man.drop_duplicates("sha256").set_index("sha256")["ruta"]
    for _, f in man[repetido].iterrows():
        nuevos_rech.append({**{k: f[k] for k in ["ruta", "etiqueta_origen", "fuente", "grupo", "split_origen"]},
                            "motivo": f"duplicado_exacto_de:{primera[f['sha256']]}"})
    info["duplicados_exactos"] = int(repetido.sum())
    info["conflictos_exactos"] = int(conflicto.sum())
    man = man[~repetido].reset_index(drop=True)

    # 2) Casi-duplicados: fusionar sus grupos para que nunca queden separados
    man["grupo_original"] = man["grupo"]
    info["pares_casi_duplicados"] = 0
    info["grupos_fusionados"] = 0
    conflictos_casi = []
    if hamming_max >= 0 and len(man) > 1:
        huellas = np.array([int(h, 16) for h in man["dhash"]], dtype=np.uint64)
        candidatos = distancias_hamming_bajas(huellas, hamming_max)
        # Confirmación con la huella fina de 256 bits: con objetos centrados sobre fondo
        # liso, la de 64 bits junta fotos distintas. Se exige la misma distancia relativa.
        finas = [int(h, 16) for h in man["dhash16"]]
        pares = [(i, j) for i, j in candidatos
                 if (finas[i] ^ finas[j]).bit_count() <= 4 * hamming_max]
        info["candidatos_dhash64"] = len(candidatos)
        info["pares_casi_duplicados"] = len(pares)
        uf = _UnionFind(man["grupo"].unique())
        g = man["grupo"].to_numpy()
        c = man["clase"].to_numpy()
        for i, j in pares:
            if g[i] != g[j]:
                uf.unir(g[i], g[j])
            if c[i] != c[j]:
                conflictos_casi.append({"ruta_a": man.at[i, "ruta"], "clase_a": c[i],
                                        "ruta_b": man.at[j, "ruta"], "clase_b": c[j]})
        man["grupo"] = man["grupo"].map(uf.raiz)
        info["grupos_fusionados"] = int(man["grupo_original"].nunique() - man["grupo"].nunique())
        info["mayor_grupo"] = int(man["grupo"].value_counts().max())
    info["conflictos_casi_duplicados"] = conflictos_casi
    rech = pd.concat([rech, pd.DataFrame(nuevos_rech)], ignore_index=True)
    return man, rech, info


def resumen(man: pd.DataFrame, rech: pd.DataFrame, info: dict) -> str:
    lineas = ["=" * 70, "RESUMEN DEL INVENTARIO", "=" * 70]
    if man.empty:
        lineas.append("No hay imágenes válidas. Revise las carpetas de datos/crudo/.")
    else:
        lineas.append(f"Imágenes válidas: {len(man)} | grupos (unidades de generalización): "
                      f"{man['grupo'].nunique()}")
        tabla = pd.crosstab(man["clase"], man["fuente"], margins=True, margins_name="TOTAL")
        lineas += ["", "Imágenes por clase y fuente:", tabla.to_string()]
        grupos = man.groupby("clase")["grupo"].nunique()
        lineas += ["", "Grupos por clase (lo que de verdad cuenta como 'datos', cap. 9):", grupos.to_string()]
    lineas += ["", f"Duplicados exactos apartados: {info.get('duplicados_exactos', 0)}",
               f"Duplicados exactos con etiquetas distintas (apartados): {info.get('conflictos_exactos', 0)}",
               f"Pares de casi-duplicados (dHash): {info.get('pares_casi_duplicados', 0)} -> "
               f"grupos fusionados: {info.get('grupos_fusionados', 0)}",
               f"Casi-duplicados con clases distintas (revíselos a mano en "
               f"conflictos_casi_duplicados.csv): {len(info.get('conflictos_casi_duplicados', []))}",
               f"Grupo más grande: {info.get('mayor_grupo', 0)} imágenes"]
    if not man.empty and info.get("mayor_grupo", 0) > max(50, 0.02 * len(man)):
        lineas.append("  AVISO: un grupo concentra demasiadas imágenes. Si no son de verdad la misma foto, "
                      "baje datos.hamming_max en config/estudio.yaml (p. ej. a 2).")
    if len(rech):
        motivos = rech["motivo"].str.split(":").str[0].value_counts()
        lineas += ["", f"Registros apartados: {len(rech)} (detalle en apartados.csv)", motivos.to_string()]
    if info.get("desconocidas"):
        lineas += ["", "ETIQUETAS SIN EQUIVALENCIA (agréguelas en config/ntp_900058.yaml -> equivalencias):"]
        lineas += [f"   {k}: {v} imágenes" for k, v in info["desconocidas"].items()]
    lineas.append("=" * 70)
    return "\n".join(lineas)


def guardar(man: pd.DataFrame, rech: pd.DataFrame, info: dict, cfg: dict) -> dict:
    dcfg = cfg["datos"]
    archivo = ruta(dcfg["inventario"])
    archivo.parent.mkdir(parents=True, exist_ok=True)
    columnas = ["ruta", "clase", "etiqueta_origen", "fuente", "grupo", "grupo_original",
                "split_origen", "ancho", "alto", "dhash", "dhash16", "sha256"]
    if not man.empty:
        man = man.sort_values("ruta")[columnas]
    man.to_csv(archivo, index=False, encoding="utf-8")
    rech.to_csv(ruta(dcfg["apartados"]), index=False, encoding="utf-8")
    pd.DataFrame(info.get("conflictos_casi_duplicados", []),
                 columns=["ruta_a", "clase_a", "ruta_b", "clase_b"]).to_csv(
        archivo.parent / "conflictos_casi_duplicados.csv", index=False, encoding="utf-8")
    from .rutas import huella_archivo
    meta = {"huella_inventario": huella_archivo(archivo), "n_imagenes": int(len(man)),
            "n_grupos": int(man["grupo"].nunique()) if len(man) else 0,
            "n_rechazados": int(len(rech)),
            **{k: v for k, v in info.items() if k != "conflictos_casi_duplicados"}}
    with open(archivo.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def grupos_por_clase(man: pd.DataFrame) -> dict:
    d = defaultdict(int)
    for c, n in man.groupby("clase")["grupo"].nunique().items():
        d[c] = int(n)
    return dict(d)


def submuestrear(man: pd.DataFrame, grupos_por_clase: int | None, semilla: int) -> pd.DataFrame:
    """Ensayos rápidos (p. ej. en CPU): conserva a lo sumo N GRUPOS por clase.
    Se submuestrea por grupo, nunca por imagen, para no romper la unidad de generalización."""
    if not grupos_por_clase:
        return man
    rng = np.random.default_rng(semilla)
    elegidos = []
    for _, sub in man.groupby("clase"):
        g = sub["grupo"].unique()
        elegidos += list(rng.permutation(g)[:int(grupos_por_clase)])
    return man[man["grupo"].isin(set(elegidos))].reset_index(drop=True)
