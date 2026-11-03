"""Panel local interactivo para EcoSegrega."""
import hashlib
import io
import json
import re
import threading
import time
from pathlib import Path

import av
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from ecosegrega.clasificador import Clasificador
from ecosegrega.rutas import cargar_config, ruta

CONFIG = "config/estudio_aumento.yaml"
MODELO = "resultados_aumento/final/modelo_final.pt"
SALIDAS = "resultados_aumento"


@st.cache_resource(show_spinner="Cargando el modelo entrenado...")
def cargar_clasificador(modelo: str, norma: str, equipo: str,
                        modificacion: int, tamano: int) -> Clasificador:
    return Clasificador(ruta(modelo), equipo, norma)


def imagen_desde_archivo(archivo):
    imagen = Image.open(io.BytesIO(archivo.getvalue()))
    imagen.load()
    return imagen.convert("RGB")


class ProcesadorCamara:
    def __init__(self, clasificador: Clasificador, esquema: str, frecuencia: int):
        self.clasificador = clasificador
        self.esquema = esquema
        self.intervalo = 1 / frecuencia
        self.ultima_inferencia = 0.0
        self.prediccion = None
        self.lock = threading.Lock()

    def __call__(self, frame):
        imagen = Image.fromarray(frame.to_ndarray(format="rgb24"))
        ahora = time.monotonic()
        with self.lock:
            if ahora - self.ultima_inferencia >= self.intervalo:
                self.prediccion = self.clasificador(imagen, self.esquema)
                self.ultima_inferencia = ahora
            prediccion = self.prediccion

        if prediccion:
            dibujo = ImageDraw.Draw(imagen, "RGBA")
            texto = (
                f"{prediccion['nombre']}  {prediccion['confianza']:.0%}"
                f"  |  Tacho {prediccion['tacho'].upper()}"
            )
            if not prediccion["seguro"]:
                texto += "  |  REVISAR"
            caja = dibujo.textbbox((16, 16), texto)
            dibujo.rounded_rectangle(
                (8, 8, caja[2] + 24, caja[3] + 24),
                radius=10,
                fill=prediccion["color"] + "E8",
                outline="#FFFFFF",
                width=2,
            )
            dibujo.text((16, 16), texto, fill="#111111", stroke_width=1, stroke_fill="#FFFFFF")
        else:
            dibujo = ImageDraw.Draw(imagen, "RGBA")
            dibujo.rounded_rectangle((8, 8, 360, 48), radius=10, fill="#111111B8")
            dibujo.text((16, 16), "Enfoca un residuo para clasificarlo", fill="white")
        return av.VideoFrame.from_image(imagen)


def mostrar_tarjeta(resultado):
    st.markdown(
        f'<div style="background:{resultado["color"]};padding:10px;border-radius:8px;'
        f'border:1px solid #888;color:#111"><b>Tacho {resultado["tacho"].upper()}</b></div>',
        unsafe_allow_html=True,
    )


def mostrar_clasificador(modelo_path, cfg, equipo):
    try:
        return cargar_clasificador(
            str(modelo_path), cfg["datos"]["norma"], equipo,
            modelo_path.stat().st_mtime_ns, modelo_path.stat().st_size,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"No se pudo cargar el modelo: {exc}")
        return None


def mostrar_inicio(modelo_path):
    st.header("Panel de clasificación de residuos")
    st.write(
        "Clasifica residuos con el modelo EcoSegrega. Usa fotos o la cámara en vivo, "
        "consulta el tacho recomendado y revisa los resultados de tu estudio."
    )
    if not modelo_path.is_file():
        st.error(f"No se encontró el modelo entrenado: {modelo_path}")
        st.info("Comprueba la ruta del modelo en la barra lateral.")
        return

    informe = ruta(SALIDAS) / "final" / "informe_prueba.md"
    contenido = informe.read_text(encoding="utf-8-sig") if informe.is_file() else ""
    valores = {}
    for metrica in ("F1 macro", "Exactitud", "Exactitud balanceada"):
        coincidencia = re.search(
            rf"^\|\s*{re.escape(metrica)}\s*\|\s*([\d.]+)",
            contenido,
            re.MULTILINE,
        )
        if coincidencia:
            valores[metrica] = float(coincidencia.group(1))

    columnas = st.columns(3)
    muestra = re.search(r"Prueba:\s*([\d,]+)\s+imágenes", contenido)
    for columna, metrica in zip(columnas, ("F1 macro", "Exactitud", "Exactitud balanceada")):
        columna.metric(metrica, f"{valores[metrica]:.1%}" if metrica in valores else "—")
    datos_modelo = st.columns(3)
    datos_modelo[0].metric("Modelo", modelo_path.name)
    datos_modelo[1].metric("Imágenes de prueba", muestra.group(1) if muestra else "No disponible")
    datos_modelo[2].metric("Predicciones en sesión", len(st.session_state.get("historial", [])))

    ganador = ruta(SALIDAS) / "estudio" / "ganador.json"
    if ganador.is_file():
        datos = json.loads(ganador.read_text(encoding="utf-8"))
        st.success(
            f"Tratamiento ganador del estudio: **{datos['tratamiento']}** · "
            f"F1 macro medio en validación cruzada: **{datos['media']:.1%}**."
        )
    if contenido:
        st.caption("Las métricas corresponden a la prueba apartada del estudio; no garantizan el resultado en cada foto.")
        st.download_button(
            "Descargar informe de evaluación",
            contenido.encode("utf-8"),
            file_name="informe_prueba_ecosegrega.md",
            mime="text/markdown",
        )


def mostrar_camara(modelo_path, cfg, equipo, esquema, frecuencia):
    st.header("Cámara en vivo")
    st.write("Apunta la cámara hacia un solo residuo. La clase, la confianza y el tacho aparecerán sobre el video.")
    st.caption(
        f"Analiza hasta {frecuencia} cuadros por segundo para reducir carga en el equipo. "
        "No graba ni guarda el video."
    )
    clf = mostrar_clasificador(modelo_path, cfg, equipo)
    if clf is None:
        return
    procesador = ProcesadorCamara(clf, esquema, frecuencia)
    webrtc_streamer(
        key="ecosegrega-camara",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": []}),
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=procesador,
        async_processing=True,
        video_html_attrs={"autoPlay": True, "controls": False, "muted": True},
    )
    st.info("Pulsa **START** y acepta el permiso de cámara del navegador. La conexión es local.")


def mostrar_fotos(modelo_path, cfg, equipo, esquema):
    st.header("Clasificar fotos")
    archivos = st.file_uploader(
        "Selecciona una o varias fotos",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        help="Las imágenes se procesan en memoria y no se guardan automáticamente.",
    )
    if not archivos:
        st.info("Carga fotos para comenzar.")
        return

    firma = tuple(
        (archivo.name, hashlib.sha256(archivo.getvalue()).hexdigest())
        for archivo in archivos
    )
    clave_resultados = (firma, esquema, str(modelo_path), equipo)
    if st.button("Clasificar fotos", type="primary"):
        clf = mostrar_clasificador(modelo_path, cfg, equipo)
        if clf is None:
            return
        resultados = []
        progreso = st.progress(0, text="Preparando imágenes...")
        for indice, archivo in enumerate(archivos, start=1):
            try:
                imagen = imagen_desde_archivo(archivo)
                prediccion = clf(imagen, esquema)
                resultados.append({
                    "archivo": archivo.name,
                    "imagen": imagen,
                    "historial_id": hashlib.sha256(
                        (archivo.name + firma[indice - 1][1] + prediccion["clase"]).encode("utf-8")
                    ).hexdigest(),
                    **prediccion,
                })
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                st.warning(f"No se pudo procesar {archivo.name}: {exc}")
            progreso.progress(indice / len(archivos), text=f"Procesando {indice} de {len(archivos)}")
        progreso.empty()
        st.session_state["clave_resultados"] = clave_resultados
        st.session_state["resultados"] = resultados
        st.session_state["umbral"] = clf.umbral

    if st.session_state.get("clave_resultados") != clave_resultados:
        st.info("Pulsa **Clasificar fotos** para actualizar los resultados con las opciones actuales.")
        return

    resultados = st.session_state.get("resultados", [])
    if not resultados:
        st.warning("No hay fotos válidas para mostrar.")
        return

    seguros = sum(resultado["seguro"] for resultado in resultados)
    columnas = st.columns(3)
    columnas[0].metric("Fotos clasificadas", len(resultados))
    columnas[1].metric("Para revisar", len(resultados) - seguros)
    columnas[2].metric("Esquema", "Municipal" if esquema == "municipal" else "No municipal")
    st.caption(f"Umbral de confianza del modelo: {st.session_state['umbral']:.0%}")

    filas_csv = []
    columnas_tarjetas = st.columns(3)
    historial = st.session_state.setdefault("historial", [])
    for indice, resultado in enumerate(resultados):
        with columnas_tarjetas[indice % len(columnas_tarjetas)]:
            st.image(resultado["imagen"], width="stretch")
            st.subheader(resultado["nombre"])
            mostrar_tarjeta(resultado)
            st.write(f"Confianza: **{resultado['confianza']:.1%}**")
            st.caption(resultado["archivo"])
            if resultado["seguro"]:
                st.success("Clasificación dentro del umbral")
            else:
                st.warning("Revisar: confianza baja")
            with st.expander("Ver otras clases probables"):
                for clase, confianza in resultado["top3"]:
                    st.write(f"{clase}: {confianza:.1%}")
        filas_csv.append({
            clave: valor for clave, valor in resultado.items()
            if clave not in ("imagen", "top3", "historial_id")
        })
        identificador = resultado["historial_id"]
        if not any(fila["id"] == identificador for fila in historial):
            historial.append({
                "id": identificador,
                "archivo": resultado["archivo"],
                "clase": resultado["clase"],
                "confianza": resultado["confianza"],
                "tacho": resultado["tacho"],
                "seguro": resultado["seguro"],
            })

    csv = pd.DataFrame(filas_csv).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar resultados CSV",
        data=csv,
        file_name="clasificacion_ecosegrega.csv",
        mime="text/csv",
    )


def mostrar_guia(cfg, esquema):
    from ecosegrega.norma import Catalogo

    st.header("Guía de separación y clases")
    catalogo = Catalogo.desde_yaml(cfg["datos"]["norma"])
    st.write(
        f"Esquema seleccionado: **{'municipal' if esquema == 'municipal' else 'no municipal'}**. "
        "La recomendación de separación se basa en el catálogo NTP configurado en el proyecto."
    )
    columnas = st.columns(2)
    for indice, clase in enumerate(catalogo.clases):
        info = catalogo.info[clase]
        tacho = catalogo.tacho(clase, esquema)
        with columnas[indice % len(columnas)]:
            st.markdown(
                f'<div style="border-left:10px solid {catalogo.color_hex(tacho)};'
                'padding:8px 14px;margin:6px 0;background:#f4f5f2;color:#1d1d1b">'
                f'<b>{catalogo.nombre(clase)}</b> · Tacho {tacho.upper()}<br>'
                f'{info.get("ejemplos", "")}</div>',
                unsafe_allow_html=True,
            )
    st.caption(
        "El clasificador asigna una clase a la imagen completa; no detecta ni delimita "
        "objetos individuales. Verifica los residuos peligrosos y los resultados marcados para revisar."
    )


def main():
    st.set_page_config(page_title="EcoSegrega | Panel", page_icon="♻️", layout="wide")
    st.title("EcoSegrega")
    st.caption("Clasificación inteligente de residuos · Panel interactivo local")

    cfg = cargar_config(CONFIG)
    modelo_default = ruta(MODELO)
    modelo = st.sidebar.text_input(
        "Ruta del modelo",
        value=str(modelo_default) if modelo_default.is_file() else MODELO,
        help="Puedes cambiarla si guardaste el modelo en otra carpeta.",
    )
    equipo = st.sidebar.selectbox("Equipo de cómputo", ["auto", "cpu"])
    opciones_esquema = ["no_municipal", "municipal"]
    esquema_default = cfg["decision"]["esquema"]
    esquema = st.sidebar.selectbox(
        "Esquema de tachos",
        opciones_esquema,
        index=opciones_esquema.index(esquema_default),
        format_func=lambda valor: "No municipal (7 colores)" if valor == "no_municipal"
        else "Municipal (4 colores)",
    )
    frecuencia = st.sidebar.select_slider(
        "Análisis de cámara",
        options=[1, 2, 4],
        value=2,
        format_func=lambda valor: f"{valor} cuadro{'s' if valor != 1 else ''}/s",
        help="Reduce la frecuencia si la computadora se vuelve lenta.",
    )
    st.sidebar.caption("La cámara y las imágenes se procesan localmente; no se graban.")
    seccion = st.sidebar.radio(
        "Secciones",
        ["Inicio", "Cámara en vivo", "Clasificar fotos", "Guía de separación"],
    )

    modelo_path = Path(modelo)
    if not modelo_path.is_absolute():
        modelo_path = ruta(modelo_path)
    if not modelo_path.is_file():
        st.error(f"No se encontró el modelo: {modelo_path}")
        st.info("Comprueba la ruta del modelo en la barra lateral.")
        return

    if seccion == "Inicio":
        mostrar_inicio(modelo_path)
        historial = st.session_state.get("historial", [])
        if historial:
            st.subheader("Actividad de esta sesión")
            actividad = pd.DataFrame(historial).drop(columns="id")
            st.dataframe(actividad, hide_index=True, width="stretch")
            if st.button("Limpiar actividad de esta sesión"):
                st.session_state["historial"] = []
                st.rerun()
    elif seccion == "Cámara en vivo":
        mostrar_camara(modelo_path, cfg, equipo, esquema, frecuencia)
    elif seccion == "Clasificar fotos":
        mostrar_fotos(modelo_path, cfg, equipo, esquema)
    else:
        mostrar_guia(cfg, esquema)


if __name__ == "__main__":
    main()
