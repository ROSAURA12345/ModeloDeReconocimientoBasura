"""Galería HTML autocontenida con el tacho de cada foto (sin torch)."""
import base64
import html
import io
from datetime import datetime


def miniatura(im, lado=220) -> str:
    im = im.copy()
    im.thumbnail((lado, lado))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=80)
    return base64.b64encode(b.getvalue()).decode()


def galeria(filas, destino, esquema, umbral):
    tarjetas = []
    for f in filas:
        texto_color = "#111" if f["tacho"] in ("blanco", "amarillo") else "#fff"
        estado = ("" if f["seguro"] else '<div class="revisar">Revisar: confianza baja</div>')
        top = " · ".join(f"{html.escape(c)} {100 * v:.0f}%" for c, v in f["top3"])
        tarjetas.append(
            f'<figure><img src="data:image/jpeg;base64,{f["mini"]}" alt="">'
            f'<figcaption><b>{html.escape(f["nombre"])}</b> {100 * f["confianza"]:.0f}%'
            f'<div class="tacho" style="background:{f["color"]};color:{texto_color}">Tacho {html.escape(f["tacho"].upper())}</div>'
            f'{estado}<small>{html.escape(f["archivo"])}<br>{top}</small></figcaption></figure>')
    pagina = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EcoSegrega · clasificación</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#f4f5f2;color:#1d1d1b}}
h1{{font-size:1.4rem}} .rejilla{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}}
figure{{margin:0;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px #0002}}
img{{width:100%;height:200px;object-fit:cover;display:block}} figcaption{{padding:10px;font-size:.92rem}}
.tacho{{margin:6px 0;padding:4px 8px;border-radius:6px;font-weight:600;border:1px solid #0003;display:inline-block}}
.revisar{{color:#b3261e;font-weight:600}} small{{color:#555;display:block;margin-top:4px;word-break:break-all}}</style></head>
<body><h1>EcoSegrega — {len(filas)} fotos clasificadas</h1>
<p>Esquema {html.escape(esquema)} · NTP 900.058:2019 · umbral de confianza {umbral:.2f} · {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="rejilla">{''.join(tarjetas)}</div></body></html>"""
    destino.write_text(pagina, encoding="utf-8")
