"""Etapa 0 — ¿Está listo el equipo? Diagnóstico del entorno (cap. 1.6).

Comprueba, en este orden: sistema y núcleos; Python y entorno virtual; versión y
VARIANTE de PyTorch (+cpu / +cu126); controlador NVIDIA y lo que PyTorch ve;
comparación medida CPU vs GPU; reproducibilidad. Termina con un veredicto.

    python etapas/e0_entorno.py            # informe completo
    python etapas/e0_entorno.py --rapido   # sin las mediciones
Guarde la salida: es lo que se adjunta al pedir ayuda y al documentar la tesis.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time

import _base  # noqa: F401

LINEA = "=" * 72


def seccion(t):
    print(f"\n{LINEA}\n{t}\n{LINEA}")


def medir_matmul(torch, disp, n=2048, rep=12):
    a = torch.randn(n, n, device=disp)
    b = torch.randn(n, n, device=disp)
    for _ in range(3):
        a @ b
    if disp.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        a @ b
    if disp.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / rep
    return dt * 1000, 2 * n ** 3 / dt / 1e9


def medir_modelo(torch, disp, amp=False, lote=32, pasos=6):
    """Imágenes/s de entrenamiento real (adelante + atrás) con mobilenet_v3_large a 224 px."""
    from torchvision import models
    m = models.mobilenet_v3_large(weights=None, num_classes=7).to(disp)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = torch.randn(lote, 3, 224, 224, device=disp)
    y = torch.randint(0, 7, (lote,), device=disp)
    esc = torch.amp.GradScaler("cuda") if amp else None
    m.train()

    def paso():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=disp.type, dtype=torch.float16, enabled=amp):
            perdida = torch.nn.functional.cross_entropy(m(x), y)
        if esc:
            esc.scale(perdida).backward()
            esc.step(opt)
            esc.update()
        else:
            perdida.backward()
            opt.step()

    paso()
    if disp.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(pasos):
        paso()
    if disp.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30 if disp.type == "cuda" else 0
    return lote * pasos / dt, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    problemas, avisos = [], []

    seccion("1. Sistema")
    print(f"  {platform.system()} {platform.release()} | {platform.machine()} | núcleos lógicos: {os.cpu_count()}")

    seccion("2. Python y entorno virtual")
    en_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"  Python {platform.python_version()} en {sys.executable}")
    print(f"  Entorno virtual activo: {'SÍ' if en_venv else 'NO'}")
    if sys.version_info < (3, 10):
        problemas.append("Python < 3.10. Instale Python 3.10–3.12 (cap. 1.2).")
    if not en_venv:
        problemas.append("No está en un entorno virtual. Ejecute:  virtualenv venv  y luego  "
                         "venv\\Scripts\\activate  (Windows)  /  source venv/bin/activate (Linux).")

    seccion("3. PyTorch")
    try:
        import torch
        import torchvision
    except ImportError:
        print("  PyTorch NO está instalado en este Python.")
        problemas.append("Instale PyTorch con la orden del LEEME (con --index-url, nunca 'pip install torch' a secas).")
        return veredicto(problemas, avisos)
    v = torch.__version__
    variante = "CPU (sin GPU)" if "+cpu" in v else (f"CUDA {torch.version.cuda}" if torch.version.cuda else "desconocida")
    print(f"  torch {v} | torchvision {torchvision.__version__} | variante: {variante}")

    seccion("4. Tarjeta NVIDIA")
    smi = shutil.which("nvidia-smi")
    cuda_driver = None
    if smi:
        try:
            salida = subprocess.run([smi], capture_output=True, text=True, timeout=20).stdout
            m = re.search(r"Driver Version:\s*([\d.]+).*?CUDA Version:\s*([\d.]+)", salida, re.S)
            if m:
                cuda_driver = m.group(2)
                print(f"  Controlador {m.group(1)} | CUDA máxima que soporta el controlador: {m.group(2)}")
                print("  (ese número es el MÁXIMO, no lo que hay instalado; cap. 1.4.2)")
        except Exception as e:  # noqa: BLE001
            print(f"  nvidia-smi falló: {e}")
    else:
        print("  nvidia-smi no encontrado (sin tarjeta NVIDIA o sin controlador).")
    disponible = torch.cuda.is_available()
    print(f"  torch.cuda.is_available(): {disponible}")
    if disponible:
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name} | {p.total_memory / 2**30:.1f} GB | capacidad {p.major}.{p.minor}")
        print(f"  cuDNN: {torch.backends.cudnn.version()}")
        if p.total_memory / 2**30 < 6:
            avisos.append(f"La GPU tiene {p.total_memory / 2**30:.1f} GB: lote 32 con mobilenet/efficientnet va bien; "
                          "si falta memoria baje a 16. Cierre el navegador al entrenar.")
        if p.major < 7:
            avisos.append("GPU anterior a Turing: la precisión mixta casi no acelera.")
        elif "GTX 16" in p.name:
            avisos.append("GTX 16xx (Turing sin núcleos tensoriales): la precisión mixta ahorra "
                          "memoria pero acelera poco. Mídalo abajo antes de darlo por hecho (cap. 4.7).")
    elif smi:
        if "+cpu" in v:
            problemas.append("Tiene tarjeta NVIDIA pero instaló la rueda +cpu de PyTorch (cap. 1.7.1).\n"
                             "      pip uninstall -y torch torchvision\n"
                             "      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
        else:
            problemas.append(f"La rueda es CUDA {torch.version.cuda} pero el controlador solo soporta "
                             f"CUDA {cuda_driver}. Reinstale con cu126 (o una versión <= {cuda_driver}) "
                             "o actualice el controlador NVIDIA.")

    if not args.rapido:
        seccion("5. Comparación medida CPU vs GPU")
        cpu = torch.device("cpu")
        ms_c, gf_c = medir_matmul(torch, cpu)
        print(f"  Matmul 2048x2048  CPU : {ms_c:8.1f} ms | {gf_c:8.0f} GFLOP/s")
        if disponible:
            gpu = torch.device("cuda")
            ms_g, gf_g = medir_matmul(torch, gpu)
            print(f"  Matmul 2048x2048  CUDA: {ms_g:8.1f} ms | {gf_g:8.0f} GFLOP/s   -> {ms_c / ms_g:.1f}x")
            print("  (es el TECHO: operación densa sin carga de imágenes; cap. 1.5.1)")
        print("\n  Entrenamiento real mobilenet_v3_large, lote 32, 224 px (adelante + atrás):")
        ips_c, _ = medir_modelo(torch, cpu, pasos=2)
        print(f"    CPU          : {ips_c:7.1f} imágenes/s")
        if disponible:
            ips_g, mem_g = medir_modelo(torch, gpu, amp=False)
            print(f"    CUDA fp32    : {ips_g:7.1f} imágenes/s | memoria pico {mem_g:.2f} GB | {ips_g / ips_c:.1f}x")
            try:
                ips_a, mem_a = medir_modelo(torch, gpu, amp=True)
                print(f"    CUDA mixta   : {ips_a:7.1f} imágenes/s | memoria pico {mem_a:.2f} GB")
                if ips_a < ips_g * 1.05:
                    avisos.append("La precisión mixta no acelera en esta GPU; puede dejarla en true por la "
                                  "memoria o ponerla en false en config/estudio.yaml.")
            except Exception as e:  # noqa: BLE001
                print(f"    CUDA mixta   : falló ({e}); ponga precision_mixta: false")
        n_img, entren = 4000, 4 * 5 * 8
        print(f"\n  Estimación del estudio factorial (4 tratamientos x 5 pliegues x ~8 épocas, {n_img} img):")
        print(f"    CPU ~{n_img * entren / ips_c / 3600:.1f} h"
              + (f" | GPU ~{n_img * entren / ips_g / 3600:.1f} h" if disponible else ""))

    seccion("6. Reproducibilidad")
    print(f"  cudnn.deterministic = {torch.backends.cudnn.deterministic} | "
          f"cudnn.benchmark = {torch.backends.cudnn.benchmark}")
    print("  (las etapas los fijan según 'determinista' de config/estudio.yaml; cap. 4.8)")

    seccion("7. Dependencias del proyecto")
    for mod in ["numpy", "pandas", "PIL", "yaml", "sklearn", "scipy", "matplotlib"]:
        try:
            m = __import__(mod)
            print(f"  {mod:12s} {getattr(m, '__version__', 'ok')}")
        except ImportError:
            problemas.append(f"Falta '{mod}': pip install -r requisitos.txt")
    for mod, para in [("roboflow", "e1_roboflow.py"),
                      ("pillow_heif", "fotos .heic de iPhone"), ("mlflow", "registro opcional")]:
        try:
            __import__(mod)
            print(f"  {mod:12s} ok (opcional, para {para})")
        except ImportError:
            print(f"  {mod:12s} no instalado (opcional, para {para})")
    veredicto(problemas, avisos, disponible)


def veredicto(problemas, avisos, gpu=False):
    seccion("VEREDICTO")
    if problemas:
        print("  HAY PROBLEMAS QUE CORREGIR:")
        for p in problemas:
            print(f"   - {p}")
    else:
        print(f"  Entorno correcto. Entrenará en {'GPU' if gpu else 'CPU'} con --equipo auto.")
    for a in avisos:
        print(f"   * {a}")
    print(LINEA)


if __name__ == "__main__":
    main()
