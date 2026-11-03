"""CPU o GPU, y todas las semillas (caps. 1.8 y 4.8).

Se decide UNA vez por ejecución. Si se pide la GPU y PyTorch no la ve, el
programa se detiene con instrucciones: nunca entrena en CPU "a escondidas".
"""
from __future__ import annotations

import os
import random
import shutil

import numpy as np
import torch

INSTALAR_CU126 = ("pip uninstall -y torch torchvision\n"
                  "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")


def elegir_equipo(pedido: str | None = "auto") -> torch.device:
    pedido = (pedido or "auto").strip().lower()
    if pedido in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "\n[ERROR] Se pidió la GPU pero PyTorch no la ve.\n"
                f"  PyTorch instalado: {torch.__version__}  ('+cpu' = rueda sin soporte de GPU)\n"
                f"  nvidia-smi: {'presente' if shutil.which('nvidia-smi') else 'no encontrado'}\n"
                f"  Corrección (cap. 1.7.1), con el entorno activado:\n    {INSTALAR_CU126}\n")
        return torch.device("cuda")
    if pedido == "cpu":
        return torch.device("cpu")
    if pedido != "auto":
        raise SystemExit(f"[ERROR] equipo '{pedido}' no reconocido: use auto, gpu o cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if shutil.which("nvidia-smi"):
        print(f"[AVISO] Hay tarjeta NVIDIA pero PyTorch ({torch.__version__}) no la ve: se usará CPU.")
    return torch.device("cpu")


def describir(d: torch.device) -> str:
    if d.type == "cuda":
        p = torch.cuda.get_device_properties(d)
        return f"GPU {p.name} ({p.total_memory / 2**30:.1f} GB) · torch {torch.__version__} · CUDA {torch.version.cuda}"
    return f"CPU ({torch.get_num_threads()} hilos) · torch {torch.__version__}"


def sembrar(semilla: int, determinista: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    if determinista:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def semilla_worker(_id: int) -> None:
    """A nivel de módulo para que Windows pueda lanzarla en los procesos del DataLoader."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def generador(semilla: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(semilla)
    return g
