"""Continuacion local supervisada del estudio; no modifica datos ni configuraciones."""
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_aumento'
STATUS = OUT / 'continuacion_estado.json'
CFG = 'config/estudio_aumento.yaml'
EXPECTED = {(a + '+' + b, str(k)) for a in ('mobilenet_v3_large', 'efficientnet_b0')
            for b in ('basico', 'fuerte') for k in range(1, 6)}

def state(stage, **extra):
    data = dict(etapa=stage, fecha=datetime.now().isoformat(timespec='seconds'), **extra)
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(STATUS)
    print(data, flush=True)

def rows():
    with (OUT / 'estudio/resultados_pliegues.csv').open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def validate():
    rs = rows()
    keys = [(r['tratamiento'], r['pliegue']) for r in rs]
    if len(keys) != len(set(keys)) or not set(keys) <= EXPECTED:
        raise RuntimeError('Resultados duplicados o tratamientos inesperados.')
    digest = hashlib.sha256((ROOT / 'datos/derivados/pliegues.csv').read_bytes()).hexdigest()
    for r in rs:
        if r['huella_pliegues'] != digest:
            raise RuntimeError('Los resultados no corresponden a los pliegues actuales.')
        stem = r['tratamiento'] + '__p' + r['pliegue']
        for rel in ('oof/' + stem + '.npz', 'historiales/' + stem + '.csv'):
            p = OUT / 'estudio' / rel
            if not p.is_file() or not p.stat().st_size:
                raise RuntimeError('Falta un archivo de resultados: ' + str(p))
    return set(keys)

def run(stage, script, *args):
    state(stage)
    log = OUT / ('continuacion_' + stage + '.log')
    with log.open('a', encoding='utf-8') as f:
        proc = subprocess.run([sys.executable, '-u', script, '--config', CFG, *args],
                              cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f'{stage} fallo (codigo {proc.returncode}); ver {log.name}')

def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    # Mantener despierto el sistema durante el trabajo; permitir apagar la pantalla.
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState(0x80000001)
    handles = []
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    try:
        # Esperar a los procesos ya existentes, sin iniciar un segundo entrenamiento.
        for pid in map(int, sys.argv[1:]):
            h = kernel.OpenProcess(0x00100000, False, pid)
            if h:
                handles.append(h)
        state('esperando_entrenamiento_actual', procesos=sys.argv[1:])
        while any(kernel.WaitForSingleObject(h, 0) == 258 for h in handles):
            time.sleep(20)
        if validate() != EXPECTED:
            run('e4_pendientes', 'etapas/e4_estudio_factorial.py', '--equipo', 'cpu')
        if validate() != EXPECTED:
            raise RuntimeError('No estan completos los 20 entrenamientos.')
        run('e5_analisis', 'etapas/e5_analisis_estadistico.py')
        final = OUT / 'final'
        if not (final / 'modelo_final.pt').exists():
            consulted = final / 'consultas_prueba.log'
            if consulted.exists() and consulted.read_text(encoding='utf-8').strip():
                raise RuntimeError('La prueba ya fue consultada pero falta el modelo. Revisar antes de repetir.')
            run('e6_modelo_final', 'etapas/e6_modelo_final.py', '--equipo', 'cpu')
        for name in ('modelo_final.pt', 'informe_prueba.md', 'predicciones_prueba.csv'):
            if not (final / name).is_file():
                raise RuntimeError('Falta salida final: ' + name)
        run('e8_explicaciones', 'etapas/e8_mapas_gradcampp.py', '--equipo', 'cpu')
        # Galeria de demostracion con ejemplos YA evaluados, no una prueba nueva.
        with (final / 'predicciones_prueba.csv').open(encoding='utf-8', newline='') as f:
            predictions = list(csv.DictReader(f))
        examples = {}
        for r in predictions:
            examples.setdefault(r['real'], r['ruta'])
        run('e7_galeria', 'etapas/e7_clasificar.py', '--equipo', 'cpu',
            '--modelo', str(final / 'modelo_final.pt'), *examples.values())
        (OUT / 'LEEME_RESULTADOS.md').write_text(
            '# EcoSegrega: resultados generados\n\n'
            '- Comparacion: estudio/informe_estadistico.md\n'
            '- Evaluacion final: final/informe_prueba.md\n'
            '- Modelo: final/modelo_final.pt\n'
            '- Galeria: clasificacion/galeria.html\n'
            '- Explicaciones: final/gradcampp_aciertos_fallos.png\n\n'
            'La galeria muestra un ejemplo por categoria tomado de la prueba ya evaluada. '
            'No representa fotografias nuevas ni una validacion adicional.\n\n'
            'Limitaciones pendientes: revision de 41 pares similares entre clases y del grupo '
            'de 97 imagenes; datos exclusivamente de Roboflow. Los resultados son preliminares '
            'hasta completar la revision de datos.\n', encoding='utf-8')
        state('completo', entrenamientos=20, modelo=str(final / 'modelo_final.pt'))
    except Exception as exc:
        traceback.print_exc()
        state('error_requiere_revision', error=str(exc))
        raise
    finally:
        for h in handles:
            kernel.CloseHandle(h)
        kernel.SetThreadExecutionState(0x80000000)

if __name__ == '__main__':
    main()
