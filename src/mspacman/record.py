"""Graba partidas del agente para el demo web.

Corre N episodios, guarda el video a 60 fps y una traza JSON con lo que el agente
"pensaba" en cada decisión (los 9 valores Q), y se queda con los mejores.

El demo se sirve estático justamente por esto: un backend de inferencia en capa
gratuita se duerme, y un demo que tarda dos minutos en despertar es peor que no
tener demo.

    python -m mspacman.record --episodios 12 --guardar 3 --salida web/public/replay
"""
from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch

from .envs import ACCIONES, make_vec_env
from .model import cargar

warnings.filterwarnings("ignore")
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 666, 777)


#: H.264/mp4 es el que reproduce todo; VP9/WebM va primero como respaldo abierto
#: (algunos Chromium de código abierto se compilan sin códecs propietarios).
CODECS = {
    ".webm": ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-row-mt", "1"],
    ".mp4": ["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-movflags", "+faststart"],
}


def codificar_video(frames, ruta_sin_ext: Path, fps: int = 60, escala: int = 3) -> list[Path]:
    """Frames RGB crudos -> un archivo por códec.

    El escalado usa vecino más cercano a propósito: es pixel art y la interpolación
    suave lo convierte en una mancha.
    """
    alto, ancho, _ = frames[0].shape
    crudo = b"".join(np.asarray(f, dtype=np.uint8).tobytes() for f in frames)
    salidas = []
    for ext, opciones in CODECS.items():
        destino = ruta_sin_ext.with_suffix(ext)
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{ancho}x{alto}", "-r", str(fps), "-i", "-",
             *opciones, "-pix_fmt", "yuv420p",
             "-vf", f"scale=iw*{escala}:ih*{escala}:flags=neighbor", str(destino)],
            stdin=subprocess.PIPE,
        )
        proc.communicate(crudo)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg falló al codificar {destino.name}")
        salidas.append(destino)
    return salidas


def grabar_episodio(modelo, seed: int, limite: int = 4000) -> dict:
    """Juega un episodio completo capturando frames y valores Q por decisión."""
    venv = make_vec_env(seed=seed, render=True, record_frames=True, monitor=False)
    tap = venv.venv.envs[0].env          # el FrameTap, bajo AtariPreprocessing
    ale = venv.venv.envs[0].unwrapped.ale

    obs = venv.reset()
    tap.frames.clear()
    traza, score = [], 0.0

    while True:
        with torch.no_grad():
            q = modelo.q_net(modelo.policy.obs_to_tensor(obs)[0]).cpu().numpy()[0]
        accion = int(np.argmax(q))
        frame_inicial = len(tap.frames)

        obs, recompensa, done, _ = venv.step(np.array([accion]))
        score += float(recompensa[0])

        traza.append({
            "f0": frame_inicial, "f1": len(tap.frames), "a": accion,
            "q": [round(float(x), 2) for x in q],
            "r": float(recompensa[0]), "score": score, "vidas": int(ale.lives()),
        })
        if done[0] or len(traza) >= limite:
            break

    frames = list(tap.frames)
    venv.close()
    return {"seed": seed, "score": score, "traza": traza, "frames": frames}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodios", type=int, default=12)
    p.add_argument("--guardar", type=int, default=3)
    p.add_argument("--salida", type=Path, default=Path("web/public/replay"))
    args = p.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    modelo = cargar()
    resultados = []
    for i, seed in enumerate(SEEDS[: args.episodios], 1):
        r = grabar_episodio(modelo, seed)
        resultados.append(r)
        print(f"ep {i:2d} (seed {seed:3d}): {r['score']:7.0f} pts | "
              f"{len(r['traza']):4d} decisiones | {len(r['frames'])/60:5.1f}s")

    puntajes = np.array([r["score"] for r in resultados])
    print(f"\nmedia {puntajes.mean():.1f} ± {puntajes.std():.1f} en {len(puntajes)} episodios")

    resultados.sort(key=lambda r: -r["score"])
    manifiesto = {
        "acciones": list(ACCIONES), "fps": 60,
        "resumen": {"episodios": len(puntajes), "media": round(float(puntajes.mean()), 1),
                    "desviacion": round(float(puntajes.std()), 1),
                    "mejor": float(puntajes.max()), "peor": float(puntajes.min())},
        "episodios": [],
    }
    for i, r in enumerate(resultados[: args.guardar], 1):
        salidas = codificar_video(r["frames"], args.salida / f"ep{i}")
        (args.salida / f"ep{i}.json").write_text(
            json.dumps(r["traza"], separators=(",", ":")), encoding="utf-8")
        manifiesto["episodios"].append({
            "id": i, "videos": [s.name for s in salidas], "traza": f"ep{i}.json",
            "score": r["score"], "decisiones": len(r["traza"]),
            "frames": len(r["frames"]), "seed": r["seed"],
        })
        print("  -> " + ", ".join(f"{s.name} ({s.stat().st_size/1e6:.1f} MB)" for s in salidas))

    (args.salida / "manifiesto.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
