"""Carga del agente entrenado."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from stable_baselines3 import DQN

PESOS = "ddqn_mspacman_500k.zip"

#: Los pesos viven en GitHub Releases, no en Drive: es versionado, tiene URL estable
#: y no depende de una cuenta personal ni de límites de descarga.
URL_PESOS = os.environ.get(
    "MSPACMAN_WEIGHTS_URL",
    "https://github.com/luisfernandomtzg/mspacman-rl/releases/download/v1.0.0/" + PESOS,
)


def ruta_pesos(destino: str | Path = PESOS) -> Path:
    """Devuelve la ruta local de los pesos, descargándolos si hace falta."""
    destino = Path(destino)
    if destino.exists():
        return destino
    print(f"Descargando pesos desde {URL_PESOS} ...")
    try:
        urllib.request.urlretrieve(URL_PESOS, destino)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"No se pudieron descargar los pesos desde {URL_PESOS}.\n"
            f"Descárgalos a mano y colócalos en '{destino}', o exporta "
            f"MSPACMAN_WEIGHTS_URL con otra ubicación.\nCausa: {e}"
        ) from e
    return destino


def cargar(destino: str | Path = PESOS, device: str = "cpu") -> DQN:
    """Carga el agente DDQN. En CPU por defecto: la inferencia no necesita GPU."""
    return DQN.load(ruta_pesos(destino), device=device)
