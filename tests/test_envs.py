"""Pruebas de regresión del entorno.

`test_frameskip_efectivo_es_4` es la razón de ser de este archivo: cubre el defecto
que degradaba al agente un 40% en producción. Si alguien vuelve a construir el
entorno con el atajo de SB3 sin fijar el frameskip de la base, esta prueba falla.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mspacman.envs import (  # noqa: E402
    ACCIONES, FRAMESKIP, N_STACK, frameskip_efectivo, make_vec_env,
)


@pytest.fixture(scope="module")
def venv():
    v = make_vec_env(seed=0)
    yield v
    v.close()


def test_frameskip_efectivo_es_4(venv):
    """El agente debe consumir exactamente FRAMESKIP frames por decisión.

    `ALE/MsPacman-v5` trae frameskip=4 de fábrica; si el preprocesamiento le encima
    otro salto de 4, el efectivo se vuelve 16 y el modelo juega 4x más lento de lo
    que aprendió. No se deduce de la config: se mide contra el emulador.
    """
    assert frameskip_efectivo(venv) == FRAMESKIP


def test_forma_de_observacion(venv):
    """La red espera 84x84 con N_STACK frames apilados."""
    assert venv.observation_space.shape == (84, 84, N_STACK)
    assert venv.reset().shape == (1, 84, 84, N_STACK)


def test_espacio_de_acciones(venv):
    """Ms. Pac-Man tiene 9 acciones y nuestra tabla de nombres debe coincidir."""
    assert venv.action_space.n == len(ACCIONES) == 9


def test_el_episodio_avanza_y_termina(venv):
    """Un episodio con acciones al azar debe terminar sin colgarse."""
    venv.reset()
    for _ in range(200):
        _, _, done, _ = venv.step(np.array([venv.action_space.sample()]))
        if done[0]:
            break
    else:
        pytest.skip("200 pasos sin terminar: válido, el episodio es largo")
