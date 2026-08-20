"""Construcción de entornos de Ms. Pac-Man.

Este módulo existe por una razón concreta: en la primera versión de este proyecto
el agente se entrenaba con un entorno y se servía con otro, y la diferencia costaba
un 40% del rendimiento. El detalle está documentado en `docs/frameskip.md`.

Regla de la casa: el frameskip efectivo se declara explícitamente y se verifica en
tests. Nunca se hereda del identificador del entorno.
"""
from __future__ import annotations

import gymnasium as gym
import ale_py
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack

gym.register_envs(ale_py)

ENV_ID = "ALE/MsPacman-v5"

#: Frames del emulador que consume cada decisión del agente.
#: El modelo publicado fue entrenado con este valor; servirlo con otro lo degrada.
FRAMESKIP = 4

#: Cuántas observaciones consecutivas ve la red (le da noción de movimiento).
N_STACK = 4

ACCIONES = ("NOOP", "UP", "RIGHT", "LEFT", "DOWN",
            "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT")


def make_raw_env(*, render: bool = False, record_frames: bool = False):
    """Entorno base con `frameskip=1`.

    `ALE/MsPacman-v5` trae `frameskip=4` de fábrica. Lo forzamos a 1 para que el
    único salto de frames sea el que aplica `AtariPreprocessing` más abajo. Si no
    se hace, los dos saltos se multiplican y el agente termina decidiendo cada 16
    frames en vez de cada 4.
    """
    env = gym.make(ENV_ID, frameskip=1, render_mode="rgb_array" if render else None)
    if record_frames:
        env = FrameTap(env)
    return env


class FrameTap(gym.Wrapper):
    """Guarda cada frame del emulador, incluidos los que el frameskip descarta.

    Se coloca *debajo* de `AtariPreprocessing` para poder reconstruir video fluido
    a 60 fps aunque el agente solo decida 15 veces por segundo.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.frames: list = []

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.frames.append(self.env.render())
        return out

    def step(self, action):
        out = self.env.step(action)
        self.frames.append(self.env.render())
        return out


def make_env(*, terminal_on_life_loss: bool = False, monitor: bool = True,
             render: bool = False, record_frames: bool = False):
    """Un entorno de Ms. Pac-Man listo para el modelo (sin vectorizar).

    `terminal_on_life_loss=True` acorta los episodios a una vida: sirve para
    entrenar (da señal más densa), no para medir el puntaje de una partida.
    """
    env = make_raw_env(render=render, record_frames=record_frames)
    if monitor:
        # Monitor va por DEBAJO del preprocesamiento a propósito: así registra la
        # recompensa cruda del juego y no una versión recortada o por vida.
        env = Monitor(env)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=FRAMESKIP,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_newaxis=True,
    )
    return env


def make_vec_env(*, seed: int | None = None, **kwargs) -> VecEnv:
    """El entorno vectorizado y apilado que espera el modelo: (84, 84, 4)."""
    venv = DummyVecEnv([lambda: make_env(**kwargs)])
    if seed is not None:
        # Sembrar hace la corrida reproducible, pero NO garantiza que dos semillas
        # distintas den partidas distintas: con política determinista la única
        # variación del entorno es el número de no-ops del reset, y semillas
        # distintas caen en el mismo arranque con frecuencia. Quien necesite
        # episodios independientes debe verificar unicidad, no confiar en la
        # semilla. Ver `record.py` y docs/frameskip.md.
        venv.seed(seed)
    return VecFrameStack(venv, n_stack=N_STACK)


def frameskip_efectivo(venv: VecEnv) -> int:
    """Frames del emulador que consume un `step()` del agente.

    Se mide contra el contador del emulador, no se deduce de la configuración.
    Es la única forma honesta de comprobarlo.
    """
    import numpy as np

    venv.reset()
    ale = venv.venv.envs[0].unwrapped.ale
    antes = ale.getFrameNumber()
    venv.step(np.array([0]))
    return ale.getFrameNumber() - antes
