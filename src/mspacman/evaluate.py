"""Evaluación del agente y comparación de pipelines de servicio.

Reproduce la medición que documenta `docs/frameskip.md`:

    python -m mspacman.evaluate --episodios 15
"""
from __future__ import annotations

import argparse
import warnings

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

from .envs import ENV_ID, N_STACK, frameskip_efectivo, make_vec_env
from .model import cargar

warnings.filterwarnings("ignore")


def _sb3_atari(frameskip_base: int | None, seed: int):
    """El atajo de SB3. Con `frameskip_base=None` reproduce el error original."""
    kwargs = {} if frameskip_base is None else {"frameskip": frameskip_base}
    venv = make_atari_env(ENV_ID, n_envs=1, seed=seed, env_kwargs=kwargs)
    return VecFrameStack(venv, n_stack=N_STACK)


PIPELINES = {
    "roto": ("make_atari_env sin fijar frameskip (el error original)",
             lambda s: _sb3_atari(None, s)),
    "sb3-corregido": ("make_atari_env con frameskip=1 en la base",
                      lambda s: _sb3_atari(1, s)),
    "propio": ("mspacman.envs.make_vec_env (el de este repo)",
               lambda s: make_vec_env(seed=s)),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodios", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    modelo = cargar()
    print(f"{'pipeline':<48}{'frameskip':>11}{'puntaje real':>22}")
    print("-" * 81)
    for _, (etiqueta, ctor) in PIPELINES.items():
        venv = ctor(args.seed)
        fs = frameskip_efectivo(venv) if hasattr(venv.venv, "envs") else "?"
        venv.close()

        venv = ctor(args.seed)
        media, desv = evaluate_policy(
            modelo, venv, n_eval_episodes=args.episodios, deterministic=True
        )
        venv.close()
        print(f"{etiqueta:<48}{fs:>7} f/acc {media:>13.1f} ± {desv:<6.1f}")


if __name__ == "__main__":
    main()
