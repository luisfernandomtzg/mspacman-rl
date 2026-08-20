# El agente no estaba mal entrenado, estaba mal servido

## Síntoma

El modelo DDQN entrenado con 500 mil pasos promediaba **914 puntos** al evaluarlo, contra
un objetivo de referencia de 2 500. La lectura fácil era "le faltan pasos de entrenamiento".
Antes de gastar horas de GPU en reentrenar, valía la pena comprobar que lo que se estaba
midiendo era realmente el agente.

## Qué se midió

El entorno de entrenamiento y el de servicio se construían con librerías distintas:

```python
# Entrenamiento (notebook)
env = gym.make("ALE/MsPacman-v5", frameskip=1)
env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, ...)

# Servicio (la primera versión de la API)
env = make_atari_env("ALE/MsPacman-v5", n_envs=1)   # SB3
```

Un comentario en el código de servicio afirmaba que el entorno quedaba *"exactamente con
los filtros con los que fue entrenado"*. Eso no se había verificado; se había supuesto.

La comprobación no se deduce de la configuración: se cuenta contra el emulador.

```python
ale = venv.venv.envs[0].unwrapped.ale
antes = ale.getFrameNumber()
venv.step(np.array([0]))
frameskip_efectivo = ale.getFrameNumber() - antes
```

| Configuración | Frameskip efectivo | Puntaje real |
| --- | ---: | ---: |
| `make_atari_env` sin fijar el frameskip de la base | **16** | 962.7 ± 165.5 |
| `make_atari_env` con `frameskip=1` en la base | 4 | **1 622.7 ± 575.5** |
| Pipeline del entrenamiento (`AtariPreprocessing`) | 4 | 1 517.3 ± 470.3 |

<sub>15 episodios por configuración, política determinista, misma semilla.</sub>

## Causa

`ALE/MsPacman-v5` **ya trae `frameskip=4` de fábrica**. El `AtariWrapper` que aplica
`make_atari_env` le encima un `MaxAndSkipEnv(skip=4)`. Los dos saltos no se sustituyen:
se multiplican. El agente entrenó decidiendo cada 4 cuadros y se puso a jugar decidiendo
cada 16 — reaccionaba cuatro veces más lento de lo que había aprendido, y los fantasmas
lo alcanzaban antes de que pudiera cambiar de dirección.

Los espacios de observación coinciden en ambos pipelines — `(84, 84, 4)` en los dos — así
que nada fallaba ni advertía. El modelo cargaba, jugaba y devolvía frames. Solo jugaba mal.

## Corrección

`src/mspacman/envs.py` declara el frameskip como constante del módulo y fuerza
`frameskip=1` en el entorno base, de modo que el único salto sea el explícito.

El resultado: **+69 % de puntaje sin reentrenar un solo paso**, y el pipeline de servicio
pasó a coincidir con el de entrenamiento dentro del margen de ruido — que es la prueba de
que ahora sí son equivalentes.

## Que no vuelva a pasar

`tests/test_envs.py::test_frameskip_efectivo_es_4` mide el frameskip contra el emulador en
cada corrida de CI. Si alguien reconstruye el entorno con el atajo, la prueba falla.

```bash
pytest tests -q
```

## Un segundo defecto, encontrado al verificar este

Al revisar las repeticiones grabadas resultó que dos de tres eran **el mismo episodio**:
470 acciones idénticas al 100 %. Doce "episodios" no eran doce muestras independientes,
y la media de 1 755 ± 392 que se reportó primero no era una estadística válida.

La primera hipótesis fue que la semilla no llegaba al emulador. **Era falsa**, y la prueba
que la habría respaldado pasaba igual con y sin el supuesto arreglo — por eso se mutó el
código a propósito antes de darla por buena. La medición real:

| Forma de sembrar | Trayectorias únicas (12 semillas) | Colisiones |
| --- | ---: | --- |
| `VecEnv.seed(s)` | 10/12 | 11, 44, 77 |
| `reset(seed=s)` | 11/12 | 44, 99 |

Ambas siembran correctamente y ambas son reproducibles. Ambas colisionan.

**La causa real:** con una política determinista, la única fuente de variación del
entorno es el número aleatorio de no-ops que `AtariPreprocessing` ejecuta en el reset —
un entero en un rango de unas decenas. Dos semillas que caen en el mismo arranque generan
la partida idéntica, incluidas las repeticiones pegajosas de acción, porque el flujo
aleatorio también parte del mismo punto. Con 12 semillas sobre ~30 arranques posibles,
que haya colisiones es lo esperable, no la excepción.

**La corrección** no es otra forma de sembrar: es no confiar en la semilla. `record.py`
firma cada episodio por su secuencia completa de acciones, descarta los repetidos y sigue
probando semillas hasta juntar el número pedido de partidas distintas. Funciona sin
importar de dónde venga la colisión.

Las 15 repeticiones finales son 15 trayectorias verificadamente únicas, con puntajes de
680 a 2 060: media **1 408 ± 512**. La desviación subió de 392 a 512 — la variación real
del agente siempre estuvo ahí, los duplicados la escondían.

## Lo que queda

Corregir el servicio no arregla el entrenamiento: el agente **se entrenó** con el pipeline
correcto, así que ~1 620 es su techo real con 500 mil pasos. Sigue lejos de los ~2 500 de
la referencia de DeepMind, que se obtienen con 200 millones de pasos. La diferencia es
presupuesto de cómputo, no un defecto.
