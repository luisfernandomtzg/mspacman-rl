"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Paso = { f0: number; f1: number; a: number; q: number[]; r: number; score: number; vidas: number };
type Episodio = { id: number; videos: string[]; traza: string; score: number; decisiones: number; frames: number; seed: number };
type Manifiesto = {
  acciones: string[];
  fps: number;
  resumen: { episodios: number; media: number; desviacion: number; mejor: number; peor: number };
  episodios: Episodio[];
};

const BASE = "/replay";

/** Índice frame -> paso, para no buscar en cada cuadro de video. */
function construirIndice(traza: Paso[], totalFrames: number): Int32Array {
  const idx = new Int32Array(Math.max(totalFrames, 1)).fill(0);
  let p = 0;
  for (let f = 0; f < idx.length; f++) {
    while (p + 1 < traza.length && f >= traza[p].f1) p++;
    idx[f] = p;
  }
  return idx;
}

export default function Reproductor({ manifiesto }: { manifiesto: Manifiesto }) {
  const [epId, setEpId] = useState(manifiesto.episodios[0]?.id ?? 1);
  const [traza, setTraza] = useState<Paso[] | null>(null);
  const [pasoIdx, setPasoIdx] = useState(0);
  const [reproduciendo, setReproduciendo] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const episodio = useMemo(
    () => manifiesto.episodios.find((e) => e.id === epId) ?? manifiesto.episodios[0],
    [manifiesto.episodios, epId],
  );

  useEffect(() => {
    let vivo = true;
    setTraza(null);
    setPasoIdx(0);
    fetch(`${BASE}/${episodio.traza}`)
      .then((r) => r.json())
      .then((d: Paso[]) => vivo && setTraza(d))
      .catch(() => vivo && setTraza([]));
    return () => { vivo = false; };
  }, [episodio]);

  const indice = useMemo(
    () => (traza && traza.length ? construirIndice(traza, episodio.frames) : null),
    [traza, episodio.frames],
  );

  // Sincroniza el panel con el cuadro que el video está mostrando de verdad.
  // requestVideoFrameCallback da el mediaTime exacto del cuadro presentado;
  // donde no existe, caemos a rAF + currentTime, que basta para 60 fps.
  const sincronizar = useCallback(() => {
    const v = videoRef.current;
    if (!v || !indice) return;
    const f = Math.min(indice.length - 1, Math.max(0, Math.round(v.currentTime * manifiesto.fps)));
    setPasoIdx(indice[f]);
  }, [indice, manifiesto.fps]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !indice) return;
    let cancelado = false;
    let handle = 0;

    type ConRVFC = HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number, md: { mediaTime: number }) => void) => number;
      cancelVideoFrameCallback?: (h: number) => void;
    };
    const vv = v as ConRVFC;

    if (typeof vv.requestVideoFrameCallback === "function") {
      const paso = (_n: number, md: { mediaTime: number }) => {
        if (cancelado) return;
        const f = Math.min(indice.length - 1, Math.max(0, Math.round(md.mediaTime * manifiesto.fps)));
        setPasoIdx(indice[f]);
        handle = vv.requestVideoFrameCallback!(paso);
      };
      handle = vv.requestVideoFrameCallback(paso);
      return () => { cancelado = true; vv.cancelVideoFrameCallback?.(handle); };
    }

    const bucle = () => {
      if (cancelado) return;
      sincronizar();
      handle = requestAnimationFrame(bucle);
    };
    handle = requestAnimationFrame(bucle);
    return () => { cancelado = true; cancelAnimationFrame(handle); };
  }, [indice, manifiesto.fps, sincronizar]);

  const paso = traza?.[pasoIdx];
  const q = paso?.q ?? [];
  const qMin = q.length ? Math.min(...q) : 0;
  const qMax = q.length ? Math.max(...q) : 1;
  const rango = qMax - qMin || 1;

  return (
    <div className="@container">
      <div className="grid gap-5 @[56rem]:grid-cols-[minmax(0,1fr)_20rem]">
        {/* ---------- Video ---------- */}
        <figure className="m-0 self-start overflow-hidden rounded-2xl border border-borde bg-panel">
          <div className="relative grid max-h-[min(72vh,42rem)] place-items-center bg-black">
            <video
              key={episodio.id}
              ref={videoRef}
              className="pixelado block max-h-[min(72vh,42rem)] w-full object-contain"
              style={{ aspectRatio: "160 / 210" }}
              playsInline
              muted
              loop
              preload="metadata"
              onPlay={() => setReproduciendo(true)}
              onPause={() => setReproduciendo(false)}
              aria-label={`Repetición del episodio ${episodio.id}, ${episodio.score} puntos`}
            >
              {episodio.videos.map((v) => (
                <source
                  key={v}
                  src={`${BASE}/${v}`}
                  type={v.endsWith(".webm") ? "video/webm" : "video/mp4"}
                />
              ))}
            </video>
            {reproduciendo && (
              <span className="latido pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-rojo/90 px-3 py-1 text-[0.7rem] font-bold tracking-wide text-white">
                ● DDQN JUGANDO
              </span>
            )}
          </div>

          <figcaption className="flex flex-wrap items-center gap-2 border-t border-borde p-3">
            <button
              type="button"
              onClick={() => {
                const v = videoRef.current;
                if (!v) return;
                v.paused ? void v.play() : v.pause();
              }}
              className="rounded-full bg-azul px-5 py-2 text-sm font-semibold text-white transition hover:brightness-110 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-azul"
            >
              {reproduciendo ? "Pausa" : "Reproducir"}
            </button>
            <button
              type="button"
              onClick={() => { const v = videoRef.current; if (v) { v.currentTime = 0; void v.play(); } }}
              className="rounded-full bg-panel-alto px-5 py-2 text-sm font-semibold text-tinta transition hover:bg-borde focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-azul"
            >
              Reiniciar
            </button>

            <div className="ms-auto flex items-center gap-2">
              <label htmlFor="sel-ep" className="text-xs text-tenue">Episodio</label>
              <select
                id="sel-ep"
                value={epId}
                onChange={(e) => setEpId(Number(e.target.value))}
                className="rounded-lg border border-borde bg-panel-alto px-2 py-1.5 text-sm text-tinta focus-visible:outline-2 focus-visible:outline-azul"
              >
                {manifiesto.episodios.map((e) => (
                  <option key={e.id} value={e.id}>#{e.id} — {e.score} pts</option>
                ))}
              </select>
            </div>
          </figcaption>
        </figure>

        {/* ---------- Panel de decisión ---------- */}
        <aside className="self-start rounded-2xl border border-borde bg-panel p-4 @[56rem]:sticky @[56rem]:top-5">
          <h2 className="m-0 text-sm font-semibold">Qué está pensando</h2>
          <p className="mt-1 mb-4 text-xs leading-relaxed text-tenue">
            Valor Q estimado por la red para cada acción. El agente ejecuta el más alto.
          </p>

          <dl className="mb-4 grid grid-cols-3 gap-2 text-center">
            {[
              ["Puntos", paso ? paso.score.toLocaleString("es-MX") : "—"],
              ["Vidas", paso ? String(paso.vidas) : "—"],
              ["Decisión", paso ? `${pasoIdx + 1}` : "—"],
            ].map(([k, v]) => (
              <div key={k} className="rounded-xl bg-panel-alto px-2 py-2">
                <dt className="text-[0.65rem] uppercase tracking-wide text-tenue">{k}</dt>
                <dd className="m-0 font-mono text-base font-semibold tabular-nums">{v}</dd>
              </div>
            ))}
          </dl>

          <ul className="m-0 list-none space-y-1 p-0">
            {manifiesto.acciones.map((nombre, i) => {
              const valor = q[i];
              const elegida = paso?.a === i;
              const ancho = valor === undefined ? 0 : ((valor - qMin) / rango) * 100;
              return (
                <li key={nombre} className="grid grid-cols-[4.5rem_1fr_3.2rem] items-center gap-2">
                  <span className={`font-mono text-[0.7rem] ${elegida ? "font-bold text-tinta" : "text-tenue"}`}>
                    {nombre}
                  </span>
                  <span
                    className="relative block h-2.5 overflow-hidden rounded-full bg-panel-alto"
                    role="meter"
                    aria-valuenow={valor ?? 0}
                    aria-valuemin={qMin}
                    aria-valuemax={qMax}
                    aria-label={`Valor Q de la acción ${nombre}`}
                  >
                    <span
                      className={`block h-full rounded-full transition-[width] duration-100 ease-out ${elegida ? "bg-verde" : "bg-azul/40"}`}
                      style={{ width: `${ancho}%` }}
                    />
                  </span>
                  <span className={`text-end font-mono text-[0.7rem] tabular-nums ${elegida ? "text-verde" : "text-tenue"}`}>
                    {valor === undefined ? "—" : valor.toFixed(1)}
                  </span>
                </li>
              );
            })}
          </ul>

          <p className="mt-4 mb-0 text-[0.7rem] leading-relaxed text-tenue">
            Repetición grabada, no inferencia en vivo: así el demo abre al instante y
            no depende de un servidor que se duerme.{" "}
            <a href="#metodo" className="text-azul underline-offset-2 hover:underline">Por qué</a>.
          </p>
        </aside>
      </div>
    </div>
  );
}
