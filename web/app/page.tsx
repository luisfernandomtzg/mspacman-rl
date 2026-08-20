import Reproductor from "@/components/Reproductor";
import manifiesto from "@/public/replay/manifiesto.json";

const RESULTADOS = [
  { agente: "Aleatorio (línea base)", pasos: "—", score: "202", nota: "5 episodios" },
  { agente: "DQN estándar", pasos: "50 mil", score: "434 ± 92", nota: "5 episodios" },
  { agente: "DDQN, servido con el defecto de frameskip", pasos: "500 mil", score: "962 ± 166", nota: "15 episodios" },
  { agente: "DDQN, servido correctamente", pasos: "500 mil", score: "1 623 ± 576", nota: "15 episodios, mismo modelo sin reentrenar", destacado: true },
];

export default function Pagina() {
  return (
    <main className="mx-auto grid max-w-5xl gap-12 px-5 py-12">
      <header className="grid gap-3">
        <p className="m-0 font-mono text-xs uppercase tracking-[0.2em] text-tenue">
          Aprendizaje por refuerzo · Atari 2600
        </p>
        <h1 className="m-0 text-3xl font-semibold tracking-tight sm:text-4xl">
          Un agente que aprendió a jugar Ms. Pac-Man
        </h1>
        <p className="m-0 max-w-2xl text-base leading-relaxed text-tenue">
          Double DQN entrenado 500 mil pasos sobre píxeles crudos, sin ninguna regla del
          juego programada a mano. Abajo puedes ver lo que la red estima para cada acción,
          cuadro por cuadro.
        </p>
      </header>

      <Reproductor manifiesto={manifiesto} />

      <section className="grid gap-4">
        <h2 className="m-0 text-xl font-semibold tracking-tight">Resultados medidos</h2>
        <div className="overflow-x-auto rounded-2xl border border-borde">
          <table className="w-full border-collapse text-sm">
            <caption className="sr-only">Puntaje promedio por configuración del agente</caption>
            <thead>
              <tr className="bg-panel text-start">
                <th scope="col" className="p-3 text-start font-semibold">Agente</th>
                <th scope="col" className="p-3 text-start font-semibold">Pasos</th>
                <th scope="col" className="p-3 text-end font-semibold">Puntaje</th>
                <th scope="col" className="p-3 text-start font-semibold">Medición</th>
              </tr>
            </thead>
            <tbody>
              {RESULTADOS.map((r) => (
                <tr key={r.agente} className={`border-t border-borde ${r.destacado ? "bg-verde/5" : ""}`}>
                  <th scope="row" className={`p-3 text-start font-medium ${r.destacado ? "text-verde" : ""}`}>
                    {r.agente}
                  </th>
                  <td className="p-3 font-mono text-tenue tabular-nums">{r.pasos}</td>
                  <td className={`p-3 text-end font-mono font-semibold tabular-nums ${r.destacado ? "text-verde" : ""}`}>
                    {r.score}
                  </td>
                  <td className="p-3 text-xs leading-snug text-tenue">{r.nota}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="m-0 text-sm leading-relaxed text-tenue">
          El objetivo de referencia del DQN original de DeepMind en Ms. Pac-Man ronda los
          2 500 puntos con 200 millones de pasos. Este agente entrenó 500 mil —{" "}
          <strong className="font-semibold text-tinta">400 veces menos</strong> — y llega a
          ~1 620. No supera la referencia, y decirlo importa más que adornar el número.
        </p>
      </section>

      <section id="metodo" className="grid gap-4 scroll-mt-8">
        <h2 className="m-0 text-xl font-semibold tracking-tight">
          El hallazgo: el modelo no estaba mal entrenado, estaba mal servido
        </h2>
        <div className="grid gap-4 text-sm leading-relaxed text-tenue @container sm:grid-cols-2">
          <p className="m-0">
            La primera versión de este proyecto servía el agente con el atajo de
            <code className="mx-1 rounded bg-panel-alto px-1.5 py-0.5 font-mono text-xs text-tinta">make_atari_env</code>
            de Stable-Baselines3. Ese atajo aplica un salto de 4 cuadros por acción — pero
            el entorno <code className="mx-1 rounded bg-panel-alto px-1.5 py-0.5 font-mono text-xs text-tinta">ALE/MsPacman-v5</code>
            ya trae otro salto de 4 de fábrica. Los dos se multiplican.
          </p>
          <p className="m-0">
            Resultado: el agente entrenó decidiendo cada 4 cuadros y se puso a jugar
            decidiendo cada 16. Reaccionaba cuatro veces más lento de lo que aprendió.
            Corregirlo subió el puntaje un{" "}
            <strong className="font-semibold text-verde">69 %</strong> sin reentrenar ni un paso,
            y el pipeline de servicio pasó a coincidir con el de entrenamiento dentro del
            margen de ruido. Hay una prueba que lo vigila para que no vuelva a pasar.
          </p>
        </div>
      </section>

      <footer className="border-t border-borde pt-6 text-sm text-tenue">
        <p className="m-0">
          Luis Fernando Martínez González ·{" "}
          <a className="text-azul underline-offset-2 hover:underline" href="https://github.com/luisfernandomtzg/mspacman-rl">
            Código y método en GitHub
          </a>{" "}
          ·{" "}
          <a className="text-azul underline-offset-2 hover:underline" href="https://www.linkedin.com/in/luisfernandomtzg">
            LinkedIn
          </a>
        </p>
      </footer>
    </main>
  );
}
