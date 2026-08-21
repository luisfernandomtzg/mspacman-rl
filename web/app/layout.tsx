import type { Metadata } from "next";
import "./globals.css";

const URL_SITIO = "https://mspacman-rl.vercel.app";
const TITULO = "Ms. Pac-Man DDQN — agente de aprendizaje por refuerzo";
const DESCRIPCION =
  "Agente DDQN entrenado con Double Deep Q-Network sobre Ms. Pac-Man (Atari 2600). " +
  "Demo con los valores Q del agente en cada decisión.";

export const metadata: Metadata = {
  // metadataBase resuelve las rutas relativas de abajo a URLs absolutas.
  // Sin esto, Next.js emite rutas relativas y ninguna plataforma las resuelve:
  // el enlace se comparte sin miniatura.
  metadataBase: new URL(URL_SITIO),
  title: TITULO,
  description: DESCRIPCION,
  openGraph: {
    type: "website",
    url: URL_SITIO,
    siteName: "Ms. Pac-Man DDQN",
    title: TITULO,
    description: DESCRIPCION,
    locale: "es_MX",
    images: [
      {
        url: "/og.png",
        width: 1200,
        height: 630,
        alt: "El agente jugando Ms. Pac-Man. Servirlo correctamente subió el puntaje de 962 a 1623 sin reentrenar.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: TITULO,
    description: DESCRIPCION,
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
