import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ms. Pac-Man DDQN — agente de aprendizaje por refuerzo",
  description:
    "Agente DDQN entrenado con Double Deep Q-Network sobre Ms. Pac-Man (Atari 2600). " +
    "Demo con los valores Q del agente en cada decisión.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
