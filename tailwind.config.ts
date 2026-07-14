import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#171714",
        paper: "#f4f2ea",
        lime: "#c7ff4a",
        moss: "#315a3a",
      },
      boxShadow: {
        soft: "0 18px 60px rgba(33, 37, 29, .10)",
      },
    },
  },
  plugins: [],
} satisfies Config;
