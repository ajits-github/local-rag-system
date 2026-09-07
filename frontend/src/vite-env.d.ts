/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "developer" | "production" (anything else falls back to "production"); see src/config/uiMode.ts. */
  readonly VITE_UI_MODE?: string;
}
