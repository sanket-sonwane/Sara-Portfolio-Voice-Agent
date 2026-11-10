/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_SARANG_API?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
