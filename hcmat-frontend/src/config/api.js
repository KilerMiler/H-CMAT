// src/config/api.js

export const API_ORIGIN =
  import.meta.env.VITE_HCMAT_API_ORIGIN || 'http://127.0.0.1:8000';

export const WS_ORIGIN =
  import.meta.env.VITE_HCMAT_WS_ORIGIN || 'ws://127.0.0.1:8000';

export const API_BASE = `${API_ORIGIN}/api/v1`;
export const WS_BASE = `${WS_ORIGIN}/api/v1`;