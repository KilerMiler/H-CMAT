/**
 * src/store/useDashboardStore.js
 */

import { create } from 'zustand';

const INITIAL_LIVE = {
  modalityMatrix: {
    speech: { feature: '—', weight: 0, local_tag: 'WAITING' },
    face: { feature: '—', weight: 0, local_tag: 'WAITING' },
    body: { feature: '—', weight: 0, local_tag: 'WAITING' },
  },
  attentionWeights: {},
  lastSeqId: -1,
};

const INITIAL_SESSION = {
  sessionId: null,
  status: 'idle', // idle | active | stopping | summarizing | done
  cultureId: 1,
  sessionStart: null,
};

export const useDashboardStore = create((set, get) => ({
  // ── SYSTEM ──────────────────────────────────────────────────────────────
  hardware: 'Detecting...',
  precision: '—',
  latency: 0,
  mpsActive: false,

  setSystemInfo: ({ device, dtype, mps_available }) =>
    set({
      hardware: device ?? 'Unknown',
      precision: dtype ?? '—',
      mpsActive: mps_available ?? false,
    }),

  setLatency: (ms) => set({ latency: ms }),

  // ── CULTURES ────────────────────────────────────────────────────────────
  cultures: [],
  culturesLoaded: false,

  setCultures: (list) =>
    set({
      cultures: Array.isArray(list) ? list : [],
      culturesLoaded: true,
    }),

  // ── SESSION ─────────────────────────────────────────────────────────────
  ...INITIAL_SESSION,

  setSession: (sessionId, cultureId) =>
    set({
      sessionId,
      cultureId,
      status: 'active',
      sessionStart: Date.now(),
      history: [],
      summary: null,
      wsErrors: [],
      ...INITIAL_LIVE,
    }),

  setSessionStatus: (status) => set({ status }),

  setCultureId: (id) => set({ cultureId: id }),

  clearSession: () =>
    set({
      ...INITIAL_SESSION,
      ...INITIAL_LIVE,
      history: [],
      summary: null,
      wsErrors: [],
    }),

  // ── LIVE ────────────────────────────────────────────────────────────────
  ...INITIAL_LIVE,

  applyMatrixUpdate: ({ seq_id, modality_matrix }) =>
    set({
      modalityMatrix: modality_matrix || INITIAL_LIVE.modalityMatrix,
      lastSeqId: seq_id,
    }),

  applyAttentionTick: ({ weights }) =>
    set({
      attentionWeights: weights || {},
    }),

  // ── HISTORY ─────────────────────────────────────────────────────────────
  history: [],

  /**
   * Backend NMS behavior:
   *   is_new_event=true
   *      → append row
   *
   *   is_new_event=false + replaces_seq_id
   *      → replace row with matching seq_id
   *
   *   is_new_event=false + replaces_seq_id=null
   *      → suppressed duplicate, do nothing
   */
  applyFusionResult: (fusionResult) =>
    set((state) => {
      const fusion = fusionResult?.holistic_fusion;
      const isNew = fusion?.is_new_event;
      const replacesSeqId = fusion?.replaces_seq_id;

      const next = {
        latency:
          typeof fusionResult?.total_latency_ms === 'number'
            ? fusionResult.total_latency_ms
            : state.latency,
      };

      if (isNew === true) {
        return {
          ...next,
          history: [...state.history, fusionResult],
        };
      }

      if (isNew === false && replacesSeqId != null) {
        const idx = state.history.findIndex((e) => e.seq_id === replacesSeqId);

        if (idx >= 0) {
          const updated = [...state.history];
          updated[idx] = fusionResult;
          return {
            ...next,
            history: updated,
          };
        }

        // Fallback if frontend missed the original row.
        return {
          ...next,
          history: [...state.history, fusionResult],
        };
      }

      if (isNew === false && replacesSeqId == null) {
        // Suppressed duplicate.
        return next;
      }

      // Defensive fallback for older backend payloads.
      return {
        ...next,
        history: [...state.history, fusionResult],
      };
    }),

  clearHistory: () => set({ history: [] }),

  // ── SUMMARY ─────────────────────────────────────────────────────────────
  summary: null,

  setSummary: (data) => set({ summary: data }),

  // ── UI FLAGS ────────────────────────────────────────────────────────────
  isSummaryModalOpen: false,
  isStarting: false,
  isSummarizing: false,

  openSummaryModal: () => set({ isSummaryModalOpen: true }),
  closeSummaryModal: () => set({ isSummaryModalOpen: false }),
  setIsStarting: (v) => set({ isStarting: v }),
  setIsSummarizing: (v) => set({ isSummarizing: v }),

  // ── WS ERRORS ───────────────────────────────────────────────────────────
  wsErrors: [],

  pushWsError: ({ code, message, seq_id }) =>
    set((state) => ({
      wsErrors: [
        ...state.wsErrors.slice(-9),
        { code, message, seq_id, ts: Date.now() },
      ],
    })),

  // ── ENVIRONMENTAL STRESSORS ─────────────────────────────────────────────
  audioNoise: 42.4,
  visualOcclusion: 12.0,
  isAutoSensing: true,

  setAudioNoise: (val) => set({ audioNoise: val }),
  setVisualOcclusion: (val) => set({ visualOcclusion: val }),
  toggleAutoSensing: () =>
    set((s) => ({ isAutoSensing: !s.isAutoSensing })),
}));