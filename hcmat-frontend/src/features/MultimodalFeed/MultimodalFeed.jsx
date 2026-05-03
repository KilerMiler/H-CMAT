/**
 * src/features/MultimodalFeed/MultimodalFeed.jsx
 *
 * Owns:
 *   - camera/mic stream
 *   - MediaRecorder
 *   - WebSocket
 *   - rolling 4s WebM window
 *   - Web Worker base64 encoding
 */

import React, { useRef, useCallback, useState, useEffect } from 'react';
import { useDashboardStore } from '../../store/useDashboardStore';
import { useHCMATSession } from '../../hooks/useHCMATSession';
import { WS_BASE } from '../../config/api';
import styles from './MultimodalFeed.module.css';

const CHUNK_INTERVAL_MS = 1000;
const STRIDE_MS = 2000;
const WINDOW_CHUNKS = 4;
const MAX_BUF = 5;

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function waitForWsOpen(ws, timeoutMs = 3000) {
  return new Promise((resolve, reject) => {
    if (!ws) return reject(new Error('WebSocket missing.'));
    if (ws.readyState === WebSocket.OPEN) return resolve();

    const started = Date.now();

    const timer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        clearInterval(timer);
        resolve();
      }

      if (ws.readyState === WebSocket.CLOSED) {
        clearInterval(timer);
        reject(new Error('WebSocket closed before opening.'));
      }

      if (Date.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error('WebSocket open timeout.'));
      }
    }, 50);
  });
}

export default function MultimodalFeed() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const recorderRef = useRef(null);
  const workerRef = useRef(null);
  const headerRef = useRef(null);
  const bufferRef = useRef([]);
  const seqRef = useRef(0);
  const intervalRef = useRef(null);
  const pendingSendsRef = useRef(new Map());

  const [cameraError, setCameraError] = useState(null);

  const status = useDashboardStore((s) => s.status);
  const isStarting = useDashboardStore((s) => s.isStarting);
  const sessionId = useDashboardStore((s) => s.sessionId);
  const cultureId = useDashboardStore((s) => s.cultureId);
  const sessionStart = useDashboardStore((s) => s.sessionStart);
  const modalityMatrix = useDashboardStore((s) => s.modalityMatrix);
  const setSessionStatus = useDashboardStore((s) => s.setSessionStatus);
  const pushWsError = useDashboardStore((s) => s.pushWsError);

  const { startSession, stopSession } = useHCMATSession();

  const isActive = status === 'active';
  const faceFeature = modalityMatrix?.face?.feature ?? '—';
  const bodyFeature = modalityMatrix?.body?.feature ?? '—';

  /**
   * Important:
   * Do not feed previous predicted intents back as user text.
   * Backend audio encoder now gets the actual WebM audio stream.
   * If later you add a text input box, return last typed utterances here.
   */
  const getTextHistory = useCallback(() => [], []);

  const initWorker = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate();
    }

    const worker = new Worker('/mediaWorker.js');

    worker.onmessage = (event) => {
      const { type, payload } = event.data || {};

      if (type === 'CHUNK_READY') {
        const { seqId, chunkPayload } = payload;
        const pending = pendingSendsRef.current.get(seqId);

        try {
          const ws = wsRef.current;

          if (!ws || ws.readyState !== WebSocket.OPEN) {
            throw new Error('WebSocket is not open.');
          }

          ws.send(JSON.stringify(chunkPayload));

          if (pending) pending.resolve();
        } catch (err) {
          console.error('[Worker→WS] Send failed:', err);
          if (pending) pending.reject(err);
        } finally {
          pendingSendsRef.current.delete(seqId);
        }
      }

      if (type === 'CHUNK_ERROR') {
        console.error('[Worker] Chunk failed:', payload);

        const pending = pendingSendsRef.current.get(payload.seqId);

        if (pending) {
          pending.reject(new Error(payload.error));
          pendingSendsRef.current.delete(payload.seqId);
        }

        pushWsError({
          code: 'WORKER_CHUNK_ERROR',
          message: payload.error,
          seq_id: payload.seqId,
        });
      }
    };

    workerRef.current = worker;
  }, [pushWsError]);

  const openWs = useCallback(
    (sid) => {
      if (wsRef.current) {
        wsRef.current.close();
      }

      const ws = new WebSocket(`${WS_BASE}/stream/${sid}`);

      ws.onopen = () => console.log('[WS] Connected');

      ws.onclose = (event) => {
        console.log('[WS] Closed', event.code, event.reason);
      };

      ws.onerror = (e) => {
        console.error('[WS] Error', e);
      };

      ws.onmessage = (e) => {
        try {
          const { type, payload } = JSON.parse(e.data);
          const store = useDashboardStore.getState();

          if (type === 'matrix_update') store.applyMatrixUpdate(payload);
          if (type === 'fusion_result') store.applyFusionResult(payload);
          if (type === 'attention_tick') store.applyAttentionTick(payload);
          if (type === 'error') {
            console.warn('[WS] Backend error:', payload);
            store.pushWsError(payload);
          }
        } catch (err) {
          console.warn('[WS] Bad message:', err);
        }
      };

      wsRef.current = ws;
      return ws;
    },
    []
  );

  const dispatch = useCallback(
    async (sid, isFinal = false) => {
      const header = headerRef.current;
      const buffer = bufferRef.current;

      if (!sid) return;
      if (!header || buffer.length === 0) return;

      const ws = wsRef.current;

      if (!ws || ws.readyState !== WebSocket.OPEN) {
        console.warn('[Dispatch] WebSocket not open; skipping chunk.');
        return;
      }

      const windowChunks = buffer.slice(-WINDOW_CHUNKS);
      const firstChunk = windowChunks[0];
      const lastChunk = windowChunks[windowChunks.length - 1];

      const baseStart = sessionStart ?? Date.now();

      const clipStartMs = Math.max(0, firstChunk.arrivedAt - baseStart);

      // Add 1s because arrivedAt is when the MediaRecorder chunk was emitted.
      const clipEndMs = Math.max(
        clipStartMs + 1,
        lastChunk.arrivedAt - baseStart + CHUNK_INTERVAL_MS
      );

      const seqId = seqRef.current++;

      const sendPromise = new Promise((resolve, reject) => {
        pendingSendsRef.current.set(seqId, { resolve, reject });
      });

      workerRef.current?.postMessage({
        type: 'SEND_CHUNK',
        payload: {
          headerChunk: header,
          windowChunks,
          meta: {
            sessionId: sid,
            seqId,
            clipStartMs,
            clipEndMs,
            isFinalClip: isFinal,
            textHistory: getTextHistory(),
            cultureId,
          },
        },
      });

      return sendPromise;
    },
    [sessionStart, cultureId, getTextHistory]
  );

  const handleStart = useCallback(async () => {
    setCameraError(null);

    let stream;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
        },
        audio: true,
      });
    } catch (err) {
      console.error('[Camera] Access failed:', err);
      setCameraError(
        'Camera or microphone access denied. Please allow permissions and try again.'
      );
      return;
    }

    streamRef.current = stream;

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => {});
    }

    const sid = await startSession();

    if (!sid) {
      stream.getTracks().forEach((t) => t.stop());

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }

      return;
    }

    initWorker();

    const ws = openWs(sid);

    try {
      await waitForWsOpen(ws);
    } catch (err) {
      console.error('[WS] Could not open:', err);
      alert(`Could not open WebSocket:\n${err.message}`);
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    headerRef.current = null;
    bufferRef.current = [];
    seqRef.current = 0;
    pendingSendsRef.current.clear();

    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')
      ? 'video/webm;codecs=vp8,opus'
      : 'video/webm';

    const recorder = new MediaRecorder(stream, { mimeType });
    recorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (!e.data || e.data.size === 0) return;

      if (headerRef.current === null) {
        headerRef.current = e.data;
        return;
      }

      bufferRef.current.push({
        blob: e.data,
        arrivedAt: Date.now(),
      });

      if (bufferRef.current.length > MAX_BUF) {
        bufferRef.current.shift();
      }
    };

    recorder.start(CHUNK_INTERVAL_MS);

    setTimeout(() => {
      if (intervalRef.current) clearInterval(intervalRef.current);

      intervalRef.current = setInterval(() => {
        dispatch(sid, false).catch((err) =>
          console.warn('[Dispatch] Chunk failed:', err)
        );
      }, STRIDE_MS);
    }, STRIDE_MS);
  }, [startSession, initWorker, openWs, dispatch]);

  const handleStop = useCallback(async () => {
    if (!sessionId) return;

    setSessionStatus('stopping');

    clearInterval(intervalRef.current);
    intervalRef.current = null;

    try {
      // Send final clip and wait until it is actually written to WS.
      await dispatch(sessionId, true);

      // Give backend time to process final inference_result before summarize.
      await wait(1800);
    } catch (err) {
      console.warn('[Stop] Final clip failed:', err);
    }

    try {
      recorderRef.current?.stop();
    } catch {}

    streamRef.current?.getTracks().forEach((t) => t.stop());

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    await stopSession(sessionId);

    wsRef.current?.close(1000, 'Session stopped');

    workerRef.current?.terminate();
    workerRef.current = null;

    streamRef.current = null;
    headerRef.current = null;
    bufferRef.current = [];
    recorderRef.current = null;
    pendingSendsRef.current.clear();
  }, [sessionId, dispatch, setSessionStatus, stopSession]);

  useEffect(() => {
    return () => {
      clearInterval(intervalRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      wsRef.current?.close();
      workerRef.current?.terminate();
      pendingSendsRef.current.clear();
    };
  }, []);

  const isBusy =
    isStarting || status === 'summarizing' || status === 'stopping';

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.liveIndicator}>
          <div
            className={`${styles.dot} ${isActive ? styles.dotActive : ''}`}
          />
          {isActive ? 'LIVE MULTIMODAL FEED' : 'MULTIMODAL FEED — STANDBY'}
        </div>

        <div className={styles.channel}>
          {sessionId ? `SID_${sessionId.slice(-6).toUpperCase()}` : 'CH_01_INPUT'}
        </div>
      </div>

      <div className={styles.videoWrapper}>
        <video ref={videoRef} autoPlay playsInline muted className={styles.video} />

        {!isActive && (
          <div className={styles.idleOverlay}>
            <div className={styles.idleIcon}>⬤</div>
            <div className={styles.idleText}>PRESS START TO BEGIN</div>
          </div>
        )}

        {cameraError && (
          <div className={styles.errorOverlay}>
            <div className={styles.errorText}>{cameraError}</div>
          </div>
        )}

        {isActive && (
          <>
            <div className={styles.overlayLabels}>
              <span className={styles.label}>FACE: {faceFeature}</span>
              <span className={styles.label}>BODY: {bodyFeature}</span>
            </div>
            <div className={styles.scanline} />
          </>
        )}
      </div>

      <button
        className={`${styles.sessionBtn} ${
          isActive ? styles.stopBtn : styles.startBtn
        }`}
        onClick={isActive ? handleStop : handleStart}
        disabled={isBusy}
      >
        {isStarting
          ? '⏳  INITIALISING...'
          : status === 'summarizing'
            ? '⏳  GENERATING SUMMARY...'
            : status === 'stopping'
              ? '⏳  STOPPING...'
              : isActive
                ? '⬛  STOP SESSION'
                : '▶  START SESSION'}
      </button>
    </div>
  );
}