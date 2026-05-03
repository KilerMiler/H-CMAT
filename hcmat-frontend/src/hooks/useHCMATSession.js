/**
 * src/hooks/useHCMATSession.js
 *
 * Handles REST lifecycle only.
 */

import { useCallback } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import { API_BASE } from '../config/api';

export function useHCMATSession() {
  const sessionId = useDashboardStore((s) => s.sessionId);
  const cultureId = useDashboardStore((s) => s.cultureId);
  const setSession = useDashboardStore((s) => s.setSession);
  const setSystemInfo = useDashboardStore((s) => s.setSystemInfo);
  const setSummary = useDashboardStore((s) => s.setSummary);
  const openSummaryModal = useDashboardStore((s) => s.openSummaryModal);
  const setIsStarting = useDashboardStore((s) => s.setIsStarting);
  const setIsSummarizing = useDashboardStore((s) => s.setIsSummarizing);
  const setSessionStatus = useDashboardStore((s) => s.setSessionStatus);
  const clearSession = useDashboardStore((s) => s.clearSession);

  const startSession = useCallback(async () => {
    setIsStarting(true);

    try {
      const healthRes = await fetch(`${API_BASE}/health`);

      if (!healthRes.ok) {
        throw new Error(`Backend health check failed: ${healthRes.status}`);
      }

      const health = await healthRes.json();

      if (health.status !== 'ok') {
        const failing = health.encoders
          ?.filter((e) => e.status !== 'loaded')
          ?.map((e) => e.name)
          ?.join(', ');

        throw new Error(`Encoders not ready: ${failing || 'unknown'}`);
      }

      setSystemInfo({
        device: health.device,
        dtype: health.dtype,
        mps_available: health.mps_available,
      });

      const sessionRes = await fetch(`${API_BASE}/inference/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'demo_user',
          culture_id: cultureId,
          modalities: ['speech', 'face', 'body'],
          config: {},
        }),
      });

      if (!sessionRes.ok) {
        const detail = await sessionRes.text();
        throw new Error(`Failed to create session: ${detail}`);
      }

      const session = await sessionRes.json();

      setSession(session.session_id, session.culture_id);
      return session.session_id;
    } catch (err) {
      console.error('[Session] Start failed:', err);
      alert(`Could not start session:\n${err.message}`);
      return null;
    } finally {
      setIsStarting(false);
    }
  }, [cultureId, setSession, setSystemInfo, setIsStarting]);

  const stopSession = useCallback(
    async (sid) => {
      const targetId = sid || sessionId;
      if (!targetId) return;

      setIsSummarizing(true);
      setSessionStatus('summarizing');

      try {
        const summaryRes = await fetch(
          `${API_BASE}/inference/session/${targetId}/summarize`,
          { method: 'POST' }
        );

        if (!summaryRes.ok) {
          const detail = await summaryRes.text();
          throw new Error(`Summarize failed: ${detail}`);
        }

        const summary = await summaryRes.json();

        setSummary(summary);
        openSummaryModal();

        const deleteRes = await fetch(
          `${API_BASE}/inference/session/${targetId}`,
          { method: 'DELETE' }
        );

        if (!deleteRes.ok) {
          console.warn('[Session] DELETE returned:', deleteRes.status);
        }
      } catch (err) {
        console.error('[Session] Stop failed:', err);
        alert(`Could not stop session cleanly:\n${err.message}`);
      } finally {
        setIsSummarizing(false);
        setSessionStatus('done');
      }
    },
    [
      sessionId,
      setSessionStatus,
      setIsSummarizing,
      setSummary,
      openSummaryModal,
    ]
  );

  const resetSession = useCallback(() => {
    clearSession();
  }, [clearSession]);

  return { startSession, stopSession, resetSession };
}