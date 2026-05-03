/**
 * src/features/ContextLogic/ContextLogic.jsx
 *
 * Profiles loaded from backend.
 * Culture can be switched live during an active session.
 */

import React, { useEffect, useState } from 'react';
import { useDashboardStore } from '../../store/useDashboardStore';
import { API_BASE } from '../../config/api';
import styles from './ContextLogic.module.css';

export default function ContextLogic() {
  const cultures = useDashboardStore((s) => s.cultures);
  const setCultures = useDashboardStore((s) => s.setCultures);
  const cultureId = useDashboardStore((s) => s.cultureId);
  const setCultureId = useDashboardStore((s) => s.setCultureId);
  const status = useDashboardStore((s) => s.status);

  const [activeProfile, setActiveProfile] = useState(null);
  const [loading, setLoading] = useState(false);

  const isSessionActive = status === 'active';

  useEffect(() => {
    let cancelled = false;

    fetch(`${API_BASE}/cultures`)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch cultures: ${r.status}`);
        return r.json();
      })
      .then((list) => {
        if (cancelled) return;

        setCultures(list);

        const initial = list.find((p) => p.id === cultureId) ?? list[0];

        if (initial) {
          setActiveProfile(initial);

          if (!cultureId) {
            setCultureId(initial.id);
          }
        }
      })
      .catch((err) => console.warn('[Cultures] Failed to load:', err));

    return () => {
      cancelled = true;
    };
  }, [cultureId, setCultures, setCultureId]);

  const handleChange = async (e) => {
    const id = parseInt(e.target.value, 10);

    setCultureId(id);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/cultures/${id}`);

      if (!res.ok) {
        throw new Error(`Culture fetch failed: ${res.status}`);
      }

      const profile = await res.json();
      setActiveProfile(profile);
    } catch (err) {
      console.warn('[Culture] Falling back to cached profile:', err);
      setActiveProfile(cultures.find((p) => p.id === id) ?? null);
    } finally {
      setLoading(false);
    }
  };

  const buildDescription = (profile) => {
    if (!profile) return 'Loading cultural profile...';

    const { weights, politeness_bias, indirect_threshold } = profile;
    const dominant = Object.entries(weights).sort((a, b) => b[1] - a[1])[0];

    return (
      `System weighting: Speech ${Math.round(weights.speech * 100)}% · ` +
      `Face ${Math.round(weights.face * 100)}% · ` +
      `Body ${Math.round(weights.body * 100)}%. ` +
      `Dominant channel: ${dominant[0].toUpperCase()}. ` +
      `Politeness bias: ${Math.round(politeness_bias * 100)}%. ` +
      `Indirect communication threshold: ${Math.round(
        indirect_threshold * 100
      )}%.`
    );
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ color: 'var(--text-secondary)' }}
        >
          <circle cx="12" cy="12" r="10" />
          <line x1="2" y1="12" x2="22" y2="12" />
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
        </svg>

        CULTURAL CONTEXT LOGIC

        {isSessionActive && (
          <span className={styles.liveSwitchBadge}>LIVE SWITCH ENABLED</span>
        )}
      </div>

      <div className={styles.selectWrapper}>
        <select
          className={styles.select}
          value={cultureId}
          onChange={handleChange}
          disabled={loading || cultures.length === 0}
        >
          {cultures.length === 0 ? (
            <option>Loading profiles...</option>
          ) : (
            cultures.map((p) => (
              <option key={p.id} value={p.id}>
                [{p.code}] {p.name}
              </option>
            ))
          )}
        </select>

        <div className={styles.chevron}>
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </div>

      {activeProfile && (
        <div className={styles.weightBars}>
          {Object.entries(activeProfile.weights).map(([channel, w]) => (
            <div key={channel} className={styles.weightRow}>
              <span className={styles.weightLabel}>
                {channel.toUpperCase()}
              </span>

              <div className={styles.weightBarBg}>
                <div
                  className={styles.weightBarFill}
                  style={{ width: `${Math.round(w * 100)}%` }}
                />
              </div>

              <span className={styles.weightValue}>
                {Math.round(w * 100)}%
              </span>
            </div>
          ))}
        </div>
      )}

      <div className={styles.descriptionBox}>
        <p className={styles.descriptionText}>
          {loading ? 'Loading...' : buildDescription(activeProfile)}
        </p>
      </div>
    </div>
  );
}