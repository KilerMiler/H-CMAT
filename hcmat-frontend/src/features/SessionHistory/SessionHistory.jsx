/**
 * src/features/SessionHistory/SessionHistory.jsx
 */

import React from 'react';
import { useDashboardStore } from '../../store/useDashboardStore';
import styles from './SessionHistory.module.css';

function intentColor(intent) {
  if (!intent) return 'var(--text-muted)';

  const upper = intent.toUpperCase();

  if (upper.includes('REFUSAL') || upper.includes('DISSONANCE')) {
    return 'var(--accent-red)';
  }

  if (upper.includes('AGREEMENT') || upper.includes('POSITIVE')) {
    return 'var(--accent-green)';
  }

  if (upper.includes('SURFACE') || upper.includes('AMBIGUOUS')) {
    return '#FBBF24';
  }

  return 'var(--accent-blue-light)';
}

export default function SessionHistory() {
  const history = useDashboardStore((s) => s.history);
  const status = useDashboardStore((s) => s.status);

  const isEmpty = history.length === 0;
  const isActive = status === 'active';

  const reversed = [...history].reverse();

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span>SESSION LOG</span>
        <span className={styles.count}>
          {history.length > 0
            ? `${history.length} EVENTS`
            : isActive
              ? 'LISTENING...'
              : 'NO SESSION'}
        </span>
      </div>

      <div className={styles.list}>
        {isEmpty && (
          <div className={styles.emptyState}>
            {isActive
              ? 'First inference window in ~4 seconds...'
              : 'Start a session to see the live event log.'}
          </div>
        )}

        {reversed.map((entry, idx) => {
          const fusion = entry?.holistic_fusion;
          const matrix = entry?.modality_matrix;
          const isLatest = idx === 0;

          return (
            <div
              key={entry.seq_id}
              className={`${styles.entry} ${isLatest ? styles.latest : ''}`}
            >
              <div className={styles.entryHeader}>
                <span className={styles.turnLabel}>
                  SEQ {entry.seq_id}
                </span>

                <span className={styles.timestamp}>
                  {entry.temporal_context
                    ? `${(entry.temporal_context.clip_start_ms / 1000).toFixed(
                        1
                      )}s`
                    : ''}
                </span>
              </div>

              <div
                className={styles.intent}
                style={{ color: intentColor(fusion?.primary_intent) }}
              >
                {fusion?.primary_intent ?? '—'}
              </div>

              <div className={styles.affect}>
                {fusion?.affective_state ?? '—'}
              </div>

              {matrix && (
                <div className={styles.miniMatrix}>
                  {Object.entries(matrix).map(([channel, detail]) => (
                    <div key={channel} className={styles.miniRow}>
                      <span className={styles.miniLabel}>
                        {channel[0].toUpperCase()}
                      </span>

                      <div className={styles.miniBarBg}>
                        <div
                          className={styles.miniBarFill}
                          style={{
                            width: `${Math.round(
                              (detail.weight ?? 0) * 100
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {fusion?.confidence != null && (
                <div className={styles.confidence}>
                  {Math.round(fusion.confidence * 100)}% conf
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}