/**
 * src/features/SessionSummaryModal/SessionSummaryModal.jsx
 *
 * NEW component. The "Grand Finale" modal that appears when the user
 * clicks Stop. Displays the holistic_summary paragraph, dominant intent,
 * dominant affect, session confidence, and full turn count.
 *
 * Closes by clicking outside or the X button, then resets the session.
 */

import React, { useCallback } from 'react';
import { useDashboardStore }  from '../../store/useDashboardStore';
import { useHCMATSession }    from '../../hooks/useHCMATSession';
import styles from './SessionSummaryModal.module.css';

export default function SessionSummaryModal() {
  const isOpen     = useDashboardStore((s) => s.isSummaryModalOpen);
  const summary    = useDashboardStore((s) => s.summary);
  const closeModal = useDashboardStore((s) => s.closeSummaryModal);
  const { resetSession } = useHCMATSession();

  const handleClose = useCallback(() => {
    closeModal();
    resetSession();
  }, [closeModal, resetSession]);

  if (!isOpen || !summary) return null;

  const confPct = Math.round((summary.session_confidence ?? 0) * 100);

  // Color-code dominant intent
  const intentColor = () => {
    const i = summary.dominant_intent?.toUpperCase() ?? '';
    if (i.includes('REFUSAL') || i.includes('DISSONANCE')) return 'var(--accent-red)';
    if (i.includes('AGREEMENT') || i.includes('POSITIVE'))  return 'var(--accent-green)';
    return 'var(--accent-blue-light)';
  };

  return (
    <div className={styles.backdrop} onClick={handleClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className={styles.header}>
          <div className={styles.title}>
            <span className={styles.titleBadge}>H-CMAT</span>
            HOLISTIC PRAGMATIC STATE REPORT
          </div>
          <button className={styles.closeBtn} onClick={handleClose}>✕</button>
        </div>

        {/* Key metrics row */}
        <div className={styles.metricsRow}>
          <div className={styles.metric}>
            <div className={styles.metricLabel}>DOMINANT INTENT</div>
            <div className={styles.metricValue} style={{ color: intentColor() }}>
              {summary.dominant_intent ?? '—'}
            </div>
          </div>
          <div className={styles.metric}>
            <div className={styles.metricLabel}>AFFECTIVE STATE</div>
            <div className={styles.metricValue} style={{ color: 'var(--accent-green)' }}>
              {summary.dominant_affect ?? '—'}
            </div>
          </div>
          <div className={styles.metric}>
            <div className={styles.metricLabel}>SESSION CONFIDENCE</div>
            <div className={styles.metricValue}>{confPct}%</div>
          </div>
          <div className={styles.metric}>
            <div className={styles.metricLabel}>TURNS ANALYSED</div>
            <div className={styles.metricValue}>{summary.turn_count ?? 0}</div>
          </div>
        </div>

        {/* Holistic summary paragraph */}
        <div className={styles.summaryBox}>
          <div className={styles.summaryLabel}>HOLISTIC ANALYSIS</div>
          <p className={styles.summaryText}>
            {summary.holistic_summary}
          </p>
        </div>

        {/* Ledger preview */}
        {summary.ledger?.length > 0 && (
          <div className={styles.ledgerSection}>
            <div className={styles.ledgerLabel}>DEDUPED EVENT LEDGER ({summary.ledger.length} events)</div>
            <div className={styles.ledgerList}>
              {summary.ledger.map((entry, i) => (
                <div key={i} className={styles.ledgerRow}>
                  <span className={styles.ledgerSeq}>T{i + 1}</span>
                  <span className={styles.ledgerIntent}>{entry.primary_intent}</span>
                  <span className={styles.ledgerAffect}>{entry.affective_state}</span>
                  <span className={styles.ledgerConf}>{Math.round(entry.confidence * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* CTA */}
        <button className={styles.newSessionBtn} onClick={handleClose}>
          START NEW SESSION
        </button>

      </div>
    </div>
  );
}