/**
 * src/features/InferenceMatrix/InferenceMatrix.jsx
 */

import React from 'react';
import { useDashboardStore } from '../../store/useDashboardStore';
import styles from './InferenceMatrix.module.css';

function MatrixRow({ title, feature, weight, localTag, isActive, isEmpty }) {
  const widthPct = isEmpty ? 0 : Math.round((weight ?? 0) * 100);
  const isLow = widthPct < 15;

  return (
    <div className={`${styles.matrixRow} ${isActive ? styles.active : ''}`}>
      <div className={styles.modalityLabel}>
        <span className={styles.modalitySub}>CHANNEL</span>
        <span className={styles.modalityTitle}>{title}</span>
      </div>

      <div className={styles.weightContainer}>
        <span className={styles.modalitySub}>WEIGHT</span>
        <div className={styles.weightBarBg}>
          <div
            className={`${styles.weightBarFill} ${isLow ? styles.low : ''}`}
            style={{ width: `${widthPct}%` }}
          />
        </div>
      </div>

      <div className={styles.rawText}>
        {isEmpty ? 'NULL' : feature || '—'}
      </div>

      <div className={`${styles.badge} ${isEmpty ? styles.inactive : ''}`}>
        {isEmpty ? 'NO ACTIVE INPUT' : localTag || 'PROCESSING...'}
      </div>
    </div>
  );
}

export default function InferenceMatrix() {
  const modalityMatrix = useDashboardStore((s) => s.modalityMatrix);
  const history = useDashboardStore((s) => s.history);
  const latency = useDashboardStore((s) => s.latency);
  const status = useDashboardStore((s) => s.status);

  const isActive = status === 'active';

  const latest = history[history.length - 1];
  const fusion = latest?.holistic_fusion;

  const confidence = fusion ? Math.round(fusion.confidence * 100) : null;
  const primaryIntent = fusion?.primary_intent ?? (isActive ? 'ANALYSING...' : '—');
  const affectState = fusion?.affective_state ?? (isActive ? 'ANALYSING...' : '—');

  const baselineFailing =
    fusion &&
    !['GENUINE AGREEMENT', 'INFORMATION SEEKING'].includes(
      fusion.primary_intent
    );

  const speech = modalityMatrix?.speech;
  const face = modalityMatrix?.face;
  const body = modalityMatrix?.body;

  const dominantChannel =
    Object.entries(modalityMatrix || {})
      .sort((a, b) => (b[1]?.weight ?? 0) - (a[1]?.weight ?? 0))[0]?.[0]
      ?.toUpperCase() ?? '—';

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span>MICRO-INFERENCE MATRIX</span>
        <span
          style={{
            fontSize: '0.65rem',
            color: isActive ? 'var(--accent-green)' : 'var(--text-muted)',
          }}
        >
          {isActive ? `⚡ LIVE — ${latency}ms` : 'REAL-TIME DECOUPLED STREAMS'}
        </span>
      </div>

      <div className={styles.matrixTable}>
        <MatrixRow
          title="Speech Channel"
          feature={speech?.feature}
          weight={speech?.weight ?? 0}
          localTag={speech?.local_tag}
          isActive={speech && speech.weight > 0.4}
        />

        <MatrixRow
          title="Facial Expressions"
          feature={face?.feature}
          weight={face?.weight ?? 0}
          localTag={face?.local_tag}
          isActive={face && face.weight > 0.4}
        />

        <MatrixRow
          title="Body / Gesture / Hand Signal"
          feature={body?.feature}
          weight={body?.weight ?? 0}
          localTag={body?.local_tag}
          isActive={body && body.weight > 0.4}
        />
      </div>

      <div className={styles.synthesisGrid}>
        <div className={styles.synthesisCard}>
          <div className={styles.cardHeader}>
            <span>STANDARD EARLY-FUSION</span>
            <span
              style={{
                color: 'var(--accent-red)',
                border: '1px solid var(--accent-red)',
                padding: '2px 6px',
                borderRadius: '2px',
              }}
            >
              LEGACY BASELINE
            </span>
          </div>

          {baselineFailing ? (
            <div className={styles.errorState}>
              <div className={styles.errorBadge}>
                SIGNAL DISTORTED: PRAGMATIC NUANCE IGNORED
              </div>
              <div className={styles.errorSubtext}>
                Detecting surface agreement only.
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', marginTop: '20px' }}>
              <div className={styles.intentTitle}>
                Intent:{' '}
                <span
                  style={{
                    color: 'var(--accent-green)',
                    fontWeight: 'bold',
                  }}
                >
                  {isActive ? 'AGREEMENT' : '—'}
                </span>
              </div>
              <div className={styles.errorSubtext} style={{ marginTop: '8px' }}>
                Failing to detect internal dissonance.
              </div>
            </div>
          )}
        </div>

        <div className={`${styles.synthesisCard} ${styles.highlight}`}>
          <div className={styles.cardHeader}>
            <span>H-CMAT HOLISTIC PRAGMATIC STATE</span>
            {confidence !== null && (
              <span style={{ color: 'var(--text-primary)' }}>
                <span style={{ color: 'var(--accent-green)' }}>●</span>{' '}
                {confidence}% Confidence
              </span>
            )}
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '16px',
              marginTop: '8px',
            }}
          >
            <div>
              <div className={styles.intentTitle}>PRIMARY INTENT</div>
              <div className={styles.intentValue}>{primaryIntent}</div>
            </div>

            <div>
              <div className={styles.intentTitle}>AFFECTIVE STATE</div>
              <div className={`${styles.intentValue} ${fusion ? styles.green : ''}`}>
                {affectState}
              </div>
            </div>
          </div>

          {fusion && (
            <div className={styles.nuanceBox}>
              H-CMAT detected <strong>{primaryIntent}</strong> with{' '}
              <strong>{confidence}%</strong> confidence. Dominant channel:{' '}
              <strong>{dominantChannel}</strong>. Affective state:{' '}
              {affectState}.
            </div>
          )}

          {!fusion && isActive && (
            <div className={styles.nuanceBox} style={{ color: 'var(--text-muted)' }}>
              Waiting for first inference window...
            </div>
          )}

          {!fusion && !isActive && (
            <div className={styles.nuanceBox} style={{ color: 'var(--text-muted)' }}>
              Start a session to see live analysis here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}