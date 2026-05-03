/**
 * src/App.jsx
 *
 * Layout is unchanged (left: feed + sensors + culture, right: matrix).
 * New additions:
 *   - SessionHistory panel in the right column
 *   - SessionSummaryModal mounted at root level
 *   - Firebase import removed
 *   - useTelemetryStream replaced by useHCMATSession (in MultimodalFeed)
 */

import React from 'react';
import DashboardLayout          from './layouts/DashboardLayout';
import MultimodalFeed           from './features/MultimodalFeed/MultimodalFeed';
import EnvironmentalStressors   from './features/EnvironmentalStressors/EnvironmentalStressors';
import InferenceMatrix          from './features/InferenceMatrix/InferenceMatrix';
import ContextLogic             from './features/ContextLogic/ContextLogic';
import SessionHistory           from './features/SessionHistory/SessionHistory';
import SessionSummaryModal      from './features/SessionSummaryModal/SessionSummaryModal';
import styles from './layouts/DashboardLayout.module.css';

function App() {
  return (
    <>
      <DashboardLayout>

        {/* ── Left Column ───────────────────────────────────── */}
        <div className={styles.column}>

          {/* Live Multimodal Feed + Start/Stop button */}
          <div className={styles.panel} style={{ flex: 1, minHeight: '320px', display: 'flex', flexDirection: 'column' }}>
            <MultimodalFeed />
          </div>

          {/* Environmental Stressors */}
          <div className={styles.panel}>
            <EnvironmentalStressors />
          </div>

          {/* Cultural Context Logic (loads from API) */}
          <div className={styles.panel}>
            <ContextLogic />
          </div>

        </div>

        {/* ── Right Column ──────────────────────────────────── */}
        <div className={styles.column}>

          {/* Inference Matrix — real-time Glass Box */}
          <div className={styles.panel} style={{ flex: 2, display: 'flex', flexDirection: 'column' }}>
            <InferenceMatrix />
          </div>

          {/* Session History — NMS-deduplicated live log */}
          <div className={styles.panel} style={{ flex: 1, minHeight: '200px', display: 'flex', flexDirection: 'column' }}>
            <SessionHistory />
          </div>

        </div>

      </DashboardLayout>

      {/* Grand Finale modal — mounted outside layout so it overlays everything */}
      <SessionSummaryModal />
    </>
  );
}

export default App;