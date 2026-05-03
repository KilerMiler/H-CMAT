// src/layouts/DashboardLayout.jsx
import React from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import styles from './DashboardLayout.module.css';

export default function DashboardLayout({ children }) {
  // FIXED: Pull out state variables individually to avoid infinite re-renders
  const hardware = useDashboardStore((state) => state.hardware);
  const precision = useDashboardStore((state) => state.precision);
  const latency = useDashboardStore((state) => state.latency);

  return (
    <div className={styles.dashboard}>
      <header className={styles.header}>
        <div className={styles.logo}>H-CMAT</div>
        <div className={styles.headerMetrics}>
          <span>Hardware: <span style={{color: 'var(--accent-green)'}}>{hardware}</span></span>
          <span>Precision: {precision}</span>
          <span>Lat: {latency}ms</span>
        </div>
      </header>
      
      <main className={styles.mainContent}>
        {children}
      </main>
    </div>
  );
}