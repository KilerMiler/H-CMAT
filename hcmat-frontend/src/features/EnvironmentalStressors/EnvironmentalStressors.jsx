// src/features/EnvironmentalStressors/EnvironmentalStressors.jsx
import React, { useEffect } from 'react';
import { useDashboardStore } from '../../store/useDashboardStore';
import styles from './EnvironmentalStressors.module.css';

export default function EnvironmentalStressors() {
  // Pull state individually to prevent infinite loops
  const isAutoSensing = useDashboardStore((state) => state.isAutoSensing);
  const toggleAutoSensing = useDashboardStore((state) => state.toggleAutoSensing);
  const audioNoise = useDashboardStore((state) => state.audioNoise);
  const setAudioNoise = useDashboardStore((state) => state.setAudioNoise);
  const visualOcclusion = useDashboardStore((state) => state.visualOcclusion);
  const setVisualOcclusion = useDashboardStore((state) => state.setVisualOcclusion);

  // Demo Telemetry Simulator
  useEffect(() => {
    let interval;
    if (isAutoSensing) {
      interval = setInterval(() => {
        // Create a slight, realistic fluctuation in the values
        // Noise fluctuates +/- 1.5 dB
        const noiseDrift = (Math.random() - 0.5) * 3;
        let newNoise = audioNoise + noiseDrift;
        // Keep it bounded
        if (newNoise < 35) newNoise = 35; 
        if (newNoise > 65) newNoise = 65;
        
        // Occlusion fluctuates +/- 0.5%
        const occDrift = (Math.random() - 0.5) * 1;
        let newOcc = visualOcclusion + occDrift;
        // Keep it bounded
        if (newOcc < 8) newOcc = 8;
        if (newOcc > 25) newOcc = 25;

        setAudioNoise(newNoise);
        setVisualOcclusion(newOcc);
      }, 1500); // Updates every 1.5 seconds for a smooth, natural feel
    }

    return () => clearInterval(interval);
  }, [isAutoSensing, audioNoise, visualOcclusion, setAudioNoise, setVisualOcclusion]);

  const getSliderStyle = (value, max) => {
    const percentage = (value / max) * 100;
    // When locked, the filled bar is slightly dimmer to look less "interactive"
    const color = isAutoSensing ? 'rgba(0, 255, 157, 0.6)' : 'var(--accent-green)';
    return {
      background: `linear-gradient(to right, ${color} ${percentage}%, var(--border-subtle) ${percentage}%)`
    };
  };

  return (
    <div className={styles.container}>
      
      {/* Header with Title and Toggle */}
      <div className={styles.headerRow}>
        <div className={styles.title}>
          {/* Simple generic sensor icon */}
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{color: 'var(--accent-green)'}}>
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
          </svg>
          ENVIRONMENTAL SENSORS
        </div>

        <div className={styles.toggleContainer}>
          <span className={styles.toggleLabel}>AUTO SENSORS</span>
          <div 
            className={`${styles.toggleSwitch} ${isAutoSensing ? styles.active : ''}`}
            onClick={toggleAutoSensing}
          >
            <div className={styles.toggleKnob}></div>
          </div>
        </div>
      </div>

      {/* Audio Noise Block */}
      <div className={`${styles.stressorBlock} ${isAutoSensing ? styles.locked : ''}`}>
        <div className={styles.labelRow}>
          <span className={styles.label}>AMBIENT AUDIO NOISE (DB)</span>
          <span className={styles.value}>{audioNoise.toFixed(1)} dB</span>
        </div>
        <input 
          type="range" 
          min="0" max="120" step="0.1"
          value={audioNoise} 
          onChange={(e) => setAudioNoise(parseFloat(e.target.value))}
          className={styles.slider}
          style={getSliderStyle(audioNoise, 120)}
          disabled={isAutoSensing}
        />
        {isAutoSensing && <div className={styles.statusText}>Locked: Active sensing enabled</div>}
      </div>

      {/* Visual Occlusion Block */}
      <div className={`${styles.stressorBlock} ${isAutoSensing ? styles.locked : ''}`}>
        <div className={styles.labelRow}>
          <span className={styles.label}>VISUAL OCCLUSION (%)</span>
          <span className={styles.value}>{visualOcclusion.toFixed(1)} %</span>
        </div>
        <input 
          type="range" 
          min="0" max="100" step="0.1"
          value={visualOcclusion} 
          onChange={(e) => setVisualOcclusion(parseFloat(e.target.value))}
          className={styles.slider}
          style={getSliderStyle(visualOcclusion, 100)}
          disabled={isAutoSensing}
        />
        {isAutoSensing && <div className={styles.statusText}>Locked: Optimized for current lighting</div>}
      </div>

    </div>
  );
}