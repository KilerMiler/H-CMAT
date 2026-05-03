// src/hooks/useTelemetryStream.js
import { useEffect } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';

export function useTelemetryStream(url) {
  // Grab the master update function from Zustand
  const updateMetrics = useDashboardStore((state) => state.updateMetrics);

  useEffect(() => {
    // Open the pipe to FastAPI
    const ws = new WebSocket(url);

    ws.onopen = () => console.log('🟢 H-CMAT Live Telemetry Connected');

    // Every time FastAPI sends a packet, update Zustand instantly
    ws.onmessage = (event) => {
      const liveData = JSON.parse(event.data);
      updateMetrics(liveData); 
    };

    ws.onclose = () => console.log('🔴 H-CMAT Telemetry Disconnected');

    // Clean up if the component unmounts
    return () => ws.close();
  }, [url, updateMetrics]);
}