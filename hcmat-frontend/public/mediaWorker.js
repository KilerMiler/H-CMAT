/**
 * public/mediaWorker.js
 *
 * H-CMAT media encoding worker.
 *
 * Responsibility:
 *   - Receive WebM header chunk + rolling WebM chunks from main thread.
 *   - Stitch them into one WebM Blob.
 *   - Convert stitched Blob to FULL Data URL.
 *   - Send prepared WebSocket chunk payload back to main thread.
 *
 * Important:
 *   The backend expects either:
 *     1. Full Data URL:
 *        data:video/webm;codecs=vp8,opus;base64,AAAA...
 *
 *     2. Raw base64:
 *        AAAA...
 *
 *   We intentionally send FULL Data URL because it is easier to debug
 *   and backend can safely strip the prefix.
 *
 *   The same WebM container carries both video and audio tracks, so the same
 *   Data URL is sent as:
 *     - video_data_base64
 *     - audio_data_base64
 */

const WORKER_VERSION = 'mediaWorker-v3-dataurl-safe';

function blobToDataUrl(blob) {
  const reader = new FileReaderSync();
  return reader.readAsDataURL(blob);
}

function stitchBlobs(headerChunk, windowChunks) {
  const validChunks = (windowChunks || [])
    .map((c) => c?.blob)
    .filter((blob) => blob && blob.size > 0);

  if (!headerChunk || headerChunk.size === 0) {
    throw new Error('Missing or empty WebM header chunk.');
  }

  if (validChunks.length === 0) {
    throw new Error('No valid rolling WebM chunks available.');
  }

  const blobType = headerChunk.type || validChunks[0]?.type || 'video/webm';

  return new Blob([headerChunk, ...validChunks], {
    type: blobType,
  });
}

self.onmessage = (event) => {
  const { type, payload } = event.data || {};

  if (type !== 'SEND_CHUNK') return;

  const { headerChunk, windowChunks, meta } = payload || {};

  const {
    sessionId,
    seqId,
    clipStartMs,
    clipEndMs,
    isFinalClip,
    textHistory,
    cultureId,
  } = meta || {};

  try {
    if (!sessionId) {
      throw new Error('Missing sessionId.');
    }

    if (seqId === undefined || seqId === null) {
      throw new Error('Missing seqId.');
    }

    if (!headerChunk) {
      throw new Error('Missing WebM header chunk.');
    }

    if (!windowChunks || windowChunks.length === 0) {
      throw new Error('No rolling window chunks available.');
    }

    const stitchedBlob = stitchBlobs(headerChunk, windowChunks);
    const mediaDataUrl = blobToDataUrl(stitchedBlob);

    if (
      typeof mediaDataUrl !== 'string' ||
      !mediaDataUrl.startsWith('data:') ||
      !mediaDataUrl.includes('base64,')
    ) {
      throw new Error('Generated media Data URL is invalid.');
    }

    // Useful browser-side diagnostics.
    // Check DevTools Console for these.
    console.log('[H-CMAT Worker] chunk ready', {
      workerVersion: WORKER_VERSION,
      seqId,
      isFinalClip,
      stitchedSizeBytes: stitchedBlob.size,
      blobType: stitchedBlob.type,
      dataUrlPrefix: mediaDataUrl.slice(0, 80),
      dataUrlLength: mediaDataUrl.length,
      windowChunkCount: windowChunks.length,
    });

    const chunkPayload = {
      type: 'chunk',
      payload: {
        session_id: sessionId,
        seq_id: seqId,
        temporal_context: {
          clip_start_ms: Math.max(0, Number(clipStartMs) || 0),
          clip_end_ms: Math.max(1, Number(clipEndMs) || 1),
          is_final_clip: Boolean(isFinalClip),
        },
        culture_id: Number(cultureId) || 1,
        text_history: Array.isArray(textHistory) ? textHistory : [],

        // Same WebM container includes audio + video.
        // Backend decodes video frames from video_data_base64 and audio waveform
        // from audio_data_base64 using PyAV.
        audio_data_base64: mediaDataUrl,
        video_data_base64: mediaDataUrl,
      },
    };

    self.postMessage({
      type: 'CHUNK_READY',
      payload: {
        seqId,
        isFinalClip: Boolean(isFinalClip),
        chunkPayload,
      },
    });
  } catch (error) {
    console.error('[H-CMAT Worker] chunk error', {
      workerVersion: WORKER_VERSION,
      seqId,
      error: error?.message || String(error),
    });

    self.postMessage({
      type: 'CHUNK_ERROR',
      payload: {
        seqId,
        error: error?.message || String(error),
      },
    });
  }
};