/**
 * public/mediaWorker.js
 *
 * Worker encodes audio WebM container as Data URL.
 *
 * Video is no longer taken from stitched WebM because rolling WebM stitching
 * can become invalid for PyAV video decoding.
 *
 * Main thread captures one JPEG frame from the live <video> element and passes
 * it as meta.videoFrameDataUrl.
 */

const WORKER_VERSION = 'mediaWorker-v4-audio-webm-video-jpeg';

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
    videoFrameDataUrl,
  } = meta || {};

  try {
    if (!sessionId) throw new Error('Missing sessionId.');
    if (seqId === undefined || seqId === null) throw new Error('Missing seqId.');

    const stitchedAudioVideoBlob = stitchBlobs(headerChunk, windowChunks);
    const audioDataUrl = blobToDataUrl(stitchedAudioVideoBlob);

    const safeVideoDataUrl =
      typeof videoFrameDataUrl === 'string' &&
      videoFrameDataUrl.startsWith('data:image/')
        ? videoFrameDataUrl
        : audioDataUrl;

    console.log('[H-CMAT Worker] chunk ready', {
      workerVersion: WORKER_VERSION,
      seqId,
      isFinalClip,
      audioBlobBytes: stitchedAudioVideoBlob.size,
      audioDataUrlPrefix: audioDataUrl.slice(0, 60),
      videoDataUrlPrefix: safeVideoDataUrl.slice(0, 60),
      windowChunkCount: windowChunks?.length ?? 0,
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

        // Audio still comes from WebM container.
        audio_data_base64: audioDataUrl,

        // Video is now one reliable JPEG frame.
        video_data_base64: safeVideoDataUrl,
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