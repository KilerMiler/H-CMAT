"""
api/routes/stream.py

WebSocket endpoint for the live H-CMAT inference loop.

WS /api/v1/stream/{session_id}

All real-time traffic flows through this single connection:
    Frontend → Backend:  chunk, heartbeat
    Backend → Frontend:  matrix_update, fusion_result, attention_tick, error

This version includes the safety fix:
    Dropping late fusion results if the session was already terminated
    (prevents require() KeyError crashes).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from config.logging import get_logger
from core.nms_ledger import apply_nms
from core.schemas import (
    AttentionTickResponse,
    FuseRequest,
    MatrixUpdateResponse,
    WSMessage,
    WSMessageType,
)

logger = get_logger(__name__)
router = APIRouter()


async def _send(ws: WebSocket, msg_type: WSMessageType, payload: Any) -> None:
    try:
        if ws.client_state != WebSocketState.CONNECTED:
            return
        envelope = WSMessage(type=msg_type, payload=payload)
        await ws.send_text(envelope.model_dump_json())
    except Exception as exc:
        logger.warning(f"WebSocket send failed ({msg_type}): {exc}")


async def _send_error(
    ws: WebSocket,
    code: str,
    message: str,
    seq_id: int | None = None,
) -> None:
    await _send(
        ws,
        WSMessageType.ERROR,
        {
            "code": code,
            "message": message,
            "seq_id": seq_id,
        },
    )


@router.websocket("/stream/{session_id}")
async def stream_endpoint(websocket: WebSocket, session_id: str):
    system = websocket.app.state.system
    session_manager = system["session_manager"]
    runner = system["runner"]
    fusion_layer = system["brain"]
    mapper = system["mapper"]

    # Validate session exists
    if session_manager.get(session_id) is None:
        await websocket.close(code=4004, reason=f"Session '{session_id}' not found.")
        logger.warning(f"WS rejected — unknown session: {session_id}")
        return

    await websocket.accept()
    logger.info(f"[{session_id}] WebSocket connected.")

    try:
        while True:
            raw = await websocket.receive_text()

            # Parse JSON
            try:
                data = json.loads(raw)
                msg_type = data.get("type")
                payload = data.get("payload", {})
            except json.JSONDecodeError:
                await _send_error(websocket, "PARSE_ERROR", "Invalid JSON frame.")
                continue

            # Heartbeat
            if msg_type == WSMessageType.HEARTBEAT:
                continue

            # Main message: CHUNK
            if msg_type == WSMessageType.CHUNK:
                pipeline_start = time.time()

                # Check session still exists (prevent late crashes)
                active_session = session_manager.get(session_id)
                if active_session is None:
                    logger.warning(
                        f"[{session_id}] Received chunk after session termination — dropping."
                    )
                    break

                # Validate FuseRequest schema
                try:
                    fuse_req = FuseRequest(**payload)
                except Exception as exc:
                    await _send_error(
                        websocket,
                        "SCHEMA_ERROR",
                        f"FuseRequest validation failed: {exc}",
                        payload.get("seq_id"),
                    )
                    continue

                # session_id mismatch
                if fuse_req.session_id != session_id:
                    await _send_error(
                        websocket,
                        "SESSION_MISMATCH",
                        (
                            f"Payload session_id '{fuse_req.session_id}' does not "
                            f"match WebSocket session_id '{session_id}'."
                        ),
                        fuse_req.seq_id,
                    )
                    continue

                # Culture validation
                if not mapper.has_id(fuse_req.culture_id):
                    await _send_error(
                        websocket,
                        "UNKNOWN_CULTURE",
                        (
                            f"Cultural profile {fuse_req.culture_id} not found. "
                            f"Available: {mapper.available_ids()}"
                        ),
                        fuse_req.seq_id,
                    )
                    continue

                # Sequence validity check
                try:
                    if not session_manager.is_seq_valid(session_id, fuse_req.seq_id):
                        continue
                except KeyError:
                    logger.warning(
                        f"[{session_id}] Session terminated during seq check — dropping chunk."
                    )
                    break

                # Update text history
                try:
                    session_manager.update_text_history(
                        session_id, fuse_req.text_history
                    )
                except KeyError:
                    logger.warning(
                        f"[{session_id}] Session terminated during text update — dropping chunk."
                    )
                    break

                # Sync culture ID
                active_session = session_manager.get(session_id)
                if active_session is None:
                    logger.warning(
                        f"[{session_id}] Session terminated before culture sync — dropping chunk."
                    )
                    break

                active_session.culture_id = fuse_req.culture_id

                text_input = fuse_req.text_history[-1] if fuse_req.text_history else ""
                culture_weights = mapper.get_weights_for_fusion(fuse_req.culture_id)

                # Run encoders
                try:
                    encoder_outputs = await runner.run_all_async(
                        text_input=text_input,
                        audio_base64=fuse_req.audio_data_base64,
                        video_base64=fuse_req.video_data_base64,
                    )
                except Exception as exc:
                    logger.error(f"[{session_id}] Encoder crashed: {exc}", exc_info=True)
                    await _send_error(
                        websocket,
                        "INFERENCE_ERROR",
                        f"Encoder failure: {exc}",
                        fuse_req.seq_id,
                    )
                    continue

                # Session may be deleted during slow encoder
                active_session = session_manager.get(session_id)
                if active_session is None:
                    logger.warning(
                        f"[{session_id}] Session terminated before fusion — dropping late result."
                    )
                    break

                # Run fusion
                try:
                    fuse_response = fusion_layer.fuse(
                        encoder_outputs=encoder_outputs,
                        culture_weights=culture_weights,
                        seq_id=fuse_req.seq_id,
                        temporal_context=fuse_req.temporal_context,
                        pipeline_start=pipeline_start,
                    )
                except Exception as exc:
                    logger.error(f"[{session_id}] Fusion crashed: {exc}", exc_info=True)
                    await _send_error(
                        websocket,
                        "FUSION_ERROR",
                        f"Fusion failure: {exc}",
                        fuse_req.seq_id,
                    )
                    continue

                # Session may terminate right after fusion
                active_session = session_manager.get(session_id)
                if active_session is None:
                    logger.warning(
                        f"[{session_id}] Session terminated after fusion — dropping late result."
                    )
                    break

                # Send MATRIX_UPDATE
                await _send(
                    websocket,
                    WSMessageType.MATRIX_UPDATE,
                    MatrixUpdateResponse(
                        seq_id=fuse_req.seq_id,
                        modality_matrix=fuse_response.modality_matrix,
                    ).model_dump(),
                )

                # Send ATTENTION_TICK
                weights = {k: v.weight for k, v in fuse_response.modality_matrix.items()}
                cross = {}
                keys = list(weights.keys())
                for i in range(len(keys)):
                    for j in range(len(keys)):
                        if i != j:
                            cross[f"{keys[i]}_to_{keys[j]}"] = round(
                                weights[keys[i]] * weights[keys[j]], 4
                            )

                await _send(
                    websocket,
                    WSMessageType.ATTENTION_TICK,
                    AttentionTickResponse(
                        ts=int(time.time() * 1000),
                        weights=cross,
                    ).model_dump(),
                )

                # SAFE NMS:
                # ---------------------------------------
                active_session = session_manager.get(session_id)
                if active_session is None:
                    logger.warning(
                        f"[{session_id}] Session terminated before NMS — dropping late result."
                    )
                    break

                apply_nms(active_session, fuse_response)
                # ---------------------------------------

                # Send FUSION_RESULT
                await _send(
                    websocket,
                    WSMessageType.FUSION_RESULT,
                    fuse_response.model_dump(),
                )

                logger.info(
                    f"[{session_id}] seq={fuse_req.seq_id} "
                    f"intent='{fuse_response.holistic_fusion.primary_intent}' "
                    f"new={fuse_response.holistic_fusion.is_new_event} "
                    f"replace={fuse_response.holistic_fusion.replaces_seq_id} "
                    f"total={fuse_response.total_latency_ms}ms"
                )

                # Final clip
                if fuse_req.temporal_context.is_final_clip:
                    active_session = session_manager.get(session_id)
                    if active_session:
                        active_session.status = "summarize_ready"
                        logger.info(
                            f"[{session_id}] Final clip received — ready for /summarize."
                        )
                    else:
                        logger.warning(
                            f"[{session_id}] Final clip arrived after session termination."
                        )
                    break

                continue

            # Unknown message
            await _send_error(
                websocket,
                "UNKNOWN_TYPE",
                f"Unknown message type '{msg_type}'. Expected: chunk | heartbeat",
            )

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected.")

    except Exception as exc:
        logger.error(f"[{session_id}] WebSocket crashed: {exc}", exc_info=True)
        try:
            await _send_error(websocket, "SERVER_ERROR", str(exc))
        except Exception:
            pass

    finally:
        logger.info(f"[{session_id}] WebSocket handler exiting.")