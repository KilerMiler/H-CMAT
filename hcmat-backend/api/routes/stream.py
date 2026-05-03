"""
api/routes/stream.py

WebSocket endpoint for the live H-CMAT inference loop.

WS /api/v1/stream/{session_id}

All real-time traffic flows through this single connection:
    Frontend → Backend:  chunk, heartbeat
    Backend → Frontend:  matrix_update, fusion_result, attention_tick, error

Current demo behavior:
    The frontend may switch culture live by sending a different valid culture_id
    in each FuseRequest. The backend strictly validates that culture_id before
    applying its weights. Invalid IDs are rejected with a WebSocket error.
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
    """
    Serialises a WSMessage envelope and sends it over the WebSocket.
    Send errors are swallowed so stale connections do not crash inference.
    """
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
        {"code": code, "message": message, "seq_id": seq_id},
    )


@router.websocket("/stream/{session_id}")
async def stream_endpoint(websocket: WebSocket, session_id: str):
    """
    Persistent bidirectional WebSocket for the entire session duration.

    Opened after POST /api/v1/inference/session returns a valid session_id.
    Closed when the user clicks Stop or when the browser connection drops.
    """
    system = websocket.app.state.system
    session_manager = system["session_manager"]
    runner = system["runner"]
    fusion_layer = system["brain"]
    mapper = system["mapper"]

    session = session_manager.get(session_id)
    if session is None:
        await websocket.close(code=4004, reason=f"Session '{session_id}' not found.")
        logger.warning(f"WS rejected — unknown session: {session_id}")
        return

    await websocket.accept()
    logger.info(f"[{session_id}] WebSocket connected.")

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
                msg_type = data.get("type")
                payload = data.get("payload", {})
            except json.JSONDecodeError:
                await _send_error(websocket, "PARSE_ERROR", "Invalid JSON frame.")
                continue

            if msg_type == WSMessageType.HEARTBEAT:
                logger.debug(f"[{session_id}] Heartbeat received.")
                continue

            if msg_type == WSMessageType.CHUNK:
                pipeline_start = time.time()

                try:
                    fuse_req = FuseRequest(**payload)
                except Exception as exc:
                    await _send_error(
                        websocket,
                        "SCHEMA_ERROR",
                        f"FuseRequest validation failed: {exc}",
                        seq_id=payload.get("seq_id"),
                    )
                    continue

                if fuse_req.session_id != session_id:
                    await _send_error(
                        websocket,
                        "SESSION_MISMATCH",
                        (
                            f"Payload session_id '{fuse_req.session_id}' does not "
                            f"match path session_id '{session_id}'."
                        ),
                        seq_id=fuse_req.seq_id,
                    )
                    continue

                # Strict culture validation.
                # This keeps live culture switching possible while preventing
                # silent fallback to the wrong profile.
                if not mapper.has_id(fuse_req.culture_id):
                    await _send_error(
                        websocket,
                        "UNKNOWN_CULTURE",
                        (
                            f"Cultural profile {fuse_req.culture_id} not found. "
                            f"Available IDs: {mapper.available_ids()}"
                        ),
                        seq_id=fuse_req.seq_id,
                    )
                    continue

                if not session_manager.is_seq_valid(session_id, fuse_req.seq_id):
                    continue

                session_manager.update_text_history(session_id, fuse_req.text_history)

                # Keep session metadata synced with live frontend culture selection.
                # This is useful for the final SQLite log.
                session = session_manager.require(session_id)
                session.culture_id = fuse_req.culture_id

                text_input = fuse_req.text_history[-1] if fuse_req.text_history else ""
                culture_weights = mapper.get_weights_for_fusion(fuse_req.culture_id)

                try:
                    encoder_outputs = await runner.run_all_async(
                        text_input=text_input,
                        audio_base64=fuse_req.audio_data_base64,
                        video_base64=fuse_req.video_data_base64,
                    )
                except Exception as exc:
                    logger.error(
                        f"[{session_id}] Encoder stage crashed: {exc}",
                        exc_info=True,
                    )
                    await _send_error(
                        websocket,
                        "INFERENCE_ERROR",
                        f"H-CMAT encoder pipeline error: {exc}",
                        seq_id=fuse_req.seq_id,
                    )
                    continue

                try:
                    fuse_response = fusion_layer.fuse(
                        encoder_outputs=encoder_outputs,
                        culture_weights=culture_weights,
                        seq_id=fuse_req.seq_id,
                        temporal_context=fuse_req.temporal_context,
                        pipeline_start=pipeline_start,
                    )
                except Exception as exc:
                    logger.error(
                        f"[{session_id}] Fusion crashed: {exc}",
                        exc_info=True,
                    )
                    await _send_error(
                        websocket,
                        "FUSION_ERROR",
                        f"H-CMAT fusion error: {exc}",
                        seq_id=fuse_req.seq_id,
                    )
                    continue

                matrix_update = MatrixUpdateResponse(
                    seq_id=fuse_req.seq_id,
                    modality_matrix=fuse_response.modality_matrix,
                )
                await _send(
                    websocket,
                    WSMessageType.MATRIX_UPDATE,
                    matrix_update.model_dump(),
                )

                weights = {
                    k: v.weight for k, v in fuse_response.modality_matrix.items()
                }
                keys = list(weights.keys())
                cross_weights: Dict[str, float] = {}

                for i in range(len(keys)):
                    for j in range(len(keys)):
                        if i != j:
                            label = f"{keys[i]}_to_{keys[j]}"
                            cross_weights[label] = round(
                                weights[keys[i]] * weights[keys[j]],
                                4,
                            )

                attention_tick = AttentionTickResponse(
                    ts=int(time.time() * 1000),
                    weights=cross_weights,
                )
                await _send(
                    websocket,
                    WSMessageType.ATTENTION_TICK,
                    attention_tick.model_dump(),
                )

                apply_nms(session_manager.require(session_id), fuse_response)

                await _send(
                    websocket,
                    WSMessageType.FUSION_RESULT,
                    fuse_response.model_dump(),
                )

                logger.info(
                    f"[{session_id}] seq={fuse_req.seq_id} "
                    f"culture={fuse_req.culture_id} "
                    f"intent='{fuse_response.holistic_fusion.primary_intent}' "
                    f"is_new={fuse_response.holistic_fusion.is_new_event} "
                    f"replace_seq={fuse_response.holistic_fusion.replaces_seq_id} "
                    f"total={fuse_response.total_latency_ms}ms"
                )

                if fuse_req.temporal_context.is_final_clip:
                    session_manager.require(session_id).status = "summarize_ready"
                    logger.info(
                        f"[{session_id}] Final clip received. "
                        "Session ready for POST /summarize."
                    )

                continue

            logger.warning(f"[{session_id}] Unknown message type: {msg_type}")
            await _send_error(
                websocket,
                "UNKNOWN_TYPE",
                f"Unknown message type '{msg_type}'. Expected: chunk | heartbeat",
            )

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected by client.")

    except Exception as exc:
        logger.error(f"[{session_id}] WebSocket crashed: {exc}", exc_info=True)
        try:
            await _send_error(websocket, "SERVER_ERROR", str(exc))
        except Exception:
            pass

    finally:
        logger.info(f"[{session_id}] WebSocket handler exiting.")