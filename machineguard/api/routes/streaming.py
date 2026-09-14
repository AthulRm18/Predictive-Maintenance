"""WebSocket streaming routes and streaming status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from machineguard.streaming.websocket_handler import broadcaster

router = APIRouter()


@router.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """Stream all predictions in real-time via WebSocket."""
    await broadcaster.connect(websocket)
    try:
        while True:
            # Keep connection alive, wait for client disconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)


@router.websocket("/ws/predictions/{machine_id}")
async def websocket_machine_predictions(
    websocket: WebSocket, machine_id: str
):
    """Stream predictions for a specific machine via WebSocket."""
    await broadcaster.connect(websocket, machine_id=machine_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket, machine_id=machine_id)


@router.get("/api/v1/streaming/status", tags=["Streaming"])
async def streaming_status():
    """Get streaming system status."""
    return {
        "websocket_connections": broadcaster.connection_count,
        "status": "active" if broadcaster.connection_count > 0 else "idle",
    }
