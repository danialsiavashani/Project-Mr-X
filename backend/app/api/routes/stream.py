import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/stream", tags=["stream"])

_latest_frames: dict[str, bytes] = {}
_viewer_counts: dict[str, int] = {}


@router.post("/push/{camera_id}")
async def push_frame(camera_id: str, request: Request):
    frame_bytes = await request.body()
    if not frame_bytes:
        raise HTTPException(status_code=400, detail="Empty frame body")
    _latest_frames[camera_id] = frame_bytes
    return {"status": "ok"}


@router.get("/should-stream/{camera_id}")
def should_stream(camera_id: str):
    """The Pi polls this — only pushes frames while someone's actually watching."""
    return {"active": _viewer_counts.get(camera_id, 0) > 0}


def mjpeg_generator(camera_id: str):
    _viewer_counts[camera_id] = _viewer_counts.get(camera_id, 0) + 1
    try:
        while True:
            frame_bytes = _latest_frames.get(camera_id)
            if frame_bytes is None:
                time.sleep(0.1)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.1)
    finally:
        # Runs when the client disconnects (tab closed, navigated away) —
        # this is what makes the counter accurate, not just increment-only.
        _viewer_counts[camera_id] = max(0, _viewer_counts.get(camera_id, 1) - 1)


@router.get("/live/{camera_id}")
def live_stream(camera_id: str):
    return StreamingResponse(
        mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )