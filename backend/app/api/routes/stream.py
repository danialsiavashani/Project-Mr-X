import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse

router = APIRouter(prefix="/stream", tags=["stream"])

_latest_frames: dict[str, bytes] = {}
_last_heartbeat: dict[str, float] = {}
VIEWER_TIMEOUT_SECONDS = 3


@router.post("/push/{camera_id}")
async def push_frame(camera_id: str, request: Request):
    frame_bytes = await request.body()
    if not frame_bytes:
        raise HTTPException(status_code=400, detail="Empty frame body")
    _latest_frames[camera_id] = frame_bytes
    return {"status": "ok"}


@router.post("/heartbeat/{camera_id}")
def heartbeat(camera_id: str):
    """Called by the viewing page's JS every second, only while it's open."""
    _last_heartbeat[camera_id] = time.time()
    return {"status": "ok"}


@router.get("/should-stream/{camera_id}")
def should_stream(camera_id: str):
    last = _last_heartbeat.get(camera_id, 0)
    return {"active": (time.time() - last) < VIEWER_TIMEOUT_SECONDS}


def mjpeg_generator(camera_id: str):
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


@router.get("/live/{camera_id}")
def live_stream(camera_id: str):
    return StreamingResponse(
        mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/view/{camera_id}", response_class=HTMLResponse)
def view_page(camera_id: str):
    """Temporary test page. Sends the heartbeat — this is the exact
    pattern the real Next.js dashboard will need to copy later."""
    return f"""
    <html><body style="background:#111;margin:0">
        <img src="/stream/live/{camera_id}" style="width:640px;max-width:100%">
        <script>
            setInterval(() => {{
                fetch('/stream/heartbeat/{camera_id}', {{ method: 'POST' }});
            }}, 1000);
        </script>
    </body></html>
    """