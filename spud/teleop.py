"""`spud teleop`: drive the arm yourself, from the keyboard or from your phone, and record demos.

Keyboard (in the MuJoCo window):
    arrows        move the gripper in x / y         PageUp / PageDown   up / down
    Home / End    rotate the gripper                Space               close / open
    Enter         save this episode, start a new    Delete              throw it away, start a new

Phone: `spud teleop --input phone` prints a link. Open it on a phone on the same Wi-Fi: two
thumb-sticks, a grip button, save / redo, and the robot's camera streamed to the page. If you
release the sticks, or the connection drops for 0.3 s, the arm stops (dead-man's switch).
The link carries a random token, so other people on the network can't drive your robot.
"""

from __future__ import annotations

import io
import json
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

# GLFW key codes
# (not Backspace: MuJoCo's viewer already uses it to reset the simulation)
ENTER, DELETE, RIGHT, LEFT, DOWN, UP, PGUP, PGDN, HOME, END, SPACE = 257, 261, 262, 263, 264, 265, 266, 267, 268, 269, 32


class KeyboardInput:
    """Each key press moves for a short burst; holding a key (OS key repeat) keeps it moving."""

    BURST = 0.15  # seconds a single press keeps moving

    def __init__(self):
        self.until = {}
        self.grip = -1.0
        self.event = None

    def on_key(self, key):
        axis = {UP: (0, 1), DOWN: (0, -1), LEFT: (1, 1), RIGHT: (1, -1), PGUP: (2, 1), PGDN: (2, -1),
                HOME: (3, 1), END: (3, -1)}
        if key in axis:
            self.until[axis[key]] = time.perf_counter() + self.BURST
        elif key == SPACE:
            self.grip = -self.grip
        elif key == ENTER:
            self.event = "save"
        elif key == DELETE:
            self.event = "discard"

    def command(self) -> np.ndarray:
        now = time.perf_counter()
        a = np.zeros(5)
        for (i, s), t in list(self.until.items()):
            if t > now:
                a[i] += s
        a[4] = self.grip
        return a

    def pop_event(self):
        e, self.event = self.event, None
        return e


class PhoneInput:
    """A tiny web server: the page posts stick positions ~20x per second and shows the camera."""

    DEADMAN = 0.3

    def __init__(self, port: int = 8765, host: str = "0.0.0.0"):
        self.token = secrets.token_urlsafe(8)
        self.cmd = np.zeros(5)
        self.cmd[4] = -1.0
        self.last = 0.0
        self.event = None
        self.frame = b""
        self.status = {"saved": 0, "episode_s": 0.0, "last": ""}
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _ok_token(self):
                q = parse_qs(urlparse(self.path).query)
                if q.get("t", [""])[0] != outer.token:
                    self.send_error(403)
                    return False
                return True

            def do_GET(self):
                if not self._ok_token():
                    return
                path = urlparse(self.path).path
                if path == "/":
                    body, ctype = PAGE.encode(), "text/html; charset=utf-8"
                elif path == "/frame.jpg":
                    with outer.lock:
                        body = outer.frame
                    ctype = "image/jpeg"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if not self._ok_token():
                    return
                n = int(self.headers.get("Content-Length", 0))
                try:
                    msg = json.loads(self.rfile.read(n) or b"{}")
                    with outer.lock:
                        outer.cmd = np.clip(np.array([msg.get(k, 0.0) for k in ("x", "y", "z", "yaw")] +
                                                     [1.0 if msg.get("grip") else -1.0], float), -1, 1)
                        outer.last = time.perf_counter()
                        if msg.get("event") in ("save", "discard"):
                            outer.event = msg["event"]
                        body = json.dumps(outer.status).encode()
                except (ValueError, TypeError):
                    self.send_error(400)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://{_lan_ip()}:{self.port}/?t={self.token}"

    def command(self) -> np.ndarray:
        with self.lock:
            a = self.cmd.copy()
            if time.perf_counter() - self.last > self.DEADMAN:
                a[:4] = 0.0  # dead-man's switch: no fresh packet, no motion
        return a

    def pop_event(self):
        with self.lock:
            e, self.event = self.event, None
        return e

    def push_frame(self, rgb: np.ndarray, quality: int = 70):
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        with self.lock:
            self.frame = buf.getvalue()

    def set_status(self, **kw):
        with self.lock:
            self.status.update(kw)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def run(env, device, record=None, window: bool = True, max_episodes: int | None = None, log=print,
        realtime: bool = True):
    """Real-time loop: input -> env.step at 50 Hz; Enter / Save writes the episode to `record`."""
    from .manip import DemoWriter

    writer = DemoWriter(record, env.meta) if record else None
    viewer = None
    if window:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data,
                                              key_callback=device.on_key if hasattr(device, "on_key") else None)
        viewer.cam.lookat[:] = [0.4, 0, 0.1]
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.3, -30, 160
    saved, ep = 0, 0
    obs = env.reset(seed=int(time.time()))
    O, A, R, D, S = [], [], [], [], []
    clock = time.perf_counter()
    try:
        while viewer is None or viewer.is_running():
            a = device.command()
            O.append(obs)
            S.append(env.state())
            obs, r, done, info = env.step(a)
            A.append(a)
            R.append(r)
            D.append(done)
            event = device.pop_event()
            if info["success"] and event is None:
                event = "save"  # placed it: keep it without making the operator press anything
                log("  placed! saving")
            if done and event is None:
                event = "discard"
            if event:
                if event == "save" and writer is not None and A:
                    D[-1] = True
                    writer.add(O, A, R, D, S, success=bool(info["success"]), operator=type(device).__name__)
                    saved += 1
                log(f"episode {ep}: {event} ({len(A) * env.DT:.1f} s); saved so far {saved}")
                ep += 1
                O, A, R, D, S = [], [], [], [], []
                obs = env.reset(seed=int(time.time() * 1000) % 2**31)
                if max_episodes and ep >= max_episodes:
                    break
            if hasattr(device, "push_frame") and len(A) % 3 == 0:
                device.push_frame(env.render("front", 320, 240))
                device.set_status(saved=saved, episode_s=round(len(A) * env.DT, 1))
            if viewer is not None:
                viewer.sync()
            clock += env.DT
            if realtime:
                time.sleep(max(0.0, clock - time.perf_counter()))
            if time.perf_counter() - clock > 0.5:
                clock = time.perf_counter()  # fell behind (slow machine): resync instead of bursting
    finally:
        if viewer is not None:
            viewer.close()
        if writer is not None:
            writer.close()
    return saved


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>spud teleop</title>
<style>
:root{--bg:#0e0c18;--fg:#ece9ff;--dim:#8f88b8;--acc:#9b6bff;--ok:#29d39a;--bad:#ff5c8a}
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;touch-action:none}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
#cam{width:100%;max-height:42vh;object-fit:contain;background:#000}
#bar{display:flex;gap:8px;padding:8px 12px;align-items:center;justify-content:space-between}
#stat{color:var(--dim);font-variant-numeric:tabular-nums}
button{flex:1;border:0;border-radius:12px;padding:14px 0;font-size:16px;font-weight:600;color:var(--fg);background:#241f3d}
#grip.on{background:var(--acc)} #save{background:var(--ok);color:#06251a} #redo{background:var(--bad);color:#2a0612}
#pads{flex:1;display:flex;gap:12px;padding:12px}
.pad{flex:1;border-radius:50%;background:radial-gradient(#211c38,#151126);position:relative;aspect-ratio:1;align-self:center;border:1px solid #2c2650}
.knob{position:absolute;width:30%;height:30%;left:35%;top:35%;border-radius:50%;background:var(--acc);opacity:.85}
.lbl{position:absolute;width:100%;bottom:-22px;text-align:center;color:var(--dim);font-size:12px}
</style></head><body>
<img id="cam" alt="robot camera">
<div id="bar"><span id="stat">connecting...</span></div>
<div id="bar"><button id="grip">grip</button><button id="save">save</button><button id="redo">redo</button></div>
<div id="pads"><div class="pad" id="L"><div class="knob"></div><div class="lbl">move x / y</div></div>
<div class="pad" id="R"><div class="knob"></div><div class="lbl">up-down / rotate</div></div></div>
<script>
const T=new URLSearchParams(location.search).get('t');
const st={x:0,y:0,z:0,yaw:0,grip:false,event:null};
function pad(id,fn){const el=document.getElementById(id),k=el.querySelector('.knob');let pid=null;
 function set(e){const r=el.getBoundingClientRect();let dx=(e.clientX-r.left)/r.width*2-1,dy=(e.clientY-r.top)/r.height*2-1;
  const m=Math.hypot(dx,dy);if(m>1){dx/=m;dy/=m}k.style.left=(35+dx*35)+'%';k.style.top=(35+dy*35)+'%';fn(dx,dy)}
 function end(){pid=null;k.style.left='35%';k.style.top='35%';fn(0,0)}
 el.addEventListener('pointerdown',e=>{pid=e.pointerId;el.setPointerCapture(pid);set(e)});
 el.addEventListener('pointermove',e=>{if(e.pointerId===pid)set(e)});
 el.addEventListener('pointerup',end);el.addEventListener('pointercancel',end)}
// The robot faces the camera: pushing the stick up moves the gripper away from you (-x in the front view).
pad('L',(dx,dy)=>{st.x=dy;st.y=dx});
pad('R',(dx,dy)=>{st.z=-dy;st.yaw=-dx});
const g=document.getElementById('grip');g.onclick=()=>{st.grip=!st.grip;g.classList.toggle('on',st.grip)};
document.getElementById('save').onclick=()=>st.event='save';
document.getElementById('redo').onclick=()=>st.event='discard';
async function tick(){try{const r=await fetch('/cmd?t='+T,{method:'POST',body:JSON.stringify(st)});st.event=null;
 const s=await r.json();document.getElementById('stat').textContent='saved '+s.saved+' demos  |  this one '+s.episode_s+' s'}
 catch(e){document.getElementById('stat').textContent='connection lost, retrying...'}setTimeout(tick,50)}
tick();
const cam=document.getElementById('cam');function frame(){const i=new Image();i.onload=()=>{cam.src=i.src;setTimeout(frame,80)};
 i.onerror=()=>setTimeout(frame,500);i.src='/frame.jpg?t='+T+'&n='+Date.now()}frame();
</script></body></html>"""
