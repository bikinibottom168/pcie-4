import time, threading, subprocess
from typing import List, Tuple, Optional
import numpy as np
import cv2
import sounddevice as sd

# Windows named pipe
import win32pipe, win32file, pywintypes

# ====================== CONFIG ======================
RTMP_URL = "rtmp://51.79.231.72:1935/xstream/test-siam-1"  # <-- เปลี่ยนปลายทางของคุณ
FFMPEG   = "ffmpeg"                                  # หรือระบุ path เต็ม

# วิดีโอ: ดัชนีอุปกรณ์ (DirectShow) ของ capture 4 ช่อง
DEVICE_INDEXES = [0, 1, 2, 3]

# วิธีเลือกอุปกรณ์เสียงให้ตรงแหล่งวิดีโอ:
# -- โหมด A: จับคู่ด้วย "คำค้นชื่ออุปกรณ์" (แนะนำ) —
#    ใส่ keyword ต่อ source (ซ้ายบน -> ขวาบน -> ซ้ายล่าง -> ขวาล่าง)
#    ตัวอย่าง: Blackmagic/AVerMedia/Magewell/USB Capture ...
AUDIO_NAME_KEYWORDS: List[List[str]] = [
    ["AVerMedia", "HD Pro", "Card1"],  # source0
    ["AVerMedia", "HD Pro", "Card2"],  # source1
    ["AVerMedia", "HD Pro", "Card3"],  # source2
    ["AVerMedia", "HD Pro", "Card4"],  # source3
]

# -- โหมด B: ถ้าไม่อยากใช้ชื่อ ให้ “ล็อก index ตายตัว” ตามลำดับแหล่งวิดีโอ
#    (ปล่อยเป็น [] ถ้าจะใช้โหมด A)
AUDIO_DEVICE_INDEXES: List[int] = []  # เช่น [3, 5, 7, 9]

# ขนาด/เฟรมเรต (ให้ทุกช่องเท่ากัน)
WIDTH, HEIGHT = 1920, 1080
FPS = 30
BITRATE = "2000k"
X264_PRESET = "slow"
GOP_SECONDS = 2

# การสลับและตรวจ motion
SWITCH_SECONDS = 120
MOTION_CHECK_FRAMES = 3
MOTION_CHECK_INTERVAL = 0.4
DIFF_THRESHOLD = 2.0        # % diff >= นี้ = มีการเคลื่อนไหว
MIN_BRIGHTNESS = 5.0        # % สว่างเฉลี่ยขั้นต่ำ

# พรีวิว
WINDOW_NAME = "Preview (2x2) — green = ACTIVE"
PREVIEW_FPS = 15
BORDER_THICK = 6

# เสียง
AUDIO_SR = 48000
AUDIO_CH = 2
AUDIO_BLOCK = 1024           # samples/CH ต่อครั้ง (ลดเพื่อลดดีเลย์)
AUDIO_FADE_MS = 120          # cross-fade เบาๆ ตอนสลับ (ป้องกันป๊อป)
# ====================================================

def ensure_size(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    if frame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if frame.shape[1] != w or frame.shape[0] != h:
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return frame

def frame_diff_percent(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
    return (diff / 255.0) * 100.0

def brightness_percent(a: np.ndarray) -> float:
    return float(a.mean() / 255.0 * 100.0)

class CaptureReader(threading.Thread):
    def __init__(self, device_index: int, w: int, h: int, fps: int):
        super().__init__(daemon=True)
        self.idx = device_index
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.w, self.h, self.fps = w, h, fps
        self.latest_lock = threading.Lock()
        self.latest: Optional[np.ndarray] = None
        self.running = True

    def run(self):
        interval = 1.0 / max(5, min(self.fps, 30))
        while self.running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                frame = ensure_size(frame, self.w, self.h)
                with self.latest_lock:
                    self.latest = frame
            else:
                with self.latest_lock:
                    self.latest = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            time.sleep(interval)

    def get_latest(self) -> np.ndarray:
        with self.latest_lock:
            if self.latest is None:
                return np.zeros((self.h, self.w, 3), dtype=np.uint8)
            return self.latest.copy()

    def close(self):
        self.running = False
        try: self.cap.release()
        except Exception: pass

def check_motion_from_reader(reader: CaptureReader) -> Tuple[bool, float, float]:
    frames = []
    for _ in range(MOTION_CHECK_FRAMES):
        frames.append(reader.get_latest())
        time.sleep(MOTION_CHECK_INTERVAL)
    avg_bright = float(np.mean([brightness_percent(f) for f in frames]))
    if avg_bright < MIN_BRIGHTNESS: return (False, 0.0, avg_bright)
    diffs = [frame_diff_percent(frames[i], frames[i+1]) for i in range(len(frames)-1)]
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return (avg_diff >= DIFF_THRESHOLD, avg_diff, avg_bright)

def make_preview_grid(frames: List[np.ndarray], active_idx: int) -> np.ndarray:
    assert len(frames) == 4
    h, w = frames[0].shape[:2]
    grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    grid[0:h, 0:w]     = frames[0]
    grid[0:h, w:2*w]   = frames[1]
    grid[h:2*h, 0:w]   = frames[2]
    grid[h:2*h, w:2*w] = frames[3]
    pos_map = {0:(0,0), 1:(0,w), 2:(h,0), 3:(h,w)}
    y, x = pos_map.get(active_idx, (0,0))
    cv2.rectangle(grid, (x, y), (x+w, y+h), (0,255,0), BORDER_THICK)
    cv2.putText(grid, f"ACTIVE #{active_idx}", (x+18, y+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    return grid

# ---------- Named Pipes (Windows) ----------
def create_connected_pipe(name: str):
    full = r"\\.\pipe\{}".format(name)
    handle = win32pipe.CreateNamedPipe(
        full,
        win32pipe.PIPE_ACCESS_OUTBOUND,
        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
        1, 1024*1024, 1024*1024, 0, None
    )
    win32pipe.ConnectNamedPipe(handle, None)
    return handle

def pipe_write(handle, data: bytes):
    win32file.WriteFile(handle, data)

# ---------- FFmpeg ----------
def launch_ffmpeg_with_pipes():
    video_pipe = r"\\.\pipe\video_pipe"
    audio_pipe = r"\\.\pipe\audio_pipe"
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", video_pipe,
        "-f", "s16le", "-ar", str(AUDIO_SR), "-ac", str(AUDIO_CH), "-i", audio_pipe,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", X264_PRESET, "-tune", "zerolatency",
        "-b:v", BITRATE, "-maxrate", BITRATE, "-bufsize", BITRATE,
        "-g", str(FPS * GOP_SECONDS), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", str(AUDIO_SR),
        "-f", "flv", RTMP_URL
    ]
    return subprocess.Popen(cmd)

# ---------- Audio helpers ----------
def list_audio_inputs():
    devs = sd.query_devices()
    return [(i, d["name"], d["max_input_channels"]) for i, d in enumerate(devs) if d["max_input_channels"] > 0]

def auto_pick_audio_index_for_source(src_idx: int) -> Optional[int]:
    """
    เลือกอุปกรณ์เสียงโดยหา keyword ตาม AUDIO_NAME_KEYWORDS[src_idx]
    ถ้าไม่พบคืน None
    """
    keywords = AUDIO_NAME_KEYWORDS[src_idx] if src_idx < len(AUDIO_NAME_KEYWORDS) else []
    if not keywords: return None
    candidates = list_audio_inputs()
    for i, name, ch in candidates:
        low = name.lower()
        if all(k.lower() in low for k in keywords if k):  # match ทุกคำ
            return i
        # ผ่อนปรน: match คำใดคำหนึ่งก็ได้
        if any(k.lower() in low for k in keywords if k):
            return i
    return None

class AudioIO:
    """จัดการสตรีมเสียงเข้า (จากอุปกรณ์) -> เขียนลง named pipe พร้อม fade ตอนสลับ"""
    def __init__(self, pipe_handle):
        self.pipe = pipe_handle
        self.stream: Optional[sd.InputStream] = None
        self.current_dev: Optional[int] = None

    def start_with_device(self, dev_index: int):
        self.stop()
        self.stream = sd.InputStream(device=dev_index, samplerate=AUDIO_SR,
                                     channels=AUDIO_CH, dtype='int16',
                                     blocksize=AUDIO_BLOCK)
        self.stream.start()
        self.current_dev = dev_index

    def stop(self):
        if self.stream is not None:
            try: self.stream.stop()
            except Exception: pass
            try: self.stream.close()
            except Exception: pass
            self.stream = None
            self.current_dev = None

    def write_once(self):
        """อ่านหนึ่งบล็อกจาก input ปัจจุบัน (หรือ silence) แล้วเขียนลง pipe"""
        if self.stream is None:
            silence = (np.zeros((AUDIO_BLOCK, AUDIO_CH), dtype=np.int16)).tobytes()
            pipe_write(self.pipe, silence)
            return
        try:
            block, _ = self.stream.read(AUDIO_BLOCK)
            pipe_write(self.pipe, block.tobytes())
        except sd.PortAudioError:
            silence = (np.zeros((AUDIO_BLOCK, AUDIO_CH), dtype=np.int16)).tobytes()
            pipe_write(self.pipe, silence)

    def crossfade_to_device(self, new_dev_index: int, fade_ms=AUDIO_FADE_MS):
        """ทำ fade-out → สลับ input → fade-in เพื่อกันป๊อปเสียง"""
        # fade-out
        steps = max(2, int((fade_ms/1000.0) * (AUDIO_SR / AUDIO_BLOCK)))
        for s in range(steps):
            scale = 1.0 - (s+1)/steps
            if self.stream is None:
                silence = (np.zeros((AUDIO_BLOCK, AUDIO_CH), dtype=np.int16)).tobytes()
                pipe_write(self.pipe, silence)
            else:
                try:
                    block, _ = self.stream.read(AUDIO_BLOCK)
                    block = (block.astype(np.int32) * scale).astype(np.int16)
                    pipe_write(self.pipe, block.tobytes())
                except sd.PortAudioError:
                    silence = (np.zeros((AUDIO_BLOCK, AUDIO_CH), dtype=np.int16)).tobytes()
                    pipe_write(self.pipe, silence)

        # switch
        self.start_with_device(new_dev_index)

        # fade-in
        for s in range(steps):
            scale = (s+1)/steps
            try:
                block, _ = self.stream.read(AUDIO_BLOCK)
                block = (block.astype(np.int32) * scale).astype(np.int16)
                pipe_write(self.pipe, block.tobytes())
            except sd.PortAudioError:
                silence = (np.zeros((AUDIO_BLOCK, AUDIO_CH), dtype=np.int16)).tobytes()
                pipe_write(self.pipe, silence)

# ----------------------------------------------------

def main():
    # ตั้ง readers วิดีโอ
    readers = [CaptureReader(idx, WIDTH, HEIGHT, FPS) for idx in DEVICE_INDEXES]
    for r in readers: r.start()
    time.sleep(1.0)

    # เปิด ffmpeg + ต่อท่อ
    ff = launch_ffmpeg_with_pipes()
    print("Waiting ffmpeg to connect pipes ...")
    vid_handle = create_connected_pipe("video_pipe")
    aud_handle = create_connected_pipe("audio_pipe")
    print("Pipes connected.")

    # เลือก active เริ่มต้นเป็นตัวแรกที่ "ขยับ"
    active = 0
    for _ in range(len(readers)):
        ok, diff, bright = check_motion_from_reader(readers[active])
        print(f"Init VIDEO dev#{DEVICE_INDEXES[active]} moving={ok} diff={diff:.2f}% bright={bright:.2f}%")
        if ok: break
        active = (active + 1) % len(readers)

    # เตรียม audio I/O
    audio = AudioIO(aud_handle)

    # หาอุปกรณ์เสียงเริ่มต้น
    if AUDIO_DEVICE_INDEXES:
        start_audio_dev = AUDIO_DEVICE_INDEXES[active]
    else:
        picked = auto_pick_audio_index_for_source(active)
        if picked is None:
            print("WARN: ไม่พบ audio โดยชื่ออุปกรณ์ → จะเงียบจนกว่าคุณจะตั้งค่า AUDIO_NAME_KEYWORDS หรือ AUDIO_DEVICE_INDEXES")
            picked = None
    if picked is not None:
        audio.start_with_device(picked)

    last_switch = time.time()
    out_frame_interval = 1.0 / FPS
    preview_interval = 1.0 / PREVIEW_FPS
    last_preview_time = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)  # ครึ่งของ grid (ปรับได้)

    try:
        while True:
            t0 = time.time()

            # ===== Video out =====
            frame = readers[active].get_latest()
            if frame is None or frame.size == 0:
                frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            pipe_write(vid_handle, frame.tobytes())

            # ===== Audio out =====
            audio.write_once()

            # ===== Preview =====
            if (t0 - last_preview_time) >= preview_interval:
                frames = [ensure_size(r.get_latest(), WIDTH, HEIGHT) for r in readers]
                grid = make_preview_grid(frames, active)
                cv2.imshow(WINDOW_NAME, grid)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    break
                elif k in (ord('1'), ord('2'), ord('3'), ord('4')):
                    new_active = int(chr(k)) - 1
                    if new_active != active:
                        # เลือกอุปกรณ์เสียงของ source ใหม่
                        if AUDIO_DEVICE_INDEXES:
                            new_dev = AUDIO_DEVICE_INDEXES[new_active]
                        else:
                            new_dev = auto_pick_audio_index_for_source(new_active)
                        if new_dev is not None:
                            audio.crossfade_to_device(new_dev)
                        active = new_active
                        last_switch = time.time()
                last_preview_time = t0

            # ===== Auto switch =====
            if time.time() - last_switch >= SWITCH_SECONDS:
                probe = active
                switched = False
                for _ in range(len(readers)-1):
                    probe = (probe + 1) % len(readers)
                    ok, diff, bright = check_motion_from_reader(readers[probe])
                    print(f"Pre-switch VIDEO dev#{DEVICE_INDEXES[probe]} moving={ok} diff={diff:.2f}% bright={bright:.2f}%")
                    if ok:
                        # จัดเสียงให้ตรงแหล่งใหม่
                        if AUDIO_DEVICE_INDEXES:
                            new_dev = AUDIO_DEVICE_INDEXES[probe]
                        else:
                            new_dev = auto_pick_audio_index_for_source(probe)
                        if new_dev is not None:
                            audio.crossfade_to_device(new_dev)
                        active = probe
                        last_switch = time.time()
                        print(f"Switched → VIDEO dev#{DEVICE_INDEXES[active]} + AUDIO dev#{new_dev if new_dev is not None else 'None'}")
                        switched = True
                        break
                if not switched:
                    print("All candidates look frozen/dark. Keep current.")
                    last_switch = time.time()

            # pace
            elapsed = time.time() - t0
            sleep_dur = out_frame_interval - elapsed
            if sleep_dur > 0: time.sleep(sleep_dur)

    finally:
        try:
            win32file.CloseHandle(vid_handle)
            win32file.CloseHandle(aud_handle)
        except Exception: pass
        for r in readers: r.close()
        try: audio.stop()
        except Exception: pass
        cv2.destroyAllWindows()
        try: ff.terminate()
        except Exception: pass

if __name__ == "__main__":
    main()
