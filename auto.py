import time, threading, subprocess, sys
from typing import List, Tuple, Optional
import numpy as np
import cv2
import sounddevice as sd

# Windows named pipe
import win32pipe, win32file, pywintypes

# ====================== CONFIG ======================
RTMP_URL = "rtmp://51.79.231.72:1935/xstream/test-siam-1"  # <-- เปลี่ยนปลายทางของคุณ
FFMPEG = "ffmpeg"                                   # หรือระบุ path เต็ม

# วิดีโอ: ดัชนีอุปกรณ์ (DirectShow) ของ capture 4 ช่อง
DEVICE_INDEXES = [0, 1, 2, 3]

# เสียง: ดัชนีอุปกรณ์ input (จาก sounddevice.query_devices())
# ต้อง mapping ให้ตรงกับแต่ละช่องวิดีโอข้างบน
AUDIO_DEVICE_INDEXES = [3, 5, 7, 9]  # <-- ตัวอย่าง, แก้ให้ตรงเครื่องคุณ!

# ขนาดและเฟรมเรต (บังคับให้ทุกช่องเท่ากัน)
WIDTH, HEIGHT = 1280, 720
FPS = 30
BITRATE = "2000k"
X264_PRESET = "slow"
GOP_SECONDS = 2

# สลับทุกกี่วินาที
SWITCH_SECONDS = 120

# ตรวจการเคลื่อนไหวของ source ถัดไป
MOTION_CHECK_FRAMES = 3
MOTION_CHECK_INTERVAL = 0.4
DIFF_THRESHOLD = 2.0        # % ความต่างเฉลี่ยต่อพิกเซล (>= ถือว่าขยับ)
MIN_BRIGHTNESS = 5.0        # % สว่างเฉลี่ยขั้นต่ำ (กันจอมืด/ดำ)

# พรีวิว
WINDOW_NAME = "Preview (2x2) — green = ACTIVE"
PREVIEW_FPS = 15
BORDER_THICK = 6

# เสียง
AUDIO_SR = 48000
AUDIO_CH = 2
AUDIO_BLOCK = 1024  # sample per channel per write (ลด/เพิ่มเพื่อลด latency)
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
    """อ่านเฟรมของแต่ละอุปกรณ์เก็บไว้เป็นเฟรมล่าสุด"""
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
        try:
            self.cap.release()
        except Exception:
            pass

def check_motion_from_reader(reader: CaptureReader) -> Tuple[bool, float, float]:
    frames = []
    for _ in range(MOTION_CHECK_FRAMES):
        f = reader.get_latest()
        frames.append(f)
        time.sleep(MOTION_CHECK_INTERVAL)
    br = [brightness_percent(f) for f in frames]
    avg_bright = float(np.mean(br))
    if avg_bright < MIN_BRIGHTNESS:
        return (False, 0.0, avg_bright)
    diffs = []
    for i in range(len(frames)-1):
        diffs.append(frame_diff_percent(frames[i], frames[i+1]))
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return (avg_diff >= DIFF_THRESHOLD, avg_diff, avg_bright)

def make_preview_grid(frames: List[np.ndarray], active_idx: int) -> np.ndarray:
    assert len(frames) == 4
    h, w = frames[0].shape[:2]
    grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    grid[0:h, 0:w] = frames[0]
    grid[0:h, w:2*w] = frames[1]
    grid[h:2*h, 0:w] = frames[2]
    grid[h:2*h, w:2*w] = frames[3]
    pos_map = {0:(0,0), 1:(0,w), 2:(h,0), 3:(h,w)}
    y, x = pos_map.get(active_idx, (0,0))
    cv2.rectangle(grid, (x, y), (x+w, y+h), (0,255,0), BORDER_THICK)
    label = f"ACTIVE #{active_idx} (video dev {DEVICE_INDEXES[active_idx]}, audio dev {AUDIO_DEVICE_INDEXES[active_idx]})"
    cv2.putText(grid, label, (x+18, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    return grid

# ---------- Named Pipes helpers (Windows) ----------
def create_connected_pipe(name: str) -> int:
    """สร้างและรอ client (ffmpeg) มา connect แล้วคืน handle"""
    full = r"\\.\pipe\{}".format(name)
    handle = win32pipe.CreateNamedPipe(
        full,
        win32pipe.PIPE_ACCESS_OUTBOUND,  # เราจะเขียนออก (Python -> ffmpeg)
        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
        1, 1024*1024, 1024*1024, 0, None
    )
    # รอ ffmpeg connect
    win32pipe.ConnectNamedPipe(handle, None)
    return handle

def pipe_write(handle, data: bytes):
    try:
        win32file.WriteFile(handle, data)
    except pywintypes.error as e:
        print("Pipe write error:", e)
        raise

# ---------------------------------------------------

def launch_ffmpeg_with_pipes():
    """
    ffmpeg อ่านจากสอง named pipes:
    - video: rawvideo bgr24  WxH@FPS
    - audio: s16le stereo @ 48 kHz
    """
    video_pipe = r"\\.\pipe\video_pipe"
    audio_pipe = r"\\.\pipe\audio_pipe"

    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "warning",

        # video from named pipe
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", video_pipe,

        # audio from named pipe
        "-f", "s16le",
        "-ar", str(AUDIO_SR),
        "-ac", str(AUDIO_CH),
        "-i", audio_pipe,

        # map
        "-map", "0:v:0",
        "-map", "1:a:0",

        "-c:v", "libx264",
        "-preset", X264_PRESET,
        "-tune", "zerolatency",
        "-b:v", BITRATE,
        "-maxrate", BITRATE,
        "-bufsize", BITRATE,
        "-g", str(FPS * GOP_SECONDS),
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", str(AUDIO_SR),

        "-f", "flv",
        RTMP_URL
    ]
    # เรียก ffmpeg ก่อน แล้วค่อยรอ connect pipes
    proc = subprocess.Popen(cmd)
    return proc

def main():
    # เตรียม video readers
    readers = [CaptureReader(idx, WIDTH, HEIGHT, FPS) for idx in DEVICE_INDEXES]
    for r in readers: r.start()
    time.sleep(1.0)  # รอให้มีเฟรม

    # เปิด ffmpeg
    ff = launch_ffmpeg_with_pipes()

    # สร้างและรอ ffmpeg connect ท่อ
    print("Waiting ffmpeg to connect pipes ...")
    vid_handle = create_connected_pipe("video_pipe")
    aud_handle = create_connected_pipe("audio_pipe")
    print("Pipes connected.")

    # เริ่มที่ตัวแรกที่ “ขยับ”
    active = 0
    for _ in range(len(readers)):
        ok, diff, bright = check_motion_from_reader(readers[active])
        print(f"Init VIDEO dev#{DEVICE_INDEXES[active]} moving={ok} diff={diff:.2f}% bright={bright:.2f}%")
        if ok: break
        active = (active + 1) % len(readers)

    # เปิดสตรีมเสียงจากอุปกรณ์ที่ active
    audio_idx = AUDIO_DEVICE_INDEXES[active]
    audio_stream = sd.InputStream(
        device=audio_idx, samplerate=AUDIO_SR, channels=AUDIO_CH, dtype='int16',
        blocksize=AUDIO_BLOCK
    )
    audio_stream.start()

    last_switch = time.time()
    frame_interval = 1.0 / FPS
    preview_interval = 1.0 / PREVIEW_FPS
    last_preview_time = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH*2//2, HEIGHT*2//2)

    try:
        while True:
            t0 = time.time()

            # ====== VIDEO: เขียนเฟรมลงท่อ ======
            frame = readers[active].get_latest()
            if frame is None or frame.size == 0:
                frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            pipe_write(vid_handle, frame.tobytes())

            # ====== AUDIO: อ่านบล็อกจาก input ปัจจุบัน แล้วเขียนลงท่อ ======
            try:
                audio_block, _ = audio_stream.read(AUDIO_BLOCK)
                pipe_write(aud_handle, audio_block.tobytes())
            except sd.PortAudioError as e:
                # ถ้าอ่านเสียงพลาด ใส่ silence กันหลุด
                silence = (np.zeros((AUDIO_BLOCK, AUDIO_CH), dtype=np.int16)).tobytes()
                pipe_write(aud_handle, silence)

            # ====== PREVIEW ======
            if (t0 - last_preview_time) >= preview_interval:
                frames = [readers[i].get_latest() for i in range(4)]
                frames = [ensure_size(f, WIDTH, HEIGHT) for f in frames]
                grid = make_preview_grid(frames, active)
                cv2.imshow(WINDOW_NAME, grid)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    break
                elif k in (ord('1'), ord('2'), ord('3'), ord('4')):
                    # manual switch (พร้อมสลับเสียง)
                    new_active = int(chr(k)) - 1
                    if new_active != active:
                        # เปลี่ยน input เสียง
                        audio_stream.stop()
                        audio_stream.close()
                        audio_idx = AUDIO_DEVICE_INDEXES[new_active]
                        audio_stream = sd.InputStream(
                            device=audio_idx, samplerate=AUDIO_SR, channels=AUDIO_CH, dtype='int16',
                            blocksize=AUDIO_BLOCK
                        )
                        audio_stream.start()
                        active = new_active
                        last_switch = time.time()
                last_preview_time = t0

            # ====== AUTO SWITCH ทุก SWITCH_SECONDS (ตรวจ motion ก่อน) ======
            if time.time() - last_switch >= SWITCH_SECONDS:
                switched = False
                probe = active
                for _ in range(len(readers)-1):
                    probe = (probe + 1) % len(readers)
                    ok, diff, bright = check_motion_from_reader(readers[probe])
                    print(f"Pre-switch VIDEO dev#{DEVICE_INDEXES[probe]} moving={ok} diff={diff:.2f}% bright={bright:.2f}%")
                    if ok:
                        # สลับเสียงไปอุปกรณ์ที่ mapping กับ probe
                        audio_stream.stop()
                        audio_stream.close()
                        audio_idx = AUDIO_DEVICE_INDEXES[probe]
                        audio_stream = sd.InputStream(
                            device=audio_idx, samplerate=AUDIO_SR, channels=AUDIO_CH, dtype='int16',
                            blocksize=AUDIO_BLOCK
                        )
                        audio_stream.start()

                        active = probe
                        last_switch = time.time()
                        print(f"Switched to VIDEO dev#{DEVICE_INDEXES[active]} / AUDIO dev#{audio_idx}")
                        switched = True
                        break
                if not switched:
                    print("All candidates look frozen/dark. Keep current.")
                    last_switch = time.time()

            # pace video write
            elapsed = time.time() - t0
            sleep_dur = frame_interval - elapsed
            if sleep_dur > 0:
                time.sleep(sleep_dur)

    finally:
        try:
            win32file.CloseHandle(vid_handle)
            win32file.CloseHandle(aud_handle)
        except Exception:
            pass
        for r in readers: r.close()
        try:
            audio_stream.stop(); audio_stream.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            ff.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
