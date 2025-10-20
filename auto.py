import time, threading, subprocess, sys, os
from typing import List, Tuple, Optional
import numpy as np
import cv2
import sounddevice as sd

# Windows named pipes
import win32pipe, win32file, pywintypes

# ====================== CONFIG ======================
# --- เปลี่ยนค่าตามระบบคุณ ---
RTMP_URL = "rtmp://51.79.231.72:1935/xstream/test-siam-1"  # <-- เปลี่ยนปลายทางของคุณ
FFMPEG   = r"ffmpeg"                                 # หรือระบุ path เต็ม เช่น r"C:\ffmpeg\bin\ffmpeg.exe"

# วิดีโอ: ดัชนีอุปกรณ์ (DirectShow) ของ capture 4 ช่อง (เรียงซ้ายบน,ขวาบน,ซ้ายล่าง,ขวาล่าง)
DEVICE_INDEXES = [0, 1, 2, 3]

# เสียง: จับคู่ “ตามลำดับเดียวกับภาพ”
#   เช่น ภาพ dev 0 → เสียง index 4, ภาพ dev 1 → เสียง index 7, ...
AUDIO_DEVICE_INDEXES = [4, 7, 11, 14]  # <-- แก้ให้ตรงเครื่องคุณ

# ขนาด/เฟรมเรต (ให้ทุกช่องเท่ากัน)
WIDTH, HEIGHT = 1280, 720
FPS = 30
BITRATE = "3500k"
X264_PRESET = "veryfast"
GOP_SECONDS = 2

# สลับและตรวจ motion
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
AUDIO_FADE_MS = 120          # cross-fade ตอนสลับเสียง
# ====================================================


# ---------------------- Utils -----------------------
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


# ----------------- Video Capture Reader --------------
class CaptureReader(threading.Thread):
    """อ่านเฟรมของแต่ละอุปกรณ์เก็บไว้เป็นเฟรมล่าสุด (เปิดพร้อมกันทั้ง 4 ช่อง)"""
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
        interval = 1.0 / max(5, min(self.fps, 30))  # จำกัดอัตราอ่าน
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
        frames.append(reader.get_latest())
        time.sleep(MOTION_CHECK_INTERVAL)
    avg_bright = float(np.mean([brightness_percent(f) for f in frames]))
    if avg_bright < MIN_BRIGHTNESS:
        return (False, 0.0, avg_bright)
    diffs = [frame_diff_percent(frames[i], frames[i+1]) for i in range(len(frames)-1)]
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return (avg_diff >= DIFF_THRESHOLD, avg_diff, avg_bright)


def make_preview_grid(frames: List[np.ndarray], active_idx: int) -> np.ndarray:
    """รวม 4 เฟรมเป็น 2x2 และตีกรอบเขียวที่ active (ลำดับ [0,1; 2,3])"""
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
    cv2.putText(grid, f"ACTIVE #{active_idx}  V:{DEVICE_INDEXES[active_idx]}  A:{AUDIO_DEVICE_INDEXES[active_idx]}",
                (x+18, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    return grid


# ----------------- Named Pipes (Windows) --------------
def create_listening_pipe(name: str):
    """สร้างท่อ (ยังไม่ Connect) — Python จะเป็นฝั่งเขียนออก (OUTBOUND)"""
    full = r"\\.\pipe\{}".format(name)
    handle = win32pipe.CreateNamedPipe(
        full,
        win32pipe.PIPE_ACCESS_OUTBOUND,
        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
        1, 1024*1024, 1024*1024, 0, None
    )
    return handle

def wait_client_connect(handle):
    try:
        win32pipe.ConnectNamedPipe(handle, None)
    except pywintypes.error as e:
        # ถ้าลูกค้าต่อมาก่อนแล้ว จะได้ ERROR_PIPE_CONNECTED (535) → ถือว่าโอเค
        if e.winerror != 535:
            raise

def pipe_write(handle, data: bytes):
    win32file.WriteFile(handle, data)


# ------------------ FFmpeg launcher -------------------
def launch_ffmpeg_with_existing_pipes():
    """
    ffmpeg จะอ่านจาก named pipes ที่มีอยู่แล้ว:
      - \\.\pipe\video_pipe : rawvideo (bgr24, WxH@FPS)
      - \\.\pipe\audio_pipe : s16le stereo @ 48k
    """
    video_pipe = r"\\.\pipe\video_pipe"
    audio_pipe = r"\\.\pipe\audio_pipe"
    cmd = [
        FFMPEG,
        "-hide_banner", "-loglevel", "info",  # เปิด log เพื่อ debug
        # video
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", video_pipe,
        # audio
        "-f", "s16le", "-ar", str(AUDIO_SR), "-ac", str(AUDIO_CH), "-i", audio_pipe,
        # map
        "-map", "0:v:0", "-map", "1:a:0",
        # encode
        "-c:v", "libx264", "-preset", X264_PRESET, "-tune", "zerolatency",
        "-b:v", BITRATE, "-maxrate", BITRATE, "-bufsize", BITRATE,
        "-g", str(FPS * GOP_SECONDS), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", str(AUDIO_SR),
        # output
        "-f", "flv", RTMP_URL
    ]
    # เก็บ stderr ไว้แสดงเป็น log
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def ffmpeg_log_printer(proc: subprocess.Popen):
    """อ่าน stderr ของ ffmpeg แล้วพิมพ์ให้ดู (background thread)"""
    try:
        if proc.stderr:
            for line in proc.stderr:
                sys.stdout.write("[ffmpeg] " + line)
    except Exception:
        pass


# ---------------------- Audio I/O ---------------------
class AudioIO:
    """อ่านเสียงจาก input (sounddevice) → เขียนลง audio named pipe พร้อม crossfade ตอนสลับ"""
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

    def crossfade_to_device(self, new_dev_index: int, fade_ms: int = AUDIO_FADE_MS):
        """fade-out → switch → fade-in เพื่อลดเสียงป๊อปตอนสลับแหล่ง"""
        steps = max(2, int((fade_ms/1000.0) * (AUDIO_SR / AUDIO_BLOCK)))

        # fade-out
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


# -------------------------- MAIN ---------------------
def main():
    # 0) ตรวจ ffmpeg
    #   ถ้า ffmpeg ไม่อยู่ใน PATH ให้แก้ตัวแปร FFMPEG เป็นพาธเต็ม
    #   ตัวนี้ไม่ hard-fail แค่เตือน (ffmpeg จะ error เองถ้าไม่พบ)
    print(f"Using FFMPEG: {FFMPEG}")

    # 1) ตั้ง readers วิดีโอ (4 แหล่ง)
    readers = [CaptureReader(idx, WIDTH, HEIGHT, FPS) for idx in DEVICE_INDEXES]
    for r in readers:
        r.start()
    time.sleep(1.0)  # รอให้มีเฟรม

    # 2) สร้างท่อรอไว้ก่อน (Python = server)
    vid_handle = create_listening_pipe("video_pipe")
    aud_handle = create_listening_pipe("audio_pipe")

    # 3) สตาร์ต ffmpeg ที่จะเปิด \\.\pipe\video_pipe และ \\.\pipe\audio_pipe
    ff = launch_ffmpeg_with_existing_pipes()
    t = threading.Thread(target=ffmpeg_log_printer, args=(ff,), daemon=True)
    t.start()

    print("Launched ffmpeg, waiting for it to connect named pipes...")

    # 4) รอ ffmpeg connect (มี timeout)
    deadline = time.time() + 20  # 20 วินาที
    connected_video = False
    connected_audio = False
    while time.time() < deadline and (not connected_video or not connected_audio):
        if not connected_video:
            try:
                wait_client_connect(vid_handle)
                connected_video = True
                print("Video pipe connected.")
            except Exception:
                pass
        if not connected_audio:
            try:
                wait_client_connect(aud_handle)
                connected_audio = True
                print("Audio pipe connected.")
            except Exception:
                pass
        time.sleep(0.1)

    if not (connected_video and connected_audio):
        print("ERROR: Failed to connect pipes in time. Check ffmpeg path/permissions/args.")
        try:
            if ff.stderr:
                # อ่านส่วนแรกของ log ให้ดู
                err = ff.stderr.read(3000)
                print("FFmpeg stderr (first 3KB):\n", err)
        except Exception:
            pass
        sys.exit(1)

    print("Pipes connected. Start streaming loop…")

    # 5) เลือก active เริ่มจากตัวแรกที่ "ขยับ"
    active = 0
    for _ in range(len(readers)):
        ok, diff, bright = check_motion_from_reader(readers[active])
        print(f"Init VIDEO dev#{DEVICE_INDEXES[active]} moving={ok} diff={diff:.2f}% bright={bright:.2f}%")
        if ok:
            break
        active = (active + 1) % len(readers)

    # 6) เตรียม Audio I/O และเริ่มจากอุปกรณ์เสียงตาม index mapping
    audio = AudioIO(aud_handle)
    try:
        audio_dev = AUDIO_DEVICE_INDEXES[active]
    except IndexError:
        print("ERROR: AUDIO_DEVICE_INDEXES ไม่ครบ 4 ตัวหรือ index เกินขอบเขต")
        sys.exit(1)
    audio.start_with_device(audio_dev)

    last_switch = time.time()
    out_frame_interval = 1.0 / FPS
    preview_interval = 1.0 / PREVIEW_FPS
    last_preview_time = 0.0
    last_frame = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    try:
        while True:
            t0 = time.time()

            # ===== VIDEO → เขียนลงท่อ =====
            frame = readers[active].get_latest()
            if frame is None or frame.size == 0:
                frame = last_frame if last_frame is not None else np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            pipe_write(vid_handle, frame.tobytes())
            last_frame = frame

            # ===== AUDIO → เขียนลงท่อ =====
            audio.write_once()

            # ===== PREVIEW =====
            if (t0 - last_preview_time) >= preview_interval:
                frames = [ensure_size(r.get_latest(), WIDTH, HEIGHT) for r in readers]
                grid = make_preview_grid(frames, active)
                cv2.imshow(WINDOW_NAME, grid)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC เพื่อออก
                    break
                elif k in (ord('1'), ord('2'), ord('3'), ord('4')):
                    new_active = int(chr(k)) - 1
                    if new_active != active:
                        # สลับเสียงไปยังอุปกรณ์ที่จับคู่กับภาพใหม่
                        try:
                            new_dev = AUDIO_DEVICE_INDEXES[new_active]
                        except IndexError:
                            print("AUDIO_DEVICE_INDEXES ไม่ครบ 4 ตัว")
                            new_dev = AUDIO_DEVICE_INDEXES[active]
                        audio.crossfade_to_device(new_dev)
                        active = new_active
                        last_switch = time.time()
                last_preview_time = t0

            # ===== AUTO SWITCH ทุก SWITCH_SECONDS (ตรวจ motion ก่อน) =====
            if time.time() - last_switch >= SWITCH_SECONDS:
                probe = active
                switched = False
                for _ in range(len(readers)-1):
                    probe = (probe + 1) % len(readers)
                    ok, diff, bright = check_motion_from_reader(readers[probe])
                    print(f"Pre-switch VIDEO dev#{DEVICE_INDEXES[probe]} moving={ok} diff={diff:.2f}% bright={bright:.2f}%")
                    if ok:
                        try:
                            new_dev = AUDIO_DEVICE_INDEXES[probe]
                        except IndexError:
                            print("AUDIO_DEVICE_INDEXES ไม่ครบ 4 ตัว — คงเสียงเดิม")
                            new_dev = AUDIO_DEVICE_INDEXES[active]
                        audio.crossfade_to_device(new_dev)
                        active = probe
                        last_switch = time.time()
                        print(f"Switched → VIDEO dev#{DEVICE_INDEXES[active]} + AUDIO dev#{new_dev}")
                        switched = True
                        break
                if not switched:
                    print("All candidates look frozen/dark. Keep current.")
                    last_switch = time.time()

            # pace
            elapsed = time.time() - t0
            sleep_dur = out_frame_interval - elapsed
            if sleep_dur > 0:
                time.sleep(sleep_dur)

    finally:
        # cleanup
        try:
            win32file.CloseHandle(vid_handle)
            win32file.CloseHandle(aud_handle)
        except Exception:
            pass
        for r in readers:
            r.close()
        try:
            audio.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            ff.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
