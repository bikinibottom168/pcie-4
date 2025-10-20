import time, threading, subprocess, sys
from typing import List, Tuple, Optional
import numpy as np
import cv2
import sounddevice as sd

# Windows named pipes
import win32pipe, win32file, pywintypes

# ====================== CONFIG ======================
RTMP_URL = "rtmp://51.79.231.72:1935/xstream/test-siam-1"  # <-- เปลี่ยนปลายทางของคุณ
FFMPEG   = r"ffmpeg"                                 # หรือพาธเต็ม เช่น r"C:\ffmpeg\bin\ffmpeg.exe"

# วิดีโอ: index ของ capture 4 ช่อง (DirectShow) เรียง [ซ้ายบน,ขวาบน,ซ้ายล่าง,ขวาล่าง]
DEVICE_INDEXES = [0, 1, 2, 3]

# เสียง: index ของ input device ตามลำดับเดียวกับภาพ (ต้องครบ 4 ค่า)
AUDIO_DEVICE_INDEXES = [4, 7, 11, 14]  # <-- แก้ให้ตรงเครื่องคุณ

# ขนาด/เฟรมเรต
WIDTH, HEIGHT = 1920, 1080
FPS   = 30
BITRATE      = "2000k"
X264_PRESET  = "slow"
GOP_SECONDS  = 2

# สลับและตรวจ motion
SWITCH_SECONDS        = 120
MOTION_CHECK_FRAMES   = 3
MOTION_CHECK_INTERVAL = 0.4
DIFF_THRESHOLD        = 2.0   # % diff >= นี้ = มีการเคลื่อนไหว
MIN_BRIGHTNESS        = 5.0   # % สว่างเฉลี่ยขั้นต่ำ

# พรีวิว
WINDOW_NAME   = "Preview (2x2) — green = ACTIVE"
PREVIEW_FPS   = 15
BORDER_THICK  = 6

# เสียง
AUDIO_SR      = 48000
AUDIO_CH      = 2
AUDIO_BLOCK   = 1024     # samples/CH per write
AUDIO_FADE_MS = 120      # crossfade ตอนสลับอุปกรณ์
AUDIO_TRY_TIMEOUT_SEC = 10   # พยายามต่ออุปกรณ์ครั้งแรกภายใน 10 วิ
AUDIO_RETRY_INTERVAL_SEC = 2 # ถ้าไม่สำเร็จ จะลองใหม่ทุก 2 วิ
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
    """อ่านเฟรมของแต่ละอุปกรณ์เก็บเป็นเฟรมล่าสุด (เปิดพร้อมกันทั้ง 4 ช่อง)"""
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
    if avg_bright < MIN_BRIGHTNESS:
        return (False, 0.0, avg_bright)
    diffs = [frame_diff_percent(frames[i], frames[i+1]) for i in range(len(frames)-1)]
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    return (avg_diff >= DIFF_THRESHOLD, avg_diff, avg_bright)


def make_preview_grid(frames: List[np.ndarray], active_idx: int) -> np.ndarray:
    """รวม 4 เฟรมเป็น 2x2 และตีกรอบเขียวแหล่งที่ active (ลำดับ [0,1; 2,3])"""
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
    label = f"ACTIVE #{active_idx}  V:{DEVICE_INDEXES[active_idx]}  A:{AUDIO_DEVICE_INDEXES[active_idx]}"
    cv2.putText(grid, label, (x+18, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    return grid


# ----------------- Named Pipes (Windows) --------------
def create_listening_pipe(name: str):
    """สร้างท่อ (ยังไม่ Connect) — Python เป็นฝั่งเขียน (OUTBOUND)"""
    full = r"\\.\pipe\{}".format(name)
    handle = win32pipe.CreateNamedPipe(
        full,
        win32pipe.PIPE_ACCESS_OUTBOUND,
        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
        1, 1024*1024, 1024*1024, 0, None
    )
    return handle

def wait_client_connect(handle, deadline_sec=20, tag=""):
    """รอ client (ffmpeg) connect พร้อม timeout"""
    end = time.time() + deadline_sec
    while time.time() < end:
        try:
            win32pipe.ConnectNamedPipe(handle, None)
            print(f"{tag} pipe connected.")
            return True
        except pywintypes.error as e:
            if e.winerror == 535:  # ERROR_PIPE_CONNECTED: client ต่อมาแล้ว
                print(f"{tag} pipe already connected.")
                return True
        time.sleep(0.1)
    print(f"Timeout waiting for {tag} pipe.")
    return False

def pipe_write(handle, data: bytes):
    win32file.WriteFile(handle, data)


# ------------------ FFmpeg launcher -------------------
def launch_ffmpeg_with_existing_pipes():
    """ffmpeg อ่านจาก named pipes ที่ Python สร้างไว้แล้ว"""
    video_pipe = r"\\.\pipe\video_pipe"
    audio_pipe = r"\\.\pipe\audio_pipe"
    cmd = [
        FFMPEG,
        "-hide_banner", "-loglevel", "info",
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
        "-f", "flv", RTMP_URL
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

def ffmpeg_log_printer(proc: subprocess.Popen):
    try:
        if proc.stderr:
            for line in proc.stderr:
                sys.stdout.write("[ffmpeg] " + line)
    except Exception:
        pass


# ---------------------- Audio I/O ---------------------
class AudioPipeWriter(threading.Thread):
    """
    เขียนเสียงลง audio pipe ตลอดเวลา:
    - เริ่มจาก 'silence mode' เพื่อให้สตรีมเปิดได้ทันที
    - พยายามเชื่อมต่ออุปกรณ์เสียงจริงภายใน 10 วิ (และ retry เป็นระยะ)
    - เมื่อพร้อม -> crossfade เปลี่ยนจาก silence -> real audio
    - เวลาเปลี่ยน source -> crossfade ไปอุปกรณ์ใหม่
    """
    def __init__(self, pipe_handle):
        super().__init__(daemon=True)
        self.pipe = pipe_handle
        self.stream: Optional[sd.InputStream] = None
        self.current_dev: Optional[int] = None
        self.lock = threading.Lock()
        self.running = True
        self.requested_dev: Optional[int] = None  # dev ที่อยากใช้ (ตาม source ปัจจุบัน)
        self.first_try_deadline = 0.0
        self.silence_mode = True  # เริ่มด้วยเงียบ

    def set_target_device(self, dev_index: int):
        with self.lock:
            self.requested_dev = dev_index
            # หากยังไม่มี stream เลย ให้เริ่มจับเวลา 10 วิ สำหรับ first-try
            if self.stream is None and self.first_try_deadline == 0.0:
                self.first_try_deadline = time.time() + AUDIO_TRY_TIMEOUT_SEC

    def _start_stream(self, dev_index: int) -> bool:
        try:
            st = sd.InputStream(device=dev_index, samplerate=AUDIO_SR,
                                channels=AUDIO_CH, dtype='int16',
                                blocksize=AUDIO_BLOCK)
            st.start()
            self.stream = st
            self.current_dev = dev_index
            return True
        except Exception as e:
            return False

    def _stop_stream(self):
        if self.stream is not None:
            try: self.stream.stop()
            except Exception: pass
            try: self.stream.close()
            except Exception: pass
            self.stream = None
            self.current_dev = None

    def _write_silence(self):
        silence = (np.zeros((AUDIO_BLOCK, AUDIO_CH), dtype=np.int16)).tobytes()
        pipe_write(self.pipe, silence)

    def _write_from_stream(self):
        if self.stream is None:
            self._write_silence()
            return
        try:
            block, _ = self.stream.read(AUDIO_BLOCK)
            pipe_write(self.pipe, block.tobytes())
        except sd.PortAudioError:
            self._write_silence()

    def _crossfade_switch(self, new_dev: int):
        # fade-out
        steps = max(2, int((AUDIO_FADE_MS/1000.0) * (AUDIO_SR / AUDIO_BLOCK)))
        for s in range(steps):
            scale = 1.0 - (s+1)/steps
            if self.stream is None:
                self._write_silence()
            else:
                try:
                    block, _ = self.stream.read(AUDIO_BLOCK)
                    block = (block.astype(np.int32) * scale).astype(np.int16)
                    pipe_write(self.pipe, block.tobytes())
                except sd.PortAudioError:
                    self._write_silence()

        # switch device
        self._stop_stream()
        if self._start_stream(new_dev):
            self.silence_mode = False
            # fade-in
            for s in range(steps):
                scale = (s+1)/steps
                try:
                    block, _ = self.stream.read(AUDIO_BLOCK)
                    block = (block.astype(np.int32) * scale).astype(np.int16)
                    pipe_write(self.pipe, block.tobytes())
                except sd.PortAudioError:
                    self._write_silence()
        else:
            # start new failed -> keep silence and retry later
            self.silence_mode = True

    def run(self):
        next_retry = 0.0
        while self.running:
            # 1) ถ้ายังไม่มี stream (เริ่มต้น) → พยายามเริ่มภายใน 10 วิแรก
            with self.lock:
                target = self.requested_dev

            now = time.time()
            if self.stream is None:
                # ใส่เสียงเงียบไปเรื่อยๆ เพื่อให้ ffmpeg ได้ audio ต่อเนื่อง
                self._write_silence()

                if target is not None:
                    # ภายใน timeout แรกให้พยายามเรื่อยๆ (spam-retry)
                    if self.first_try_deadline != 0.0 and now <= self.first_try_deadline:
                        if self._start_stream(target):
                            self.silence_mode = False
                            continue  # next loop จะอ่านจาก stream
                    # หมด 10 วิแล้ว → ตั้ง retry เป็นช่วงๆ
                    if now >= self.first_try_deadline and now >= next_retry:
                        if self._start_stream(target):
                            self.silence_mode = False
                        else:
                            self.silence_mode = True
                            next_retry = now + AUDIO_RETRY_INTERVAL_SEC
            else:
                # 2) มี stream แล้ว → ถ้า target เปลี่ยน (เปลี่ยน source) ให้ crossfade ไปอุปกรณ์ใหม่
                if target is not None and target != self.current_dev:
                    self._crossfade_switch(target)
                # เขียนบล็อกเสียงจาก stream
                self._write_from_stream()

        # cleanup
        self._stop_stream()

    def stop(self):
        self.running = False


# -------------------------- MAIN ---------------------
def main():
    print(f"Using FFMPEG: {FFMPEG}")

    # 1) ตั้ง readers วิดีโอ (4 แหล่ง)
    readers = [CaptureReader(idx, WIDTH, HEIGHT, FPS) for idx in DEVICE_INDEXES]
    for r in readers: r.start()
    time.sleep(1.0)  # รอให้มีเฟรม

    # 2) สร้างท่อรอไว้ก่อน
    vid_handle = create_listening_pipe("video_pipe")
    aud_handle = create_listening_pipe("audio_pipe")

    # 3) สตาร์ต ffmpeg
    ff = launch_ffmpeg_with_existing_pipes()
    threading.Thread(target=ffmpeg_log_printer, args=(ff,), daemon=True).start()

    print("Launched ffmpeg, waiting it to connect pipes...")

    # 4) รอ ffmpeg connect ทั้ง 2 ท่อ (ภาพ/เสียง)
    ok_v = wait_client_connect(vid_handle, deadline_sec=20, tag="Video")
    ok_a = wait_client_connect(aud_handle, deadline_sec=20, tag="Audio")
    if not (ok_v and ok_a):
        print("ERROR: Pipe connection failed. Check ffmpeg path/permissions/args.")
        try:
            if ff.stderr:
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
        if ok: break
        active = (active + 1) % len(readers)

    # 6) สร้าง audio writer (เริ่มด้วย silence mode) แล้วตั้ง target device ตามภาพ active
    audio_writer = AudioPipeWriter(aud_handle)
    audio_writer.start()
    try:
        audio_writer.set_target_device(AUDIO_DEVICE_INDEXES[active])
    except IndexError:
        print("ERROR: AUDIO_DEVICE_INDEXES ไม่ครบ 4 ตัวหรือ index เกินขอบเขต")
        sys.exit(1)

    last_switch = time.time()
    out_frame_interval = 1.0 / FPS
    preview_interval   = 1.0 / PREVIEW_FPS
    last_preview_time  = 0.0
    last_frame         = None

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

            # ===== PREVIEW =====
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
                        # สลับอุปกรณ์เสียงให้ตรงกับภาพใหม่ (audio_writer จะ crossfade เอง)
                        try:
                            audio_writer.set_target_device(AUDIO_DEVICE_INDEXES[new_active])
                        except IndexError:
                            print("AUDIO_DEVICE_INDEXES ไม่ครบ 4 ตัว")
                        active = new_active
                        last_switch = time.time()
                last_preview_time = t0

            # ===== AUTO SWITCH (ทุก SWITCH_SECONDS + ตรวจ motion ก่อน) =====
            if time.time() - last_switch >= SWITCH_SECONDS:
                probe = active
                switched = False
                for _ in range(len(readers)-1):
                    probe = (probe + 1) % len(readers)
                    ok, diff, bright = check_motion_from_reader(readers[probe])
                    print(f"Pre-switch VIDEO dev#{DEVICE_INDEXES[probe]} moving={ok} diff={diff:.2f}% bright={bright:.2f}%")
                    if ok:
                        try:
                            audio_writer.set_target_device(AUDIO_DEVICE_INDEXES[probe])
                        except IndexError:
                            print("AUDIO_DEVICE_INDEXES ไม่ครบ 4 ตัว — คงเสียงเดิม")
                        active = probe
                        last_switch = time.time()
                        print(f"Switched → VIDEO dev#{DEVICE_INDEXES[active]} + AUDIO dev#{AUDIO_DEVICE_INDEXES[active]}")
                        switched = True
                        break
                if not switched:
                    print("All candidates look frozen/dark. Keep current.")
                    last_switch = time.time()

            # pace video
            elapsed = time.time() - t0
            sleep_dur = out_frame_interval - elapsed
            if sleep_dur > 0:
                time.sleep(sleep_dur)

    finally:
        # cleanup
        try: audio_writer.stop()
        except Exception: pass
        try:
            win32file.CloseHandle(vid_handle)
            win32file.CloseHandle(aud_handle)
        except Exception: pass
        for r in readers: r.close()
        cv2.destroyAllWindows()
        try: ff.terminate()
        except Exception: pass


if __name__ == "__main__":
    main()
