import os, time, threading, subprocess
from typing import List, Tuple, Optional
import numpy as np
import cv2

# ====================== CONFIG ======================
RTMP_URL = "rtmp://51.79.231.72:1935/xstream/test-siam-1"  # <-- เปลี่ยนเป็นปลายทางของคุณ
FFMPEG = "ffmpeg"                                   # หรือระบุ path เต็ม

# ดัชนีอุปกรณ์ (DirectShow) ของ capture card 4 ช่อง
DEVICE_INDEXES = [0, 1, 2, 3]

# ขนาดและเฟรมเรต (บังคับให้ทุกช่องเท่ากัน)
WIDTH, HEIGHT = 1280, 720
FPS = 30
BITRATE = "2000k"
X264_PRESET = "veryfast"
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
PREVIEW_FPS = 15            # เฟรมเรตหน้าต่างพรีวิว
BORDER_THICK = 6            # ความหนากรอบสีเขียว
# ====================================================


def open_capture(idx: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

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
    """
    อ่านเฟรมของแต่ละอุปกรณ์อย่างต่อเนื่อง เก็บเฟรมล่าสุดไว้ให้ดึงใช้
    """
    def __init__(self, device_index: int, w: int, h: int, fps: int):
        super().__init__(daemon=True)
        self.device_index = device_index
        self.cap = open_capture(device_index, w, h, fps)
        self.w, self.h, self.fps = w, h, fps
        self.latest_lock = threading.Lock()
        self.latest: Optional[np.ndarray] = None
        self.running = True

    def run(self):
        interval = 1.0 / max(5, min(self.fps, 30))  # จำกัดอัตราอ่านเพื่อไม่กินเครื่อง
        while self.running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                frame = ensure_size(frame, self.w, self.h)
                with self.latest_lock:
                    self.latest = frame
            else:
                # ถ้าอ่านไม่ได้ ใส่ภาพดำกันช่องว่าง
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


def launch_ffmpeg(width: int, height: int, fps: int, bitrate: str, rtmp_url: str):
    cmd = [
        FFMPEG,
        "-hide_banner", "-loglevel", "warning",

        # video from stdin
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",

        # audio silent (บาง player ต้องการสตรีมมีเสียง)
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",

        "-map", "0:v:0",
        "-map", "1:a:0",

        "-c:v", "libx264",
        "-preset", X264_PRESET,
        "-tune", "zerolatency",
        "-b:v", bitrate,
        "-maxrate", bitrate,
        "-bufsize", bitrate,
        "-g", str(fps * GOP_SECONDS),
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",

        "-f", "flv",
        rtmp_url
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


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
    """
    รวม 4 เฟรมเป็น 2x2 แล้วตีกรอบเขียวช่องที่ active
    ลำดับวาง: [0,1; 2,3]
    """
    assert len(frames) == 4
    h, w = frames[0].shape[:2]

    # วางตำแหน่ง
    grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    grid[0:h, 0:w] = frames[0]
    grid[0:h, w:2*w] = frames[1]
    grid[h:2*h, 0:w] = frames[2]
    grid[h:2*h, w:2*w] = frames[3]

    # คำนวณพิกัดกรอบของ active
    pos_map = {
        0: (0, 0),
        1: (0, w),
        2: (h, 0),
        3: (h, w),
    }
    y, x = pos_map.get(active_idx, (0, 0))

    cv2.rectangle(grid, (x, y), (x + w, y + h), (0, 255, 0), BORDER_THICK)
    label = f"ACTIVE #{active_idx} (dev {DEVICE_INDEXES[active_idx]})"
    cv2.putText(grid, label, (x + 18, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    return grid


def main():
    # เตรียม readers
    readers = [CaptureReader(idx, WIDTH, HEIGHT, FPS) for idx in DEVICE_INDEXES]
    for r in readers:
        r.start()

    # รอให้มีเฟรมแรก
    time.sleep(1.0)

    ff = launch_ffmpeg(WIDTH, HEIGHT, FPS, BITRATE, RTMP_URL)
    if ff.stdin is None:
        print("Cannot open ffmpeg stdin.")
        return

    # เริ่มที่ตัวแรกที่ “ขยับ”
    active = 0
    for _ in range(len(readers)):
        ok, diff, bright = check_motion_from_reader(readers[active])
        print(f"Init dev#{DEVICE_INDEXES[active]} moving={ok} diff={diff:.2f}% bright={bright:.2f}%")
        if ok:
            break
        active = (active + 1) % len(readers)

    last_switch = time.time()
    out_frame_interval = 1.0 / FPS
    preview_interval = 1.0 / PREVIEW_FPS
    last_preview_time = 0.0

    last_frame = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH*2//2, HEIGHT*2//2)  # ย่อจอพรีวิวให้เห็นทั้ง 2x2

    try:
        while True:
            t0 = time.time()

            # เอาเฟรมล่าสุดของช่องที่ active ไปออกอากาศ
            frame = readers[active].get_latest()
            if frame is None or frame.size == 0:
                # fallback ใช้เฟรมก่อนหน้า/จอดำ
                frame = last_frame if last_frame is not None else np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            try:
                ff.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("FFmpeg pipe broken. Exiting.")
                break

            last_frame = frame

            # อัพเดทหน้าพรีวิวตามรอบ
            if (t0 - last_preview_time) >= preview_interval:
                frames = [ensure_size(r.get_latest(), WIDTH, HEIGHT) for r in readers]
                grid = make_preview_grid(frames, active)
                cv2.imshow(WINDOW_NAME, grid)
                # กด ESC = ออก / กด 1-4 = สลับทันที (option)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    break
                elif k in (ord('1'), ord('2'), ord('3'), ord('4')):
                    active = int(chr(k)) - 1
                    last_switch = time.time()
                last_preview_time = t0

            # ถึงเวลาจะสลับ? ตรวจ motion ของช่องถัดไป
            if time.time() - last_switch >= SWITCH_SECONDS:
                switched = False
                probe = active
                for _ in range(len(readers)-1):
                    probe = (probe + 1) % len(readers)
                    ok, diff, bright = check_motion_from_reader(readers[probe])
                    print(f"Pre-switch dev#{DEVICE_INDEXES[probe]} moving={ok} diff={diff:.2f}% bright={bright:.2f}%")
                    if ok:
                        active = probe
                        last_switch = time.time()
                        print(f"Switched to dev#{DEVICE_INDEXES[active]}")
                        switched = True
                        break
                if not switched:
                    # ไม่มีตัวไหนขยับเลย → คงตัวเดิม
                    print("All candidates look frozen/dark. Keep current.")
                    last_switch = time.time()

            # คุมเฟรมเรตสำหรับการส่งออก
            elapsed = time.time() - t0
            sleep_dur = out_frame_interval - elapsed
            if sleep_dur > 0:
                time.sleep(sleep_dur)

    finally:
        try:
            ff.stdin.close()
        except Exception:
            pass
        ff.terminate()
        for r in readers:
            r.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
