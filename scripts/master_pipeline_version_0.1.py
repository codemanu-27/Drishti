import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pytz
import xml.etree.ElementTree as ET

# ==============================================================================
# 1. PROJECT DIRECTORY & AUTO-DISCOVERY CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_FOLDER = PROJECT_ROOT / "data" / "dataset" / "images"
XML_FOLDER = PROJECT_ROOT / "data" / "dataset" / "annotations" / "xmls"
AUDIT_CSV = PROJECT_ROOT / "data" / "dataset" / "sync_audit_log.csv"

IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
XML_FOLDER.mkdir(parents=True, exist_ok=True)

# Fixed GUI Canvas Dimensions (Fits comfortably on 1080p / 720p screens without stretching)
CANVAS_W = 1360
CANVAS_H = 760
SIDEBAR_W = 400
VIEWPORT_W = CANVAS_W - SIDEBAR_W  # 960px dedicated to the video frame
VIEWPORT_H = CANVAS_H              # 760px height

# Standard RD2022 Classes (Hotkeys 0-9 and B)
CLASS_HOTKEYS = {
    '0': ('D00', 'Longitudinal Crack'),
    '1': ('D01', 'Long. Construction Joint'),
    '2': ('D10', 'Transverse Crack'),
    '3': ('D11', 'Trans. Construction Joint'),
    '4': ('D20', 'Alligator Crack'),
    '5': ('D40', 'Pothole / Other Corruption'),
    '6': ('D43', 'Crosswalk Blur'),
    '7': ('D44', 'White Line Blur'),
    '8': ('D50', 'Repair (D50)'),
    '9': ('Repair', 'Repair Surface'),
    'b': ('Block crack', 'Block Crack')
}

UTC_ZONE = pytz.utc
IST_ZONE = pytz.timezone('Asia/Kolkata')


def discover_sessions():
    """Automatically pairs every .mp4 file in data/raw/ with its matching .txt file."""
    sessions = []
    for mp4_path in sorted(RAW_DIR.glob("*.mp4")):
        txt_path = RAW_DIR / f"{mp4_path.stem}.txt"
        if txt_path.exists():
            sessions.append({
                'prefix': mp4_path.stem,
                'video_path': mp4_path,
                'gps_path': txt_path,
                'expected_start_delay_range': (3.5, 5.5),
                'expected_end_delay_range': (2.0, 5.5),
                'motion_speed_threshold': 0.45
            })
    return sessions


# ==============================================================================
# 2. GUI LOADING SCREEN & ASPECT-RATIO VIEWPORT MAPPER
# ==============================================================================
def show_status_screen(window_name, title, subtitle, progress_ratio=None):
    """Displays a crisp, readable status screen inside the OpenCV window during heavy tasks."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), (24, 24, 28), dtype=np.uint8)
    
    cv2.putText(canvas, title, (80, 280), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (80, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2, cv2.LINE_AA)
    
    if progress_ratio is not None:
        bar_x, bar_y, bar_w, bar_h = 80, 400, CANVAS_W - 160, 36
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 70), -1)
        fill_w = int(bar_w * np.clip(progress_ratio, 0.0, 1.0))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 100), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 2)
        pct_text = f"{int(progress_ratio * 100)}%"
        cv2.putText(canvas, pct_text, (bar_x + bar_w // 2 - 30, bar_y + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, canvas)
    cv2.waitKey(1)


class ViewportTransform:
    """
    Fits any video resolution (Portrait or Landscape) into the left viewport
    WITHOUT stretching or distorting text, and converts mouse clicks back to
    exact original image pixel coordinates.
    """
    def __init__(self, orig_w, orig_h, view_w=VIEWPORT_W, view_h=VIEWPORT_H):
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.view_w = view_w
        self.view_h = view_h
        
        scale_w = (view_w - 20) / float(orig_w)
        scale_h = (view_h - 20) / float(orig_h)
        self.scale = min(scale_w, scale_h)
        
        self.disp_w = int(round(orig_w * self.scale))
        self.disp_h = int(round(orig_h * self.scale))
        
        self.pad_x = (view_w - self.disp_w) // 2
        self.pad_y = (view_h - self.disp_h) // 2

    def canvas_to_image(self, cx, cy):
        """Converts mouse (cx, cy) on canvas to original frame (ix, iy)."""
        ix = int(round((cx - self.pad_x) / self.scale))
        iy = int(round((cy - self.pad_y) / self.scale))
        ix = int(np.clip(ix, 0, self.orig_w - 1))
        iy = int(np.clip(iy, 0, self.orig_h - 1))
        return ix, iy

    def image_to_canvas(self, ix, iy):
        """Converts original frame (ix, iy) to canvas coordinates."""
        cx = int(round(ix * self.scale)) + self.pad_x
        cy = int(round(iy * self.scale)) + self.pad_y
        return cx, cy

    def is_inside_image(self, cx, cy):
        return (self.pad_x <= cx <= self.pad_x + self.disp_w and
                self.pad_y <= cy <= self.pad_y + self.disp_h)


# ==============================================================================
# 3. CUSTOM MOUSE HANDLER (DRAG DIRECTLY TO DRAW BOX)
# ==============================================================================
class MouseAnnotator:
    def __init__(self, transform: ViewportTransform):
        self.transform = transform
        self.dragging = False
        self.start_pt = (0, 0)      # Canvas coords
        self.curr_pt = (0, 0)       # Canvas coords
        self.mouse_pos = (0, 0)     # Current cursor for crosshairs
        self.pending_box = None     # (xmin, ymin, xmax, ymax) in ORIGINAL image coords
        self.right_click_undo = False

    def callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            # Right-click instantly triggers Undo of the last box
            self.right_click_undo = True
            self.dragging = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.transform.is_inside_image(x, y):
                self.dragging = True
                self.start_pt = (x, y)
                self.curr_pt = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.curr_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.dragging = False
                self.curr_pt = (x, y)
                ix1, iy1 = self.transform.canvas_to_image(*self.start_pt)
                ix2, iy2 = self.transform.canvas_to_image(*self.curr_pt)
                xmin, xmax = min(ix1, ix2), max(ix1, ix2)
                ymin, ymax = min(iy1, iy2), max(iy1, iy2)
                
                # Require minimum 10x10 pixel box in original resolution
                if (xmax - xmin) >= 10 and (ymax - ymin) >= 10:
                    self.pending_box = (xmin, ymin, xmax, ymax)


# ==============================================================================
# 4. KINEMATIC & OPTICAL-FLOW SYNCHRONIZATION ENGINE
# ==============================================================================
def detect_video_motion_onset(window_name, session_prefix, video_path, max_scan_seconds=15.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 4.5, 30.0, 0.0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    max_frames = int(min(total_frames, max_scan_seconds * fps))
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 4.5, fps, video_duration, total_frames

    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (320, 180)), cv2.COLOR_BGR2GRAY)
    motion_scores = []

    step = max(1, int(fps // 6))  # Sample ~6 times per second for fast startup
    for f_idx in range(step, max_frames, step):
        show_status_screen(
            window_name,
            f"Synchronizing Session: {session_prefix}",
            f"Scanning initial stationary frames for motion onset ({f_idx}/{max_frames})...",
            progress_ratio=f_idx / max(1, max_frames)
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(cv2.resize(curr_frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        
        roi_prev = prev_gray[72:, :]
        roi_curr = curr_gray[72:, :]
        diff = cv2.absdiff(roi_curr, roi_prev)
        motion_scores.append((f_idx / fps, float(np.mean(diff))))
        prev_gray = curr_gray

    cap.release()

    if len(motion_scores) < 4:
        return 4.5, fps, video_duration, total_frames

    initial_scores = [s for t, s in motion_scores if t <= 2.0]
    baseline = np.median(initial_scores) if initial_scores else motion_scores[0][1]
    std_dev = np.std(initial_scores) if len(initial_scores) > 2 else 1.0
    threshold = max(baseline + 3.0 * std_dev, baseline * 1.6, 2.5)

    for i in range(len(motion_scores) - 2):
        if (motion_scores[i][1] > threshold and 
            motion_scores[i+1][1] > threshold and 
            motion_scores[i+2][1] > threshold):
            return motion_scores[i][0], fps, video_duration, total_frames

    return 4.5, fps, video_duration, total_frames


def synchronize_session(window_name, session):
    show_status_screen(window_name, f"Loading GPS Telemetry: {session['prefix']}", "Reading and filtering 1Hz satellite logs...", 0.1)
    
    df = pd.read_csv(session['gps_path'])
    df.columns = [c.strip() for c in df.columns]
    df['utc_time'] = pd.to_datetime(df['date time']).dt.tz_localize(UTC_ZONE)
    df.sort_values('utc_time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['smooth_speed'] = df['speed(m/s)'].rolling(window=3, min_periods=1, center=True).median()

    gps_t0 = df['utc_time'].iloc[0]
    gps_tend = df['utc_time'].iloc[-1]
    gps_duration = (gps_tend - gps_t0).total_seconds()

    tau_video_move, fps, video_duration, total_frames = detect_video_motion_onset(
        window_name, session['prefix'], session['video_path']
    )

    total_slack = gps_duration - video_duration
    min_start, max_start = session['expected_start_delay_range']
    min_end, _ = session['expected_end_delay_range']

    slack_estimated_delay = np.clip(total_slack * (4.5 / 8.0), min_start, max_start) if total_slack > 0 else 4.5

    moving_rows = df[df['smooth_speed'] >= session['motion_speed_threshold']]
    if not moving_rows.empty:
        t_gps_move = moving_rows['utc_time'].iloc[0]
        gps_move_elapsed = (t_gps_move - gps_t0).total_seconds()
        kinematic_delay = gps_move_elapsed - tau_video_move
    else:
        kinematic_delay = slack_estimated_delay

    if min_start <= kinematic_delay <= max_start:
        final_start_delay = 0.65 * kinematic_delay + 0.35 * slack_estimated_delay
    else:
        final_start_delay = slack_estimated_delay

    if total_slack > min_end:
        final_start_delay = min(final_start_delay, total_slack - min_end)
    final_start_delay = max(0.0, final_start_delay)

    video_start_utc = gps_t0 + timedelta(seconds=final_start_delay)
    return df, video_start_utc, fps, total_frames, final_start_delay


def interpolate_gps_at_timestamp(df, target_utc):
    times = df['utc_time']
    if target_utc <= times.iloc[0]:
        r = df.iloc[0]
        return float(r['latitude']), float(r['longitude']), float(r['speed(m/s)']), float(r['accuracy(m)'])
    if target_utc >= times.iloc[-1]:
        r = df.iloc[-1]
        return float(r['latitude']), float(r['longitude']), float(r['speed(m/s)']), float(r['accuracy(m)'])

    idx_after = int(times.searchsorted(target_utc))
    idx_before = idx_after - 1

    t0 = times.iloc[idx_before]
    t1 = times.iloc[idx_after]
    total_dt = (t1 - t0).total_seconds()
    if total_dt <= 0:
        r = df.iloc[idx_before]
        return float(r['latitude']), float(r['longitude']), float(r['speed(m/s)']), float(r['accuracy(m)'])

    alpha = (target_utc - t0).total_seconds() / total_dt

    if df['smooth_speed'].iloc[idx_before] < 0.25 and df['smooth_speed'].iloc[idx_after] < 0.25:
        r = df.iloc[idx_before] if alpha < 0.5 else df.iloc[idx_after]
        return float(r['latitude']), float(r['longitude']), float(r['speed(m/s)']), float(r['accuracy(m)'])

    lat = (1 - alpha) * df['latitude'].iloc[idx_before] + alpha * df['latitude'].iloc[idx_after]
    lon = (1 - alpha) * df['longitude'].iloc[idx_before] + alpha * df['longitude'].iloc[idx_after]
    spd = (1 - alpha) * df['speed(m/s)'].iloc[idx_before] + alpha * df['speed(m/s)'].iloc[idx_after]
    acc = (1 - alpha) * df['accuracy(m)'].iloc[idx_before] + alpha * df['accuracy(m)'].iloc[idx_after]
    return float(lat), float(lon), float(spd), float(acc)


# ==============================================================================
# 5. PASCAL VOC (RD2022) XML GENERATOR
# ==============================================================================
def save_voc_xml(filename, img_shape, boxes, gps_data, is_clean):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'images'
    ET.SubElement(annotation, 'filename').text = filename
    
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'RD2022_Custom_India'
    ET.SubElement(source, 'timestamp_ist').text = str(gps_data['timestamp_ist'])
    ET.SubElement(source, 'latitude').text = f"{gps_data['latitude']:.8f}"
    ET.SubElement(source, 'longitude').text = f"{gps_data['longitude']:.8f}"
    ET.SubElement(source, 'speed_mps').text = f"{gps_data['speed']:.3f}"
    ET.SubElement(source, 'gps_accuracy_m').text = f"{gps_data['accuracy']:.2f}"
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_shape[1])
    ET.SubElement(size, 'height').text = str(img_shape[0])
    ET.SubElement(size, 'depth').text = str(img_shape[2])
    ET.SubElement(annotation, 'segmented').text = '0'
    
    if is_clean:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = 'Clean'
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = '1'
        ET.SubElement(bndbox, 'ymin').text = '1'
        ET.SubElement(bndbox, 'xmax').text = '2'
        ET.SubElement(bndbox, 'ymax').text = '2'
    else:
        for (xmin, ymin, xmax, ymax), class_id in boxes:
            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = class_id 
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(xmin))
            ET.SubElement(bndbox, 'ymin').text = str(int(ymin))
            ET.SubElement(bndbox, 'xmax').text = str(int(xmax))
            ET.SubElement(bndbox, 'ymax').text = str(int(ymax))
            
    xml_filename = filename.replace('.jpg', '.xml')
    xml_filepath = XML_FOLDER / xml_filename
    
    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ", level=0)
    tree.write(str(xml_filepath), encoding='utf-8', xml_declaration=True)


# ==============================================================================
# 6. SPLIT-SCREEN DASHBOARD RENDERER (ZERO DISTORTION)
# ==============================================================================
def render_dashboard(frame, transform: ViewportTransform, mouse: MouseAnnotator,
                     session_prefix, sample_idx, total_samples, frame_idx,
                     gps_dict, boxes, status_banner, mode="ANNOTATE"):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), (30, 30, 34), dtype=np.uint8)

    # 1. Draw aspect-ratio-preserved road image in Left Viewport
    resized_img = cv2.resize(frame, (transform.disp_w, transform.disp_h), interpolation=cv2.INTER_AREA)
    y1, y2 = transform.pad_y, transform.pad_y + transform.disp_h
    x1, x2 = transform.pad_x, transform.pad_x + transform.disp_w
    canvas[y1:y2, x1:x2] = resized_img

    # Subtle border around the video frame
    cv2.rectangle(canvas, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (90, 90, 100), 1)

    # 2. Draw confirmed bounding boxes
    for (ix1, iy1, ix2, iy2), cid in boxes:
        cx1, cy1 = transform.image_to_canvas(ix1, iy1)
        cx2, cy2 = transform.image_to_canvas(ix2, iy2)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 60, 255), 2)
        
        # High-contrast label pill above box
        (tw, th), _ = cv2.getTextSize(cid, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        lbl_y = max(y1 + th + 6, cy1)
        cv2.rectangle(canvas, (cx1, lbl_y - th - 8), (cx1 + tw + 12, lbl_y), (0, 60, 255), -1)
        cv2.putText(canvas, cid, (cx1 + 6, lbl_y - 4), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Draw active mouse drag box or crosshair
    mx, my = mouse.mouse_pos
    if mode == "ANNOTATE" and transform.is_inside_image(mx, my):
        # Crosshair guide lines
        cv2.line(canvas, (x1, my), (x2, my), (180, 255, 180), 1)
        cv2.line(canvas, (mx, y1), (mx, y2), (180, 255, 180), 1)

    if mouse.dragging:
        sx, sy = mouse.start_pt
        ex, ey = mouse.curr_pt
        cv2.rectangle(canvas, (sx, sy), (ex, ey), (0, 255, 0), 2)

    if mouse.pending_box is not None:
        px1, py1, px2, py2 = mouse.pending_box
        cx1, cy1 = transform.image_to_canvas(px1, py1)
        cx2, cy2 = transform.image_to_canvas(px2, py2)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 255, 255), 3)

    # 4. Draw Right Sidebar Panel (X = 960 to 1360)
    sx0 = VIEWPORT_W
    cv2.rectangle(canvas, (sx0, 0), (CANVAS_W, CANVAS_H), (18, 18, 22), -1)
    cv2.line(canvas, (sx0, 0), (sx0, CANVAS_H), (80, 80, 95), 2)

    pad = sx0 + 20
    # Header & Progress
    cv2.putText(canvas, "RD2022 ANNOTATOR", (pad, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Session : {session_prefix}   (Sample {sample_idx}/{total_samples})",
                (pad, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1, cv2.LINE_AA)

    # Progress bar
    prog = sample_idx / max(1, total_samples)
    cv2.rectangle(canvas, (pad, 80), (CANVAS_W - 20, 96), (50, 50, 60), -1)
    cv2.rectangle(canvas, (pad, 80), (pad + int((SIDEBAR_W - 40) * prog), 96), (0, 200, 120), -1)

    # Telemetry Card
    cv2.rectangle(canvas, (pad, 110), (CANVAS_W - 20, 265), (32, 32, 40), -1)
    cv2.rectangle(canvas, (pad, 110), (CANVAS_W - 20, 265), (70, 70, 85), 1)

    is_moving = gps_dict['speed'] >= 0.35
    motion_txt = "MOVING" if is_moving else "STATIONARY"
    motion_col = (0, 230, 110) if is_moving else (0, 170, 255)

    cv2.putText(canvas, f"Frame #{frame_idx:06d}   [{motion_txt}]", (pad + 12, 138),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, motion_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"IST  : {gps_dict['timestamp_ist']}", (pad + 12, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Lat  : {gps_dict['latitude']:.7f}", (pad + 12, 196),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Lon  : {gps_dict['longitude']:.7f}", (pad + 12, 224),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Spd  : {gps_dict['speed']:.2f} m/s  (Acc: {gps_dict['accuracy']:.1f}m)", (pad + 12, 252),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 230, 255), 1, cv2.LINE_AA)

    # Status / Feedback Banner
    cv2.rectangle(canvas, (pad, 278), (CANVAS_W - 20, 322), (45, 45, 25), -1)
    cv2.rectangle(canvas, (pad, 278), (CANVAS_W - 20, 322), (0, 210, 255), 1)
    cv2.putText(canvas, status_banner[:42], (pad + 10, 306),
                cv2.FONT_HERSHEY_DUPLEX, 0.56, (0, 240, 255), 1, cv2.LINE_AA)

    # Controls or Class Selection Menu
    if mode == "ANNOTATE":
        cv2.putText(canvas, f"BOXES ON FRAME: {len(boxes)}", (pad, 355),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 160) if boxes else (160, 160, 160), 1, cv2.LINE_AA)

        instructions = [
            ("MOUSE DRAG", "Draw Damage Box Directly"),
            ("RIGHT CLICK", "Undo Last Drawn Box"),
            ("[C] Key", "Mark Road CLEAN & Save"),
            ("[N] / SPACE", "Save Drawn Boxes & Next"),
            ("[Z] Key", "Undo Last Box"),
            ("[S] Key", "Skip Blurry/Bad Frame"),
            ("[Q] / ESC", "Save Audit Log & Quit")
        ]
        y_cursor = 392
        for k_str, desc_str in instructions:
            cv2.putText(canvas, k_str, (pad, y_cursor), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"- {desc_str}", (pad + 125, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (230, 230, 230), 1, cv2.LINE_AA)
            y_cursor += 32

        # Mini class cheat-sheet at bottom of sidebar
        cv2.line(canvas, (pad, 625), (CANVAS_W - 20, 625), (60, 60, 75), 1)
        cv2.putText(canvas, "Tip: Just click & drag on the road", (pad, 655),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (160, 220, 160), 1, cv2.LINE_AA)
        cv2.putText(canvas, "to draw a box. No 'D' key needed!", (pad, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (160, 220, 160), 1, cv2.LINE_AA)

    elif mode == "CLASS_SELECT":
        cv2.rectangle(canvas, (pad, 335), (CANVAS_W - 20, CANVAS_H - 15), (25, 45, 25), -1)
        cv2.rectangle(canvas, (pad, 335), (CANVAS_W - 20, CANVAS_H - 15), (0, 255, 120), 2)
        cv2.putText(canvas, "SELECT DAMAGE CLASS KEY:", (pad + 12, 365),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 120), 1, cv2.LINE_AA)

        y_cursor = 398
        for k_char, (cid, cdesc) in CLASS_HOTKEYS.items():
            label_line = f"[{k_char.upper()}]  {cid:11s} : {cdesc}"
            cv2.putText(canvas, label_line, (pad + 12, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            y_cursor += 28

        cv2.putText(canvas, "[ENTER/SPACE] Default D40  |  [ESC] Cancel", (pad + 12, CANVAS_H - 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

    return canvas

# ==============================================================================
# 6B. INTERACTIVE SESSION SELECTOR MENU (GUI)
# ==============================================================================
def select_session_gui(window_name, sessions):
    """
    Displays an interactive menu inside the OpenCV window allowing the user
    to choose which specific (video + GPS) pair to process, or 'A' for All.
    """
    selected_idx = 0  # Highlight first session by default

    while True:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), (24, 24, 28), dtype=np.uint8)

        # Title Header
        cv2.putText(canvas, "SELECT VIDEO & GPS SESSION TO ANNOTATE", (80, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Press the Number Key [1-9] (or UP/DOWN arrows + ENTER) to select:", (80, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 200, 200), 1, cv2.LINE_AA)

        # List discovered sessions
        y_start = 180
        box_h = 54
        for idx, s in enumerate(sessions):
            y_top = y_start + idx * (box_h + 12)
            if y_top + box_h > CANVAS_H - 120:
                break  # Prevent overflow if many files exist

            is_selected = (idx == selected_idx)
            bg_col = (40, 75, 40) if is_selected else (36, 36, 44)
            border_col = (0, 255, 120) if is_selected else (80, 80, 95)

            # Check how many frames are already annotated for this session
            existing_xmls = len(list(XML_FOLDER.glob(f"{s['prefix']}_frame_*.xml")))

            cv2.rectangle(canvas, (80, y_top), (CANVAS_W - 80, y_top + box_h), bg_col, -1)
            cv2.rectangle(canvas, (80, y_top), (CANVAS_W - 80, y_top + box_h), border_col, 2)

            label = f"[{idx + 1}]   {s['prefix']}   -->   Video: {s['video_path'].name}   |   GPS: {s['gps_path'].name}"
            status = f"(Annotated so far: {existing_xmls} frames)"

            cv2.putText(canvas, label, (105, y_top + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, status, (CANVAS_W - 440, y_top + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 255) if existing_xmls else (160, 160, 160), 1, cv2.LINE_AA)

        # Bottom Controls Bar
        cv2.rectangle(canvas, (80, CANVAS_H - 95), (CANVAS_W - 80, CANVAS_H - 35), (32, 32, 40), -1)
        cv2.rectangle(canvas, (80, CANVAS_H - 95), (CANVAS_W - 80, CANVAS_H - 35), (0, 210, 255), 1)
        footer_text = "[1-9] Select Session Immediately    |    [ENTER] Start Highlighted    |    [A] Process ALL    |    [Q/ESC] Quit"
        cv2.putText(canvas, footer_text, (105, CANVAS_H - 58),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 240, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue

        # Quit
        if key == 27 or chr(key).lower() == 'q':
            return []

        # Process ALL sessions
        if chr(key).lower() == 'a':
            return sessions

        # Enter key launches currently highlighted session
        if key in (13, 32):
            return [sessions[selected_idx]]

        # W / S or Up / Down navigation
        if chr(key).lower() == 'w':
            selected_idx = (selected_idx - 1) % len(sessions)
        elif chr(key).lower() == 's':
            selected_idx = (selected_idx + 1) % len(sessions)

        # Direct number hotkeys (1 to 9)
        if ord('1') <= key <= ord('9'):
            choice = key - ord('1')
            if choice < len(sessions):
                return [sessions[choice]]

# ==============================================================================
# 7. MAIN EXECUTION LOOP
# ==============================================================================
def main():
    window_name = "RD2022 Road Damage Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    all_sessions = discover_sessions()
    if not all_sessions:
        print(f"ERROR: No matching (.mp4 + .txt) pairs found inside {RAW_DIR}")
        return

    # Ask the user which specific video + GPS file pair they want to work on
    sessions = select_session_gui(window_name, all_sessions)
    if not sessions:
        cv2.destroyAllWindows()
        print("No session selected. Exiting.")
        return

    audit_records = []

    for session in sessions:
        df, video_start_utc, fps, total_frames, start_delay = synchronize_session(window_name, session)

        cap = cv2.VideoCapture(str(session['video_path']))
        if not cap.isOpened():
            continue

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        transform = ViewportTransform(orig_w, orig_h)
        mouse = MouseAnnotator(transform)
        cv2.setMouseCallback(window_name, mouse.callback)

        frame_step = max(1, int(round(fps)))  # Exactly 1 frame per second
        sample_indices = list(range(0, total_frames, frame_step))
        total_samples = len(sample_indices)

        quit_all = False
        status_banner = f"Sync OK (+{start_delay:.2f}s offset). Drag mouse to box."

        for s_idx, frame_count in enumerate(sample_indices, start=1):
            image_filename = f"{session['prefix']}_frame_{frame_count:06d}.jpg"
            xml_filepath = XML_FOLDER / image_filename.replace('.jpg', '.xml')

            # Fast resume: skip frames already annotated in earlier sessions
            if xml_filepath.exists():
                continue

            # Show quick jump indicator so user knows it is seeking the video
            show_status_screen(
                window_name,
                f"Loading Frame {frame_count} ({s_idx}/{total_samples})",
                f"Session: {session['prefix']} -> {image_filename}",
                progress_ratio=s_idx / max(1, total_samples)
            )

            # Direct seek to target frame (eliminates sequential decoding lag)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break

            time_offset_seconds = frame_count / fps
            frame_utc = video_start_utc + timedelta(seconds=time_offset_seconds)
            frame_ist = frame_utc.astimezone(IST_ZONE)
            lat, lon, spd, acc = interpolate_gps_at_timestamp(df, frame_utc)

            gps_dict = {
                'timestamp_ist': frame_ist.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'latitude': lat,
                'longitude': lon,
                'speed': spd,
                'accuracy': acc
            }

            boxes = []
            is_clean = False
            mouse.pending_box = None

            # Interactive Frame Loop (30Hz UI refresh for smooth mouse dragging)
            while True:
                if mouse.right_click_undo:
                    mouse.right_click_undo = False
                    if boxes:
                        removed = boxes.pop()
                        status_banner = f"Removed box ({removed[1]})"
                    else:
                        status_banner = "No boxes to undo."

                # Check if user just finished dragging a rectangle with the mouse
                if mouse.pending_box is not None:
                    status_banner = "Box drawn! Press 0-9 or B to assign class."
                    while mouse.pending_box is not None:
                        dash = render_dashboard(
                            frame, transform, mouse, session['prefix'],
                            s_idx, total_samples, frame_count, gps_dict,
                            boxes, status_banner, mode="CLASS_SELECT"
                        )
                        cv2.imshow(window_name, dash)
                        ckey = cv2.waitKey(20) & 0xFF
                        if ckey == 255:
                            continue

                        if ckey == 27:  # ESC cancels current box
                            mouse.pending_box = None
                            status_banner = "Cancelled box. Drag again or press C."
                            break
                        elif ckey in (13, 32):  # Enter or Space defaults to D40
                            boxes.append((mouse.pending_box, 'D40'))
                            mouse.pending_box = None
                            status_banner = "Added D40! Drag more or press N to save."
                            break
                        else:
                            ch = chr(ckey).lower() if ckey < 128 else ''
                            if ch in CLASS_HOTKEYS:
                                chosen_cid = CLASS_HOTKEYS[ch][0]
                                boxes.append((mouse.pending_box, chosen_cid))
                                mouse.pending_box = None
                                status_banner = f"Added {chosen_cid}! Drag more or press N."
                                break

                # Render normal annotation dashboard
                dash = render_dashboard(
                    frame, transform, mouse, session['prefix'],
                    s_idx, total_samples, frame_count, gps_dict,
                    boxes, status_banner, mode="ANNOTATE"
                )
                cv2.imshow(window_name, dash)

                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    continue

                ch = chr(key).lower() if key < 128 else ''

                if ch == 'c':
                    is_clean = True
                    status_banner = f"Saved {image_filename} as CLEAN!"
                    break

                elif ch == 'n' or key in (13, 32):
                    if len(boxes) > 0:
                        status_banner = f"Saved {len(boxes)} damages on {image_filename}!"
                        break
                    else:
                        status_banner = "Draw a box first, or press [C] for Clean!"

                elif ch == 'z':
                    if boxes:
                        removed = boxes.pop()
                        status_banner = f"Undid last box ({removed[1]})"
                    else:
                        status_banner = "No boxes to undo."

                elif ch == 's':
                    boxes = None
                    status_banner = f"Skipped {image_filename}"
                    break

                elif ch == 'q' or key == 27:
                    quit_all = True
                    break

            if quit_all:
                break

            if boxes is not None:
                image_filepath = IMAGES_FOLDER / image_filename
                cv2.imwrite(str(image_filepath), frame)
                save_voc_xml(image_filename, frame.shape, boxes, gps_dict, is_clean)

                audit_records.append({
                    'session': session['prefix'],
                    'frame_file': image_filename,
                    'timestamp_ist': gps_dict['timestamp_ist'],
                    'latitude': lat,
                    'longitude': lon,
                    'speed_mps': spd,
                    'status': 'Clean' if is_clean else f"{len(boxes)}_Damages"
                })

        cap.release()
        if quit_all:
            break

    cv2.destroyAllWindows()
    if audit_records:
        pd.DataFrame(audit_records).to_csv(AUDIT_CSV, index=False)
        print(f"Saved synchronization audit log to: {AUDIT_CSV}")
    print("Pipeline session finished cleanly.")


if __name__ == "__main__":

    main()