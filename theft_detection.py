#!/usr/bin/env python3
"""
theft_detection.py — Accurate Theft Detection (v3)

Alert fires ONLY when all 3 conditions met:
  1. Person confirmed near object for >= INTERACT_MIN_SEC seconds
  2. Object then disappears (gone from detection)
  3. Person physically moves >= PERSON_LEAVE_PX from where countdown started

If person stays seated → false alarm suppressed silently.
"""
import os, sys, cv2, time, threading, numpy as np
from datetime import datetime

# ── Auto-export yolo11s.onnx ─────────────────────────────────────────────────
MODEL_PATH = "yolo11s.onnx"
if not os.path.exists(MODEL_PATH):
    print("[Setup] Exporting yolo11s.onnx (Better detection than nano) …")
    try:
        from ultralytics import YOLO
        YOLO("yolo11s.pt").export(format="onnx")
        print("[Setup] Done.")
    except Exception as e:
        sys.exit(f"[Error] {e}")
import onnxruntime as ort

# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_URL       = "rtsp://admin:cctv%40321@192.168.1.72:554/cam/realmonitor?channel=5&subtype=1"
CONF_THRESH_PERSON = 0.25
CONF_THRESH_CLASS  = {
    63: 0.05,  # Laptop (extremely low to catch difficult angles and foreground laptops)
    67: 0.05,  # Phone (extremely low because cell phones are tiny)
    24: 0.15,  # Backpack
    26: 0.15,  # Handbag
    28: 0.15,  # Suitcase
    73: 0.15,  # Book
}
INTERACT_PX      = 150        # px: person must be this close to object
INTERACT_MIN_SEC = 12.0       # seconds near object before interaction is CONFIRMED
PERSON_LEAVE_PX  = 120        # px: person must move this far for alert to fire
COUNTDOWN_SEC    = 20.0       # seconds after disappear before alert (if person left)
MISS_GRACE_FRAMES = 9         # consecutive absent frames before countdown starts (~3s at 3fps)
OBJ_MAX_GONE     = 150        # inference frames (~50s at 3fps) before slot retires
PER_MAX_GONE     = 60         # inference frames (~20s) before person slot retires
EVIDENCE_DIR     = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# COCO class_id → display label (Step 3: valuable objects to monitor)
OBJECT_CLASSES = {
    63: "Laptop",   24: "Backpack",  26: "Handbag",
    67: "Phone",    28: "Suitcase",  73: "Book",
}

# State labels (Step 12: system status)
ST_NORMAL = "Normal Monitoring"
ST_NEAR   = "Person Near Object"
ST_WATCH  = "Interaction Confirmed"
ST_MISS   = "Object Missing"
ST_THEFT  = "Possible Theft!"

# BGR colours
C_GREEN  = (  0, 200,   0)
C_YELLOW = (  0, 225, 225)
C_ORANGE = (  0, 150, 255)
C_RED    = ( 30,  30, 230)
C_CYAN   = (200, 170,   0)
C_WHITE  = (255, 255, 255)
C_GRAY   = (140, 140, 140)


# ── Centroid Tracker (Step 2: unique IDs) ────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_gone=60, max_dist=200):
        self.nxt = 1
        self.pos  = {}   # id → (cx,cy)
        self.box  = {}   # id → (x1,y1,x2,y2)
        self.gone = {}   # id → frames absent
        self.mg   = max_gone
        self.md   = max_dist

    def _add(self, b):
        cx = (b[0]+b[2])//2; cy = (b[1]+b[3])//2
        i  = self.nxt; self.nxt += 1
        self.pos[i]  = (cx, cy)
        self.box[i]  = tuple(b)
        self.gone[i] = 0

    def _rm(self, i):
        self.pos.pop(i, None)
        self.box.pop(i, None)
        self.gone.pop(i, None)

    def _iou(self, a, b):
        ix = max(0, min(a[2],b[2]) - max(a[0],b[0]))
        iy = max(0, min(a[3],b[3]) - max(a[1],b[1]))
        u  = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy
        return ix*iy / max(u, 1e-6)

    def update(self, bboxes):
        for i in list(self.pos):
            if self.gone[i] > 0:
                self.gone[i] += 1
                if self.gone[i] > self.mg:
                    self._rm(i)
        if not bboxes:
            return
        dets = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in bboxes]
        ids  = list(self.pos.keys())
        if not ids:
            for b in bboxes: self._add(b)
            return
        D = np.array([[np.linalg.norm(np.subtract(self.pos[i], d)) for d in dets] for i in ids])
        ur, uc = set(), set()
        for r in D.min(1).argsort():
            c = int(D[r].argmin())
            if r in ur or c in uc or D[r,c] > self.md: continue
            i = ids[r]
            self.pos[i]  = dets[c]
            self.box[i]  = tuple(bboxes[c])
            self.gone[i] = 0
            ur.add(r); uc.add(c)
        for r, i in enumerate(ids):
            if r not in ur: self.gone[i] = max(self.gone[i], 1)
        for c, b in enumerate(bboxes):
            if c not in uc:
                if not any(self._iou(self.box[i], b) > 0.45
                           for i in self.pos if self.gone[i] == 0):
                    self._add(b)

    def visible(self):
        return {i: self.box[i] for i in self.pos if self.gone[i] == 0}


# ── State Manager (Steps 4-9: full state machine) ────────────────────────────
class StateManager:
    def __init__(self):
        self.st       = {}   # key → state string
        self.pid      = {}   # key → associated person id
        self.istart   = {}   # key → when ST_NEAR phase started
        self.ipos     = {}   # key → person (cx,cy) during interaction
        self.cdstart  = {}   # key → countdown start timestamp
        self.cd_ppos  = {}   # key → person (cx,cy) when countdown started
        self.lpos     = {}   # key → last seen object bbox (x1,y1,x2,y2)

    # Step 6: update while object IS visible
    def process_visible(self, lbl, oid, bbox, people, now):
        key  = (lbl, oid)
        prev = self.st.get(key, ST_NORMAL)
        self.lpos[key] = bbox
        cx = (bbox[0]+bbox[2])//2; cy = (bbox[1]+bbox[3])//2

        # Step 8: object reappeared → cancel any countdown
        if prev == ST_MISS:
            self.st[key] = ST_NORMAL
            for d in (self.cdstart, self.cd_ppos, self.pid, self.istart, self.ipos):
                d.pop(key, None)
            return ST_NORMAL

        # Step 4: find closest person
        best_pid, best_d, best_p = None, float("inf"), None
        for pid, pb in people.items():
            px = (pb[0]+pb[2])//2; py = (pb[1]+pb[3])//2
            d  = float(np.linalg.norm([cx-px, cy-py]))
            if d < best_d:
                best_d = d; best_pid = pid; best_p = (px, py)

        # Step 5: interaction detection
        if best_pid is not None and best_d <= INTERACT_PX:
            if prev == ST_NORMAL:
                self.st[key]     = ST_NEAR
                self.pid[key]    = best_pid
                self.istart[key] = now
                self.ipos[key]   = best_p
            elif prev == ST_NEAR:
                self.ipos[key] = best_p
                self.pid[key]  = best_pid
                # Confirm interaction only after sustained proximity
                if now - self.istart.get(key, now) >= INTERACT_MIN_SEC:
                    self.st[key] = ST_WATCH
            elif prev == ST_WATCH:
                self.ipos[key] = best_p
                self.pid[key]  = best_pid
        else:
            # Person moved away
            if prev in (ST_NEAR, ST_WATCH):
                self.st[key] = ST_NORMAL
                for d in (self.pid, self.istart, self.ipos): d.pop(key, None)

        return self.st.get(key, ST_NORMAL)

    # Step 7: update while object is NOT visible
    def process_missing(self, lbl, oid, people, now):
        """Returns (theft_triggered: bool, payload: dict or None)"""
        key  = (lbl, oid)
        prev = self.st.get(key, ST_NORMAL)

        # Step 7: confirmed interaction + object disappeared → start countdown
        if prev == ST_WATCH:
            self.st[key]    = ST_MISS
            self.cdstart[key] = now
            pid = self.pid.get(key)
            if pid and pid in people:
                pb = people[pid]
                self.cd_ppos[key] = ((pb[0]+pb[2])//2, (pb[1]+pb[3])//2)
            else:
                self.cd_ppos[key] = None

        elif prev == ST_MISS:
            elapsed = now - self.cdstart.get(key, now)
            if elapsed >= COUNTDOWN_SEC:
                pid = self.pid.get(key)
                # Step 13 false-alarm check: did person physically move?
                person_moved = True   # if invisible → assume they left
                ini_p = self.cd_ppos.get(key)
                if ini_p:
                    for _pid, pb in people.items():
                        cur_p = ((pb[0]+pb[2])//2, (pb[1]+pb[3])//2)
                        if float(np.linalg.norm(np.subtract(cur_p, ini_p))) <= PERSON_LEAVE_PX:
                            person_moved = False
                            break

                if person_moved:
                    # Step 9: THEFT confirmed
                    self.st[key] = ST_THEFT
                    return True, dict(
                        suspect_id = pid or "?",
                        category   = lbl,
                        oid        = oid,
                        last_pos   = self.lpos.get(key),
                        timestamp  = datetime.now().strftime("%H:%M:%S"),
                        date       = datetime.now().strftime("%Y-%m-%d"),
                    )
                else:
                    # Person never moved → suppress false alarm
                    self.st[key] = ST_NORMAL
                    for d in (self.cdstart, self.cd_ppos, self.pid): d.pop(key, None)
                    return False, {"suppressed": True}

        return False, None

    def get_remaining(self, lbl, oid, now):
        key = (lbl, oid)
        if self.st.get(key) == ST_MISS:
            return max(0.0, COUNTDOWN_SEC - (now - self.cdstart.get(key, now)))
        return None

    def clear(self, lbl, oid):
        key = (lbl, oid)
        self.st[key] = ST_NORMAL
        for d in (self.cdstart, self.cd_ppos, self.pid, self.istart, self.ipos):
            d.pop(key, None)


# ── Shared State ──────────────────────────────────────────────────────────────
_fl = threading.Lock()   # frame lock
_rl = threading.Lock()   # render lock

_frame    = None
_fid      = 0
_pid_inf  = 0
_cam_fps  = 0.0
_inf_fps  = 0.0

_reset_flag = threading.Event()

# Written by inference thread, read by display thread
_R = dict(
    people   = {},    # pid → (x1,y1,x2,y2)
    items    = [],    # non-NORMAL objects: list of dicts
    alerts   = [],    # active theft alerts
    history  = [],    # event log (last 15)
    status   = ST_NORMAL,
    suppressed = 0,   # count of suppressed false alarms
)


# ── Camera Thread ─────────────────────────────────────────────────────────────
def _cam_thread():
    global _frame, _fid, _cam_fps
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;5000000|timeout;5000000")
    cap = None; fails = 0

    def _open():
        nonlocal cap
        if cap: cap.release()
        print("[Camera] Connecting …")
        cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    _open()
    t0 = time.time(); fc = 0
    while True:
        ok = cap.grab() if (cap and cap.isOpened()) else False
        if not ok:
            fails += 1
            if fails > 80: _open(); fails = 0
            time.sleep(0.01); continue
        fails = 0
        ok, frm = cap.retrieve()
        if not ok or frm is None: continue
        oh, ow = frm.shape[:2]
        if ow > 800: frm = cv2.resize(frm, (800, int(oh * 800 / ow)))
        with _fl: _frame = frm; _fid += 1
        fc += 1
        now = time.time()
        if now - t0 >= 2: _cam_fps = fc / (now - t0); fc = 0; t0 = now


# ── Letterbox ─────────────────────────────────────────────────────────────────
def _lb(img, sz=640):
    h, w = img.shape[:2]; s = min(sz/h, sz/w)
    nh, nw = int(h*s), int(w*s)
    c = np.full((sz, sz, 3), 114, np.uint8)
    pl, pt = (sz-nw)//2, (sz-nh)//2
    c[pt:pt+nh, pl:pl+nw] = cv2.resize(img, (nw, nh))
    return c, s, pl, pt


# ── Evidence Saving (Step 10) ─────────────────────────────────────────────────
_rec_lock = threading.Lock(); _recorder = None; _rec_stop = 0.; _rec_on = False

def _save_shot(frm, alert):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p  = os.path.join(EVIDENCE_DIR, f"theft_{ts}_P{alert['suspect_id']}.jpg")
    cv2.imwrite(p, frm); print(f"[Evidence] Screenshot → {p}")

def _start_rec(frm):
    global _recorder, _rec_stop, _rec_on
    with _rec_lock:
        if _rec_on: return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p  = os.path.join(EVIDENCE_DIR, f"evidence_{ts}.avi")
        h, w = frm.shape[:2]
        _recorder = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"XVID"), 15, (w, h))
        _rec_stop = time.time() + 30; _rec_on = True
        print(f"[Evidence] Recording → {p} (30s)")

def _write_ev(frm):
    global _rec_on, _recorder
    with _rec_lock:
        if not _rec_on: return
        if time.time() > _rec_stop:
            if _recorder: _recorder.release(); _recorder = None
            _rec_on = False; return
        if _recorder: _recorder.write(frm)


# ── Inference Thread ──────────────────────────────────────────────────────────
def _inf_thread():
    global _pid_inf, _inf_fps

    print("[AI] Loading yolo11s.onnx …")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0].name

    # Step 1+2: separate tracker per class
    p_trk  = CentroidTracker(max_gone=PER_MAX_GONE, max_dist=300)
    o_trks = {lb: CentroidTracker(max_gone=OBJ_MAX_GONE, max_dist=10000)
              for lb in set(OBJECT_CLASSES.values())}
    smgr   = StateManager()
    alerted = set()   # keys that already fired evidence

    MIN_INT = 1/3.0; last_t = 0.; last_fr = -1; t0 = time.time(); fc = 0; sup_cnt = 0

    while True:
        if _reset_flag.is_set():
            alerted.clear(); _reset_flag.clear()

        now = time.time()
        if now - last_t < MIN_INT: time.sleep(0.005); continue
        with _fl: frm = _frame; cur = _fid
        if frm is None or cur == last_fr: time.sleep(0.01); continue
        last_t = time.time()

        # ── Run YOLO ─────────────────────────────────────────────────────────
        canvas, sc, pl, pt = _lb(frm)
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        raw  = sess.run(None, {inp: blob})[0][0]   # shape (84, 8400)
        oh, ow = frm.shape[:2]

        def _parse(class_index, conf_thresh, person_filter=False):
            scores = raw[4 + class_index]
            mask   = scores >= conf_thresh
            if not mask.any(): return []
            conf_vals = scores[mask].tolist()
            cx, cy, bw, bh = raw[0][mask], raw[1][mask], raw[2][mask], raw[3][mask]
            pre = []
            pre_scores = []
            for (a, b, c_, d), s in zip(zip(
                np.clip(((cx-bw/2)-pl)/sc, 0, ow).astype(int),
                np.clip(((cy-bh/2)-pt)/sc, 0, oh).astype(int),
                np.clip(((cx+bw/2)-pl)/sc, 0, ow).astype(int),
                np.clip(((cy+bh/2)-pt)/sc, 0, oh).astype(int)), conf_vals):
                if person_filter and (d-b) > 0 and (c_-a)/(d-b) < 0.22: continue
                pre.append([a, b, c_, d])
                pre_scores.append(float(s))
            if not pre: return []
            # NMS — removes duplicate boxes for the same person/object
            xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in pre]
            idxs = cv2.dnn.NMSBoxes(xywh, pre_scores, conf_thresh, 0.20)
            if len(idxs) == 0: return []
            idxs = idxs.flatten() if hasattr(idxs, 'flatten') else [i[0] for i in idxs]
            
            # Strict spatial suppression for objects (fixes multiple detections of parts of the same laptop)
            final_boxes = []
            idxs_sorted = sorted(idxs.tolist(), key=lambda i: pre_scores[i], reverse=True)
            for i in idxs_sorted:
                b = pre[i]
                cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                suppress = False
                for fb in final_boxes:
                    fcx, fcy = (fb[0]+fb[2])/2, (fb[1]+fb[3])/2
                    dist = ((cx-fcx)**2 + (cy-fcy)**2)**0.5
                    if class_index > 0:  # Objects only
                        # Suppress if centers are close, or if center falls inside a higher-confidence box
                        if dist < 80 or (fb[0] < cx < fb[2] and fb[1] < cy < fb[3]):
                            suppress = True
                            break
                if not suppress:
                    final_boxes.append(b)
            return final_boxes

        # Step 1: detect people (COCO class 0)
        p_trk.update(_parse(0, CONF_THRESH_PERSON, person_filter=True))
        active_p = p_trk.visible()

        # Step 1: detect valuable objects
        cat_raw = {}
        for cid, lbl in OBJECT_CLASSES.items():
            bxs = _parse(cid, CONF_THRESH_CLASS.get(cid, 0.20))
            if bxs: cat_raw.setdefault(lbl, []).extend(bxs)
            
        # Filter out objects being carried/worn (only for bags)
        for lbl in list(cat_raw.keys()):
            valid_bxs = []
            for b in cat_raw[lbl]:
                is_carried = False
                if lbl in ["Backpack", "Handbag", "Suitcase"]:
                    obj_area = (b[2]-b[0]) * (b[3]-b[1])
                    for pb in active_p.values():
                        ix = max(0, min(b[2], pb[2]) - max(b[0], pb[0]))
                        iy = max(0, min(b[3], pb[3]) - max(b[1], pb[1]))
                        if ix > 0 and iy > 0:
                            if (ix * iy) / max(obj_area, 1) > 0.40:
                                is_carried = True
                                break
                if not is_carried:
                    valid_bxs.append(b)
            cat_raw[lbl] = valid_bxs

        for lbl, trk in o_trks.items():
            trk.update(cat_raw.get(lbl, []))

        # ── State machine ─────────────────────────────────────────────────────
        now_t     = time.time()
        items_out = []
        new_alerts = list(_R["alerts"])
        new_hist   = list(_R["history"])
        worst      = ST_NORMAL
        rank       = {ST_NORMAL:0, ST_NEAR:1, ST_WATCH:2, ST_MISS:3, ST_THEFT:4}

        for lbl, trk in o_trks.items():
            vis = trk.visible()
            for oid in list(trk.pos.keys()):
                key = (lbl, oid)

                if oid in vis:
                    st = smgr.process_visible(lbl, oid, vis[oid], active_p, now_t)
                else:
                    # Grace period: only process as missing after MISS_GRACE_FRAMES
                    # consecutive absent frames — prevents 1-2 frame YOLO misses
                    # from triggering a countdown for a laptop that's still there.
                    frames_gone = trk.gone.get(oid, 0)
                    if frames_gone >= MISS_GRACE_FRAMES:
                        triggered, payload = smgr.process_missing(lbl, oid, active_p, now_t)
                        if triggered and key not in alerted:
                            alerted.add(key)
                            new_alerts.append(payload)
                            new_hist.append(payload)
                            # Step 10: save evidence
                            threading.Thread(target=_save_shot,
                                             args=(frm.copy(), payload), daemon=True).start()
                            threading.Thread(target=_start_rec,
                                             args=(frm.copy(),), daemon=True).start()
                        elif payload and payload.get("suppressed"):
                            sup_cnt += 1
                    st = smgr.st.get(key, ST_NORMAL)

                # Add all tracked items to render list (so user can see they are detected)
                items_out.append(dict(
                    lbl       = lbl,
                    oid       = oid,
                    st        = st,
                    box       = smgr.lpos.get(key),
                    pid       = smgr.pid.get(key),
                    remaining = smgr.get_remaining(lbl, oid, now_t),
                ))

                if rank.get(st, 0) > rank.get(worst, 0): worst = st

        with _rl:
            _R["people"]     = active_p
            _R["items"]      = items_out
            _R["alerts"]     = new_alerts
            _R["history"]    = new_hist[-15:]
            _R["status"]     = worst
            _R["suppressed"] = sup_cnt
        _pid_inf = cur; last_fr = cur

        fc += 1; now2 = time.time()
        if now2 - t0 >= 2: _inf_fps = fc / (now2 - t0); fc = 0; t0 = now2


# ── Draw Helpers ──────────────────────────────────────────────────────────────
def _put(img, txt, pos, sc=0.5, col=C_WHITE, th=1):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

def _fade(img, x1, y1, x2, y2, alpha=0.6, col=(0, 0, 0)):
    x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(img.shape[1],x2), min(img.shape[0],y2)
    if x2 <= x1 or y2 <= y1: return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, col, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
    img[y1:y2, x1:x2] = roi

def _st_col(st):
    return {ST_NORMAL:C_GREEN, ST_NEAR:C_YELLOW, ST_WATCH:C_ORANGE,
            ST_MISS:C_ORANGE, ST_THEFT:C_RED}.get(st, C_WHITE)


# ── Main Display Loop ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=_cam_thread,  daemon=True).start()
    threading.Thread(target=_inf_thread,  daemon=True).start()
    print("Controls:  Q = Quit    R = Reset Alerts")
    print(f"Theft requires: {INTERACT_MIN_SEC}s interaction + {COUNTDOWN_SEC}s missing + person moved {PERSON_LEAVE_PX}px")

    TOP_H = 36; BOT_H = 55
    lr_fid = -1; lr_pid = -1; lr_t = time.time()
    d_fps = 0.; t_fps = time.time()

    while True:
        # Rate limit display to ~20fps
        now = time.time()
        if now - lr_t < 0.05:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k in (ord('r'), ord('R')):
                with _rl: _R["alerts"] = []; _reset_flag.set()
            continue
        lr_t = now

        with _fl:  frm = _frame
        with _rl:  data = dict(_R); cf_fps = _cam_fps; if_fps = _inf_fps

        if frm is None: cv2.waitKey(5); continue

        disp = frm.copy(); H, W = disp.shape[:2]; now_t = time.time()

        # ── 1. People: subtle thin green boxes ───────────────────────────────
        for pid, (x1, y1, x2, y2) in data["people"].items():
            cv2.rectangle(disp, (x1,y1), (x2,y2), C_GREEN, 1, cv2.LINE_AA)
            _put(disp, f"P{pid}", (max(x1,0), max(y1-3, 12)), 0.38, C_GREEN, 1)

        # ── 2. Object items (only non-NORMAL state) ───────────────────────────
        for item in data["items"]:
            st  = item["st"]; box = item["box"]
            lbl = item["lbl"]; oid = item["oid"]
            rem = item["remaining"]; pid_ = item["pid"]
            col = _st_col(st)

            if box:
                x1, y1, x2, y2 = box
                cx = (x1+x2)//2; cy = (y1+y2)//2
                thick = 3 if st in (ST_WATCH, ST_MISS, ST_THEFT) else (2 if st == ST_NEAR else 1)
                cv2.rectangle(disp, (x1,y1), (x2,y2), col, thick, cv2.LINE_AA)

                # Only draw text tags if the object is missing or stolen
                if st in (ST_MISS, ST_THEFT):
                    tag = lbl
                    tw = len(tag)*9 + 8
                    cv2.rectangle(disp, (x1, max(y1-22,0)), (x1+tw, y1), (0,0,0), -1)
                    _put(disp, tag, (x1+4, max(y1-6,13)), 0.45, col, 1)

                # Association line: person → object
                if pid_ and pid_ in data["people"]:
                    pb = data["people"][pid_]
                    px = (pb[0]+pb[2])//2; py = (pb[1]+pb[3])//2
                    cv2.line(disp, (cx,cy), (px,py), col, 1, cv2.LINE_AA)

                # Countdown marker for missing objects
                if st == ST_MISS:
                    flash = int(now_t * 2.5) % 2 == 0
                    if flash:
                        cv2.rectangle(disp, (x1-4,y1-4), (x2+4,y2+4), C_RED, 3, cv2.LINE_AA)
                        cv2.drawMarker(disp, (cx,cy), C_RED, cv2.MARKER_CROSS, 28, 3, cv2.LINE_AA)
                    if rem is not None:
                        cd_text = f"Missing! Alert in {rem:.0f}s"
                        tw2 = len(cd_text)*9 + 8
                        cv2.rectangle(disp, (x1, y2), (x1+tw2, y2+24), (0,0,0), -1)
                        _put(disp, cd_text, (x1+4, y2+17), 0.5, C_RED, 2)
                    if pid_:
                        _put(disp, f"Last assoc: P{pid_}", (x1, y1-28), 0.38, C_RED, 1)

        # ── 3. Top status bar ─────────────────────────────────────────────────
        _fade(disp, 0, 0, W, TOP_H, 0.72)
        sc_str = data["status"]; sc_col = _st_col(sc_str)
        cv2.line(disp, (0, TOP_H), (W, TOP_H), sc_col, 2)
        _put(disp, f"STATUS: {sc_str}", (10, 24), 0.62, sc_col, 2)
        ts_now = datetime.now().strftime("%H:%M:%S")
        _put(disp, f"Cam:{cf_fps:.0f}  AI:{if_fps:.1f}fps  {ts_now}", (W-240, 24), 0.42, C_WHITE, 1)

        # ── 4. Bottom info bar ────────────────────────────────────────────────
        _fade(disp, 0, H-BOT_H, W, H, 0.68)
        cv2.line(disp, (0, H-BOT_H), (W, H-BOT_H), (80,80,80), 1)

        hist = data["history"]
        for i, h in enumerate(reversed(hist[-2:])):
            y = H - BOT_H + 18 + i*22
            _put(disp, f"[{h['timestamp']}] P{h['suspect_id']} took {h['category']}", (10, y), 0.42, C_YELLOW, 1)

        sup = data["suppressed"]
        if sup > 0:
            _put(disp, f"False alarms blocked: {sup}", (W-220, H-BOT_H+18), 0.38, C_GRAY, 1)

        d_fps = 0.9*d_fps + 0.1*(1/max(now_t-t_fps, 1e-6)); t_fps = now_t
        _put(disp, f"UI:{d_fps:.0f}fps", (W-75, H-8), 0.35, C_GRAY, 1)

        # ── 5. Theft alert bar (Step 10+11) — clean bottom strip ─────────────
        if data["alerts"]:
            a    = data["alerts"][0]
            ay1  = H - BOT_H - 85
            ay2  = H - BOT_H
            _fade(disp, 0, ay1, W, ay2, 0.88, (0, 0, 150))
            flash = int(now_t * 2) % 2 == 0
            brd   = C_RED if flash else (100, 30, 30)
            cv2.rectangle(disp, (0, ay1), (W, ay2), brd, 3, cv2.LINE_AA)
            line1 = f"[!] THEFT: Person {a['suspect_id']} took {a['category'].upper()}  |  {a['timestamp']}"
            line2 = f"Evidence saved to /{EVIDENCE_DIR}/    |    Press R to clear alert    |    Alerts: {len(data['alerts'])}"
            _put(disp, line1, (12, ay1+30), 0.58, C_WHITE, 2)
            _put(disp, line2, (12, ay1+58), 0.4,  (210,210,210), 1)

        # Write evidence frame
        _write_ev(disp)

        cv2.imshow("Theft Detection | Surveillance", disp)
        lr_fid = _fid; lr_pid = _pid_inf

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k in (ord('r'), ord('R')):
            with _rl: _R["alerts"] = []; _reset_flag.set()

    cv2.destroyAllWindows()
    if _rec_on and _recorder: _recorder.release()
