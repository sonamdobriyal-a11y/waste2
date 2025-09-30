import base64
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from threading import Lock

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# Ensure project root is importable (so we can import src.*)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision import detect_utensil_ellipse, segment_food_in_utensil, estimate_area_and_volume


app = Flask(__name__, static_folder='static', template_folder='templates')

DB_PATH = os.path.join(PROJECT_ROOT, 'utensil_sessions.db')
_DB_LOCK = Lock()
_DB_INIT = False


def _init_db():
    global _DB_INIT
    with _DB_LOCK:
        if _DB_INIT:
            return
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS utensil_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    utensil_type TEXT NOT NULL,
                    average_volume_ml REAL,
                    average_percent_fill REAL,
                    sample_count INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
            _run_schema_migrations(conn)
        _DB_INIT = True


def _write_session_to_db(session):
    payload = (
        session['utensil_type'],
        session['average_volume_ml'],
        session['average_percent_fill'],
        session['sample_count'],
        session['started_at'].isoformat(timespec='seconds') + 'Z',
        session['ended_at'].isoformat(timespec='seconds') + 'Z',
        datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    )
    with _DB_LOCK:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO utensil_sessions (
                    utensil_type,
                    average_volume_ml,
                    average_percent_fill,
                    sample_count,
                    started_at,
                    ended_at,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()


IST_TZ = timezone(timedelta(hours=5, minutes=30))
UTC_TZ = timezone.utc


def _to_datetime_utc(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except ValueError:
            try:
                dt = datetime.strptime(str(value), '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_TZ)
    else:
        dt = dt.astimezone(UTC_TZ)
    return dt


def _format_ts_for_display(value):
    dt = _to_datetime_utc(value)
    if dt is None:
        return None
    return dt.astimezone(IST_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')


def _run_schema_migrations(conn):
    info = conn.execute("PRAGMA table_info(utensil_sessions)").fetchall()
    if not info:
        return
    columns = {row[1]: row for row in info}
    avg_volume_col = columns.get('average_volume_ml')
    if avg_volume_col and avg_volume_col[3]:
        try:
            conn.executescript(
                """
                BEGIN;
                ALTER TABLE utensil_sessions RENAME TO utensil_sessions__old;
                CREATE TABLE utensil_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    utensil_type TEXT NOT NULL,
                    average_volume_ml REAL,
                    average_percent_fill REAL,
                    sample_count INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                INSERT INTO utensil_sessions (
                    id,
                    utensil_type,
                    average_volume_ml,
                    average_percent_fill,
                    sample_count,
                    started_at,
                    ended_at,
                    created_at
                )
                SELECT
                    id,
                    utensil_type,
                    average_volume_ml,
                    average_percent_fill,
                    sample_count,
                    started_at,
                    ended_at,
                    created_at
                FROM utensil_sessions__old;
                DROP TABLE utensil_sessions__old;
                COMMIT;
                """
            )
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise


def _fetch_recent_sessions(limit=200):
    with _DB_LOCK:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    id,
                    utensil_type,
                    average_volume_ml,
                    average_percent_fill,
                    sample_count,
                    started_at,
                    ended_at,
                    created_at
                FROM utensil_sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    sessions = []
    for row in rows:
        avg_volume = float(row['average_volume_ml']) if row['average_volume_ml'] is not None else None
        avg_percent = float(row['average_percent_fill']) if row['average_percent_fill'] is not None else None
        sessions.append({
            'id': row['id'],
            'utensil_type': row['utensil_type'],
            'average_volume_ml': avg_volume,
            'average_percent_fill': avg_percent,
            'sample_count': row['sample_count'],
            'started_at': _format_ts_for_display(row['started_at']),
            'ended_at': _format_ts_for_display(row['ended_at']),
            'created_at': _format_ts_for_display(row['created_at']),
        })
    return sessions


def _dashboard_metrics(sessions):
    total_sessions = len(sessions)
    volume_values = [s['average_volume_ml'] for s in sessions if s['average_volume_ml'] is not None]
    percent_values = [s['average_percent_fill'] for s in sessions if s['average_percent_fill'] is not None]
    total_volume = sum(volume_values) if volume_values else 0.0
    mean_volume = (total_volume / len(volume_values)) if volume_values else None
    mean_percent = (sum(percent_values) / len(percent_values)) if percent_values else None
    return {
        'total_sessions': total_sessions,
        'total_volume_ml': total_volume,
        'mean_volume_ml': mean_volume,
        'mean_percent_fill': mean_percent,
    }


class UtensilSessionTracker:
    """Aggregates per-utensil volume readings until the utensil leaves the frame."""

    def __init__(self, tolerance_frames=2):
        self.tolerance_frames = tolerance_frames
        self.lock = Lock()
        self._clear_state()

    def _clear_state(self):
        self.active = False
        self.label = None
        self.volumes = []
        self.percents = []
        self.started_at = None
        self.last_seen_at = None
        self.missed_frames = 0
        self.frame_count = 0

    def update(self, utensil_present, utensil_label, volume_ml, percent_fill):
        now = datetime.utcnow()
        with self.lock:
            recorded_session = None

            if utensil_present:
                if self.active and self.label and utensil_label and utensil_label != self.label:
                    recorded_session = self._finalize_locked(now)

                if not self.active:
                    self._start_locked(utensil_label, now)

                self.missed_frames = 0
                self.last_seen_at = now
                self.frame_count += 1

                if volume_ml is not None:
                    try:
                        self.volumes.append(float(volume_ml))
                    except (TypeError, ValueError):
                        pass
                if percent_fill is not None:
                    try:
                        self.percents.append(float(percent_fill))
                    except (TypeError, ValueError):
                        pass
            else:
                if self.active:
                    self.missed_frames += 1
                    if self.missed_frames >= self.tolerance_frames:
                        recorded_session = self._finalize_locked(now)

            return recorded_session

    def _start_locked(self, label, started_at):
        self.active = True
        self.label = label or 'unknown'
        self.started_at = started_at
        self.last_seen_at = started_at
        self.volumes = []
        self.percents = []
        self.missed_frames = 0
        self.frame_count = 0

    def _finalize_locked(self, ended_at):
        if not self.active:
            return None

        volume_values = [v for v in self.volumes if v is not None]
        percent_values = [p for p in self.percents if p is not None]

        if not volume_values and not percent_values:
            self._clear_state()
            return None

        avg_volume = (sum(volume_values) / len(volume_values)) if volume_values else None
        avg_percent = (sum(percent_values) / len(percent_values)) if percent_values else None
        sample_count = max(self.frame_count, len(volume_values), len(percent_values))

        session = {
            'utensil_type': self.label or 'unknown',
            'average_volume_ml': avg_volume,
            'average_percent_fill': avg_percent,
            'sample_count': sample_count,
            'started_at': self.started_at or ended_at,
            'ended_at': self.last_seen_at or ended_at,
        }
        _write_session_to_db(session)
        self._clear_state()
        return session


_init_db()
SESSION_TRACKER = UtensilSessionTracker()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    sessions = _fetch_recent_sessions(limit=200)
    metrics = _dashboard_metrics(sessions)
    return render_template('dashboard.html', sessions=sessions, metrics=metrics)


def _decode_image_from_base64(data_url: str):
    # Accepts either data URL (data:image/jpeg;base64,...) or raw base64
    if ',' in data_url and data_url.strip().startswith('data:'):
        b64 = data_url.split(',', 1)[1]
    else:
        b64 = data_url
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def _encode_image_to_data_url(img_bgr):
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"


@app.route('/process', methods=['POST'])
def process():
    payload = request.get_json(silent=True) or {}
    img_data = payload.get('image')
    utensil = payload.get('utensil', 'auto')
    diameter_mm = payload.get('diameter_mm')
    height_mm = payload.get('assumed_height_mm', 15.0)

    if img_data is None:
        return jsonify({'error': 'missing image'}), 400

    frame = _decode_image_from_base64(img_data)
    if frame is None:
        return jsonify({'error': 'bad image'}), 400

    display = frame.copy()

    # Sensible Hough bounds based on frame size
    h, w = frame.shape[:2]
    min_radius = max(30, min(h, w) // 8)
    max_radius = max(60, min(h, w) // 2)

    ellipse = detect_utensil_ellipse(frame, utensil_hint=utensil,
                                     min_radius=min_radius, max_radius=max_radius, debug=False)

    percent_fill = None
    est_volume_ml = None

    if ellipse is not None:
        # Draw ellipse
        (cx, cy), (MA, ma), angle = ellipse
        cv2.ellipse(display, (int(cx), int(cy)), (int(MA/2), int(ma/2)), angle, 0, 360, (0,255,255), 2)

        seg_mask, _ = segment_food_in_utensil(frame, ellipse, debug=False)
        if seg_mask is not None:
            # Overlay segmentation
            overlay = display.copy()
            overlay[seg_mask > 0] = (0, 0, 255)
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

            # Compute metrics
            d_mm = float(diameter_mm) if diameter_mm not in (None, '') else None
            percent_fill, est_volume_ml = estimate_area_and_volume(
                ellipse, seg_mask, d_mm, utensil, float(height_mm or 0)
            )

    session_record = SESSION_TRACKER.update(ellipse is not None, utensil, est_volume_ml, percent_fill)

    # HUD text
    y = 24
    txts = []
    txts.append(f"Utensil: {utensil}")
    if percent_fill is not None:
        txts.append(f"Fill: {percent_fill:.1f}%")
    if est_volume_ml is not None:
        txts.append(f"Vol: {est_volume_ml:.0f} ml")
    if ellipse is None:
        txts.append("Utensil not detected")
    for i, t in enumerate(txts):
        cv2.putText(display, t, (10, 30 + i*y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    data_url = _encode_image_to_data_url(display)
    response = {
        'percent_fill': percent_fill,
        'volume_ml': est_volume_ml,
        'overlay': data_url,
    }

    if session_record:
        response['session_recorded'] = True
        response['session_summary'] = {
            'utensil_type': session_record['utensil_type'],
            'average_volume_ml': session_record['average_volume_ml'],
            'average_percent_fill': session_record['average_percent_fill'],
            'sample_count': session_record['sample_count'],
            'started_at': session_record['started_at'].isoformat(timespec='seconds') + 'Z',
            'ended_at': session_record['ended_at'].isoformat(timespec='seconds') + 'Z',
            'started_at_ist': _format_ts_for_display(session_record['started_at']),
            'ended_at_ist': _format_ts_for_display(session_record['ended_at']),
        }

    return jsonify(response)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    app.run(host='0.0.0.0', port=port, debug=True)
