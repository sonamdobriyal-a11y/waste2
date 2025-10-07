import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from threading import Lock

import cv2
import firebase_admin
import numpy as np
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
from google.api_core import exceptions as google_exceptions
from queue import Empty, Queue
from flask import Flask, jsonify, render_template, request, Response, stream_with_context

# Ensure project root is importable (so we can import src.*)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from src.vision import detect_utensil_ellipse, segment_food_in_utensil, estimate_area_and_volume


app = Flask(__name__, static_folder='static', template_folder='templates')

FIREBASE_PROJECT_ID = os.environ.get('FIREBASE_PROJECT_ID')
FIRESTORE_COLLECTION = 'utensil_sessions'
_FIRESTORE_CLIENT = None
_FIRESTORE_LOCK = Lock()


class FirestoreNotConfigured(RuntimeError):
    """Raised when Firestore is unreachable or not initialised."""
    pass



def _build_firebase_credentials():
    service_account_base64 = os.environ.get('FIREBASE_SERVICE_ACCOUNT_BASE64')
    if service_account_base64:
        service_account_base64 = service_account_base64.strip()
    service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if service_account_json:
        service_account_json = service_account_json.strip()
    credentials_file = os.environ.get('FIREBASE_CREDENTIALS_FILE')
    if credentials_file:
        credentials_file = credentials_file.strip()

    if service_account_base64:
        try:
            decoded = base64.b64decode(service_account_base64).decode('utf-8')
        except Exception as exc:
            raise RuntimeError('FIREBASE_SERVICE_ACCOUNT_BASE64 is not valid base64 data') from exc
        service_account_json = decoded

    if service_account_json:
        try:
            info = json.loads(service_account_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError('FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON') from exc
        return credentials.Certificate(info)

    if credentials_file:
        credentials_path = credentials_file
        if not os.path.isabs(credentials_path):
            credentials_path = os.path.join(PROJECT_ROOT, credentials_path)
        if not os.path.exists(credentials_path):
            raise RuntimeError(f"FIREBASE_CREDENTIALS_FILE {credentials_path} does not exist")
        return credentials.Certificate(credentials_path)

    return None



def _init_firestore():
    global _FIRESTORE_CLIENT
    if _FIRESTORE_CLIENT is not None:
        return _FIRESTORE_CLIENT

    with _FIRESTORE_LOCK:
        if _FIRESTORE_CLIENT is not None:
            return _FIRESTORE_CLIENT

        if not firebase_admin._apps:
            cred = _build_firebase_credentials()
            options = {}
            if FIREBASE_PROJECT_ID:
                options['projectId'] = FIREBASE_PROJECT_ID
            firebase_admin.initialize_app(credential=cred, options=options or None)

        _FIRESTORE_CLIENT = firestore.client()
    return _FIRESTORE_CLIENT


class DashboardEventBroker:
    """Tracks dashboard listeners and pushes events when sessions persist."""

    def __init__(self):
        self._lock = Lock()
        self._subscribers = []

    def subscribe(self):
        queue = Queue()
        with self._lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue):
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def publish(self, event):
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(event)

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



def _write_session_to_db(session):
    client = _init_firestore()

    started_at = _to_datetime_utc(session.get('started_at')) or datetime.utcnow().replace(tzinfo=UTC_TZ)
    ended_at = _to_datetime_utc(session.get('ended_at')) or started_at
    created_at = datetime.utcnow().replace(tzinfo=UTC_TZ)

    avg_volume = session.get('average_volume_ml')
    if avg_volume is not None:
        try:
            avg_volume = float(avg_volume)
        except (TypeError, ValueError):
            avg_volume = None

    avg_percent = session.get('average_percent_fill')
    if avg_percent is not None:
        try:
            avg_percent = float(avg_percent)
        except (TypeError, ValueError):
            avg_percent = None

    sample_count = session.get('sample_count') or 0
    try:
        sample_count = int(sample_count)
    except (TypeError, ValueError):
        sample_count = 0

    payload = {
        'utensil_type': session.get('utensil_type', 'unknown'),
        'average_volume_ml': avg_volume,
        'average_percent_fill': avg_percent,
        'sample_count': sample_count,
        'started_at': started_at,
        'ended_at': ended_at,
        'created_at': created_at,
    }

    collection = client.collection(FIRESTORE_COLLECTION)
    doc_ref = collection.document()
    try:
        doc_ref.set(payload)
    except google_exceptions.NotFound as exc:
        raise FirestoreNotConfigured('Firestore database is not initialised for this project.') from exc
    except google_exceptions.GoogleAPICallError as exc:
        message = getattr(exc, 'message', str(exc))
        raise FirestoreNotConfigured(f'Unable to write to Firestore: {message}') from exc
    else:
        DASHBOARD_EVENTS.publish({
            'type': 'session_saved',
            'id': doc_ref.id,
            'created_at': created_at.isoformat(),
        })


def _fetch_recent_sessions(limit=200):
    client = _init_firestore()
    query = (
        client.collection(FIRESTORE_COLLECTION)
        .order_by('created_at', direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    sessions = []

    try:
        documents = list(query.stream())
    except google_exceptions.NotFound as exc:
        raise FirestoreNotConfigured('Firestore database is not initialised for this project.') from exc
    except google_exceptions.GoogleAPICallError as exc:
        message = getattr(exc, 'message', str(exc))
        raise FirestoreNotConfigured(f'Unable to read from Firestore: {message}') from exc

    for doc in documents:
        data = doc.to_dict() or {}
        avg_volume = data.get('average_volume_ml')
        if avg_volume is not None:
            try:
                avg_volume = float(avg_volume)
            except (TypeError, ValueError):
                avg_volume = None
        avg_percent = data.get('average_percent_fill')
        if avg_percent is not None:
            try:
                avg_percent = float(avg_percent)
            except (TypeError, ValueError):
                avg_percent = None
        sample_count = data.get('sample_count') or 0
        try:
            sample_count = int(sample_count)
        except (TypeError, ValueError):
            sample_count = 0

        sessions.append({
            'id': doc.id,
            'utensil_type': data.get('utensil_type', 'unknown'),
            'average_volume_ml': avg_volume,
            'average_percent_fill': avg_percent,
            'sample_count': sample_count,
            'started_at': _format_ts_for_display(data.get('started_at')),
            'ended_at': _format_ts_for_display(data.get('ended_at')),
            'created_at': _format_ts_for_display(data.get('created_at')),
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

        try:
            _write_session_to_db(session)
        except FirestoreNotConfigured as exc:
            app.logger.error('Failed to persist session to Firestore: %s', exc)

        self._clear_state()
        return session


DASHBOARD_EVENTS = DashboardEventBroker()
SESSION_TRACKER = UtensilSessionTracker()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard/events')
def dashboard_events():
    queue = DASHBOARD_EVENTS.subscribe()

    def generate():
        try:
            while True:
                try:
                    event = queue.get(timeout=30)
                except Empty:
                    yield ': keep-alive\n\n'
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            DASHBOARD_EVENTS.unsubscribe(queue)

    response = Response(stream_with_context(generate()), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


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
        cv2.ellipse(display, (int(cx), int(cy)), (int(MA/2), int(ma/2)), angle, 0, 360, (0, 255, 255), 2)

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
        cv2.putText(display, t, (10, 30 + i * y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

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
