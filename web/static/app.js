(() => {
  const sel = (q) => document.querySelector(q);

  const video = sel('#video');
  const cap = sel('#cap');
  const ctx = cap.getContext('2d');
  const overlay = sel('#overlay');
  const stats = sel('#stats');
  const utensilEl = sel('#utensil');
  const diameterEl = sel('#diameter');
  const heightEl = sel('#height');
  const fpsEl = sel('#fps');
  const startBtn = sel('#start');
  const stopBtn = sel('#stop');

  const processUrl = (window.APP_CONFIG && window.APP_CONFIG.processUrl) || '/process';

  let running = false;
  let stream = null;
  let lastTick = 0;

  async function start() {
    if (running) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      video.srcObject = stream;
      await video.play();
      running = true;
      tick();
    } catch (e) {
      console.error(e);
      alert('Camera access error: ' + e.message);
    }
  }

  function stop() {
    running = false;
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
  }

  async function tick(ts) {
    if (!running) return;
    const maxFps = Math.max(1, Math.min(10, parseInt(fpsEl.value || '3', 10)));
    const interval = 1000 / maxFps;
    if (!lastTick || (performance.now() - lastTick) >= interval) {
      lastTick = performance.now();
      await captureAndSend();
    }
    requestAnimationFrame(tick);
  }

  async function captureAndSend() {
    const W = 640, H = 480;
    cap.width = W; cap.height = H;
    ctx.drawImage(video, 0, 0, W, H);
    const blob = await new Promise(res => cap.toBlob(res, 'image/jpeg', 0.8));
    if (!blob) return;
    const b64 = await blobToDataURL(blob);

    const payload = {
      image: b64,
      utensil: utensilEl.value,
      diameter_mm: diameterEl.value ? Number(diameterEl.value) : null,
      assumed_height_mm: heightEl.value ? Number(heightEl.value) : 15,
    };

    try {
      const resp = await fetch(processUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (data.overlay) overlay.src = data.overlay;
      const pf = data.percent_fill != null ? data.percent_fill.toFixed(1) + '%' : '—';
      const vol = data.volume_ml != null ? data.volume_ml.toFixed(0) + ' ml' : '—';
      stats.innerText = `Fill: ${pf}   Volume: ${vol}`;
    } catch (e) {
      console.error('process error', e);
    }
  }

  function blobToDataURL(blob) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = reject;
      r.readAsDataURL(blob);
    });
  }

  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);
})();

