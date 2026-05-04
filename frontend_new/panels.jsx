// panels.jsx — Camera views, error graph, status bar, controls.

// ── Procedural "camera view" ────────────────────────────────────────────────
// Renders a fake first-person view from a pose by ray-marching the occupancy
// grid. Walls are textured with a hash. Different palettes for "real camera"
// vs "3D-GS rendered best particle view".
function CameraCanvas({ pose, label, sublabel, palette, accent }) {
  const ref = React.useRef(null);
  const wrapRef = React.useRef(null);
  const [size, setSize] = React.useState({ w: 320, h: 200 });

  React.useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: Math.max(60, r.width), h: Math.max(60, r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  React.useEffect(() => {
    let raf;
    const draw = () => {
      const c = ref.current;
      if (!c || !pose) { raf = requestAnimationFrame(draw); return; }
      const dpr = window.devicePixelRatio || 1;
      const W = size.w, H = size.h;
      if (c.width !== W * dpr || c.height !== H * dpr) {
        c.width = W * dpr; c.height = H * dpr;
        c.style.width = W + 'px'; c.style.height = H + 'px';
      }
      const ctx = c.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // sky/floor gradient
      const sky = ctx.createLinearGradient(0, 0, 0, H);
      sky.addColorStop(0, palette.skyTop);
      sky.addColorStop(0.5, palette.skyMid);
      sky.addColorStop(0.5, palette.floorTop);
      sky.addColorStop(1, palette.floorBot);
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, W, H);

      // raycast columns
      const FOV = Math.PI / 2.6;
      const cols = Math.max(60, Math.floor(W / 2));
      const colW = W / cols;
      const ox = pose.x, oy = pose.y;
      for (let i = 0; i < cols; i++) {
        const a = pose.theta + (i / (cols - 1) - 0.5) * FOV;
        const cosA = Math.cos(a), sinA = Math.sin(a);
        let dist = 0, hit = false, hitV = 0;
        const maxD = 220;
        const stepLen = 1.0;
        for (let d = 0; d < maxD; d += stepLen) {
          const x = ox + cosA * d, y = oy + sinA * d;
          const ix = x | 0, iy = y | 0;
          if (ix < 0 || iy < 0 || ix >= MAP_W || iy >= MAP_H) { dist = d; hit = true; hitV = 1; break; }
          const v = OCC.grid[iy * MAP_W + ix];
          if (v === 1) { dist = d; hit = true; hitV = 1; break; }
          if (v === 2) { dist = d; hit = true; hitV = 2; break; }
        }
        if (!hit) continue;
        // perspective correction
        const corr = dist * Math.cos(a - pose.theta);
        const wallH = Math.min(H, (H * 24) / Math.max(1, corr));
        const yTop = (H - wallH) / 2;
        const shade = Math.max(0.18, 1 - corr / 120);
        // hash for vertical stripe variation
        const hx = ((ox + cosA * dist) * 17 + (oy + sinA * dist) * 31) | 0;
        const stripe = ((hx % 7) + 7) % 7 / 7;
        const baseR = palette.wall[0], baseG = palette.wall[1], baseB = palette.wall[2];
        const r = baseR * shade * (0.85 + stripe * 0.3);
        const g = baseG * shade * (0.85 + stripe * 0.3);
        const b = baseB * shade * (0.85 + stripe * 0.3);
        ctx.fillStyle = hitV === 2
          ? `rgba(${baseR*0.4|0},${baseG*0.4|0},${baseB*0.4|0},${shade})`
          : `rgb(${r|0},${g|0},${b|0})`;
        ctx.fillRect(i * colW, yTop, colW + 0.6, wallH);
      }

      // crosshair
      ctx.strokeStyle = palette.crosshair;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(W / 2 - 6, H / 2); ctx.lineTo(W / 2 + 6, H / 2);
      ctx.moveTo(W / 2, H / 2 - 6); ctx.lineTo(W / 2, H / 2 + 6);
      ctx.stroke();

      // scanlines for "real camera" feel
      if (palette.scanlines) {
        ctx.fillStyle = 'rgba(0,0,0,0.08)';
        for (let y = 0; y < H; y += 2) ctx.fillRect(0, y, W, 1);
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [pose, size, palette]);

  return (
    <div className="cam-card">
      <div className="cam-hd">
        <span className="cam-dot" style={{ background: accent }} />
        <span className="cam-lbl">{label}</span>
        <span className="cam-sub">{sublabel}</span>
      </div>
      <div className="cam-body" ref={wrapRef}>
        <canvas ref={ref} />
        <div className="cam-corner tl">REC ●</div>
        <div className="cam-corner tr">{pose ? `θ=${(pose.theta * 180 / Math.PI).toFixed(0)}°` : ''}</div>
        <div className="cam-corner bl">{pose ? `(${pose.x.toFixed(1)},${pose.y.toFixed(1)})` : ''}</div>
        <div className="cam-corner br">640×400 · 30fps</div>
      </div>
    </div>
  );
}

// ── Error graph ─────────────────────────────────────────────────────────────
// mode: 'magnitude' (single euclidean line) or 'components' (x, y, θ lines)
function ErrorGraph({ history, theme, mode = 'magnitude' }) {
  const ref = React.useRef(null);
  const wrapRef = React.useRef(null);
  const [size, setSize] = React.useState({ w: 320, h: 140 });

  React.useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: Math.max(60, r.width), h: Math.max(60, r.height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  React.useEffect(() => {
    let raf;
    const draw = () => {
      const c = ref.current;
      if (!c) { raf = requestAnimationFrame(draw); return; }
      const dpr = window.devicePixelRatio || 1;
      const W = size.w, H = size.h;
      if (c.width !== W * dpr || c.height !== H * dpr) {
        c.width = W * dpr; c.height = H * dpr;
        c.style.width = W + 'px'; c.style.height = H + 'px';
      }
      const ctx = c.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const PAD_L = 32, PAD_R = 8, PAD_T = 12, PAD_B = 18;
      const gw = W - PAD_L - PAD_R, gh = H - PAD_T - PAD_B;

      const hist = history || [];
      const tEnd = hist.length ? hist[hist.length - 1].t : 0;
      // Fixed 30s window. The "now" line sits at ~82% of the plot — the strip
      // to its right is left empty so the current value isn't pinned to the
      // edge and is easier to read.
      const WINDOW = 30;
      const RIGHT_PAD_FRAC = 0.18;
      const plotW = gw * (1 - RIGHT_PAD_FRAC);
      const tStart = tEnd - WINDOW;
      const xOf = (t) => PAD_L + ((t - tStart) / WINDOW) * plotW;
      const nowX = xOf(tEnd);

      // grid
      ctx.strokeStyle = theme === 'dark' ? 'rgba(120,140,170,0.12)' : 'rgba(40,50,70,0.15)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = PAD_T + (gh * i) / 4;
        ctx.beginPath();
        ctx.moveTo(PAD_L, y); ctx.lineTo(W - PAD_R, y); ctx.stroke();
      }
      for (let i = 0; i <= 6; i++) {
        const x = PAD_L + (plotW * i) / 6;
        ctx.beginPath();
        ctx.moveTo(PAD_L + x - PAD_L, PAD_T);
        ctx.moveTo(PAD_L + (plotW * i) / 6, PAD_T);
        ctx.lineTo(PAD_L + (plotW * i) / 6, PAD_T + gh);
        ctx.stroke();
      }
      // "now" marker: vertical line at tEnd
      ctx.strokeStyle = theme === 'dark' ? 'rgba(220,225,235,0.35)' : 'rgba(40,50,70,0.5)';
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(nowX, PAD_T);
      ctx.lineTo(nowX, PAD_T + gh);
      ctx.stroke();
      ctx.setLineDash([]);

      // y range
      let yMin = 0, yMax = 0;
      if (mode === 'components') {
        for (const p of hist) {
          if (p.t < tStart) continue;
          const exPx = p.ex || 0, eyPx = p.ey || 0;
          const ethDeg = (p.eth || 0) * 180 / Math.PI;
          const m = Math.max(Math.abs(exPx), Math.abs(eyPx), Math.abs(ethDeg));
          if (m > yMax) yMax = m;
        }
        yMax = Math.max(10, yMax * 1.15);
        yMin = -yMax;
      } else {
        for (const p of hist) if (p.t >= tStart && p.err > yMax) yMax = p.err;
        yMax = Math.max(10, yMax * 1.15);
        yMin = 0;
      }
      const yOf = (v) => PAD_T + gh - ((v - yMin) / (yMax - yMin)) * gh;

      // axis labels
      ctx.fillStyle = theme === 'dark' ? 'rgba(180,200,220,0.55)' : 'rgba(40,50,70,0.65)';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      for (let i = 0; i <= 4; i++) {
        const v = yMax + (yMin - yMax) * (i / 4);
        const y = PAD_T + (gh * i) / 4;
        const lbl = mode === 'components'
          ? (Math.abs(v) < 0.001 ? '0' : v.toFixed(0))
          : v.toFixed(0) + 'px';
        ctx.fillText(lbl, PAD_L - 4, y + 3);
      }
      // zero line for components mode
      if (mode === 'components') {
        ctx.strokeStyle = theme === 'dark' ? 'rgba(180,200,220,0.25)' : 'rgba(40,50,70,0.3)';
        ctx.lineWidth = 1;
        const zy = yOf(0);
        ctx.beginPath(); ctx.moveTo(PAD_L, zy); ctx.lineTo(W - PAD_R, zy); ctx.stroke();
      }
      ctx.textAlign = 'center';
      for (let i = 0; i <= 6; i++) {
        const t = tStart + WINDOW * (i / 6);
        const x = PAD_L + (plotW * i) / 6;
        const rel = t - tEnd;
        ctx.fillText(rel.toFixed(0) + 's', x, H - 4);
      }
      // "now" tick label — small, near the bottom of the plot, away from the legend
      ctx.textAlign = 'left';
      ctx.fillStyle = theme === 'dark' ? 'rgba(220,225,235,0.7)' : 'rgba(40,50,70,0.85)';
      ctx.fillText('now', nowX + 4, PAD_T + gh - 4);

      // clip to plot region so a not-yet-full window doesn't draw outside
      ctx.save();
      ctx.beginPath();
      ctx.rect(PAD_L, PAD_T, gw, gh);
      ctx.clip();

      const drawSeries = (getter, color, fill = false) => {
        if (hist.length < 2) return;
        if (fill) {
          ctx.beginPath();
          let started = false;
          const baseY = mode === 'components' ? yOf(0) : PAD_T + gh;
          for (const p of hist) {
            if (p.t < tStart) continue;
            const x = xOf(p.t), y = yOf(getter(p));
            if (!started) { ctx.moveTo(x, baseY); ctx.lineTo(x, y); started = true; }
            else ctx.lineTo(x, y);
          }
          if (started) {
            const lastP = hist[hist.length - 1];
            ctx.lineTo(xOf(lastP.t), baseY);
            ctx.closePath();
            const grad = ctx.createLinearGradient(0, PAD_T, 0, PAD_T + gh);
            grad.addColorStop(0, color.replace(/[\d.]+\)$/, '0.35)'));
            grad.addColorStop(1, color.replace(/[\d.]+\)$/, '0.02)'));
            ctx.fillStyle = grad;
            ctx.fill();
          }
        }
        ctx.beginPath();
        let started = false;
        for (const p of hist) {
          if (p.t < tStart) continue;
          const x = xOf(p.t), y = yOf(getter(p));
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        ctx.stroke();
        // last value pulse
        const last = hist[hist.length - 1];
        const lx = xOf(last.t), ly = yOf(getter(last));
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2); ctx.fill();
      };

      if (mode === 'components') {
        drawSeries((p) => p.ex || 0, 'rgba(120,220,255,1)');
        drawSeries((p) => p.ey || 0, 'rgba(255,180,60,1)');
        drawSeries((p) => ((p.eth || 0) * 180 / Math.PI), 'rgba(255,120,180,1)');
      } else {
        drawSeries((p) => p.err, 'rgba(255,180,60,1)', true);
      }
      ctx.restore();

      // legend (top-left inside plot, away from the now-line + right padding)
      const legend = mode === 'components'
        ? [
            { c: 'rgba(120,220,255,1)', l: 'Δx (px)' },
            { c: 'rgba(255,180,60,1)',  l: 'Δy (px)' },
            { c: 'rgba(255,120,180,1)', l: 'Δθ (°)' },
          ]
        : [{ c: 'rgba(255,180,60,1)', l: '‖Δp‖ (px)' }];
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      let lx0 = PAD_L + 6;
      for (let i = 0; i < legend.length; i++) {
        const item = legend[i];
        ctx.fillStyle = item.c;
        ctx.fillRect(lx0, PAD_T + 5, 8, 2);
        ctx.fillStyle = theme === 'dark' ? 'rgba(220,225,235,0.8)' : 'rgba(40,50,70,0.85)';
        ctx.fillText(item.l, lx0 + 12, PAD_T + 9);
        lx0 += 12 + ctx.measureText(item.l).width + 14;
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [history, size, theme, mode]);

  return <div className="graph-canvas-wrap" ref={wrapRef}><canvas ref={ref} /></div>;
}

Object.assign(window, { CameraCanvas, ErrorGraph });
