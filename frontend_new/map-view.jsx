// map-view.jsx — Live occupancy-grid map with pan/zoom, particles,
// ground-truth + estimate markers, covariance ellipse, and click-drag prior.

// Render the occupancy grid to an offscreen canvas once (cheap) — we reuse it.
function makeMapBitmap(theme = 'dark') {
  const c = document.createElement('canvas');
  c.width = MAP_W; c.height = MAP_H;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(MAP_W, MAP_H);
  const palettes = {
    dark: { free: [30, 32, 38], occ: [200, 205, 215], unk: [16, 17, 21] },
    light: { free: [248, 247, 244], occ: [40, 44, 52], unk: [220, 218, 212] },
  };
  const pal = palettes[theme] || palettes.dark;
  for (let i = 0; i < OCC.grid.length; i++) {
    const v = OCC.grid[i];
    const col = v === 1 ? pal.occ : v === 2 ? pal.unk : pal.free;
    img.data[i * 4 + 0] = col[0];
    img.data[i * 4 + 1] = col[1];
    img.data[i * 4 + 2] = col[2];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// Particle weight → color. Cool (low) → hot (high).
function weightColor(w, wMax) {
  const t = Math.max(0, Math.min(1, w / wMax));
  // interpolate dark cyan → cyan → amber → hot pink
  const stops = [
    [0.00, [40, 90, 130]],
    [0.35, [80, 200, 230]],
    [0.70, [255, 200, 90]],
    [1.00, [255, 90, 140]],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i];
      const u = (t - t0) / (t1 - t0);
      return [
        c0[0] + (c1[0] - c0[0]) * u,
        c0[1] + (c1[1] - c0[1]) * u,
        c0[2] + (c1[2] - c0[2]) * u,
      ];
    }
  }
  return stops[stops.length - 1][1];
}

function MapView({
  filter, paused, theme, particleStyle,
  layers, // {grid, particles, robots, heatmap, covariance, trail}
  onSetPrior,
  estPath, truthPath,
}) {
  const wrapRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const overlayRef = React.useRef(null);
  const [size, setSize] = React.useState({ w: 800, h: 600 });
  // view: pixels per map unit, plus pan offset (in map units, of top-left corner)
  const [view, setView] = React.useState({ scale: 2.4, cx: MAP_W / 2, cy: MAP_H / 2 });
  const viewRef = React.useRef(view);
  viewRef.current = view;

  const bitmap = React.useMemo(() => makeMapBitmap(theme), [theme]);

  // Drag-to-set-prior interaction state
  const [priorDrag, setPriorDrag] = React.useState(null); // {x0,y0,x1,y1}
  const [hover, setHover] = React.useState(null);

  // Resize observer
  React.useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: Math.max(100, r.width), h: Math.max(100, r.height) });
    });
    ro.observe(el);
    const r = el.getBoundingClientRect();
    setSize({ w: Math.max(100, r.width), h: Math.max(100, r.height) });
    return () => ro.disconnect();
  }, []);

  // Map ↔ screen helpers
  const mapToScreen = React.useCallback((mx, my) => {
    const v = viewRef.current;
    return {
      x: (mx - v.cx) * v.scale + size.w / 2,
      y: (my - v.cy) * v.scale + size.h / 2,
    };
  }, [size]);
  const screenToMap = React.useCallback((sx, sy) => {
    const v = viewRef.current;
    return {
      x: (sx - size.w / 2) / v.scale + v.cx,
      y: (sy - size.h / 2) / v.scale + v.cy,
    };
  }, [size]);

  // Wheel zoom + pan
  React.useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const sx = e.clientX - r.left, sy = e.clientY - r.top;
      const before = screenToMap(sx, sy);
      const factor = Math.exp(-e.deltaY * 0.0015);
      setView((v) => {
        const ns = Math.max(0.6, Math.min(8, v.scale * factor));
        // Keep the cursor's map point fixed
        const dx = (sx - size.w / 2) / ns;
        const dy = (sy - size.h / 2) / ns;
        return { scale: ns, cx: before.x - dx, cy: before.y - dy };
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [screenToMap, size]);

  // Mouse interactions: middle/right drag to pan; left drag to set prior
  const dragStateRef = React.useRef(null);
  const onMouseDown = (e) => {
    const el = canvasRef.current;
    const r = el.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    if (e.button === 0 && !e.shiftKey) {
      // start prior drag
      const m = screenToMap(sx, sy);
      dragStateRef.current = { kind: 'prior', start: m };
      setPriorDrag({ x0: m.x, y0: m.y, x1: m.x, y1: m.y });
    } else {
      // pan
      dragStateRef.current = { kind: 'pan', sx, sy, cx: viewRef.current.cx, cy: viewRef.current.cy };
    }
  };
  const onMouseMove = (e) => {
    const el = canvasRef.current;
    const r = el.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    setHover(screenToMap(sx, sy));
    const ds = dragStateRef.current;
    if (!ds) return;
    if (ds.kind === 'pan') {
      const v = viewRef.current;
      setView({ scale: v.scale, cx: ds.cx - (sx - ds.sx) / v.scale, cy: ds.cy - (sy - ds.sy) / v.scale });
    } else if (ds.kind === 'prior') {
      const m = screenToMap(sx, sy);
      setPriorDrag({ x0: ds.start.x, y0: ds.start.y, x1: m.x, y1: m.y });
    }
  };
  const onMouseUp = () => {
    const ds = dragStateRef.current;
    dragStateRef.current = null;
    if (ds && ds.kind === 'prior' && priorDrag) {
      const dx = priorDrag.x1 - priorDrag.x0;
      const dy = priorDrag.y1 - priorDrag.y0;
      const mag = Math.hypot(dx, dy);
      // require a small drag distance to set orientation, else default to 0
      const theta = mag > 4 ? Math.atan2(dy, dx) : 0;
      onSetPrior?.({ x: priorDrag.x0, y: priorDrag.y0, theta });
      setPriorDrag(null);
    }
  };
  const onMouseLeave = () => {
    setHover(null);
    if (dragStateRef.current?.kind === 'pan') dragStateRef.current = null;
  };

  // Main render loop — draws everything onto a single canvas every frame
  React.useEffect(() => {
    let raf;
    const draw = () => {
      const c = canvasRef.current;
      if (!c) { raf = requestAnimationFrame(draw); return; }
      const ctx = c.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      if (c.width !== size.w * dpr || c.height !== size.h * dpr) {
        c.width = size.w * dpr; c.height = size.h * dpr;
        c.style.width = size.w + 'px'; c.style.height = size.h + 'px';
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.w, size.h);

      // background grid (subtle)
      const bg = theme === 'dark' ? '#0c0d10' : '#efece6';
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, size.w, size.h);

      const v = viewRef.current;
      const tlx = v.cx - size.w / (2 * v.scale);
      const tly = v.cy - size.h / (2 * v.scale);

      // grid dots
      ctx.fillStyle = theme === 'dark' ? 'rgba(120,140,170,0.07)' : 'rgba(40,50,70,0.07)';
      const step = 20;
      const startX = Math.floor(tlx / step) * step;
      const startY = Math.floor(tly / step) * step;
      for (let mx = startX; mx < tlx + size.w / v.scale + step; mx += step) {
        for (let my = startY; my < tly + size.h / v.scale + step; my += step) {
          const s = mapToScreen(mx, my);
          ctx.fillRect(s.x - 1, s.y - 1, 2, 2);
        }
      }

      // occupancy grid bitmap (the "PGM background")
      if (layers.grid) {
        const tl = mapToScreen(0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.globalAlpha = 0.95;
        ctx.drawImage(bitmap, tl.x, tl.y, MAP_W * v.scale, MAP_H * v.scale);
        ctx.globalAlpha = 1;
      }

      // map outline
      const o0 = mapToScreen(0, 0), o1 = mapToScreen(MAP_W, MAP_H);
      ctx.strokeStyle = theme === 'dark' ? 'rgba(120,160,200,0.25)' : 'rgba(40,60,90,0.25)';
      ctx.lineWidth = 1;
      ctx.strokeRect(o0.x, o0.y, o1.x - o0.x, o1.y - o0.y);

      // Truth path trail
      if (layers.trail && truthPath?.length > 1) {
        ctx.strokeStyle = theme === 'dark' ? 'rgba(80,220,160,0.35)' : 'rgba(20,140,90,0.5)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        for (let i = 0; i < truthPath.length; i++) {
          const s = mapToScreen(truthPath[i].x, truthPath[i].y);
          if (i === 0) ctx.moveTo(s.x, s.y); else ctx.lineTo(s.x, s.y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (layers.trail && estPath?.length > 1) {
        ctx.strokeStyle = theme === 'dark' ? 'rgba(255,180,60,0.55)' : 'rgba(180,100,0,0.7)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < estPath.length; i++) {
          const s = mapToScreen(estPath[i].x, estPath[i].y);
          if (i === 0) ctx.moveTo(s.x, s.y); else ctx.lineTo(s.x, s.y);
        }
        ctx.stroke();
      }

      // particles
      const N = filter.N;
      // find max weight for normalization
      let wMax = 0;
      for (let i = 0; i < N; i++) if (filter.pw[i] > wMax) wMax = filter.pw[i];

      if (layers.particles && layers.heatmap) {
        // density heatmap on a low-res grid
        const cellPx = 8;
        const cellsX = Math.ceil(size.w / cellPx);
        const cellsY = Math.ceil(size.h / cellPx);
        const acc = new Float32Array(cellsX * cellsY);
        for (let i = 0; i < N; i++) {
          const s = mapToScreen(filter.px[i], filter.py[i]);
          const cx = (s.x / cellPx) | 0;
          const cy = (s.y / cellPx) | 0;
          if (cx < 0 || cy < 0 || cx >= cellsX || cy >= cellsY) continue;
          acc[cy * cellsX + cx] += filter.pw[i];
        }
        let maxA = 0;
        for (let i = 0; i < acc.length; i++) if (acc[i] > maxA) maxA = acc[i];
        if (maxA > 0) {
          for (let cy = 0; cy < cellsY; cy++) {
            for (let cx = 0; cx < cellsX; cx++) {
              const a = acc[cy * cellsX + cx] / maxA;
              if (a < 0.02) continue;
              const col = weightColor(a, 1);
              ctx.fillStyle = `rgba(${col[0]|0},${col[1]|0},${col[2]|0},${0.18 + a * 0.6})`;
              ctx.fillRect(cx * cellPx, cy * cellPx, cellPx + 0.5, cellPx + 0.5);
            }
          }
        }
      } else if (layers.particles) {
        for (let i = 0; i < N; i++) {
          const s = mapToScreen(filter.px[i], filter.py[i]);
          if (s.x < -10 || s.y < -10 || s.x > size.w + 10 || s.y > size.h + 10) continue;
          const col = weightColor(filter.pw[i], wMax);
          const alpha = 0.4 + 0.6 * Math.min(1, filter.pw[i] / wMax);
          ctx.fillStyle = `rgba(${col[0]|0},${col[1]|0},${col[2]|0},${alpha})`;
          if (particleStyle === 'arrow') {
            const len = Math.max(3.5, Math.min(7, v.scale * 1.1));
            ctx.strokeStyle = ctx.fillStyle;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(s.x, s.y);
            ctx.lineTo(s.x + Math.cos(filter.pt[i]) * len, s.y + Math.sin(filter.pt[i]) * len);
            ctx.stroke();
          } else if (particleStyle === 'triangle') {
            const len = Math.max(3, Math.min(6, v.scale));
            const a = filter.pt[i];
            ctx.beginPath();
            ctx.moveTo(s.x + Math.cos(a) * len, s.y + Math.sin(a) * len);
            ctx.lineTo(s.x + Math.cos(a + 2.5) * len * 0.6, s.y + Math.sin(a + 2.5) * len * 0.6);
            ctx.lineTo(s.x + Math.cos(a - 2.5) * len * 0.6, s.y + Math.sin(a - 2.5) * len * 0.6);
            ctx.closePath();
            ctx.fill();
          } else {
            // dot + heading line
            ctx.beginPath();
            ctx.arc(s.x, s.y, 1.8, 0, Math.PI * 2);
            ctx.fill();
            const len = Math.max(3, Math.min(6, v.scale));
            ctx.strokeStyle = ctx.fillStyle;
            ctx.lineWidth = 0.9;
            ctx.beginPath();
            ctx.moveTo(s.x, s.y);
            ctx.lineTo(s.x + Math.cos(filter.pt[i]) * len, s.y + Math.sin(filter.pt[i]) * len);
            ctx.stroke();
          }
        }
      }

      // covariance ellipse
      if (layers.robots && layers.covariance) {
        const cov = filter.getCov();
        const est = filter.getEst();
        // eigenvalues/vectors of 2x2 covariance
        const a = cov.xx, b = cov.xy, d = cov.yy;
        const tr = a + d, det = a * d - b * b;
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
        const angle = Math.atan2(2 * b, a - d) / 2;
        const sigmaScale = 2.0; // 2σ
        const rx = sigmaScale * Math.sqrt(Math.max(0.5, l1)) * v.scale;
        const ry = sigmaScale * Math.sqrt(Math.max(0.5, l2)) * v.scale;
        const cs = mapToScreen(est.x, est.y);
        ctx.save();
        ctx.translate(cs.x, cs.y);
        ctx.rotate(angle);
        ctx.strokeStyle = 'rgba(255,180,60,0.8)';
        ctx.fillStyle = 'rgba(255,180,60,0.10)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      }

      // ground truth marker (green)
      if (layers.robots) {
        const truth = filter.getTruth();
        const ts = mapToScreen(truth.x, truth.y);
        ctx.save();
        ctx.translate(ts.x, ts.y);
        ctx.rotate(truth.theta);
        const R = 10;
        ctx.fillStyle = 'rgba(80,220,160,0.95)';
        ctx.strokeStyle = '#0c0d10';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(R, 0);
        ctx.lineTo(-R * 0.7, R * 0.7);
        ctx.lineTo(-R * 0.4, 0);
        ctx.lineTo(-R * 0.7, -R * 0.7);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
        ctx.fillStyle = 'rgba(80,220,160,1)';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillText('GT', ts.x + 12, ts.y - 10);

        // estimate marker (amber, hollow)
        const est = filter.getEst();
        const es = mapToScreen(est.x, est.y);
        ctx.save();
        ctx.translate(es.x, es.y);
        ctx.rotate(est.theta);
        ctx.strokeStyle = 'rgba(255,180,60,1)';
        ctx.fillStyle = 'rgba(255,180,60,0.18)';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(R, 0);
        ctx.lineTo(-R * 0.7, R * 0.7);
        ctx.lineTo(-R * 0.4, 0);
        ctx.lineTo(-R * 0.7, -R * 0.7);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
        ctx.fillStyle = 'rgba(255,180,60,1)';
        ctx.fillText('EST', es.x + 12, es.y + 14);
      }

      // prior drag arrow
      if (priorDrag) {
        const a = mapToScreen(priorDrag.x0, priorDrag.y0);
        const b = mapToScreen(priorDrag.x1, priorDrag.y1);
        ctx.strokeStyle = 'rgba(120,220,255,0.95)';
        ctx.fillStyle = 'rgba(120,220,255,0.18)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(a.x, a.y, 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        // arrowhead
        const ang = Math.atan2(b.y - a.y, b.x - a.x);
        const ah = 8;
        ctx.beginPath();
        ctx.moveTo(b.x, b.y);
        ctx.lineTo(b.x - Math.cos(ang - 0.4) * ah, b.y - Math.sin(ang - 0.4) * ah);
        ctx.lineTo(b.x - Math.cos(ang + 0.4) * ah, b.y - Math.sin(ang + 0.4) * ah);
        ctx.closePath();
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [filter, size, theme, particleStyle, layers, priorDrag, bitmap, mapToScreen, estPath, truthPath]);

  return (
    <div ref={wrapRef} className="map-wrap">
      <canvas
        ref={canvasRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
        onContextMenu={(e) => e.preventDefault()}
        style={{ display: 'block', cursor: dragStateRef.current?.kind === 'pan' ? 'grabbing' : 'crosshair' }}
      />

      {/* Map HUD */}
      <div className="map-hud-tl">
        <div className="hud-chip">MAP / occupancy_grid.pgm</div>
        <div className="hud-chip dim">{MAP_W}×{MAP_H} · 0.05 m/px</div>
      </div>
      <div className="map-hud-tr">
        <div className="hud-chip">×{view.scale.toFixed(2)}</div>
      </div>
      <div className="map-hud-bl">
        {hover ? (
          <div className="hud-chip mono">
            x={hover.x.toFixed(1)} &nbsp; y={hover.y.toFixed(1)} &nbsp;
            ({(hover.x * 0.05).toFixed(2)}m, {(hover.y * 0.05).toFixed(2)}m)
          </div>
        ) : (
          <div className="hud-chip mono dim">left-drag: set prior · right-drag: pan · wheel: zoom</div>
        )}
      </div>
      <div className="map-hud-br">
        <button className="zoom-btn" onClick={() => setView((v) => ({ ...v, scale: Math.min(8, v.scale * 1.25) }))}>+</button>
        <button className="zoom-btn" onClick={() => setView((v) => ({ ...v, scale: Math.max(0.6, v.scale / 1.25) }))}>−</button>
        <button className="zoom-btn" onClick={() => setView({ scale: 2.4, cx: MAP_W / 2, cy: MAP_H / 2 })}>⊡</button>
      </div>
    </div>
  );
}

Object.assign(window, { MapView });
