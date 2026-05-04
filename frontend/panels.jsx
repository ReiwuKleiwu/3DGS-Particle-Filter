function CameraImageCard({ imageSrc, label, sublabel, accent, pose, resolution }) {
  const poseLabel = pose ? `θ=${(pose.yaw * 180 / Math.PI).toFixed(1)}°` : 'θ=—';
  const positionLabel = pose ? `(${pose.x.toFixed(3)}, ${pose.y.toFixed(3)})` : '(—, —)';
  return (
    <div className="cam-card">
      <div className="cam-hd">
        <span className="cam-dot" style={{ background: accent }} />
        <span className="cam-lbl">{label}</span>
        <span className="cam-sub">{sublabel}</span>
      </div>
      <div className="cam-body">
        {imageSrc ? <img src={imageSrc} alt={label} /> : null}
        <div className="cam-corner tl">LIVE</div>
        <div className="cam-corner tr">{poseLabel}</div>
        <div className="cam-corner bl">{positionLabel}</div>
        <div className="cam-corner br">{resolution}</div>
      </div>
    </div>
  );
}

function ErrorGraph({ history, theme, mode = 'magnitude' }) {
  const ref = React.useRef(null);
  const wrapRef = React.useRef(null);
  const [size, setSize] = React.useState({ w: 320, h: 140 });

  React.useEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      const rect = element.getBoundingClientRect();
      setSize({ w: Math.max(60, rect.width), h: Math.max(60, rect.height) });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  React.useEffect(() => {
    let frameId;
    const draw = () => {
      const canvas = ref.current;
      if (!canvas) {
        frameId = requestAnimationFrame(draw);
        return;
      }
      const dpr = window.devicePixelRatio || 1;
      const W = size.w;
      const H = size.h;
      if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        canvas.style.width = `${W}px`;
        canvas.style.height = `${H}px`;
      }
      const context = canvas.getContext('2d');
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, W, H);

      const PAD_L = 42;
      const PAD_R = 8;
      const PAD_T = 12;
      const PAD_B = 18;
      const graphWidth = W - PAD_L - PAD_R;
      const graphHeight = H - PAD_T - PAD_B;
      const data = history || [];
      const tEnd = data.length ? data[data.length - 1].t : 0;
      const windowSeconds = 30;
      const rightPadFraction = 0.18;
      const plotWidth = graphWidth * (1 - rightPadFraction);
      const tStart = tEnd - windowSeconds;
      const xOf = (t) => PAD_L + ((t - tStart) / windowSeconds) * plotWidth;
      const nowX = xOf(tEnd);

      context.strokeStyle = theme === 'dark' ? 'rgba(120,140,170,0.12)' : 'rgba(40,50,70,0.15)';
      context.lineWidth = 1;
      for (let index = 0; index <= 4; index += 1) {
        const y = PAD_T + (graphHeight * index) / 4;
        context.beginPath();
        context.moveTo(PAD_L, y);
        context.lineTo(W - PAD_R, y);
        context.stroke();
      }
      for (let index = 0; index <= 6; index += 1) {
        const x = PAD_L + (plotWidth * index) / 6;
        context.beginPath();
        context.moveTo(x, PAD_T);
        context.lineTo(x, PAD_T + graphHeight);
        context.stroke();
      }
      context.strokeStyle = theme === 'dark' ? 'rgba(220,225,235,0.35)' : 'rgba(40,50,70,0.5)';
      context.setLineDash([2, 3]);
      context.beginPath();
      context.moveTo(nowX, PAD_T);
      context.lineTo(nowX, PAD_T + graphHeight);
      context.stroke();
      context.setLineDash([]);

      let yMin = 0;
      let yMax = 0;
      if (mode === 'components') {
        data.forEach((point) => {
          if (point.t < tStart) return;
          const m = Math.max(Math.abs(point.ex || 0), Math.abs(point.ey || 0), Math.abs((point.eth || 0) * 180 / Math.PI));
          if (m > yMax) yMax = m;
        });
        yMax = Math.max(0.1, yMax * 1.15);
        yMin = -yMax;
      } else {
        data.forEach((point) => {
          if (point.t >= tStart && point.err > yMax) yMax = point.err;
        });
        yMax = Math.max(0.1, yMax * 1.15);
      }
      const yOf = (value) => PAD_T + graphHeight - ((value - yMin) / (yMax - yMin)) * graphHeight;

      context.fillStyle = theme === 'dark' ? 'rgba(180,200,220,0.55)' : 'rgba(40,50,70,0.65)';
      context.font = '9px "JetBrains Mono", monospace';
      context.textAlign = 'right';
      for (let index = 0; index <= 4; index += 1) {
        const value = yMax + (yMin - yMax) * (index / 4);
        const y = PAD_T + (graphHeight * index) / 4;
        const label = mode === 'components' ? (Math.abs(value) < 0.0001 ? '0' : value.toFixed(2)) : `${value.toFixed(2)}m`;
        context.fillText(label, PAD_L - 4, y + 3);
      }
      if (mode === 'components') {
        context.strokeStyle = theme === 'dark' ? 'rgba(180,200,220,0.25)' : 'rgba(40,50,70,0.3)';
        context.beginPath();
        context.moveTo(PAD_L, yOf(0));
        context.lineTo(W - PAD_R, yOf(0));
        context.stroke();
      }
      context.textAlign = 'center';
      for (let index = 0; index <= 6; index += 1) {
        const t = tStart + windowSeconds * (index / 6);
        const x = PAD_L + (plotWidth * index) / 6;
        context.fillText(`${(t - tEnd).toFixed(0)}s`, x, H - 4);
      }
      context.textAlign = 'left';
      context.fillStyle = theme === 'dark' ? 'rgba(220,225,235,0.7)' : 'rgba(40,50,70,0.85)';
      context.fillText('now', nowX + 4, PAD_T + graphHeight - 4);

      context.save();
      context.beginPath();
      context.rect(PAD_L, PAD_T, graphWidth, graphHeight);
      context.clip();

      const drawSeries = (getter, color, fill = false) => {
        if (data.length < 2) return;
        if (fill) {
          context.beginPath();
          let started = false;
          const baseY = mode === 'components' ? yOf(0) : PAD_T + graphHeight;
          for (const point of data) {
            if (point.t < tStart) continue;
            const x = xOf(point.t);
            const y = yOf(getter(point));
            if (!started) {
              context.moveTo(x, baseY);
              context.lineTo(x, y);
              started = true;
            } else {
              context.lineTo(x, y);
            }
          }
          if (started) {
            const last = data[data.length - 1];
            context.lineTo(xOf(last.t), baseY);
            context.closePath();
            const gradient = context.createLinearGradient(0, PAD_T, 0, PAD_T + graphHeight);
            gradient.addColorStop(0, color.replace(/,\s*1\)$/u, ', 0.35)'));
            gradient.addColorStop(1, color.replace(/,\s*1\)$/u, ', 0.02)'));
            context.fillStyle = gradient;
            context.fill();
          }
        }
        context.beginPath();
        let started = false;
        for (const point of data) {
          if (point.t < tStart) continue;
          const x = xOf(point.t);
          const y = yOf(getter(point));
          if (!started) {
            context.moveTo(x, y);
            started = true;
          } else {
            context.lineTo(x, y);
          }
        }
        context.strokeStyle = color;
        context.lineWidth = 1.6;
        context.stroke();
        const last = data[data.length - 1];
        const lx = xOf(last.t);
        const ly = yOf(getter(last));
        context.fillStyle = color;
        context.beginPath();
        context.arc(lx, ly, 3, 0, Math.PI * 2);
        context.fill();
      };

      if (mode === 'components') {
        drawSeries((point) => point.ex || 0, 'rgba(120,220,255,1)');
        drawSeries((point) => point.ey || 0, 'rgba(255,180,60,1)');
        drawSeries((point) => (point.eth || 0) * 180 / Math.PI, 'rgba(255,120,180,1)');
      } else {
        drawSeries((point) => point.err || 0, 'rgba(255,180,60,1)', true);
      }

      context.restore();

      const legend = mode === 'components'
        ? [
            { c: 'rgba(120,220,255,1)', l: 'Δx (m)' },
            { c: 'rgba(255,180,60,1)', l: 'Δy (m)' },
            { c: 'rgba(255,120,180,1)', l: 'Δθ (°)' },
          ]
        : [{ c: 'rgba(255,180,60,1)', l: '‖Δp‖ (m)' }];
      context.font = '9px "JetBrains Mono", monospace';
      context.textAlign = 'left';
      let legendX = PAD_L + 6;
      legend.forEach((item) => {
        context.fillStyle = item.c;
        context.fillRect(legendX, PAD_T + 5, 8, 2);
        context.fillStyle = theme === 'dark' ? 'rgba(220,225,235,0.8)' : 'rgba(40,50,70,0.85)';
        context.fillText(item.l, legendX + 12, PAD_T + 9);
        legendX += 12 + context.measureText(item.l).width + 14;
      });

      frameId = requestAnimationFrame(draw);
    };

    frameId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frameId);
  }, [history, size, theme, mode]);

  return <div className="graph-canvas-wrap" ref={wrapRef}><canvas ref={ref} /></div>;
}

Object.assign(window, { CameraImageCard, ErrorGraph });
