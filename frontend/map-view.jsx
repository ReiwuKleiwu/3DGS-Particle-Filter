function weightColor(weight, maxWeight) {
  const safeMax = Math.max(maxWeight || 0, 1e-9);
  const t = Math.max(0, Math.min(1, weight / safeMax));
  const stops = [
    [0.00, [40, 90, 130]],
    [0.35, [80, 200, 230]],
    [0.70, [255, 200, 90]],
    [1.00, [255, 90, 140]],
  ];
  for (let index = 1; index < stops.length; index += 1) {
    if (t <= stops[index][0]) {
      const [t0, c0] = stops[index - 1];
      const [t1, c1] = stops[index];
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

function computeCovarianceFromParticles(particles) {
  if (!particles || particles.length === 0) return null;
  let weightSum = 0;
  let meanX = 0;
  let meanY = 0;
  for (const particle of particles) {
    const weight = Math.max(0, particle.weight);
    weightSum += weight;
    meanX += particle.x * weight;
    meanY += particle.y * weight;
  }
  if (weightSum <= 0) return null;
  meanX /= weightSum;
  meanY /= weightSum;
  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (const particle of particles) {
    const weight = Math.max(0, particle.weight) / weightSum;
    const dx = particle.x - meanX;
    const dy = particle.y - meanY;
    xx += dx * dx * weight;
    yy += dy * dy * weight;
    xy += dx * dy * weight;
  }
  return { xx, yy, xy };
}

function MapView({ snapshot, mapMetadata, mapImage, layers, particleStyle, estimatedPath, groundTruthPath, onSetPrior }) {
  const wrapRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const [size, setSize] = React.useState({ w: 800, h: 600 });
  const [view, setView] = React.useState({ scale: 1, cx: 0, cy: 0 });
  const viewRef = React.useRef(view);
  viewRef.current = view;
  const [priorDrag, setPriorDrag] = React.useState(null);
  const [hover, setHover] = React.useState(null);
  const dragStateRef = React.useRef(null);

  React.useEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      const rect = element.getBoundingClientRect();
      setSize({ w: Math.max(100, rect.width), h: Math.max(100, rect.height) });
    });
    observer.observe(element);
    const rect = element.getBoundingClientRect();
    setSize({ w: Math.max(100, rect.width), h: Math.max(100, rect.height) });
    return () => observer.disconnect();
  }, []);

  React.useEffect(() => {
    if (!mapMetadata || size.w <= 0 || size.h <= 0) return;
    const fitScale = Math.min(size.w / mapMetadata.width, size.h / mapMetadata.height) * 0.96;
    setView({ scale: fitScale, cx: mapMetadata.width / 2, cy: mapMetadata.height / 2 });
  }, [mapMetadata, size.w, size.h]);

  const worldToImage = React.useCallback((x, y) => {
    if (!mapMetadata) return { x: 0, y: 0 };
    const [originX, originY, originYaw] = mapMetadata.origin;
    const dx = x - originX;
    const dy = y - originY;
    const cosine = Math.cos(originYaw);
    const sine = Math.sin(originYaw);
    const mapX = (cosine * dx + sine * dy) / mapMetadata.resolution;
    const mapY = (-sine * dx + cosine * dy) / mapMetadata.resolution;
    return { x: mapX, y: mapMetadata.height - mapY };
  }, [mapMetadata]);

  const imageToWorld = React.useCallback((imageX, imageY) => {
    if (!mapMetadata) return { x: 0, y: 0 };
    const mapY = mapMetadata.height - imageY;
    const [originX, originY, originYaw] = mapMetadata.origin;
    const cosine = Math.cos(originYaw);
    const sine = Math.sin(originYaw);
    const dx = mapMetadata.resolution * (cosine * imageX - sine * mapY);
    const dy = mapMetadata.resolution * (sine * imageX + cosine * mapY);
    return { x: originX + dx, y: originY + dy };
  }, [mapMetadata]);

  const imageToScreen = React.useCallback((imageX, imageY) => {
    const currentView = viewRef.current;
    return {
      x: (imageX - currentView.cx) * currentView.scale + size.w / 2,
      y: (imageY - currentView.cy) * currentView.scale + size.h / 2,
    };
  }, [size.w, size.h]);

  const screenToImage = React.useCallback((screenX, screenY) => {
    const currentView = viewRef.current;
    return {
      x: (screenX - size.w / 2) / currentView.scale + currentView.cx,
      y: (screenY - size.h / 2) / currentView.scale + currentView.cy,
    };
  }, [size.w, size.h]);

  const worldToScreen = React.useCallback((x, y) => {
    const imagePoint = worldToImage(x, y);
    return imageToScreen(imagePoint.x, imagePoint.y);
  }, [worldToImage, imageToScreen]);

  const screenToWorld = React.useCallback((screenX, screenY) => {
    const imagePoint = screenToImage(screenX, screenY);
    return imageToWorld(imagePoint.x, imagePoint.y);
  }, [screenToImage, imageToWorld]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const onWheel = (event) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const screenX = event.clientX - rect.left;
      const screenY = event.clientY - rect.top;
      const before = screenToImage(screenX, screenY);
      const factor = Math.exp(-event.deltaY * 0.0015);
      setView((currentView) => {
        const nextScale = Math.max(0.2, Math.min(20, currentView.scale * factor));
        const nextCx = before.x - (screenX - size.w / 2) / nextScale;
        const nextCy = before.y - (screenY - size.h / 2) / nextScale;
        return { scale: nextScale, cx: nextCx, cy: nextCy };
      });
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [screenToImage, size.w, size.h]);

  function beginPriorDrag(screenX, screenY) {
    if (!mapMetadata) return;
    const startWorld = screenToWorld(screenX, screenY);
    dragStateRef.current = { kind: 'prior', startWorld };
    setPriorDrag({ x0: startWorld.x, y0: startWorld.y, x1: startWorld.x, y1: startWorld.y });
  }

  function beginPan(screenX, screenY) {
    dragStateRef.current = {
      kind: 'pan',
      screenX,
      screenY,
      cx: viewRef.current.cx,
      cy: viewRef.current.cy,
    };
  }

  function handleMouseDown(event) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const screenX = event.clientX - rect.left;
    const screenY = event.clientY - rect.top;
    if (event.button === 0 && !event.shiftKey) {
      beginPriorDrag(screenX, screenY);
    } else {
      beginPan(screenX, screenY);
    }
  }

  function handleMouseMove(event) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const screenX = event.clientX - rect.left;
    const screenY = event.clientY - rect.top;
    setHover(screenToWorld(screenX, screenY));

    const dragState = dragStateRef.current;
    if (!dragState) return;

    if (dragState.kind === 'pan') {
      const currentView = viewRef.current;
      setView({
        scale: currentView.scale,
        cx: dragState.cx - (screenX - dragState.screenX) / currentView.scale,
        cy: dragState.cy - (screenY - dragState.screenY) / currentView.scale,
      });
      return;
    }

    if (dragState.kind === 'prior') {
      const currentWorld = screenToWorld(screenX, screenY);
      setPriorDrag({
        x0: dragState.startWorld.x,
        y0: dragState.startWorld.y,
        x1: currentWorld.x,
        y1: currentWorld.y,
      });
    }
  }

  function finishDrag() {
    const dragState = dragStateRef.current;
    dragStateRef.current = null;
    if (!dragState || dragState.kind !== 'prior' || !priorDrag) return;
    const dx = priorDrag.x1 - priorDrag.x0;
    const dy = priorDrag.y1 - priorDrag.y0;
    const magnitude = Math.hypot(dx, dy);
    const theta = magnitude > 0.05 ? Math.atan2(dy, dx) : 0;
    onSetPrior?.({ x: priorDrag.x0, y: priorDrag.y0, theta });
    setPriorDrag(null);
  }

  function cancelPan() {
    if (dragStateRef.current?.kind === 'pan') dragStateRef.current = null;
  }

  React.useEffect(() => {
    const handleWindowMouseUp = () => finishDrag();
    window.addEventListener('mouseup', handleWindowMouseUp);
    return () => window.removeEventListener('mouseup', handleWindowMouseUp);
  }, [priorDrag]);

  React.useEffect(() => {
    let frameId = null;
    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        frameId = requestAnimationFrame(draw);
        return;
      }

      const context = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== size.w * dpr || canvas.height !== size.h * dpr) {
        canvas.width = size.w * dpr;
        canvas.height = size.h * dpr;
        canvas.style.width = `${size.w}px`;
        canvas.style.height = `${size.h}px`;
      }
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, size.w, size.h);
      context.fillStyle = '#0c0d10';
      context.fillRect(0, 0, size.w, size.h);

      if (mapImage && mapMetadata && layers.grid) {
        const topLeft = imageToScreen(0, 0);
        context.imageSmoothingEnabled = false;
        context.globalAlpha = 0.95;
        context.drawImage(mapImage, topLeft.x, topLeft.y, mapMetadata.width * view.scale, mapMetadata.height * view.scale);
        context.globalAlpha = 1;
      }

      if (mapMetadata) {
        const topLeft = imageToScreen(0, 0);
        const bottomRight = imageToScreen(mapMetadata.width, mapMetadata.height);
        context.strokeStyle = 'rgba(120,160,200,0.25)';
        context.lineWidth = 1;
        context.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
      }

      if (layers.trail && groundTruthPath?.length > 1) {
        context.strokeStyle = 'rgba(80,220,160,0.35)';
        context.lineWidth = 1.5;
        context.setLineDash([4, 4]);
        context.beginPath();
        groundTruthPath.forEach((point, index) => {
          const screen = worldToScreen(point.x, point.y);
          if (index === 0) context.moveTo(screen.x, screen.y); else context.lineTo(screen.x, screen.y);
        });
        context.stroke();
        context.setLineDash([]);
      }

      if (layers.trail && estimatedPath?.length > 1) {
        context.strokeStyle = 'rgba(255,180,60,0.55)';
        context.lineWidth = 1.5;
        context.beginPath();
        estimatedPath.forEach((point, index) => {
          const screen = worldToScreen(point.x, point.y);
          if (index === 0) context.moveTo(screen.x, screen.y); else context.lineTo(screen.x, screen.y);
        });
        context.stroke();
      }

      const particles = snapshot?.particles || [];
      let maxWeight = 0;
      particles.forEach((particle) => { if (particle.weight > maxWeight) maxWeight = particle.weight; });

      if (layers.particles && layers.heatmap && particles.length > 0) {
        const cellSize = 8;
        const cellsX = Math.ceil(size.w / cellSize);
        const cellsY = Math.ceil(size.h / cellSize);
        const density = new Float32Array(cellsX * cellsY);
        particles.forEach((particle) => {
          const screen = worldToScreen(particle.x, particle.y);
          const cellX = Math.floor(screen.x / cellSize);
          const cellY = Math.floor(screen.y / cellSize);
          if (cellX < 0 || cellY < 0 || cellX >= cellsX || cellY >= cellsY) return;
          density[cellY * cellsX + cellX] += particle.weight;
        });
        let maxDensity = 0;
        density.forEach((value) => { if (value > maxDensity) maxDensity = value; });
        if (maxDensity > 0) {
          for (let y = 0; y < cellsY; y += 1) {
            for (let x = 0; x < cellsX; x += 1) {
              const alpha = density[y * cellsX + x] / maxDensity;
              if (alpha < 0.02) continue;
              const color = weightColor(alpha, 1);
              context.fillStyle = `rgba(${color[0] | 0},${color[1] | 0},${color[2] | 0},${0.18 + alpha * 0.6})`;
              context.fillRect(x * cellSize, y * cellSize, cellSize + 0.5, cellSize + 0.5);
            }
          }
        }
      } else if (layers.particles) {
        particles.forEach((particle) => {
          const screen = worldToScreen(particle.x, particle.y);
          if (screen.x < -10 || screen.y < -10 || screen.x > size.w + 10 || screen.y > size.h + 10) return;
          const color = weightColor(particle.weight, maxWeight || 1);
          const alpha = 0.4 + 0.6 * Math.min(1, particle.weight / Math.max(maxWeight, 1e-9));
          context.fillStyle = `rgba(${color[0] | 0},${color[1] | 0},${color[2] | 0},${alpha})`;
          const forward = worldToScreen(
            particle.x + Math.cos(particle.yaw) * 0.08,
            particle.y + Math.sin(particle.yaw) * 0.08,
          );
          if (particleStyle === 'arrow') {
            context.strokeStyle = context.fillStyle;
            context.lineWidth = 1.2;
            context.beginPath();
            context.moveTo(screen.x, screen.y);
            context.lineTo(forward.x, forward.y);
            context.stroke();
          } else if (particleStyle === 'triangle') {
            const sideA = worldToScreen(
              particle.x + Math.cos(particle.yaw + 2.5) * 0.05,
              particle.y + Math.sin(particle.yaw + 2.5) * 0.05,
            );
            const sideB = worldToScreen(
              particle.x + Math.cos(particle.yaw - 2.5) * 0.05,
              particle.y + Math.sin(particle.yaw - 2.5) * 0.05,
            );
            context.beginPath();
            context.moveTo(forward.x, forward.y);
            context.lineTo(sideA.x, sideA.y);
            context.lineTo(sideB.x, sideB.y);
            context.closePath();
            context.fill();
          } else {
            context.beginPath();
            context.arc(screen.x, screen.y, 1.8, 0, Math.PI * 2);
            context.fill();
            context.strokeStyle = context.fillStyle;
            context.lineWidth = 0.9;
            context.beginPath();
            context.moveTo(screen.x, screen.y);
            context.lineTo(forward.x, forward.y);
            context.stroke();
          }
        });
      }

      if (layers.robots && layers.covariance && particles.length > 1) {
        const covariance = computeCovarianceFromParticles(particles);
        const estimate = snapshot?.estimated_pose;
        if (covariance && estimate && mapMetadata) {
          const a = covariance.xx;
          const b = covariance.xy;
          const d = covariance.yy;
          const trace = a + d;
          const determinant = a * d - b * b;
          const discriminant = Math.sqrt(Math.max(0, (trace * trace) / 4 - determinant));
          const lambda1 = trace / 2 + discriminant;
          const lambda2 = trace / 2 - discriminant;
          const angleWorld = Math.atan2(2 * b, a - d) / 2;
          const angleMap = angleWorld - mapMetadata.origin[2];
          const radiusX = 2.0 * Math.sqrt(Math.max(1e-6, lambda1)) / mapMetadata.resolution * view.scale;
          const radiusY = 2.0 * Math.sqrt(Math.max(1e-6, lambda2)) / mapMetadata.resolution * view.scale;
          const center = worldToScreen(estimate.x, estimate.y);
          context.save();
          context.translate(center.x, center.y);
          context.rotate(-angleMap);
          context.strokeStyle = 'rgba(255,180,60,0.8)';
          context.fillStyle = 'rgba(255,180,60,0.10)';
          context.lineWidth = 1.2;
          context.setLineDash([3, 3]);
          context.beginPath();
          context.ellipse(0, 0, radiusX, radiusY, 0, 0, Math.PI * 2);
          context.fill();
          context.stroke();
          context.setLineDash([]);
          context.restore();
        }
      }

      if (layers.robots && snapshot?.ground_truth_pose) {
        drawRobotMarker(context, worldToScreen, snapshot.ground_truth_pose, 'rgba(80,220,160,0.95)', '#0c0d10', 'GT');
      }
      if (layers.robots && snapshot?.estimated_pose) {
        drawRobotMarker(context, worldToScreen, snapshot.estimated_pose, 'rgba(255,180,60,0.18)', 'rgba(255,180,60,1)', 'EST');
      }

      if (priorDrag) {
        const start = worldToScreen(priorDrag.x0, priorDrag.y0);
        const end = worldToScreen(priorDrag.x1, priorDrag.y1);
        context.strokeStyle = 'rgba(120,220,255,0.95)';
        context.fillStyle = 'rgba(120,220,255,0.18)';
        context.lineWidth = 2;
        context.beginPath();
        context.arc(start.x, start.y, 12, 0, Math.PI * 2);
        context.fill();
        context.stroke();
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
        const angle = Math.atan2(end.y - start.y, end.x - start.x);
        const arrow = 8;
        context.beginPath();
        context.moveTo(end.x, end.y);
        context.lineTo(end.x - Math.cos(angle - 0.4) * arrow, end.y - Math.sin(angle - 0.4) * arrow);
        context.lineTo(end.x - Math.cos(angle + 0.4) * arrow, end.y - Math.sin(angle + 0.4) * arrow);
        context.closePath();
        context.fill();
      }

      frameId = requestAnimationFrame(draw);
    };

    frameId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frameId);
  }, [snapshot, mapMetadata, mapImage, layers, particleStyle, priorDrag, size, view.scale, estimatedPath, groundTruthPath, worldToScreen, imageToScreen]);

  return (
    <div ref={wrapRef} className="map-wrap">
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={finishDrag}
        onMouseLeave={() => { setHover(null); cancelPan(); }}
        onContextMenu={(event) => event.preventDefault()}
        style={{ display: 'block', cursor: dragStateRef.current?.kind === 'pan' ? 'grabbing' : 'crosshair' }}
      />

      <div className="map-hud-tl">
        <div className="hud-chip">MAP / {mapMetadata ? 'map.pgm' : 'not available'}</div>
        <div className="hud-chip dim">{mapMetadata ? `${mapMetadata.width}×${mapMetadata.height} · ${mapMetadata.resolution.toFixed(3)} m/px` : 'waiting for metadata'}</div>
      </div>
      <div className="map-hud-tr">
        <div className="hud-chip">×{view.scale.toFixed(2)}</div>
      </div>
      <div className="map-hud-bl">
        {hover ? (
          <div className="hud-chip mono">
            x={hover.x.toFixed(3)} y={hover.y.toFixed(3)}
          </div>
        ) : (
          <div className="hud-chip mono dim">left-drag: set prior · right-drag: pan · wheel: zoom</div>
        )}
      </div>
      <div className="map-hud-br">
        <button className="zoom-btn" onClick={() => setView((currentView) => ({ ...currentView, scale: Math.min(20, currentView.scale * 1.25) }))}>+</button>
        <button className="zoom-btn" onClick={() => setView((currentView) => ({ ...currentView, scale: Math.max(0.2, currentView.scale / 1.25) }))}>−</button>
        <button className="zoom-btn" onClick={() => {
          if (!mapMetadata) return;
          const fitScale = Math.min(size.w / mapMetadata.width, size.h / mapMetadata.height) * 0.96;
          setView({ scale: fitScale, cx: mapMetadata.width / 2, cy: mapMetadata.height / 2 });
        }}>⊡</button>
      </div>
    </div>
  );
}

function drawRobotMarker(context, worldToScreen, pose, fillStyle, strokeStyle, label) {
  const center = worldToScreen(pose.x, pose.y);
  const tip = worldToScreen(pose.x + Math.cos(pose.yaw) * 0.18, pose.y + Math.sin(pose.yaw) * 0.18);
  const left = worldToScreen(pose.x + Math.cos(pose.yaw + 2.55) * 0.12, pose.y + Math.sin(pose.yaw + 2.55) * 0.12);
  const right = worldToScreen(pose.x + Math.cos(pose.yaw - 2.55) * 0.12, pose.y + Math.sin(pose.yaw - 2.55) * 0.12);
  context.fillStyle = fillStyle;
  context.strokeStyle = strokeStyle;
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(tip.x, tip.y);
  context.lineTo(left.x, left.y);
  context.lineTo(center.x, center.y);
  context.lineTo(right.x, right.y);
  context.closePath();
  context.fill();
  context.stroke();
  context.fillStyle = strokeStyle;
  context.font = '10px "JetBrains Mono", monospace';
  context.fillText(label, center.x + 12, center.y + (label === 'GT' ? -10 : 14));
}

Object.assign(window, { MapView });
