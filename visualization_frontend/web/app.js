const POLL_INTERVAL_MS = 300;

const connectionStatus = document.getElementById("connection-status");
const summaryMetrics = document.getElementById("summary-metrics");
const worldMeta = document.getElementById("world-meta");
const imageMeta = document.getElementById("image-meta");
const estimatedPose = document.getElementById("estimated-pose");
const groundTruthPose = document.getElementById("ground-truth-pose");
const metricsDetail = document.getElementById("metrics-detail");
const observationImage = document.getElementById("observation-image");
const renderImage = document.getElementById("render-image");
const resetStatus = document.getElementById("reset-status");
const toolDefaults = document.getElementById("tool-defaults");
const togglePoseToolButton = document.getElementById("toggle-pose-tool-button");
const worldCanvas = document.getElementById("world-canvas");
const worldContext = worldCanvas.getContext("2d");

let lastUpdateIndex = null;
let latestSnapshot = null;
let mapMetadata = null;
let mapImage = null;
let resetDefaults = { yaw: 0.0, sigma_x: 0.5, sigma_y: 0.5, sigma_yaw: 0.5 };
let poseToolEnabled = false;
let poseToolDragState = null;

const viewport = {
  scale: 1.0,
  minScale: 0.5,
  maxScale: 8.0,
  offsetX: 0.0,
  offsetY: 0.0,
  isDragging: false,
  dragMoved: false,
  lastPointerX: 0.0,
  lastPointerY: 0.0,
};

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function poseToText(pose) {
  if (!pose) {
    return "not available";
  }
  return [
    `x:   ${formatNumber(pose.x)}`,
    `y:   ${formatNumber(pose.y)}`,
    `yaw: ${formatNumber(pose.yaw)}`,
  ].join("\n");
}

function metricsToText(snapshot) {
  const metrics = snapshot.metrics;
  return [
    `best_particle_index:        ${metrics.best_particle_index}`,
    `best_score:                 ${formatNumber(metrics.best_score, 6)}`,
    `effective_particle_count:   ${formatNumber(metrics.effective_particle_count, 2)}`,
    `render_and_score_ms:        ${formatNumber(metrics.render_and_score_milliseconds, 1)}`,
    `resampled:                  ${metrics.resampled}`,
    `received_at_unix_seconds:   ${formatNumber(snapshot.received_at_unix_seconds, 3)}`,
    `zoom:                       ${formatNumber(viewport.scale, 2)}x`,
  ].join("\n");
}

function renderMetricCards(snapshot) {
  const metrics = snapshot.metrics;
  const items = [
    ["Update", snapshot.update_index],
    ["Best score", formatNumber(metrics.best_score, 6)],
    ["Neff", formatNumber(metrics.effective_particle_count, 2)],
    ["Render+score ms", formatNumber(metrics.render_and_score_milliseconds, 1)],
    ["Best particle", metrics.best_particle_index],
    ["Resampled", metrics.resampled ? "yes" : "no"],
  ];

  summaryMetrics.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="metric-card">
          <span class="label">${label}</span>
          <span class="value">${value}</span>
        </div>`
    )
    .join("");
}

function computeBounds(snapshot) {
  const xs = snapshot.particles.map((particle) => particle.x);
  const ys = snapshot.particles.map((particle) => particle.y);
  xs.push(snapshot.estimated_pose.x);
  ys.push(snapshot.estimated_pose.y);
  if (snapshot.ground_truth_pose) {
    xs.push(snapshot.ground_truth_pose.x);
    ys.push(snapshot.ground_truth_pose.y);
  }

  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);

  const width = Math.max(maxX - minX, 0.5);
  const height = Math.max(maxY - minY, 0.5);
  const paddingX = width * 0.15;
  const paddingY = height * 0.15;

  minX -= paddingX;
  maxX += paddingX;
  minY -= paddingY;
  maxY += paddingY;

  return { minX, maxX, minY, maxY };
}

function worldToCanvasAuto(x, y, bounds) {
  const padding = 24;
  const usableWidth = worldCanvas.width - padding * 2;
  const usableHeight = worldCanvas.height - padding * 2;
  const normalizedX = (x - bounds.minX) / (bounds.maxX - bounds.minX || 1);
  const normalizedY = (y - bounds.minY) / (bounds.maxY - bounds.minY || 1);
  return {
    x: padding + normalizedX * usableWidth,
    y: worldCanvas.height - padding - normalizedY * usableHeight,
  };
}

function worldToCanvasMap(x, y, metadata) {
  const [originX, originY, originYaw] = metadata.origin;
  const dx = x - originX;
  const dy = y - originY;
  const cosine = Math.cos(originYaw);
  const sine = Math.sin(originYaw);
  const mapX = (cosine * dx + sine * dy) / metadata.resolution;
  const mapY = (-sine * dx + cosine * dy) / metadata.resolution;
  return {
    x: (mapX / metadata.width) * worldCanvas.width,
    y: worldCanvas.height - (mapY / metadata.height) * worldCanvas.height,
  };
}

function canvasToWorldMap(canvasX, canvasY, metadata) {
  const baseX = (canvasX - viewport.offsetX) / viewport.scale;
  const baseY = (canvasY - viewport.offsetY) / viewport.scale;
  const mapX = (baseX / worldCanvas.width) * metadata.width;
  const mapY = ((worldCanvas.height - baseY) / worldCanvas.height) * metadata.height;

  const [originX, originY, originYaw] = metadata.origin;
  const cosine = Math.cos(originYaw);
  const sine = Math.sin(originYaw);
  const dx = metadata.resolution * (cosine * mapX - sine * mapY);
  const dy = metadata.resolution * (sine * mapX + cosine * mapY);

  return {
    x: originX + dx,
    y: originY + dy,
  };
}

function drawHeadingArrow(x, y, yaw, color, lineWidth, size = 12) {
  const dx = Math.cos(yaw) * size;
  const dy = -Math.sin(yaw) * size;

  worldContext.strokeStyle = color;
  worldContext.fillStyle = color;
  worldContext.lineWidth = lineWidth / viewport.scale;
  worldContext.beginPath();
  worldContext.moveTo(x, y);
  worldContext.lineTo(x + dx, y + dy);
  worldContext.stroke();

  const leftX = x + dx - Math.cos(yaw - Math.PI / 6) * (size * 0.4);
  const leftY = y + dy + Math.sin(yaw - Math.PI / 6) * (size * 0.4);
  const rightX = x + dx - Math.cos(yaw + Math.PI / 6) * (size * 0.4);
  const rightY = y + dy + Math.sin(yaw + Math.PI / 6) * (size * 0.4);

  worldContext.beginPath();
  worldContext.moveTo(x + dx, y + dy);
  worldContext.lineTo(leftX, leftY);
  worldContext.lineTo(rightX, rightY);
  worldContext.closePath();
  worldContext.fill();
}

function canvasPointFromEvent(event) {
  const rect = worldCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (worldCanvas.width / rect.width),
    y: (event.clientY - rect.top) * (worldCanvas.height / rect.height),
  };
}

function zoomAround(canvasX, canvasY, zoomMultiplier) {
  const newScale = Math.max(viewport.minScale, Math.min(viewport.maxScale, viewport.scale * zoomMultiplier));
  if (newScale === viewport.scale) {
    return;
  }

  const baseX = (canvasX - viewport.offsetX) / viewport.scale;
  const baseY = (canvasY - viewport.offsetY) / viewport.scale;
  viewport.scale = newScale;
  viewport.offsetX = canvasX - baseX * viewport.scale;
  viewport.offsetY = canvasY - baseY * viewport.scale;
}

function resetViewport() {
  viewport.scale = 1.0;
  viewport.offsetX = 0.0;
  viewport.offsetY = 0.0;
}

function renderToolDefaults() {
  toolDefaults.textContent = [
    `yaw:       ${formatNumber(resetDefaults.yaw)}`,
    `sigma_x:   ${formatNumber(resetDefaults.sigma_x)}`,
    `sigma_y:   ${formatNumber(resetDefaults.sigma_y)}`,
    `sigma_yaw: ${formatNumber(resetDefaults.sigma_yaw)}`,
  ].join("\n");
}

function updatePoseToolUi() {
  togglePoseToolButton.classList.toggle("button-active", poseToolEnabled);
  worldCanvas.classList.toggle("world-canvas-tool-mode", poseToolEnabled);
  if (poseToolEnabled) {
    resetStatus.textContent = "Pose tool active. Click and drag on the map to set x, y, yaw.";
    worldCanvas.style.cursor = "crosshair";
  } else {
    if (!resetStatus.textContent.includes("queued")) {
      resetStatus.textContent = "Tool inactive.";
    }
    worldCanvas.style.cursor = "grab";
  }
}

function drawPoseToolPreview(mapper) {
  if (!poseToolDragState) {
    return;
  }
  const start = mapper(poseToolDragState.startWorld.x, poseToolDragState.startWorld.y);
  const yaw = poseToolDragState.previewYaw;
  drawHeadingArrow(start.x, start.y, yaw, "#ffb454", 4, 22 / viewport.scale);
}

function drawWorldBackground(snapshot) {
  worldContext.setTransform(1, 0, 0, 1, 0, 0);
  worldContext.clearRect(0, 0, worldCanvas.width, worldCanvas.height);
  worldContext.setTransform(viewport.scale, 0, 0, viewport.scale, viewport.offsetX, viewport.offsetY);

  if (mapMetadata && mapImage) {
    worldContext.imageSmoothingEnabled = false;
    worldContext.drawImage(mapImage, 0, 0, worldCanvas.width, worldCanvas.height);
    worldMeta.textContent = `map underlay | zoom=${formatNumber(viewport.scale, 2)}x | drag=pan | wheel=zoom | dblclick=reset viewport${poseToolEnabled ? ' | pose tool active: drag to set prior' : ''}`;
    return { mapper: (x, y) => worldToCanvasMap(x, y, mapMetadata) };
  }

  worldContext.imageSmoothingEnabled = true;
  const bounds = computeBounds(snapshot);
  worldContext.fillStyle = "#0b0f14";
  worldContext.fillRect(0, 0, worldCanvas.width, worldCanvas.height);

  worldContext.strokeStyle = "#1f2630";
  worldContext.lineWidth = 1 / viewport.scale;
  for (let x = 0; x <= 10; x += 1) {
    const screenX = (x / 10) * worldCanvas.width;
    worldContext.beginPath();
    worldContext.moveTo(screenX, 0);
    worldContext.lineTo(screenX, worldCanvas.height);
    worldContext.stroke();
  }
  for (let y = 0; y <= 8; y += 1) {
    const screenY = (y / 8) * worldCanvas.height;
    worldContext.beginPath();
    worldContext.moveTo(0, screenY);
    worldContext.lineTo(worldCanvas.width, screenY);
    worldContext.stroke();
  }

  worldMeta.textContent = `x:[${formatNumber(bounds.minX)}, ${formatNumber(bounds.maxX)}] y:[${formatNumber(bounds.minY)}, ${formatNumber(bounds.maxY)}] | zoom=${formatNumber(viewport.scale, 2)}x | drag=pan | wheel=zoom | dblclick=reset viewport`;
  return { mapper: (x, y) => worldToCanvasAuto(x, y, bounds) };
}

function renderWorld(snapshot) {
  const worldProjection = drawWorldBackground(snapshot);

  snapshot.particles.forEach((particle) => {
    const point = worldProjection.mapper(particle.x, particle.y);
    const alpha = Math.max(0.08, Math.min(1.0, particle.weight * snapshot.particles.length * 0.85));
    worldContext.fillStyle = `rgba(255, 180, 84, ${alpha})`;
    worldContext.beginPath();
    worldContext.arc(point.x, point.y, 2.6 / viewport.scale, 0, Math.PI * 2);
    worldContext.fill();
  });

  snapshot.particles
    .slice()
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 24)
    .forEach((particle) => {
      const point = worldProjection.mapper(particle.x, particle.y);
      drawHeadingArrow(point.x, point.y, particle.yaw, "rgba(255, 180, 84, 0.65)", 1.2, 10 / viewport.scale);
    });

  const estimatedPoint = worldProjection.mapper(snapshot.estimated_pose.x, snapshot.estimated_pose.y);
  drawHeadingArrow(estimatedPoint.x, estimatedPoint.y, snapshot.estimated_pose.yaw, "#42d392", 3, 16 / viewport.scale);

  if (snapshot.ground_truth_pose) {
    const groundTruthPoint = worldProjection.mapper(snapshot.ground_truth_pose.x, snapshot.ground_truth_pose.y);
    drawHeadingArrow(groundTruthPoint.x, groundTruthPoint.y, snapshot.ground_truth_pose.yaw, "#69a8ff", 3, 16 / viewport.scale);
  }

  drawPoseToolPreview(worldProjection.mapper);
  worldContext.setTransform(1, 0, 0, 1, 0, 0);
}

function renderSnapshot(snapshot) {
  latestSnapshot = snapshot;
  connectionStatus.textContent = `Connected. Showing update ${snapshot.update_index}.`;
  renderMetricCards(snapshot);
  renderWorld(snapshot);

  observationImage.src = `data:image/jpeg;base64,${snapshot.images.observation_jpeg_base64}`;
  renderImage.src = `data:image/png;base64,${snapshot.images.best_render_png_base64}`;

  estimatedPose.textContent = poseToText(snapshot.estimated_pose);
  groundTruthPose.textContent = poseToText(snapshot.ground_truth_pose);
  metricsDetail.textContent = metricsToText(snapshot);
  imageMeta.textContent = `stamp=${snapshot.image_stamp_seconds}.${String(snapshot.image_stamp_nanoseconds).padStart(9, "0")}`;
}

async function submitResetPayload(payload) {
  const response = await fetch('/api/reset-particle-filter', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.detail || `HTTP ${response.status}`);
  }
  return result;
}

function buildResetPayload(meanPose) {
  return {
    prior: {
      mean: {
        x: meanPose.x,
        y: meanPose.y,
        yaw: meanPose.yaw,
      },
      sigma_x: resetDefaults.sigma_x,
      sigma_y: resetDefaults.sigma_y,
      sigma_yaw: resetDefaults.sigma_yaw,
    },
  };
}

async function queueResetFromPose(meanPose, sourceLabel) {
  const payload = buildResetPayload(meanPose);
  try {
    const result = await submitResetPayload(payload);
    resetStatus.textContent = `${sourceLabel} prior queued at ${new Date(result.command.issued_at_unix_seconds * 1000).toLocaleTimeString()} | x=${formatNumber(meanPose.x)}, y=${formatNumber(meanPose.y)}, yaw=${formatNumber(meanPose.yaw)}`;
  } catch (error) {
    resetStatus.textContent = `${sourceLabel} prior failed: ${error}`;
  }
}

function beginPoseToolDrag(startPoint) {
  if (!mapMetadata) {
    resetStatus.textContent = 'Pose tool requires a map underlay.';
    return;
  }
  const startWorld = canvasToWorldMap(startPoint.x, startPoint.y, mapMetadata);
  poseToolDragState = {
    startCanvas: startPoint,
    currentCanvas: startPoint,
    startWorld,
    previewYaw: resetDefaults.yaw,
  };
}

function updatePoseToolDrag(currentPoint) {
  if (!poseToolDragState) {
    return;
  }
  poseToolDragState.currentCanvas = currentPoint;
  const deltaX = currentPoint.x - poseToolDragState.startCanvas.x;
  const deltaY = currentPoint.y - poseToolDragState.startCanvas.y;
  if (Math.hypot(deltaX, deltaY) > 2.0) {
    poseToolDragState.previewYaw = Math.atan2(-(deltaY), deltaX);
  }
  if (latestSnapshot) {
    renderSnapshot(latestSnapshot);
  }
}

async function endPoseToolDrag() {
  if (!poseToolDragState) {
    return;
  }
  const meanPose = {
    x: poseToolDragState.startWorld.x,
    y: poseToolDragState.startWorld.y,
    yaw: poseToolDragState.previewYaw,
  };
  poseToolDragState = null;
  poseToolEnabled = false;
  updatePoseToolUi();
  if (latestSnapshot) {
    renderSnapshot(latestSnapshot);
  }
  await queueResetFromPose(meanPose, 'Map tool');
}

function installCanvasInteractions() {
  worldCanvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    const point = canvasPointFromEvent(event);
    const zoomMultiplier = Math.exp(-event.deltaY * 0.0015);
    zoomAround(point.x, point.y, zoomMultiplier);
    if (latestSnapshot) {
      renderSnapshot(latestSnapshot);
    }
  }, { passive: false });

  worldCanvas.addEventListener('mousedown', (event) => {
    const point = canvasPointFromEvent(event);
    viewport.isDragging = true;
    viewport.dragMoved = false;
    viewport.lastPointerX = point.x;
    viewport.lastPointerY = point.y;

    if (poseToolEnabled) {
      beginPoseToolDrag(point);
      worldCanvas.style.cursor = 'crosshair';
      return;
    }

    worldCanvas.style.cursor = 'grabbing';
  });

  worldCanvas.addEventListener('mousemove', (event) => {
    const point = canvasPointFromEvent(event);

    if (poseToolEnabled && poseToolDragState) {
      updatePoseToolDrag(point);
      return;
    }

    if (!viewport.isDragging) {
      return;
    }

    const deltaX = point.x - viewport.lastPointerX;
    const deltaY = point.y - viewport.lastPointerY;
    if (Math.abs(deltaX) > 0.5 || Math.abs(deltaY) > 0.5) {
      viewport.dragMoved = true;
    }
    viewport.offsetX += deltaX;
    viewport.offsetY += deltaY;
    viewport.lastPointerX = point.x;
    viewport.lastPointerY = point.y;
    if (latestSnapshot) {
      renderSnapshot(latestSnapshot);
    }
  });

  async function stopDragging() {
    const hadPoseToolDrag = poseToolEnabled && poseToolDragState;
    viewport.isDragging = false;
    viewport.dragMoved = false;
    worldCanvas.style.cursor = poseToolEnabled ? 'crosshair' : 'grab';
    if (hadPoseToolDrag) {
      await endPoseToolDrag();
    }
  }

  worldCanvas.addEventListener('mouseup', stopDragging);
  worldCanvas.addEventListener('mouseleave', stopDragging);
  window.addEventListener('mouseup', stopDragging);

  worldCanvas.addEventListener('dblclick', () => {
    resetViewport();
    if (latestSnapshot) {
      renderSnapshot(latestSnapshot);
    }
  });

  worldCanvas.style.cursor = 'grab';
}

async function loadMapUnderlay() {
  try {
    const response = await fetch('/api/map-metadata', { cache: 'no-store' });
    if (response.status === 204 || !response.ok) {
      return;
    }
    mapMetadata = await response.json();
    mapImage = new Image();
    mapImage.src = mapMetadata.image_url;
    await mapImage.decode();
  } catch (error) {
    console.warn('Map underlay unavailable:', error);
  }
}

async function loadResetDefaults() {
  try {
    const response = await fetch('/api/reset-defaults', { cache: 'no-store' });
    if (!response.ok) {
      return;
    }
    resetDefaults = await response.json();
    renderToolDefaults();
  } catch (error) {
    console.warn('Reset defaults unavailable:', error);
  }
}

async function pollLatest() {
  try {
    const response = await fetch('/api/latest', { cache: 'no-store' });
    if (response.status === 204) {
      connectionStatus.textContent = 'Waiting for particle filter snapshots...';
      return;
    }
    if (!response.ok) {
      connectionStatus.textContent = `Viewer API error: ${response.status}`;
      return;
    }

    const snapshot = await response.json();
    if (snapshot.update_index !== lastUpdateIndex) {
      lastUpdateIndex = snapshot.update_index;
      renderSnapshot(snapshot);
    }
  } catch (error) {
    connectionStatus.textContent = `Connection failed: ${error}`;
  }
}

async function pollResetStatus() {
  try {
    const response = await fetch('/api/reset-particle-filter/pending', { cache: 'no-store' });
    if (response.status === 204) {
      return;
    }
    if (!response.ok) {
      return;
    }
    const command = await response.json();
    resetStatus.textContent = `Pending reset queued at ${new Date(command.issued_at_unix_seconds * 1000).toLocaleTimeString()}`;
  } catch (_) {
    return;
  }
}

togglePoseToolButton.addEventListener('click', () => {
  poseToolEnabled = !poseToolEnabled;
  poseToolDragState = null;
  updatePoseToolUi();
  if (latestSnapshot) {
    renderSnapshot(latestSnapshot);
  }
});

renderToolDefaults();
updatePoseToolUi();

(async function start() {
  installCanvasInteractions();
  await loadMapUnderlay();
  await loadResetDefaults();
  setInterval(pollLatest, POLL_INTERVAL_MS);
  setInterval(pollResetStatus, 1000);
  pollLatest();
  pollResetStatus();
})();
