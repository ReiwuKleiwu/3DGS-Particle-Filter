const POLL_INTERVAL_MS = 50;
const MAX_PATH_POINTS = 600;
const MAX_ERROR_POINTS = 1200;
const DEFAULT_LAYERS = {
  grid: true,
  particles: true,
  robots: true,
  heatmap: false,
  covariance: true,
  trail: true,
};

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

function wrapAngle(angle) {
  let value = angle;
  while (value > Math.PI) value -= 2 * Math.PI;
  while (value < -Math.PI) value += 2 * Math.PI;
  return value;
}

function computePoseError(estimatedPose, groundTruthPose) {
  if (!estimatedPose || !groundTruthPose) return null;
  const dx = estimatedPose.x - groundTruthPose.x;
  const dy = estimatedPose.y - groundTruthPose.y;
  const dyaw = wrapAngle(estimatedPose.yaw - groundTruthPose.yaw);
  return { dx, dy, dyaw, distance: Math.hypot(dx, dy) };
}

function computeParticleCovariance(particles) {
  if (!particles || particles.length === 0) return null;
  let weightSum = 0;
  let meanX = 0;
  let meanY = 0;
  let sinYaw = 0;
  let cosYaw = 0;
  for (const particle of particles) {
    const weight = Math.max(0, particle.weight);
    weightSum += weight;
    meanX += particle.x * weight;
    meanY += particle.y * weight;
    sinYaw += Math.sin(particle.yaw) * weight;
    cosYaw += Math.cos(particle.yaw) * weight;
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
  return {
    xx,
    yy,
    xy,
    spread: Math.sqrt(Math.max(0, xx + yy)),
    meanPose: { x: meanX, y: meanY, yaw: Math.atan2(sinYaw, cosYaw) },
  };
}

function deriveConvergence(covariance) {
  if (!covariance) return null;
  return Math.max(0, Math.min(1, 1 - covariance.spread / 2.5));
}

function deriveUpdateRateHz(snapshot) {
  const renderAndScoreMs = snapshot?.metrics?.render_and_score_milliseconds;
  if (renderAndScoreMs === null || renderAndScoreMs === undefined || renderAndScoreMs <= 0) return null;
  return 1000 / renderAndScoreMs;
}

function appendCapped(history, point, maxLength) {
  const next = history.concat(point);
  return next.length > maxLength ? next.slice(next.length - maxLength) : next;
}

function poseToWorldText(pose) {
  if (!pose) return ['x: —', 'y: —', 'θ: —'];
  return [
    `x: ${formatNumber(pose.x, 3)} m`,
    `y: ${formatNumber(pose.y, 3)} m`,
    `θ: ${formatNumber((pose.yaw * 180) / Math.PI, 1)}°`,
  ];
}

function mergeLiveFilterConfig(baseConfig, filterState) {
  if (!baseConfig) return null;
  if (!filterState) return baseConfig;
  return {
    ...baseConfig,
    particle_count: filterState.particle_count ?? baseConfig.particle_count,
    resample_threshold_ratio: filterState.resample_threshold_ratio ?? baseConfig.resample_threshold_ratio,
    measurement: {
      ...(baseConfig.measurement || {}),
      ...(filterState.measurement || {}),
    },
    motion_noise: {
      ...(baseConfig.motion_noise || {}),
      ...(filterState.motion_noise || {}),
    },
    runtime: {
      ...(baseConfig.runtime || {}),
      ...(filterState.runtime || {}),
    },
    initialization: {
      ...(baseConfig.initialization || {}),
      ...(filterState.initialization || {}),
    },
  };
}

function App() {
  const [snapshot, setSnapshot] = React.useState(null);
  const [connectionState, setConnectionState] = React.useState('Connecting...');
  const [mapMetadata, setMapMetadata] = React.useState(null);
  const [mapImage, setMapImage] = React.useState(null);
  const [layers, setLayers] = React.useState(DEFAULT_LAYERS);
  const [graphMode, setGraphMode] = React.useState('magnitude');
  const [resetDefaults, setResetDefaults] = React.useState({ yaw: 0.0, sigma_x: 0.5, sigma_y: 0.5, sigma_yaw: 0.5 });
  const [filterConfig, setFilterConfig] = React.useState(null);
  const [priorPreset, setPriorPreset] = React.useState('medium');
  const [resetStatus, setResetStatus] = React.useState('Left-drag on the map to place a pending prior.');
  const [estimatedPath, setEstimatedPath] = React.useState([]);
  const [groundTruthPath, setGroundTruthPath] = React.useState([]);
  const [errorHistory, setErrorHistory] = React.useState([]);
  const [rightTab, setRightTab] = React.useState('views');
  const [pendingPrior, setPendingPrior] = React.useState(null);
  const lastUpdateRef = React.useRef(null);

  React.useEffect(() => {
    let cancelled = false;

    async function loadMap() {
      try {
        const response = await fetch('/api/map-metadata', { cache: 'no-store' });
        if (response.status === 204 || !response.ok) return;
        const metadata = await response.json();
        if (cancelled) return;
        setMapMetadata(metadata);
        const image = new Image();
        image.src = metadata.image_url;
        await image.decode();
        if (!cancelled) setMapImage(image);
      } catch (_) {
        return;
      }
    }

    async function loadResetDefaults() {
      try {
        const response = await fetch('/api/reset-defaults', { cache: 'no-store' });
        if (!response.ok) return;
        const defaults = await response.json();
        if (!cancelled) setResetDefaults(defaults);
      } catch (_) {
        return;
      }
    }

    async function loadFilterConfig() {
      try {
        const response = await fetch('/api/filter-config', { cache: 'no-store' });
        if (!response.ok) return;
        const config = await response.json();
        if (!cancelled) setFilterConfig(config);
      } catch (_) {
        return;
      }
    }

    loadMap();
    loadResetDefaults();
    loadFilterConfig();

    return () => {
      cancelled = true;
    };
  }, []);

  React.useEffect(() => {
    document.body.dataset.theme = 'dark';
  }, []);

  React.useEffect(() => {
    const interval = window.setInterval(async () => {
      try {
        const response = await fetch('/api/latest', { cache: 'no-store' });
        if (response.status === 204) {
          setConnectionState('Waiting for particle filter snapshots...');
          return;
        }
        if (!response.ok) {
          setConnectionState(`Viewer API error ${response.status}`);
          return;
        }
        const nextSnapshot = await response.json();
        if (lastUpdateRef.current === nextSnapshot.update_index) return;
        lastUpdateRef.current = nextSnapshot.update_index;
        setSnapshot(nextSnapshot);
        setFilterConfig((current) => mergeLiveFilterConfig(current, nextSnapshot.filter_state));
        setConnectionState(`Connected · update ${nextSnapshot.update_index}`);

        const nextError = computePoseError(nextSnapshot.estimated_pose, nextSnapshot.ground_truth_pose);
        const timestamp = nextSnapshot.received_at_unix_seconds || Date.now() / 1000;
        if (nextSnapshot.estimated_pose) {
          setEstimatedPath((history) => appendCapped(history, { x: nextSnapshot.estimated_pose.x, y: nextSnapshot.estimated_pose.y }, MAX_PATH_POINTS));
        }
        if (nextSnapshot.ground_truth_pose) {
          setGroundTruthPath((history) => appendCapped(history, { x: nextSnapshot.ground_truth_pose.x, y: nextSnapshot.ground_truth_pose.y }, MAX_PATH_POINTS));
        }
        if (nextError) {
          setErrorHistory((history) => appendCapped(history, {
            t: timestamp,
            err: nextError.distance,
            ex: nextError.dx,
            ey: nextError.dy,
            eth: nextError.dyaw,
          }, MAX_ERROR_POINTS));
        }
      } catch (error) {
        setConnectionState(`Connection failed: ${error}`);
      }
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, []);

  const particleCount = snapshot ? snapshot.particles.length : 0;
  const covariance = React.useMemo(() => computeParticleCovariance(snapshot?.particles || []), [snapshot]);
  const poseError = React.useMemo(() => computePoseError(snapshot?.estimated_pose, snapshot?.ground_truth_pose), [snapshot]);
  const updateRateHz = React.useMemo(() => deriveUpdateRateHz(snapshot), [snapshot]);
  const convergence = React.useMemo(() => deriveConvergence(covariance), [covariance]);
  const bestPose = React.useMemo(() => {
    if (!snapshot) return null;
    const bestParticlePose = snapshot.metrics?.best_particle_pose;
    if (bestParticlePose) return bestParticlePose;
    const bestParticle = snapshot.particles?.[snapshot.metrics.best_particle_index];
    return bestParticle ? { x: bestParticle.x, y: bestParticle.y, yaw: bestParticle.yaw } : snapshot.estimated_pose;
  }, [snapshot]);
  const bestRenderSrc = snapshot ? `data:image/png;base64,${snapshot.images.best_render_png_base64}` : null;
  const observationSrc = snapshot ? `data:image/jpeg;base64,${snapshot.images.observation_jpeg_base64}` : null;
  const spreadMeters = covariance ? covariance.spread : null;
  const isLive = snapshot ? ((Date.now() / 1000) - snapshot.received_at_unix_seconds) < 1.5 : false;
  const liveFilterConfig = React.useMemo(() => mergeLiveFilterConfig(filterConfig, snapshot?.filter_state), [filterConfig, snapshot]);
  const priorPresets = React.useMemo(() => buildPriorPresets(liveFilterConfig || filterConfig || { initial_pose_prior: resetDefaults }), [liveFilterConfig, filterConfig, resetDefaults]);
  const capabilities = liveFilterConfig?.capabilities || filterConfig?.capabilities || {};
  const runtimeControlsSupported = Boolean(capabilities.pause_resume || capabilities.single_step);
  const localizationMode = liveFilterConfig?.initialization?.mode || filterConfig?.initialization?.mode || 'local';

  function buildResetPayload(meanPose, sigmas) {
    return {
      prior: {
        mean: {
          x: meanPose.x,
          y: meanPose.y,
          yaw: meanPose.yaw,
        },
        sigma_x: sigmas.sigma_x,
        sigma_y: sigmas.sigma_y,
        sigma_yaw: sigmas.sigma_yaw,
      },
    };
  }

  async function submitResetPayload(payload, label) {
    const response = await fetch('/api/reset-particle-filter', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || `HTTP ${response.status}`);
    }
    setResetStatus(`${label} queued · ${new Date(result.command.issued_at_unix_seconds * 1000).toLocaleTimeString()}`);
  }

  async function submitControlCommand(command, label, optimisticUpdate) {
    if (optimisticUpdate) {
      setFilterConfig((current) => optimisticUpdate(current || liveFilterConfig || filterConfig));
    }
    const response = await fetch('/api/control-command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(command),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || `HTTP ${response.status}`);
    }
    setResetStatus(`${label} queued · ${new Date(result.command.issued_at_unix_seconds * 1000).toLocaleTimeString()}`);
  }

  function handleSetPrior(priorPose) {
    if (localizationMode === 'global') {
      setResetStatus('Switch to local mode to place a manual prior.');
      return;
    }
    setPendingPrior(priorPose);
    setResetStatus(`Pending prior · x=${formatNumber(priorPose.x, 3)} m · y=${formatNumber(priorPose.y, 3)} m · θ=${formatNumber((priorPose.theta * 180) / Math.PI, 1)}°`);
  }

  async function applyPendingPrior() {
    if (!pendingPrior) return;
    const sigmas = priorPresets[priorPreset];
    await submitResetPayload(
      buildResetPayload({ x: pendingPrior.x, y: pendingPrior.y, yaw: pendingPrior.theta }, sigmas),
      'Map prior',
    );
    setPendingPrior(null);
  }

  function cancelPendingPrior() {
    setPendingPrior(null);
    setResetStatus('Pending prior cleared.');
  }

  async function handleGlobalReset() {
    if (localizationMode === 'global') {
      await submitControlCommand(
        { type: 'global_reset_particle_filter' },
        'Global reset',
        (current) => ({
          ...current,
          initialization: { ...(current?.initialization || {}), mode: 'global' },
        }),
      );
      setPendingPrior(null);
      return;
    }

    const basePrior = liveFilterConfig?.initial_pose_prior || filterConfig?.initial_pose_prior;
    if (!basePrior) {
      setResetStatus('Local reset unavailable until filter config is loaded.');
      return;
    }
    const sigmas = priorPresets[priorPreset];
    await submitResetPayload(
      buildResetPayload(basePrior.mean, sigmas),
      'Local reset',
    );
    setPendingPrior(null);
  }

  async function handleLocalizationModeChange(mode) {
    if (mode === localizationMode) return;
    await submitControlCommand(
      { type: 'set_localization_mode', mode },
      `Mode ${mode}`,
      (current) => ({
        ...current,
        initialization: { ...(current?.initialization || {}), mode },
      }),
    );
    if (mode === 'global') {
      setPendingPrior(null);
      setResetStatus('Global mode selected. Use reset to reinitialize across map free space.');
    } else {
      setResetStatus('Local mode selected. You can place a prior by dragging on the map.');
    }
  }

  async function handleParticleCountChange(value) {
    await submitControlCommand(
      { type: 'set_particle_filter_parameters', particle_count: value },
      `Particle count ${value}`,
      (current) => ({ ...current, particle_count: value }),
    );
  }

  async function handleResampleThresholdChange(value) {
    await submitControlCommand(
      { type: 'set_particle_filter_parameters', resample_threshold_ratio: value },
      `Resample threshold ${value.toFixed(2)}`,
      (current) => ({ ...current, resample_threshold_ratio: value }),
    );
  }

  async function handleTemperatureChange(value) {
    await submitControlCommand(
      { type: 'set_particle_filter_parameters', temperature: value },
      `Temperature ${value.toFixed(3)}`,
      (current) => ({
        ...current,
        measurement: { ...(current?.measurement || {}), temperature: value },
      }),
    );
  }

  async function handleMotionNoiseChange(key, value) {
    const currentNoise = (liveFilterConfig?.motion_noise || filterConfig?.motion_noise || {});
    const nextNoise = {
      x_meters: currentNoise.x_meters ?? 0.02,
      y_meters: currentNoise.y_meters ?? 0.02,
      yaw_radians: currentNoise.yaw_radians ?? 0.017453292519943295,
      [key]: value,
    };
    await submitControlCommand(
      { type: 'set_particle_filter_parameters', motion_noise: nextNoise },
      `Motion noise ${key}=${value.toFixed(3)}`,
      (current) => ({
        ...current,
        motion_noise: { ...(current?.motion_noise || {}), ...nextNoise },
      }),
    );
  }

  async function handleTogglePause() {
    const paused = Boolean(liveFilterConfig?.runtime?.paused);
    await submitControlCommand(
      { type: paused ? 'resume_particle_filter' : 'pause_particle_filter' },
      paused ? 'Resume' : 'Pause',
      (current) => ({
        ...current,
        runtime: { ...(current?.runtime || {}), paused: !paused },
      }),
    );
  }

  async function handleStepOnce() {
    await submitControlCommand(
      { type: 'step_particle_filter' },
      'Step once',
      (current) => ({
        ...current,
        runtime: { ...(current?.runtime || {}), paused: true },
      }),
    );
  }

  function toggleLayer(key) {
    setLayers((current) => ({ ...current, [key]: !current[key] }));
  }

  const mapInfoText = mapMetadata
    ? {
        resolution: `${formatNumber(mapMetadata.resolution, 3)} m/px`,
        origin: `(${formatNumber(mapMetadata.origin[0], 2)}, ${formatNumber(mapMetadata.origin[1], 2)}, ${formatNumber((mapMetadata.origin[2] * 180) / Math.PI, 1)}°)`,
        cells: `${mapMetadata.width}×${mapMetadata.height} cells`,
      }
    : null;

  const truthPoseLines = poseToWorldText(snapshot?.ground_truth_pose);
  const estimatedPoseLines = poseToWorldText(snapshot?.estimated_pose);
  const estimateExtra = covariance ? [
    `σx: ${formatNumber(Math.sqrt(Math.max(0, covariance.xx)), 3)} m`,
    `σy: ${formatNumber(Math.sqrt(Math.max(0, covariance.yy)), 3)} m`,
  ] : [];

  const historyTail = errorHistory.slice(-150);
  const meanError = historyTail.length ? historyTail.reduce((sum, item) => sum + item.err, 0) / historyTail.length : null;
  const minError = historyTail.length ? Math.min(...historyTail.map((item) => item.err)) : null;
  const maxError = historyTail.length ? Math.max(...historyTail.map((item) => item.err)) : null;

  return (
    <>
      <div className="topbar">
        <div className="brand">
          <div className="brand-mark"></div>
          <div>
            <div className="title">3DGS-PF</div>
            <div className="sub">localization · console</div>
          </div>
        </div>
        <div className="stats">
          <div className="stat"><div className="k">Status</div><div className={`v ${isLive ? 'ok' : 'warn'}`}>{isLive ? 'LIVE' : 'STALE'}</div></div>
          <div className="stat"><div className="k">Update</div><div className="v">{snapshot ? snapshot.update_index : '—'}</div></div>
          <div className="stat"><div className="k">Particles</div><div className="v">{particleCount || '—'}</div></div>
          <div className="stat"><div className="k">ESS</div><div className={`v ${snapshot && snapshot.metrics.effective_particle_count > particleCount * 0.5 ? 'ok' : 'warn'}`}>{snapshot ? formatNumber(snapshot.metrics.effective_particle_count, 0) : '—'}</div></div>
          <div className="stat"><div className="k">Error</div><div className={`v ${poseError && poseError.distance < 0.15 ? 'ok' : poseError && poseError.distance < 0.5 ? 'warn' : 'err'}`}>{poseError ? formatNumber(poseError.distance, 3) : '—'}{poseError ? <span className="unit">m</span> : null}</div></div>
          <div className="stat"><div className="k">Spread</div><div className={`v ${spreadMeters !== null && spreadMeters < 0.35 ? 'ok' : 'warn'}`}>{spreadMeters !== null ? formatNumber(spreadMeters, 3) : '—'}{spreadMeters !== null ? <span className="unit">m</span> : null}</div></div>
          <div className="stat"><div className="k">Conv</div><div className={`v ${convergence !== null && convergence > 0.7 ? 'ok' : 'warn'}`}>{convergence !== null ? `${Math.round(convergence * 100)}` : '—'}{convergence !== null ? <span className="unit">%</span> : null}</div></div>
          <div className="stat"><div className="k">Rate</div><div className="v ok">{updateRateHz !== null ? formatNumber(updateRateHz, 1) : '—'}{updateRateHz !== null ? <span className="unit">Hz</span> : null}</div></div>
        </div>
        <div className="right">
          <div className={`led ${isLive ? '' : 'warn'}`}></div>
          <span>{connectionState}</span>
        </div>
      </div>

      <div className="root">
        <div className="rail">
          <div className="rail-section">
            <h3>Map Source</h3>
            <div className="pgm-meta">
              <div><b>{mapMetadata ? 'map.pgm' : 'waiting for map'}</b></div>
              <div>{mapInfoText ? mapInfoText.resolution : '—'}</div>
              <div>origin {mapInfoText ? mapInfoText.origin : '—'}</div>
              <div>{mapInfoText ? mapInfoText.cells : '—'}</div>
            </div>
          </div>

          <div className="rail-section">
            <h3>Layers</h3>
            {[
              { key: 'grid', name: 'Occupancy grid', swatch: '#9aa3b2' },
              { key: 'particles', name: 'Particles', swatch: 'oklch(0.78 0.14 200)' },
              { key: 'heatmap', name: 'Density heatmap', swatch: 'oklch(0.78 0.14 75)' },
              { key: 'robots', name: 'Robot markers', swatch: 'oklch(0.78 0.16 145)' },
              { key: 'covariance', name: '2σ covariance', swatch: 'oklch(0.78 0.14 75)' },
              { key: 'trail', name: 'Path history', swatch: '#7d8493' },
            ].map((layer) => (
              <div key={layer.key} className="layer-toggle" onClick={() => toggleLayer(layer.key)}>
                <span className="name"><span className="swatch" style={{ background: layer.swatch }}></span>{layer.name}</span>
                <span className="switch" data-on={layers[layer.key] ? '1' : '0'}><i /></span>
              </div>
            ))}
          </div>

          <div className="rail-section">
            <h3>Legend</h3>
            <div className="legend-row"><span className="dot gt"></span>Ground truth pose</div>
            <div className="legend-row"><span className="dot" style={{ background: 'rgba(255,79,216,1)' }}></span>AMCL pose</div>
            <div className="legend-row"><span className="dot est"></span>PF estimate</div>
            <div className="legend-row"><span className="dot" style={{ background: 'oklch(0.78 0.14 200)' }}></span>Particle (high w)</div>
            <div className="legend-row"><span className="dot" style={{ background: 'oklch(0.5 0.08 230)' }}></span>Particle (low w)</div>
          </div>

          <div className="rail-section">
            <h3>Pose · Truth</h3>
            <div className="pgm-meta" style={{ fontFamily: 'var(--mono)' }}>{truthPoseLines.map((line) => <div key={line}>{line}</div>)}</div>
          </div>

          <div className="rail-section">
            <h3>Pose · Estimate</h3>
            <div className="pgm-meta" style={{ fontFamily: 'var(--mono)' }}>{estimatedPoseLines.concat(estimateExtra).map((line) => <div key={line}>{line}</div>)}</div>
          </div>
        </div>

        <div className="map-col">
          <MapView
            snapshot={snapshot}
            mapMetadata={mapMetadata}
            mapImage={mapImage}
            layers={layers}
            particleStyle="dot"
            estimatedPath={estimatedPath}
            groundTruthPath={groundTruthPath}
            onSetPrior={handleSetPrior}
          />
        </div>

        <div className="right-col">
          <div className="col-tabs">
            <button className={`col-tab ${rightTab === 'views' ? 'on' : ''}`} onClick={() => setRightTab('views')}>Views <span className="badge">cam · err</span></button>
            <button className={`col-tab ${rightTab === 'filter' ? 'on' : ''}`} onClick={() => setRightTab('filter')}>Filter <span className="badge">params</span></button>
          </div>

          {rightTab === 'views' && (
            <div className="right-pane">
              <div className="right-section">
                <div className="head"><span>Camera Views</span><span className="pill">live</span></div>
                <CameraImageCard
                  imageSrc={observationSrc}
                  label="ROBOT.cam0"
                  sublabel="live observation · /oakd/rgb/preview/image_raw"
                  accent="oklch(0.78 0.16 145)"
                  pose={snapshot?.ground_truth_pose}
                  resolution={observationSrc ? '320×240' : '—'}
                />
                <CameraImageCard
                  imageSrc={bestRenderSrc}
                  label="3DGS.render"
                  sublabel={snapshot ? `best particle · score=${formatNumber(snapshot.metrics.best_score, 6)}` : 'best particle render'}
                  accent="oklch(0.78 0.14 200)"
                  pose={bestPose}
                  resolution={bestRenderSrc ? '320×240' : '—'}
                />
              </div>
              <div className="right-section graph-card">
                <div className="head">
                  <span>Localization Error</span>
                  <div className="graph-tabs">
                    <button className={`gtab ${graphMode === 'magnitude' ? 'on' : ''}`} onClick={() => setGraphMode('magnitude')}>‖Δp‖</button>
                    <button className={`gtab ${graphMode === 'components' ? 'on' : ''}`} onClick={() => setGraphMode('components')}>Δx Δy Δθ</button>
                  </div>
                  <span className="pill">last 30s</span>
                </div>
                <ErrorGraph history={errorHistory} theme="dark" mode={graphMode} />
                <div className="graph-meta">
                  <div className="item"><span className="k">Now</span><span className="v warn">{formatNumber(poseError?.distance, 3)} m</span></div>
                  <div className="item"><span className="k">Mean</span><span className="v">{formatNumber(meanError, 3)} m</span></div>
                  <div className="item"><span className="k">Min</span><span className="v ok">{formatNumber(minError, 3)} m</span></div>
                  <div className="item"><span className="k">Max</span><span className="v">{formatNumber(maxError, 3)} m</span></div>
                </div>
              </div>
            </div>
          )}

          {rightTab === 'filter' && (
            <div className="right-pane scroll">
              <FilterControls
                filterConfig={liveFilterConfig || filterConfig}
                snapshot={snapshot}
                priorPreset={priorPreset}
                setPriorPreset={setPriorPreset}
                localizationMode={localizationMode}
                onLocalizationModeChange={handleLocalizationModeChange}
                onGlobalReset={handleGlobalReset}
                onParticleCountChange={handleParticleCountChange}
                onResampleThresholdChange={handleResampleThresholdChange}
                onTemperatureChange={handleTemperatureChange}
                onMotionNoiseChange={handleMotionNoiseChange}
                onTogglePause={handleTogglePause}
                onStepOnce={handleStepOnce}
              />
            </div>
          )}
        </div>
      </div>

      <div className="controls">
        {!pendingPrior && localizationMode === 'local' && <span className="ctrl-meta">tip: <b>left-drag</b> on map to set new prior · preset <b>{priorPreset}</b></span>}
        {!pendingPrior && localizationMode === 'global' && <span className="ctrl-meta">mode <b>global</b> · reset reinitializes over the full free-space map</span>}
        <div className="ctrl-divider"></div>
        <span className="ctrl-meta">mode <b>{localizationMode}</b> · renderer <b>{snapshot ? formatNumber(snapshot.metrics.render_and_score_milliseconds, 1) : '—'}</b> ms · best idx <b>{snapshot ? snapshot.metrics.best_particle_index : '—'}</b> · resampled <b>{snapshot ? (snapshot.metrics.resampled ? 'yes' : 'no') : '—'}</b></span>
        <div className="ctrl-spacer"></div>
        {pendingPrior ? (
          <div className="prior-banner">
            <span className="dot"></span>
            NEW PRIOR @ ({formatNumber(pendingPrior.x, 2)}, {formatNumber(pendingPrior.y, 2)}) θ={formatNumber((pendingPrior.theta * 180) / Math.PI, 0)}°
            <button className="ctrl-btn primary" style={{ height: '22px', padding: '0 10px', marginLeft: '4px' }} onClick={applyPendingPrior}>APPLY &amp; RESTART</button>
            <button className="ctrl-btn" style={{ height: '22px', padding: '0 10px' }} onClick={cancelPendingPrior}>CANCEL</button>
          </div>
        ) : (
          <div className="status-chip">{resetStatus}</div>
        )}
      </div>
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
