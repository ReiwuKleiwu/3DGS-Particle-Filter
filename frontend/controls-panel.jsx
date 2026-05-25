function buildPriorPresets(filterConfig) {
  const baseSigmaX = filterConfig?.initial_pose_prior?.sigma_x ?? 0.5;
  const baseSigmaY = filterConfig?.initial_pose_prior?.sigma_y ?? 0.5;
  const baseSigmaYaw = filterConfig?.initial_pose_prior?.sigma_yaw ?? 0.5;
  return {
    tight: {
      label: 'tight',
      sigma_x: Math.max(0.05, baseSigmaX * 0.3),
      sigma_y: Math.max(0.05, baseSigmaY * 0.3),
      sigma_yaw: Math.max(0.05, baseSigmaYaw * 0.3),
    },
    medium: {
      label: 'medium',
      sigma_x: baseSigmaX,
      sigma_y: baseSigmaY,
      sigma_yaw: baseSigmaYaw,
    },
    wide: {
      label: 'wide',
      sigma_x: Math.max(baseSigmaX, baseSigmaX * 2.2),
      sigma_y: Math.max(baseSigmaY, baseSigmaY * 2.2),
      sigma_yaw: Math.max(baseSigmaYaw, baseSigmaYaw * 2.2),
    },
  };
}

function FilterControls({
  filterConfig,
  snapshot,
  priorPreset,
  setPriorPreset,
  localizationMode,
  onLocalizationModeChange,
  onGlobalReset,
  onParticleCountChange,
  onResampleThresholdChange,
  onTemperatureChange,
  onMotionNoiseChange,
  onTogglePause,
  onStepOnce,
}) {
  const presets = React.useMemo(() => buildPriorPresets(filterConfig), [filterConfig]);
  const capabilities = filterConfig?.capabilities || {};
  const particleCount = filterConfig?.particle_count ?? snapshot?.particles?.length ?? 0;
  const resampleRatio = filterConfig?.resample_threshold_ratio ?? 0.5;
  const ess = snapshot?.metrics?.effective_particle_count ?? null;
  const essRatio = ess !== null && particleCount > 0 ? ess / particleCount : null;
  const temperature = filterConfig?.measurement?.temperature ?? null;
  const motionNoise = filterConfig?.motion_noise ?? null;
  const paused = Boolean(filterConfig?.runtime?.paused);
  const isGlobalMode = localizationMode === 'global';

  const [draftParticleCount, setDraftParticleCount] = React.useState(particleCount || 256);
  const [draftResampleRatio, setDraftResampleRatio] = React.useState(resampleRatio);
  const [draftTemperature, setDraftTemperature] = React.useState(temperature ?? 0.02);
  const [draftNoiseX, setDraftNoiseX] = React.useState(motionNoise?.x_meters ?? 0.02);
  const [draftNoiseY, setDraftNoiseY] = React.useState(motionNoise?.y_meters ?? 0.02);
  const [draftNoiseYaw, setDraftNoiseYaw] = React.useState(motionNoise?.yaw_radians ?? 0.017453292519943295);

  React.useEffect(() => setDraftParticleCount(particleCount || 256), [particleCount]);
  React.useEffect(() => setDraftResampleRatio(resampleRatio), [resampleRatio]);
  React.useEffect(() => setDraftTemperature(temperature ?? 0.02), [temperature]);
  React.useEffect(() => setDraftNoiseX(motionNoise?.x_meters ?? 0.02), [motionNoise?.x_meters]);
  React.useEffect(() => setDraftNoiseY(motionNoise?.y_meters ?? 0.02), [motionNoise?.y_meters]);
  React.useEffect(() => setDraftNoiseYaw(motionNoise?.yaw_radians ?? 0.017453292519943295), [motionNoise?.yaw_radians]);

  return (
    <div className="fc-wrap">
      <div className="fc-section">
        <div className="fc-h">LOCALIZATION MODE</div>
        <div className="fc-row">
          <div className="fc-lbl">Reset / relocalize strategy</div>
          <div className="fc-seg">
            {['local', 'global'].map((key) => (
              <button
                key={key}
                className={localizationMode === key ? 'on' : ''}
                onClick={() => onLocalizationModeChange(key)}
                disabled={!capabilities.localization_mode}
              >
                {key}
              </button>
            ))}
          </div>
        </div>
        <div className="fc-hint">
          {isGlobalMode
            ? 'Global mode reinitializes particles across free map space and ignores map-drawn priors.'
            : 'Local mode uses the configured Gaussian prior or a map-drawn prior.'}
        </div>
      </div>

      <div className="fc-section">
        <div className="fc-h">PRIOR</div>
        <div className="fc-row">
          <div className="fc-lbl">Spread preset</div>
          <div className="fc-seg">
            {['tight', 'medium', 'wide'].map((key) => (
              <button key={key} className={priorPreset === key ? 'on' : ''} onClick={() => setPriorPreset(key)} disabled={isGlobalMode}>{key}</button>
            ))}
          </div>
        </div>
        <div className="fc-row tworow">
          <div className="fc-mini">
            <div className="fc-mlbl">σx</div>
            <div className="fc-mval">{presets[priorPreset].sigma_x.toFixed(2)}<span> m</span></div>
          </div>
          <div className="fc-mini">
            <div className="fc-mlbl">σy</div>
            <div className="fc-mval">{presets[priorPreset].sigma_y.toFixed(2)}<span> m</span></div>
          </div>
          <div className="fc-mini">
            <div className="fc-mlbl">σθ</div>
            <div className="fc-mval">{(presets[priorPreset].sigma_yaw * 180 / Math.PI).toFixed(0)}<span> °</span></div>
          </div>
        </div>
        <div className="fc-hint">{isGlobalMode ? 'Switch back to local mode to place and apply a manual prior from the map.' : 'Left-drag the map to place a pending prior. Apply uses the spread preset above.'}</div>
      </div>

      <div className="fc-section">
        <div className="fc-h">SAMPLES</div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Particle count</span><span className="v">{draftParticleCount || '—'}</span></div>
          <input
            className="fc-slider"
            type="range"
            min={16}
            max={2048}
            step={16}
            value={draftParticleCount || 256}
            onChange={(event) => setDraftParticleCount(Number(event.target.value))}
            onMouseUp={() => draftParticleCount !== particleCount && onParticleCountChange(draftParticleCount)}
            onTouchEnd={() => draftParticleCount !== particleCount && onParticleCountChange(draftParticleCount)}
            disabled={!capabilities.particle_count}
          />
          <div className="fc-rng"><span>64</span><span>2048</span></div>
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Resample @ ESS &lt;</span><span className="v">{(draftResampleRatio * 100).toFixed(0)}% · {draftParticleCount ? (draftResampleRatio * draftParticleCount).toFixed(0) : '—'}</span></div>
          <input
            className="fc-slider"
            type="range"
            min={0.1}
            max={0.95}
            step={0.05}
            value={draftResampleRatio}
            onChange={(event) => setDraftResampleRatio(Number(event.target.value))}
            onMouseUp={() => draftResampleRatio !== resampleRatio && onResampleThresholdChange(draftResampleRatio)}
            onTouchEnd={() => draftResampleRatio !== resampleRatio && onResampleThresholdChange(draftResampleRatio)}
            disabled={!capabilities.resample_threshold}
          />
        </div>
        <div className="fc-gauge">
          <div className="fc-gauge-track">
            <div className="fc-gauge-thr" style={{ left: `${draftResampleRatio * 100}%` }} />
            <div className={`fc-gauge-fill ${essRatio !== null && essRatio > draftResampleRatio ? 'ok' : 'warn'}`} style={{ width: `${Math.min(1, Math.max(0, essRatio ?? 0)) * 100}%` }} />
          </div>
          <div className="fc-gauge-meta">
            <span>ESS</span>
            <span className="mono">{ess !== null ? ess.toFixed(0) : '—'} / {draftParticleCount || '—'}</span>
            <span className={`chip ${essRatio !== null && essRatio > draftResampleRatio ? 'ok' : 'warn'}`}>
              {essRatio !== null && essRatio > draftResampleRatio ? 'healthy' : 'near resample'}
            </span>
          </div>
        </div>
      </div>

      <div className="fc-section">
        <div className="fc-h">MEASUREMENT</div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Temperature</span><span className="v">{draftTemperature.toFixed(3)}</span></div>
          <input
            className="fc-slider"
            type="range"
            min={0.005}
            max={0.06}
            step={0.0025}
            value={draftTemperature}
            onChange={(event) => setDraftTemperature(Number(event.target.value))}
            onMouseUp={() => draftTemperature !== (temperature ?? 0.02) && onTemperatureChange(draftTemperature)}
            onTouchEnd={() => draftTemperature !== (temperature ?? 0.02) && onTemperatureChange(draftTemperature)}
            disabled={!capabilities.temperature}
          />
          <div className="fc-rng"><span>winner-take-all</span><span>flat</span></div>
        </div>
      </div>

      <div className="fc-section">
        <div className="fc-h">MOTION NOISE</div>
        <div className="fc-row">
          <div className="fc-lbl"><span>σx</span><span className="v">{draftNoiseX.toFixed(3)} m</span></div>
          <input
            className="fc-slider"
            type="range"
            min={0.0}
            max={0.2}
            step={0.005}
            value={draftNoiseX}
            onChange={(event) => setDraftNoiseX(Number(event.target.value))}
            onMouseUp={() => draftNoiseX !== (motionNoise?.x_meters ?? 0.02) && onMotionNoiseChange('x_meters', draftNoiseX)}
            onTouchEnd={() => draftNoiseX !== (motionNoise?.x_meters ?? 0.02) && onMotionNoiseChange('x_meters', draftNoiseX)}
            disabled={!capabilities.motion_noise}
          />
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>σy</span><span className="v">{draftNoiseY.toFixed(3)} m</span></div>
          <input
            className="fc-slider"
            type="range"
            min={0.0}
            max={0.2}
            step={0.005}
            value={draftNoiseY}
            onChange={(event) => setDraftNoiseY(Number(event.target.value))}
            onMouseUp={() => draftNoiseY !== (motionNoise?.y_meters ?? 0.02) && onMotionNoiseChange('y_meters', draftNoiseY)}
            onTouchEnd={() => draftNoiseY !== (motionNoise?.y_meters ?? 0.02) && onMotionNoiseChange('y_meters', draftNoiseY)}
            disabled={!capabilities.motion_noise}
          />
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>σθ</span><span className="v">{(draftNoiseYaw * 180 / Math.PI).toFixed(1)} °</span></div>
          <input
            className="fc-slider"
            type="range"
            min={0.0}
            max={0.35}
            step={0.005}
            value={draftNoiseYaw}
            onChange={(event) => setDraftNoiseYaw(Number(event.target.value))}
            onMouseUp={() => draftNoiseYaw !== (motionNoise?.yaw_radians ?? 0.017453292519943295) && onMotionNoiseChange('yaw_radians', draftNoiseYaw)}
            onTouchEnd={() => draftNoiseYaw !== (motionNoise?.yaw_radians ?? 0.017453292519943295) && onMotionNoiseChange('yaw_radians', draftNoiseYaw)}
            disabled={!capabilities.motion_noise}
          />
        </div>
      </div>

      <div className="fc-section">
        <div className="fc-h">RUN CONTROL</div>
        <div className="fc-actions">
          <button className="fc-btn" onClick={onTogglePause} disabled={!capabilities.pause_resume}>{paused ? '▶ RESUME' : '❚❚ PAUSE'}</button>
          <button className="fc-btn" onClick={onStepOnce} disabled={!capabilities.single_step || !paused}>⤳ STEP ONCE</button>
          <button className="fc-btn danger" onClick={onGlobalReset} disabled={capabilities.global_reset === false}>{isGlobalMode ? '↻ GLOBAL RESET' : '↻ LOCAL RESET'}</button>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { FilterControls, buildPriorPresets });
