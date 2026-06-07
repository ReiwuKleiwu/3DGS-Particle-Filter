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

function ParticleFilterModules({
  filterConfig,
  snapshot,
  onRougheningChange,
  onRecoveryChange,
  onAdaptiveParticleCountChange,
}) {
  const capabilities = filterConfig?.capabilities || {};
  const pf = filterConfig?.particle_filter || {};
  const recovery = filterConfig?.recovery || {};
  const adaptive = filterConfig?.adaptive_particle_count || {};
  const metricName = filterConfig?.measurement?.metric_name || 'lpips';
  const profile = recovery.absolute_score_profiles?.[metricName] || recovery.absolute_score_profiles?.default || {};
  const roughened = snapshot?.metrics?.roughening_particle_count ?? 0;
  const injected = snapshot?.metrics?.random_particle_count ?? 0;
  const recoveryRatio = snapshot?.metrics?.random_particle_ratio ?? 0;

  const [rougheningEnabled, setRougheningEnabled] = React.useState(Boolean(pf.roughening_enabled));
  const [rougheningMode, setRougheningMode] = React.useState(pf.roughening_mode || 'resample_only');
  const [rougheningRatio, setRougheningRatio] = React.useState(pf.roughening_ratio ?? 0);
  const [rougheningSigmaX, setRougheningSigmaX] = React.useState(pf.roughening_sigma_x ?? 0.05);
  const [rougheningSigmaY, setRougheningSigmaY] = React.useState(pf.roughening_sigma_y ?? 0.05);
  const [rougheningSigmaYaw, setRougheningSigmaYaw] = React.useState(pf.roughening_sigma_yaw ?? 0.05);
  const [recoveryEnabled, setRecoveryEnabled] = React.useState(Boolean(recovery.enabled));
  const [recoveryStrategy, setRecoveryStrategy] = React.useState(recovery.strategy || 'absolute_score');
  const [recoveryMaxRatio, setRecoveryMaxRatio] = React.useState(recovery.random_particle_max_ratio ?? 0.3);
  const [bestThreshold, setBestThreshold] = React.useState(profile.best_score_threshold ?? 0.45);
  const [medianThreshold, setMedianThreshold] = React.useState(profile.median_score_threshold ?? 0.48);
  const [absoluteRatio, setAbsoluteRatio] = React.useState(profile.random_particle_ratio ?? 0.3);
  const [badUpdates, setBadUpdates] = React.useState(profile.consecutive_bad_updates ?? 3);
  const [adaptiveEnabled, setAdaptiveEnabled] = React.useState(Boolean(adaptive.enabled));
  const [adaptiveMinCount, setAdaptiveMinCount] = React.useState(adaptive.min_particle_count ?? 128);
  const [adaptiveMediumCount, setAdaptiveMediumCount] = React.useState(adaptive.medium_particle_count ?? 256);
  const [adaptiveMaxCount, setAdaptiveMaxCount] = React.useState(adaptive.max_particle_count ?? filterConfig?.particle_count ?? 256);
  const [adaptiveStableUpdates, setAdaptiveStableUpdates] = React.useState(adaptive.stable_required_updates ?? 8);
  const [adaptiveUnstableUpdates, setAdaptiveUnstableUpdates] = React.useState(adaptive.unstable_required_updates ?? 2);

  React.useEffect(() => setRougheningEnabled(Boolean(pf.roughening_enabled)), [pf.roughening_enabled]);
  React.useEffect(() => setRougheningMode(pf.roughening_mode || 'resample_only'), [pf.roughening_mode]);
  React.useEffect(() => setRougheningRatio(pf.roughening_ratio ?? 0), [pf.roughening_ratio]);
  React.useEffect(() => setRougheningSigmaX(pf.roughening_sigma_x ?? 0.05), [pf.roughening_sigma_x]);
  React.useEffect(() => setRougheningSigmaY(pf.roughening_sigma_y ?? 0.05), [pf.roughening_sigma_y]);
  React.useEffect(() => setRougheningSigmaYaw(pf.roughening_sigma_yaw ?? 0.05), [pf.roughening_sigma_yaw]);
  React.useEffect(() => setRecoveryEnabled(Boolean(recovery.enabled)), [recovery.enabled]);
  React.useEffect(() => setRecoveryStrategy(recovery.strategy || 'absolute_score'), [recovery.strategy]);
  React.useEffect(() => setRecoveryMaxRatio(recovery.random_particle_max_ratio ?? 0.3), [recovery.random_particle_max_ratio]);
  React.useEffect(() => setBestThreshold(profile.best_score_threshold ?? 0.45), [profile.best_score_threshold]);
  React.useEffect(() => setMedianThreshold(profile.median_score_threshold ?? 0.48), [profile.median_score_threshold]);
  React.useEffect(() => setAbsoluteRatio(profile.random_particle_ratio ?? 0.3), [profile.random_particle_ratio]);
  React.useEffect(() => setBadUpdates(profile.consecutive_bad_updates ?? 3), [profile.consecutive_bad_updates]);
  React.useEffect(() => setAdaptiveEnabled(Boolean(adaptive.enabled)), [adaptive.enabled]);
  React.useEffect(() => setAdaptiveMinCount(adaptive.min_particle_count ?? 128), [adaptive.min_particle_count]);
  React.useEffect(() => setAdaptiveMediumCount(adaptive.medium_particle_count ?? 256), [adaptive.medium_particle_count]);
  React.useEffect(() => setAdaptiveMaxCount(adaptive.max_particle_count ?? filterConfig?.particle_count ?? 256), [adaptive.max_particle_count, filterConfig?.particle_count]);
  React.useEffect(() => setAdaptiveStableUpdates(adaptive.stable_required_updates ?? 8), [adaptive.stable_required_updates]);
  React.useEffect(() => setAdaptiveUnstableUpdates(adaptive.unstable_required_updates ?? 2), [adaptive.unstable_required_updates]);

  function submitRoughening(patch) {
    onRougheningChange({
      enabled: rougheningEnabled,
      mode: rougheningMode,
      ratio: rougheningRatio,
      sigma_x: rougheningSigmaX,
      sigma_y: rougheningSigmaY,
      sigma_yaw: rougheningSigmaYaw,
      ...patch,
    });
  }

  function submitRecovery(patch) {
    const profiles = {
      ...(recovery.absolute_score_profiles || {}),
      [metricName]: {
        best_score_threshold: bestThreshold,
        median_score_threshold: medianThreshold,
        random_particle_ratio: absoluteRatio,
        consecutive_bad_updates: badUpdates,
        ...(patch.profile || {}),
      },
    };
    const payload = {
      enabled: recoveryEnabled,
      strategy: recoveryStrategy,
      random_particle_max_ratio: recoveryMaxRatio,
      absolute_score_profiles: profiles,
      ...patch,
    };
    delete payload.profile;
    onRecoveryChange(payload);
  }

  function submitAdaptiveParticleCount(patch) {
    onAdaptiveParticleCountChange({
      enabled: adaptiveEnabled,
      min_particle_count: adaptiveMinCount,
      medium_particle_count: adaptiveMediumCount,
      max_particle_count: adaptiveMaxCount,
      stable_required_updates: adaptiveStableUpdates,
      unstable_required_updates: adaptiveUnstableUpdates,
      ...patch,
    });
  }

  return (
    <div className="fc-wrap">
      <div className="fc-section">
        <div className="fc-h">ADAPTIVE PARTICLES</div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Module</span><span className="v">{adaptiveEnabled ? 'enabled' : 'disabled'}</span></div>
          <div className="fc-seg">
            <button
              className={adaptiveEnabled ? 'on' : ''}
              onClick={() => { setAdaptiveEnabled(true); submitAdaptiveParticleCount({ enabled: true }); }}
              disabled={!capabilities.adaptive_particle_count}
            >on</button>
            <button
              className={!adaptiveEnabled ? 'on' : ''}
              onClick={() => { setAdaptiveEnabled(false); submitAdaptiveParticleCount({ enabled: false }); }}
              disabled={!capabilities.adaptive_particle_count}
            >off</button>
          </div>
        </div>
        <div className="fc-row tworow">
          <div className="fc-mini">
            <div className="fc-mlbl">current</div>
            <div className="fc-mval">{snapshot?.particles?.length ?? '—'}</div>
          </div>
          <div className="fc-mini">
            <div className="fc-mlbl">target</div>
            <div className="fc-mval">{adaptive.target_particle_count ?? '—'}</div>
          </div>
          <div className="fc-mini">
            <div className="fc-mlbl">max</div>
            <div className="fc-mval">{adaptive.max_particle_count ?? adaptiveMaxCount}</div>
          </div>
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Min / medium</span><span className="v">{adaptiveMinCount} / {adaptiveMediumCount}</span></div>
          <input className="fc-slider" type="range" min={16} max={1024} step={16} value={adaptiveMinCount}
            onChange={(event) => setAdaptiveMinCount(Number(event.target.value))}
            onMouseUp={(event) => submitAdaptiveParticleCount({ min_particle_count: Number(event.currentTarget.value) })}
            onTouchEnd={(event) => submitAdaptiveParticleCount({ min_particle_count: Number(event.currentTarget.value) })}
            disabled={!capabilities.adaptive_particle_count || !adaptiveEnabled}
          />
          <input className="fc-slider" type="range" min={16} max={2048} step={16} value={adaptiveMediumCount}
            onChange={(event) => setAdaptiveMediumCount(Number(event.target.value))}
            onMouseUp={(event) => submitAdaptiveParticleCount({ medium_particle_count: Number(event.currentTarget.value) })}
            onTouchEnd={(event) => submitAdaptiveParticleCount({ medium_particle_count: Number(event.currentTarget.value) })}
            disabled={!capabilities.adaptive_particle_count || !adaptiveEnabled}
          />
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Max count</span><span className="v">{adaptiveMaxCount}</span></div>
          <input className="fc-slider" type="range" min={16} max={2048} step={16} value={adaptiveMaxCount}
            onChange={(event) => setAdaptiveMaxCount(Number(event.target.value))}
            onMouseUp={(event) => submitAdaptiveParticleCount({ max_particle_count: Number(event.currentTarget.value) })}
            onTouchEnd={(event) => submitAdaptiveParticleCount({ max_particle_count: Number(event.currentTarget.value) })}
            disabled={!capabilities.adaptive_particle_count || !adaptiveEnabled}
          />
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Stable / unstable frames</span><span className="v">{adaptiveStableUpdates} / {adaptiveUnstableUpdates}</span></div>
          <input className="fc-slider" type="range" min={1} max={30} step={1} value={adaptiveStableUpdates}
            onChange={(event) => setAdaptiveStableUpdates(Number(event.target.value))}
            onMouseUp={(event) => submitAdaptiveParticleCount({ stable_required_updates: Number(event.currentTarget.value) })}
            onTouchEnd={(event) => submitAdaptiveParticleCount({ stable_required_updates: Number(event.currentTarget.value) })}
            disabled={!capabilities.adaptive_particle_count || !adaptiveEnabled}
          />
          <input className="fc-slider" type="range" min={1} max={10} step={1} value={adaptiveUnstableUpdates}
            onChange={(event) => setAdaptiveUnstableUpdates(Number(event.target.value))}
            onMouseUp={(event) => submitAdaptiveParticleCount({ unstable_required_updates: Number(event.currentTarget.value) })}
            onTouchEnd={(event) => submitAdaptiveParticleCount({ unstable_required_updates: Number(event.currentTarget.value) })}
            disabled={!capabilities.adaptive_particle_count || !adaptiveEnabled}
          />
        </div>
        <div className="fc-hint">
          stable {adaptive.stable_update_count ?? 0} · unstable {adaptive.unstable_update_count ?? 0} · spread {(adaptive.xy_spread_meters ?? 0).toFixed(2)}m · reason {adaptive.last_resize_reason || '—'}
        </div>
      </div>

      <div className="fc-section">
        <div className="fc-h">ROUGHENING</div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Module</span><span className="v">{rougheningEnabled ? 'enabled' : 'disabled'}</span></div>
          <div className="fc-seg">
            <button
              className={rougheningEnabled ? 'on' : ''}
              onClick={() => { setRougheningEnabled(true); submitRoughening({ enabled: true }); }}
              disabled={!capabilities.pf_modules}
            >on</button>
            <button
              className={!rougheningEnabled ? 'on' : ''}
              onClick={() => { setRougheningEnabled(false); submitRoughening({ enabled: false }); }}
              disabled={!capabilities.pf_modules}
            >off</button>
          </div>
        </div>
        <div className="fc-row">
          <div className="fc-lbl">Mode</div>
          <div className="fc-seg">
            {['always', 'resample_only'].map((mode) => (
              <button
                key={mode}
                className={rougheningMode === mode ? 'on' : ''}
                onClick={() => { setRougheningMode(mode); submitRoughening({ mode }); }}
                disabled={!capabilities.pf_modules || !rougheningEnabled}
              >{mode.replace('_', ' ')}</button>
            ))}
          </div>
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Ratio</span><span className="v">{(rougheningRatio * 100).toFixed(0)}% · last {roughened}</span></div>
          <input className="fc-slider" type="range" min={0} max={0.3} step={0.01} value={rougheningRatio}
            onChange={(event) => setRougheningRatio(Number(event.target.value))}
            onMouseUp={(event) => submitRoughening({ ratio: Number(event.currentTarget.value) })}
            onTouchEnd={(event) => submitRoughening({ ratio: Number(event.currentTarget.value) })}
            disabled={!capabilities.pf_modules || !rougheningEnabled}
          />
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>σx / σy</span><span className="v">{rougheningSigmaX.toFixed(2)} / {rougheningSigmaY.toFixed(2)} m</span></div>
          <input className="fc-slider" type="range" min={0} max={0.3} step={0.01} value={rougheningSigmaX}
            onChange={(event) => setRougheningSigmaX(Number(event.target.value))}
            onMouseUp={(event) => submitRoughening({ sigma_x: Number(event.currentTarget.value) })}
            onTouchEnd={(event) => submitRoughening({ sigma_x: Number(event.currentTarget.value) })}
            disabled={!capabilities.pf_modules || !rougheningEnabled}
          />
          <input className="fc-slider" type="range" min={0} max={0.3} step={0.01} value={rougheningSigmaY}
            onChange={(event) => setRougheningSigmaY(Number(event.target.value))}
            onMouseUp={(event) => submitRoughening({ sigma_y: Number(event.currentTarget.value) })}
            onTouchEnd={(event) => submitRoughening({ sigma_y: Number(event.currentTarget.value) })}
            disabled={!capabilities.pf_modules || !rougheningEnabled}
          />
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>σθ</span><span className="v">{(rougheningSigmaYaw * 180 / Math.PI).toFixed(1)}°</span></div>
          <input className="fc-slider" type="range" min={0} max={0.5} step={0.01} value={rougheningSigmaYaw}
            onChange={(event) => setRougheningSigmaYaw(Number(event.target.value))}
            onMouseUp={(event) => submitRoughening({ sigma_yaw: Number(event.currentTarget.value) })}
            onTouchEnd={(event) => submitRoughening({ sigma_yaw: Number(event.currentTarget.value) })}
            disabled={!capabilities.pf_modules || !rougheningEnabled}
          />
        </div>
      </div>

      <div className="fc-section">
        <div className="fc-h">RECOVERY</div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Module</span><span className="v">{recoveryEnabled ? 'enabled' : 'disabled'}</span></div>
          <div className="fc-seg">
            <button className={recoveryEnabled ? 'on' : ''} onClick={() => { setRecoveryEnabled(true); submitRecovery({ enabled: true }); }} disabled={!capabilities.pf_modules}>on</button>
            <button className={!recoveryEnabled ? 'on' : ''} onClick={() => { setRecoveryEnabled(false); submitRecovery({ enabled: false }); }} disabled={!capabilities.pf_modules}>off</button>
          </div>
        </div>
        <div className="fc-row">
          <div className="fc-lbl">Strategy</div>
          <div className="fc-seg">
            {['absolute_score', 'augmented_mcl'].map((strategy) => (
              <button
                key={strategy}
                className={recoveryStrategy === strategy ? 'on' : ''}
                onClick={() => { setRecoveryStrategy(strategy); submitRecovery({ strategy }); }}
                disabled={!capabilities.pf_modules || !recoveryEnabled}
              >{strategy.replace('_', ' ')}</button>
            ))}
          </div>
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Max global injection</span><span className="v">{(recoveryMaxRatio * 100).toFixed(0)}% · last {injected}</span></div>
          <input className="fc-slider" type="range" min={0} max={0.8} step={0.05} value={recoveryMaxRatio}
            onChange={(event) => setRecoveryMaxRatio(Number(event.target.value))}
            onMouseUp={(event) => submitRecovery({ random_particle_max_ratio: Number(event.currentTarget.value) })}
            onTouchEnd={(event) => submitRecovery({ random_particle_max_ratio: Number(event.currentTarget.value) })}
            disabled={!capabilities.pf_modules || !recoveryEnabled}
          />
          <div className="fc-hint">Current recovery ratio: {(recoveryRatio * 100).toFixed(1)}%</div>
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>{metricName} thresholds</span><span className="v">best {bestThreshold.toFixed(2)} · med {medianThreshold.toFixed(2)}</span></div>
          <input className="fc-slider" type="range" min={0.1} max={0.8} step={0.01} value={bestThreshold}
            onChange={(event) => setBestThreshold(Number(event.target.value))}
            onMouseUp={(event) => submitRecovery({ profile: { best_score_threshold: Number(event.currentTarget.value) } })}
            onTouchEnd={(event) => submitRecovery({ profile: { best_score_threshold: Number(event.currentTarget.value) } })}
            disabled={!capabilities.pf_modules || !recoveryEnabled || recoveryStrategy !== 'absolute_score'}
          />
          <input className="fc-slider" type="range" min={0.1} max={0.8} step={0.01} value={medianThreshold}
            onChange={(event) => setMedianThreshold(Number(event.target.value))}
            onMouseUp={(event) => submitRecovery({ profile: { median_score_threshold: Number(event.currentTarget.value) } })}
            onTouchEnd={(event) => submitRecovery({ profile: { median_score_threshold: Number(event.currentTarget.value) } })}
            disabled={!capabilities.pf_modules || !recoveryEnabled || recoveryStrategy !== 'absolute_score'}
          />
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Absolute injection</span><span className="v">{(absoluteRatio * 100).toFixed(0)}% after {badUpdates} bad</span></div>
          <input className="fc-slider" type="range" min={0} max={0.8} step={0.05} value={absoluteRatio}
            onChange={(event) => setAbsoluteRatio(Number(event.target.value))}
            onMouseUp={(event) => submitRecovery({ profile: { random_particle_ratio: Number(event.currentTarget.value) } })}
            onTouchEnd={(event) => submitRecovery({ profile: { random_particle_ratio: Number(event.currentTarget.value) } })}
            disabled={!capabilities.pf_modules || !recoveryEnabled || recoveryStrategy !== 'absolute_score'}
          />
          <input className="fc-slider" type="range" min={1} max={10} step={1} value={badUpdates}
            onChange={(event) => setBadUpdates(Number(event.target.value))}
            onMouseUp={(event) => submitRecovery({ profile: { consecutive_bad_updates: Number(event.currentTarget.value) } })}
            onTouchEnd={(event) => submitRecovery({ profile: { consecutive_bad_updates: Number(event.currentTarget.value) } })}
            disabled={!capabilities.pf_modules || !recoveryEnabled || recoveryStrategy !== 'absolute_score'}
          />
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { ParticleFilterModules });
