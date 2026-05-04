// controls-panel.jsx — Filter parameter controls.
// Lives in the right column under the FILTER tab. Mutates filter.params live.

function FilterControls({ filter, particleCount, setParticleCount, paused, setPaused, onReset, onStep, priorPreset, setPriorPreset }) {
  const [, force] = React.useReducer((x) => x + 1, 0);
  const p = filter.params;

  const update = (patch) => {
    filter.setParams(patch);
    force();
  };

  // Sync prior preset → priorSigma values
  React.useEffect(() => {
    const presets = {
      tight:  { priorSigmaXY: 6,  priorSigmaT: 0.15 },
      medium: { priorSigmaXY: 18, priorSigmaT: 0.6 },
      wide:   { priorSigmaXY: 45, priorSigmaT: 1.4 },
    };
    filter.setParams(presets[priorPreset] || presets.medium);
    force();
  }, [priorPreset, filter]);

  // ESS bar live update
  React.useEffect(() => {
    let raf;
    const tick = () => { force(); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  let essInv = 0;
  for (let i = 0; i < filter.N; i++) essInv += filter.pw[i] * filter.pw[i];
  const ess = 1 / essInv;
  const essRatio = ess / filter.N;

  return (
    <div className="fc-wrap">
      {/* Prior */}
      <div className="fc-section">
        <div className="fc-h">PRIOR</div>
        <div className="fc-row">
          <div className="fc-lbl">Spread preset</div>
          <div className="fc-seg">
            {['tight', 'medium', 'wide'].map((k) => (
              <button key={k} className={priorPreset === k ? 'on' : ''} onClick={() => setPriorPreset(k)}>{k}</button>
            ))}
          </div>
        </div>
        <div className="fc-row tworow">
          <div className="fc-mini">
            <div className="fc-mlbl">σx,y</div>
            <div className="fc-mval">{p.priorSigmaXY.toFixed(0)}<span> px</span></div>
          </div>
          <div className="fc-mini">
            <div className="fc-mlbl">σθ</div>
            <div className="fc-mval">{(p.priorSigmaT * 180 / Math.PI).toFixed(0)}<span> °</span></div>
          </div>
        </div>
        <div className="fc-hint">left-drag the map to set a prior pose · spread above controls cloud width</div>
      </div>

      {/* Particle count */}
      <div className="fc-section">
        <div className="fc-h">SAMPLES</div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Particle count</span><span className="v">{particleCount}</span></div>
          <input className="fc-slider" type="range" min={50} max={2000} step={50}
                 value={particleCount}
                 onChange={(e) => setParticleCount(Number(e.target.value))} />
          <div className="fc-rng"><span>50</span><span>fast</span><span>robust</span><span>2000</span></div>
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Resample @ ESS &lt;</span><span className="v">{(p.resampleRatio * 100).toFixed(0)}% · {(p.resampleRatio * filter.N).toFixed(0)}</span></div>
          <input className="fc-slider" type="range" min={0.1} max={0.95} step={0.05}
                 value={p.resampleRatio}
                 onChange={(e) => update({ resampleRatio: Number(e.target.value) })} />
          <div className="fc-rng"><span>preserve diversity</span><span>aggressive resample</span></div>
        </div>
        {/* Live ESS gauge */}
        <div className="fc-gauge">
          <div className="fc-gauge-track">
            <div className="fc-gauge-thr" style={{ left: `${p.resampleRatio * 100}%` }} />
            <div className={"fc-gauge-fill " + (essRatio > p.resampleRatio ? "ok" : "warn")}
                 style={{ width: `${Math.min(1, essRatio) * 100}%` }} />
          </div>
          <div className="fc-gauge-meta">
            <span>ESS</span>
            <span className="mono">{ess.toFixed(0)} / {filter.N}</span>
            <span className={"chip " + (essRatio > p.resampleRatio ? "ok" : "warn")}>
              {essRatio > p.resampleRatio ? "healthy" : "near resample"}
            </span>
          </div>
        </div>
      </div>

      {/* Measurement */}
      <div className="fc-section">
        <div className="fc-h">MEASUREMENT</div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Temperature</span><span className="v">{p.temperature.toFixed(2)}</span></div>
          <input className="fc-slider" type="range" min={0.1} max={5} step={0.05}
                 value={p.temperature}
                 onChange={(e) => update({ temperature: Number(e.target.value) })} />
          <div className="fc-rng"><span>winner-take-all</span><span>flat</span></div>
        </div>
        <div className="fc-hint">Image-likelihood softness. Low = fast lock-on, can latch wrong mode. High = patient, can stay indecisive.</div>
      </div>

      {/* Motion */}
      <div className="fc-section">
        <div className="fc-h">MOTION NOISE</div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Translational σ</span><span className="v">{(p.motionNoise * 100).toFixed(0)}%</span></div>
          <input className="fc-slider" type="range" min={0.01} max={0.6} step={0.01}
                 value={p.motionNoise}
                 onChange={(e) => update({ motionNoise: Number(e.target.value) })} />
          <div className="fc-rng"><span>trust odom</span><span>spread</span></div>
        </div>
        <div className="fc-row">
          <div className="fc-lbl"><span>Rotational σ</span><span className="v">{(p.turnNoise).toFixed(3)} rad</span></div>
          <input className="fc-slider" type="range" min={0.005} max={0.5} step={0.005}
                 value={p.turnNoise}
                 onChange={(e) => update({ turnNoise: Number(e.target.value) })} />
        </div>
      </div>

      {/* Run control */}
      <div className="fc-section">
        <div className="fc-h">RUN CONTROL</div>
        <div className="fc-actions">
          <button className={"fc-btn " + (paused ? "primary" : "")} onClick={() => setPaused(!paused)}>
            {paused ? "▶ RESUME" : "❚❚ PAUSE"}
          </button>
          <button className="fc-btn" onClick={onStep} disabled={!paused}>⤳ STEP ONCE</button>
          <button className="fc-btn danger" onClick={onReset}>↻ GLOBAL RESET</button>
        </div>
        <div className="fc-hint">Pause freezes state for inspection. Step advances exactly one observation cycle.</div>
      </div>
    </div>
  );
}

Object.assign(window, { FilterControls });
