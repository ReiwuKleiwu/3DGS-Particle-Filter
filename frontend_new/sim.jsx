// sim.jsx — Procedural floorplan + particle filter simulator (mocked).
// Generates a believable office occupancy grid (PGM-style) and runs a small
// MCL particle filter so the dashboard has live, plausible data.

const MAP_W = 480;
const MAP_H = 320;

// ── Procedural floorplan ────────────────────────────────────────────────────
// Returns { grid: Uint8Array(MAP_W*MAP_H), W, H } where 0=free, 1=occupied,
// 2=unknown. Layout: outer walls, several rooms, doors, scattered furniture.
function buildOccupancyGrid() {
  const grid = new Uint8Array(MAP_W * MAP_H);
  const set = (x, y, v) => {
    if (x < 0 || y < 0 || x >= MAP_W || y >= MAP_H) return;
    grid[y * MAP_W + x] = v;
  };
  const rect = (x0, y0, x1, y1, v) => {
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) set(x, y, v);
  };
  const wallH = (x0, x1, y, t = 2) => rect(x0, y, x1, y + t - 1, 1);
  const wallV = (x, y0, y1, t = 2) => rect(x, y0, x + t - 1, y1, 1);

  // unknown halo around the map
  for (let y = 0; y < MAP_H; y++) for (let x = 0; x < MAP_W; x++) grid[y * MAP_W + x] = 2;
  // free interior
  rect(20, 20, MAP_W - 21, MAP_H - 21, 0);

  // Outer walls
  wallH(20, MAP_W - 21, 20);
  wallH(20, MAP_W - 21, MAP_H - 22);
  wallV(20, 20, MAP_H - 21);
  wallV(MAP_W - 22, 20, MAP_H - 21);

  // Inner partitions — a corridor + 4 rooms
  // Vertical spine at x=200
  wallV(200, 20, 130);
  wallV(200, 170, MAP_H - 22);
  // Horizontal partition at y=150 on the left
  wallH(20, 130, 150);
  wallH(170, 200, 150);
  // Horizontal partition at y=160 on the right
  wallH(200, 320, 160);
  wallH(360, MAP_W - 22, 160);
  // Vertical partition right side at x=380
  wallV(380, 20, 100);
  wallV(380, 140, 160);

  // Furniture / clutter (small obstacles to make scans interesting)
  rect(60, 50, 80, 70, 1);    // desk
  rect(110, 50, 130, 65, 1);  // desk
  rect(60, 100, 90, 115, 1);  // table
  rect(40, 200, 75, 235, 1);  // couch
  rect(120, 210, 145, 230, 1);// chair cluster
  rect(240, 50, 280, 80, 1);  // table
  rect(310, 50, 340, 70, 1);  // shelf
  rect(230, 200, 290, 220, 1);// long table
  rect(330, 200, 360, 230, 1);// printer
  rect(410, 40, 440, 70, 1);  // small office desk
  rect(410, 200, 445, 240, 1);// big shelf

  return { grid, W: MAP_W, H: MAP_H };
}

const OCC = buildOccupancyGrid();

function isFree(x, y) {
  const ix = Math.round(x), iy = Math.round(y);
  if (ix < 1 || iy < 1 || ix >= MAP_W - 1 || iy >= MAP_H - 1) return false;
  return OCC.grid[iy * MAP_W + ix] === 0;
}

// ── Ground-truth path ───────────────────────────────────────────────────────
// A hand-tuned waypoint loop through the corridors. The robot follows it with
// a constant speed, easing through corners. Returns { x, y, theta } at time t.
const PATH = [
  { x: 60, y: 230 },
  { x: 60, y: 180 },
  { x: 150, y: 180 },
  { x: 150, y: 100 },
  { x: 60, y: 100 },
  { x: 60, y: 50 },
  { x: 180, y: 50 },
  { x: 180, y: 130 },
  { x: 240, y: 130 },
  { x: 240, y: 180 },
  { x: 350, y: 180 },
  { x: 350, y: 130 },
  { x: 430, y: 130 },
  { x: 430, y: 250 },
  { x: 300, y: 250 },
  { x: 300, y: 280 },
  { x: 80, y: 280 },
];

const PATH_SEGS = PATH.map((p, i) => {
  const q = PATH[(i + 1) % PATH.length];
  const dx = q.x - p.x, dy = q.y - p.y;
  return { p, q, len: Math.hypot(dx, dy), theta: Math.atan2(dy, dx) };
});
const PATH_TOTAL = PATH_SEGS.reduce((a, s) => a + s.len, 0);

function poseAtDistance(d) {
  const dist = ((d % PATH_TOTAL) + PATH_TOTAL) % PATH_TOTAL;
  let acc = 0;
  for (const s of PATH_SEGS) {
    if (dist <= acc + s.len) {
      const u = (dist - acc) / s.len;
      return { x: s.p.x + (s.q.x - s.p.x) * u, y: s.p.y + (s.q.y - s.p.y) * u, theta: s.theta };
    }
    acc += s.len;
  }
  return { ...PATH[0], theta: 0 };
}

// ── Tiny seedable RNG (deterministic resets) ────────────────────────────────
function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng) {
  // Box-Muller
  const u = Math.max(1e-9, rng()), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── Particle filter state ───────────────────────────────────────────────────
// Particles: typed arrays for px, py, ptheta, weight. We expose a step()
// function and a reset(prior) function. The "measurement" is the pose at
// distance traveled — in real life this would be a 3D-GS render likelihood.
function createFilter(N = 300, seed = 1234) {
  const rng = mulberry32(seed);
  const px = new Float32Array(N);
  const py = new Float32Array(N);
  const pt = new Float32Array(N);
  const pw = new Float32Array(N);

  // Mutable parameters — the controls panel writes into this object live.
  const params = {
    temperature: 1.0,        // weight softness; <1 = sharper, >1 = flatter
    motionNoise: 0.18,       // fraction of motion magnitude
    turnNoise: 0.05,         // rad / step
    resampleRatio: 0.5,      // resample when ESS < N * ratio
    priorSigmaXY: 18,        // px
    priorSigmaT: 0.6,        // rad
  };

  let trueDist = 0;
  let estX = 0, estY = 0, estTheta = 0;
  let cov = { xx: 200, yy: 200, xy: 0 }; // start uncertain
  let errorHistory = []; // {t, err}
  let elapsed = 0;
  let convergence = 0; // 0 = unconverged, 1 = converged

  function seedAround(x, y, theta, sigmaXY = 30, sigmaT = Math.PI) {
    for (let i = 0; i < N; i++) {
      let tries = 0, qx, qy;
      do {
        qx = x + gauss(rng) * sigmaXY;
        qy = y + gauss(rng) * sigmaXY;
        tries++;
      } while (!isFree(qx, qy) && tries < 8);
      px[i] = qx; py[i] = qy;
      pt[i] = theta + gauss(rng) * sigmaT;
      pw[i] = 1 / N;
    }
    convergence = 0;
  }

  // Initial: spread particles widely (kidnapped robot situation)
  function globalSeed() {
    for (let i = 0; i < N; i++) {
      let qx, qy, tries = 0;
      do {
        qx = 25 + rng() * (MAP_W - 50);
        qy = 25 + rng() * (MAP_H - 50);
        tries++;
      } while (!isFree(qx, qy) && tries < 20);
      px[i] = qx; py[i] = qy;
      pt[i] = (rng() * 2 - 1) * Math.PI;
      pw[i] = 1 / N;
    }
    convergence = 0;
  }

  globalSeed();

  function reset(prior) {
    trueDist = 0;
    elapsed = 0;
    errorHistory = [];
    if (prior) seedAround(prior.x, prior.y, prior.theta, params.priorSigmaXY, params.priorSigmaT);
    else globalSeed();
    cov = { xx: 200, yy: 200, xy: 0 };
  }

  function step(dt) {
    // Advance ground truth along the path
    const SPEED = 22; // px/sec
    trueDist += SPEED * dt;
    elapsed += dt;
    const truth = poseAtDistance(trueDist);

    // 1. Motion update — same control input applied to every particle with noise
    const motion = SPEED * dt;
    const motionNoise = motion * params.motionNoise;
    const turnNoise = params.turnNoise;
    for (let i = 0; i < N; i++) {
      const m = motion + gauss(rng) * motionNoise;
      const th = pt[i] + gauss(rng) * turnNoise;
      let nx = px[i] + Math.cos(th) * m;
      let ny = py[i] + Math.sin(th) * m;
      if (!isFree(nx, ny)) {
        // bump: keep position but penalize weight slightly later
        nx = px[i]; ny = py[i];
      }
      px[i] = nx; py[i] = ny; pt[i] = th;
    }

    // 2. Measurement update — weight each particle by gaussian distance to truth
    // (stand-in for "render this particle's view from 3D-GS and compare to camera")
    const SIGMA = 18; // px
    const STH = 0.5;  // rad
    let wsum = 0;
    for (let i = 0; i < N; i++) {
      const dx = px[i] - truth.x;
      const dy = py[i] - truth.y;
      let dth = pt[i] - truth.theta;
      // wrap
      while (dth > Math.PI) dth -= 2 * Math.PI;
      while (dth < -Math.PI) dth += 2 * Math.PI;
      const d2 = (dx * dx + dy * dy) / (SIGMA * SIGMA) + (dth * dth) / (STH * STH);
      // temperature softens / sharpens the weight distribution
      const w = Math.exp(-0.5 * d2 / Math.max(0.05, params.temperature)) + 1e-6;
      pw[i] = w;
      wsum += w;
    }
    for (let i = 0; i < N; i++) pw[i] /= wsum;

    // 3. ESS — resample if low
    let essInv = 0;
    for (let i = 0; i < N; i++) essInv += pw[i] * pw[i];
    const ess = 1 / essInv;

    if (ess < N * params.resampleRatio) {
      // systematic resample
      const r = rng() / N;
      let c = pw[0];
      let i = 0;
      const nx = new Float32Array(N), ny = new Float32Array(N), nth = new Float32Array(N);
      for (let m = 0; m < N; m++) {
        const u = r + m / N;
        while (u > c && i < N - 1) { i++; c += pw[i]; }
        // jitter slightly to avoid degeneracy
        nx[m] = px[i] + gauss(rng) * 1.2;
        ny[m] = py[i] + gauss(rng) * 1.2;
        nth[m] = pt[i] + gauss(rng) * 0.04;
      }
      for (let m = 0; m < N; m++) { px[m] = nx[m]; py[m] = ny[m]; pt[m] = nth[m]; pw[m] = 1 / N; }
    }

    // 4. Estimate — weighted mean
    let mx = 0, my = 0, sx = 0, cs = 0;
    for (let i = 0; i < N; i++) {
      mx += px[i] * pw[i];
      my += py[i] * pw[i];
      sx += Math.sin(pt[i]) * pw[i];
      cs += Math.cos(pt[i]) * pw[i];
    }
    estX = mx; estY = my;
    estTheta = Math.atan2(sx, cs);

    // 5. Covariance for ellipse
    let cxx = 0, cyy = 0, cxy = 0;
    for (let i = 0; i < N; i++) {
      const dx = px[i] - estX, dy = py[i] - estY;
      cxx += dx * dx * pw[i];
      cyy += dy * dy * pw[i];
      cxy += dx * dy * pw[i];
    }
    cov = { xx: cxx, yy: cyy, xy: cxy };

    // convergence: ease toward 1 as cov shrinks
    const spread = Math.sqrt(cxx + cyy);
    const target = Math.max(0, Math.min(1, 1 - spread / 80));
    convergence += (target - convergence) * Math.min(1, dt * 1.5);

    const err = Math.hypot(estX - truth.x, estY - truth.y);
    const ex = estX - truth.x;
    const ey = estY - truth.y;
    let eth = estTheta - truth.theta;
    while (eth > Math.PI) eth -= 2 * Math.PI;
    while (eth < -Math.PI) eth += 2 * Math.PI;
    errorHistory.push({ t: elapsed, err, ex, ey, eth });
    if (errorHistory.length > 4000) errorHistory.shift();

    return { truth, est: { x: estX, y: estY, theta: estTheta }, ess, cov, err };
  }

  return {
    N, px, py, pt, pw,
    step, reset,
    params,
    setParams: (patch) => Object.assign(params, patch),
    getTruth: () => poseAtDistance(trueDist),
    getEst: () => ({ x: estX, y: estY, theta: estTheta }),
    getCov: () => cov,
    getErrorHistory: () => errorHistory,
    getElapsed: () => elapsed,
    getConvergence: () => convergence,
  };
}

Object.assign(window, { OCC, MAP_W, MAP_H, createFilter, poseAtDistance, isFree });
