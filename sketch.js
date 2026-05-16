// ─── Mouse / keyboard ────────────────────────────────────────────────────────

function mousePressed() {
  // Details button (bottom-right) handled by HTML — skip if click lands there
  let btnW = 180, btnH = 48, btnX = SZ_W - btnW - 32, btnY = SZ_H - btnH - 32;
  if (mouseX > btnX && mouseX < btnX+btnW && mouseY > btnY && mouseY < btnY+btnH) return;

  // Canvas clicks do nothing — orb handles mic toggle via its own click event
}

window.addEventListener('click', ()=>{ if(playbackActive && !window._sessionActive) stopOnUserInteraction = true; });
window.addEventListener('mousemove', ()=>{ if(playbackActive && !window._sessionActive) stopOnUserInteraction = true; });

window.stopMicIfActive = function() {
  if (inputSource === 'mic') finishMicAnalysis();
};

window.fadeVisualization = function() {
  if (state === 'visualising') { state = 'fading'; fadeAlpha = 0; }
};

window.stopAudioPlayback = function() {
  if (playbackAudio) { playbackAudio.pause(); playbackAudio = null; }
  playbackActive = false;
};

window.resetSketch = function() {
  _sessionId++;                          // invalidates any in-flight async fetch
  clearInterval(window._revealTimer);
  clearTimeout(window._vizStopTimer);
  if (playbackAudio) { playbackAudio.pause(); playbackAudio.currentTime = 0; }
  playbackActive = false; stopOnUserInteraction = false;
  inputSource = null;
  state = 'idle'; fadeAlpha = 0;
  statusMsg = '';
  background(245, 244, 242);
  drawBackground(-1);
};

window.clearCanvasOnly = function() {
  clearInterval(window._revealTimer);
  clearTimeout(window._vizStopTimer);
  if (playbackAudio) { playbackAudio.pause(); playbackAudio = null; }
  playbackActive = false; stopOnUserInteraction = false;
  inputSource = null;
  state = 'idle';
  drawBackground(-1);
  // does NOT call showOrb or reset session state
};

// ─── Global state ─────────────────────────────────────────────────────────────

let _sessionId = 0;   // incremented on every reset; async callbacks check this before drawing
let fft = null;    // lazy-init on first user gesture (avoids AudioContext errors)
let mic = null;
let inputSource = null;
let statusMsg = '';
let SZ_W, SZ_H;
let _ctx = null;   // raw 2D context — captured in setup()

let state = 'idle';
let vParams = { numOrigins:8, lineWeight:1.2, bass:0.3, treble:0.3 };
let vizFrame = 0;
let fadeAlpha = 0;
let mediaRecorder = null;
let recordedChunks = [];
let playbackAudio = null;
let playbackActive = false;
let stopOnUserInteraction = false;

let dispMeanEntropy='\u2014', dispStdEntropy='\u2014', dispRMS='\u2014';
let dispKeynote='\u2014', dispSignal='\u2014';
let mEntropy=9.0, mEntropySD=0.5, mRMS=0.02, mKeynote=400, mSignal=0.05;
let detectedCategories = [];

// ─── YAMNet category layout ───────────────────────────────────────────────────

// Y axis = logarithmic frequency (high freq → top / low Y, low freq → bottom / high Y)
// Approximate centre frequency per category, mapped log10(20–10000 Hz) → 0–1
const CATEGORY_Y = {
  insects:     0.06,  // ~5000 Hz
  birds:       0.12,  // ~3000 Hz
  alarm_siren: 0.20,  // ~2000 Hz
  bells:       0.27,  // ~1500 Hz
  voices:      0.35,  // ~800 Hz
  music:       0.40,  // ~600 Hz
  amphibians:  0.44,  // ~500 Hz
  handling:    0.47,  // ~500 Hz
  white_noise: 0.50,  // broadband
  silence:     0.50,  // neutral
  rain:        0.53,  // ~400 Hz
  water:       0.57,  // ~300 Hz
  wind:        0.61,  // ~200 Hz
  fire:        0.65,  // ~180 Hz
  mammals:     0.68,  // ~150 Hz
  footsteps:   0.71,  // ~120 Hz
  object_moving:0.74, // ~100 Hz
  construction:0.77,  // ~80 Hz
  machinery:   0.80,  // ~70 Hz
  wind_howl:   0.83,  // ~60 Hz
  thunder:     0.86,  // ~50 Hz
  vehicles:    0.90,  // ~50 Hz
  low_freq:    0.94,  // ~30 Hz
};
const CATEGORY_COLORS = {
  rain:[60,130,255], water:[0,210,255], thunder:[140,40,220], fire:[255,110,20],
  wind:[180,215,230], amphibians:[40,200,90], insects:[180,220,0], birds:[255,215,0],
  mammals:[180,100,40], vehicles:[90,90,90], voices:[255,60,60], footsteps:[200,160,100],
  machinery:[220,100,0], bells:[220,200,100], music:[255,80,200], object_moving:[150,150,150],
  alarm_siren:[255,0,30], construction:[200,150,0], low_freq:[60,60,130],
  white_noise:[230,230,230], wind_howl:[170,215,255], handling:[200,200,200],
  silence:[245,248,252],
};

// ─── Mic-analysis accumulators ────────────────────────────────────────────────

let liveSumE=0, liveSumBass=0, liveSumTreble=0, liveSumFlux=0;
let liveFrames=0;
const getMaxMicFrames = () => (window._sessMaxRecordSecs || 30) * 60;
let lastSpectrum = [];
let allTrailPoints = [];

// ─── p5 lifecycle ─────────────────────────────────────────────────────────────

function setup() {
  let cnv = createCanvas(windowWidth, windowHeight);
  _ctx = cnv.elt.getContext('2d');
  SZ_W = windowWidth; SZ_H = windowHeight;
  drawBackground(-1);
  // fft is lazy-init in toggleMic to avoid AudioContext-before-gesture errors
}

function windowResized() {
  SZ_W = windowWidth; SZ_H = windowHeight;
  if (state === 'visualising' || state === 'fading') return;
  resizeCanvas(windowWidth, windowHeight);
  drawBackground(-1);
}

// ─── Background ───────────────────────────────────────────────────────────────

function drawBackground(en) {
  if (en === undefined) en = -1;
  let ctx = _ctx;
  if (!ctx) return;
  if (window.blackBgMode) {
    ctx.fillStyle = 'rgb(0,0,0)';
    ctx.fillRect(0, 0, SZ_W, SZ_H);
    return;
  }
  ctx.fillStyle = 'rgb(245,244,242)';
  ctx.fillRect(0, 0, SZ_W, SZ_H);
  let grad = ctx.createRadialGradient(
    SZ_W*0.5, SZ_H*0.5, SZ_W*0.01,
    SZ_W*0.5, SZ_H*0.5, Math.max(SZ_W, SZ_H)*0.85
  );
  if (en < 0) {
    grad.addColorStop(0.00, 'rgb(255,255,255)');
    grad.addColorStop(0.50, 'rgb(245,245,248)');
    grad.addColorStop(1.00, 'rgb(218,220,225)');
  } else if (en < 0.35) {
    grad.addColorStop(0.00, 'rgb(248,252,255)');
    grad.addColorStop(0.40, 'rgb(235,242,252)');
    grad.addColorStop(0.75, 'rgb(210,222,240)');
    grad.addColorStop(1.00, 'rgb(185,200,225)');
  } else if (en < 0.65) {
    grad.addColorStop(0.00, 'rgb(255,255,252)');
    grad.addColorStop(0.40, 'rgb(250,248,242)');
    grad.addColorStop(0.75, 'rgb(235,228,218)');
    grad.addColorStop(1.00, 'rgb(215,205,190)');
  } else {
    grad.addColorStop(0.00, 'rgb(255,254,248)');
    grad.addColorStop(0.40, 'rgb(252,246,232)');
    grad.addColorStop(0.75, 'rgb(238,225,205)');
    grad.addColorStop(1.00, 'rgb(218,200,175)');
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, SZ_W, SZ_H);
}

// ─── Drawing helpers ──────────────────────────────────────────────────────────

function getLineColour(sdNorm, sigNorm, hueShift, sat) {
  let r = constrain(int(lerp(35,75,sdNorm)*sat)+hueShift, 15, 105);
  let g = constrain(int(lerp(50,52,sdNorm)*sat)+int(random(-4,4)), 18, 85);
  let b = constrain(int(lerp(80,38,sdNorm)*sat)+int(-hueShift*0.4), 15, 115);
  return {r, g, b};
}

function drawBurningLine(x,y,nx,ny,r,g,b,alpha,wt) {
  stroke(min(255,r+80),min(255,g+80),min(255,b+80),alpha*0.15); strokeWeight(wt*12.0); line(x,y,nx,ny);
  stroke(min(255,r+50),min(255,g+50),min(255,b+50),alpha*0.22); strokeWeight(wt*7.0);  line(x,y,nx,ny);
  stroke(min(255,r+25),min(255,g+25),min(255,b+25),alpha*0.35); strokeWeight(wt*4.0);  line(x,y,nx,ny);
  stroke(max(0,r-30),max(0,g-30),max(0,b-30),alpha*0.70);       strokeWeight(wt*2.2);  line(x,y,nx,ny);
  stroke(r,g,b,alpha);                                            strokeWeight(wt);      line(x,y,nx,ny);
  stroke(min(255,r+35),min(255,g+35),min(255,b+35),alpha*0.65); strokeWeight(wt*0.38); line(x,y,nx,ny);
}

function buildParticlePath(ox,oy,startAngle,bw,sdNorm,sigNorm,rmsAlpha,rmsNorm,pitchNorm,entropyNorm,scaleMult,depth) {
  if (depth > 2) return [];
  let curlDir = random() > 0.5 ? 1 : -1;
  let x = ox, y = oy, angle = startAngle;
  let speed = lerp(10.0,22.0,rmsNorm)*random(0.4,1.8)*scaleMult;
  let steps = int(lerp(120,320,rmsNorm)*random(0.8,1.2));
  let wBase = random(0.12,0.65)*bw*scaleMult;
  let style = random();
  let curlAmount;
  let sc=lerp(0.35,0.06,sdNorm), gc=lerp(0.60,0.22,sdNorm), ac=lerp(0.80,0.48,sdNorm);
  if      (style < sc) curlAmount = random(0.002,0.012);
  else if (style < gc) curlAmount = random(0.012,0.040);
  else if (style < ac) curlAmount = random(0.040,0.090);
  else                 curlAmount = random(0.090,0.250);
  let spiralStart = int(steps*lerp(0.30,0.10,pitchNorm)*random(0.7,1.0));
  let hueShift = random(-12,12);
  let sat = lerp(0.7,1.0,entropyNorm);
  let col = getLineColour(sdNorm, sigNorm, hueShift, sat);
  let baseR=col.r, baseG=col.g, baseB=col.b;
  // Entropy drives noise turbulence: low entropy = smooth predictable paths,
  // high entropy = erratic, frequently redirected paths
  let noiseTurb  = lerp(0.008, 0.055, entropyNorm); // angle perturbation per step
  let noiseScale = lerp(0.003, 0.012, entropyNorm); // spatial frequency of the noise field
  let pts = [];
  for (let s=0; s<steps; s++) {
    let t = s/steps;
    if (s < spiralStart) {
      angle += curlAmount*curlDir*0.12;
    } else {
      let sp = s-spiralStart;
      angle += curlAmount*curlDir*(1.0+sp*lerp(0.010,0.030,pitchNorm));
      speed = max(0.2, speed*0.968);
    }
    // Replace fixed noise warp with entropy-scaled turbulence
    angle += map(noise(x*noiseScale, y*noiseScale+s*0.02), 0, 1, -noiseTurb, noiseTurb);
    let nx = x+cos(angle)*speed, ny = y+sin(angle)*speed;
    if (nx<-100||nx>SZ_W+100||ny<-100||ny>SZ_H+100) break;
    let wt    = lerp(wBase, wBase*0.04, t);
    let alpha = lerp(240, 0, t)*rmsAlpha;
    let tn    = noise(x*0.018, y*0.018);
    pts.push({x,y,nx,ny,r:baseR,g:baseG,b:baseB,alpha:alpha*map(tn,0,1,0.55,1.0),wt,tn});
    if (t>0.05 && t<0.75 && depth<2) {
      let spawnProb = lerp(0.010,0.030,constrain((entropyNorm+sdNorm)*0.5,0,1));
      if (random() < spawnProb) {
        // Low entropy: children branch narrowly (converging); high entropy: wide diverging forks
        let branchSpread = lerp(PI*0.15, PI*0.90, entropyNorm);
        let childAngle = angle+(random()>0.5?1:-1)*random(branchSpread*0.4, branchSpread);
        let childSd = min(1.0, sdNorm+random(0.35,0.7));
        let childPts = buildParticlePath(nx,ny,childAngle,bw,childSd,sigNorm,
                         rmsAlpha,rmsNorm,pitchNorm,entropyNorm,scaleMult*random(0.45,0.80),depth+1);
        pts.push(...childPts);
      }
    }
    x = nx; y = ny;
  }
  return pts;
}

function drawCrossingEffects() {
  for (let i=0; i<allTrailPoints.length; i++) {
    for (let j=i+1; j<allTrailPoints.length; j++) {
      if (!allTrailPoints[i].length||!allTrailPoints[j].length) continue;
      let pi = allTrailPoints[i][int(random(allTrailPoints[i].length))];
      let pj = allTrailPoints[j][int(random(allTrailPoints[j].length))];
      if (pi && pj && dist(pi.x,pi.y,pj.x,pj.y) < 20) {
        let startR=random(6,14), tightness=random(2.5,5.0);
        let coilDir=random()>0.5?1:-1, coilA=random(TWO_PI);
        let csteps = int(startR/tightness*6);
        for (let s=0; s<csteps; s++) {
          let theta = s*0.18;
          let r  = max(0.3, startR-theta*tightness);
          let r2 = max(0.3, r-0.18*tightness);
          let x1=pi.x+cos(coilA+theta*coilDir)*r,  y1=pi.y+sin(coilA+theta*coilDir)*r;
          let x2=pi.x+cos(coilA+(theta+0.18)*coilDir)*r2, y2=pi.y+sin(coilA+(theta+0.18)*coilDir)*r2;
          let t = s/csteps;
          stroke(60,60,70,lerp(140,0,t)); strokeWeight(lerp(0.8,0.1,t)); line(x1,y1,x2,y2);
          if (r <= 0.3) break;
        }
      }
    }
  }
}

// ─── draw() ───────────────────────────────────────────────────────────────────

function draw() {
  // Keep canvas clear during idle and analysing states
  if (state === 'idle' || state === 'analysing') {
    background(window.blackBgMode ? 0 : 245, window.blackBgMode ? 0 : 244, window.blackBgMode ? 0 : 242);
  }

  // Idle: 3D model is rendered by <model-viewer> HTML element, nothing to draw here
  if (state === 'idle') {
    cursor(ARROW);
    return;
  }

  // Analysing via mic: accumulate FFT data
  if (state === 'analysing' && inputSource === 'mic' && fft) {
    let spectrum = fft.analyze();
    let energy   = fft.getEnergy('bass','treble');
    let bass     = fft.getEnergy('bass','lowMid');
    let treble   = fft.getEnergy('highMid','treble');
    let flux = 0;
    if (lastSpectrum.length > 0) {
      for (let i=0; i<spectrum.length; i++) flux += abs(spectrum[i]-(lastSpectrum[i]||0));
      flux /= spectrum.length;
    }
    lastSpectrum = spectrum.slice();
    liveSumE+=energy; liveSumBass+=bass; liveSumTreble+=treble; liveSumFlux+=flux; liveFrames++;
    statusMsg = 'RECORDING  '+int((liveFrames/getMaxMicFrames())*100)+'%';
    if (liveFrames >= getMaxMicFrames()) finishMicAnalysis();
  }

  // Visualising: category dots drawn periodically on top of particle lines
  if (state === 'visualising') {
    if (detectedCategories.length > 0 && vizFrame % 40 === 0) {
      let cat  = detectedCategories[int(random(detectedCategories.length))];
      let col  = CATEGORY_COLORS[cat] || [200,200,200];
      let xPos = map(vizFrame, 0, 1800, 40, SZ_W-40);
      let yNorm = (CATEGORY_Y[cat] !== undefined) ? CATEGORY_Y[cat] : 0.5;
      let yPos  = yNorm * SZ_H + random(-14,14);
      noStroke(); fill(col[0],col[1],col[2],200); ellipse(xPos,yPos,21,21);
    }
    vizFrame++;
    if (playbackActive && stopOnUserInteraction) {
      if (playbackAudio) { playbackAudio.pause(); playbackAudio = null; }
      playbackActive = false;
    }
  }

  // Fading: overlay that wipes canvas back to idle
  if (state === 'fading') {
    fadeAlpha += 2.0;
    let ctx = _ctx, a = fadeAlpha/255;
    // Hide prompt overlay and sd button during fade for self-directed
    if (window._selfDirected) {
      const promptEl = document.getElementById('session-prompt-text');
      if (promptEl) {
        promptEl.style.opacity = '0';
        promptEl.classList.add('hidden');
      }
      const _sdBtnFade = document.getElementById('sd-session-prompt-btn');
      if (_sdBtnFade) _sdBtnFade.style.display = 'none';
    }
    if (ctx) {
      let grad = ctx.createRadialGradient(SZ_W*0.5,SZ_H*0.5,SZ_W*0.01,SZ_W*0.5,SZ_H*0.5,Math.max(SZ_W,SZ_H)*0.85);
      grad.addColorStop(0.00, `rgba(255,255,255,${a})`);
      grad.addColorStop(1.00, `rgba(218,220,225,${a})`);
      ctx.fillStyle = grad; ctx.fillRect(0,0,SZ_W,SZ_H);
    }
    if (fadeAlpha >= 255) {
      state = 'idle';
      if (window.setPlaybackActive) window.setPlaybackActive(false);
      clear();
      drawBackground(-1);
      statusMsg = '';
      // Restore default UI: orb, prompt overlay, home button
      if (window.showOrb) window.showOrb();
      const promptEl = document.getElementById('session-prompt-text');
      if (promptEl) {
        // Only restore prompt overlay if not self-directed or session inactive
        if (!window._selfDirected || !window._sessionActive) {
          promptEl.style.opacity = '';
          promptEl.classList.remove('hidden');
        }
      }
      if (window._sessionActive) {
        const homeBtn = document.getElementById('home-btn');
        if (homeBtn) homeBtn.style.display = 'flex';
      }
      // Resume ambient prompts now that fade is fully done
      if (window._selfDirected && window._sessionActive && window._resumeAmbientPrompts) {
        window._resumeAmbientPrompts('drawing');
      }
      // Always restore sd buttons after fade if session is still active
      if (window._selfDirected && window._sessionActive) {
        const _sdBtn = document.getElementById('sd-session-prompt-btn');
        if (_sdBtn) _sdBtn.style.display = '';
      }
    }
  }

  cursor(ARROW);
}

// ─── Visualisation ────────────────────────────────────────────────────────────

function startVisualising() {
  // Map acoustic metrics to visual parameters
  let entropyNorm = constrain(map(mEntropy,  6.7, 8.3, 0.0, 1.0), 0, 1);
  let sdNorm      = constrain(map(mEntropySD,0.10,0.45, 0.0, 1.0), 0, 1);
  let pitchNorm   = constrain(map(mKeynote,  71,  2035, 0.0, 1.0), 0, 1);
  let sigNorm     = constrain(map(mSignal,   0.0, 0.545,0.0, 1.0), 0, 1);
  let rmsAlpha    = constrain(map(mRMS,      0.001,0.35,0.3, 1.0), 0.3, 1.0);
  let rmsNorm     = constrain(map(mRMS,      0.001,0.35,0.0, 1.0), 0.0, 1.0);

  // NaN guard — bad metrics still produce a drawing
  if (isNaN(sdNorm))      sdNorm      = 0.3;
  if (isNaN(entropyNorm)) entropyNorm = 0.5;
  if (isNaN(pitchNorm))   pitchNorm   = 0.4;
  if (isNaN(sigNorm))     sigNorm     = 0.3;
  if (isNaN(rmsAlpha))    rmsAlpha    = 0.6;
  if (isNaN(rmsNorm))     rmsNorm     = 0.3;

  // Blend mean entropy + SD into one complexity score that drives visual density
  // so the visual and the order→disorder slider tell the same story
  let complexityNorm = constrain((entropyNorm + sdNorm) * 0.5, 0, 1);

  let baseWeight = lerp(2.5, 0.5, rmsNorm);

  vizFrame = 0; fadeAlpha = 0;
  drawBackground(complexityNorm);
  state = 'visualising'; statusMsg = '';
  if (window.setPlaybackActive) window.setPlaybackActive(true);
  if (window.hideOrb) window.hideOrb();
  // Pause prompts during drawing
  if (window._pauseAmbientPrompts) window._pauseAmbientPrompts('drawing');
  if (window['onVisualisationStart']) { window['onVisualisationStart'](); window['onVisualisationStart'] = null; }
  clearInterval(window._revealTimer);
  if (window.startVisualRecording) window.startVisualRecording(playbackAudio);
  allTrailPoints = [];

  // Origins: sparse (low complexity) → dense (high complexity)
  const _isMobile = windowWidth < 640;
  let numOrigins = int(lerp(2, 14, complexityNorm));
  if (_isMobile) numOrigins = Math.max(1, int(numOrigins * 0.12));
  let origins = [];
  randomSeed(99);
  for (let o=0; o<numOrigins; o++) {
    if (o === 0) origins.push({x: SZ_W*random(0.2,0.5), y: SZ_H*random(0.3,0.7)});
    else         origins.push({x: SZ_W*random(0.05,0.95), y: SZ_H*random(0.05,0.95)});
  }
  randomSeed(42);

  let particlesPerOrigin = int(lerp(4, 12, complexityNorm));
  if (_isMobile) particlesPerOrigin = Math.max(2, int(particlesPerOrigin * 0.12));
  let allPaths = [];
  let seedCounter = 0;

  for (let o=0; o<origins.length; o++) {
    let ox = origins[o].x, oy = origins[o].y;
    for (let i=0; i<particlesPerOrigin; i++) {
      let startAngle = random(TWO_PI);
      randomSeed(42+seedCounter*31); noiseSeed(42+seedCounter*31);
      let path = buildParticlePath(ox,oy,startAngle,baseWeight,
                   sdNorm,sigNorm,rmsAlpha,rmsNorm,pitchNorm,entropyNorm,1.0,0);
      if (path.length > 0) {
        allPaths.push(path);
        allTrailPoints.push(path.filter((_,idx)=>idx%5===0).map(p=>({x:p.nx,y:p.ny})));
      }
      seedCounter++;
    }
    noStroke();
    fill(40,40,45,180); ellipse(ox,oy,4,4);
    fill(40,40,45,40);  ellipse(ox,oy,10,10);
  }

  const _targetDur = (window._lastRecordingDuration || 30) * 1000;

  if (window._sessionActive) {
    // Guided session: draw segments sequentially (one path completes before the next),
    // timed to fill the full target duration so there is always movement to trace.
    const allSegs = [];
    for (const p of allPaths) {
      for (let s = 0; s < p.length - 1; s++) allSegs.push(p[s]);
    }
    // Raise minimum tick to 180ms for calm pace, even for complex drawings.
    // segsPerTick scales up with complexity so the full duration is always filled.
    const MIN_TICK_MS = 180;
    const _tickMs_s = Math.max(MIN_TICK_MS, Math.floor(_targetDur / Math.max(allSegs.length, 1)));
    const _segsPerTick = Math.max(1, Math.round(allSegs.length / (_targetDur / _tickMs_s)));
    let si = 0;
    window._revealTimer = setInterval(() => {
      if (si >= allSegs.length) {
        clearInterval(window._revealTimer);
        drawCrossingEffects();
        // Resume prompts after drawing ends
        if (window._resumeAmbientPrompts) window._resumeAmbientPrompts('drawing');
        return;
      }
      const end = Math.min(allSegs.length, si + _segsPerTick);
      for (; si < end; si++) {
        const a = allSegs[si];
        if (!a) continue;
        drawBurningLine(a.x, a.y, a.nx, a.ny, a.r, a.g, a.b, a.alpha, a.wt);
        if (a.tn > 0.72 && Math.random() < 0.18) {
          stroke(Math.min(255,a.r+60),Math.min(255,a.g+80),Math.min(255,a.b+40),a.alpha*0.10);
          strokeWeight(a.wt*6.0); line(a.x, a.y, a.nx, a.ny);
        }
      }
    }, _tickMs_s);

  } else {
    // Self-guided: existing simultaneous drawing logic, unchanged.
    let maxSteps = allPaths.length > 0 ? Math.max(...allPaths.map(p=>p.length)) : 0;
    let step = 0;
    const _tickMs = 16;
    const _stepsPerTick = maxSteps / (_targetDur / _tickMs);
    let _stepAccum = 0;
    window._revealTimer = setInterval(() => {
      if (step >= maxSteps) {
        clearInterval(window._revealTimer);
        drawCrossingEffects();
        // Resume prompts after drawing ends
        if (window._resumeAmbientPrompts) window._resumeAmbientPrompts('drawing');
        return;
      }
      _stepAccum += _stepsPerTick;
      const nextStep = Math.min(maxSteps, Math.floor(_stepAccum));
      for (let s = step; s < nextStep; s++) {
        for (let p of allPaths) {
          if (s >= p.length - 1) continue;
          let a=p[s], b=p[s+1];
          if (!a||!b) continue;
          drawBurningLine(a.x,a.y,b.x,b.y,a.r,a.g,a.b,a.alpha,a.wt);
          if (a.tn > 0.72 && Math.random() < 0.18) {
            stroke(Math.min(255,a.r+60),Math.min(255,a.g+80),Math.min(255,a.b+40),a.alpha*0.10);
            strokeWeight(a.wt*6.0); line(a.x,a.y,b.x,b.y);
          }
        }
      }
      step = nextStep;
    }, _tickMs);
  }
}

// ─── File analysis ────────────────────────────────────────────────────────────

async function analyseFileOffline(file) {
  // Stop any running session cleanly
  clearInterval(window._revealTimer);
  clearTimeout(window._vizStopTimer);
  if (playbackAudio) { playbackAudio.pause(); playbackAudio = null; }
  playbackActive = false;
  inputSource = null;
  detectedCategories = [];
  const _mySession = _sessionId;
  if (window.updateSoundmarks) window.updateSoundmarks([]);

  const overlay = document.getElementById('info-overlay');
  if (overlay) overlay.classList.add('hidden');
  if (window.hideOrb) window.hideOrb();
  state = 'idle'; statusMsg = 'ANALYSING  ' + file.name;
  drawBackground(-1);

  // Hide "how long do you have" modal if present
  const howlongModal = document.getElementById('howlong-modal');
  if (howlongModal) howlongModal.classList.add('hidden');

  try {
    let formData = new FormData();
    formData.append('file', file);
    let response = await fetch('/analyse', { method:'POST', body:formData });
    if (!response.ok) { let e=await response.json(); throw new Error(e.error||'Server error '+response.status); }
    let data = await response.json();

    detectedCategories = (data && Array.isArray(data.detected_categories)) ? data.detected_categories : [];

    let m = (data&&data.measures) ? data.measures : {};
    let p = (data&&data.params)   ? data.params   : {};
    mEntropy  = (typeof m.mean_entropy==='number') ? m.mean_entropy : 9.0;
    mEntropySD= (typeof m.std_entropy ==='number') ? m.std_entropy  : 0.5;
    mRMS      = (typeof m.mean_rms    ==='number') ? m.mean_rms     : 0.02;
    mKeynote  = (typeof m.keynote_hz  ==='number') ? m.keynote_hz   : 400;
    mSignal   = (typeof m.signal      ==='number') ? m.signal       : 0.05;

    dispMeanEntropy = (typeof m.mean_entropy==='number') ? m.mean_entropy.toFixed(4) : '\u2014';
    dispStdEntropy  = (typeof m.std_entropy ==='number') ? m.std_entropy.toFixed(4)  : '\u2014';
    dispRMS         = (typeof m.mean_rms    ==='number') ? m.mean_rms.toFixed(4)     : '\u2014';
    dispKeynote     = (typeof m.keynote_hz  ==='number') ? m.keynote_hz.toFixed(1)+' Hz' : '\u2014';
    dispSignal      = (typeof m.signal      ==='number') ? m.signal.toFixed(4)       : '\u2014';

    if (window.updateDetailsBox)
      window.updateDetailsBox(dispMeanEntropy, dispStdEntropy, dispRMS, dispKeynote, dispSignal,
        (typeof m.mean_entropy==='number' && typeof m.std_entropy==='number')
          ? constrain((map(m.mean_entropy,6.7,8.3,0,1)+map(m.std_entropy,0.10,0.45,0,1))*0.5, 0, 1)
          : null);
    if (window.updateSoundmarks)
      window.updateSoundmarks(detectedCategories);

    vParams.numOrigins = (typeof p.numOrigins==='number') ? p.numOrigins : 8;
    vParams.lineWeight = (typeof p.lineWeight==='number') ? p.lineWeight : 1.2;

    let segStart = (typeof data.segment_start   ==='number') ? data.segment_start    : 0;
    let segDur   = (typeof data.segment_duration==='number') ? data.segment_duration : 30;

    inputSource = 'file';

    // Set up audio BEFORE startVisualising so the recorder captures both canvas + audio together
    if (playbackAudio) { playbackAudio.pause(); playbackAudio = null; }
    const _url = URL.createObjectURL(file);
    playbackAudio = new Audio(_url);
    if (window.soundMuted) playbackAudio.muted = true;

    // Abort if reset happened while fetch was in flight
    if (_sessionId !== _mySession) return;
    window._lastRecordingDuration = segDur;
    startVisualising();

    // Fade out after segment ends
    clearTimeout(window._vizStopTimer);
    window._vizStopTimer = setTimeout(() => {
      if (state === 'visualising') { state = 'fading'; fadeAlpha = 0; }
    }, segDur * 1000);

    // Begin audio playback once ready
    playbackAudio.addEventListener('canplay', () => {
      if (!playbackAudio) return;
      playbackAudio.currentTime = segStart;
      playbackAudio.play().catch(e => console.warn('Playback error:', e));
      setTimeout(() => {
        if (playbackAudio) { playbackAudio.pause(); URL.revokeObjectURL(_url); playbackAudio = null; }
      }, segDur * 1000);
    }, { once: true });

  } catch (err) {
    console.error('[analyseFileOffline] Server error:', err.message);
    if (_sessionId !== _mySession) return;
    inputSource = 'file';
    startVisualising();
  }
}

// ─── Mic recording ────────────────────────────────────────────────────────────

function finishMicAnalysis() {
  if (mic)  { mic.stop(); }
  if (fft)  { fft.setInput(); }
  inputSource = null;

  // Use accumulated live FFT data if available, otherwise keep current values
  if (liveFrames > 0) {
    let meanE = liveSumE/liveFrames, meanB = liveSumBass/liveFrames, meanT = liveSumTreble/liveFrames;
    mRMS      = meanE/255*0.15;
    mEntropySD= constrain(map(liveSumFlux/liveFrames, 0,20, 0.1,0.6), 0.1, 0.6);
    mEntropy  = constrain(map(meanE,  0,255, 6.7,8.3), 6.7, 8.3);
    mKeynote  = constrain(map(meanT,  0,255, 71,2035), 71, 2035);
    mSignal   = constrain(map(meanB,  0,255, 0.0,0.545), 0, 0.545);
  }
  // else: keep previous mEntropy etc. — server response in onstop will override

  dispRMS          = nf(mRMS,1,4);
  dispSignal       = nf(mSignal,1,4);
  dispMeanEntropy  = nf(mEntropy,1,4);
  dispStdEntropy   = nf(mEntropySD,1,4);
  dispKeynote      = nf(mKeynote,1,1)+' Hz';

  vParams.numOrigins = liveFrames>0 ? int(map(liveSumE/liveFrames, 0,255, 4,32)) : vParams.numOrigins;
  vParams.lineWeight = 1.2;

  window._lastRecordingDuration = liveFrames > 0 ? liveFrames / 60 : 30;

  if (window.setOrbRecording) window.setOrbRecording(false);

  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  // onstop callback will call startVisualising() once audio is processed
}

function toggleMic() {
  // Lazy-init FFT here — requires AudioContext which needs a prior user gesture
  if (!fft) fft = new p5.FFT(0.8, 1024);

  if (inputSource === 'mic') {
    // Already recording → stop
    finishMicAnalysis();
    return;
  }

  // Stop any running visualisation before starting a new recording
  clearInterval(window._revealTimer);
  clearTimeout(window._vizStopTimer);
  if (playbackAudio) { playbackAudio.pause(); playbackAudio = null; }
  playbackActive = false;
  state = 'idle';
  drawBackground(-1);

  if (!mic) mic = new p5.AudioIn();
  mic.start(async () => {
    if (!fft) fft = new p5.FFT(0.8, 1024);
    fft.setInput(mic);
    inputSource = 'mic';
    liveSumE=0; liveSumBass=0; liveSumTreble=0; liveSumFlux=0; liveFrames=0;
    lastSpectrum=[]; detectedCategories=[];
    state = 'analysing'; statusMsg = 'RECORDING  0%';
    if (window.setOrbRecording) window.setOrbRecording(true);

    let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };

    mediaRecorder.onstop = async () => {
      if (window._suppressVisualization) { window._suppressVisualization = false; return; }
      const _mySession = _sessionId;
      const rawBlob = new Blob(recordedChunks, { type: 'audio/webm' });

      // ── Start playback + visualisation immediately with live metrics ──
      if (playbackAudio) { playbackAudio.pause(); playbackAudio = null; }
      const _micURL = URL.createObjectURL(rawBlob);
      playbackAudio = new Audio(_micURL);
      if (window.soundMuted) playbackAudio.muted = true;
      playbackActive = true;
      stopOnUserInteraction = false;
      playbackAudio.play().catch(e => console.warn('Mic playback error:', e));

      if (_sessionId !== _mySession) return;
      inputSource = 'mic';
      startVisualising();

      function _onVizDone() {
        clearTimeout(window._vizStopTimer);
        if (playbackAudio) playbackAudio.removeEventListener('ended', _onVizDone);
        if (state !== 'visualising') return;
        if (window.onVisualisationComplete) {
          const cb = window.onVisualisationComplete;
          window.onVisualisationComplete = null;
          cb();
        } else {
          if (window._pauseAmbientPrompts) window._pauseAmbientPrompts('drawing');
          if (window.setPlaybackActive) window.setPlaybackActive(false);
          state = 'fading'; fadeAlpha = 0;
        }
      }
      if (playbackAudio) playbackAudio.addEventListener('ended', _onVizDone, { once: true });
      const _recDur = window._lastRecordingDuration || 30;
      clearTimeout(window._vizStopTimer);
      window._vizStopTimer = setTimeout(_onVizDone, (_recDur + 1.5) * 1000);

      // ── Background: encode + send to server for soundmarks/refined measures ──
      (async () => {
        try {
          let arrayBuf = await rawBlob.arrayBuffer();
          let audioCtx = new AudioContext({ sampleRate: 16000 });
          let decoded  = await Promise.race([
            audioCtx.decodeAudioData(arrayBuf),
            new Promise((_,reject) => setTimeout(() => reject(new Error('decode timeout')), 2000))
          ]);
          audioCtx.close();
          let pcm    = decoded.getChannelData(0);
          let wavBuf = new ArrayBuffer(44 + pcm.length * 2);
          let view   = new DataView(wavBuf);
          const ws   = (o,s) => { for(let i=0;i<s.length;i++) view.setUint8(o+i,s.charCodeAt(i)); };
          ws(0,'RIFF'); view.setUint32(4,36+pcm.length*2,true);
          ws(8,'WAVE'); ws(12,'fmt '); view.setUint32(16,16,true);
          view.setUint16(20,1,true); view.setUint16(22,1,true);
          view.setUint32(24,16000,true); view.setUint32(28,32000,true);
          view.setUint16(32,2,true); view.setUint16(34,16,true);
          ws(36,'data'); view.setUint32(40,pcm.length*2,true);
          let off = 44;
          for (let i=0; i<pcm.length; i++, off+=2) {
            let s = Math.max(-1,Math.min(1,pcm[i]));
            view.setInt16(off, s<0 ? s*0x8000 : s*0x7FFF, true);
          }
          let wavBlob  = new Blob([wavBuf], { type:'audio/wav' });
          let formData = new FormData();
          formData.append('file', wavBlob, 'recording.wav');
          let response = await fetch('/analyse', { method:'POST', body:formData });
          let data     = await response.json();
          if (_sessionId !== _mySession) return;

          detectedCategories = (data&&Array.isArray(data.detected_categories)) ? data.detected_categories : [];
          let m = (data&&data.measures) ? data.measures : {};
          mEntropy  = (typeof m.mean_entropy==='number') ? m.mean_entropy : mEntropy;
          mEntropySD= (typeof m.std_entropy ==='number') ? m.std_entropy  : mEntropySD;
          mRMS      = (typeof m.mean_rms    ==='number') ? m.mean_rms     : mRMS;
          mKeynote  = (typeof m.keynote_hz  ==='number') ? m.keynote_hz   : mKeynote;
          mSignal   = (typeof m.signal      ==='number') ? m.signal       : mSignal;
          dispMeanEntropy = nf(mEntropy,1,4); dispStdEntropy = nf(mEntropySD,1,4);
          dispRMS = nf(mRMS,1,4); dispKeynote = nf(mKeynote,1,1)+' Hz'; dispSignal = nf(mSignal,1,4);
          if (window.updateDetailsBox)
            window.updateDetailsBox(dispMeanEntropy, dispStdEntropy, dispRMS, dispKeynote, dispSignal,
              (typeof m.mean_entropy==='number' && typeof m.std_entropy==='number')
              ? constrain((map(m.mean_entropy,6.7,8.3,0,1)+map(m.std_entropy,0.10,0.45,0,1))*0.5, 0, 1)
              : null);
          if (window.updateSoundmarks) window.updateSoundmarks(detectedCategories);
        } catch (err) {
          console.error('[server analysis] error:', err.message);
        }
      })();
    };

    mediaRecorder.start();

  }, () => { statusMsg = 'MICROPHONE ACCESS DENIED'; });
}


