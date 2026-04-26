"""
USV Digital Twin — Replay Visualization Server
==============================================
Servidor de visualização pós-processamento.

Arquitetura:
- benchmark_engine: única fonte de verdade física
- replay_exporter: persistência temporal
- visualization_player: reprodução para frontend

Este servidor NÃO roda PhysicsEngine e NÃO integra dinâmica em runtime.
"""

import os
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

from geometry_engine import GeometryEngine
from visualization_player import PlayerLoop, VisualizationPlayer


app = Flask(__name__)
app.config['SECRET_KEY'] = 'usv_digital_twin'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

VIEW = {
    'replay_dir': os.path.join(os.getcwd(), 'training_runs', 'replays'),
    'status': 'idle',
}

geo = GeometryEngine(L=0.8, D=0.1)
player = VisualizationPlayer(VIEW['replay_dir'])
player_loop = None


def _emit_replay_state(payload):
    socketio.emit('state', payload)


def _safe_payload(data):
    return data if isinstance(data, dict) else {}


def _catalog_payload():
    trials = player.list_trials()
    history = []
    for item in trials:
        history.append({
            'saved_at_epoch_s': 0,
            'label': item['run_id'],
            'benchmark_mode': item['benchmark_mode'],
            'scenario': {
                'trials': item['trial'],
                'max_steps': item['frame_count'],
                'dt': None,
            },
            'controllers': {
                item['controller']: {
                    'score': 0.0,
                    'success_rate': 0.0,
                    'collision_rate': 0.0,
                    'mean_tracking_error_m': 0.0,
                    'mean_final_error_m': 0.0,
                    'mean_attitude_error_deg': 0.0,
                    'mean_final_attitude_error_deg': 0.0,
                    'mean_time_s': item['duration_s'],
                    'mean_energy_score': 0.0,
                    'mean_compute_ms': 0.0,
                }
            },
            'replay': item,
        })
    return {
        'trials': trials,
        'history': history,
    }


def _initial_load_latest():
    catalog = player.list_trials()
    if not catalog:
        VIEW['status'] = 'empty'
        return
    latest = catalog[-1]['run_id']
    if player.load_primary(latest):
        player.select_trials([latest])
        VIEW['status'] = 'ready'


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@socketio.on('connect')
def on_connect():
    emit('ready', {'mesh': geo.to_dict()['mesh']})
    emit('benchmark_status', {
        'status': VIEW['status'],
        'running': False,
    })
    payload = _catalog_payload()
    emit('replay_catalog', payload)
    emit('benchmark_history', {'history': payload['history']})
    emit('replay_status', player.status())


@socketio.on('start')
def on_start():
    player.play()
    emit('status', {'running': True, 'mode': 'replay'})


@socketio.on('pause')
def on_pause():
    playing = player.toggle_pause()
    emit('status', {'paused': (not playing), 'mode': 'replay'})


@socketio.on('reset')
def on_reset():
    player.reset()
    emit('status', {'reset': True, 'mode': 'replay'})


@socketio.on('set_replay_speed')
def on_set_replay_speed(data):
    payload = _safe_payload(data)
    speed = player.set_speed(payload.get('speed', 1.0))
    emit('replay_status', player.status())
    emit('status', {'replay_speed': speed})


@socketio.on('seek_replay')
def on_seek_replay(data):
    payload = _safe_payload(data)
    if 'ratio' in payload:
        playhead = player.seek_ratio(float(payload.get('ratio', 0.0)))
    else:
        playhead = player.seek_time(float(payload.get('time_s', 0.0)))
    emit('replay_status', player.status())
    emit('status', {'replay_seek_s': playhead})


@socketio.on('load_replay')
def on_load_replay(data):
    payload = _safe_payload(data)
    run_id = str(payload.get('run_id', '')).strip()
    if not run_id:
        emit('benchmark_error', {'message': 'run_id inválido para replay.'})
        return
    if not player.load_primary(run_id):
        emit('benchmark_error', {'message': f'Replay não encontrado: {run_id}'})
        return
    player.select_trials([run_id])
    emit('replay_status', player.status())
    emit('status', {'replay_loaded': run_id})


@socketio.on('select_replay_trials')
def on_select_replay_trials(data):
    payload = _safe_payload(data)
    run_ids = payload.get('run_ids', [])
    if not isinstance(run_ids, list):
        run_ids = []
    selected = player.select_trials([str(rid) for rid in run_ids])
    emit('replay_status', player.status())
    emit('status', {'selected_replay_trials': selected})


@socketio.on('refresh_replays')
def on_refresh_replays():
    player.refresh_catalog()
    payload = _catalog_payload()
    emit('replay_catalog', payload)
    emit('benchmark_history', {'history': payload['history']})
    emit('replay_status', player.status())


@socketio.on('run_benchmark')
def on_run_benchmark(_data):
    player.refresh_catalog()
    payload = _catalog_payload()
    emit('benchmark_error', {
        'message': 'Benchmark em runtime foi desativado no visualization_server. Execute benchmark_engine para gerar novos replays.'
    })
    emit('replay_catalog', payload)
    emit('benchmark_history', {'history': payload['history']})


@socketio.on('set_controller')
@socketio.on('set_waypoint')
@socketio.on('add_waypoint')
@socketio.on('set_waypoints')
@socketio.on('update_lqr_weights')
@socketio.on('add_obstacle')
@socketio.on('clear_obstacles')
@socketio.on('noise_scale')
@socketio.on('environmental_disturbance')
def on_mutation_blocked(*_args, **_kwargs):
    emit('status', {
        'warning': 'Mutação de dinâmica está desativada no replay viewer.',
        'mode': 'replay',
    })


# ─────────────────────────────────────────────
# HTML + THREE.JS (template inline)
# ─────────────────────────────────────────────

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>USV Digital Twin</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #020d14;
    --panel:     rgba(0, 255, 200, 0.04);
    --border:    rgba(0, 255, 200, 0.15);
    --accent:    #00ffc8;
    --accent2:   #00a8ff;
    --warn:      #ff6b35;
    --danger:    #ff2255;
    --text:      #c8ffe8;
    --dim:       rgba(200, 255, 232, 0.4);
    --font-mono: 'Share Tech Mono', monospace;
    --font-hud:  'Orbitron', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    overflow: hidden;
    height: 100vh;
    width: 100vw;
  }

  #canvas-container {
    position: absolute;
    inset: 0;
  }

  canvas { display: block; }

  /* ── HUD overlay ── */
  #hud {
    position: absolute;
    inset: 0;
    pointer-events: none;
    display: grid;
    grid-template-columns: 280px 1fr 280px;
    grid-template-rows: 1fr auto;
    gap: 12px;
    padding: 16px;
  }

  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px;
    backdrop-filter: blur(8px);
    pointer-events: all;
  }

  .panel-title {
    font-family: var(--font-hud);
    font-size: 9px;
    letter-spacing: 3px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }

  /* ── telemetria ── */
  #panel-telem {
    grid-column: 1;
    grid-row: 1;
    align-self: start;
  }

  .telem-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    border-bottom: 1px solid rgba(0,255,200,0.05);
  }

  .telem-label { font-size: 10px; color: var(--dim); }

  .telem-value {
    font-family: var(--font-hud);
    font-size: 13px;
    color: var(--accent);
    letter-spacing: 1px;
  }

  .telem-value.warn  { color: var(--warn); }
  .telem-value.danger { color: var(--danger); }

  /* ── sonar radar ── */
  #panel-sonar {
    grid-column: 1;
    grid-row: 1;
    align-self: end;
    margin-top: 320px;
  }

  #sonar-canvas {
    display: block;
    margin: 0 auto;
    border-radius: 50%;
    border: 1px solid var(--border);
  }

  /* ── controles ── */
  #panel-controls {
    grid-column: 3;
    grid-row: 1;
    align-self: start;
  }

  .btn-group {
    display: flex;
    gap: 6px;
    margin-bottom: 10px;
    flex-wrap: wrap;
  }

  .btn {
    font-family: var(--font-hud);
    font-size: 9px;
    letter-spacing: 2px;
    padding: 7px 12px;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--dim);
    cursor: pointer;
    border-radius: 2px;
    transition: all 0.15s;
    text-transform: uppercase;
  }

  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn.active { background: rgba(0,255,200,0.1); border-color: var(--accent); color: var(--accent); }
  .btn.danger-btn { border-color: var(--danger); color: var(--danger); }
  .btn.danger-btn:hover { background: rgba(255,34,85,0.1); }

  .slider-group { margin-bottom: 10px; }
  .slider-label {
    font-size: 9px;
    color: var(--dim);
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
  }

  input[type=range] {
    width: 100%;
    -webkit-appearance: none;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    outline: none;
  }

  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px; height: 12px;
    background: var(--accent);
    border-radius: 50%;
    cursor: pointer;
  }

  .mini-input {
    width: 100%;
    background: rgba(0, 20, 30, 0.7);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 7px 8px;
    border-radius: 2px;
    font-family: var(--font-mono);
    font-size: 10px;
  }

  .benchmark-status {
    font-size: 9px;
    color: var(--dim);
    margin-bottom: 8px;
    line-height: 1.4;
  }

  .benchmark-board {
    display: grid;
    gap: 6px;
    margin-top: 10px;
  }

  .benchmark-card {
    border: 1px solid var(--border);
    background: rgba(0, 255, 200, 0.03);
    padding: 8px;
    border-radius: 2px;
  }

  .benchmark-card strong {
    font-family: var(--font-hud);
    font-size: 10px;
    letter-spacing: 1px;
    color: var(--accent);
  }

  .benchmark-card .meta {
    margin-top: 4px;
    font-size: 9px;
    color: var(--dim);
    line-height: 1.45;
  }

  /* ── bottom bar ── */
  #bottom-bar {
    grid-column: 1 / -1;
    grid-row: 2;
    display: flex;
    gap: 12px;
    align-items: center;
  }

  #panel-status {
    flex: 1;
    padding: 10px 14px;
  }

  .status-row {
    display: flex;
    gap: 24px;
    align-items: center;
  }

  .status-item {
    display: flex;
    gap: 6px;
    align-items: center;
    font-size: 10px;
  }

  .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent);
    animation: pulse 1.5s infinite;
  }

  .dot.offline { background: var(--danger); box-shadow: 0 0 6px var(--danger); animation: none; }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* ── crosshair center ── */
  #crosshair {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    opacity: 0.3;
  }

  .ch-h, .ch-v {
    position: absolute;
    background: var(--accent);
  }

  .ch-h { width: 20px; height: 1px; top: 50%; left: 50%; transform: translate(-50%, -50%); }
  .ch-v { width: 1px; height: 20px; top: 50%; left: 50%; transform: translate(-50%, -50%); }

  /* ── depth gauge ── */
  #depth-gauge {
    position: absolute;
    right: 295px;
    top: 50%;
    transform: translateY(-50%);
    width: 24px;
    height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    pointer-events: none;
  }

  .gauge-bar {
    width: 4px;
    height: 100%;
    background: var(--border);
    border-radius: 2px;
    position: relative;
    overflow: hidden;
  }

  #gauge-fill {
    position: absolute;
    bottom: 0;
    width: 100%;
    background: linear-gradient(to top, var(--accent2), var(--accent));
    border-radius: 2px;
    transition: height 0.3s ease;
  }

  .gauge-label {
    font-family: var(--font-hud);
    font-size: 8px;
    color: var(--dim);
    writing-mode: vertical-rl;
    letter-spacing: 2px;
  }

  /* ── scan line overlay ── */
  #scanlines {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0, 0, 0, 0.03) 2px,
      rgba(0, 0, 0, 0.03) 4px
    );
    z-index: 10;
  }

  /* ── waypoint instruction ── */
  #wp-hint {
    position: absolute;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%);
    font-family: var(--font-hud);
    font-size: 9px;
    letter-spacing: 3px;
    color: var(--accent);
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
    text-transform: uppercase;
  }

  #wp-hint.visible { opacity: 0.7; }
</style>
</head>
<body>

<div id="canvas-container"></div>
<div id="scanlines"></div>

<div id="hud">

  <!-- TELEMETRIA -->
  <div class="panel" id="panel-telem">
    <div class="panel-title">Telemetry</div>

    <div class="telem-row">
      <span class="telem-label">DEPTH</span>
      <span class="telem-value" id="t-depth">0.00 m</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">POSITION X</span>
      <span class="telem-value" id="t-x">0.00 m</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">POSITION Y</span>
      <span class="telem-value" id="t-y">0.00 m</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">SURGE VEL</span>
      <span class="telem-value" id="t-u">0.00 m/s</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">HEAVE VEL</span>
      <span class="telem-value" id="t-w">0.00 m/s</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">DENSITY</span>
      <span class="telem-value" id="t-rho">1000 kg/m³</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">ROLL</span>
      <span class="telem-value" id="t-phi">0.0°</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">PITCH</span>
      <span class="telem-value" id="t-tht">0.0°</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">THRUSTER</span>
      <span class="telem-value" id="t-thr">0%</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">BALLAST</span>
      <span class="telem-value" id="t-bal">0%</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">SIM TIME</span>
      <span class="telem-value" id="t-time">0.0 s</span>
    </div>
    <div class="telem-row">
      <span class="telem-label">CONTROLLER</span>
      <span class="telem-value" id="t-ctrl">—</span>
    </div>
  </div>

  <!-- SONAR DISPLAY -->
  <div class="panel" id="panel-sonar">
    <div class="panel-title">Sonar 360°</div>
    <canvas id="sonar-canvas" width="200" height="200"></canvas>
  </div>

  <!-- CONTROLES -->
  <div class="panel" id="panel-controls">
    <div class="panel-title">Mission Control</div>

    <div class="btn-group">
      <button class="btn active" id="btn-start" onclick="startSim()">Start</button>
      <button class="btn" id="btn-pause" onclick="pauseSim()">Pause</button>
      <button class="btn danger-btn" onclick="resetSim()">Reset</button>
    </div>

    <div class="panel-title" style="margin-top:12px">Controller</div>
    <div class="btn-group">
      <button class="btn active" id="ctrl-lqr" onclick="setCtrl('lqr')">LQR</button>
      <button class="btn" id="ctrl-mpc" onclick="setCtrl('mpc')">MPC</button>
      <button class="btn" id="ctrl-rl"  onclick="setCtrl('rl')">RL</button>
    </div>

    <div class="panel-title" style="margin-top:12px">Camera</div>
    <div class="btn-group">
      <button class="btn active" id="cam-follow" onclick="setCamera('follow')">Follow</button>
      <button class="btn" id="cam-free"   onclick="setCamera('free')">Free</button>
      <button class="btn" id="cam-top"    onclick="setCamera('top')">Top</button>
    </div>

    <div class="panel-title" style="margin-top:12px">LQR Weights</div>

    <div class="slider-group">
      <div class="slider-label"><span>q_z (depth)</span><span id="qz-val">10</span></div>
      <input type="range" min="1" max="100" value="10" id="sl-qz"
             oninput="updateWeight('q_z', this.value, 'qz-val')">
    </div>

    <div class="slider-group">
      <div class="slider-label"><span>q_phi (roll)</span><span id="qphi-val">5</span></div>
      <input type="range" min="1" max="50" value="5" id="sl-qphi"
             oninput="updateWeight('q_phi', this.value, 'qphi-val')">
    </div>

    <div class="slider-group">
      <div class="slider-label"><span>r_thrust</span><span id="rthr-val">1</span></div>
      <input type="range" min="1" max="20" value="1" id="sl-rthr"
             oninput="updateWeight('r_thrust_power', this.value, 'rthr-val')">
    </div>

    <div class="slider-group">
      <div class="slider-label"><span>Sensor Noise</span><span id="noise-val">0.5</span></div>
      <input type="range" min="0" max="20" value="5" id="sl-noise"
             oninput="updateNoise(this.value)">
    </div>

    <div class="panel-title" style="margin-top:12px">Environment</div>
    <div class="btn-group">
      <button class="btn" onclick="addObstacleMode()">Add Obstacle</button>
      <button class="btn danger-btn" onclick="clearObstacles()">Clear All</button>
    </div>

    <div class="panel-title" style="margin-top:12px">Waypoints</div>
    <div class="btn-group">
      <button class="btn" onclick="wpMode()">Click to Set</button>
      <button class="btn danger-btn" onclick="clearWaypoints()">Clear</button>
    </div>

    <div class="panel-title" style="margin-top:12px">Replay</div>
    <div class="benchmark-status" id="benchmark-status">Carregue um replay gerado pelo benchmark_engine.</div>

    <div class="btn-group">
      <button class="btn" id="btn-refresh-replays" onclick="refreshReplays()">Refresh Logs</button>
    </div>

    <div class="slider-group">
      <div class="slider-label"><span>Trial</span><span id="replay-run-label">none</span></div>
      <select id="replay-run-select" class="mini-input" onchange="loadSelectedReplay()"></select>
    </div>

    <div class="slider-group">
      <div class="slider-label"><span>Timeline</span><span id="replay-time">0.0 / 0.0 s</span></div>
      <input type="range" min="0" max="1000" value="0" id="replay-scrub" oninput="seekReplay(this.value)">
    </div>

    <div class="panel-title" style="margin-top:10px">Playback Speed</div>
    <div class="btn-group">
      <button class="btn active" id="spd-1x" onclick="setReplaySpeed(1)">1x</button>
      <button class="btn" id="spd-2x" onclick="setReplaySpeed(2)">2x</button>
      <button class="btn" id="spd-5x" onclick="setReplaySpeed(5)">5x</button>
    </div>

    <div class="slider-group">
      <div class="slider-label"><span>Compare Trials</span><span id="selected-trials-count">1</span></div>
      <select id="replay-compare-select" class="mini-input" multiple size="4" onchange="selectReplayTrials()"></select>
    </div>

    <div class="benchmark-board" id="benchmark-results"></div>
    <div class="benchmark-status" style="margin-top:10px">Replay catalog</div>
    <div class="benchmark-board" id="benchmark-history"></div>
  </div>

  <!-- STATUS BAR -->
  <div id="bottom-bar">
    <div class="panel panel-status">
      <div class="status-row">
        <div class="status-item">
          <div class="dot" id="conn-dot"></div>
          <span id="conn-status">CONNECTING</span>
        </div>
        <div class="status-item">
          <span style="color:var(--dim)">FPS</span>
          <span id="fps-display" style="color:var(--accent);font-family:var(--font-hud);font-size:11px">—</span>
        </div>
        <div class="status-item">
          <span style="color:var(--dim)">WAYPOINTS</span>
          <span id="wp-count" style="color:var(--accent);font-family:var(--font-hud);font-size:11px">0</span>
        </div>
        <div class="status-item">
          <span style="color:var(--dim)">OBSTACLES</span>
          <span id="obs-count" style="color:var(--accent);font-family:var(--font-hud);font-size:11px">0</span>
        </div>
        <div class="status-item" style="margin-left:auto">
          <span style="color:var(--dim);font-size:9px;letter-spacing:2px">USV DIGITAL TWIN v1.0</span>
        </div>
      </div>
    </div>
  </div>

</div>

<!-- DEPTH GAUGE -->
<div id="depth-gauge">
  <span class="gauge-label">DEPTH</span>
  <div class="gauge-bar">
    <div id="gauge-fill" style="height:0%"></div>
  </div>
</div>

<!-- CROSSHAIR -->
<div id="crosshair">
  <div class="ch-h"></div>
  <div class="ch-v"></div>
</div>

<!-- WAYPOINT HINT -->
<div id="wp-hint">Click scene to place waypoint</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
<script>
// ─── Three.js Setup ───────────────────────────────────────────
const W = window.innerWidth, H = window.innerHeight;
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(W, H);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.getElementById('canvas-container').appendChild(renderer.domElement);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, W/H, 0.1, 500);
camera.position.set(-10, 8, -10);
camera.lookAt(0, 5, 0);

// ─── Iluminação submarina ─────────────────────────────────────
scene.fog = new THREE.FogExp2(0x001a2e, 0.04);

// luz ambiente fraca — penumbra submarina
const ambLight = new THREE.AmbientLight(0x0a2040, 0.8);
scene.add(ambLight);

// luz direcional suave — "luz solar filtrada"
const sunLight = new THREE.DirectionalLight(0x4488ff, 0.6);
sunLight.position.set(10, 20, 10);
sunLight.castShadow = true;
scene.add(sunLight);

// point lights verdes — bioluminescência
const bioLight1 = new THREE.PointLight(0x00ffc8, 0.8, 15);
bioLight1.position.set(0, 3, 0);
scene.add(bioLight1);

const bioLight2 = new THREE.PointLight(0x00a8ff, 0.5, 20);
bioLight2.position.set(5, 5, 5);
scene.add(bioLight2);

// ─── Ambiente — pool subaquático ─────────────────────────────
// fundo
const floorGeo  = new THREE.PlaneGeometry(60, 60, 30, 30);
const floorMat  = new THREE.MeshStandardMaterial({
  color: 0x001428,
  roughness: 0.8,
  metalness: 0.1,
  wireframe: false,
});
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI/2;
floor.position.y = 0;
floor.receiveShadow = true;
scene.add(floor);

// grade de fundo
const gridHelper = new THREE.GridHelper(60, 30, 0x00ffc820, 0x00ffc808);
gridHelper.position.y = 0.01;
scene.add(gridHelper);

// superfície da água — semi-transparente
const waterGeo = new THREE.PlaneGeometry(60, 60);
const waterMat = new THREE.MeshStandardMaterial({
  color: 0x001e3c,
  transparent: true,
  opacity: 0.3,
  roughness: 0.1,
  metalness: 0.5,
  side: THREE.DoubleSide,
});
const water = new THREE.Mesh(waterGeo, waterMat);
water.rotation.x = -Math.PI/2;
water.position.y = 10;
scene.add(water);

// partículas — plâncton / partículas em suspensão
const particleCount = 500;
const particleGeo   = new THREE.BufferGeometry();
const positions = new Float32Array(particleCount * 3);
for(let i = 0; i < particleCount; i++) {
  positions[i*3]   = (Math.random() - 0.5) * 40;
  positions[i*3+1] = Math.random() * 10;
  positions[i*3+2] = (Math.random() - 0.5) * 40;
}
particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const particleMat = new THREE.PointsMaterial({
  color: 0x00ffc8,
  size: 0.05,
  transparent: true,
  opacity: 0.4,
});
scene.add(new THREE.Points(particleGeo, particleMat));

// ─── Modelo do veículo USV ────────────────────────────────────
let vehicleMesh = null;
let thrusterArrow = null;
let velocityArrow = null;
let envelopeMesh = null;
let vehicleLight  = null;

function buildVehicleMesh(meshParams) {
  const group = new THREE.Group();
  const L = meshParams.L || 0.8;
  const D = meshParams.D || 0.1;
  const R = D/2;
  const L_vk   = meshParams.L_vk   || 0.25;
  const L_cyl  = meshParams.L_cyl  || 0.3;

  // material metálico
  const mat = new THREE.MeshStandardMaterial({
    color: 0x1a4060,
    roughness: 0.3,
    metalness: 0.8,
    emissive: 0x001020,
    emissiveIntensity: 0.2,
  });

  // corpo cilíndrico
  const cylGeo = new THREE.CylinderGeometry(R, R, L_cyl, 24);
  cylGeo.rotateZ(Math.PI/2);
  const cyl = new THREE.Mesh(cylGeo, mat);
  cyl.castShadow = true;
  group.add(cyl);

  // cone frontal (Von Kármán aproximado como cone)
  const coneGeoF = new THREE.ConeGeometry(R, L_vk, 24);
  coneGeoF.rotateZ(Math.PI/2);
  const coneF = new THREE.Mesh(coneGeoF, mat);
  coneF.position.x = (L_cyl + L_vk) / 2;
  coneF.castShadow = true;
  group.add(coneF);

  // cone traseiro
  const coneGeoB = new THREE.ConeGeometry(R, L_vk, 24);
  coneGeoB.rotateZ(-Math.PI/2);
  const coneB = new THREE.Mesh(coneGeoB, mat);
  coneB.position.x = -(L_cyl + L_vk) / 2;
  coneB.castShadow = true;
  group.add(coneB);

  // luz de status no veículo
  vehicleLight = new THREE.PointLight(0x00ffc8, 1.5, 3);
  group.add(vehicleLight);

  // seta do propulsor
  const arrowDir = new THREE.Vector3(1, 0, 0);
  thrusterArrow = new THREE.ArrowHelper(arrowDir, new THREE.Vector3(0, 0, 0), 0.3, 0xff6b35, 0.1, 0.06);
  group.add(thrusterArrow);

  // vetor de velocidade (body frame projetado no mundo)
  velocityArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 0.2, 0x00a8ff, 0.08, 0.05);
  group.add(velocityArrow);

  // envelope de Monte Carlo/comparação entre trials
  const envGeo = new THREE.BoxGeometry(1, 1, 1);
  const envMat = new THREE.MeshBasicMaterial({
    color: 0x00ffc8,
    wireframe: true,
    transparent: true,
    opacity: 0.25,
  });
  envelopeMesh = new THREE.Mesh(envGeo, envMat);
  envelopeMesh.visible = false;
  scene.add(envelopeMesh);

  return group;
}

// ─── Sonar rays ───────────────────────────────────────────────
const sonarLines = [];
const sonarDirs = [
  [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,-1], [0,0,1]
];
const sonarColors = [0x00ffc8, 0x00ffc8, 0x00a8ff, 0x00a8ff, 0xffcc00, 0xffcc00];

sonarDirs.forEach((dir, i) => {
  const mat = new THREE.LineBasicMaterial({
    color: sonarColors[i],
    transparent: true,
    opacity: 0.6,
  });
  const geo    = new THREE.BufferGeometry();
  const points = new Float32Array(6);
  geo.setAttribute('position', new THREE.BufferAttribute(points, 3));
  const line = new THREE.Line(geo, mat);
  scene.add(line);
  sonarLines.push(line);
});

// ─── Trajetória ribbon ────────────────────────────────────────
const maxTrajPoints = 500;
const trajPositions = new Float32Array(maxTrajPoints * 3);
const trajColors    = new Float32Array(maxTrajPoints * 3);
const trajGeo = new THREE.BufferGeometry();
trajGeo.setAttribute('position', new THREE.BufferAttribute(trajPositions, 3));
trajGeo.setAttribute('color',    new THREE.BufferAttribute(trajColors, 3));
trajGeo.setDrawRange(0, 0);
const trajMat  = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
const trajLine = new THREE.Line(trajGeo, trajMat);
scene.add(trajLine);

// ─── Waypoint markers ─────────────────────────────────────────
const waypointMeshes = [];

function createWaypointMarker(pos) {
  const geo  = new THREE.SphereGeometry(0.2, 8, 8);
  const mat  = new THREE.MeshStandardMaterial({
    color: 0x00ffc8,
    emissive: 0x00ffc8,
    emissiveIntensity: 0.8,
    transparent: true,
    opacity: 0.7,
  });
  const mesh = new THREE.Mesh(geo, mat);
  // NED → Three.js: x→x, y→z, z→-y (z down = depth)
  mesh.position.set(pos[0], -pos[2] + 10, pos[1]);

  // anel ao redor
  const ringGeo = new THREE.TorusGeometry(0.4, 0.02, 8, 32);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x00ffc8, transparent: true, opacity: 0.4 });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  mesh.add(ring);

  scene.add(mesh);
  waypointMeshes.push(mesh);
  return mesh;
}

function clearWaypointMarkers() {
  waypointMeshes.forEach(m => scene.remove(m));
  waypointMeshes.length = 0;
}

// ─── Obstáculos ───────────────────────────────────────────────
const obstacleMeshes = new Map();

function updateObstacleMesh(id, pos, radius) {
  if (!obstacleMeshes.has(id)) {
    const geo = new THREE.SphereGeometry(radius, 16, 16);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xff2255,
      emissive: 0xff2255,
      emissiveIntensity: 0.3,
      transparent: true,
      opacity: 0.6,
      wireframe: false,
    });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
    obstacleMeshes.set(id, mesh);
  }
  const mesh = obstacleMeshes.get(id);
  mesh.position.set(pos[0], -pos[2] + 10, pos[1]);
}

function clearObstacleMeshes() {
  obstacleMeshes.forEach(m => scene.remove(m));
  obstacleMeshes.clear();
}

// ─── Câmera ───────────────────────────────────────────────────
let cameraMode = 'follow';
let camTheta = -Math.PI/4, camPhi = Math.PI/4, camDist = 15;
let isDragging = false, lastMX = 0, lastMY = 0;

renderer.domElement.addEventListener('mousedown', e => { isDragging = true; lastMX = e.clientX; lastMY = e.clientY; });
renderer.domElement.addEventListener('mouseup',   () => isDragging = false);
renderer.domElement.addEventListener('mousemove', e => {
  if (!isDragging || cameraMode !== 'free') return;
  const dx = e.clientX - lastMX, dy = e.clientY - lastMY;
  camTheta -= dx * 0.005;
  camPhi    = Math.max(0.1, Math.min(Math.PI/2 - 0.1, camPhi - dy * 0.005));
  lastMX = e.clientX; lastMY = e.clientY;
});
renderer.domElement.addEventListener('wheel', e => {
  camDist = Math.max(3, Math.min(60, camDist + e.deltaY * 0.01));
});

// ─── WebSocket ────────────────────────────────────────────────
const socket = io();
let meshParams = null;
let lastState  = null;
let frameCount = 0, lastFpsTime = performance.now();
let wpPlacementMode = false, obsPlacementMode = false;
let benchmarkData = null;
let benchmarkHistory = [];

socket.on('connect', () => {
  document.getElementById('conn-dot').classList.remove('offline');
  document.getElementById('conn-status').textContent = 'ONLINE';
});

socket.on('disconnect', () => {
  document.getElementById('conn-dot').classList.add('offline');
  document.getElementById('conn-status').textContent = 'OFFLINE';
});

socket.on('ready', data => {
  meshParams = data.mesh;
  vehicleMesh = buildVehicleMesh(meshParams);
  scene.add(vehicleMesh);
  socket.emit('start');
});

socket.on('state', data => {
  lastState = data;
  updateHUD(data);
  updateScene(data);
  if (data?.replay) {
    const duration = Number(data.replay.duration_s || 0);
    const playhead = Number(data.replay.playhead_s || data.time || 0);
    const ratio = duration > 0 ? Math.max(0, Math.min(1, playhead / duration)) : 0;
    document.getElementById('replay-scrub').value = String(Math.round(ratio * 1000));
    document.getElementById('replay-time').textContent = `${playhead.toFixed(1)} / ${duration.toFixed(1)} s`;
    document.getElementById('replay-run-label').textContent = String(data.replay.run_id || 'none');
  }
});

socket.on('status', data => {
  if (data.waypoints) {
    clearWaypointMarkers();
    data.waypoints.forEach(wp => createWaypointMarker(wp));
    document.getElementById('wp-count').textContent = data.waypoints.length;
  }
  if (data.obstacles !== undefined) {
    document.getElementById('obs-count').textContent = data.obstacles;
  }
});

socket.on('benchmark_status', data => {
  const status = String(data?.status || 'idle');
  document.getElementById('benchmark-status').textContent =
    status === 'empty'
      ? 'Nenhum replay encontrado. Execute benchmark_engine para gerar logs.'
      : 'Replay viewer pronto.';
});

socket.on('benchmark_progress', data => {
  if (!data) return;
  document.getElementById('benchmark-status').textContent = String(data.message || 'Atualizando replay...');
});

socket.on('benchmark_complete', data => {
  benchmarkData = data;
  renderBenchmark(data);
});

socket.on('benchmark_history', data => {
  benchmarkHistory = Array.isArray(data.history) ? data.history : [];
  renderBenchmarkHistory(benchmarkHistory);
});

socket.on('benchmark_error', data => {
  document.getElementById('benchmark-status').textContent =
    String(data?.message || 'Erro no replay viewer.');
});

socket.on('replay_catalog', data => {
  const trials = Array.isArray(data?.trials) ? data.trials : [];
  updateReplaySelects(trials);
  document.getElementById('benchmark-status').textContent =
    trials.length ? `Replays disponíveis: ${trials.length}` : 'Nenhum replay disponível.';
});

socket.on('replay_status', data => {
  const replay = data || {};
  const duration = Number(replay.duration_s || 0);
  const playhead = Number(replay.playhead_s || 0);
  const ratio = duration > 0 ? Math.max(0, Math.min(1, playhead / duration)) : 0;
  document.getElementById('replay-scrub').value = String(Math.round(ratio * 1000));
  document.getElementById('replay-time').textContent = `${playhead.toFixed(1)} / ${duration.toFixed(1)} s`;
  document.getElementById('replay-run-label').textContent = String(replay.primary_run_id || 'none');
  const selected = Array.isArray(replay.selected_run_ids) ? replay.selected_run_ids : [];
  document.getElementById('selected-trials-count').textContent = String(selected.length || 0);
});

function refreshReplays() {
  socket.emit('refresh_replays');
}

function loadSelectedReplay() {
  const sel = document.getElementById('replay-run-select');
  const runId = String(sel.value || '');
  if (!runId) return;
  socket.emit('load_replay', { run_id: runId });
}

function setReplaySpeed(speed) {
  socket.emit('set_replay_speed', { speed });
  [1,2,5].forEach(v => {
    const btn = document.getElementById(`spd-${v}x`);
    if (btn) btn.classList.toggle('active', Number(v) === Number(speed));
  });
}

function seekReplay(sliderValue) {
  const ratio = Number(sliderValue) / 1000;
  socket.emit('seek_replay', { ratio });
}

function selectReplayTrials() {
  const sel = document.getElementById('replay-compare-select');
  const runIds = Array.from(sel.selectedOptions).map(opt => opt.value);
  socket.emit('select_replay_trials', { run_ids: runIds });
}

function updateReplaySelects(trials) {
  const primary = document.getElementById('replay-run-select');
  const compare = document.getElementById('replay-compare-select');
  primary.innerHTML = '';
  compare.innerHTML = '';
  trials.forEach(t => {
    const label = `${String(t.controller || 'ctrl').toUpperCase()} T${Number(t.trial || 0)} | ${String(t.run_id)}`;
    const o1 = document.createElement('option');
    o1.value = String(t.run_id || '');
    o1.textContent = label;
    primary.appendChild(o1);

    const o2 = document.createElement('option');
    o2.value = String(t.run_id || '');
    o2.textContent = label;
    compare.appendChild(o2);
  });
}

function _normalizeBenchmarkCategories(data) {
  if (data?.categories && typeof data.categories === 'object') {
    return data.categories;
  }
  return { benchmark: data || {} };
}

function _normalizeBenchmarkRanking(data) {
  if (Array.isArray(data?.ranking)) {
    return data.ranking;
  }
  if (data?.controllers && typeof data.controllers === 'object') {
    return Object.entries(data.controllers).map(([controller, metrics]) => ({
      controller,
      ...(metrics || {}),
    }));
  }
  return [];
}

function _formatBenchmarkItem(item) {
  return {
    controller: String(item.controller || 'unknown'),
    score: Number(item.score || 0),
    success_rate: Number(item.success_rate || 0),
    collision_rate: Number(item.collision_rate || 0),
    mean_tracking_error_m: Number(item.mean_tracking_error_m || 0),
    mean_final_error_m: Number(item.mean_final_error_m || 0),
    mean_attitude_error_deg: Number(item.mean_attitude_error_deg || 0),
    mean_final_attitude_error_deg: Number(item.mean_final_attitude_error_deg || 0),
    mean_time_s: Number(item.mean_time_s || 0),
    mean_energy_score: Number(item.mean_energy_score || 0),
    mean_compute_ms: Number(item.mean_compute_ms || 0),
  };
}

function _renderBenchmarkCards(title, payload) {
  const ranking = _normalizeBenchmarkRanking(payload).map(_formatBenchmarkItem);
  const scenario = payload?.scenario || {};
  if (!ranking.length) {
    return `
      <div class="benchmark-card">
        <strong>${title}</strong>
        <div class="meta">Sem dados nesta categoria.</div>
      </div>
    `;
  }

  const modeTag = scenario.benchmark_mode ? `modo ${String(scenario.benchmark_mode).toUpperCase()}` : '';
  const targetTag = Array.isArray(scenario.waypoints) && scenario.waypoints.length
    ? `waypoints ${scenario.waypoints.length}`
    : `hold ${Array.isArray(scenario.hold_position) ? scenario.hold_position.map(v => Number(v).toFixed(1)).join(', ') : 'n/a'}`;

  return `
    <div class="benchmark-card">
      <strong>${title}</strong>
      <div class="meta">
        ${modeTag} | trials ${scenario.trials ?? '—'} | ${targetTag}
      </div>
    </div>
    ${ranking.map((item, index) => `
      <div class="benchmark-card">
        <strong>#${index + 1} ${item.controller.toUpperCase()}</strong>
        <div class="meta">
          score ${item.score.toFixed(1)} | sucesso ${(item.success_rate * 100).toFixed(0)}% | colisão ${(item.collision_rate * 100).toFixed(0)}%<br>
          erro médio ${item.mean_tracking_error_m.toFixed(2)} m | erro final ${item.mean_final_error_m.toFixed(2)} m<br>
          atitude média ${item.mean_attitude_error_deg.toFixed(1)}° | atitude final ${item.mean_final_attitude_error_deg.toFixed(1)}°<br>
          tempo ${item.mean_time_s.toFixed(1)} s | energia ${item.mean_energy_score.toFixed(2)} | CPU ${item.mean_compute_ms.toFixed(2)} ms
        </div>
      </div>
    `).join('')}
  `;
}

function renderBenchmark(data) {
  const root = document.getElementById('benchmark-results');
  const categories = _normalizeBenchmarkCategories(data);
  const categoryNames = Object.keys(categories);
  if (!categoryNames.length) {
    root.innerHTML = '<div class="benchmark-card"><div class="meta">Sem dados de benchmark.</div></div>';
    return;
  }

  root.innerHTML = categoryNames.map(name =>
    _renderBenchmarkCards(name.toUpperCase(), categories[name])
  ).join('');
}

function renderBenchmarkHistory(history) {
  const root = document.getElementById('benchmark-history');
  if (!history.length) {
    root.innerHTML = '<div class="benchmark-card"><div class="meta">Sem histórico salvo ainda.</div></div>';
    return;
  }

  const items = history.slice().reverse();
  root.innerHTML = items.map((entry, idx) => {
    const when = new Date((entry.saved_at_epoch_s || 0) * 1000);
    const whenText = Number.isNaN(when.getTime()) ? 'sem data' : when.toLocaleString();
    const label = entry.label ? String(entry.label) : null;
    const categories = _normalizeBenchmarkCategories(entry);
    const summary = Object.entries(categories).map(([name, payload]) => {
      const scenario = payload?.scenario || {};
      const ranking = _normalizeBenchmarkRanking(payload).map(_formatBenchmarkItem);
      const rankingText = ranking.map(item =>
        `${item.controller.toUpperCase()}: score ${item.score.toFixed(1)}, sucesso ${(item.success_rate * 100).toFixed(0)}%, colisão ${(item.collision_rate * 100).toFixed(0)}%`
      ).join('<br>');
      const scenarioText = `trials ${scenario.trials ?? '—'} | max_steps ${scenario.max_steps ?? '—'} | dt ${scenario.dt ?? '—'}`;
      return `<strong>${String(name).toUpperCase()}</strong><br>${scenarioText}<br>${rankingText || 'Sem ranking salvo.'}`;
    }).join('<br><br>');
    return `
      <div class="benchmark-card">
        <strong>${label ? label.toUpperCase() : `LOG #${history.length - idx}`}</strong>
        <div class="meta">
          ${whenText}<br>
          ${summary || 'Sem ranking salvo.'}
        </div>
      </div>
    `;
  }).join('');
}

// ─── Atualização da cena ──────────────────────────────────────
function updateScene(data) {
  if (!vehicleMesh) return;

  // posição NED → Three.js (y-up, z-forward)
  // NED: x=Norte, y=Leste, z=Down
  // THREE: x=x, y=up (=-z_ned+offset), z=y_ned
  const [x, y, z] = data.position;
  vehicleMesh.position.set(x, -z + 10, y);

  // quaternion
  const [qw, qx, qy, qz] = data.quaternion;
  vehicleMesh.quaternion.set(qx, -qz, qy, qw);

  // propulsor arrow
  if (thrusterArrow) {
    const thr = data.thruster;
    const theta = thr.theta_deg * Math.PI/180;
    const phi   = thr.phi_deg   * Math.PI/180;
    const dir = new THREE.Vector3(
      Math.cos(theta),
      Math.sin(theta) * Math.sin(phi),
      Math.sin(theta) * Math.cos(phi),
    ).normalize();
    thrusterArrow.setDirection(dir);
    const p = Math.abs(thr.power);
    thrusterArrow.setLength(0.1 + p * 0.5, 0.1, 0.06);
  }

  // luz do veículo pulsa com thrust
  if (vehicleLight) {
    const p = Math.abs(data.thruster.power);
    vehicleLight.intensity = 0.5 + p * 2.0;
  }

  // vetor de velocidade
  if (velocityArrow) {
    const [u, v, w] = data.velocity_linear || [0, 0, 0];
    const speed = Math.sqrt(u*u + v*v + w*w);
    const [qwv, qxv, qyv, qzv] = data.quaternion;
    const q = new THREE.Quaternion(qxv, -qzv, qyv, qwv);
    const velWorld = new THREE.Vector3(u, -w, v).applyQuaternion(q);
    const dir = speed > 1e-4 ? velWorld.clone().normalize() : new THREE.Vector3(1, 0, 0);
    velocityArrow.setDirection(dir);
    velocityArrow.setLength(0.15 + Math.min(2.5, speed), 0.08, 0.04);
  }

  // sonar rays
  data.sonar.forEach((ray, i) => {
    if (i >= sonarLines.length) return;
    const line = sonarLines[i];
    const dir  = ray.direction;
    const dist = ray.hit ? ray.distance : 0.5;

    // transforma direção do body frame pra world
    const [qwv, qxv, qyv, qzv] = data.quaternion;
    const q  = new THREE.Quaternion(qxv, -qzv, qyv, qwv);
    const d3 = new THREE.Vector3(dir[0], -dir[2], dir[1]).applyQuaternion(q);

    const start = vehicleMesh.position.clone();
    const end   = start.clone().add(d3.multiplyScalar(dist));

    const pos = line.geometry.attributes.position;
    pos.setXYZ(0, start.x, start.y, start.z);
    pos.setXYZ(1, end.x,   end.y,   end.z);
    pos.needsUpdate = true;

    // opacidade por confiança
    line.material.opacity = ray.hit ? 0.3 + ray.confidence * 0.5 : 0.15;
  });

  // trajetória ribbon
  const traj = data.trajectory;
  let n = Math.min(traj.length, maxTrajPoints);
  for (let i = 0; i < n; i++) {
    const [tx, ty, tz] = traj[i];
    trajPositions[i*3]   = tx;
    trajPositions[i*3+1] = -tz + 10;
    trajPositions[i*3+2] = ty;

    // cor por índice — mais recente = mais brilhante
    const t = i / n;
    trajColors[i*3]   = t * 0;
    trajColors[i*3+1] = t * 1;
    trajColors[i*3+2] = t * 0.8;
  }
  trajGeo.attributes.position.needsUpdate = true;
  trajGeo.attributes.color.needsUpdate    = true;
  trajGeo.setDrawRange(0, n);

  // obstáculos dinâmicos
  const dynObs = data.dynamic_obstacles || [];
  dynObs.forEach((obs, i) => {
    updateObstacleMesh(i, obs.position, obs.radius);
  });

  // remove obstáculos extras
  if (obstacleMeshes.size > dynObs.length) {
    for (let i = dynObs.length; i < obstacleMeshes.size; i++) {
      const m = obstacleMeshes.get(i);
      if (m) scene.remove(m);
      obstacleMeshes.delete(i);
    }
  }

  // envelope de comparação entre múltiplos trials
  if (envelopeMesh) {
    const env = data.comparison_envelope;
    if (env && env.min && env.max) {
      const mn = env.min;
      const mx = env.max;
      const centerX = (mn[0] + mx[0]) * 0.5;
      const centerY = ((-mx[2] + 10) + (-mn[2] + 10)) * 0.5;
      const centerZ = (mn[1] + mx[1]) * 0.5;
      const sizeX = Math.max(0.1, Math.abs(mx[0] - mn[0]));
      const sizeY = Math.max(0.1, Math.abs(mx[2] - mn[2]));
      const sizeZ = Math.max(0.1, Math.abs(mx[1] - mn[1]));
      envelopeMesh.visible = true;
      envelopeMesh.position.set(centerX, centerY, centerZ);
      envelopeMesh.scale.set(sizeX, sizeY, sizeZ);
    } else {
      envelopeMesh.visible = false;
    }
  }

  // profundidade gauge
  const depthPct = Math.min(100, (z / 10) * 100);
  document.getElementById('gauge-fill').style.height = depthPct + '%';
}

// ─── HUD ──────────────────────────────────────────────────────
function updateHUD(data) {
  const rad2deg = 180/Math.PI;
  const [x, y, z] = data.position;
  const [u, v, w] = data.velocity_linear;
  const [phi, tht] = data.euler;

  document.getElementById('t-depth').textContent = z.toFixed(2) + ' m';
  document.getElementById('t-x').textContent     = x.toFixed(2) + ' m';
  document.getElementById('t-y').textContent     = y.toFixed(2) + ' m';
  document.getElementById('t-u').textContent     = u.toFixed(2) + ' m/s';
  document.getElementById('t-w').textContent     = w.toFixed(2) + ' m/s';
  document.getElementById('t-phi').textContent   = (phi*rad2deg).toFixed(1) + '°';
  document.getElementById('t-tht').textContent   = (tht*rad2deg).toFixed(1) + '°';
  document.getElementById('t-time').textContent  = data.time.toFixed(1) + ' s';
  const ekfErr = Number(data?.errors?.ekf_position_error_m || 0);
  document.getElementById('t-ctrl').textContent  = `${data.controller.toUpperCase()} | EKF ${ekfErr.toFixed(2)}m`;

  const rho = data.ballast.density_avg || 1000;
  document.getElementById('t-rho').textContent = rho.toFixed(0) + ' kg/m³';
  const rhoEl = document.getElementById('t-rho');
  rhoEl.className = 'telem-value' + (rho > 1040 ? ' warn' : rho > 1048 ? ' danger' : '');

  document.getElementById('t-thr').textContent = (data.thruster.power * 100).toFixed(0) + '%';
  document.getElementById('t-bal').textContent = (data.command.ballast_cmd * 100).toFixed(0) + '%';

  // sonar canvas 2D
  updateSonarDisplay(data.sonar);

  // FPS
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime > 500) {
    const fps = (frameCount / (now - lastFpsTime) * 1000).toFixed(0);
    document.getElementById('fps-display').textContent = fps;
    frameCount = 0;
    lastFpsTime = now;
  }
}

// ─── Sonar 2D display ─────────────────────────────────────────
const sonarCtx = document.getElementById('sonar-canvas').getContext('2d');
const SONAR_LABEL = ['F', 'B', 'S', 'P', '↓', '↑'];

function updateSonarDisplay(sonarData) {
  const c = sonarCtx;
  const W = 200, H = 200, cx = W/2, cy = H/2;
  const maxR = 7.0, dispR = 85;

  c.clearRect(0, 0, W, H);

  // fundo
  c.fillStyle = 'rgba(0,20,30,0.8)';
  c.beginPath(); c.arc(cx, cy, 92, 0, Math.PI*2); c.fill();

  // círculos de distância
  [1,2,4,7].forEach(d => {
    const r = (d/maxR) * dispR;
    c.beginPath(); c.arc(cx, cy, r, 0, Math.PI*2);
    c.strokeStyle = 'rgba(0,255,200,0.1)';
    c.lineWidth = 1;
    c.stroke();
    c.fillStyle = 'rgba(0,255,200,0.25)';
    c.font = '7px Share Tech Mono';
    c.fillText(d+'m', cx + r + 2, cy - 2);
  });

  // linhas cruzadas
  c.strokeStyle = 'rgba(0,255,200,0.15)';
  c.lineWidth = 1;
  [[cx,cy-dispR,cx,cy+dispR],[cx-dispR,cy,cx+dispR,cy]].forEach(([x1,y1,x2,y2]) => {
    c.beginPath(); c.moveTo(x1,y1); c.lineTo(x2,y2); c.stroke();
  });

  // ângulos dos 6 transdutores no display 2D (projetados)
  const angles = [0, Math.PI, Math.PI/2, -Math.PI/2, null, null];

  sonarData.forEach((ray, i) => {
    if (angles[i] === null) return;
    const ang  = angles[i];
    const dist = ray.hit ? Math.min(ray.distance, maxR) : maxR;
    const r    = (dist/maxR) * dispR;
    const ex   = cx + Math.cos(ang) * r;
    const ey   = cy - Math.sin(ang) * r;

    // raio
    c.beginPath(); c.moveTo(cx, cy); c.lineTo(ex, ey);
    c.strokeStyle = ray.hit ? `rgba(0,255,200,${0.3 + ray.confidence*0.5})` : 'rgba(0,255,200,0.1)';
    c.lineWidth = 1.5; c.stroke();

    // hit marker
    if (ray.hit) {
      c.beginPath(); c.arc(ex, ey, 3, 0, Math.PI*2);
      c.fillStyle = `rgba(255,107,53,${0.5 + ray.confidence*0.5})`;
      c.fill();
    }

    // label
    const lx = cx + Math.cos(ang) * (dispR + 10);
    const ly = cy - Math.sin(ang) * (dispR + 10);
    c.fillStyle = 'rgba(200,255,232,0.5)';
    c.font = '8px Share Tech Mono';
    c.textAlign = 'center';
    c.fillText(SONAR_LABEL[i], lx, ly + 3);
  });

  // up/down separado — barra vertical
  [4, 5].forEach(i => {
    const ray = sonarData[i];
    if (!ray) return;
    const isDown = (i === 4);
    const x = isDown ? cx - 15 : cx + 15;
    const dist = ray.hit ? Math.min(ray.distance, maxR) : maxR;
    const h = (dist/maxR) * 40;
    const yStart = isDown ? cy : cy - h;

    c.fillStyle = ray.hit ? 'rgba(0,255,200,0.3)' : 'rgba(0,255,200,0.05)';
    c.fillRect(x - 3, yStart, 6, h);
    c.fillStyle = 'rgba(200,255,232,0.4)';
    c.font = '7px Share Tech Mono';
    c.textAlign = 'center';
    c.fillText(SONAR_LABEL[i], x, isDown ? cy + 50 : cy - 50);
  });

  // centro
  c.beginPath(); c.arc(cx, cy, 4, 0, Math.PI*2);
  c.fillStyle = '#00ffc8'; c.fill();
}

// ─── Controles UI ─────────────────────────────────────────────
function startSim()  { socket.emit('start'); document.getElementById('btn-start').classList.add('active'); }
function pauseSim()  {
  socket.emit('pause');
  const btn = document.getElementById('btn-pause');
  btn.classList.toggle('active');
  btn.textContent = btn.classList.contains('active') ? 'Resume' : 'Pause';
}
function resetSim()  { socket.emit('reset'); }

function setCtrl(c) {
  socket.emit('set_controller', { controller: c });
  ['lqr','mpc','rl'].forEach(x => document.getElementById('ctrl-'+x).classList.remove('active'));
  document.getElementById('ctrl-'+c).classList.add('active');
}

function setCamera(mode) {
  cameraMode = mode;
  ['follow','free','top'].forEach(x => document.getElementById('cam-'+x).classList.remove('active'));
  document.getElementById('cam-'+mode).classList.add('active');
}

function updateWeight(key, val, labelId) {
  document.getElementById(labelId).textContent = val;
  const data = {}; data[key] = parseFloat(val);
  socket.emit('update_lqr_weights', data);
}

function updateNoise(val) {
  document.getElementById('noise-val').textContent = (val/10).toFixed(1);
  socket.emit('noise_scale', { scale: val/10 });
}

function wpMode() {
  wpPlacementMode = !wpPlacementMode;
  obsPlacementMode = false;
  const hint = document.getElementById('wp-hint');
  hint.classList.toggle('visible', wpPlacementMode);
  hint.textContent = 'Click scene to place waypoint';
}

function addObstacleMode() {
  obsPlacementMode = !obsPlacementMode;
  wpPlacementMode = false;
  const hint = document.getElementById('wp-hint');
  hint.classList.toggle('visible', obsPlacementMode);
  hint.textContent = 'Click scene to add obstacle';
}

function clearWaypoints() {
  clearWaypointMarkers();
  socket.emit('set_waypoint', { position: [5, 0, 5] });
}

function clearObstacles() {
  clearObstacleMeshes();
  socket.emit('clear_obstacles');
}

// ─── Raycasting para waypoints/obstáculos ─────────────────────
const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();
const clickPlane = new THREE.Plane(new THREE.Vector3(0,1,0), -5);

renderer.domElement.addEventListener('click', e => {
  if (!wpPlacementMode && !obsPlacementMode) return;

  mouse.x = (e.clientX / window.innerWidth)  * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  const target = new THREE.Vector3();
  raycaster.ray.intersectPlane(clickPlane, target);

  if (target) {
    // Three.js → NED: x=x, y=z_ned=-(y-10), z=y
    const nedPos = [target.x, target.z, -(target.y - 10)];

    if (wpPlacementMode) {
      socket.emit('add_waypoint', { position: nedPos });
      createWaypointMarker(nedPos);
      wpPlacementMode = false;
      document.getElementById('wp-hint').classList.remove('visible');
    } else {
      socket.emit('add_obstacle', { position: nedPos, radius: 0.5 });
      obsPlacementMode = false;
      document.getElementById('wp-hint').classList.remove('visible');
    }
  }
});

// ─── Loop de animação ─────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);

  const t = performance.now() * 0.001;

  // câmera
  if (vehicleMesh) {
    const target = vehicleMesh.position.clone();

    if (cameraMode === 'follow') {
      const offset = new THREE.Vector3(
        -Math.cos(camTheta) * camDist,
        camDist * 0.5,
        -Math.sin(camTheta) * camDist,
      );
      camera.position.lerp(target.clone().add(offset), 0.05);
      camera.lookAt(target);
    } else if (cameraMode === 'top') {
      camera.position.lerp(target.clone().add(new THREE.Vector3(0, camDist*1.5, 0)), 0.05);
      camera.lookAt(target);
    }
    // free: mouse controls
  }

  // animação da superfície da água
  water.position.y = 10 + Math.sin(t * 0.3) * 0.05;

  // bio light pulsa
  bioLight1.intensity = 0.5 + Math.sin(t * 1.2) * 0.3;
  bioLight2.intensity = 0.3 + Math.sin(t * 0.8 + 1) * 0.2;

  // partículas flutuam lentamente
  const ppos = particleGeo.attributes.position;
  for(let i = 0; i < particleCount; i++) {
    ppos.array[i*3+1] += Math.sin(t + i) * 0.0005;
    if (ppos.array[i*3+1] > 10) ppos.array[i*3+1] = 0;
  }
  ppos.needsUpdate = true;

  renderer.render(scene, camera);
}

animate();

// ─── Resize ───────────────────────────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

console.log('USV Digital Twin — Frontend initialized');
</script>
</body>
</html>'''


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    _initial_load_latest()
    player_loop = PlayerLoop(player, _emit_replay_state, hz=60.0)
    player_loop.start()

    print("\n" + "=" * 58)
    print("  USV Digital Twin — Replay Visualization Server")
    print("  Replay dir:", VIEW['replay_dir'])
    print("  Abra: http://localhost:5000")
    print("=" * 58 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
