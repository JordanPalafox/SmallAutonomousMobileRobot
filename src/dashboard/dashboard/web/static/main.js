'use strict';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_LINEAR_VEL  = 0.5;
const MAX_ANGULAR_VEL = 2.0;
const MAX_HISTORY     = 20;

// Teleop velocities — start at teleop_twist_keyboard's defaults (speed 0.5,
// turn 1.0) and are adjustable in ±10% steps with the on-screen +/- buttons or
// the +/- keys, exactly like teleop_twist_keyboard. A full WASD press commands
// `_teleopSpeed` m/s linear and `_teleopTurn` rad/s angular.
let _teleopSpeed = 0.10;  // m/s  (linear)
let _teleopTurn  = 0.21;  // rad/s (angular)
const TELEOP_STEP_UP    = 1.1;   // +key → ×1.1 (≈ +10%)
const TELEOP_STEP_DOWN  = 0.9;   // -key → ×0.9 (≈ -10%), matches teleop_twist_keyboard
const TELEOP_MIN_SPEED  = 0.05;
const TELEOP_MAX_SPEED  = 1.0;
const TELEOP_MIN_TURN   = 0.1;
const TELEOP_MAX_TURN   = 3.0;

// Map world-space parameters — updated dynamically from /api/map/info
let MAP_ORIGIN_X   = -10.0;  // m
let MAP_ORIGIN_Y   = -10.0;  // m
let MAP_RESOLUTION = 0.05;   // m/pixel

// ---------------------------------------------------------------------------
// Session counters
// ---------------------------------------------------------------------------

let _missionCount = 0;
let _eventCount   = 0;
let _startTime    = Date.now();

// ---------------------------------------------------------------------------
// Map state
// ---------------------------------------------------------------------------

// Single Image element reused for map updates (avoids object churn)
const _mapImg = new Image();

// Letterbox rect computed each draw frame: {drawX, drawY, drawW, drawH, mapW, mapH}
let _mapRender = null;

let _currentPose = {x: 0, y: 0, theta: 0};
let _currentScan = null;
let _currentPlan = [];

// Waypoints: {name: {x, y, theta}}
let _waypoints = {};

// Edit mode
let _editMode    = false;
let _pendingWp   = null;  // {x, y, theta} world coords of pending placement
let _dragWp      = null;  // {wx0, wy0, wx1, wy1} live drag preview

// ---------------------------------------------------------------------------
// Teleop state
// ---------------------------------------------------------------------------

let _keysDown       = new Set();
let _teleopInterval = null;

// ---------------------------------------------------------------------------
// Socket.IO
// ---------------------------------------------------------------------------

const socket = io({
  reconnection: true,
  reconnectionAttempts: Infinity,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 10000,
});

socket.on('connect',    () => setConnectionStatus(true));
socket.on('disconnect', () => setConnectionStatus(false));

socket.on('robot_pose', (data) => {
  _eventCount++;
  _currentPose = {x: data.x, y: data.y, theta: data.theta};
  updatePose(data);
});

socket.on('robot_state', (data) => {
  _eventCount++;
  updateState(data.state, data.mission);
  updateNavStatus(data.state);
});

socket.on('telemetry', (data) => {
  _eventCount++;
  updateTelemetry(data);
  if (data && data.system_mode && data.system_mode !== _systemMode) {
    applyModeToUI(data.system_mode);
  }
});

socket.on('scan_data', (data) => { _currentScan = data; });

socket.on('nav_plan', (data) => { _currentPlan = data.points || []; });

// ---------------------------------------------------------------------------
// Connection status
// ---------------------------------------------------------------------------

function setConnectionStatus(connected) {
  const cls = connected ? 'conn-dot connected' : 'conn-dot disconnected';
  const lbl = connected ? 'Connected' : 'Disconnected — reconnecting…';
  document.getElementById('connDot').className   = cls;
  document.getElementById('footerDot').className = cls;
  document.getElementById('connLabel').textContent  = lbl;
  document.getElementById('footerLabel').textContent = connected ? 'Connected' : 'Disconnected';
}

// ---------------------------------------------------------------------------
// Pose update
// ---------------------------------------------------------------------------

function updatePose(data) {
  setText('poseX',     `${data.x.toFixed(3)} m`);
  setText('poseY',     `${data.y.toFixed(3)} m`);
  setText('poseTheta', `${data.theta.toFixed(4)} rad`);
}

// ---------------------------------------------------------------------------
// Robot state + mission
// ---------------------------------------------------------------------------

function updateState(state, mission) {
  const badge = document.getElementById('stateBadge');
  if (!badge) return;
  badge.textContent = state || 'UNKNOWN';
  badge.className   = 'state-badge ' + (state || '');

  const box = document.getElementById('missionBox');
  if (!box) return;
  if (!mission || typeof mission !== 'object') {
    box.innerHTML = '<p class="dim">No active mission</p>';
    return;
  }
  box.innerHTML = [
    ['Pallet', mission.pallet_id || '—'],
    ['Source', formatLocation(mission.source)],
    ['Level',  String(mission.source_level || '—')],
    ['Dest',   formatLocation(mission.destination)],
  ].map(([k, v]) =>
    `<div class="mission-row"><span class="mission-key">${k}</span><span class="mission-val">${v}</span></div>`
  ).join('');
}

// ---------------------------------------------------------------------------
// Nav status badge
// ---------------------------------------------------------------------------

function updateNavStatus(state) {
  const badge = document.getElementById('navStatusBadge');
  if (!badge) return;
  badge.textContent = state || 'UNKNOWN';
  badge.className   = 'nav-status-badge ' + (state || '').split(':')[0];
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

function updateTelemetry(data) {
  const lin = data.linear_vel  ?? 0;
  const ang = data.angular_vel ?? 0;
  setText('velLinear',  `${lin.toFixed(3)} m/s`);
  setText('velAngular', `${ang.toFixed(3)} rad/s`);
  setStyle('barLinear',  'width', `${Math.min(Math.abs(lin) / MAX_LINEAR_VEL  * 100, 100)}%`);
  setStyle('barAngular', 'width', `${Math.min(Math.abs(ang) / MAX_ANGULAR_VEL * 100, 100)}%`);

  updateLifter(data.lifter_level ?? 0);

  if (data.map_png) {
    _mapImg.src = 'data:image/png;base64,' + data.map_png;
  }
  if (data.map_info) {
    applyMapInfo(data.map_info);
  }
}

function applyMapInfo(info) {
  if (typeof info.origin_x   === 'number') MAP_ORIGIN_X   = info.origin_x;
  if (typeof info.origin_y   === 'number') MAP_ORIGIN_Y   = info.origin_y;
  if (typeof info.resolution === 'number') MAP_RESOLUTION = info.resolution;

  const badge = document.getElementById('mapSourceBadge');
  if (badge) {
    const live = info.source === 'live';
    badge.textContent = live ? 'LIVE' : 'STATIC MAP';
    badge.className   = 'map-source-badge ' + (live ? 'live' : 'static');
  }
}

async function loadMapInfo() {
  try {
    const res = await fetch('/api/map/info');
    if (!res.ok) return;
    const info = await res.json();
    applyMapInfo(info);
  } catch (_) {}
}

// ---------------------------------------------------------------------------
// Lifter
// ---------------------------------------------------------------------------

// SPI HAL (Tang Nano) only reaches levels 0-3 on the real robot.
const LIFTER_MAX = 3;

function initLifter() {
  const row = document.getElementById('lifterButtons');
  if (!row) return;
  row.innerHTML = '';
  for (let i = 0; i <= LIFTER_MAX; i++) {
    const btn = document.createElement('button');
    btn.className   = 'lifter-btn';
    btn.id          = `lift${i}`;
    btn.type        = 'button';
    btn.textContent = String(i);
    btn.addEventListener('click', () => setLifter(i));
    row.appendChild(btn);
  }
  updateLifter(0);
}

// Highlight the current (reported) level. Driven by /lifter_status telemetry.
function updateLifter(level) {
  for (let i = 0; i <= LIFTER_MAX; i++) {
    const btn = document.getElementById(`lift${i}`);
    if (btn) btn.classList.toggle('active', i === level);
  }
  setText('lifterText', String(level));
}

// Command a target level → /lifter_level (via POST /api/lifter).
async function setLifter(level) {
  try {
    const res = await fetch('/api/lifter', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({level}),
    });
    const json = await res.json();
    if (json.ok) {
      showToast(`Lifter → ${level}`, 'success');
    } else {
      showToast(`Error: ${json.error || 'unknown'}`, 'error');
    }
  } catch (err) {
    showToast(`Network error: ${err.message}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
      btn.classList.add('active');
      const pane = document.getElementById(`tab-${btn.dataset.tab}`);
      if (pane) pane.classList.remove('hidden');
    });
  });
}

// ---------------------------------------------------------------------------
// Waypoints — load full {name: {x, y, theta}} data
// ---------------------------------------------------------------------------

async function loadWaypoints() {
  try {
    const res  = await fetch('/api/waypoints');
    const data = await res.json();
    _waypoints = data.waypoints || {};

    const select = document.getElementById('waypointSelect');
    if (!select) return;
    const names = Object.keys(_waypoints).sort();
    select.innerHTML = '<option value="">— select waypoint —</option>';
    names.forEach(name => {
      const opt = document.createElement('option');
      opt.value       = name;
      opt.textContent = name;
      select.appendChild(opt);
    });
  } catch (err) {
    console.warn('Could not load waypoints:', err);
  }
}

// ---------------------------------------------------------------------------
// Navigation control
// ---------------------------------------------------------------------------

async function deleteWaypoint(name) {
  if (!confirm(`Delete waypoint "${name}"?`)) return;
  try {
    const res = await fetch(`/api/waypoints/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
    if (res.ok) {
      showToast(`Waypoint "${name}" deleted`, 'success');
      await loadWaypoints();
    } else {
      const data = await res.json().catch(() => ({}));
      showToast(`Delete failed: ${data.error || res.status}`, 'error');
    }
  } catch (err) {
    showToast(`Network error: ${err.message}`, 'error');
  }
}

async function sendGoalWaypoint(name) {
  try {
    const res  = await fetch('/api/goal', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({waypoint: name}),
    });
    const json = await res.json();
    if (json.ok) {
      showToast(name ? `Navigating to: ${name}` : 'Navigation cancelled', 'success');
    } else {
      showToast(`Error: ${json.error || 'unknown'}`, 'error');
    }
  } catch (err) {
    showToast(`Network error: ${err.message}`, 'error');
  }
}

function initNavControls() {
  const btnGo     = document.getElementById('btnGoWaypoint');
  const btnCancel = document.getElementById('btnCancelNav');

  btnGo?.addEventListener('click', () => {
    const sel = document.getElementById('waypointSelect');
    const wp  = sel ? sel.value : '';
    if (!wp) { showToast('Select a waypoint first.', 'error'); return; }
    sendGoalWaypoint(wp);
  });

  btnCancel?.addEventListener('click', () => sendGoalWaypoint(''));
}

// ---------------------------------------------------------------------------
// Quick mission control — one-click missions + abort
// ---------------------------------------------------------------------------

async function _sendQuickMission(type) {
  try {
    const res  = await fetch('/api/mission', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({type}),
    });
    const json = await res.json();
    if (!res.ok || json.error) {
      showToast(`Error: ${json.error || res.status}`, 'error');
    } else {
      showToast(`Misión enviada: ${type}`, 'success');
    }
  } catch (err) {
    showToast(`Network error: ${err.message}`, 'error');
  }
}

function initQuickControl() {
  document.getElementById('btnMissionRollers')
    ?.addEventListener('click', () => _sendQuickMission('ROLLER_TO_TRUCK'));
  document.getElementById('btnMissionRacks')
    ?.addEventListener('click', () => _sendQuickMission('RACK_TO_TRUCK'));
  document.getElementById('btnMissionPick')
    ?.addEventListener('click', () => _sendQuickMission('PICK_ONLY'));
  document.getElementById('btnMissionAbort')?.addEventListener('click', () => {
    if (!confirm('¿Abortar la misión actual?')) return;
    _smControl({action: 'abort'});
    showToast('Abort enviado', 'success');
  });
}

// ---------------------------------------------------------------------------
// MAP CANVAS — coordinate transforms
// ---------------------------------------------------------------------------

/**
 * Convert ROS world coords (m) → canvas pixel coords.
 * Requires _mapRender to be set by drawMapOverlay.
 */
function worldToCanvas(wx, wy) {
  if (!_mapRender) return [0, 0];
  const {drawX, drawY, drawW, drawH, mapW, mapH} = _mapRender;
  const px = (wx - MAP_ORIGIN_X) / MAP_RESOLUTION;
  const py = mapH - (wy - MAP_ORIGIN_Y) / MAP_RESOLUTION;
  return [drawX + px * drawW / mapW, drawY + py * drawH / mapH];
}

/**
 * Convert canvas pixel coords → ROS world coords (m).
 */
function canvasToWorld(cx, cy) {
  if (!_mapRender) return [0, 0];
  const {drawX, drawY, drawW, drawH, mapW, mapH} = _mapRender;
  const px = (cx - drawX) * mapW / drawW;
  const py = (cy - drawY) * mapH / drawH;
  return [
    px * MAP_RESOLUTION + MAP_ORIGIN_X,
    (mapH - py) * MAP_RESOLUTION + MAP_ORIGIN_Y,
  ];
}

// ---------------------------------------------------------------------------
// MAP CANVAS — main draw loop (10 Hz)
// ---------------------------------------------------------------------------

function drawMapOverlay() {
  const canvas = document.getElementById('mapCanvas');
  if (!canvas) return;

  const container = canvas.parentElement;
  const cw = container.clientWidth;
  const ch = container.clientHeight;
  if (!cw || !ch) return;

  // Update canvas resolution to match CSS display size
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width  = cw;
    canvas.height = ch;
  }

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, cw, ch);

  // ---- Draw map image (letterboxed) ----
  if (_mapImg.complete && _mapImg.naturalWidth > 0) {
    const iw = _mapImg.naturalWidth;
    const ih = _mapImg.naturalHeight;
    const imgRatio = iw / ih;
    const canRatio = cw / ch;

    let drawW, drawH, drawX, drawY;
    if (imgRatio >= canRatio) {
      drawW = cw;
      drawH = cw / imgRatio;
      drawX = 0;
      drawY = (ch - drawH) / 2;
    } else {
      drawH = ch;
      drawW = ch * imgRatio;
      drawX = (cw - drawW) / 2;
      drawY = 0;
    }

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(_mapImg, drawX, drawY, drawW, drawH);
    _mapRender = {drawX, drawY, drawW, drawH, mapW: iw, mapH: ih};
  } else {
    _mapRender = null;
    // Waiting placeholder
    ctx.fillStyle = '#0a0f1e';
    ctx.fillRect(0, 0, cw, ch);
    ctx.fillStyle = '#4a5f80';
    ctx.font = '14px "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Waiting for map data…', cw / 2, ch / 2);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    return;
  }

  // ---- Draw LiDAR scan (red dots) ----
  if (_currentScan && _currentScan.ranges) {
    const {ranges, angle_min, angle_increment, range_max} = _currentScan;
    ctx.fillStyle = 'rgba(255, 55, 55, 0.65)';
    ranges.forEach((r, i) => {
      if (r <= 0 || r >= (range_max || 12)) return;
      const angle = _currentPose.theta + angle_min + i * angle_increment;
      const [cx2, cy2] = worldToCanvas(
        _currentPose.x + r * Math.cos(angle),
        _currentPose.y + r * Math.sin(angle)
      );
      ctx.beginPath();
      ctx.arc(cx2, cy2, 1.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // ---- Draw navigation path (orange) ----
  if (_currentPlan && _currentPlan.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255, 100, 0, 0.85)';
    ctx.lineWidth   = 2;
    _currentPlan.forEach((pt, i) => {
      const [cx2, cy2] = worldToCanvas(pt.x, pt.y);
      i === 0 ? ctx.moveTo(cx2, cy2) : ctx.lineTo(cx2, cy2);
    });
    ctx.stroke();

    ctx.fillStyle = '#ff9900';
    _currentPlan.forEach((pt) => {
      const [cx2, cy2] = worldToCanvas(pt.x, pt.y);
      ctx.beginPath();
      ctx.arc(cx2, cy2, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // ---- Draw waypoints ----
  drawWaypoints(ctx);

  // ---- Draw drag preview (RViz-style "click & drag pose") ----
  if (_dragWp) {
    const [c0x, c0y] = worldToCanvas(_dragWp.wx0, _dragWp.wy0);
    const [c1x, c1y] = worldToCanvas(_dragWp.wx1, _dragWp.wy1);
    const dragColor = '#cc88ff';

    ctx.save();
    ctx.shadowColor = dragColor;
    ctx.shadowBlur  = 8;

    // Origin marker
    ctx.beginPath();
    ctx.arc(c0x, c0y, 8, 0, Math.PI * 2);
    ctx.fillStyle = dragColor + '40';
    ctx.fill();
    ctx.strokeStyle = dragColor;
    ctx.lineWidth   = 2;
    ctx.stroke();

    const distPx = Math.hypot(c1x - c0x, c1y - c0y);
    if (distPx > 4) {
      const ang = Math.atan2(c1y - c0y, c1x - c0x);
      // Shaft
      ctx.beginPath();
      ctx.moveTo(c0x, c0y);
      ctx.lineTo(c1x, c1y);
      ctx.strokeStyle = dragColor;
      ctx.lineWidth   = 3;
      ctx.stroke();
      // Arrowhead
      const ahLen = 12;
      ctx.beginPath();
      ctx.moveTo(c1x, c1y);
      ctx.lineTo(c1x - ahLen * Math.cos(ang - 0.45),
                 c1y - ahLen * Math.sin(ang - 0.45));
      ctx.lineTo(c1x - ahLen * Math.cos(ang + 0.45),
                 c1y - ahLen * Math.sin(ang + 0.45));
      ctx.closePath();
      ctx.fillStyle = dragColor;
      ctx.fill();
    }
    ctx.restore();
  }

  // ---- Draw robot arrow ----
  const [rx, ry] = worldToCanvas(_currentPose.x, _currentPose.y);
  ctx.save();
  ctx.translate(rx, ry);
  ctx.rotate(-_currentPose.theta);
  ctx.fillStyle   = '#00ff88';
  ctx.strokeStyle = '#00aa55';
  ctx.lineWidth   = 1;
  ctx.shadowColor = '#00ff88';
  ctx.shadowBlur  = 8;
  ctx.beginPath();
  ctx.moveTo(10, 0);
  ctx.lineTo(-6, -6);
  ctx.lineTo(-4, 0);
  ctx.lineTo(-6, 6);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

setInterval(drawMapOverlay, 100);

// ---------------------------------------------------------------------------
// Waypoint rendering
// ---------------------------------------------------------------------------

function waypointColor(name) {
  if (name.startsWith('truck'))  return '#ffd166';
  if (name.startsWith('rack'))   return '#00c8ff';
  if (name.startsWith('roller')) return '#66ee88';
  return '#cc88ff';
}

function drawWaypoints(ctx) {
  Object.entries(_waypoints).forEach(([name, wp]) => {
    const [cx, cy] = worldToCanvas(wp.x, wp.y);
    const color    = waypointColor(name);
    const theta    = wp.theta || 0;

    // Direction arrow (in canvas: negate sin due to Y-flip)
    const dx =  Math.cos(theta);
    const dy = -Math.sin(theta);

    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur  = 6;

    // Circle fill
    ctx.beginPath();
    ctx.arc(cx, cy, 8, 0, Math.PI * 2);
    ctx.fillStyle = color + '28';
    ctx.fill();

    // Circle stroke
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.stroke();

    // Heading arrow
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + 14 * dx, cy + 14 * dy);
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.stroke();

    // Arrowhead
    const ax = cx + 14 * dx;
    const ay = cy + 14 * dy;
    const headAngle = Math.atan2(dy, dx);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - 5 * Math.cos(headAngle - 0.5), ay - 5 * Math.sin(headAngle - 0.5));
    ctx.lineTo(ax - 5 * Math.cos(headAngle + 0.5), ay - 5 * Math.sin(headAngle + 0.5));
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();

    ctx.restore();

    // Label background
    ctx.font = 'bold 10px "Courier New", monospace';
    const tw = ctx.measureText(name).width;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    ctx.fillRect(cx - tw / 2 - 3, cy - 23, tw + 6, 13);

    // Label text
    ctx.fillStyle     = color;
    ctx.textAlign     = 'center';
    ctx.textBaseline  = 'bottom';
    ctx.fillText(name, cx, cy - 11);
    ctx.textAlign    = 'left';
    ctx.textBaseline = 'alphabetic';
  });
}

// ---------------------------------------------------------------------------
// Map canvas — click handling
// ---------------------------------------------------------------------------

function initMapInteraction() {
  const canvas = document.getElementById('mapCanvas');
  if (!canvas) return;

  const DRAG_THRESHOLD_PX = 10;
  let downCx = 0, downCy = 0;

  const evToCanvas = (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    return [(e.clientX - rect.left) * scaleX,
            (e.clientY - rect.top)  * scaleY];
  };

  canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    // Don't start a drag while the popup is up
    if (!document.getElementById('waypointPopup').classList.contains('hidden')) return;
    const [cx, cy] = evToCanvas(e);
    downCx = cx; downCy = cy;
    const [wx, wy] = canvasToWorld(cx, cy);
    _dragWp = {wx0: wx, wy0: wy, wx1: wx, wy1: wy};
    drawMapOverlay();
  });

  window.addEventListener('mousemove', (e) => {
    if (!_dragWp) return;
    const [cx, cy] = evToCanvas(e);
    const [wx, wy] = canvasToWorld(cx, cy);
    _dragWp.wx1 = wx;
    _dragWp.wy1 = wy;
    drawMapOverlay();
  });

  window.addEventListener('mouseup', (e) => {
    if (!_dragWp || e.button !== 0) return;
    const drag = _dragWp;
    _dragWp = null;

    const [cx, cy] = evToCanvas(e);
    const dragDistPx = Math.hypot(cx - downCx, cy - downCy);
    drawMapOverlay();

    if (_editMode) {
      // Short click in edit mode → delete the waypoint under the cursor
      // (if any).  Long drag → create a new waypoint with heading.
      if (dragDistPx < DRAG_THRESHOLD_PX) {
        for (const [name, wp] of Object.entries(_waypoints)) {
          const [wpx, wpy] = worldToCanvas(wp.x, wp.y);
          const dist = Math.hypot(cx - wpx, cy - wpy);
          if (dist < 14) {
            deleteWaypoint(name);
            return;
          }
        }
        return;   // click on empty space in edit mode: do nothing
      }
      // Map Y-axis: world Y up, canvas Y down → atan2 in world coords directly
      const theta = Math.atan2(drag.wy1 - drag.wy0, drag.wx1 - drag.wx0);
      _pendingWp = {x: drag.wx0, y: drag.wy0, theta};
      showWaypointPopup(drag.wx0, drag.wy0, theta);
    } else if (dragDistPx < DRAG_THRESHOLD_PX) {
      // Short click outside edit mode → navigate to nearby waypoint
      for (const [name, wp] of Object.entries(_waypoints)) {
        const [wpx, wpy] = worldToCanvas(wp.x, wp.y);
        const dist = Math.hypot(cx - wpx, cy - wpy);
        if (dist < 14) {
          sendGoalWaypoint(name);
          const sel = document.getElementById('waypointSelect');
          if (sel) sel.value = name;
          return;
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Edit mode toggle
// ---------------------------------------------------------------------------

function initEditMode() {
  const btn  = document.getElementById('btnEditWaypoints');
  const hint = document.getElementById('editHint');
  const canvas = document.getElementById('mapCanvas');
  if (!btn) return;

  btn.addEventListener('click', () => {
    _editMode = !_editMode;
    btn.classList.toggle('active', _editMode);
    hint.classList.toggle('hidden', !_editMode);
    if (canvas) canvas.classList.toggle('edit-mode', _editMode);
  });

  // ESC cancels: in-flight drag, then edit mode + popup
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (_dragWp) {
      _dragWp = null;
      drawMapOverlay();
      return;
    }
    if (_editMode) {
      _editMode = false;
      btn.classList.remove('active');
      hint.classList.add('hidden');
      if (canvas) canvas.classList.remove('edit-mode');
      hideWaypointPopup();
    }
  });
}

// ---------------------------------------------------------------------------
// Waypoint popup
// ---------------------------------------------------------------------------

// Next free name for a type, mirroring the server's authoritative scheme
// (<type>_<n>).  Shown only as a preview — the server assigns the final name
// under a lock so two quick saves can't collide.
function _nextWaypointName(type) {
  const re = new RegExp(`^${type}_(\\d+)$`);
  let maxN = 0;
  Object.keys(_waypoints).forEach((name) => {
    const m = name.match(re);
    if (m) maxN = Math.max(maxN, parseInt(m[1], 10));
  });
  return `${type}_${maxN + 1}`;
}

function _selectedWpType() {
  return document.querySelector('#wpTypeGroup .wp-type-btn.active')?.dataset.type || 'roller';
}

function _setWpType(type) {
  document.querySelectorAll('#wpTypeGroup .wp-type-btn').forEach((b) => {
    b.classList.toggle('active', b.dataset.type === type);
  });
  _updateWpNamePreview();
}

function _updateWpNamePreview() {
  setText('wpNamePreview', `Name: ${_nextWaypointName(_selectedWpType())}`);
}

function showWaypointPopup(wx, wy, theta) {
  setText('wpCoords', `x: ${wx.toFixed(3)} m,  y: ${wy.toFixed(3)} m`);
  setText('wpHeading',
    `θ: ${theta.toFixed(3)} rad (${(theta * 180 / Math.PI).toFixed(1)}°)`);

  _setWpType('roller');   // reset to default selection each time the popup opens

  document.getElementById('waypointPopup').classList.remove('hidden');
}

function hideWaypointPopup() {
  document.getElementById('waypointPopup').classList.add('hidden');
  _pendingWp = null;
}

function initWaypointPopup() {
  document.getElementById('btnCancelWaypoint')?.addEventListener('click', hideWaypointPopup);

  // Type toggle buttons: highlight the picked one and refresh the name preview.
  document.querySelectorAll('#wpTypeGroup .wp-type-btn').forEach((btn) => {
    btn.addEventListener('click', () => _setWpType(btn.dataset.type));
  });

  document.getElementById('btnSaveWaypoint')?.addEventListener('click', async () => {
    if (!_pendingWp) return;
    const type  = _selectedWpType();
    const theta = _pendingWp.theta || 0;

    try {
      const res = await fetch('/api/waypoints', {
        method:  'POST',
        headers: {'Content-Type': 'application/json'},
        body:    JSON.stringify({type, x: _pendingWp.x, y: _pendingWp.y, theta}),
      });
      const json = await res.json();
      if (json.ok) {
        hideWaypointPopup();
        await loadWaypoints();
        showToast(`Waypoint "${json.name || type}" saved`, 'success');
      } else {
        showToast(`Error: ${json.error || 'unknown'}`, 'error');
      }
    } catch (err) {
      showToast(`Network error: ${err.message}`, 'error');
    }
  });
}

// ---------------------------------------------------------------------------
// Teleop
// ---------------------------------------------------------------------------

function _isInputFocused() {
  const tag = document.activeElement ? document.activeElement.tagName : '';
  return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA';
}

function _computeTeleopVelocities() {
  let linear = 0, angular = 0;
  if (_keysDown.has('w') || _keysDown.has('arrowup'))    linear  += _teleopSpeed;
  if (_keysDown.has('s') || _keysDown.has('arrowdown'))  linear  -= _teleopSpeed;
  if (_keysDown.has('a') || _keysDown.has('arrowleft'))  angular += _teleopTurn;
  if (_keysDown.has('d') || _keysDown.has('arrowright')) angular -= _teleopTurn;
  return {linear, angular};
}

function _startTeleopInterval() {
  if (_teleopInterval !== null) return;
  _teleopInterval = setInterval(async () => {
    if (_keysDown.size === 0) { _stopTeleopInterval(); return; }
    const {linear, angular} = _computeTeleopVelocities();
    updateTeleopBars(linear, angular);
    try {
      await fetch('/api/teleop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({linear, angular}),
      });
    } catch (_) {}
  }, 100);
}

function _stopTeleopInterval() {
  if (_teleopInterval !== null) { clearInterval(_teleopInterval); _teleopInterval = null; }
}

function _highlightKeys() {
  const map = {w: 'keyW', a: 'keyA', s: 'keyS', d: 'keyD'};
  Object.entries(map).forEach(([k, id]) => {
    document.getElementById(id)?.classList.toggle('pressed', _keysDown.has(k));
  });
}

function _updateSpeedReadout() {
  setText('tSpeedVal', _teleopSpeed.toFixed(2));
  setText('tTurnVal',  _teleopTurn.toFixed(2));
}

function _adjustTeleopSpeed(factor) {
  // Scale linear and angular together by `factor`, clamped to sane bounds —
  // same behaviour as teleop_twist_keyboard's q/z keys.
  _teleopSpeed = Math.min(TELEOP_MAX_SPEED, Math.max(TELEOP_MIN_SPEED, _teleopSpeed * factor));
  _teleopTurn  = Math.min(TELEOP_MAX_TURN,  Math.max(TELEOP_MIN_TURN,  _teleopTurn  * factor));
  _updateSpeedReadout();
  // If a movement key is held, reflect the new speed on the bars right away.
  const {linear, angular} = _computeTeleopVelocities();
  updateTeleopBars(linear, angular);
}

function initTeleop() {
  document.addEventListener('keydown', (e) => {
    if (_isInputFocused()) return;
    const key = e.key.toLowerCase();
    if (key === ' ') { e.preventDefault(); triggerEstop(); return; }
    // Speed adjust (+/-) — both linear and angular, ±10% per press.
    if (key === '+' || key === '=') { e.preventDefault(); _adjustTeleopSpeed(TELEOP_STEP_UP);   return; }
    if (key === '-' || key === '_') { e.preventDefault(); _adjustTeleopSpeed(TELEOP_STEP_DOWN); return; }
    const valid = new Set(['w','a','s','d','arrowup','arrowdown','arrowleft','arrowright']);
    if (!valid.has(key)) return;
    e.preventDefault();
    _keysDown.add(key);
    _highlightKeys();
    _startTeleopInterval();
  });

  document.addEventListener('keyup', (e) => {
    if (_isInputFocused()) return;
    _keysDown.delete(e.key.toLowerCase());
    _highlightKeys();
    if (_keysDown.size === 0) {
      _stopTeleopInterval();
      updateTeleopBars(0, 0);
      fetch('/api/teleop/stop', {method: 'POST'}).catch(() => {});
    }
  });

  document.getElementById('btnEstop')?.addEventListener('click', triggerEstop);
  document.getElementById('btnSpeedUp')?.addEventListener('click',   () => _adjustTeleopSpeed(TELEOP_STEP_UP));
  document.getElementById('btnSpeedDown')?.addEventListener('click', () => _adjustTeleopSpeed(TELEOP_STEP_DOWN));
  _updateSpeedReadout();
}

function triggerEstop() {
  _keysDown.clear();
  _stopTeleopInterval();
  _highlightKeys();
  updateTeleopBars(0, 0);
  fetch('/api/teleop/stop', {method: 'POST'}).catch(() => {});
}

function updateTeleopBars(linear, angular) {
  const linPct = Math.min(Math.abs(linear) / _teleopSpeed * 50, 50);
  const angPct = Math.min(Math.abs(angular) / _teleopTurn  * 50, 50);
  const tBarL  = document.getElementById('tBarLinear');
  const tBarA  = document.getElementById('tBarAngular');
  if (tBarL) { tBarL.style.width = linPct + '%'; tBarL.style.left = linear  >= 0 ? '50%' : (50 - linPct) + '%'; }
  if (tBarA) { tBarA.style.width = angPct + '%'; tBarA.style.left = angular >= 0 ? '50%' : (50 - angPct) + '%'; }
  setText('tVelLinear',  linear.toFixed(2));
  setText('tVelAngular', angular.toFixed(2));
}

// ---------------------------------------------------------------------------
// Unified SLAM mode: mapping + localisation run together.  No mode switch.
// We keep the helpers for backwards compat (server-side /api/mode still
// works) but the dashboard no longer exposes a UI to toggle.
// ---------------------------------------------------------------------------

let _systemMode = 'NAVIGATION';

function applyModeToUI(mode) {
  _systemMode = mode || 'NAVIGATION';
  // Always enable navigation controls — there is no separate mapping
  // mode that blocks them anymore.
  ['btnGoWaypoint', 'btnCancelNav', 'waypointSelect'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.disabled = false; el.title = ''; }
  });
  // Save Map is always available too (map is always being built).
  const saveBtn = document.getElementById('btnSaveMap');
  if (saveBtn) saveBtn.classList.remove('hidden');
}

async function fetchCurrentMode() {
  try {
    const res  = await fetch('/api/mode');
    const data = await res.json();
    if (data && data.mode) applyModeToUI(data.mode);
  } catch (e) { /* ignore */ }
}

function initSystemMode() {
  // Unified mode: just enable everything and fetch once for compat.
  applyModeToUI('NAVIGATION');
  fetchCurrentMode();

  // Save Map button
  document.getElementById('btnSaveMap')?.addEventListener('click', async () => {
    const btn = document.getElementById('btnSaveMap');
    btn.classList.add('saving');
    try {
      const res  = await fetch('/api/map/save', {method: 'POST'});
      const data = await res.json();
      if (res.ok && data.success) {
        showToast('Map saved to ~/ros2_maps/', 'success');
      } else {
        showToast(`Save failed: ${data.message || 'unknown error'}`, 'error');
      }
    } catch (err) {
      showToast(`Save error: ${err}`, 'error');
    } finally {
      btn.classList.remove('saving');
    }
  });

}

// ---------------------------------------------------------------------------
// Mission form
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initLifter();
  loadWaypoints();
  initNavControls();
  initQuickControl();
  initTeleop();
  initEditMode();
  initMapInteraction();
  initWaypointPopup();
  initSystemMode();
  initVoice();

  // Preload map and fetch metadata immediately
  _mapImg.src = '/api/map';
  loadMapInfo();
});

// ---------------------------------------------------------------------------
// Mission history
// ---------------------------------------------------------------------------

function addHistoryItem(mission) {
  const list = document.getElementById('historyList');
  if (!list) return;
  list.querySelector('.dim')?.remove();

  const li = document.createElement('li');
  li.className = 'history-item';

  let meta = '';
  if (mission.type === 'CUSTOM') {
    const src = mission.source?.waypoint || '?';
    const dest = (typeof mission.destination === 'string')
      ? mission.destination
      : (mission.destination?.waypoint || '?');
    meta = `${src} → ${dest}`;
  } else {
    const cands = mission.source?.candidates;
    meta = (cands && cands.length) ? cands.join(',') : 'all';
    meta += ' → (QR)';
  }

  li.innerHTML =
    `<span class="hist-id">${mission.id || '—'}</span>` +
    `<span class="hist-meta"> · ${mission.type} · ${meta}</span>`;
  list.insertBefore(li, list.firstChild);
  while (list.children.length > MAX_HISTORY) list.removeChild(list.lastChild);
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------

let _toastTimer = null;

function showToast(msg, type = 'success') {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.className   = `toast ${type}`;
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { toast.className = 'toast hidden'; }, 4000);
}

// ---------------------------------------------------------------------------
// Uptime clock
// ---------------------------------------------------------------------------

function startUptimeClock() {
  setInterval(() => {
    const s = Math.floor((Date.now() - _startTime) / 1000);
    setText('statUptime', `${pad(Math.floor(s/3600))}:${pad(Math.floor(s%3600/60))}:${pad(s%60)}`);
    setText('statEvents', String(_eventCount));
  }, 1000);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setStyle(id, prop, value) {
  const el = document.getElementById(id);
  if (el) el.style[prop] = value;
}

function pad(n) { return String(n).padStart(2, '0'); }

function formatLocation(loc) {
  if (!loc) return '—';
  return loc.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// Voice tab
// ---------------------------------------------------------------------------

let _mediaRecorder  = null;
let _audioChunks    = [];
let _voiceRecording = false;

socket.on('voice_result', (data) => {
  if (data && data.word) {
    _showVoiceResult(data.word, data.action);
  }
});

async function initVoice() {
  // Load vocabulary from server
  try {
    const res  = await fetch('/api/voice/status');
    const data = await res.json();
    const statusEl = document.getElementById('voiceModelStatus');
    const vocabEl  = document.getElementById('voiceVocab');

    if (data.ready) {
      if (statusEl) statusEl.textContent = `HMM ready — ${data.vocabulary.length} words`;
      if (vocabEl) {
        vocabEl.innerHTML = data.vocabulary
          .map(w => `<span class="voice-vocab-chip">${w}</span>`)
          .join('');
      }
    } else {
      if (statusEl) statusEl.textContent = 'Models not found — train first';
    }
  } catch (_) {}

  const btn = document.getElementById('btnVoiceRecord');
  if (!btn) return;

  // Touch/pointer events for hold-to-record
  btn.addEventListener('pointerdown', (e) => { e.preventDefault(); _startRecording(); });
  btn.addEventListener('pointerup',   (e) => { e.preventDefault(); _stopRecording();  });
  btn.addEventListener('pointerleave',(e) => { if (_voiceRecording) _stopRecording(); });

  // Keyboard shortcut: hold V to record
  document.addEventListener('keydown', (e) => {
    if (e.code === 'KeyV' && !_voiceRecording && !_isInputFocused()) {
      e.preventDefault();
      _startRecording();
    }
  });
  document.addEventListener('keyup', (e) => {
    if (e.code === 'KeyV' && _voiceRecording) {
      e.preventDefault();
      _stopRecording();
    }
  });
}

async function _startRecording() {
  if (_voiceRecording) return;
  _voiceRecording = true;

  const btn = document.getElementById('btnVoiceRecord');
  _setVoiceStatus('recording', '&#9679; Recording…');
  btn?.classList.add('recording');
  btn && (btn.textContent = '⬛ Release to Send');

  _audioChunks = [];

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {sampleRate: 16000, channelCount: 1, echoCancellation: true},
    });

    const opts = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? {mimeType: 'audio/webm;codecs=opus'}
      : {};
    _mediaRecorder = new MediaRecorder(stream, opts);

    _mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) _audioChunks.push(e.data);
    };

    _mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      await _sendAudio();
    };

    _mediaRecorder.start(100); // collect in 100ms chunks
  } catch (err) {
    _setVoiceStatus('error', 'Mic error: ' + err.message);
    _voiceRecording = false;
    btn?.classList.remove('recording');
    btn && (btn.textContent = '● Hold to Record');
  }
}

function _stopRecording() {
  if (!_voiceRecording) return;
  _voiceRecording = false;

  const btn = document.getElementById('btnVoiceRecord');
  btn?.classList.remove('recording');
  btn && (btn.textContent = '● Hold to Record');
  _setVoiceStatus('processing', 'Processing…');

  if (_mediaRecorder && _mediaRecorder.state !== 'inactive') {
    _mediaRecorder.stop();
  }
}

async function _sendAudio() {
  if (_audioChunks.length === 0) {
    _setVoiceStatus('error', 'No audio captured');
    return;
  }

  const mime = _mediaRecorder?.mimeType || 'audio/webm';
  const blob = new Blob(_audioChunks, {type: mime});

  // Decode + resample to a 16 kHz mono 16-bit PCM WAV right here in the
  // browser. That way the backend only needs scipy to read it — no pydub or
  // ffmpeg on the server, which is what was failing ("Could not decode
  // audio"). If WebAudio decoding is unavailable we ship the raw container
  // and let the server fall back to pydub.
  const form = new FormData();
  try {
    const wav = await _blobToWav16k(blob);
    form.append('audio', wav, 'recording.wav');
  } catch (err) {
    console.warn('[voice] client-side WAV encode failed, sending raw blob:', err);
    form.append('audio', blob, 'recording.webm');
  }

  try {
    const res  = await fetch('/api/voice', {method: 'POST', body: form});
    const json = await res.json();

    if (res.ok && json.word) {
      _showVoiceResult(json.word, json.action);
    } else {
      const msg = json.error + (json.reason ? ' — ' + json.reason : '');
      _setVoiceStatus('error', 'Error: ' + (msg || 'unknown'));
    }
  } catch (err) {
    _setVoiceStatus('error', 'Network error: ' + err.message);
  }
}

// Decode any browser-recorded audio blob (webm/opus, ogg, …) into a
// 16 kHz mono 16-bit PCM WAV Blob using the WebAudio API.
async function _blobToWav16k(blob) {
  const arrayBuf = await blob.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) throw new Error('AudioContext unavailable');

  const ctx = new AC();
  let decoded;
  try {
    // Promise form; some Safari builds need the callback form, handled below.
    decoded = await new Promise((resolve, reject) => {
      const p = ctx.decodeAudioData(arrayBuf, resolve, reject);
      if (p && typeof p.then === 'function') p.then(resolve, reject);
    });
  } finally {
    if (ctx.close) ctx.close();
  }

  // Resample to 16 kHz mono through an OfflineAudioContext.
  const targetRate = 16000;
  const frames = Math.max(1, Math.ceil(decoded.duration * targetRate));
  const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const off = new OAC(1, frames, targetRate);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return _encodeWav16(rendered.getChannelData(0), targetRate);
}

// Float32 samples in [-1, 1] → mono 16-bit PCM WAV Blob.
function _encodeWav16(samples, sampleRate) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buf);
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + n * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);            // PCM fmt chunk size
  view.setUint16(20, 1, true);             // format = PCM
  view.setUint16(22, 1, true);             // channels = 1
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (mono, 16-bit)
  view.setUint16(32, 2, true);             // block align
  view.setUint16(34, 16, true);            // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, n * 2, true);

  let off = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return new Blob([view], {type: 'audio/wav'});
}

function _showVoiceResult(word, action) {
  setText('voiceResultWord', word);
  _setVoiceStatus('done', 'Sent: ' + word + _voiceActionLabel(action));

  // History
  const list = document.getElementById('voiceHistoryList');
  if (list) {
    list.querySelector('.dim')?.remove();
    const li = document.createElement('li');
    li.className   = 'history-item';
    li.textContent = word + _voiceActionLabel(action) + '  ' + new Date().toLocaleTimeString();
    list.insertBefore(li, list.firstChild);
    while (list.children.length > MAX_HISTORY) list.removeChild(list.lastChild);
  }

  setTimeout(() => _setVoiceStatus('', 'Ready'), 3000);
}

// Short human label for the robot action a voice word triggered (if any).
function _voiceActionLabel(action) {
  if (!action || !action.kind) return '';
  if (action.kind === 'teleop')  return ` → drive ${action.duration}s`;
  if (action.kind === 'mission') return ` → ${action.type}`;
  if (action.kind === 'ignored') return ` → ignored (${action.reason})`;
  return '';
}

function _setVoiceStatus(cls, text) {
  const el = document.getElementById('voiceStatus');
  if (!el) return;
  el.className  = 'voice-status' + (cls ? ' ' + cls : '');
  el.innerHTML  = text;
}

// ---------------------------------------------------------------------------
// State machine debug panel
// ---------------------------------------------------------------------------

let _smOutcomes = {};       // {state_name: [outcome, ...]}
let _smLastState = null;
const _SM_LOG_MAX = 20;

async function initSmDebugPanel() {
  // Load per-state outcomes map for the Force-outcome dropdown.
  try {
    const res = await fetch('/api/sm/outcomes');
    _smOutcomes = await res.json();
  } catch (err) {
    console.warn('Could not load /api/sm/outcomes:', err);
    _smOutcomes = {};
  }

  // Initial snapshot pull (in case the server emitted before this tab existed).
  try {
    const res = await fetch('/api/sm/snapshot');
    const data = await res.json();
    if (data.snapshot) _renderSmSnapshot(data.snapshot);
    if (Array.isArray(data.transitions)) {
      data.transitions.forEach(_appendSmTransition);
    }
  } catch (err) { /* ignored — socket will catch up */ }

  document.getElementById('smBtnPause').addEventListener('click',
    () => _smControl({action: 'pause'}));
  document.getElementById('smBtnResume').addEventListener('click',
    () => _smControl({action: 'resume'}));
  document.getElementById('smBtnStep').addEventListener('click',
    () => _smControl({action: 'step'}));
  document.getElementById('smBtnAbort').addEventListener('click', () => {
    if (!confirm('Abort the current mission?')) return;
    _smControl({action: 'abort'});
  });
  document.getElementById('smStepModeToggle').addEventListener('change', (ev) => {
    _smControl({action: 'set_step_mode', value: ev.target.checked});
  });
  document.getElementById('smBtnForce').addEventListener('click', () => {
    const sel = document.getElementById('smForceSelect');
    const value = sel.value;
    if (!value) return;
    _smControl({action: 'force_outcome', value});
    sel.value = '';
  });
}

async function _smControl(payload) {
  try {
    await fetch('/api/sm/control', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
  } catch (err) {
    console.warn('sm/control failed:', err);
  }
}

function _renderSmSnapshot(snap) {
  if (!snap) return;
  // Drive the main state badge from the snapshot too (published at 2 Hz, and on
  // connect) so it shows the real SM state immediately — /robot_state is
  // volatile and only fires on a state CHANGE, leaving a late dashboard stuck
  // on UNKNOWN until the next transition.
  const badge = document.getElementById('stateBadge');
  if (badge && snap.state) {
    badge.textContent = snap.state;
    badge.className   = 'state-badge ' + snap.state;
  }
  setText('smCurrentState',     snap.state || '—');
  setText('smMissionId',        snap.mission_id || '—');
  setText('smMissionType',      snap.mission_type || '—');
  setText('smCurrentCandidate', snap.current_candidate || '—');
  setText('smQrValue',          snap.qr_value || snap.qr_detected || '—');
  setText('smResolvedDest',     snap.resolved_dest || '—');
  setText('smCandidateQueue',
    Array.isArray(snap.candidate_queue) && snap.candidate_queue.length
      ? snap.candidate_queue.join(' → ')
      : '—');

  // Repopulate the force-outcome dropdown when the state changes.
  if (snap.state !== _smLastState) {
    _smLastState = snap.state;
    const sel = document.getElementById('smForceSelect');
    if (sel) {
      const outs = _smOutcomes[snap.state] || [];
      sel.innerHTML = '<option value="">— select —</option>'
        + outs.map(o => `<option value="${o}">${o}</option>`).join('');
    }
  }

  // Debug flags — waiting hint + step-mode toggle echo.
  const dbg = snap.debug || {};
  const toggle = document.getElementById('smStepModeToggle');
  if (toggle && toggle.checked !== !!dbg.step_mode) toggle.checked = !!dbg.step_mode;
  const hint = document.getElementById('smWaitingHint');
  if (hint) {
    if (dbg.waiting_state) {
      hint.textContent = `⏸ Waiting for step at ${dbg.waiting_state}…`;
    } else if (dbg.pause) {
      hint.textContent = '⏸ Paused';
    } else if (dbg.abort) {
      hint.textContent = '⛔ Aborted — clear with /sm/control {action:"clear_abort"}';
    } else {
      hint.textContent = '';
    }
  }

  const pre = document.getElementById('smBlackboardJson');
  if (pre) pre.textContent = JSON.stringify(snap, null, 2);
}

function _appendSmTransition(tr) {
  const list = document.getElementById('smTransitionLog');
  if (!list || !tr) return;
  // Clear the "no transitions" placeholder once we have data.
  if (list.firstElementChild && list.firstElementChild.classList.contains('dim')) {
    list.innerHTML = '';
  }
  const li = document.createElement('li');
  const ts = (typeof tr.t === 'number') ? tr.t.toFixed(2) : '—';
  li.innerHTML = `<span class="mono dim">${ts}</span> `
    + `<span class="mono">${tr.state}</span> `
    + `<span class="accent">→ ${tr.outcome}</span>`;
  list.insertBefore(li, list.firstChild);
  while (list.children.length > _SM_LOG_MAX) {
    list.removeChild(list.lastChild);
  }
}

socket.on('sm_snapshot', _renderSmSnapshot);
socket.on('sm_transition', _appendSmTransition);
socket.on('sm_transition_bulk', (arr) => {
  if (!Array.isArray(arr)) return;
  arr.forEach(_appendSmTransition);
});

// ---------------------------------------------------------------------------
// Mission builder (3 mission types — ROLLER_TO_TRUCK / RACK_TO_TRUCK / CUSTOM)
// ---------------------------------------------------------------------------

let _zones = {zones: {}, qr_aliases: {}};

async function initMissionBuilder() {
  const form = document.getElementById('missionForm');
  if (!form) return;

  try {
    const res = await fetch('/api/sm/zones');
    _zones = await res.json();
  } catch (err) {
    console.warn('Could not load /api/sm/zones:', err);
  }

  const typeSel = document.getElementById('missionType');
  typeSel.addEventListener('change', _renderMissionForm);
  document.getElementById('customScanQr').addEventListener('change', _updateMissionPreview);
  document.getElementById('customSource').addEventListener('change', _updateMissionPreview);
  document.getElementById('customDest').addEventListener('change', _updateMissionPreview);
  document.getElementById('customSkipAlign').addEventListener('change', _updateMissionPreview);
  document.getElementById('missionIdInput').addEventListener('input', _updateMissionPreview);
  document.getElementById('pickupLevel').addEventListener('input', _updateMissionPreview);
  document.getElementById('placeLevel').addEventListener('input', _updateMissionPreview);

  _renderMissionForm();

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const mission = _buildMissionJson();
    if (!mission) return;

    const btn = form.querySelector('.btn-submit');
    btn.disabled = true;
    try {
      const res = await fetch('/api/mission', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(mission),
      });
      const json = await res.json();
      if (!res.ok || json.error) {
        showToast(`Error: ${json.error || 'Unknown error'}`, 'error');
        return;
      }
      showToast(`Mission sent: ${mission.type}`, 'success');
      addHistoryItem(json.mission);
      _missionCount++;
      setText('statMissions', String(_missionCount));
    } catch (err) {
      showToast(`Network error: ${err.message}`, 'error');
    } finally {
      btn.disabled = false;
    }
  });
}

function _renderMissionForm() {
  const type = document.getElementById('missionType').value;
  const searchSec = document.getElementById('missionSearchSection');
  const customSec = document.getElementById('missionCustomSection');
  const pickupInput = document.getElementById('pickupLevel');

  if (type === 'CUSTOM') {
    searchSec.classList.add('hidden');
    customSec.classList.remove('hidden');
    _populateCustomDropdowns();
  } else {
    customSec.classList.add('hidden');
    searchSec.classList.remove('hidden');
    _populateCandidateList(type);
  }

  // Sensible default pickup levels per type.
  if (type === 'ROLLER_TO_TRUCK') pickupInput.value = '1';
  else if (type === 'RACK_TO_TRUCK') pickupInput.value = '3';

  _updateMissionPreview();
}

function _populateCandidateList(type) {
  const list = document.getElementById('candidateList');
  list.innerHTML = '';
  const zones = _zones.zones || {};
  let pool = [];
  if (type === 'ROLLER_TO_TRUCK') {
    pool = zones.rollers || [];
  } else if (type === 'RACK_TO_TRUCK') {
    pool = (zones.racks_l1 || []).concat(zones.racks_l2 || []);
  }
  if (pool.length === 0) {
    list.innerHTML = '<span class="dim">No zones defined</span>';
    return;
  }
  pool.forEach(name => {
    const label = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = name;
    cb.addEventListener('change', _updateMissionPreview);
    label.appendChild(cb);
    label.appendChild(document.createTextNode(name));
    list.appendChild(label);
  });
}

function _populateCustomDropdowns() {
  const all = [];
  Object.values(_zones.zones || {}).forEach(arr => all.push(...arr));
  const srcSel = document.getElementById('customSource');
  const destSel = document.getElementById('customDest');
  srcSel.innerHTML = all.map(n => `<option value="${n}">${n}</option>`).join('');
  destSel.innerHTML = '<option value="auto_from_qr">(auto from QR)</option>'
    + all.map(n => `<option value="${n}">${n}</option>`).join('');
}

function _buildMissionJson() {
  const type = document.getElementById('missionType').value;
  const idVal = document.getElementById('missionIdInput').value.trim();
  const pickup = parseInt(document.getElementById('pickupLevel').value, 10);
  const place = parseInt(document.getElementById('placeLevel').value, 10);

  if (Number.isNaN(pickup) || pickup < 0 || pickup > 7) {
    showToast('Pickup level must be 0–7', 'error'); return null;
  }
  if (Number.isNaN(place) || place < 0 || place > 7) {
    showToast('Place level must be 0–7', 'error'); return null;
  }

  const base = {type, pickup_level: pickup, place_level: place};
  if (idVal) base.id = idVal;

  if (type === 'CUSTOM') {
    const src = document.getElementById('customSource').value;
    const dest = document.getElementById('customDest').value;
    const scan = document.getElementById('customScanQr').checked;
    const skip = document.getElementById('customSkipAlign').checked;
    if (!src) { showToast('Pick a source waypoint', 'error'); return null; }
    base.source = {waypoint: src, scan_qr: scan};
    base.destination = (dest === 'auto_from_qr') ? 'auto_from_qr' : {waypoint: dest};
    if (skip) base.skip_alignment = true;
    return base;
  }

  // Search-type missions.
  const checked = Array.from(
    document.querySelectorAll('#candidateList input:checked')
  ).map(cb => cb.value);
  base.source = checked.length ? {candidates: checked} : {};
  base.destination = 'auto_from_qr';
  return base;
}

function _updateMissionPreview() {
  const pre = document.getElementById('missionPreview');
  if (!pre) return;
  const mission = _buildMissionJson();
  pre.textContent = mission ? JSON.stringify(mission, null, 2) : '';
}
