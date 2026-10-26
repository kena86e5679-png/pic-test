const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(express.static(path.join(__dirname)));

// 1 公尺格狀陣列標記物系統 (預設建立 -12m ~ +12m 共 25x25 = 625 個 1m 標記點，並支援動態擴展)
const GRID_STEP = 1.0;
const GRID_HALF = 12; // -12..12 共 25 行列 (A ~ Y, 1 ~ 25)

function rowLabel(idx) {
  let s = '';
  let n = idx;
  while (n >= 0) {
    s = String.fromCharCode(65 + (n % 26)) + s;
    n = Math.floor(n / 26) - 1;
  }
  return s;
}

function createInitialGridMarkers() {
  const markers = {};
  for (let r = -GRID_HALF; r <= GRID_HALF; r++) {
    const rStr = rowLabel(r + GRID_HALF);
    for (let c = -GRID_HALF; c <= GRID_HALF; c++) {
      const colNum = c + GRID_HALF + 1;
      const id = `${rStr}-${colNum}`;
      const gx = +(c * GRID_STEP).toFixed(2);
      const gz = +(r * GRID_STEP).toFixed(2);
      markers[id] = {
        id,
        row: r + GRID_HALF,
        col: c + GRID_HALF,
        gx,
        gz,
        x: gx,
        y: 1.2,
        z: gz,
        nx: 0,
        nz: 1,
        snapped: false,
        active: false,
        activatedBy: null,
        timestamp: 0,
      };
    }
  }
  return markers;
}

const state = {
  players: {},   // id -> { id, x, z, yaw, speed, lastSeen, hasTargetPhoto }
  traces: {},    // 後台儲存每位玩家的歷史軌跡點 id -> [{x, z}, ...] (用於計算範圍邊界，不直接畫雜亂軌跡線)
  structures: [], // 玩家生成的虛擬建築 [{ gid, x, z, w, d, h, kind, rot, owner }]
  markers: createInitialGridMarkers(),
  walls: [],
};

function recordTrace(playerId, x, z) {
  if (!state.traces[playerId]) state.traces[playerId] = [];
  const list = state.traces[playerId];
  const last = list[list.length - 1];
  if (!last || Math.hypot(x - last.x, z - last.z) >= 0.25) {
    list.push({ x: +Number(x).toFixed(2), z: +Number(z).toFixed(2) });
    if (list.length > 600) list.shift();
  }
}

function broadcast(data, excludeWs = null) {
  const msg = JSON.stringify(data);
  wss.clients.forEach((client) => {
    if (client !== excludeWs && client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  });
}

wss.on('connection', (ws) => {
  let clientId = 'P_' + Math.random().toString(36).substring(2, 6).toUpperCase();
  let isDashboard = false;

  ws.send(JSON.stringify({
    type: 'init',
    id: clientId,
    players: state.players,
    traces: state.traces,
    structures: state.structures,
    markers: state.markers,
    walls: state.walls,
  }));

  ws.on('message', (raw) => {
    try {
      const data = JSON.parse(raw);

      if (data.type === 'registerRole') {
        if (data.role === 'dashboard') {
          isDashboard = true;
          delete state.players[clientId];
          broadcast({ type: 'playerLeft', id: clientId });
          ws.send(JSON.stringify({
            type: 'stateSnapshot',
            players: state.players,
            traces: state.traces,
            structures: state.structures,
            markers: state.markers,
            walls: state.walls,
          }));
          return;
        } else {
          if (data.id) clientId = data.id;
          const px = Number(data.x) || 0;
          const pz = Number(data.z) || 0;
          state.players[clientId] = {
            id: clientId,
            x: px,
            z: pz,
            yaw: Number(data.yaw) || 0,
            speed: Number(data.speed) || 0,
            hasTargetPhoto: !!data.hasTargetPhoto,
            lastSeen: Date.now(),
          };
          recordTrace(clientId, px, pz);
          broadcast({ type: 'playerJoined', player: state.players[clientId], traces: state.traces }, ws);
        }
      }

      if (isDashboard) return;

      if (!state.players[clientId]) {
        state.players[clientId] = {
          id: clientId,
          x: 0,
          z: 0,
          yaw: 0,
          speed: 0,
          hasTargetPhoto: true,
          lastSeen: Date.now(),
        };
      }
      const p = state.players[clientId];
      p.lastSeen = Date.now();

      if (data.type === 'move') {
        p.x = Number(data.x) || 0;
        p.z = Number(data.z) || 0;
        p.yaw = Number(data.yaw) || 0;
        p.speed = Number(data.speed) || 0;
        recordTrace(clientId, p.x, p.z);
        broadcast({ type: 'playerMove', player: p, traces: state.traces }, ws);
      }
      else if (data.type === 'syncStructures') {
        if (Array.isArray(data.structures)) {
          // 更新該玩家生成的建築清單
          state.structures = state.structures.filter((s) => s.owner !== clientId).concat(
            data.structures.map((s) => Object.assign({}, s, { owner: clientId }))
          );
          broadcast({ type: 'structuresUpdated', structures: state.structures }, ws);
        }
      }
      else if (data.type === 'snapMarker') {
        const m = state.markers[data.id];
        if (m) {
          // 1m 網格允許在 +-0.42m 內微調貼合真實牆面
          const dx = Math.max(-0.42, Math.min(0.42, (data.x || m.gx) - m.gx));
          const dz = Math.max(-0.42, Math.min(0.42, (data.z || m.gz) - m.gz));
          m.x = +(m.gx + dx).toFixed(2);
          m.z = +(m.gz + dz).toFixed(2);
          m.y = +(data.y || 1.2).toFixed(2);
          m.nx = data.nx || 0;
          m.nz = data.nz || 1;
          m.snapped = true;
          broadcast({ type: 'markerUpdated', marker: m });
        }
      }
      else if (data.type === 'activateMarker') {
        const m = state.markers[data.id];
        if (m) {
          m.active = true;
          m.activatedBy = clientId;
          m.timestamp = Date.now();
          if (Array.isArray(data.walls)) {
            data.walls.forEach((w) => {
              if (state.walls.length < 400) state.walls.push(w);
            });
          }
          broadcast({
            type: 'markerActivated',
            marker: m,
            playerId: clientId,
            walls: state.walls,
          });
        }
      }
    } catch (e) {
      console.error('WS parse error:', e);
    }
  });

  ws.on('close', () => {
    if (state.players[clientId]) {
      delete state.players[clientId];
      broadcast({ type: 'playerLeft', id: clientId });
    }
  });
});

setInterval(() => {
  const now = Date.now();
  for (const id in state.players) {
    if (now - state.players[id].lastSeen > 30000) {
      delete state.players[id];
    }
  }
  broadcast({
    type: 'stateSnapshot',
    players: state.players,
    traces: state.traces,
    structures: state.structures,
    markers: state.markers,
    walls: state.walls,
  });
}, 500);

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`====================================================`);
  console.log(`AR 1m網格定位與多人連機伺服器已啟動：http://localhost:${PORT}`);
  console.log(`伺服器即時地圖監控端：http://localhost:${PORT}/dashboard.html`);
  console.log(`====================================================`);
});
