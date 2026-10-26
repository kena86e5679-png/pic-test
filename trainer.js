// ============================================================================
// SPATIAL AI TRAINER (trainer.js)
//   1. 結合 COCO-SSD 物件偵測 + 地牆透視幾何分析 (Perspective Horizon & Verticals)
//   2. 內建 TensorFlow.js 多任務神經網路 (Spatial MLP Classifier)：
//      輸入 18 維空間視覺特徵向量 -> 預測 [空間型態, 中央動線阻擋狀態, 最佳虛擬建築補充類型]
//   3. 支援使用者在網頁上新增/修改框選區域 (轉角、門、牆壁死路、阻擋桌椅) 與前後景層次，
//      按下「確認修正並回饋訓練模型」即刻執行 model.fit() 更新權重，並儲存至 localStorage & JSON
// ============================================================================
(() => {
'use strict';

const $ = (id) => document.getElementById(id);
const cv = $('workCv');
const ctx = cv.getContext('2d');

let cocoModel = null;
let spatialNet = null; // TensorFlow.js 可訓練神經網路
let currentImg = null;
let currentFeatVec = new Array(18).fill(0); // 18維空間幾何與語義特徵向量
let elements = []; // [{ id, x, y, w, h, type, layer, action, dist, score }]
let selectedElId = null;

// 訓練數據庫 (自動持久化於 localStorage，亦可匯出成 spatial_dataset.json)
let dataset = [];

const TYPOLOGY_KEYS = ['corridor', 'corner_space', 'cluttered_room', 'dead_end', 'open_hall'];
const PATHC_KEYS = ['clear', 'blocked_obj', 'blocked_wall', 'door_way'];
const SUPP_KEYS = ['room', 'alley', 'cover', 'wall'];

const TYPE_LABELS = {
  corner_convex: { name: '陽角 (凸轉角)', color: '#6effe1', defaultAction: 'attach_wall' },
  corner_concave: { name: '陰角 (凹轉角)', color: '#38bdf8', defaultAction: 'attach_room' },
  door_open: { name: '門扇／通道開口', color: '#a78bfa', defaultAction: 'pass_through' },
  wall_deadend: { name: '封閉牆面／死路', color: '#f87171', defaultAction: 'stop_prioritize_real' },
  obstacle_block: { name: '阻擋物 (桌椅家具)', color: '#fb923c', defaultAction: 'detour_avoid' },
  obstacle_cover: { name: '半高掩體 (矮桌/沙發)', color: '#facc15', defaultAction: 'use_as_cover' },
};

const LAYER_LABELS = {
  foreground: '前景 (0~2m)',
  midground: '中景 (2~5m)',
  background: '後景 (5m+)',
};

const ACTION_LABELS = {
  attach_room: '附著生成 2.4m 實心房間',
  attach_wall: '附著延伸 2.4m 實心牆',
  use_as_cover: '作為掩體／補充 1.1m 胸牆',
  detour_avoid: '阻擋動線（實體優先・需繞行）',
  stop_prioritize_real: '死路邊界（禁止虛擬穿牆）',
  pass_through: '保持通道淨空（禁止封堵）',
};

function log(msg) {
  const box = $('logBox');
  const time = new Date().toLocaleTimeString();
  box.innerHTML = `<div>[${time}] ${msg}</div>` + box.innerHTML;
}

// ============================================================================
// 1. 建立可於瀏覽器端即時訓練的 TensorFlow.js 神經網路模型
//    輸入：18 維影像空間特徵 (地平線高度、左中右邊緣密度、垂直線比例、前景遮擋率、桌椅面積比、門框特徵等)
//    輸出：13 維編碼 (5種空間型態 + 4種中央阻擋狀態 + 4種補充建築建議)
// ============================================================================
async function initModels() {
  log('正在初始化 TensorFlow.js 空間神經網路與 COCO-SSD 視覺模型...');
  try {
    // 嘗試載入先前在瀏覽器訓練過的權重
    try {
      spatialNet = await tf.loadLayersModel('localstorage://ar-spatial-mlp-v1');
      spatialNet.compile({
        optimizer: tf.train.adam(0.015),
        loss: 'meanSquaredError',
      });
      log('已從本地載入先前訓練的空間判讀神經網路權重！');
    } catch (e) {
      spatialNet = tf.sequential();
      spatialNet.add(tf.layers.dense({ inputShape: [18], units: 32, activation: 'relu' }));
      spatialNet.add(tf.layers.dense({ units: 24, activation: 'relu' }));
      spatialNet.add(tf.layers.dense({ units: 13, activation: 'sigmoid' }));
      spatialNet.compile({
        optimizer: tf.train.adam(0.015),
        loss: 'meanSquaredError',
      });
      log('已建立全新 3 層式 Spatial MLP 神經網路 (18 -> 32 -> 24 -> 13)。');
    }

    // 載入已儲存的訓練數據集
    const rawDs = localStorage.getItem('arSpatialDataset.v1');
    if (rawDs) {
      dataset = JSON.parse(rawDs);
      $('stSamples').textContent = dataset.length;
    }

    if (window.cocoSsd) {
      cocoModel = await window.cocoSsd.load({ base: 'lite_mobilenet_v2' });
      log('COCO-SSD 室內物件偵測模型載入完成。');
    }
  } catch (err) {
    log('模型初始化警告: ' + err.message);
  }
  // 預設載入一張合成範例室內空間供立刻體驗
  loadSyntheticRoomSample();
}

// ============================================================================
// 2. 單張照片的空間幾何與語義特徵提取器 (18-Dim Spatial Feature Extractor)
// ============================================================================
function extractImageGeometryAndLines(imgEl) {
  const W = 160, H = 104;
  const off = document.createElement('canvas');
  off.width = W; off.height = H;
  const octx = off.getContext('2d');
  octx.drawImage(imgEl, 0, 0, W, H);
  const px = octx.getImageData(0, 0, W, H).data;

  const gray = new Float32Array(W * H);
  const gx = new Float32Array(W * H);
  const gy = new Float32Array(W * H);
  const mag = new Float32Array(W * H);

  for (let i = 0; i < W * H; i++) {
    gray[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
  }

  let edgeL = 0, edgeC = 0, edgeR = 0;
  let vertEdges = 0, horizEdges = 0;
  let lowerCenterEdge = 0, upperCenterEdge = 0;

  // 記錄每一直行的垂直邊緣強度（用於找尋牆角垂直線與門框）
  const colVertScore = new Float32Array(W);
  // 記錄每一橫列的水平邊緣強度（用於找尋地牆交界踢腳線）
  const rowHorizScore = new Float32Array(H);

  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const k = y * W + x;
      const dx = (-gray[k - W - 1] + gray[k - W + 1] - 2 * gray[k - 1] + 2 * gray[k + 1] - gray[k + W - 1] + gray[k + W + 1]) / 4;
      const dy = (-gray[k - W - 1] - 2 * gray[k - W] - gray[k - W + 1] + gray[k + W - 1] + 2 * gray[k + W] + gray[k + W + 1]) / 4;
      const m = Math.hypot(dx, dy);
      gx[k] = dx; gy[k] = dy; mag[k] = m;

      if (m > 14) {
        if (x < W * 0.33) edgeL++;
        else if (x < W * 0.66) edgeC++;
        else edgeR++;

        if (Math.abs(dx) > Math.abs(dy) * 1.5) {
          vertEdges++;
          colVertScore[x] += m;
        } else if (Math.abs(dy) > Math.abs(dx) * 1.5) {
          horizEdges++;
          if (y > H * 0.35 && y < H * 0.92) rowHorizScore[y] += m;
        }

        if (y > H * 0.58 && x > W * 0.28 && x < W * 0.72) lowerCenterEdge++;
        if (y <= H * 0.58 && x > W * 0.28 && x < W * 0.72) upperCenterEdge++;
      }
    }
  }

  // 找尋最強的地牆水平交界高度 (Horizon / Floor-Wall Boundary Y)
  let bestFloorY = H * 0.68, bestRowScore = 0;
  for (let y = Math.floor(H * 0.40); y < Math.floor(H * 0.90); y++) {
    if (rowHorizScore[y] > bestRowScore) {
      bestRowScore = rowHorizScore[y];
      bestFloorY = y;
    }
  }
  const floorBoundaryNorm = bestFloorY / H; // 數值越大代表牆離鏡頭越近 (死路)，越小代表景深越深

  // 找尋前 3 強的垂直結構線 (轉角或門框候選)
  const vPeaks = [];
  for (let x = 10; x < W - 10; x++) {
    const score = colVertScore[x - 1] + colVertScore[x] + colVertScore[x + 1];
    if (score > H * 18) {
      let isMax = true;
      for (let d = -6; d <= 6; d++) {
        if (colVertScore[x + d] > colVertScore[x]) { isMax = false; break; }
      }
      if (isMax) vPeaks.push({ xNorm: x / W, score });
    }
  }
  vPeaks.sort((a, b) => b.score - a.score);

  return {
    floorBoundaryNorm,
    edgeLNorm: edgeL / (W * H * 0.33),
    edgeCNorm: edgeC / (W * H * 0.34),
    edgeRNorm: edgeR / (W * H * 0.33),
    vertRatio: vertEdges / Math.max(1, vertEdges + horizEdges),
    lowerCenterTexture: lowerCenterEdge / (W * H * 0.2),
    upperCenterTexture: upperCenterEdge / (W * H * 0.2),
    vPeaks: vPeaks.slice(0, 4),
  };
}

async function analyzeCurrentPhoto() {
  if (!currentImg) return;
  elements = [];
  const geo = extractImageGeometryAndLines(currentImg);

  // 1. 執行 COCO-SSD 偵測室內阻擋物 (桌子、椅子、沙發、門、盆栽、螢幕等)
  let cocoPreds = [];
  if (cocoModel) {
    try { cocoPreds = await cocoModel.detect(currentImg); } catch (e) {}
  }

  let fgObstacleArea = 0;
  let centerBlockedByObj = 0;
  let doorDetected = 0;

  for (const p of cocoPreds) {
    const [bx, by, bw, bh] = p.bbox;
    const x = bx / currentImg.width;
    const y = by / currentImg.height;
    const w = bw / currentImg.width;
    const h = bh / currentImg.height;
    const bottomY = y + h;

    // 根據物體底部在畫面中的 Y 位置推算前、中、後景距離
    const layer = bottomY > 0.76 ? 'foreground' : bottomY > 0.52 ? 'midground' : 'background';
    const estDist = +(1.5 / Math.max(0.12, bottomY - 0.32)).toFixed(1);

    let type = 'obstacle_block';
    if (['couch', 'chair', 'bed', 'potted plant'].includes(p.class)) {
      type = 'obstacle_cover';
    } else if (['dining table', 'desk', 'refrigerator', 'tv'].includes(p.class)) {
      type = 'obstacle_block';
    } else if (p.class === 'door') {
      type = 'door_open';
      doorDetected = 1;
    }

    if (layer === 'foreground' || layer === 'midground') {
      fgObstacleArea += w * h;
      if (x + w > 0.35 && x < 0.65 && bottomY > 0.62) {
        centerBlockedByObj = 1;
      }
    }

    elements.push({
      id: 'EL_' + Math.random().toString(36).substring(2, 6).toUpperCase(),
      x: +x.toFixed(3),
      y: +y.toFixed(3),
      w: +w.toFixed(3),
      h: +h.toFixed(3),
      type,
      layer,
      action: TYPE_LABELS[type].defaultAction,
      dist: estDist,
      rawClass: p.class,
    });
  }

  // 2. 根據垂直結構線峰值，自動標註空間中的「轉角 (Corner)」或「門扇 (Door)」
  if (geo.vPeaks.length >= 2 && Math.abs(geo.vPeaks[0].xNorm - geo.vPeaks[1].xNorm) > 0.12 && Math.abs(geo.vPeaks[0].xNorm - geo.vPeaks[1].xNorm) < 0.36) {
    // 兩條平行接近的強垂直線 -> 判定為門扇開口或通道
    const x1 = Math.min(geo.vPeaks[0].xNorm, geo.vPeaks[1].xNorm);
    const x2 = Math.max(geo.vPeaks[0].xNorm, geo.vPeaks[1].xNorm);
    doorDetected = 1;
    elements.push({
      id: 'EL_' + Math.random().toString(36).substring(2, 6).toUpperCase(),
      x: +x1.toFixed(3),
      y: 0.14,
      w: +(x2 - x1).toFixed(3),
      h: +(geo.floorBoundaryNorm - 0.14).toFixed(3),
      type: 'door_open',
      layer: geo.floorBoundaryNorm > 0.72 ? 'foreground' : 'midground',
      action: 'pass_through',
      dist: +(1.6 / Math.max(0.15, geo.floorBoundaryNorm - 0.3)).toFixed(1),
    });
  } else {
    geo.vPeaks.slice(0, 2).forEach((pk, idx) => {
      const isConvex = idx === 0 && geo.edgeCNorm > geo.edgeLNorm;
      const cType = isConvex ? 'corner_convex' : 'corner_concave';
      elements.push({
        id: 'EL_' + Math.random().toString(36).substring(2, 6).toUpperCase(),
        x: +Math.max(0.02, pk.xNorm - 0.06).toFixed(3),
        y: 0.12,
        w: 0.12,
        h: +(Math.min(0.82, geo.floorBoundaryNorm) - 0.12).toFixed(3),
        type: cType,
        layer: 'midground',
        action: TYPE_LABELS[cType].defaultAction,
        dist: +(1.8 / Math.max(0.15, geo.floorBoundaryNorm - 0.28)).toFixed(1),
      });
    });
  }

  // 3. 若地牆交線非常低 (floorBoundaryNorm > 0.76) 且中央紋理平滑 -> 判定前方為封閉牆面死路
  if (geo.floorBoundaryNorm > 0.76 && geo.upperCenterTexture < 0.18 && !doorDetected) {
    elements.push({
      id: 'EL_WALL',
      x: 0.22,
      y: 0.08,
      w: 0.56,
      h: +(geo.floorBoundaryNorm - 0.08).toFixed(3),
      type: 'wall_deadend',
      layer: 'foreground',
      action: 'stop_prioritize_real',
      dist: 1.6,
    });
  }

  // 4. 組裝 18 維特徵向量，送入 TensorFlow.js Spatial MLP 進行推論
  currentFeatVec = [
    geo.floorBoundaryNorm,
    geo.edgeLNorm,
    geo.edgeCNorm,
    geo.edgeRNorm,
    geo.vertRatio,
    geo.lowerCenterTexture,
    geo.upperCenterTexture,
    fgObstacleArea,
    centerBlockedByObj,
    doorDetected,
    geo.vPeaks.length / 4,
    elements.filter((e) => e.type.startsWith('corner')).length / 3,
    elements.filter((e) => e.type.startsWith('obstacle')).length / 4,
    elements.filter((e) => e.layer === 'foreground').length / 4,
    elements.filter((e) => e.layer === 'midground').length / 4,
    elements.filter((e) => e.layer === 'background').length / 4,
    geo.edgeLNorm > geo.edgeRNorm ? 1 : 0,
    geo.floorBoundaryNorm > 0.74 ? 1 : 0,
  ];

  runNeuralInference();
  renderCanvas();
  renderElementList();
}

function runNeuralInference() {
  if (!spatialNet) return;
  tf.tidy(() => {
    const input = tf.tensor2d([currentFeatVec]);
    const pred = spatialNet.predict(input).dataSync();

    // 若已經有使用者訓練過的樣本，優先使用神經網路預測；否則結合啟發式初始值
    if (dataset.length >= 2) {
      let bestTypo = 0, bestPath = 0, bestSupp = 0;
      for (let i = 0; i < 5; i++) if (pred[i] > pred[bestTypo]) bestTypo = i;
      for (let i = 0; i < 4; i++) if (pred[5 + i] > pred[5 + bestPath]) bestPath = i;
      for (let i = 0; i < 4; i++) if (pred[9 + i] > pred[9 + bestSupp]) bestSupp = i;

      $('selTypology').value = TYPOLOGY_KEYS[bestTypo];
      $('selPathC').value = PATHC_KEYS[bestPath];
      $('selSupplement').value = SUPP_KEYS[bestSupp];
      const conf = Math.round(Math.max(pred[bestTypo], pred[5 + bestPath], pred[9 + bestSupp]) * 100);
      $('stAcc').textContent = conf + '%';
      $('aiPredBadge').textContent = `MLP 模型推論 (信心度 ${conf}%)`;
    } else {
      // 初始啟發式推論
      const hasBlock = elements.some((e) => e.type.startsWith('obstacle') && e.layer !== 'background');
      const hasWall = elements.some((e) => e.type === 'wall_deadend');
      const hasDoor = elements.some((e) => e.type === 'door_open');
      const hasCorner = elements.some((e) => e.type.startsWith('corner'));

      $('selTypology').value = hasWall ? 'dead_end' : hasBlock ? 'cluttered_room' : hasCorner ? 'corner_space' : 'corridor';
      $('selPathC').value = hasBlock ? 'blocked_obj' : hasWall ? 'blocked_wall' : hasDoor ? 'door_way' : 'clear';
      $('selSupplement').value = hasWall ? 'room' : hasBlock ? 'cover' : hasCorner ? 'alley' : 'wall';
      $('stAcc').textContent = '啟發式初始';
    }
  });
}

// ============================================================================
// 3. 回饋修正與瀏覽器端即時訓練 (Online Backpropagation Training)
// ============================================================================
function buildTargetVectorFromUI() {
  const vec = new Array(13).fill(0);
  const tIdx = Math.max(0, TYPOLOGY_KEYS.indexOf($('selTypology').value));
  const pIdx = Math.max(0, PATHC_KEYS.indexOf($('selPathC').value));
  const sIdx = Math.max(0, SUPP_KEYS.indexOf($('selSupplement').value));
  vec[tIdx] = 1;
  vec[5 + pIdx] = 1;
  vec[9 + sIdx] = 1;
  return vec;
}

async function trainModelWithCorrection() {
  if (!spatialNet) return;
  const targetVec = buildTargetVectorFromUI();

  // 將使用者修正後的 Ground Truth 存入數據庫
  const record = {
    id: 'SAMPLE_' + (dataset.length + 1),
    timestamp: new Date().toISOString(),
    features: currentFeatVec.map((v) => +Number(v).toFixed(4)),
    target: targetVec,
    labels: {
      typology: $('selTypology').value,
      supplement: $('selSupplement').value,
      pathL: $('selPathL').value,
      pathC: $('selPathC').value,
      pathR: $('selPathR').value,
      maxDepth: parseFloat($('inpDepth').value) || 4.5,
    },
    elements: JSON.parse(JSON.stringify(elements)),
  };
  dataset.push(record);
  localStorage.setItem('arSpatialDataset.v1', JSON.stringify(dataset));
  $('stSamples').textContent = dataset.length;

  // 準備訓練 Batch (包含歷史所有已修正樣本，進行 25 個 Epochs 微調)
  const xs = tf.tensor2d(dataset.map((d) => d.features));
  const ys = tf.tensor2d(dataset.map((d) => d.target));

  $('modelStatus').textContent = 'Training...';
  const history = await spatialNet.fit(xs, ys, {
    epochs: 25,
    shuffle: true,
  });

  const finalLoss = history.history.loss[history.history.loss.length - 1];
  $('stLoss').textContent = finalLoss.toFixed(4);
  $('modelStatus').textContent = 'Updated ✓';

  xs.dispose();
  ys.dispose();

  try {
    await spatialNet.save('localstorage://ar-spatial-mlp-v1');
  } catch (e) {}

  runNeuralInference();
  log(`已完成第 #${dataset.length} 筆空間修正訓練！Loss 降至 ${finalLoss.toFixed(4)}，權重已更新並寫入本地模型。`);
}

// ============================================================================
// 4. 畫布繪製與滑鼠互動框選新增空間元素
// ============================================================================
let dragBox = null;

function renderCanvas() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (currentImg) {
    ctx.drawImage(currentImg, 0, 0, cv.width, cv.height);
  } else {
    ctx.fillStyle = '#0c0f16';
    ctx.fillRect(0, 0, cv.width, cv.height);
  }

  // 繪製地牆交界估計基準線
  if (currentFeatVec[0] > 0) {
    const fy = currentFeatVec[0] * cv.height;
    ctx.save();
    ctx.strokeStyle = 'rgba(110, 255, 225, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(0, fy);
    ctx.lineTo(cv.width, fy);
    ctx.stroke();
    ctx.fillStyle = 'rgba(110, 255, 225, 0.8)';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('地牆交界基準線 (Floor-Wall Horizon)', 10, fy - 6);
    ctx.restore();
  }

  // 繪製所有空間元素與物件框
  for (const el of elements) {
    const info = TYPE_LABELS[el.type] || TYPE_LABELS.obstacle_block;
    const rx = el.x * cv.width, ry = el.y * cv.height;
    const rw = el.w * cv.width, rh = el.h * cv.height;

    ctx.save();
    ctx.strokeStyle = info.color;
    ctx.lineWidth = el.id === selectedElId ? 3.0 : 2.0;
    if (el.layer === 'background') ctx.setLineDash([4, 4]);
    else if (el.layer === 'midground') ctx.setLineDash([8, 4]);
    ctx.strokeRect(rx, ry, rw, rh);

    ctx.fillStyle = info.color + '26';
    ctx.fillRect(rx, ry, rw, rh);

    // 標籤背景與文字
    ctx.setLineDash([]);
    const tag = `${info.name} [${LAYER_LABELS[el.layer]}] ${el.dist}m`;
    ctx.font = 'bold 11px -apple-system, "PingFang TC", monospace';
    const tw = ctx.measureText(tag).width + 10;
    ctx.fillStyle = 'rgba(8, 9, 12, 0.85)';
    ctx.fillRect(rx, Math.max(0, ry - 20), tw, 20);
    ctx.fillStyle = info.color;
    ctx.fillText(tag, rx + 5, Math.max(14, ry - 6));
    ctx.restore();
  }

  // 繪製正在拖曳中的新框
  if (dragBox) {
    ctx.save();
    ctx.strokeStyle = '#6effe1';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(dragBox.x, dragBox.y, dragBox.w, dragBox.h);
    ctx.restore();
  }
}

function renderElementList() {
  const container = $('elementList');
  $('elCount').textContent = `${elements.length} 項`;
  if (!elements.length) {
    container.innerHTML = `<div style="color:var(--dim);font-size:12px;padding:8px;">尚未偵測到特定物件，可在左側照片上直接按住滑鼠拖曳框選「轉角、門、牆面或桌椅」。</div>`;
    return;
  }
  container.innerHTML = elements.map((el) => {
    const info = TYPE_LABELS[el.type] || TYPE_LABELS.obstacle_block;
    return `
      <div class="el-item ${el.id === selectedElId ? 'active' : ''}" data-id="${el.id}">
        <div class="el-head">
          <span style="color:${info.color};">● ${info.name} (${el.dist}m)</span>
          <button class="del-btn" data-del="${el.id}">刪除誤判</button>
        </div>
        <div class="el-controls">
          <select data-field="type" data-id="${el.id}">
            ${Object.keys(TYPE_LABELS).map((k) => `<option value="${k}" ${el.type === k ? 'selected' : ''}>${TYPE_LABELS[k].name}</option>`).join('')}
          </select>
          <select data-field="layer" data-id="${el.id}">
            ${Object.keys(LAYER_LABELS).map((k) => `<option value="${k}" ${el.layer === k ? 'selected' : ''}>${LAYER_LABELS[k]}</option>`).join('')}
          </select>
          <select data-field="action" data-id="${el.id}">
            ${Object.keys(ACTION_LABELS).map((k) => `<option value="${k}" ${el.action === k ? 'selected' : ''}>${ACTION_LABELS[k]}</option>`).join('')}
          </select>
        </div>
      </div>
    `;
  }).join('');

  container.querySelectorAll('select').forEach((sel) => {
    sel.addEventListener('change', (e) => {
      const id = e.target.getAttribute('data-id');
      const field = e.target.getAttribute('data-field');
      const target = elements.find((x) => x.id === id);
      if (target) {
        target[field] = e.target.value;
        renderCanvas();
        renderElementList();
      }
    });
  });

  container.querySelectorAll('[data-del]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const id = e.target.getAttribute('data-del');
      elements = elements.filter((x) => x.id !== id);
      renderCanvas();
      renderElementList();
    });
  });
}

// 畫布滑鼠框選事件
function getCanvasPos(e) {
  const rect = cv.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) / rect.width) * cv.width,
    y: ((e.clientY - rect.top) / rect.height) * cv.height,
  };
}

cv.addEventListener('pointerdown', (e) => {
  const p = getCanvasPos(e);
  dragBox = { startX: p.x, startY: p.y, x: p.x, y: p.y, w: 0, h: 0 };
});

window.addEventListener('pointermove', (e) => {
  const p = getCanvasPos(e);
  if (p.x >= 0 && p.x <= cv.width && p.y >= 0 && p.y <= cv.height) {
    $('cursorCoord').textContent = `X: ${(p.x / cv.width).toFixed(2)}, Y: ${(p.y / cv.height).toFixed(2)}`;
  }
  if (!dragBox) return;
  dragBox.x = Math.min(dragBox.startX, p.x);
  dragBox.y = Math.min(dragBox.startY, p.y);
  dragBox.w = Math.abs(p.x - dragBox.startX);
  dragBox.h = Math.abs(p.y - dragBox.startY);
  renderCanvas();
});

window.addEventListener('pointerup', () => {
  if (!dragBox) return;
  if (dragBox.w > 18 && dragBox.h > 18) {
    const type = $('drawType').value;
    const layer = $('drawLayer').value;
    const bottomY = (dragBox.y + dragBox.h) / cv.height;
    const estDist = +(1.6 / Math.max(0.15, bottomY - 0.28)).toFixed(1);
    const newEl = {
      id: 'EL_' + Math.random().toString(36).substring(2, 6).toUpperCase(),
      x: +(dragBox.x / cv.width).toFixed(3),
      y: +(dragBox.y / cv.height).toFixed(3),
      w: +(dragBox.w / cv.width).toFixed(3),
      h: +(dragBox.h / cv.height).toFixed(3),
      type,
      layer,
      action: TYPE_LABELS[type].defaultAction,
      dist: estDist,
    };
    elements.push(newEl);
    selectedElId = newEl.id;
    log(`已手動標註新增 [${TYPE_LABELS[type].name}] 於 ${LAYER_LABELS[layer]}。`);
  }
  dragBox = null;
  renderCanvas();
  renderElementList();
});

// ============================================================================
// 5. 載入範例合成室內空間與照片上傳匯出
// ============================================================================
function loadSyntheticRoomSample() {
  const off = document.createElement('canvas');
  off.width = 800; off.height = 520;
  const c = off.getContext('2d');

  // 繪製一個具有天花板、左右透視牆、遠處門扇、轉角與前景阻擋桌子的測試室內空間
  c.fillStyle = '#1e232d'; c.fillRect(0, 0, 800, 110); // 天花板
  c.fillStyle = '#2b313e'; c.fillRect(0, 110, 800, 240); // 正面遠牆
  c.fillStyle = '#222733'; // 左側透視牆
  c.beginPath(); c.moveTo(0, 0); c.lineTo(190, 110); c.lineTo(190, 350); c.lineTo(0, 520); c.closePath(); c.fill();
  c.fillStyle = '#252b38'; // 右側透視牆
  c.beginPath(); c.moveTo(800, 0); c.lineTo(610, 110); c.lineTo(610, 350); c.lineTo(800, 520); c.closePath(); c.fill();
  c.fillStyle = '#363e4d'; // 地板
  c.beginPath(); c.moveTo(190, 350); c.lineTo(610, 350); c.lineTo(800, 520); c.lineTo(0, 520); c.closePath(); c.fill();

  // 結構線條 (牆角線與地牆交界線)
  c.strokeStyle = '#6c7a96'; c.lineWidth = 3;
  c.beginPath();
  c.moveTo(190, 110); c.lineTo(190, 350); // 左陰角垂直線
  c.moveTo(610, 110); c.lineTo(610, 350); // 右陰角垂直線
  c.moveTo(190, 350); c.lineTo(610, 350); // 地牆水平交線
  c.moveTo(190, 350); c.lineTo(0, 520);
  c.moveTo(610, 350); c.lineTo(800, 520);
  c.stroke();

  // 中景門扇開口
  c.fillStyle = '#151922';
  c.fillRect(360, 150, 85, 200);
  c.strokeStyle = '#9aa8c7';
  c.strokeRect(360, 150, 85, 200);

  // 前景阻擋物 (一張擋在中央偏右的桌子)
  c.fillStyle = '#5c4332';
  c.fillRect(390, 340, 190, 110);
  c.strokeStyle = '#8c6b53';
  c.strokeRect(390, 340, 190, 110);

  const img = new Image();
  img.onload = () => {
    currentImg = img;
    analyzeCurrentPhoto();
    // 預先補上範例前景桌子的標註示範
    elements.push({
      id: 'EL_DEMO_DESK',
      x: +(390 / 800).toFixed(3),
      y: +(340 / 520).toFixed(3),
      w: +(190 / 800).toFixed(3),
      h: +(110 / 520).toFixed(3),
      type: 'obstacle_block',
      layer: 'foreground',
      action: 'detour_avoid',
      dist: 1.8,
    });
    $('selPathC').value = 'blocked_obj';
    renderCanvas();
    renderElementList();
    log('已載入範例空間（包含左右陰角、遠處門扇與前景阻擋桌子）。');
  };
  img.src = off.toDataURL();
}

$('fileInput').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = new Image();
    img.onload = () => {
      currentImg = img;
      log(`已讀取空間照片: ${file.name} (${img.width}x${img.height})，開始執行空間層次判讀...`);
      analyzeCurrentPhoto();
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
});

$('btnSample').addEventListener('click', loadSyntheticRoomSample);
$('btnReAnalyze').addEventListener('click', analyzeCurrentPhoto);
$('btnTrain').addEventListener('click', trainModelWithCorrection);

$('btnExport').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(dataset, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `spatial_dataset_${dataset.length}samples.json`;
  a.click();
  log(`已匯出包含 ${dataset.length} 筆空間標註的訓練數據庫 JSON！`);
});

$('importJson').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async (ev) => {
    try {
      const imported = JSON.parse(ev.target.result);
      if (Array.isArray(imported) && imported.length > 0) {
        dataset = dataset.concat(imported);
        localStorage.setItem('arSpatialDataset.v1', JSON.stringify(dataset));
        $('stSamples').textContent = dataset.length;
        log(`成功匯入 ${imported.length} 筆空間數據，正在批次訓練模型...`);
        await trainModelWithCorrection();
      }
    } catch (err) {
      log('JSON 匯入格式錯誤: ' + err.message);
    }
  };
  reader.readAsText(file);
});

initModels();
})();
