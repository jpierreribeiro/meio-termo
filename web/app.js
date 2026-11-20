// Pipeline no navegador. Espelha referencia_web.py, que foi validado contra o recorte
// do treino (fotos.recortar). Nada sai do aparelho: a foto é lida, processada e descartada.
/* global ort */

const ENTRADA = 640;                      // YuNet 2023mar: entrada fixa 640x640, BGR 0-255
const LADO = 160, OLHOS_Y = 0.40, DIST_OLHOS = 0.28, MIN_DIST = 12, SCORE_MIN = 0.8;
// fotos de celular e webcam (luz de tela, rosto um pouco de lado) às vezes ficam abaixo de 0,8,
// que foi calibrado nas fotos de urna; nesse caso tenta de novo com 0,6
const SCORE_RESERVA = 0.6, SCORE_PISO = 0.5;
const ZOOMS = [0.95, 1.0, 1.05];          // média de 6 variações (3 escalas x espelho)
const MEDIA = [0.485, 0.456, 0.406], DESVIO = [0.229, 0.224, 0.225];
const MAX_LADO_FOTO = 2048;

let sessoes = null;

// OffscreenCanvas com 2d só existe no Safari 16.4+; antes disso, um <canvas> comum serve
function tela(w, h) {
  if (typeof OffscreenCanvas !== "undefined") {
    try { const c = new OffscreenCanvas(w, h); if (c.getContext("2d")) return c; } catch { /* segue */ }
  }
  const c = document.createElement("canvas"); c.width = w; c.height = h; return c;
}

// decodifica a foto: createImageBitmap respeitando a orientação EXIF; se o navegador
// recusar (opções não suportadas, formato estranho), tenta pelo <img>, que no Safari
// também abre HEIC
async function decodificar(arquivo) {
  try { return await createImageBitmap(arquivo, { imageOrientation: "from-image" }); } catch { /* segue */ }
  const url = URL.createObjectURL(arquivo);
  try {
    const img = new Image();
    img.decoding = "async"; img.src = url;
    await img.decode();
    return img;
  } catch {
    throw new Error("este formato de imagem não abriu neste navegador (tente JPG ou PNG)");
  } finally { URL.revokeObjectURL(url); }
}

export async function carregarModelos(progresso) {
  if (sessoes) return sessoes;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const baixar = async (url, peso) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`não consegui baixar ${url} (${r.status})`);
    const total = +r.headers.get("content-length") || 0;
    const leitor = r.body.getReader();
    const partes = [];
    let recebido = 0;
    for (;;) {
      const { done, value } = await leitor.read();
      if (done) break;
      partes.push(value);
      recebido += value.length;
      if (total) progresso?.(peso.base + peso.fatia * recebido / total);
    }
    const buf = new Uint8Array(recebido);
    let o = 0;
    for (const p of partes) { buf.set(p, o); o += p.length; }
    return buf;
  };
  const [det, cls] = await Promise.all([
    baixar("modelo/face_detection_yunet_2023mar.onnx", { base: 0, fatia: 0.03 }),
    baixar("modelo/aluno2_w16.onnx", { base: 0.03, fatia: 0.97 }),
  ]);
  const opcoes = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  sessoes = {
    det: await ort.InferenceSession.create(det, opcoes),
    cls: await ort.InferenceSession.create(cls, opcoes),
  };
  return sessoes;
}

// ---------- foto -> pixels ----------
export async function lerFoto(arquivo) {
  const bmp = await decodificar(arquivo);
  const bw = bmp.naturalWidth || bmp.width, bh = bmp.naturalHeight || bmp.height;
  const esc = Math.min(1, MAX_LADO_FOTO / Math.max(bw, bh));
  const w = Math.max(1, Math.round(bw * esc)), h = Math.max(1, Math.round(bh * esc));
  const c = tela(w, h);
  const g = c.getContext("2d");
  g.imageSmoothingQuality = "high";
  g.drawImage(bmp, 0, 0, w, h);
  bmp.close?.();
  return g.getImageData(0, 0, w, h);   // RGBA
}

// ---------- detecção (YuNet) ----------
async function detectar(img, det) {
  // margem de 25% com borda replicada (como no teste externo): ajuda quando o rosto
  // ocupa a foto inteira, caso comum em selfies
  const m = Math.floor(Math.max(img.width, img.height) / 4);
  const W = img.width + 2 * m, H = img.height + 2 * m;
  const esc = Math.min(ENTRADA / W, ENTRADA / H);
  const nw = Math.max(1, Math.round(img.width * esc)), nh = Math.max(1, Math.round(img.height * esc));
  const mp = Math.round(m * esc);
  const c = tela(img.width, img.height);
  c.getContext("2d").putImageData(img, 0, 0);
  const p = tela(nw, nh);
  const gp = p.getContext("2d");
  gp.imageSmoothingQuality = "high";
  gp.drawImage(c, 0, 0, nw, nh);
  const px = gp.getImageData(0, 0, nw, nh).data;

  const plano = ENTRADA * ENTRADA;
  const x = new Float32Array(3 * plano);       // zeros fora da área (letterbox)
  const lim = (v, n) => (v < 0 ? 0 : v >= n ? n - 1 : v);
  const pw = Math.min(ENTRADA, nw + 2 * mp), ph = Math.min(ENTRADA, nh + 2 * mp);
  for (let yy = 0; yy < ph; yy++) {
    const sy = lim(yy - mp, nh);
    for (let xx = 0; xx < pw; xx++) {
      const s = (sy * nw + lim(xx - mp, nw)) * 4, d = yy * ENTRADA + xx;
      x[d] = px[s + 2];               // B
      x[plano + d] = px[s + 1];       // G
      x[2 * plano + d] = px[s];       // R
    }
  }
  const out = await det.run({ input: new ort.Tensor("float32", x, [1, 3, ENTRADA, ENTRADA]) });

  const cands = [];
  for (const s of [8, 16, 32]) {
    const cols = ENTRADA / s;
    const cls = out[`cls_${s}`].data, obj = out[`obj_${s}`].data;
    const bb = out[`bbox_${s}`].data, kp = out[`kps_${s}`].data;
    for (let i = 0; i < cls.length; i++) {
      const score = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
      if (score < SCORE_PISO) continue;
      const r = Math.floor(i / cols), col = i % cols;
      const cx = (col + bb[4 * i]) * s, cy = (r + bb[4 * i + 1]) * s;
      const bw = Math.exp(bb[4 * i + 2]) * s, bh = Math.exp(bb[4 * i + 3]) * s;
      // de volta para a foto original: tira a margem e desfaz a escala
      const volta = (vx, vy) => [(vx - mp) / esc, (vy - mp) / esc];
      const [x0, y0] = volta(cx - bw / 2, cy - bh / 2);
      const pontos = [];
      for (let k = 0; k < 5; k++) pontos.push(volta((kp[10 * i + 2 * k] + col) * s, (kp[10 * i + 2 * k + 1] + r) * s));
      cands.push({ score, caixa: [x0, y0, bw / esc, bh / esc], pontos });
    }
  }
  cands.sort((a, b) => b.score - a.score);
  const finais = [];
  for (const d of cands) if (finais.every(f => iou(d.caixa, f.caixa) <= 0.3)) finais.push(d);
  return finais;
}

function iou(a, b) {
  const iw = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9);
}

// ---------- alinhamento (igual a fotos.recortar; borda replicada) ----------
// transparente=true é só para o retrato exibido na tela: fora da foto fica vazio em vez
// de repetir a borda (a borda repetida vira listras). O modelo sempre recebe a borda replicada.
function alinhar(img, rosto, zoom = 1, lado = LADO, transparente = false) {
  const [[x1, y1], [x2, y2]] = [rosto.pontos[0], rosto.pontos[1]].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const dist = Math.hypot(x2 - x1, y2 - y1);
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  const esc = lado * DIST_OLHOS / dist * zoom;
  const cos = Math.cos(ang) / esc, sen = Math.sin(ang) / esc;
  const { width: w, height: h, data: src } = img;
  const out = new Uint8ClampedArray(lado * lado * 4);
  for (let v = 0; v < lado; v++) {
    for (let u = 0; u < lado; u++) {
      const du = u - lado / 2, dv = v - lado * OLHOS_Y;
      let xs = cx + cos * du - sen * dv, ys = cy + sen * du + cos * dv;
      if (transparente && (xs < -0.5 || ys < -0.5 || xs > w - 0.5 || ys > h - 0.5)) continue;
      xs = Math.min(Math.max(xs, 0), w - 1); ys = Math.min(Math.max(ys, 0), h - 1);
      const x0 = Math.floor(xs), y0 = Math.floor(ys);
      const x1_ = Math.min(x0 + 1, w - 1), y1_ = Math.min(y0 + 1, h - 1);
      const fx = xs - x0, fy = ys - y0;
      const a = (y0 * w + x0) * 4, b = (y0 * w + x1_) * 4, c = (y1_ * w + x0) * 4, d = (y1_ * w + x1_) * 4;
      const o = (v * lado + u) * 4;
      for (let k = 0; k < 3; k++) {
        const top = src[a + k] * (1 - fx) + src[b + k] * fx;
        const bot = src[c + k] * (1 - fx) + src[d + k] * fx;
        out[o + k] = top * (1 - fy) + bot * fy + 0.5;
      }
      out[o + 3] = 255;
    }
  }
  return new ImageData(out, lado, lado);
}

// saturação média (HSV, escala 0-255 do OpenCV) no centro do rosto: < 12 = preto e branco
function saturacao(rec) {
  let soma = 0, n = 0;
  for (let v = 40; v < 120; v++) for (let u = 40; u < 120; u++) {
    const o = (v * LADO + u) * 4, r = rec.data[o], g = rec.data[o + 1], b = rec.data[o + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    soma += mx ? 255 * (mx - mn) / mx : 0; n++;
  }
  return soma / n;
}

// ---------- análise completa ----------
export async function analisar(img, progresso) {
  const { det, cls } = await carregarModelos();
  progresso?.("Procurando o rosto…");
  const todos = await detectar(img, det);
  let rostos = todos.filter(r => r.score >= SCORE_MIN);
  if (!rostos.length) rostos = todos.filter(r => r.score >= SCORE_RESERVA);
  console.info("[meio-termo] foto", img.width + "x" + img.height, "melhor score do rosto:", todos[0]?.score?.toFixed(3) ?? "nenhum");
  if (!rostos.length) return { erro: "sem_rosto" };
  rostos.sort((a, b) => b.caixa[2] * b.caixa[3] - a.caixa[2] * a.caixa[3]);
  const p = rostos[0];
  if (rostos.length > 1 && rostos[1].caixa[2] * rostos[1].caixa[3] > 0.3 * p.caixa[2] * p.caixa[3]) {
    return { erro: "varios_rostos" };
  }
  const [a, b] = p.pontos;
  if (Math.hypot(a[0] - b[0], a[1] - b[1]) < MIN_DIST) return { erro: "rosto_pequeno" };

  progresso?.("Calculando…");
  const recs = ZOOMS.map(z => alinhar(img, p, z));
  const plano = LADO * LADO, n = recs.length * 2;
  const x = new Float32Array(n * 3 * plano);
  recs.forEach((rec, i) => {
    for (const espelho of [false, true]) {
      const base = (i * 2 + (espelho ? 1 : 0)) * 3 * plano;
      for (let v = 0; v < LADO; v++) for (let u = 0; u < LADO; u++) {
        const su = espelho ? LADO - 1 - u : u;
        const o = (v * LADO + su) * 4, d = v * LADO + u;
        for (let k = 0; k < 3; k++) x[base + k * plano + d] = (rec.data[o + k] / 255 - MEDIA[k]) / DESVIO[k];
      }
    }
  });
  const out = await cls.run({ imagem: new ort.Tensor("float32", x, [n, 3, LADO, LADO]) });
  // 5 logits por variação: [0:3] branca/parda/preta e [3:5] outra/indígena, softmax separados
  const z = out.logits.data;
  const prob = [0, 0, 0];
  let indigena = 0;
  const softmax = l => { const mx = Math.max(...l), e = l.map(v => Math.exp(v - mx)), s = e.reduce((a, b) => a + b); return e.map(v => v / s); };
  for (let i = 0; i < n; i++) {
    softmax([z[5 * i], z[5 * i + 1], z[5 * i + 2]]).forEach((v, k) => { prob[k] += v / n; });
    indigena += softmax([z[5 * i + 3], z[5 * i + 4]])[1] / n;
  }
  return { prob, indigena, recorte: recs[1], retrato: alinhar(img, p, 0.9, 480, true), saturacao: saturacao(recs[1]), score: p.score };
}
