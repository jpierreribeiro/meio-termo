"""Referência em Python do pipeline que o site roda em JavaScript (app.js).

Mesmo algoritmo, passo a passo, sem OpenCV: serve para validar que o recorte do navegador
bate com o do treino (fotos.recortar, que usa cv2.FaceDetectorYN + warpAffine).

  python referencia_web.py foto.jpg    -> imprime as probabilidades
  python referencia_web.py --validar   -> compara com o pipeline OpenCV em fotos externas
"""
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

RAIZ = Path(__file__).parent / "web"
ENTRADA = 640
LADO, OLHOS_Y, DIST_OLHOS, MIN_DIST, SCORE_MIN = 160, 0.40, 0.28, 12, 0.8
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], np.float32)
CLASSES = ["branca", "parda", "preta"]


def detectar(rgb: np.ndarray, sess) -> list[dict]:
    """Letterbox 640x640 (canto superior esquerdo), BGR 0-255, decodifica e faz NMS."""
    h, w = rgb.shape[:2]
    m = max(h, w) // 4  # margem de 25% com borda replicada (igual ao app.js)
    esc = min(ENTRADA / (w + 2 * m), ENTRADA / (h + 2 * m))
    nw, nh = max(1, round(w * esc)), max(1, round(h * esc))
    mp = round(m * esc)
    peq = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR), np.float32)
    peq = np.pad(peq, ((mp, mp), (mp, mp), (0, 0)), mode="edge")[:ENTRADA, :ENTRADA]
    x = np.zeros((ENTRADA, ENTRADA, 3), np.float32)
    x[:peq.shape[0], :peq.shape[1]] = peq[..., ::-1]  # RGB -> BGR
    saidas = dict(zip([o.name for o in sess.get_outputs()],
                      sess.run(None, {"input": x.transpose(2, 0, 1)[None]})))
    cands = []
    for s in (8, 16, 32):
        cols = ENTRADA // s
        cls, obj = saidas[f"cls_{s}"][0, :, 0], saidas[f"obj_{s}"][0, :, 0]
        bb, kp = saidas[f"bbox_{s}"][0], saidas[f"kps_{s}"][0]
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        for i in np.nonzero(score >= SCORE_MIN)[0]:
            r, c = divmod(i, cols)
            cx, cy = (c + bb[i, 0]) * s, (r + bb[i, 1]) * s
            bw, bh = math.exp(bb[i, 2]) * s, math.exp(bb[i, 3]) * s
            pts = [(((kp[i, 2 * k] + c) * s - mp) / esc, ((kp[i, 2 * k + 1] + r) * s - mp) / esc)
                   for k in range(5)]
            cands.append({"score": float(score[i]),
                          "caixa": ((cx - bw / 2 - mp) / esc, (cy - bh / 2 - mp) / esc,
                                    bw / esc, bh / esc),
                          "pontos": pts})
    cands.sort(key=lambda d: -d["score"])
    finais = []
    for d in cands:  # NMS, IoU 0.3
        if all(iou(d["caixa"], f["caixa"]) <= 0.3 for f in finais):
            finais.append(d)
    return finais


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def alinhar(rgb: np.ndarray, rosto: dict, zoom: float = 1.0) -> np.ndarray:
    """Mesmo M de fotos.recortar; amostragem bilinear com borda replicada (clamp)."""
    (x1, y1), (x2, y2) = sorted(rosto["pontos"][:2])
    dist = math.hypot(x2 - x1, y2 - y1)
    ang = math.atan2(y2 - y1, x2 - x1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    esc = LADO * DIST_OLHOS / dist * zoom
    # inversa: ponto de saída (u,v) -> ponto na foto (x,y)
    cos, sen = math.cos(ang) / esc, math.sin(ang) / esc
    u, v = np.meshgrid(np.arange(LADO, dtype=np.float32), np.arange(LADO, dtype=np.float32))
    du, dv = u - LADO / 2, v - LADO * OLHOS_Y
    xs = cx + cos * du - sen * dv
    ys = cy + sen * du + cos * dv
    h, w = rgb.shape[:2]
    xs, ys = np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1_, y1_ = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (xs - x0)[..., None], (ys - y0)[..., None]
    f = rgb.astype(np.float32)
    top = f[y0, x0] * (1 - fx) + f[y0, x1_] * fx
    bot = f[y1_, x0] * (1 - fx) + f[y1_, x1_] * fx
    return np.clip(top * (1 - fy) + bot * fy + 0.5, 0, 255).astype(np.uint8)


def analisar(rgb, det, cls):
    rostos = detectar(rgb, det)
    if not rostos:
        return None, "sem_rosto"
    rostos.sort(key=lambda d: -d["caixa"][2] * d["caixa"][3])
    p = rostos[0]
    if len(rostos) > 1 and rostos[1]["caixa"][2] * rostos[1]["caixa"][3] > 0.3 * p["caixa"][2] * p["caixa"][3]:
        return None, "varios_rostos"
    (a, b) = p["pontos"][:2]
    if math.hypot(a[0] - b[0], a[1] - b[1]) < MIN_DIST:
        return None, "rosto_pequeno"
    return classificar(rgb, p, cls), alinhar(rgb, p)


ZOOMS = (0.95, 1.0, 1.05)


def classificar(rgb, rosto, cls, tta=True):
    """Média de 6 variações (3 escalas x espelhado): o modelo oscila até ~10 pontos com
    deslocamentos de poucos pixels ou espelhamento; a média estabiliza a resposta."""
    recs = [alinhar(rgb, rosto, z) for z in (ZOOMS if tta else (1.0,))]
    if tta:
        recs += [r[:, ::-1] for r in recs]
    x = np.stack([((r.astype(np.float32) / 255 - MEDIA) / DESVIO).transpose(2, 0, 1)
                  for r in recs])
    z = cls.run(None, {"imagem": x})[0]
    # [0:3] branca/parda/preta, [3:5] outra/indígena: softmax separados; devolve 4 valores
    return np.r_[softmax(z[:, :3]).mean(0), softmax(z[:, 3:])[:, 1].mean()]


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sessoes():
    return (ort.InferenceSession(str(RAIZ / "modelo" / "face_detection_yunet_2023mar.onnx")),
            ort.InferenceSession(str(RAIZ / "modelo" / "aluno2_w16.onnx")))


def validar(n=300):
    import random
    import cv2
    sys.path.insert(0, str(RAIZ.parent))
    import fotos
    det_cv = cv2.FaceDetectorYN.create(str(fotos.MODELO), "", (320, 320), SCORE_MIN, 0.3, 50)
    det, cls = sessoes()
    base = Path.home() / "Documents" / "pardos-organizacao"
    arqs = sorted((base / "autoral" / "pardos").iterdir()) + sorted((base / "dataset_utk" / "Branco").iterdir())
    random.seed(0)
    difs, ok = [], 0
    for p in random.sample(arqs, n):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        m = max(bgr.shape[:2]) // 4  # mesma margem do externo.py
        bgr = cv2.copyMakeBorder(bgr, m, m, m, m, cv2.BORDER_REPLICATE)
        ref, st, _, _ = fotos.recortar(bgr, det_cv)
        if ref is None:
            continue
        prob, rec = analisar(bgr[..., ::-1].copy(), det, cls)
        if prob is None:
            continue
        x = ((ref[..., ::-1].astype(np.float32) / 255 - MEDIA) / DESVIO).transpose(2, 0, 1)[None]
        z = cls.run(None, {"imagem": x})[0]
        difs.append(np.abs(prob - np.r_[softmax(z[:, :3])[0], softmax(z[:, 3:])[0, 1]]).max())
        ok += 1
    d = np.array(difs) * 100
    print(f"{ok} fotos comparadas: diferença na maior probabilidade, em pontos: "
          f"mediana {np.median(d):.1f}, p90 {np.percentile(d, 90):.1f}, máx {d.max():.1f}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--validar"]:
        validar()
    else:
        det, cls = sessoes()
        prob, _ = analisar(np.asarray(Image.open(sys.argv[1]).convert("RGB")), det, cls)
        print(dict(zip(CLASSES + ["indigena"], np.round(prob * 100, 1))) if prob is not None else _)
