"""Monta o conjunto de TESTE EXTERNO (fotos fora do padrão TSE), com o mesmo recorte do TSE.

Fontes:
  autoral_parda     ~/Documents/pardos-organizacao/autoral/pardos     (rótulo do JP)
  autoral_indigena  ~/Documents/pardos-organizacao/autoral/indigenas  (rótulo do JP)
  utk_branca        UTKFace "Branco" (amostra)
  utk_preta         UTKFace "Negro"  (amostra)

As fotos do TSE que estavam misturadas na base autoral (F{UF}{SQ}_div.*) são excluídas:
não são externas. Duplicatas exatas (md5) e quase-duplicatas (hash perceptual do recorte,
ex.: a mesma pessoa em dois prints) são removidas.

Nunca entra no treino. Saída: dados/externo/{fonte}/*.jpg + dados/externo.csv
"""
import argparse
import csv
import hashlib
import random
import re
from pathlib import Path

import cv2
import numpy as np

import fotos

BASE = Path.home() / "Documents" / "pardos-organizacao"
FONTES = {
    "autoral_parda": (BASE / "autoral" / "pardos", "parda"),
    "autoral_indigena": (BASE / "autoral" / "indigenas", "indigena"),
    "utk_branca": (BASE / "dataset_utk" / "Branco", "branca"),
    "utk_preta": (BASE / "dataset_utk" / "Negro", "preta"),
}
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TSE = re.compile(r"^F[A-Z]{2}\d+_div\.", re.I)


def ahash(img: np.ndarray) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA)
    bits = (g > g.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utk-por-classe", type=int, default=1500)
    args = ap.parse_args()

    det = cv2.FaceDetectorYN.create(str(fotos.MODELO), "", (320, 320), fotos.SCORE_MIN, 0.3, 50)
    saida = fotos.DADOS / "externo"
    linhas, vistos_md5, vistos_hash = [], set(), []
    random.seed(0)

    for fonte, (pasta, raca) in FONTES.items():
        arquivos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXT)
        if fonte.startswith("utk"):
            arquivos = random.sample(arquivos, min(args.utk_por_classe, len(arquivos)))
        cont = {}
        (saida / fonte).mkdir(parents=True, exist_ok=True)
        for p in arquivos:
            if TSE.match(p.name):
                cont["foto_do_tse"] = cont.get("foto_do_tse", 0) + 1
                continue
            dados = p.read_bytes()
            md5 = hashlib.md5(dados).hexdigest()
            if md5 in vistos_md5:
                cont["duplicata"] = cont.get("duplicata", 0) + 1
                continue
            vistos_md5.add(md5)
            img = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                cont["ilegivel"] = cont.get("ilegivel", 0) + 1
                continue
            # margem: o UTK já vem recortado rente ao rosto e o detector precisa de contexto
            m = max(img.shape[:2]) // 4
            img = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_REPLICATE)
            rosto, status, score, dist = fotos.recortar(img, det)
            if rosto is not None:
                h = ahash(rosto)
                if any(bin(h ^ v).count("1") <= 12 for v in vistos_hash):
                    status = "quase_duplicata"
                else:
                    vistos_hash.append(h)
            cont[status] = cont.get(status, 0) + 1
            if status != "ok":
                continue
            nome = f"{fonte}/{md5[:16]}.jpg"
            cv2.imwrite(str(saida / nome), rosto, [cv2.IMWRITE_JPEG_QUALITY, 92])
            linhas.append({"arquivo": f"externo/{nome}", "fonte": fonte, "raca": raca,
                           "origem": p.name, "score": round(score, 3),
                           "dist_olhos": round(dist, 1)})
        print(f"{fonte}: {len(arquivos)} arquivos -> {cont}", flush=True)

    with open(fotos.DADOS / "externo.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(linhas[0]))
        wr.writeheader()
        wr.writerows(linhas)
    print(f"{len(linhas)} recortes no teste externo -> dados/externo.csv")


if __name__ == "__main__":
    main()
