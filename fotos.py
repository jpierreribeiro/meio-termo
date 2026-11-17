"""Baixa o zip de fotos de uma UF/ano do TSE, recorta e alinha os rostos e apaga o zip.

Para cada foto cujo candidato está em dados/rotulos.csv.gz (e foi sorteado pela amostragem):
  detecta o rosto (YuNet), nivela pelos olhos, recorta 160x160 e salva em
  dados/rostos/{ano}/{uf}/{sq}.jpg. Registra tudo (inclusive rejeições) em
  dados/manifestos/{ano}_{uf}.csv — a existência desse arquivo marca a UF/ano como feita.

Uso:
  python fotos.py --ano 2024 --uf AC            # uma UF
  python fotos.py --ano 2024 --uf AC --taxa 1   # sem amostragem (piloto)
  python fotos.py --todos                       # todos os anos e UFs (para a VPS)
"""
import argparse
import csv
import math
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAIZ = Path(__file__).parent
DADOS = RAIZ / "dados"
MODELO = RAIZ / "modelos" / "face_detection_yunet_2023mar.onnx"
URL = ("https://cdn.tse.jus.br/estatistica/sead/eleicoes/eleicoes{ano}/fotos/"
       "foto_cand{ano}_{uf}_div.zip")

UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
       "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO", "BR"]
ANOS = [2018, 2020, 2022, 2024]  # 2016: o TSE publicou tudo em preto e branco

# Fração de PESSOAS mantidas por classe (sorteio pelo pessoa_id, então a mesma pessoa
# entra ou sai em todos os anos). Calibrado para ~25-30 mil pessoas nas classes grandes.
TAXAS = {"branca": 0.06, "parda": 0.08, "preta": 0.25, "indigena": 1.0, "amarela": 1.0}
TAXA_INCONSISTENTE = 0.30  # quem mudou de declaração: útil como conjunto "ambíguo"

SAIDA = 160              # lado do recorte
OLHOS_Y = 0.40           # altura dos olhos no recorte (fração)
DIST_OLHOS = 0.28        # distância entre olhos no recorte (fração do lado)
MIN_DIST_OLHOS = 12      # px na foto original; abaixo disso o rosto é pequeno demais
SCORE_MIN = 0.80

NOME_FOTO = re.compile(r"^F([A-Z]{2})(\d+)_div\.(jpe?g|png)$", re.I)
CAMPOS = ["arquivo", "ano", "uf", "sq", "pessoa_id", "genero", "raca", "racas_historico",
          "n_racas", "split", "status", "score", "dist_olhos", "saturacao"]


def sorteado(pessoa_id: str, raca: str, n_racas: int, taxa_fixa: float | None) -> bool:
    taxa = taxa_fixa if taxa_fixa is not None else (
        max(TAXAS[raca], TAXA_INCONSISTENTE) if n_racas > 1 else TAXAS[raca])
    # bits diferentes dos usados no split, para não correlacionar amostra e split
    return (int(pessoa_id[8:16], 16) % 10_000) < taxa * 10_000


def baixar(ano: int, uf: str, destino: Path) -> bool:
    """Baixa com retomada (Range) e só aceita o arquivo se o tamanho bater e o zip abrir.

    A conexão com o CDN do TSE às vezes cai no meio sem levantar erro: o arquivo
    chegava truncado e só quebrava no zipfile."""
    url = URL.format(ano=ano, uf=uf)
    parcial = destino.with_suffix(".parcial")
    total = None
    for tentativa in range(150):  # ~2h de paciência se a internet cair
        ja = parcial.stat().st_size if parcial.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={ja}-"} if ja else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 200:  # servidor ignorou o Range: recomeça do zero
                    ja = 0
                    total = int(r.headers["Content-Length"])
                elif r.status == 206:
                    total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                with open(parcial, "ab" if ja else "wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if e.code == 416 and total is not None:  # já está completo
                pass
            else:
                print(f"  falha ao baixar ({e})", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"  falha ao baixar ({e})", flush=True)

        tamanho = parcial.stat().st_size if parcial.exists() else 0
        if total is not None and tamanho == total and zipfile.is_zipfile(parcial):
            parcial.rename(destino)
            return True
        if total is not None and tamanho > total:
            parcial.unlink()
        espera = min(60, 5 * (tentativa + 1))
        print(f"  {tamanho / 1e6:.0f}/{(total or 0) / 1e6:.0f} MB; retomando em {espera}s",
              flush=True)
        time.sleep(espera)
    raise RuntimeError(f"não consegui baixar {url}")


def recortar(img: np.ndarray, det) -> tuple[np.ndarray | None, str, float, float]:
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, rostos = det.detect(img)
    if rostos is None or len(rostos) == 0:
        return None, "sem_rosto", 0.0, 0.0
    rostos = sorted(rostos, key=lambda r: r[2] * r[3], reverse=True)
    principal = rostos[0]
    if len(rostos) > 1 and rostos[1][2] * rostos[1][3] > 0.3 * principal[2] * principal[3]:
        return None, "varios_rostos", float(principal[14]), 0.0

    olhos = sorted([(principal[4], principal[5]), (principal[6], principal[7])])
    (x1, y1), (x2, y2) = olhos
    dist = math.hypot(x2 - x1, y2 - y1)
    score = float(principal[14])
    if score < SCORE_MIN:
        return None, "score_baixo", score, dist
    if dist < MIN_DIST_OLHOS:
        return None, "rosto_pequeno", score, dist

    angulo = math.degrees(math.atan2(y2 - y1, x2 - x1))
    centro = ((x1 + x2) / 2, (y1 + y2) / 2)
    escala = SAIDA * DIST_OLHOS / dist
    M = cv2.getRotationMatrix2D(centro, angulo, escala)
    M[0, 2] += SAIDA / 2 - centro[0]
    M[1, 2] += SAIDA * OLHOS_Y - centro[1]
    rosto = cv2.warpAffine(img, M, (SAIDA, SAIDA), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
    return rosto, "ok", score, dist


def processar(ano: int, uf: str, rotulos: pd.DataFrame, taxa_fixa: float | None,
              manter_zip: bool) -> None:
    manifesto = DADOS / "manifestos" / f"{ano}_{uf}.csv"
    if manifesto.exists():
        print(f"{ano} {uf}: já feito")
        return

    alvo = rotulos[(rotulos.ano == ano) & (rotulos.uf == uf)]
    alvo = alvo[[sorteado(p, r, n, taxa_fixa)
                 for p, r, n in zip(alvo.pessoa_id, alvo.raca, alvo.n_racas)]]
    if alvo.empty:
        print(f"{ano} {uf}: nenhum candidato rotulado")
        return
    por_sq = {sq: row for sq, row in zip(alvo.sq, alvo.itertuples(index=False))}

    zip_path = DADOS / "tmp" / f"foto_cand{ano}_{uf}_div.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        livre_gb = shutil.disk_usage(DADOS).free / 1024**3
        if livre_gb < 5:
            sys.exit(f"PARADO: só {livre_gb:.1f} GB livres antes de baixar {ano} {uf}")
        print(f"{ano} {uf}: baixando...", flush=True)
        if not baixar(ano, uf, zip_path):
            print(f"{ano} {uf}: zip não existe no TSE")
            return

    det = cv2.FaceDetectorYN.create(str(MODELO), "", (320, 320), SCORE_MIN, 0.3, 50)
    pasta = DADOS / "rostos" / str(ano) / uf
    pasta.mkdir(parents=True, exist_ok=True)
    linhas, contagem = [], {}
    with zipfile.ZipFile(zip_path) as z:
        for nome in z.namelist():
            m = NOME_FOTO.match(Path(nome).name)
            if not m or m.group(2) not in por_sq:
                continue
            row = por_sq[m.group(2)]
            img = cv2.imdecode(np.frombuffer(z.read(nome), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                rosto, status, score, dist = None, "ilegivel", 0.0, 0.0
            else:
                rosto, status, score, dist = recortar(img, det)
            arquivo, saturacao = "", 0.0
            if rosto is not None:
                # centro do rosto; < 12 = foto em preto e branco (2016 inteiro é assim)
                hsv = cv2.cvtColor(rosto[40:120, 40:120], cv2.COLOR_BGR2HSV)
                saturacao = float(hsv[..., 1].mean())
                arquivo = f"rostos/{ano}/{uf}/{row.sq}.jpg"
                cv2.imwrite(str(DADOS / arquivo), rosto, [cv2.IMWRITE_JPEG_QUALITY, 92])
            contagem[status] = contagem.get(status, 0) + 1
            linhas.append({"arquivo": arquivo, "ano": ano, "uf": uf, "sq": row.sq,
                           "pessoa_id": row.pessoa_id, "genero": row.genero,
                           "raca": row.raca, "racas_historico": row.racas_historico,
                           "n_racas": row.n_racas, "split": row.split, "status": status,
                           "score": round(score, 3), "dist_olhos": round(dist, 1),
                           "saturacao": round(saturacao, 1)})

    manifesto.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifesto.with_suffix(".parcial")
    with open(tmp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CAMPOS)
        wr.writeheader()
        wr.writerows(linhas)
    tmp.rename(manifesto)  # só marca como feito depois de escrever tudo
    if not manter_zip:
        zip_path.unlink()
    sem_foto = len(por_sq) - len(linhas)
    print(f"{ano} {uf}: {len(por_sq)} sorteados, {sem_foto} sem foto no zip, {contagem}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano", type=int)
    ap.add_argument("--uf")
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--taxa", type=float, help="sobrescreve a amostragem (1 = todos)")
    ap.add_argument("--manter-zip", action="store_true")
    args = ap.parse_args()

    rotulos = pd.read_csv(DADOS / "rotulos.csv.gz", dtype={"sq": str, "pessoa_id": str})
    if args.todos:
        pares = [(a, u) for a in ANOS for u in UFS]
    elif args.ano and args.uf:
        pares = [(args.ano, args.uf.upper())]
    else:
        sys.exit("use --ano e --uf, ou --todos")
    for ano, uf in pares:
        processar(ano, uf, rotulos, args.taxa, args.manter_zip)


if __name__ == "__main__":
    main()
