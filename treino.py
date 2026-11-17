"""Treino e avaliação do classificador de cor/raça autodeclarada (5 classes do IBGE).

Dois modos:
  --modo sonda     extrai embeddings de um backbone congelado (CLIP ViT-B/16) e treina
                   regressão logística. Linha de base barata; roda até em CPU com subamostra.
  --modo ajuste    fine-tuning do backbone inteiro (precisa de GPU: Kaggle).

Treina SÓ com pessoas de declaração consistente (n_racas == 1). Avalia em:
  - test consistente: matriz de confusão, recall por classe, acurácia balanceada,
    e a métrica binária "parda vs. resto"
  - conjunto ambíguo (pessoas que mudaram de declaração): para onde o modelo as joga,
    comparado com o histórico de declarações delas

Uso local (teste rápido):  python treino.py --modo sonda --por-classe 300
GPU (Colab):               python treino.py --modo ajuste --dados /content/dados --epocas 8
Indígena vs. resto:        python treino.py --modo ajuste --binario indigena ...
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

CLASSES = ["branca", "parda", "preta", "indigena", "amarela"]  # redefinida por --classes
BACKBONE = "vit_base_patch16_clip_224.openai"                  # redefinido por --backbone
RECORTE = 1.0  # fração central do recorte 160x160 usada (--recorte-central); 1 = inteiro
ORIGEM = CLASSES  # raças do TSE que entram (--classes); difere de CLASSES só no modo binário
BINARIO = None    # --binario indigena: CLASSES vira ["outra", "indigena"]
ALVO = "parda"    # classe da métrica "X vs. resto"
WORKERS = min(8, os.cpu_count() or 2)


def rotular(df: pd.DataFrame) -> pd.DataFrame:
    """No modo binário, troca o rótulo por alvo/"outra" (inclusive no histórico)."""
    if not BINARIO:
        return df
    mapa = lambda r: r if r == BINARIO else "outra"
    df = df.assign(raca=df.raca.map(mapa))
    if "racas_historico" in df:
        df = df.assign(racas_historico=df.racas_historico.map(
            lambda h: "|".join(sorted({mapa(r) for r in h.split("|")}))))
    return df


class Rostos(Dataset):
    def __init__(self, df, raiz, transform):
        self.arquivos = [raiz / a for a in df.arquivo]
        self.y = df.raca.map(CLASSES.index).to_numpy()
        self.t = transform

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, i):
        img = Image.open(self.arquivos[i]).convert("RGB")
        if RECORTE < 1:
            # olhos ficam a 40% da altura: centraliza o corte um pouco abaixo deles
            w, h = img.size
            lado = RECORTE * w
            cx, cy = w / 2, h * 0.5
            img = img.crop((round(cx - lado / 2), round(cy - lado / 2),
                            round(cx + lado / 2), round(cy + lado / 2)))
        return self.t(img), self.y[i]


def carregar(raiz: Path, por_classe: int | None, semente: int):
    m = pd.read_csv(raiz / "manifesto.csv", dtype={"sq": str, "pessoa_id": str})
    m = m[(m.status == "ok") & m.raca.isin(ORIGEM)]
    if "saturacao" in m:  # 2016 inteiro e parte de 2018/2020 vieram em preto e branco
        m = m[m.saturacao >= 12]
    # ambíguo: só quem transitou entre classes que estão no modelo
    m = m[m.racas_historico.map(lambda h: all(r in ORIGEM for r in h.split("|")))]
    m = rotular(m)
    n_hist = m.racas_historico.str.count(r"\|") + 1
    cons = m[n_hist == 1]   # no binário, quem foi parda|branca é "outra" consistente
    ambiguo = m[n_hist > 1]
    partes = {s: cons[cons.split == s] for s in ["train", "val", "test"]}
    if por_classe:
        # limita por PESSOA (todas as fotos da pessoa vêm juntas)
        def limitar(df, n):
            pessoas = (df.drop_duplicates("pessoa_id").groupby("raca", group_keys=False)
                       .apply(lambda g: g.sample(min(len(g), n), random_state=semente)))
            return df[df.pessoa_id.isin(pessoas.pessoa_id)]
        partes["train"] = limitar(partes["train"], por_classe)
        for s in ["val", "test"]:
            partes[s] = limitar(partes[s], max(50, por_classe // 8))
        ambiguo = ambiguo.sample(min(len(ambiguo), por_classe * 2), random_state=semente)
    return partes, ambiguo


def relatorio(y, p_prob, titulo):
    pred = p_prob.argmax(1)
    presentes = sorted(set(y) | set(pred))
    print(f"\n=== {titulo} ({len(y)} fotos) ===")
    print(f"acurácia balanceada: {balanced_accuracy_score(y, pred):.3f}")
    print(classification_report(y, pred, labels=presentes,
                                target_names=[CLASSES[i] for i in presentes], digits=3,
                                zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=range(len(CLASSES))),
                      index=[f"real {c}" for c in CLASSES], columns=CLASSES)
    print(cm.to_string())
    ip = CLASSES.index(ALVO)
    yb, pb = (y == ip), (pred == ip)
    tp = (yb & pb).sum()
    print(f"\n{ALVO} vs. resto: precisão {tp / max(pb.sum(), 1):.3f}  "
          f"recall {tp / max(yb.sum(), 1):.3f}  acurácia {(yb == pb).mean():.3f}")
    return {"acuracia_balanceada": float(balanced_accuracy_score(y, pred)),
            "matriz": cm.values.tolist()}


def relatorio_ambiguo(amb: pd.DataFrame, prob: np.ndarray):
    amb = amb.assign(previsto=[CLASSES[i] for i in prob.argmax(1)])
    print(f"\n=== conjunto ambíguo ({len(amb)} fotos de quem mudou de declaração) ===")
    print("linhas = histórico de declarações; colunas = classe prevista (% da linha)")
    top = amb.racas_historico.value_counts().head(6).index
    t = pd.crosstab(amb[amb.racas_historico.isin(top)].racas_historico,
                    amb[amb.racas_historico.isin(top)].previsto, normalize="index")
    print((t * 100).round(1).to_string())
    acerto_hist = np.mean([p in h.split("|") for p, h in zip(amb.previsto, amb.racas_historico)])
    print(f"previsão bate com ALGUMA das declarações da pessoa: {acerto_hist:.1%}")


def relatorio_externo(ext: pd.DataFrame, prob: np.ndarray):
    """Fotos fora do padrão TSE. Cada fonte tem uma classe só, então o que dá para medir
    é: dentro de cada fonte, quanto o modelo acerta e para onde vão os erros."""
    ext = ext.assign(previsto=[CLASSES[i] for i in prob.argmax(1)],
                     p_alvo=prob[:, CLASSES.index(ALVO)])
    print(f"\n=== TESTE EXTERNO ({len(ext)} fotos fora do TSE) ===")
    print("linhas = fonte (rótulo); colunas = classe prevista (% da linha)")
    t = pd.crosstab(ext.fonte + " [" + ext.raca + "]", ext.previsto, normalize="index")
    print((t.reindex(columns=CLASSES, fill_value=0) * 100).round(1).to_string())
    pos = ext.raca == ALVO
    pred = ext.previsto == ALVO
    tp = (pos & pred).sum()
    print(f"\n{ALVO} vs. resto no externo: recall {tp / max(pos.sum(), 1):.3f}  "
          f"precisão {tp / max(pred.sum(), 1):.3f}  "
          f"(precisão aqui depende da mistura de fontes: leia com cuidado)")
    return {f: float((g.previsto == g.raca).mean()) for f, g in ext.groupby("fonte")}


def transformacoes(modelo, treino: bool):
    cfg = timm.data.resolve_data_config({}, model=modelo)
    return timm.data.create_transform(**cfg, is_training=treino,
                                      hflip=0.5 if treino else 0.0,
                                      scale=(0.6, 1.0), ratio=(0.9, 1.1),  # rosto já alinhado
                                      color_jitter=None)  # cor é o sinal: não distorcer


@torch.no_grad()
def embeddings(modelo, df, raiz, dev, lote):
    """Embeddings normalizados, com cache em disco por (backbone, recorte).

    Só calcula as imagens que ainda não estão no cache: trocar classes, amostra ou
    classificador não exige recalcular nada."""
    cache_path = raiz / "cache" / f"emb_{BACKBONE.replace('/', '_')}_rc{RECORTE}.npz"
    cache = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            cache = dict(zip(z["arquivos"], z["vetores"]))
    faltam = df[~df.arquivo.isin(cache)].drop_duplicates("arquivo")
    if len(faltam):
        print(f"  calculando {len(faltam)} embeddings ({len(df) - len(faltam)} no cache)",
              flush=True)
        dl = DataLoader(Rostos(faltam.assign(raca=CLASSES[0]), raiz,
                               transformacoes(modelo, False)), batch_size=lote, num_workers=WORKERS)
        novos = np.concatenate([modelo(x.to(dev)).float().cpu().numpy() for x, _ in dl])
        novos /= np.linalg.norm(novos, axis=1, keepdims=True)
        cache.update(zip(faltam.arquivo, novos))
        cache_path.parent.mkdir(exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, arquivos=np.array(list(cache)), vetores=np.stack(list(cache.values())))
        tmp.rename(cache_path)
    return np.stack([cache[a] for a in df.arquivo])


def modo_sonda(partes, ambiguo, externo, raiz, dev, lote):
    modelo = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(dev).eval()
    t0 = time.time()
    emb = {s: embeddings(modelo, df, raiz, dev, lote) for s, df in partes.items()}
    emb_amb = embeddings(modelo, ambiguo, raiz, dev, lote)
    emb_ext = embeddings(modelo, externo, raiz, dev, lote) if externo is not None else None
    print(f"embeddings em {time.time() - t0:.0f}s")
    y = {s: df.raca.map(CLASSES.index).to_numpy() for s, df in partes.items()}
    melhor = None
    for C in [0.3, 1, 3, 10]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(emb["train"], y["train"])
        ba = balanced_accuracy_score(y["val"], clf.predict(emb["val"]))
        print(f"C={C}: val acurácia balanceada {ba:.3f}")
        if melhor is None or ba > melhor[0]:
            melhor = (ba, clf)
    clf = melhor[1]

    def prob(e):  # garante as 5 colunas mesmo se alguma classe faltou no treino
        p = np.zeros((len(e), len(CLASSES)))
        p[:, clf.classes_] = clf.predict_proba(e)
        return p
    r = relatorio(y["test"], prob(emb["test"]), "test (consistentes)")
    relatorio_ambiguo(ambiguo, prob(emb_amb))
    if emb_ext is not None:
        r["externo"] = relatorio_externo(externo, prob(emb_ext))
    return r


def modo_ajuste(partes, ambiguo, externo, raiz, dev, lote, epocas, saida):
    modelo = timm.create_model(BACKBONE, pretrained=True, num_classes=len(CLASSES)).to(dev)
    tr = DataLoader(Rostos(partes["train"], raiz, transformacoes(modelo, True)),
                    batch_size=lote, shuffle=True, num_workers=WORKERS, drop_last=True)
    contagem = np.bincount(tr.dataset.y, minlength=len(CLASSES)).astype(float)
    peso = torch.tensor((contagem.sum() / np.maximum(contagem, 1)) ** 0.5, dtype=torch.float)
    perda = nn.CrossEntropyLoss(weight=(peso / peso.mean()).to(dev), label_smoothing=0.1)
    # backbone com lr baixa (já sabe rostos), cabeça com lr alta
    cabeca = list(modelo.get_classifier().parameters())
    ids = {id(p) for p in cabeca}
    corpo = [p for p in modelo.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([{"params": corpo, "lr": 1e-5}, {"params": cabeca, "lr": 1e-3}],
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-5, 1e-3],
                                                total_steps=epocas * len(tr), pct_start=0.1)
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    @torch.no_grad()
    def prever(df):
        modelo.eval()
        dl = DataLoader(Rostos(df, raiz, transformacoes(modelo, False)), batch_size=lote * 2,
                        num_workers=WORKERS)
        ps = []
        for x, _ in dl:
            with torch.autocast(dev, enabled=amp):
                ps.append(modelo(x.to(dev)).float().softmax(1).cpu().numpy())
        return np.concatenate(ps)

    melhor = -1
    for ep in range(epocas):
        modelo.train()
        t0, soma = time.time(), 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=amp):
                l = perda(modelo(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            soma += l.item()
        yv = partes["val"].raca.map(CLASSES.index).to_numpy()
        ba = balanced_accuracy_score(yv, prever(partes["val"]).argmax(1))
        print(f"época {ep + 1}/{epocas}: perda {soma / len(tr):.3f}  "
              f"val acurácia balanceada {ba:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if ba > melhor:
            melhor = ba
            torch.save(modelo.state_dict(), saida / "modelo.pt")

    modelo.load_state_dict(torch.load(saida / "modelo.pt", map_location=dev))
    yt = partes["test"].raca.map(CLASSES.index).to_numpy()
    r = relatorio(yt, prever(partes["test"]), "test (consistentes)")
    relatorio_ambiguo(ambiguo, prever(ambiguo))
    if externo is not None:
        r["externo"] = relatorio_externo(externo, prever(externo))
    return r


def main():
    global CLASSES, BACKBONE, RECORTE, ORIGEM, BINARIO, ALVO
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["sonda", "ajuste"], default="sonda")
    ap.add_argument("--dados", type=Path, default=Path(__file__).parent / "dados",
                    help="pasta com manifesto.csv e rostos/")
    ap.add_argument("--saida", type=Path, default=Path(__file__).parent / "resultados")
    ap.add_argument("--por-classe", type=int, help="limite de pessoas por classe no treino")
    ap.add_argument("--epocas", type=int, default=8)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--classes", nargs="+", default=["branca", "parda", "preta", "indigena"],
                    help="amarela fica de fora por padrão (rótulo do TSE pouco confiável)")
    ap.add_argument("--externo", type=Path, help="CSV do teste externo (externo.py)")
    ap.add_argument("--recorte-central", type=float, default=1.0,
                    help="usa só a fração central do recorte (ex. 0.62 = sem bordas)")
    ap.add_argument("--backbone", default=BACKBONE,
                    help="na CPU, vit_base_patch32_clip_224.openai é ~4x mais rápido")
    ap.add_argument("--binario", help="ex.: indigena -> modelo indígena vs. outra")
    ap.add_argument("--tag", default="", help="sufixo do arquivo de métricas")
    args = ap.parse_args()

    BACKBONE, RECORTE, ORIGEM, BINARIO = (args.backbone, args.recorte_central,
                                          args.classes, args.binario)
    CLASSES = ["outra", BINARIO] if BINARIO else list(ORIGEM)
    ALVO = BINARIO or ("parda" if "parda" in CLASSES else CLASSES[-1])
    torch.manual_seed(args.semente)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.saida.mkdir(parents=True, exist_ok=True)
    partes, ambiguo = carregar(args.dados, args.por_classe, args.semente)
    externo = None
    if args.externo:
        externo = pd.read_csv(args.externo)
        externo = rotular(externo[externo.raca.isin(ORIGEM)])
        if "saturacao" in externo:
            externo = externo[externo.saturacao >= 12]
    print(f"dispositivo: {dev} | backbone: {BACKBONE} | recorte central: {RECORTE} | "
          f"classes: {CLASSES}")
    for s, df in partes.items():
        print(f"{s}: {len(df)} fotos, {df.pessoa_id.nunique()} pessoas, "
              f"{df.raca.value_counts().to_dict()}")
    print(f"ambíguo: {len(ambiguo)} fotos")

    if externo is not None:
        print(f"externo: {len(externo)} fotos, {externo.fonte.value_counts().to_dict()}")

    if args.modo == "sonda":
        r = modo_sonda(partes, ambiguo, externo, args.dados, dev, args.lote)
    else:
        r = modo_ajuste(partes, ambiguo, externo, args.dados, dev, args.lote, args.epocas,
                        args.saida)
    (args.saida / f"metricas_{args.modo}{args.tag}.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
