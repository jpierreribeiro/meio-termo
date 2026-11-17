"""Destila DOIS professores num único aluno pequeno com duas saídas:

  saída A (3 logits): branca / parda / preta    <- professor ajuste_3classes (calibrado)
  saída B (2 logits): outra / indígena          <- professor ajuste_indigena (calibrado aqui)

O aluno vê todas as fotos coloridas (inclusive indígenas e quem mudou de declaração) e
aprende as probabilidades dos dois professores. Mesmo tamanho do aluno anterior.

Uso (Colab):
  python destilar2.py --dados /content/dados --saida "$SAIDA/aluno2" \
      --prof3 "$SAIDA/ajuste_3classes/modelo.pt" --cal3 "$SAIDA/calibracao/calibracao.json" \
      --profi "$SAIDA/ajuste_indigena/modelo.pt" --externo /content/dados/externo_adorno.csv \
      --cache3 "$SAIDA/aluno/logits_professor.npz"
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import destilar
import treino
from destilar import Fotos, LADO, MEDIA, DESVIO, transf_aluno, prever, ece

TRES = ["branca", "parda", "preta"]
QUATRO = TRES + ["indigena"]


def logits_de(caminho_pt, n_saidas, df, args, dev, cache: Path, cache_extra: Path | None = None):
    """Logits de um professor para todas as fotos de df, reaproveitando caches por arquivo."""
    mapa = {}
    for c in [cache_extra, cache]:
        if c and c.exists():
            z = np.load(c, allow_pickle=True)
            mapa.update(zip(z["arquivos"], z["logits"]))
    faltam = df[~df.arquivo.isin(mapa)].drop_duplicates("arquivo")
    if len(faltam):
        prof = timm.create_model(args.backbone, pretrained=False, num_classes=n_saidas)
        prof.load_state_dict(torch.load(caminho_pt, map_location="cpu"))
        prof = prof.to(dev)
        print(f"  {caminho_pt.parent.name}: {len(faltam)} fotos ({len(df) - len(faltam)} em cache)",
              flush=True)
        t0 = time.time()
        L = prever(prof, treino.Rostos(faltam.assign(raca="branca"), args.dados,
                                       treino.transformacoes(prof, False)), dev, args.lote * 2)
        print(f"  {time.time() - t0:.0f}s", flush=True)
        mapa.update(zip(faltam.arquivo, L.numpy()))
        np.savez(cache, arquivos=np.array(list(mapa)), logits=np.stack(list(mapa.values())))
        del prof
        if dev == "cuda":
            torch.cuda.empty_cache()
    return torch.as_tensor(np.stack([mapa[a] for a in df.arquivo]))


def calibrar(logits, y):
    log_t = torch.zeros(1, requires_grad=True)
    b = torch.zeros(logits.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([log_t, b], lr=0.1, max_iter=500)
    yt = torch.as_tensor(y, dtype=torch.long)

    def passo():
        opt.zero_grad()
        l = F.cross_entropy(logits / log_t.exp() + b, yt)
        l.backward()
        return l
    opt.step(passo)
    return log_t.exp().item(), b.detach()


def ece_bin(p, y, bins=15):
    """ECE da probabilidade de indígena (não da classe prevista): o que o site vai mostrar."""
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p > lo) & (p <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--prof3", type=Path, required=True)
    ap.add_argument("--cal3", type=Path)
    ap.add_argument("--profi", type=Path, required=True)
    ap.add_argument("--cache3", type=Path, help="logits do professor 3 classes já calculados")
    ap.add_argument("--backbone", default="vit_base_patch16_clip_224.openai")
    ap.add_argument("--aluno", default="mobilenetv3_large_100")
    ap.add_argument("--externo", type=Path)
    ap.add_argument("--epocas", type=int, default=12)
    ap.add_argument("--lote", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--peso-indigena", type=float, default=4.0,
                    help="sobreamostragem de fotos de indígenas no treino do aluno")
    ap.add_argument("--limite", type=int, help="só para teste rápido")
    args = ap.parse_args()

    treino.BACKBONE = args.backbone
    args.saida.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    m = pd.read_csv(args.dados / "manifesto.csv", dtype={"sq": str, "pessoa_id": str})
    m = m[(m.status == "ok") & (m.saturacao >= 12) & m.raca.isin(QUATRO)]
    m = m[m.racas_historico.map(lambda h: all(r in QUATRO for r in h.split("|")))]
    hist = m.racas_historico.str.split("|")
    m = m.assign(
        tri_cons=(m.n_racas == 1) & m.raca.isin(TRES),             # avalia a saída A
        ind_y=hist.map(lambda h: 1 if h == ["indigena"] else (0 if "indigena" not in h else -1)),
    ).reset_index(drop=True)                                       # ind_y -1 = ambíguo
    if args.limite:
        m = m.sample(frac=1, random_state=0)
        m = m[m.groupby(["split", "ind_y"]).cumcount() < args.limite].reset_index(drop=True)
    print(f"{len(m)} fotos | indígenas consistentes {(m.ind_y == 1).sum()} | "
          f"ambíguos indígena/outra {(m.ind_y == -1).sum()}", flush=True)

    # 1) professores
    print("professores:", flush=True)
    L3 = logits_de(args.prof3, 3, m, args, dev, args.saida / "logits_prof3.npz", args.cache3)
    Li = logits_de(args.profi, 2, m, args, dev, args.saida / "logits_profi.npz")
    T3, b3 = 1.0, torch.zeros(3)
    if args.cal3 and args.cal3.exists():
        c = json.loads(args.cal3.read_text())
        T3, b3 = c["T"], torch.tensor(c["b"])
    P3 = torch.softmax(L3 / T3 + b3, 1).numpy()

    val = (m.split == "val").to_numpy() & (m.ind_y >= 0).to_numpy()
    Ti, bi = calibrar(Li[val], m.ind_y[val].to_numpy())
    Pi = torch.softmax(Li / Ti + bi, 1).numpy()
    te_i = (m.split == "test").to_numpy() & (m.ind_y >= 0).to_numpy()
    yi = m.ind_y[te_i].to_numpy()
    p_cru = torch.softmax(Li[te_i], 1).numpy()[:, 1]
    print(f"\nprofessor indígena: T={Ti:.3f} b={bi.numpy().round(3)} | ECE de P(indígena) no teste: "
          f"antes {ece_bin(p_cru, yi):.1%}, calibrado {ece_bin(Pi[te_i, 1], yi):.1%} | "
          f"AUC {roc_auc_score(yi, Pi[te_i, 1]):.3f}", flush=True)

    # 2) aluno com duas saídas
    alvo = np.concatenate([P3, Pi], 1)              # 5 colunas
    tr = (m.split == "train").to_numpy()
    aluno = timm.create_model(args.aluno, pretrained=True, num_classes=5)
    for w in aluno.get_classifier().parameters():
        torch.nn.init.zeros_(w)
    aluno = aluno.to(dev)
    ds_tr = Fotos(m.arquivo[tr], args.dados, alvo[tr], transf_aluno(True))
    pesos = np.where(m.ind_y[tr] != 0, args.peso_indigena, 1.0)
    # sobreamostrar não distorce o alvo: cada foto tem a própria probabilidade a imitar
    dl = DataLoader(ds_tr, batch_size=args.lote, num_workers=treino.WORKERS, drop_last=True,
                    sampler=WeightedRandomSampler(pesos, num_samples=len(pesos), replacement=True))
    va = (m.split == "val").to_numpy()
    ds_va = Fotos(m.arquivo[va], args.dados, alvo[va], transf_aluno(False))
    print(f"\naluno {args.aluno}: treino {tr.sum()} fotos, indígenas sobreamostrados x"
          f"{args.peso_indigena:g}", flush=True)
    opt = torch.optim.AdamW(aluno.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epocas * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    def perda_de(z, a):
        la = -(a[:, :3] * F.log_softmax(z[:, :3], 1)).sum(1).mean()
        lb = -(a[:, 3:] * F.log_softmax(z[:, 3:], 1)).sum(1).mean()
        return la + lb

    def kl(p, q):
        return float((p * (np.log(np.clip(p, 1e-9, 1)) - np.log(np.clip(q, 1e-9, 1)))).sum(1).mean())

    melhor = math.inf
    for ep in range(args.epocas):
        aluno.train()
        t0, soma = time.time(), 0.0
        for x, a in dl:
            x, a = x.to(dev), a.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                z = aluno(x)
            perda = perda_de(z.float(), a)
            opt.zero_grad(set_to_none=True)
            scaler.scale(perda).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            soma += perda.item()
        z = prever(aluno, ds_va, dev, args.lote * 2)
        k3 = kl(alvo[va, :3], torch.softmax(z[:, :3], 1).numpy())
        ki = kl(alvo[va, 3:], torch.softmax(z[:, 3:], 1).numpy())
        print(f"época {ep + 1}/{args.epocas}: perda {soma / len(dl):.4f}  val KL 3 classes {k3:.4f}"
              f"  val KL indígena {ki:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if k3 + ki < melhor:
            melhor = k3 + ki
            torch.save(aluno.state_dict(), args.saida / "aluno.pt")

    # 3) avaliação
    aluno.load_state_dict(torch.load(args.saida / "aluno.pt", map_location=dev))
    te = (m.split == "test").to_numpy()
    z = prever(aluno, Fotos(m.arquivo[te], args.dados, alvo[te], transf_aluno(False)), dev,
               args.lote * 2)
    pa3, pai = torch.softmax(z[:, :3], 1).numpy(), torch.softmax(z[:, 3:], 1).numpy()[:, 1]
    mt = m[te].reset_index(drop=True)
    tc = mt.tri_cons.to_numpy()
    y3 = mt.raca[tc].map(TRES.index).to_numpy()
    r = {}
    print("\n=== saída A: branca/parda/preta (teste, consistentes) ===")
    r["prof3"] = destilar.resumo(P3[te][tc], y3, "professor")
    r["aluno3"] = destilar.resumo(pa3[tc], y3, "aluno")

    ic = (mt.ind_y >= 0).to_numpy()
    yi = mt.ind_y[ic].to_numpy()
    print("\n=== saída B: indígena (teste) ===")
    for nome, p in [("professor", Pi[te, 1][ic]), ("aluno", pai[ic])]:
        pred = p >= 0.5
        r[f"ind_{nome}"] = {"auc": round(roc_auc_score(yi, p), 4),
                            "recall": round(float(pred[yi == 1].mean()), 4),
                            "precisao": round(float(yi[pred].mean()) if pred.any() else 0, 4),
                            "ece_p_indigena": round(ece_bin(p, yi), 4)}
        print(f"{nome:10s} " + "  ".join(f"{k} {v}" for k, v in r[f"ind_{nome}"].items()))
    print("P(indígena) média por grupo (aluno):")
    g = np.where(mt.ind_y == 1, "indígena consistente",
                 np.where(mt.ind_y == -1, "mudou indígena<->outra", "sempre " + mt.raca))
    print((pd.Series(pai).groupby(g).mean() * 100).round(1).to_string())

    if args.externo:
        ext = pd.read_csv(args.externo)
        ext = ext[ext.raca.isin(QUATRO) & (ext.saturacao >= 12)].reset_index(drop=True)
        ze = prever(aluno, Fotos(ext.arquivo, args.dados, np.zeros((len(ext), 5)),
                                 transf_aluno(False)), dev, args.lote * 2)
        pe3, pei = torch.softmax(ze[:, :3], 1).numpy(), torch.softmax(ze[:, 3:], 1).numpy()[:, 1]
        t = pd.DataFrame(pe3 * 100, columns=TRES).assign(indigena=pei * 100, fonte=ext.fonte)
        print("\n=== teste externo (aluno, médias %) ===")
        print(t.groupby("fonte").mean().round(1).to_string())
        print("fração com P(indígena) >= 50%:")
        print((t.assign(i50=t.indigena >= 50).groupby("fonte").i50.mean() * 100).round(1).to_string())

    # 4) exporta
    aluno = aluno.float().cpu().eval()
    onnx = args.saida / "modelo.onnx"
    torch.onnx.export(aluno, torch.randn(1, 3, LADO, LADO), str(onnx), opset_version=17,
                      input_names=["imagem"], output_names=["logits"],
                      dynamic_axes={"imagem": {0: "n"}, "logits": {0: "n"}}, dynamo=False)
    meta = {"saidas": {"A": {"logits": [0, 3], "classes": TRES},
                       "B": {"logits": [3, 5], "classes": ["outra", "indigena"]}},
            "lado": LADO, "media": MEDIA, "desvio": DESVIO, "aluno": args.aluno,
            "calibracao_indigena": {"T": Ti, "b": bi.tolist()}, "metricas": r}
    (args.saida / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nexportado: {onnx} ({onnx.stat().st_size / 1e6:.1f} MB) + meta.json")


if __name__ == "__main__":
    main()
