#!/usr/bin/env python3
"""Intervalos de confiança (bootstrap por pessoa) e resultados por região e gênero.

Usa as mesmas entradas e filtros de analises.py (seção 3 e 5), só para o professor.
Saída: artigo/numeros/09_intervalos.json e 10_subgrupos.csv

Uso: .venv/bin/python artigo/intervalos.py [--reps 2000]
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from analises import RAIZ, OUT, TRES, QUATRO, softmax, auc

REGIAO = {**dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "Norte"),
          **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
          **dict.fromkeys(["DF", "GO", "MS", "MT"], "Centro-Oeste"),
          **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
          **dict.fromkeys(["PR", "RS", "SC"], "Sul")}


def bal(P, y):
    pred = P.argmax(1)
    return float(np.mean([(pred[y == k] == k).mean() for k in range(3) if (y == k).any()]))


def recs(P, y):
    pred = P.argmax(1)
    return [float((pred[y == k] == k).mean()) if (y == k).any() else np.nan for k in range(3)]


def carregar():
    m = pd.read_csv(RAIZ / "dados/manifesto.csv", dtype={"sq": str, "pessoa_id": str})
    cor = m[(m.status == "ok") & (m.saturacao >= 12)]
    z3 = np.load(RAIZ / "modelos/aluno2/logits_prof3.npz"); zi = np.load(RAIZ / "modelos/aluno2/logits_profi.npz")
    L3 = pd.DataFrame(z3["logits"], index=z3["arquivos"]); Li = pd.DataFrame(zi["logits"], index=zi["arquivos"])
    c3 = json.loads((RAIZ / "modelos/calibracao_prof3.json").read_text())
    ci = json.loads((RAIZ / "modelos/aluno2/meta.json").read_text())["calibracao_indigena"]
    d = cor[cor.raca.isin(QUATRO) & cor.racas_historico.map(lambda h: all(x in QUATRO for x in h.split("|")))].copy()
    d = d[d.arquivo.isin(L3.index)].reset_index(drop=True)
    P3 = softmax(L3.loc[d.arquivo].to_numpy() / c3["T"] + np.array(c3["b"]))
    d["pind"] = softmax(Li.loc[d.arquivo].to_numpy() / ci["T"] + np.array(ci["b"]))[:, 1]
    d["regiao"] = d.uf.map(REGIAO)
    return d, P3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=1000); a = ap.parse_args()
    rng = np.random.default_rng(20261117)
    d, P3 = carregar()
    te = (d.split == "test").to_numpy()
    cons = te & (d.n_racas == 1).to_numpy() & d.raca.isin(TRES).to_numpy()
    dc = d[cons].reset_index(drop=True); Pc = P3[cons]
    yc = dc.raca.map({k: i for i, k in enumerate(TRES)}).to_numpy()

    hist = d.racas_historico.str.split("|")
    d["ind_y"] = hist.map(lambda h: 1 if h == ["indigena"] else (0 if "indigena" not in h else -1))
    ti = te & (d.ind_y >= 0).to_numpy()
    di = d[ti].reset_index(drop=True)

    def boot(df, f):
        """Reamostra pessoas (com reposição) e reaplica f aos índices das fotos sorteadas."""
        grupos = df.groupby("pessoa_id").indices
        chaves = list(grupos); idx = [grupos[k] for k in chaves]
        vals = []
        for _ in range(a.reps):
            s = rng.integers(0, len(idx), len(idx))
            vals.append(f(np.concatenate([idx[j] for j in s])))
        return np.array(vals)

    out = {"reps": a.reps, "unidade": "pessoa", "pessoas_teste_consistentes": int(dc.pessoa_id.nunique()),
           "pessoas_teste_indigena": int(di.pessoa_id.nunique())}
    b = boot(dc, lambda i: bal(Pc[i], yc[i]))
    out["acuracia_balanceada"] = {"ponto": bal(Pc, yc), "ic95": np.quantile(b, [.025, .975]).tolist()}
    r = boot(dc, lambda i: recs(Pc[i], yc[i])[1])
    out["recall_parda"] = {"ponto": recs(Pc, yc)[1], "ic95": np.quantile(r, [.025, .975]).tolist()}
    yi, pi = di.ind_y.to_numpy(), di.pind.to_numpy()
    bi = boot(di, lambda i: auc(pi[i], yi[i]))
    out["auc_indigena"] = {"ponto": auc(pi, yi), "ic95": np.quantile(bi, [.025, .975]).tolist()}
    bri = boot(di, lambda i: ((pi[i] >= .5) & (yi[i] == 1)).sum() / max(1, (yi[i] == 1).sum()))
    out["recall_indigena_05"] = {"ponto": float(((pi >= .5) & (yi == 1)).sum() / yi.sum()), "ic95": np.quantile(bri, [.025, .975]).tolist()}
    (OUT / "09_intervalos.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)); print("-> 09_intervalos.json")

    linhas = []
    for col in ["regiao", "genero"]:
        for v, g in dc.groupby(col):
            i = g.index.to_numpy()
            bb = boot(g.reset_index(drop=True), lambda j, i=i: bal(Pc[i][j], yc[i][j]))
            rr = recs(Pc[i], yc[i])
            linhas.append({"recorte": col, "grupo": v, "fotos": len(i), "pessoas": g.pessoa_id.nunique(),
                           "n_branca": int((yc[i] == 0).sum()), "n_parda": int((yc[i] == 1).sum()), "n_preta": int((yc[i] == 2).sum()),
                           "acuracia_balanceada": bal(Pc[i], yc[i]), "ic_inf": np.quantile(bb, .025), "ic_sup": np.quantile(bb, .975),
                           "recall_branca": rr[0], "recall_parda": rr[1], "recall_preta": rr[2]})
    t = pd.DataFrame(linhas); t.round(4).to_csv(OUT / "10_subgrupos.csv", index=False); print("-> 10_subgrupos.csv")
    print(json.dumps(out, indent=1)); print(t.round(3).to_string())


if __name__ == "__main__":
    main()
