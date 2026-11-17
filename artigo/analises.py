#!/usr/bin/env python3
"""Recalcula, a partir dos arquivos locais, todos os números citados no artigo.

Entradas (nada é baixado):
  dados/rotulos.csv.gz              candidaturas do TSE 2016-2024 (rotulos.py)
  dados/manifesto.csv               fotos recortadas e o split por pessoa (fotos.py)
  modelos/aluno2/logits_prof3.npz   logits do professor de 3 classes (CLIP ViT-B/16 ajustado)
  modelos/aluno2/logits_profi.npz   logits do professor indígena × outra
  modelos/aluno2/meta.json          calibração do professor indígena (T, b)
  modelos/calibracao_prof3.json     calibração do professor de 3 classes (T, b)
  modelos/aluno2/aluno2_w16.onnx    aluno publicado no site
  dados/externo_adorno.csv          só as linhas utk_* (UTKFace); fotos de redes sociais NÃO entram
Saída: artigo/numeros/*.json|csv

Uso: .venv/bin/python artigo/analises.py [--sem-aluno]
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "artigo" / "numeros"
TRES = ["branca", "parda", "preta"]
QUATRO = TRES + ["indigena"]
CENSO = np.array([0.435, 0.453, 0.102])  # Censo 2022: branca 43,5%, parda 45,3%, preta 10,2% (IBGE, 2023)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def ece(p, y, bins=15):
    conf = p.max(1); pred = p.argmax(1); acc = pred == y; e = 0
    for a, b in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        s = (conf > a) & (conf <= b)
        if s.any(): e += s.mean() * abs(acc[s].mean() - conf[s].mean())
    return float(e)


def ece_bin(p, y, bins=15):
    e = 0
    for a, b in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        s = (p > a) & (p <= b) if a > 0 else (p >= a) & (p <= b)
        if s.any(): e += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(e)


def auc(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def metricas3(p, y):
    pred = p.argmax(1)
    cm = np.zeros((3, 3), int)
    for a, b in zip(y, pred): cm[a, b] += 1
    rec = cm.diagonal() / cm.sum(1); prec = cm.diagonal() / cm.sum(0)
    nll = float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
    brier = float(((p - np.eye(3)[y]) ** 2).sum(1).mean())
    return {"n": int(len(y)), "acuracia_balanceada": float(rec.mean()),
            "recall": dict(zip(TRES, rec.round(4).tolist())), "precisao": dict(zip(TRES, prec.round(4).tolist())),
            "matriz": cm.tolist(), "ece": ece(p, y), "nll": nll, "brier": brier}


def salvar(nome, obj):
    (OUT / nome).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"-> {nome}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sem-aluno", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 1. base de candidaturas ----------
    r = pd.read_csv(RAIZ / "dados/rotulos.csv.gz", dtype={"sq": str, "pessoa_id": str})
    pessoas = r.drop_duplicates("pessoa_id")
    rep = pessoas[pessoas.n_eleicoes >= 2]
    mud = rep[rep.n_racas > 1]
    pares = mud.racas_historico.str.split("|").map(lambda h: "|".join(sorted(h)))
    base = {
        "candidaturas": int(len(r)), "pessoas": int(len(pessoas)),
        "anos": sorted(r.ano.unique().tolist()),
        "candidaturas_por_ano": r.groupby("ano").size().to_dict(),
        "raca_por_ano_pct": (r.groupby("ano").raca.value_counts(normalize=True).unstack().fillna(0) * 100).round(2).to_dict(orient="index"),
        "pessoas_2mais_eleicoes": int(len(rep)),
        "mudaram_declaracao": int(len(mud)), "mudaram_pct": round(100 * len(mud) / len(rep), 2),
        "historicos_mais_comuns_entre_quem_mudou_pct": (pares.value_counts(normalize=True).head(8) * 100).round(2).to_dict(),
    }
    salvar("01_base_tse.json", base)

    # ---------- 2. fotos ----------
    m = pd.read_csv(RAIZ / "dados/manifesto.csv", dtype={"sq": str, "pessoa_id": str})
    cor = m[(m.status == "ok") & (m.saturacao >= 12)]
    fotos = {
        "fotos_total": int(len(m)), "fotos_por_ano": m.groupby("ano").size().to_dict(),
        "fotos_coloridas": int(len(cor)), "coloridas_por_ano": cor.groupby("ano").size().to_dict(),
        "coloridas_por_raca_consistencia": {f"{k[0]}|{'consistente' if k[1] else 'mudou'}": int(v)
                                            for k, v in cor.groupby(["raca", "consistente"]).size().items()},
        "pessoas_distintas_coloridas": int(cor.pessoa_id.nunique()),
        "split_por_pessoa": cor.groupby("split").pessoa_id.nunique().to_dict(),
        "split_fotos": cor.split.value_counts().to_dict(),
        "dist_olhos_mediana_px": float(cor.dist_olhos.median()),
    }
    salvar("02_fotos.json", fotos)

    # ---------- 3. professor de 3 classes ----------
    z3 = np.load(RAIZ / "modelos/aluno2/logits_prof3.npz"); zi = np.load(RAIZ / "modelos/aluno2/logits_profi.npz")
    L3 = pd.DataFrame(z3["logits"], index=z3["arquivos"]); Li = pd.DataFrame(zi["logits"], index=zi["arquivos"])
    c3 = json.loads((RAIZ / "modelos/calibracao_prof3.json").read_text())
    meta = json.loads((RAIZ / "modelos/aluno2/meta.json").read_text())
    ci = meta["calibracao_indigena"]
    d = cor[cor.raca.isin(QUATRO) & cor.racas_historico.map(lambda h: all(x in QUATRO for x in h.split("|")))].copy()
    d = d[d.arquivo.isin(L3.index)].reset_index(drop=True)
    P3 = softmax(L3.loc[d.arquivo].to_numpy() / c3["T"] + np.array(c3["b"]))
    P3cru = softmax(L3.loc[d.arquivo].to_numpy())
    Pi = softmax(Li.loc[d.arquivo].to_numpy() / ci["T"] + np.array(ci["b"]))[:, 1]
    d["pb"], d["pp"], d["ppr"], d["pind"] = P3[:, 0], P3[:, 1], P3[:, 2], Pi
    d["nota"] = d.pp + 2 * d.ppr
    te = d.split == "test"
    cons3 = te & (d.n_racas == 1) & d.raca.isin(TRES)
    y = d.raca[cons3].map({k: i for i, k in enumerate(TRES)}).to_numpy()
    prof = {"calibracao": {"T": c3["T"], "b": c3["b"]},
            "teste_consistentes_calibrado": metricas3(P3[cons3.to_numpy()], y),
            "teste_consistentes_sem_calibrar": metricas3(P3cru[cons3.to_numpy()], y)}
    # ambíguos: a classe prevista está no histórico da pessoa?
    amb = te & (d.n_racas > 1) & d.racas_historico.map(lambda h: set(h.split("|")) <= set(TRES))
    pred = np.array(TRES)[P3[amb.to_numpy()].argmax(1)]
    prof["ambiguos_teste"] = {"n": int(amb.sum()),
                              "previsao_dentro_do_historico_pct": round(100 * float(np.mean([p in h.split("|") for p, h in zip(pred, d.racas_historico[amb])])), 2)}
    # P(parda) entre pardos consistentes
    pc = d.pp[cons3 & (d.raca == "parda")]
    prof["p_parda_pardos_consistentes"] = {"mediana": round(float(pc.median()) * 100, 1),
                                           "pct_acima_70": round(float((pc > .7).mean()) * 100, 1),
                                           "p99": round(float(pc.quantile(.99)) * 100, 1), "max": round(float(pc.max()) * 100, 1)}
    # prior do Censo: reponderação p_k * pi_k / sum
    Pc = P3[cons3.to_numpy()] * CENSO; Pc /= Pc.sum(1, keepdims=True)
    prof["prior_censo"] = {"media_por_classe_verdadeira": {TRES[k]: dict(zip(TRES, (Pc[y == k].mean(0) * 100).round(1).tolist())) for k in range(3)}}
    salvar("03_professor_3classes.json", prof)

    # nota 0-2 por grupo de histórico (teste, todas as fotos do teste dentro das 3 classes)
    t3 = d[te & d.racas_historico.map(lambda h: set(h.split("|")) <= set(TRES))].copy()
    t3["grupo"] = t3.racas_historico.map(lambda h: "|".join(sorted(set(h.split("|")))))
    g = t3.groupby("grupo").agg(n=("nota", "size"), nota_media=("nota", "mean"), nota_dp=("nota", "std"),
                                p_branca=("pb", "mean"), p_parda=("pp", "mean"), p_preta=("ppr", "mean"))
    g.round(3).to_csv(OUT / "04_nota_por_historico.csv"); print("-> 04_nota_por_historico.csv")
    np.savez_compressed(OUT / "04_notas_teste.npz", nota=t3.nota.to_numpy(), grupo=t3.grupo.to_numpy())

    # ---------- 5. indígena ----------
    hist = d.racas_historico.str.split("|")
    d["ind_y"] = hist.map(lambda h: 1 if h == ["indigena"] else (0 if "indigena" not in h else -1))
    ti = te & (d.ind_y >= 0)
    yi = d.ind_y[ti].to_numpy(); pi = d.pind[ti].to_numpy()
    picru = softmax(Li.loc[d.arquivo[ti]].to_numpy())[:, 1]
    tp = ((pi >= .5) & (yi == 1)).sum()
    ind = {"n_teste": int(ti.sum()), "indigenas_teste": int(yi.sum()), "prevalencia_pct": round(100 * yi.mean(), 2),
           "auc": auc(pi, yi), "recall_05": float(tp / yi.sum()), "precisao_05": float(tp / (pi >= .5).sum()),
           "ece_sem_calibrar": ece_bin(picru, yi), "ece_calibrado": ece_bin(pi, yi), "calibracao": ci}
    grp = np.where(d.ind_y == 1, "indígena consistente", np.where(d.ind_y == -1, "mudou indígena/outra", "sempre " + d.raca))
    ind["p_indigena_media_por_grupo_teste"] = (pd.Series(d.pind[te].to_numpy()).groupby(grp[te.to_numpy()]).mean() * 100).round(1).to_dict()
    salvar("05_indigena.json", ind)
    d.to_parquet(OUT / "_probs_professor.parquet") if False else None

    if a.sem_aluno: return
    # ---------- 6. aluno (ONNX publicado) no teste + UTKFace ----------
    import onnxruntime as ort
    from PIL import Image
    M = np.array([0.485, 0.456, 0.406], np.float32); D = np.array([0.229, 0.224, 0.225], np.float32)
    s = ort.InferenceSession(str(RAIZ / "modelos/aluno2/aluno2_w16.onnx"))

    def rodar(arqs, espelho=False):
        out = []
        for i in range(0, len(arqs), 128):
            x = np.stack([np.asarray(Image.open(RAIZ / "dados" / f).convert("RGB").resize((160, 160)), np.float32) / 255 for f in arqs[i:i + 128]])
            x = ((x - M) / D).transpose(0, 3, 1, 2)
            z = s.run(None, {"imagem": x.astype(np.float32)})[0]
            if espelho:
                z2 = s.run(None, {"imagem": x[..., ::-1].copy().astype(np.float32)})[0]
                out.append(np.c_[(softmax(z[:, :3]) + softmax(z2[:, :3])) / 2, (softmax(z[:, 3:])[:, 1] + softmax(z2[:, 3:])[:, 1]) / 2,
                                 np.abs(softmax(z[:, :3]) - softmax(z2[:, :3])).max(1)])
            else:
                out.append(np.c_[softmax(z[:, :3]), softmax(z[:, 3:])[:, 1]])
        return np.concatenate(out)

    arqs = d.arquivo[te].tolist()
    A = rodar(arqs, espelho=True)
    da = d[te].reset_index(drop=True)
    c = (da.n_racas == 1) & da.raca.isin(TRES)
    ya = da.raca[c].map({k: i for i, k in enumerate(TRES)}).to_numpy()
    Pa = A[c.to_numpy(), :3]
    Pprof = da[["pb", "pp", "ppr"]][c].to_numpy()
    yi2 = da.ind_y.to_numpy(); ok = yi2 >= 0
    al = {"n_fotos_teste": int(len(da)), "saida_A_teste_consistentes": metricas3(Pa, ya),
          "concordancia_argmax_com_professor_pct": round(100 * float((Pa.argmax(1) == Pprof.argmax(1)).mean()), 2),
          "dif_media_nota_vs_professor": round(float(np.abs((Pa[:, 1] + 2 * Pa[:, 2]) - (Pprof[:, 1] + 2 * Pprof[:, 2])).mean()), 3),
          "saida_B": {"auc": auc(A[ok, 3], yi2[ok]), "recall_05": float(((A[ok, 3] >= .5) & (yi2[ok] == 1)).sum() / yi2[ok].sum()),
                      "precisao_05": float(((A[ok, 3] >= .5) & (yi2[ok] == 1)).sum() / (A[ok, 3] >= .5).sum()), "ece": ece_bin(A[ok, 3], yi2[ok])},
          "instabilidade_espelho_pontos": {"mediana": round(float(np.median(A[:, 4])) * 100, 1), "p90": round(float(np.quantile(A[:, 4], .9)) * 100, 1)}}
    salvar("06_aluno.json", al)

    ext = pd.read_csv(RAIZ / "dados/externo_adorno.csv")
    ext = ext[ext.fonte.str.startswith("utk") & (ext.saturacao >= 12)].reset_index(drop=True)
    E = rodar(ext.arquivo.tolist(), espelho=True)
    t = pd.DataFrame(E[:, :4] * 100, columns=TRES + ["indigena"]).assign(fonte=ext.fonte)
    utk = {"n": t.fonte.value_counts().to_dict(), "media_pct": t.groupby("fonte")[TRES + ["indigena"]].mean().round(1).to_dict(orient="index"),
           "acerto_argmax_pct": {f: round(100 * float((np.array(TRES)[E[(ext.fonte == f).to_numpy(), :3].argmax(1)] == f.split("_")[1]).mean()), 1)
                                 for f in t.fonte.unique()}}
    salvar("07_utkface.json", utk)


if __name__ == "__main__":
    main()
