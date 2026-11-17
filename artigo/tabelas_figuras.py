#!/usr/bin/env python3
"""Monta as tabelas (artigo/tabelas/*.csv) e as figuras (artigo/figuras/*.png) a partir de artigo/numeros/.
Nenhum número é recalculado aqui além de formatação. Uso: .venv/bin/python artigo/tabelas_figuras.py"""
import csv, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = Path(__file__).resolve().parent; N = A / "numeros"; T = A / "tabelas"; F = A / "figuras"
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
j = lambda f: json.loads((N / f).read_text())
br = lambda v, d=1: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
inteiro = lambda v: f"{int(v):,}".replace(",", ".")
pct = lambda v, d=1: br(100 * v, d)


def salvar(nome, linhas):
    with (T / nome).open("w", newline="", encoding="utf-8") as f: csv.writer(f).writerows(linhas)
    print("->", nome)


base, fotos, prof, ind, al, utk = (j(x) for x in ["01_base_tse.json", "02_fotos.json", "03_professor_3classes.json",
                                                 "05_indigena.json", "06_aluno.json", "07_utkface.json"])
# Tabela 1: candidaturas por ano e cor/raça declarada
L = [["Eleição", "Candidaturas", "Branca (%)", "Parda (%)", "Preta (%)", "Indígena (%)", "Amarela (%)"]]
for ano, n in base["candidaturas_por_ano"].items():
    r = base["raca_por_ano_pct"][ano]
    L.append([ano, inteiro(n)] + [br(r[k], 1) for k in ["branca", "parda", "preta", "indigena", "amarela"]])
L.append(["Total", inteiro(base["candidaturas"]), "", "", "", "", ""])
salvar("t1_candidaturas.csv", L)

# Tabela 2: fotos coloridas usadas
c = fotos["coloridas_por_raca_consistencia"]
L = [["Cor/raça declarada na foto", "Declaração sempre igual", "Mudou entre eleições", "Total"]]
for k, nome in [("branca", "Branca"), ("parda", "Parda"), ("preta", "Preta"), ("indigena", "Indígena"), ("amarela", "Amarela*")]:
    a, b = c[f"{k}|consistente"], c[f"{k}|mudou"]
    L.append([nome, inteiro(a), inteiro(b), inteiro(a + b)])
tot_a = sum(v for k, v in c.items() if k.endswith("consistente")); tot_b = sum(v for k, v in c.items() if k.endswith("mudou"))
L.append(["Total", inteiro(tot_a), inteiro(tot_b), inteiro(tot_a + tot_b)])
salvar("t2_fotos.csv", L)

# Tabela 3: matriz de confusão do professor calibrado (teste, consistentes)
m = np.array(prof["teste_consistentes_calibrado"]["matriz"]); rec = prof["teste_consistentes_calibrado"]["recall"]
pre = prof["teste_consistentes_calibrado"]["precisao"]
L = [["Declarada \\ Prevista", "Branca", "Parda", "Preta", "Total", "Revocação (%)"]]
for i, k in enumerate(["Branca", "Parda", "Preta"]):
    L.append([k] + [inteiro(v) for v in m[i]] + [inteiro(m[i].sum()), pct(list(rec.values())[i])])
L.append(["Precisão (%)"] + [pct(v) for v in pre.values()] + ["", ""])
salvar("t3_confusao.csv", L)

# Tabela 4: escala 0-2 por histórico
g = pd.read_csv(N / "04_nota_por_historico.csv")
ordem = ["branca", "branca|parda", "parda", "parda|preta", "preta", "branca|preta", "branca|parda|preta"]
nome = {"branca": "Sempre branca", "branca|parda": "Branca e parda", "parda": "Sempre parda", "parda|preta": "Parda e preta",
        "preta": "Sempre preta", "branca|preta": "Branca e preta", "branca|parda|preta": "Branca, parda e preta"}
L = [["Declarações da pessoa (2016–2024)", "Fotos", "P(branca)", "P(parda)", "P(preta)", "Escala 0–2: média (DP)"]]
for k in ordem:
    r = g[g.grupo == k].iloc[0]
    L.append([nome[k], inteiro(r.n), pct(r.p_branca), pct(r.p_parda), pct(r.p_preta), f"{br(r.nota_media, 2)} ({br(r.nota_dp, 2)})"])
salvar("t4_escala.csv", L)

# Tabela 5: desempenho dos modelos (teste, consistentes)
P, Pc, S = prof["teste_consistentes_calibrado"], prof["teste_consistentes_sem_calibrar"], al["saida_A_teste_consistentes"]
L = [["Modelo", "Acurácia balanceada (%)", "Revocação parda (%)", "Precisão parda (%)", "ECE (%)", "NLL"],
     ["Sonda linear (CLIP ViT-B/16 congelado)¹", "73,6", "62,1", "64,1", "–", "–"],
     ["Professor ViT-B/16 ajustado, sem calibrar", pct(Pc["acuracia_balanceada"]), pct(Pc["recall"]["parda"]), pct(Pc["precisao"]["parda"]), pct(Pc["ece"]), br(Pc["nll"], 3)],
     ["Professor ViT-B/16 ajustado, calibrado", pct(P["acuracia_balanceada"]), pct(P["recall"]["parda"]), pct(P["precisao"]["parda"]), pct(P["ece"]), br(P["nll"], 3)],
     ["Aluno MobileNetV3 (site), com espelhamento", pct(S["acuracia_balanceada"]), pct(S["recall"]["parda"]), pct(S["precisao"]["parda"]), pct(S["ece"]), br(S["nll"], 3)]]
salvar("t5_modelos.csv", L)

# Tabela 6: indígena
B = al["saida_B"]
L = [["Modelo", "AUC", "Revocação em P ≥ 50% (%)", "Precisão em P ≥ 50% (%)", "ECE de P(indígena) (%)"],
     ["Professor, sem calibrar", br(ind["auc"], 3), "–", "–", pct(ind["ece_sem_calibrar"])],
     ["Professor, calibrado", br(ind["auc"], 3), pct(ind["recall_05"]), pct(ind["precisao_05"]), pct(ind["ece_calibrado"])],
     ["Aluno (site), com espelhamento", br(B["auc"], 3), pct(B["recall_05"]), pct(B["precisao_05"]), pct(B["ece"])]]
salvar("t6_indigena.csv", L)

# Tabela 8: professor calibrado por região e gênero (intervalos.py)
sg = pd.read_csv(N / "10_subgrupos.csv")
L = [["Grupo", "Fotos", "Acurácia balanceada (%) [IC 95%]", "Revocação branca (%)", "Revocação parda (%)", "Revocação preta (%)"]]
ordem = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul", "FEMININO", "MASCULINO"]
rot = {"FEMININO": "Mulheres", "MASCULINO": "Homens"}
for g in ordem:
    r = sg[sg.grupo == g].iloc[0]
    L.append([rot.get(g, g), inteiro(r.fotos), f"{pct(r.acuracia_balanceada)} [{pct(r.ic_inf)}–{pct(r.ic_sup)}]",
              pct(r.recall_branca), pct(r.recall_parda), pct(r.recall_preta)])
salvar("t8_subgrupos.csv", L)

# Tabela 7: iluminação simulada (sem correção)
txt = (N / "08_iluminacao_simulada.txt").read_text().split("== mundo cinza ==")[0]
nomes = {"original": "Original", "escuro (-1 stop)": "Metade da luz (−1 stop)", "muito escuro (-2)": "Um quarto da luz (−2 stops)",
         "claro (+1)": "Dobro da luz (+1 stop)", "luz quente": "Luz quente (amarelada)", "escuro+quente": "Metade da luz e quente",
         "luz fria": "Luz fria (azulada)"}
L = [["Condição simulada", "Acurácia balanceada (%)", "P(branca) média entre brancos (%)", "Mudança média na escala 0–2"]]
for linha in txt.splitlines():
    mm = re.match(r"(.+?)\s+bal-acc ([\d.]+) \| brancos→P\(branca\)\s+([\d.]+)% \| desvio da nota vs original ([\d.]+)", linha)
    if mm: L.append([nomes[mm.group(1).strip()], pct(float(mm.group(2))), br(float(mm.group(3))), br(float(mm.group(4)), 2)])
salvar("t7_iluminacao.csv", L)

# ---------------- figuras (escala de cinza, sem rostos)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 10})
z = np.load(N / "04_notas_teste.npz", allow_pickle=True)
fig, ax = plt.subplots(figsize=(6.3, 3.4), dpi=300)
ords = ["branca", "branca|parda", "parda", "parda|preta", "preta"]
dados = [z["nota"][z["grupo"] == k] for k in ords]
bp = ax.boxplot(dados, orientation="horizontal", widths=.55, showfliers=False, patch_artist=True, medianprops=dict(color="black", lw=1.4))
for b in bp["boxes"]: b.set(facecolor="#d9d9d9", edgecolor="black")
ax.set_yticks(range(1, 6), [nome[k] + f"\n(n = {inteiro(len(d))})" for k, d in zip(ords, dados)])
ax.set_xlim(0, 2); ax.set_xticks([0, .5, 1, 1.5, 2], ["0\nbranca", "0,5", "1\nparda", "1,5", "2\npreta"])
ax.set_xlabel("Escala 0–2 = P(parda) + 2 × P(preta)"); ax.invert_yaxis()
ax.grid(axis="x", color="#cccccc", lw=.6); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(F / "fig1_escala_por_historico.png"); plt.close(fig); print("-> fig1")

# diagrama de confiabilidade do professor calibrado (recalcula as faixas a partir dos logits)
import sys; sys.path.insert(0, str(A))
from analises import softmax
RAIZ = A.parent
man = pd.read_csv(RAIZ / "dados/manifesto.csv", dtype={"pessoa_id": str})
man = man[(man.saturacao >= 12) & (man.split == "test") & (man.n_racas == 1) & man.raca.isin(["branca", "parda", "preta"])]
zz = np.load(RAIZ / "modelos/aluno2/logits_prof3.npz"); Lg = pd.DataFrame(zz["logits"], index=zz["arquivos"])
man = man[man.arquivo.isin(Lg.index)]
c3 = json.loads((RAIZ / "modelos/calibracao_prof3.json").read_text())
p = softmax(Lg.loc[man.arquivo].to_numpy() / c3["T"] + np.array(c3["b"])); y = man.raca.map({"branca": 0, "parda": 1, "preta": 2}).to_numpy()
assert len(y) == prof["teste_consistentes_calibrado"]["n"]
conf, acc = p.max(1), p.argmax(1) == y
bordas = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
for a, b in zip(bordas[:-1], bordas[1:]):
    s = (conf > a) & (conf <= b)
    if s.sum() >= 20: xs.append(conf[s].mean()); ys.append(acc[s].mean()); ns.append(s.sum())
fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=300)
ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
ax.plot(xs, ys, "o-", color="black", ms=4)
ax.set_xlim(.3, 1); ax.set_ylim(.3, 1)
ax.set_xlabel("Probabilidade da classe mais provável"); ax.set_ylabel("Fração de acertos")
ax.set_xticks([.4, .6, .8, 1], ["40%", "60%", "80%", "100%"]); ax.set_yticks([.4, .6, .8, 1], ["40%", "60%", "80%", "100%"])
ax.grid(color="#e0e0e0", lw=.6)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(F / "fig2_confiabilidade.png"); plt.close(fig); print("-> fig2")
