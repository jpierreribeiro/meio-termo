# Rastreabilidade dos números do artigo

Cada número do texto aponta para o arquivo e o campo de onde saiu. Para regenerar tudo:

```bash
.venv/bin/python artigo/analises.py          # numeros/*.json|csv  (~15 min na CPU por causa do aluno)
.venv/bin/python artigo/tabelas_figuras.py   # tabelas/*.csv e figuras/*.png
.venv/bin/python artigo/build_artigo.py --pdf
.venv/bin/python artigo/checar_marcadores.py # marcadores de texto de LLM
```

| Número no texto | Onde aparece | Fonte |
|---|---|---|
| 1.569.137 candidaturas; 1.178.425 pessoas | Resumo, 4.1, Tab. 1 | `01_base_tse.json`: `candidaturas`, `pessoas` |
| % por cor e ano (51,5% → 47,1% branca etc.) | 4.1, Tab. 1 | `01_base_tse.json`: `raca_por_ano_pct` |
| 290.724 pessoas com 2+ eleições; 80.584 (27,7%) mudaram | Resumo, 4.1, 5, 6 | `01_base_tse.json`: `pessoas_2mais_eleicoes`, `mudaram_*` |
| 67,3% branca–parda; 25,8% parda–preta; 2,6% branca–preta | Resumo, 4.1 | `01_base_tse.json`: `historicos_mais_comuns_entre_quem_mudou_pct` |
| 113.491 fotos usadas | Resumo, 3.1 | log de `destilar2.py` (Colab) = filtro de `analises.py` seção 3 |
| 117.703 fotos coloridas; 3.625 amarela; Tab. 2 | 3.1, Tab. 2 | `02_fotos.json`: `coloridas_por_raca_consistencia` |
| Sonda 73,6% / 62,1% / 64,1% | 4.2, Tab. 3 | **log Colab** (sonda B/16, todos os dados); não refeito localmente |
| Professor 75,0%; revocação 80,0/65,7/79,4; matriz | Resumo, 4.2, Tab. 3 e 4 | `03_professor_3classes.json`: `teste_consistentes_calibrado` |
| Sem calibrar 74,8%; parda 62,3%; NLL 0,601 → 0,591; ECE 2,6% | 4.4, Tab. 3 | `03_professor_3classes.json`: `teste_consistentes_sem_calibrar` e `_calibrado` |
| 17 e 39 erros branca↔preta | 4.2 | `matriz` acima |
| 3.613 fotos de quem mudou; 96,8% | Resumo, 4.2 | `03_professor_3classes.json`: `ambiguos_teste` |
| P(parda) mediana 59,2%; 31,6% > 70%; máx. 92,8% | 4.2 | `03_professor_3classes.json`: `p_parda_pardos_consistentes` |
| Escala 0,32 / 0,63 / 0,97 / 1,45 / 1,66 | 4.3, Tab. 6, Fig. 1 | `04_nota_por_historico.csv`, `04_notas_teste.npz` |
| Censo: preta 47,5% e parda 42,9% para sempre pretos | 4.4, 5 | `03_professor_3classes.json`: `prior_censo` |
| 70,7% e 24,0% (proporções iguais) | 4.4, 5 | `04_nota_por_historico.csv`, linha `preta` |
| Indígena: 11.376 fotos; 424 (3,7%); AUC 0,919; 53,8% / 83,8%; ECE 24,6% → 0,5% | 4.5, Tab. 7 | `05_indigena.json` |
| P(indígena) média 53,2% / 18,9% / 1,2–2,3% | 4.5 | `05_indigena.json`: `p_indigena_media_por_grupo_teste` |
| Aluno 74,2%; ECE 1,7%; concordância 90,5%; Δ escala 0,09 | Resumo, 4.6, Tab. 3 | `06_aluno.json` |
| Aluno indígena AUC 0,899; 50,2% / 87,7%; ECE 0,4% | 4.6, Tab. 7 | `06_aluno.json`: `saida_B` |
| Espelhamento: mediana 2,7; p90 9,6 pontos | 4.6 | `06_aluno.json`: `instabilidade_espelho_pontos` |
| UTKFace 89,8% / 89,4%; médias 77,6% / 77,1%; indígena 1,3% / 2,1% | 4.6 | `07_utkface.json` |
| Iluminação: 70,3–72,5%; Δ 0,04–0,08 | 4.6, Tab. 8 | `08_iluminacao_simulada.txt` (gerado por `iluminacao.py`) |
| 8,4 MB; diferença fp16 × fp32 | 3.3 | `ls -la modelos/aluno2/aluno2_w16.onnx`; conferência registrada em `docs/METODO.md` §4.5 |
| 90.957 fotos de treino do aluno; 6 épocas; 1:27 | 3.3 | logs do Colab (`destilar2.py`, `treino.py`) |
| Censo 2022: 45,3 / 43,5 / 10,2 | 1 | IBGE (2023) — conferir na fonte |

## Divergências conhecidas

- **Sonda:** o número vem do log do Colab, com outro recorte do conjunto de teste. Está marcado na nota da Tabela 3.
- **Iluminação × Tabela 3:** o "Original" da Tabela 7 (72,3%) é menor que o do aluno na Tabela 3 (74,2%). A Tabela 7 usa uma subamostra de 1.500 fotos, com uma passagem só e sem a média com a foto espelhada. As duas medições são internamente consistentes, mas não são comparáveis entre si.
- **Prior do Censo:** o Colab usou parda = 45,5% (erro de digitação). O `analises.py` usa 45,3%. A diferença nos resultados é de no máximo 0,1 ponto.
| IC 95% da acurácia balanceada (74,0–76,0%), da revocação parda (63,8–67,6%), da AUC indígena (0,902–0,936) e da revocação indígena (48,7–59,0%) | 4.2, 4.5 | `09_intervalos.json` (`intervalos.py`, bootstrap por `pessoa_id`, 1.000 repetições, semente 20261117) |
| Região e gênero: 75,0% Sudeste, 67,1% Nordeste; branca 92,0% Sul e 56,3% Nordeste; parda 57,6% mulheres e 70,1% homens; Tab. 5 | 4.2, Tab. 5 | `10_subgrupos.csv` (`intervalos.py`) |
