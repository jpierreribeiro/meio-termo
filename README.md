# Meio-Termo: o que o rosto diz sobre a cor declarada

Quanto uma foto do rosto informa sobre a cor ou raça que a pessoa **declara**? Este projeto responde com as bases públicas do TSE: 1,57 milhão de candidaturas (2016–2024) e 113 mil fotos de candidatos (2018–2024).

- **Demonstração:** https://www.jpierreribeiro.com.br/meio-termo/ (o modelo roda no navegador e a foto não sai do aparelho)
- **Artigo (NBR 6022, em revisão):** [`artigo/artigo_cor_declarada_ABNT.pdf`](artigo/artigo_cor_declarada_ABNT.pdf)

## Principais resultados

| | |
|---|---|
| Pessoas que concorreram 2+ vezes e mudaram a declaração | **27,7%** (2/3 entre branca e parda) |
| Acurácia balanceada (branca, parda, preta), CLIP ViT-B/16 ajustado | **75,0%** (IC 95%: 74,0–76,0%) |
| Revocação da classe parda | 65,7% |
| Erros entre branca e preta | abaixo de 2% |
| Fotos de quem mudou em que a previsão estava entre as declarações já usadas | 96,8% |
| Modelo do site (MobileNetV3 destilado, 8,4 MB) | 74,2% |

O teto de cerca de 75% acompanha a instabilidade do próprio rótulo: se o mesmo rosto declara cores diferentes em eleições diferentes, nenhum classificador acerta todas. Por isso o site mostra uma **distribuição** (como candidatos com rosto parecido se declararam), e não um veredito.

## O que este projeto não é

- **Não decide nada sobre ninguém.** Com 75% de acurácia e um rótulo que muda para 1 em cada 4 pessoas, o modelo não serve para bancas de heteroidentificação, cotas, seleção ou qualquer decisão individual.
- **Não mede a cor "verdadeira"** nem como terceiros veem a pessoa. Mede a associação entre aparência e autodeclaração entre candidatos.
- **Não publica rostos.** Os recortes e as tabelas de rótulos são dados pessoais sensíveis (LGPD, art. 11) e não estão neste repositório.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `rotulos.py`, `fotos.py` | Download das bases do TSE, ligação de pessoas entre eleições, detecção e recorte dos rostos |
| `treino.py`, `calibrar.py`, `destilar2.py`, `colab/` | Treino do professor (CLIP ViT-B/16), calibração e destilação para o aluno (MobileNetV3) |
| `artigo/` | Texto-fonte (`artigo.md`), análises (`analises.py`, `intervalos.py`), tabelas, figuras e o `.docx`/`.pdf` |
| `artigo/numeros/` | Todos os números do artigo, já agregados. `rastreabilidade.md` diz de onde sai cada um |
| `web/` | O site Meio-Termo (estático, onnxruntime-web) com o modelo de 8,4 MB |
| `docs/` | Método, decisões, pendências e como coletar os dados |

## Reproduzir

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python rotulos.py          # candidaturas do TSE -> dados/rotulos.csv.gz
.venv/bin/python fotos.py --todos    # fotos -> dados/rostos/ (~30 GB de download)
```

O treino do professor foi feito em GPU no Colab (`colab/treino_colab.ipynb`). Os detalhes e os parâmetros estão em [`docs/METODO.md`](docs/METODO.md) e a coleta, em [`docs/COLETA.md`](docs/COLETA.md).

Para rodar o site localmente: `cd web && python3 -m http.server 8765`.

## Autor e licença

Jean Pierre Ribeiro dos Santos, pesquisador independente. Código sob licença MIT. Os dados do TSE são públicos e devem ser obtidos na fonte: https://dadosabertos.tse.jus.br/.
