# Coleta dos dados (TSE)

Como baixar as bases do TSE e montar os recortes localmente. Os recortes **não** são publicados.

Dataset de rostos com cor/raça **autodeclarada** a partir das bases abertas do TSE
(eleições 2016–2024). Todas as classes vêm da mesma fonte e do mesmo padrão de foto,
para o modelo não aprender "de onde veio a imagem" em vez da pessoa.

## Pipeline

| passo | script | saída |
|---|---|---|
| 1. rótulos | `rotulos.py` | `dados/rotulos.csv.gz` (1,57 mi candidaturas, 1,18 mi pessoas) |
| 2. fotos | `fotos.py --todos` | `dados/rostos/{ano}/{uf}/{sq}.jpg` 160×160 alinhado + `dados/manifestos/` |
| 3. pacote | `empacotar.py` | `dados/kaggle/` pronto para `kaggle datasets create` |

- **Pessoa** = nome normalizado + data de nascimento (CPF não é público).
- **split** (train/val/test 80/10/10) é por pessoa: a mesma pessoa em anos diferentes
  nunca aparece em dois splits.
- **Amostragem** por pessoa (`TAXAS` em `fotos.py`), ~1,6 GB no final.
- `n_racas > 1` = a pessoa mudou de declaração entre eleições (27,7% de quem concorreu
  mais de uma vez; branca↔parda é 2/3 disso). Treinar com `consistente == True`;
  usar os inconsistentes como conjunto de avaliação "ambíguo".
- `fotos.py` é retomável: UF/ano com manifesto pronto é pulada.

## Rodar na VPS

Requisitos: Ubuntu, 2+ vCPU, 2 GB RAM, **30 GB de disco** (o maior zip, SP, tem
~2–3 GB e é apagado depois; o que fica são ~1,6 GB). Download total ~30 GB.

```bash
# da máquina local: só código + modelo (rótulos dá para regenerar ou copiar, 27 MB)
rsync -av --exclude .venv --exclude dados/rostos --exclude dados/tmp \
      --exclude dados/manifestos --exclude 'dados/rotulos/*.zip' \
      ~/Documents/pardos-tse/ usuario@VPS:pardos-tse/

# na VPS
curl -LsSf https://astral.sh/uv/install.sh | sh
cd pardos-tse && uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt
[ -f dados/rotulos.csv.gz ] || .venv/bin/python rotulos.py
tmux new -s tse '.venv/bin/python fotos.py --todos 2>&1 | tee fotos.log'

# no fim
.venv/bin/python empacotar.py
export KAGGLE_USERNAME=... KAGGLE_KEY=...      # ou ~/.kaggle/kaggle.json
.venv/bin/kaggle datasets create -p dados/kaggle --dir-mode zip
```

O dataset sobe **privado** (`isPrivate: true`): são rostos de pessoas reais com
cor/raça, dado sensível pela LGPD (art. 11). Não tornar público.
