"""Monta a tabela de rótulos a partir das bases abertas do TSE (consulta_cand).

Uma linha por candidatura (ano + SQ_CANDIDATO), com:
- raca: cor/raça declarada NAQUELA eleição
- pessoa_id: hash de nome normalizado + data de nascimento (o CPF não é público),
  usado para ligar a mesma pessoa entre eleições e para separar train/val/test
- racas_historico / n_racas: todas as declarações da pessoa nos anos baixados
- split: train/val/test decidido pela pessoa, nunca pela foto

Uso:  python rotulos.py --anos 2016 2018 2020 2022 2024
Saída: dados/rotulos.csv.gz
"""
import argparse
import hashlib
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).parent
DIR_ROT = RAIZ / "dados" / "rotulos"
URL = "https://cdn.tse.jus.br/estatistica/sead/odsele/consulta_cand/consulta_cand_{ano}.zip"

RACAS = {
    "BRANCA": "branca",
    "PARDA": "parda",
    "PRETA": "preta",
    "INDÍGENA": "indigena",
    "AMARELA": "amarela",
}
COLUNAS = ["ANO_ELEICAO", "SG_UF", "SQ_CANDIDATO", "NM_CANDIDATO",
           "DT_NASCIMENTO", "DS_GENERO", "DS_COR_RACA"]


def normalizar_nome(nome: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    return " ".join(sem_acento.upper().split())


def hash_pessoa(nome: str, nasc: str) -> str:
    return hashlib.sha1(f"{normalizar_nome(nome)}|{nasc}".encode()).hexdigest()[:16]


def split_de(pessoa_id: str) -> str:
    # determinístico: a mesma pessoa cai sempre no mesmo split, em qualquer máquina
    b = int(pessoa_id[:8], 16) % 100
    return "train" if b < 80 else "val" if b < 90 else "test"


def baixar(ano: int) -> Path:
    destino = DIR_ROT / f"consulta_cand_{ano}.zip"
    if not destino.exists():
        DIR_ROT.mkdir(parents=True, exist_ok=True)
        print(f"baixando consulta_cand {ano}...")
        urllib.request.urlretrieve(URL.format(ano=ano), destino)
    return destino


def ler_ano(ano: int) -> pd.DataFrame:
    with zipfile.ZipFile(baixar(ano)) as z:
        with z.open(f"consulta_cand_{ano}_BRASIL.csv") as f:
            d = pd.read_csv(f, sep=";", encoding="latin1", usecols=COLUNAS, dtype=str)
    d = d.drop_duplicates("SQ_CANDIDATO")  # o CSV repete a candidatura por turno
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anos", type=int, nargs="+", default=[2016, 2018, 2020, 2022, 2024])
    args = ap.parse_args()

    d = pd.concat([ler_ano(a) for a in args.anos], ignore_index=True)
    d["raca"] = d["DS_COR_RACA"].map(RACAS)
    nasc_ok = d["DT_NASCIMENTO"].str.match(r"\d\d/\d\d/\d{4}$", na=False)
    d = d[d["raca"].notna() & nasc_ok].copy()

    d["pessoa_id"] = [hash_pessoa(n, b) for n, b in zip(d["NM_CANDIDATO"], d["DT_NASCIMENTO"])]
    hist = d.groupby("pessoa_id")["raca"].agg(lambda s: "|".join(sorted(set(s))))
    d["racas_historico"] = d["pessoa_id"].map(hist)
    d["n_racas"] = d["racas_historico"].str.count(r"\|") + 1
    d["n_eleicoes"] = d.groupby("pessoa_id")["ANO_ELEICAO"].transform("nunique")
    d["split"] = d["pessoa_id"].map(split_de)

    out = d.rename(columns={"ANO_ELEICAO": "ano", "SG_UF": "uf", "SQ_CANDIDATO": "sq",
                            "DS_GENERO": "genero"})
    out = out[["ano", "uf", "sq", "pessoa_id", "genero", "raca",
               "racas_historico", "n_racas", "n_eleicoes", "split"]]
    destino = RAIZ / "dados" / "rotulos.csv.gz"
    out.to_csv(destino, index=False)

    print(f"{len(out)} candidaturas, {out.pessoa_id.nunique()} pessoas -> {destino}")
    print(out.groupby(["ano", "raca"]).size().unstack(fill_value=0))
    pessoas = out.drop_duplicates("pessoa_id")
    multi = pessoas[pessoas.n_eleicoes > 1]
    print(f"\npessoas em >1 eleição: {len(multi)}; "
          f"com declarações diferentes: {(multi.n_racas > 1).sum()} "
          f"({(multi.n_racas > 1).mean():.1%})")
    print(multi[multi.n_racas > 1].racas_historico.value_counts().head(8))


if __name__ == "__main__":
    main()
