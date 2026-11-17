"""Junta os manifestos num só e prepara a pasta para subir ao Kaggle como dataset PRIVADO.

Uso:
  python empacotar.py                                   # monta dados/kaggle/
  kaggle datasets create -p dados/kaggle --dir-mode zip   # primeira vez (sobe privado)
  kaggle datasets version -p dados/kaggle --dir-mode zip -m "mais UFs"   # atualizações

dados/kaggle/ contém só o que sobe: manifesto.csv, dataset-metadata.json e rostos/
(hardlinks dos recortes: arquivos reais para o CLI do Kaggle, sem ocupar disco extra).
"""
import json
import os
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).parent / "dados"
SAIDA = DADOS / "kaggle"


def main():
    partes = sorted((DADOS / "manifestos").glob("*.csv"))
    m = pd.concat([pd.read_csv(p, dtype={"sq": str, "pessoa_id": str}) for p in partes])
    ok = m[m.status == "ok"].copy()
    ok["consistente"] = ok.n_racas == 1

    SAIDA.mkdir(exist_ok=True)
    ok.to_csv(SAIDA / "manifesto.csv", index=False)
    for arquivo in ok.arquivo:
        destino = SAIDA / arquivo
        if not destino.exists():
            destino.parent.mkdir(parents=True, exist_ok=True)
            os.link(DADOS / arquivo, destino)

    meta = SAIDA / "dataset-metadata.json"
    if not meta.exists():
        usuario = os.environ.get("KAGGLE_USERNAME", "SEU_USUARIO")
        meta.write_text(json.dumps({
            "title": "rostos-tse-autodeclaracao",
            "id": f"{usuario}/rostos-tse-autodeclaracao",
            "isPrivate": True,
            "licenses": [{"name": "other"}],
        }, indent=2))

    print(f"{len(partes)} UF/anos, {len(m)} fotos processadas, {len(ok)} recortes ok")
    print(m.status.value_counts().to_string())
    print("\npessoas por classe (só consistentes):")
    print(ok[ok.consistente].groupby("raca").pessoa_id.nunique().to_string())
    print(f"\npronto em {SAIDA}  ->  kaggle datasets create -p {SAIDA} --dir-mode zip")


if __name__ == "__main__":
    main()
