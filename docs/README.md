# pardos-tse: comece aqui

Pesquisa sobre quanto uma foto do rosto informa sobre a cor ou raça **autodeclarada**, usando as bases públicas do TSE (candidatos de 2016 a 2024). O projeto gera três coisas:

1. **Artigo** (NBR 6022): `artigo/artigo_cor_declarada_ABNT.docx` e `.pdf`. O texto-fonte está em `artigo/artigo.md`.
2. **Site de demonstração "Meio-Termo"**: `web/`. Roda no navegador, a foto não sai do aparelho e o site **não está publicado**.
3. **Modelos**: `modelos/`. O professor (CLIP ViT-B/16) fica no Google Drive e o aluno (MobileNetV3, 8,4 MB) está no site.

## Documentos

| Arquivo | Para quê |
|---|---|
| `docs/METODO.md` | Tudo o que foi feito, do download ao site, com os parâmetros |
| `docs/DECISOES.md` | O que foi decidido, quando e por quê, e o que o projeto **não** faz |
| `docs/HANDOFF.md` | Onde o trabalho parou e por onde seguir (17/11) |
| `docs/PENDENCIAS.md` | O que falta: decisões do autor, conferências do texto e análises |
| `artigo/rastreabilidade.md` | De onde vem cada número do artigo |
| `docs/COLETA.md` | Como baixar os dados e rodar o pipeline de fotos |
| `web/LEIAME.md` | Como testar e publicar o site |

## Regras do projeto

- **Dado sensível:** rosto com cor/raça é dado sensível (LGPD, art. 11). Os recortes e o dataset ficam privados e não são redistribuídos.
- **Nada de decisão individual:** o modelo não decide nada sobre ninguém (cotas, bancas, seleção). A acurácia fica perto de 75%, e o rótulo é uma autodeclaração que muda para 1 em cada 4 pessoas.
- **Só TSE e UTKFace no artigo:** fotos coletadas em redes sociais e no WhatsApp não entram em nenhum número publicado.
- **Escrita:** antes de cada versão do artigo, rodar `artigo/checar_marcadores.py`. Ele usa o léxico de marcadores de texto de LLM da pesquisa `~/Documents/pesquisa-uso-de-IA`, e a meta é zero no núcleo.

## Regenerar o artigo

```bash
.venv/bin/python artigo/analises.py            # números (precisa de dados/ e modelos/)
.venv/bin/python artigo/tabelas_figuras.py
.venv/bin/python artigo/build_artigo.py --pdf
.venv/bin/python artigo/checar_marcadores.py
```
