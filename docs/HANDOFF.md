# Handoff (17/11/2026, atualizado à tarde)

> **Atualização 17/11 à tarde:** autoria preenchida, referências conferidas, IC e Tabela 5 (região e gênero) adicionados, código público em https://github.com/jpierreribeiro/meio-termo e site na Railway (`~/Documents/uruquim-site`). Restam 3 marcações `{{red:...}}`: CEP, adorno indígena e página final de Stark e Hutson. Ver `PENDENCIAS.md`. O projeto agora é repositório git.

Estado do projeto para quem continuar, seja pessoa ou sessão do Claude. Os detalhes estão em `METODO.md`, `DECISOES.md` e `PENDENCIAS.md`. Este arquivo diz **onde parou** e **por onde seguir**.

## Onde parou

| Entrega | Estado |
|---|---|
| Artigo (NBR 6022) | Texto completo em `artigo/artigo.md`, com `.docx` e `.pdf` gerados em 16/11, por volta das 14h30. `checar_marcadores.py` dá **0 ocorrências** em 4.136 palavras. Restam **17 marcações `{{red:...}}`**, todas listadas em `PENDENCIAS.md`. |
| Números do artigo | Todos rastreados em `artigo/rastreabilidade.md`. Os que vêm só do log do Colab: sonda (73,6%), contagem de 113.491 fotos e épocas de treino. |
| Site Meio-Termo (`web/`) | Pronto e testado localmente, **não publicado**. Publicar é decisão do autor (D19). |
| Modelos | Aluno `modelos/aluno2/aluno2_w16.onnx` (8,4 MB, é o mesmo de `web/modelo/`). Os professores CLIP ViT-B/16 ficam no **Google Drive**, não nesta máquina; os logits estão em `modelos/aluno2/logits_*.npz`. |
| Dados | `dados/manifesto.csv` (187.481 recortes), `dados/rostos/`, `dados/rotulos.csv.gz`. Privados (LGPD, art. 11), não redistribuir. |

O projeto **não é repositório git**. Não há histórico de versões além das datas dos arquivos.

## Próximos passos

### Bloqueados no autor (não dá para avançar sem ele)
1. Autoria: 4 campos `[PREENCHER]` no topo de `artigo.md`.
2. Destino (evento, periódico ou preprint), que define o modelo e o limite de páginas.
3. Consulta ao CEP sobre dispensa (Res. CNS 510/2016).
4. Publicar o código? Onde?
5. Publicar o site? Comando pronto: `npx wrangler pages deploy web --project-name meio-termo`.

### Possíveis sem o autor, com o que está na máquina
- **Intervalos de confiança:** bootstrap por pessoa (`pessoa_id`) da acurácia balanceada e da AUC. As notas por foto estão em `artigo/numeros/04_notas_teste.npz`.
- **Resultados por UF e gênero:** colunas `uf` e `genero` do `manifesto.csv`. Não há coluna de idade. Para idade, seria preciso trazer a data de nascimento de `rotulos.csv.gz`.
- **Conferência das referências:** a lista está em `PENDENCIAS.md`, "Do texto". Exige consultar as fontes. Não inventar páginas nem números.

### Exigem Colab (professor CLIP no Drive)
- Sonda linear com o mesmo recorte de teste, para tirar a nota ¹ da Tabela 3.
- Adorno nas fotos indígenas do TSE (CLIP zero-shot ou anotação manual), que resolve o `[PENDENTE]` da seção 4.5.

### Exigem coleta com pessoas (fora do alcance agora)
- Fotos de celular de voluntários com termo de consentimento.
- Várias avaliações por foto (percepção de terceiros × autodeclaração).

## Regras que não podem ser quebradas

- Nenhum número do artigo pode vir de fotos de redes sociais ou de WhatsApp (D17). Só TSE e UTKFace.
- Nenhum rosto aparece no artigo, e todos os resultados são agregados.
- O modelo não decide nada sobre ninguém, e o texto não deve sugerir esse uso.
- Depois de qualquer mudança no texto: `analises.py` → `tabelas_figuras.py` → `build_artigo.py --pdf` → `checar_marcadores.py` (meta: zero no núcleo). Ver `docs/README.md`.
- Número novo no texto exige uma linha nova em `artigo/rastreabilidade.md`.
