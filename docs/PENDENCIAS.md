# Pendências

## Do autor (decisões que só você pode tomar)

- [x] **Autoria:** Jean Pierre Ribeiro dos Santos, pesquisador independente, autor único (17/11).
- [x] **Destino:** preprint no SciELO Preprints (17/11). Falta submeter; o SciELO pede declaração de ética e de disponibilidade de dados.
- [ ] **Comitê de Ética:** confirmar com o CEP se a pesquisa se enquadra na dispensa da Resolução CNS 510/2016 (art. 1º, parágrafo único) ou se precisa de submissão. Rosto com cor/raça é dado sensível, mesmo sendo público. (19/11: a nota em vermelho saiu do texto do artigo; a decisão continua pendente e é exigida pelo SciELO.)
- [x] **Código:** público em https://github.com/jpierreribeiro/meio-termo (MIT), sem `dados/` nem `modelos/` (17/11).
- [x] **Site Meio-Termo:** publicado na Railway (projeto `uruquim`, serviço `site`) em https://www.jpierreribeiro.com.br/meio-termo/. Fonte do hub: `~/Documents/uruquim-site` (`./sync.sh && railway up --service site --ci`). Domínio registrado na Umbler, DNS na Cloudflare (proxy desligado). O uruquim.com.br saiu do serviço em 18/11.

## Do texto (marcadas em vermelho no .docx)

- [x] **Bancas de heteroidentificação:** frase reescrita com Dias e Tavares Junior (2018).
- [x] **Referências conferidas em 17/11** (Muniz reescrito conforme o resumo; Bueno e Dunning confirmado; DOIs de Campos e Machado, Janusz e Muniz; DOU de 10/04/2018, p. 34; título do IBGE). **Falta só a página final de Stark e Hutson (2022)**, que começa na p. 922. Em 19/11 a referência passou a citar só a p. 922 (os PDFs da Fordham e da UW bloqueiam download automático); trocar por p. 922-final quando conferir. Antes: Muniz (2012) e Bueno e Dunning (2017), conferir se dizem o que o texto atribui a eles. Conferir também as páginas e os números de Campos e Machado (2015), Janusz (2021), Stark e Hutson (2022) e Dias e Tavares Junior (2018), a data do DOU da Portaria nº 4/2018 e o título exato da publicação do IBGE (2023).
- [x] **Datas de acesso:** TSE 14/11/2026, timm 15/11/2026 (datas dos arquivos).

## Análises que fortaleceriam o artigo

- [ ] **Adorno nas fotos indígenas do TSE:** medir a revocação com e sem adorno nas fotos de teste do TSE, por exemplo com um classificador CLIP zero-shot ou com uma anotação manual de uma amostra. Hoje a dependência de adornos é só uma hipótese no texto (4.5).
- [x] **Região e gênero** feitos (`intervalos.py`, Tabela 5). Idade continua pendente. Antes: o desempenho muda entre estados? E entre homens e mulheres? Os dados já estão em `manifesto.csv` (`uf`, `genero`).
- [x] **Intervalos de confiança** feitos (`intervalos.py`, bootstrap por pessoa). Antes: fazer bootstrap por pessoa para a acurácia balanceada e a AUC. Hoje os valores aparecem sem IC.
- [ ] **Repetir a sonda localmente** com o mesmo recorte de teste dos outros modelos, para tirar a nota de rodapé da Tabela 3.
- [ ] **Teste com fotos de celular obtidas com consentimento** (voluntários, termo assinado), para medir o efeito real de câmera, sombra e contraluz. É o ponto fraco mais visível na demonstração.
- [ ] **Avaliação por várias pessoas** de uma amostra de fotos, para comparar autodeclaração e percepção de terceiros. Responderia diretamente à pergunta original do projeto.
