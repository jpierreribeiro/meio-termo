# Decisões

Registro das escolhas que mudaram o rumo do projeto, com o motivo de cada uma. A mais recente vem por último. As datas são de 2026.

| # | Decisão | Motivo |
|---|---|---|
| D1 | Usar só o TSE como fonte de rótulo e de foto. | Todas as classes têm o mesmo padrão de foto (oficial, frontal, fundo claro). Com várias fontes, o modelo aprenderia de onde a foto veio em vez de aprender o rosto. |
| D2 | Separar treino, validação e teste por pessoa, não por foto. | A mesma pessoa aparece em várias eleições. Separando por foto, o teste mediria memória de rosto. |
| D3 | Treinar só com declarações consistentes e usar quem mudou como conjunto "ambíguo" de avaliação. | 27,7% de quem concorreu mais de uma vez mudou de declaração. Esses rótulos não são errados: mostram onde a declaração se divide. |
| D4 | Excluir as fotos de 2016. | O TSE publicou as fotos de 2016 em escala de cinza. O modelo usa a cor da pele. |
| D5 | Trocar o Kaggle pelo Colab (GPU T4) com o Google Drive. | A conta do Kaggle sem verificação de telefone não libera internet nem GPU nos notebooks. |
| D6 | Mostrar uma distribuição de probabilidade, não um rótulo. | O autor nunca quis classificar alguém como "pardo" ou "branco". A pergunta honesta é o que pessoas com rosto parecido declararam. |
| D7 | Não afirmar que o modelo mede "como a pessoa é vista socialmente". | O rótulo é autodeclaração. Medir percepção exigiria vários avaliadores por foto, o que não é viável agora. |
| D8 | Aceitar o teto de cerca de 75% de acurácia balanceada. | O ajuste fino (0,748) quase não supera a sonda linear (0,736), e a validação parou de subir na 3ª época. Isso indica um limite do rótulo, não da arquitetura. |
| D9 | Calibrar as probabilidades (temperatura + viés por classe) e relatar ECE. | Dizer "54%" exige que, entre as fotos com 54%, cerca de 54% sejam daquela classe. |
| D10 | Destilar para MobileNetV3 com entrada de 160 px. | Precisa rodar no celular, dentro do navegador, a custo zero. A destilação perde 0,8 ponto de acurácia balanceada. |
| D11 | Rejeitar a quantização int8 e usar pesos fp16. | O int8 mudava a resposta em 10,6 pontos em média. Com fp16 a mudança é de 0,04. |
| D12 | Fazer a média de 6 variações (3 escalas × espelhamento) no navegador. | O modelo oscila até cerca de 10 pontos com o espelhamento ou com poucos pixels de deslocamento. A média reduz o p90 de 9,3 para 5,3. |
| D13 | Incluir uma segunda saída, indígena × outra (16/11). | Uma criança com aparência indígena saía como "preta 82%", porque o site só tinha o modelo de 3 classes. |
| D14 | Juntar as duas saídas numa lista só de 4 categorias somando 100% (16/11). | O autor considerou confusa a tela com "64% preta" em destaque e "98% indígena" num bloco separado. |
| D15 | Interface pensada para o celular, com a foto e o botão como moldura e o resultado embaixo (16/11). | O site é para demonstração. A maioria das pessoas vai abrir no celular. |
| D16 | Não acrescentar correção automática de iluminação (16/11). | A simulação com 1.500 fotos mostrou que mudanças globais de luz movem a resposta em 2 a 4 pontos. O caso que motivou o teste (pessoa branca com "parda 74%") não muda com correção de luz: o resultado vinha do rosto. |
| D17 | No artigo, usar só dados do TSE e do UTKFace (16/11). | As fotos coletadas em redes sociais e no WhatsApp não têm consentimento. Rosto com cor/raça é dado sensível (LGPD, art. 11). |
| D18 | Formatar o artigo pela NBR 6022, reaproveitando o padrão do artigo de marcadores de IA do autor (16/11). | Escolha do autor. A autoria e o destino ainda estão em aberto. |
| D19 | Manter os dados e o modelo de pesquisa privados e não publicar o site sem aprovação explícita. | Dado sensível de pessoas reais. O site faz inferência local e não guarda fotos, mas a publicação é uma decisão do autor. |

## O que o projeto não faz (e por quê)

- **Não decide nada sobre ninguém** (cotas, bancas de heteroidentificação, seleção). A acurácia é de cerca de 75% e o rótulo é autodeclaração, que muda para 1 em cada 4 pessoas.
- **Não estima a cor "verdadeira" de uma pessoa.** Esse conceito não existe no sistema brasileiro: a classificação oficial é autodeclarada.
- **Não mede identidade indígena.** A identidade indígena é autodeclaração e reconhecimento da comunidade. O modelo detecta sobretudo adornos e pintura.
