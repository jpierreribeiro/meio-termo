# Método

Descrição técnica de tudo o que foi feito, na ordem em que os dados passam. Os números citados aqui vêm de `artigo/numeros/` (gerados por `artigo/analises.py`), salvo quando marcado como **[log Colab]**, que são números de execuções na GPU que não foram refeitas localmente.

## 1. Rótulos: candidaturas do TSE

- **Fonte:** arquivos `consulta_cand_{ano}.zip` do Repositório de Dados Eleitorais do TSE, eleições de 2016, 2018, 2020, 2022 e 2024. Script: `rotulos.py`.
- **Campos usados:** ano, UF, `SQ_CANDIDATO`, nome, data de nascimento, gênero e `DS_COR_RACA`.
- **Uma linha por candidatura:** são 1.569.137 candidaturas de 1.178.425 pessoas.
- **Identificação da pessoa entre eleições:** o CPF não é público. Por isso, a pessoa é identificada pelo SHA-1 de nome normalizado (sem acento, maiúsculo, espaços simples) com a data de nascimento, truncado em 16 caracteres hexadecimais (`pessoa_id`). Homônimos com a mesma data de nascimento se fundem, e quem mudou a grafia do nome se separa. Nenhum dos dois casos foi medido.
- **Histórico:** `racas_historico` traz todas as declarações distintas da pessoa nos cinco anos, e `n_racas` conta quantas são.
- **Divisão treino/validação/teste:** é feita pela pessoa, nunca pela foto. O bucket é `int(pessoa_id[:8], 16) % 100`: abaixo de 80 vai para treino, de 80 a 89 para validação e de 90 em diante para teste. A divisão é determinística, então a mesma pessoa cai sempre no mesmo conjunto, em qualquer ano e em qualquer máquina.

## 2. Fotos

- **Fonte:** arquivos de fotos de candidatos do TSE por UF e ano. Script: `fotos.py`.
- **2016 ficou de fora:** quase todas as fotos de 2016 foram publicadas em escala de cinza. Das 56.960 fotos recortadas, só 48 são coloridas. Os rótulos de 2016 continuam no histórico.
- **Amostragem por classe declarada:** a taxa por pessoa (`TAXAS`) é branca 6%, parda 8%, preta 25%, indígena 100% e amarela 100%. Quem mudou de declaração tem taxa mínima de 30%. O objetivo foi equilibrar as classes sem baixar centenas de milhares de fotos de brancos e pardos.
- **Detecção e alinhamento:** YuNet (OpenCV, versão 2023mar).
  - Regras de aceite: score mínimo de 0,80, distância entre os olhos de pelo menos 12 px na foto original e rejeição de fotos em que um segundo rosto tem mais de 30% da área do principal.
  - Recorte de 160×160 px, rotacionado para nivelar os olhos, com os olhos a 40% da altura e a distância entre eles igual a 28% do lado. A interpolação é cúbica e a borda é replicada.
- **Saída:** 187.481 recortes (`dados/manifesto.csv`). Destes, 117.703 são coloridos, com saturação média no centro do rosto de pelo menos 12 na escala HSV 0–255 do OpenCV. Eles vêm de 93.385 pessoas. A mediana da distância entre os olhos é 32,1 px.
- **Conjunto usado nos modelos:** fotos coloridas cuja declaração e cujo histórico inteiro estão em {branca, parda, preta, indígena}, num total de 113.491 fotos. A cor amarela ficou fora porque tem poucos exemplos e o recorte facial não é o foco do trabalho.

## 3. Rótulos de treino e conjunto "ambíguo"

- **Treino só com declaração consistente:** os modelos treinam apenas com pessoas consistentes (`n_racas == 1`).
- **Conjunto ambíguo:** quem mudou de declaração entre eleições forma o conjunto "ambíguo", usado só para avaliação. A pergunta é se a classe prevista está entre as declarações que a pessoa já usou.

## 4. Modelos

### 4.1 Sonda linear (linha de base)
- **Modelo:** embeddings do CLIP ViT-B/16 congelado (`vit_base_patch16_clip_224.openai`, timm) com regressão logística, `class_weight="balanced"`, C escolhido na validação entre {0,3; 1; 3; 10}.
- **Resultado [log Colab]:** acurácia balanceada de 0,736 no teste e recall de parda de 62,1%.

### 4.2 Professor de 3 classes (branca, parda, preta)
- **Modelo:** ajuste fino do mesmo ViT-B/16 inteiro.
- **Otimização:** AdamW com taxa de 1e-5 no corpo e 1e-3 na cabeça, decaimento de pesos de 0,05, OneCycle e 6 épocas. A perda é entropia cruzada com *label smoothing* de 0,1 e peso por classe igual à raiz da frequência inversa.
- **Aumento de dados:** só RandomResizedCrop (escala de 0,6 a 1,0, razão de 0,9 a 1,1) e espelhamento horizontal. **Não há alteração de cor nem de brilho**, porque a cor é o sinal.
- **Escolha do modelo:** fica o melhor checkpoint na validação. A validação estabilizou em 0,745–0,746 a partir da 3ª época [log Colab].
- **Script:** `treino.py --modo ajuste --classes branca parda preta`.

### 4.3 Professor indígena × outra
- **Modelo:** o mesmo ajuste, em modo binário (`--binario indigena`), com 4 épocas.
- **Desbalanceamento:** no treino, a proporção é de 1 indígena para 27 outras.

### 4.4 Calibração
- **Professor de 3 classes:** temperatura T e viés b por classe, ajustados na validação por L-BFGS, com p = softmax(z/T + b) (`calibrar.py`). Resultado: T = 0,865, b = (0,143; 0,074; −0,218).
- **Professor indígena:** o mesmo procedimento, dentro de `destilar2.py`. Resultado: T = 0,627, b = (1,408; −1,408).
- **Métricas:** NLL, Brier e ECE com 15 faixas. O diagrama de confiabilidade está em `confiabilidade.png`.
- **Proporções da classificação de 3 classes:** as probabilidades valem para as 3 classes em proporção equilibrada, que é a do treino. A variante com as proporções do Censo 2022 (branca 43,5%, parda 45,3%, preta 10,2%; a execução no Colab usou 45,5% por engano de digitação, sem efeito visível nos resultados) multiplica pelas proporções e renormaliza.
- **Proporções da classificação indígena:** ela fica com a prevalência da base, de 3,7% de indígenas no teste.

### 4.5 Aluno (modelo do site)
- **Modelo:** MobileNetV3-Large 1.0 (timm), com entrada de 160 px e 5 saídas: [0:3] branca/parda/preta e [3:5] outra/indígena. A cabeça começa zerada.
- **Destilação:**
  - O alvo de cada foto é a probabilidade calibrada de cada professor, e a perda é a entropia cruzada com alvo suave, somada nas duas saídas.
  - AdamW com taxa de 2e-3 e OneCycle, 12 épocas.
  - Treino com as 90.957 fotos do split de treino. As de indígenas e de quem mudou entre indígena e outra foram sobreamostradas 4 vezes.
- **Exportação:** ONNX opset 17, com os pesos convertidos para fp16 e um nó Cast para fp32. O modelo final tem 8,4 MB. A diferença média para o fp32 é de 0,03 ponto percentual e a máxima de 1,1.
- **Script:** `destilar2.py`.

## 5. Inferência no navegador (site)

- **Pipeline:** `web/app.js`, espelhado em Python por `referencia_web.py`. A sequência é:
  1. YuNet no navegador, com entrada de 640×640 e margem de 25% com borda replicada;
  2. o mesmo alinhamento do treino, com interpolação bilinear;
  3. 6 passagens (escalas 0,95, 1 e 1,05, cada uma com e sem espelhamento), cuja média é o resultado.
- **Por que 6 passagens:** o espelhamento sozinho muda a resposta em até cerca de 10 pontos no p90, e a média dessas variações reduz a instabilidade.
- **Saída exibida:** P(indígena) e, para as outras três categorias, (1 − P(indígena)) × P(classe | 3 classes). As quatro somam 100%.
- **Foto em escala de cinza:** saturação menor que 12 gera um aviso.

## 6. Testes adicionais

- **UTKFace:** fotos de brancos e negros do UTKFace (Zhang; Song; Qi, 2017), recortadas pelo mesmo pipeline. Servem para verificar se o modelo funciona fora do padrão de foto do TSE.
- **Iluminação simulada** (`artigo/iluminacao.py`):
  - 500 fotos de teste consistentes por classe.
  - O ganho é aplicado no espaço linear do sRGB, nestas condições: −1 e −2 stops, +1 stop, luz quente, luz quente e escura e luz fria.
  - Duas correções foram testadas: média cinza e normalização de exposição.
- **Oclusão:** usada só para diagnóstico de casos isolados e não entra no artigo.

## 7. O que foi excluído do artigo e por quê

- **Fotos coletadas fora do TSE** (Instagram, TikTok, Google, Reddit, WhatsApp e afins): foram coletadas sem consentimento. Ficam fora de qualquer número publicado (decisão em `DECISOES.md`).
- **Testes "com adorno / sem adorno" com essas fotos:** saem pelo mesmo motivo.
