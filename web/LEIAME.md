# Meio-Termo — site estático

Tudo roda no navegador (onnxruntime-web). Não há servidor nem armazenamento de fotos.

    index.html                                  ferramenta (foto → boletim + régua)
    artigo.html                                 o artigo apresentado: números, cinco achados, limites, método
    estilo.css                                  identidade visual compartilhada
    graficos.js                                 régua 0–2, escala por histórico e matriz de confusão (SVG; dados das tabelas do artigo)
    app.js                                      detecção (YuNet) + alinhamento + classificação
    modelo/face_detection_yunet_2023mar.onnx    0,2 MB
    modelo/aluno2_w16.onnx                      8,4 MB (MobileNetV3 destilado, pesos fp16; 5 saídas: branca/parda/preta + outra/indígena)
    modelo/meta.json

Identidade: urna eletrônica e boletim de urna. Tinta preta sobre papel, teclas pretas e uma
cor só: o laranja da tecla CORRIGE, reservado para quem mudou de declaração. Não existe
tecla CONFIRMA, de propósito. Fontes: Archivo (títulos), Source Serif 4 (texto), IBM Plex Mono (boletim).

Os números de graficos.js e artigo.html foram copiados de artigo/tabelas/ e artigo/numeros/.
Se o artigo mudar, atualize os dois.

Testar local:  cd web && python3 -m http.server 8765  ->  http://127.0.0.1:8765
Publicado em https://uruquim.com.br/meio-termo/ (Railway, projeto uruquim). Para atualizar:
  cd ~/Documents/uruquim-site && ./sync.sh && railway up --service site --ci

Referência em Python do mesmo pipeline: ../referencia_web.py (validada contra o recorte do treino).
O antigo exemplos.html (comparação de formatos de resultado) foi movido para docs/rascunhos/.
