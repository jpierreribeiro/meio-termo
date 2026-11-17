# Meio-Termo — site estático

Tudo roda no navegador (onnxruntime-web). Não há servidor nem armazenamento de fotos.

    index.html                                  reportagem: contexto, gráficos, ferramenta (foto -> resultado) e limites
    historia.js                                 gráficos da reportagem e ligação da ferramenta com o modelo
    estilo.css                                  visual (Newsreader + Public Sans; cinzas para as categorias, azul para quem mudou)
    app.js                                      detecção (YuNet) + alinhamento + classificação
    artigo.pdf                                  cópia do artigo (sync.sh copia de ../artigo/)
    modelo/face_detection_yunet_2023mar.onnx    0,2 MB
    modelo/aluno2_w16.onnx                      8,4 MB (MobileNetV3 destilado, pesos fp16; 5 saídas: branca/parda/preta + outra/indígena)
    modelo/meta.json

A versão anterior do site (identidade de urna eletrônica, artigo.html, graficos.js) está em ../docs/rascunhos/site-v1/.

Os números de index.html e historia.js foram copiados de artigo/tabelas/ e artigo/numeros/.
Se o artigo mudar, atualize os dois.

Testar local:  cd web && python3 -m http.server 8765  ->  http://127.0.0.1:8765
Publicado em https://uruquim.com.br/meio-termo/ (Railway, projeto uruquim). Para atualizar:
  cd ~/Documents/uruquim-site && ./sync.sh && railway up --service site --ci

Referência em Python do mesmo pipeline: ../referencia_web.py (validada contra o recorte do treino).
O antigo exemplos.html (comparação de formatos de resultado) foi movido para docs/rascunhos/.
