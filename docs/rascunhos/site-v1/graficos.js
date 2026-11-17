// Gráficos do Meio-Termo em SVG puro. Dados: artigo/tabelas/t4_escala.csv e t3_confusao.csv
// (conjunto de teste do professor calibrado). Escala 0–2 = P(parda) + 2·P(preta).

export const GRUPOS = [
  { nome: "Sempre branca", n: 2543, m: 0.322, dp: 0.320, p: [71.9, 24.0, 4.1], mudou: false },
  { nome: "Branca e parda", n: 2496, m: 0.628, dp: 0.382, p: [45.3, 46.6, 8.1], mudou: true },
  { nome: "Sempre parda", n: 2615, m: 0.975, dp: 0.437, p: [24.1, 54.3, 21.6], mudou: false },
  { nome: "Parda e preta", n: 988, m: 1.449, dp: 0.354, p: [6.7, 41.7, 51.6], mudou: true },
  { nome: "Sempre preta", n: 2181, m: 1.655, dp: 0.337, p: [5.3, 24.0, 70.7], mudou: false },
  { nome: "Branca e preta", n: 85, m: 1.178, dp: 0.517, p: [20.0, 42.1, 37.9], mudou: true, raro: true },
  { nome: "Branca, parda e preta", n: 44, m: 1.193, dp: 0.496, p: [18.2, 44.3, 37.5], mudou: true, raro: true },
];

export const MATRIZ = {
  classes: ["Branca", "Parda", "Preta"],
  m: [[2034, 492, 17], [485, 1718, 412], [39, 410, 1732]],
};

const NS = "http://www.w3.org/2000/svg";
const fmt = (v, d = 2) => v.toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtN = v => v.toLocaleString("pt-BR");
const el = (tag, attrs = {}, txt) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (txt != null) e.textContent = txt;
  return e;
};

// ---------- dica flutuante (hover / toque) ----------
let dica;
function mostrarDica(html, ev) {
  if (!dica) { dica = document.createElement("div"); dica.className = "dica-flutuante"; dica.setAttribute("role", "status"); document.body.append(dica); }
  dica.innerHTML = html; dica.hidden = false;
  const x = Math.min(ev.clientX + 14, innerWidth - dica.offsetWidth - 8), y = ev.clientY + 16;
  dica.style.left = Math.max(8, x) + "px";
  dica.style.top = (y + dica.offsetHeight > innerHeight ? ev.clientY - dica.offsetHeight - 12 : y) + "px";
}
function esconderDica() { if (dica) dica.hidden = true; }
const textoGrupo = g => `<b>${g.nome}</b> (${fmtN(g.n)} fotos)<br>Em média, o modelo deu:<br>`
  + `branca ${fmt(g.p[0], 0)}% · parda ${fmt(g.p[1], 0)}% · preta ${fmt(g.p[2], 0)}%`;
function comDica(alvo, html) {
  alvo.addEventListener("pointerenter", ev => mostrarDica(html, ev));
  alvo.addEventListener("pointermove", ev => mostrarDica(html, ev));
  alvo.addEventListener("pointerleave", esconderDica);
}

// ---------- régua: os grupos do artigo numa linha, e "você" se houver ----------
// Linhas, de cima para baixo: você · quem mudou (laranja) · eixo · quem nunca mudou · nomes do eixo
export function regua(caixa, { voce = null } = {}) {
  const desenhar = () => {
    const W = Math.max(280, caixa.clientWidth), pad = 10;
    const temVoce = voce != null;
    const y0 = temVoce ? 30 : 0;
    const X = v => pad + (v / 2) * (W - 2 * pad);
    // rótulos que se encostariam descem (ou sobem) uma linha
    const larg = t => t.length * 6.6 + 8, linhaDe = new Map();
    for (const mudou of [true, false]) {
      const fim = [-1e9, -1e9];
      for (const g of GRUPOS.filter(g => !g.raro && g.mudou === mudou).sort((a, b) => a.m - b.m)) {
        const x = X(g.m), w = larg(g.nome), x0 = Math.max(0, Math.min(W - w, x - w / 2));
        const r = x0 >= fim[0] ? 0 : 1;
        fim[r] = x0 + w; linhaDe.set(g, r);
      }
    }
    const extraMud = [...linhaDe].some(([g, r]) => g.mudou && r) ? 13 : 0;
    const extraEst = [...linhaDe].some(([g, r]) => !g.mudou && r) ? 13 : 0;
    const yMud = y0 + 22 + extraMud, yEixo = yMud + 24, yEst = yEixo + 28, yNome = yEst + extraEst + 26, H = yNome + 8;
    const svg = el("svg", { class: "regua", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
      "aria-label": "Linha de branca a preta com a posição média de cada grupo de candidatos" + (temVoce ? "; com a posição do seu rosto" : "") });
    const defs = el("defs");
    const gr = el("linearGradient", { id: "gradiente-regua" });
    gr.append(el("stop", { offset: "0", "stop-color": "var(--r0)" }), el("stop", { offset: ".5", "stop-color": "var(--r1)" }), el("stop", { offset: "1", "stop-color": "var(--r2)" }));
    defs.append(gr); svg.append(defs);
    svg.append(el("rect", { class: "faixa", x: X(0), y: yEixo - 3, width: X(2) - X(0), height: 6, rx: 3 }));
    for (const v of [0, 1, 2]) svg.append(el("line", { class: "eixo", x1: X(v), x2: X(v), y1: yEixo - 7, y2: yEixo + 7 }));
    [["Branca", 0, "start"], ["Parda", 1, "middle"], ["Preta", 2, "end"]].forEach(([t, v, a]) =>
      svg.append(el("text", { class: "marca-eixo", x: X(v), y: yNome, "text-anchor": a }, t)));

    for (const g of GRUPOS.filter(g => !g.raro)) {
      const x = X(g.m), r = linhaDe.get(g), y = g.mudou ? yMud - 13 * r : yEst + 13 * r;
      const a = x < larg(g.nome) / 2 ? "start" : x > W - larg(g.nome) / 2 ? "end" : "middle";
      const grupo = el("g", { tabindex: 0 });
      grupo.append(el("line", { x1: x, x2: x, y1: g.mudou ? y + 4 : yEixo + 4, y2: g.mudou ? yEixo - 4 : y - 11,
        stroke: g.mudou ? "var(--laranja)" : "var(--tinta-3)", "stroke-width": 1 }));
      grupo.append(el("circle", { class: "ponto" + (g.mudou ? " mudou" : ""), cx: x, cy: yEixo, r: 5, stroke: "var(--recibo)", "stroke-width": 2 }));
      grupo.append(el("text", { class: "grupo" + (g.mudou ? " mudou" : ""), x: a === "start" ? x - 4 : a === "end" ? x + 4 : x, y: y, "text-anchor": a }, g.nome.toLowerCase()));
      grupo.append(el("rect", { x: x - 30, y: Math.min(y, yEixo) - 12, width: 60, height: Math.abs(y - yEixo) + 22, fill: "transparent" }));
      comDica(grupo, textoGrupo(g));
      svg.append(grupo);
    }
    if (temVoce) {
      const x = X(voce), a = x < 70 ? "start" : x > W - 70 ? "end" : "middle";
      svg.append(el("line", { x1: x, x2: x, y1: 16, y2: yEixo + 8, stroke: "var(--tinta)", "stroke-width": 2 }));
      svg.append(el("path", { class: "voce", d: `M${x - 7},${yEixo - 15} L${x + 7},${yEixo - 15} L${x},${yEixo - 5} Z` }));
      svg.append(el("text", { class: "voce-txt", x: a === "start" ? x - 2 : a === "end" ? x + 2 : x, y: 11, "text-anchor": a }, "você"));
    }
    caixa.replaceChildren(svg);
  };
  desenhar();
  if (!caixa._observador) {
    let w = caixa.clientWidth;
    caixa._observador = new ResizeObserver(() => { if (Math.abs(caixa.clientWidth - w) > 4) { w = caixa.clientWidth; desenhar(); } });
    caixa._observador.observe(caixa);
  }
  caixa._redesenhar = desenhar;
}

// ---------- escala por histórico: ponto = média, traço = ±1 DP ----------
export function escala(caixa) {
  const desenhar = () => {
    const W = Math.max(280, caixa.clientWidth), estreito = W < 560;
    const rotW = estreito ? 0 : 170, pad = 12, linha = estreito ? 46 : 34, topo = 8;
    const H = topo + GRUPOS.length * linha + 34;
    const X = v => rotW + pad + (v / 2) * (W - rotW - 2 * pad);
    const svg = el("svg", { class: "regua", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
      "aria-label": "Média e desvio padrão da escala de 0 a 2 para cada histórico de declaração. Valores na tabela abaixo." });
    for (const v of [0, 0.5, 1, 1.5, 2]) {
      svg.append(el("line", { x1: X(v), x2: X(v), y1: topo, y2: H - 26, stroke: "var(--linha)", "stroke-width": v % 1 ? 1 : 1.5, "stroke-dasharray": v % 1 ? "2 4" : "" }));
    }
    [["Branca · 0", 0, "start"], ["Parda · 1", 1, "middle"], ["Preta · 2", 2, "end"]].forEach(([t, v, a]) =>
      svg.append(el("text", { class: "marca-eixo", x: X(v), y: H - 6, "text-anchor": a }, t)));
    GRUPOS.forEach((g, i) => {
      const yc = topo + i * linha + (estreito ? 30 : linha / 2);
      const grupo = el("g", { tabindex: 0 });
      const cor = g.mudou ? "var(--laranja)" : "var(--tinta)";
      if (estreito) grupo.append(el("text", { class: "grupo" + (g.mudou ? " mudou" : ""), x: X(0), y: yc - 12 }, `${g.nome} · ${fmtN(g.n)}`));
      else grupo.append(el("text", { class: "grupo" + (g.mudou ? " mudou" : ""), x: 0, y: yc + 4, style: "font-size:12px" }, `${g.nome} (${fmtN(g.n)})`));
      grupo.append(el("line", { x1: X(Math.max(0, g.m - g.dp)), x2: X(Math.min(2, g.m + g.dp)), y1: yc, y2: yc, stroke: cor, "stroke-width": 2, "stroke-linecap": "round", opacity: g.raro ? .5 : 1 }));
      grupo.append(el("circle", { cx: X(g.m), cy: yc, r: 6, fill: cor, stroke: "var(--recibo)", "stroke-width": 2, opacity: g.raro ? .6 : 1 }));
      grupo.append(el("rect", { x: 0, y: yc - linha / 2, width: W, height: linha, fill: "transparent" }));
      comDica(grupo, textoGrupo(g));
      svg.append(grupo);
    });
    caixa.replaceChildren(svg);
  };
  desenhar();
  let w = caixa.clientWidth;
  new ResizeObserver(() => { if (Math.abs(caixa.clientWidth - w) > 4) { w = caixa.clientWidth; desenhar(); } }).observe(caixa);
}

// ---------- matriz de confusão: intensidade = % da linha ----------
export function matriz(tabela) {
  const { classes, m } = MATRIZ;
  const cab = `<thead><tr><th scope="col">Declarou ↓ · Previsto →</th>${classes.map(c => `<th scope="col">${c}</th>`).join("")}</tr></thead>`;
  const corpo = m.map((linha, i) => {
    const tot = linha.reduce((a, b) => a + b, 0);
    return `<tr><th scope="row">${classes[i]}</th>${linha.map((v, j) => {
      const pct = v / tot * 100, mix = Math.round(8 + pct * 0.85);
      const fundo = `color-mix(in srgb, var(--tinta) ${mix}%, var(--recibo))`;
      const txt = mix > 45 ? "var(--recibo)" : "var(--tinta)";
      return `<td style="background:${fundo};color:${txt}" title="${classes[i]} → ${classes[j]}: ${fmtN(v)} de ${fmtN(tot)}">${fmt(pct, 1)}%<small>${fmtN(v)}</small></td>`;
    }).join("")}</tr>`;
  }).join("");
  tabela.innerHTML = cab + `<tbody>${corpo}</tbody>`;
}
