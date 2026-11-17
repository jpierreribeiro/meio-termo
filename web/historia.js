// Gráficos da reportagem e a ligação da ferramenta com o modelo (app.js).
// Números: artigo/tabelas/t3_confusao.csv e t4_escala.csv (professor calibrado, conjunto de teste).
import { carregarModelos, lerFoto, analisar } from "./app.js";

const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, txt) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (txt != null) e.textContent = txt;
  return e;
};

// posição média de cada grupo na linha (0 = branca, 2 = preta): P(parda) + 2·P(preta)
const GRUPOS = [
  { nome: "sempre branca", m: 0.322, mudou: false },
  { nome: "branca e parda", m: 0.628, mudou: true },
  { nome: "sempre parda", m: 0.975, mudou: false },
  { nome: "parda e preta", m: 1.449, mudou: true },
  { nome: "sempre preta", m: 1.655, mudou: false },
];

// ---------- aparecer ao rolar ----------
const obs = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) { e.target.classList.add("visivel"); obs.unobserve(e.target); }
}), { threshold: 0.35 });
document.querySelectorAll(".surge").forEach(s => obs.observe(s));

// ---------- 100 candidatos ----------
// 28 de 100 mudaram (27,7%), espalhados para não formar um bloco
const MUDARAM = new Set([2, 6, 9, 13, 17, 21, 24, 28, 33, 36, 41, 44, 47, 52, 55, 59, 63, 66, 71, 74, 78, 81, 85, 88, 92, 95, 97, 99]);
$("cem").innerHTML = Array.from({ length: 100 }, (_, i) => `<span style="--i:${i}"${MUDARAM.has(i) ? ' class="mudou"' : ""}></span>`).join("");

// ---------- a linha de branca a preta ----------
function linha(svg, voce = null) {
  const W = Math.max(300, svg.clientWidth || svg.parentElement.clientWidth), pad = 12;
  const temVoce = voce != null;
  const yVoce = 16, yMud = temVoce ? 66 : 26, yEixo = yMud + 30, yEst = yEixo + 34, yFim = yEst + 34, H = yFim + 6;
  const X = v => pad + (v / 2) * (W - 2 * pad);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H);
  svg.replaceChildren();
  const defs = el("defs"), g = el("linearGradient", { id: "grad-linha" });
  g.append(el("stop", { offset: "0", "stop-color": "var(--c0)" }), el("stop", { offset: ".5", "stop-color": "var(--c1)" }), el("stop", { offset: "1", "stop-color": "var(--c2)" }));
  defs.append(g); svg.append(defs);
  svg.append(el("rect", { x: X(0), y: yEixo - 4, width: X(2) - X(0), height: 8, rx: 4, fill: "url(#grad-linha)" }));
  [["branca", 0, "start"], ["parda", 1, "middle"], ["preta", 2, "end"]].forEach(([t, v, a]) =>
    svg.append(el("text", { x: X(v), y: yFim, "text-anchor": a, class: "forte" }, t)));
  const ancora = x => x < 60 ? "start" : x > W - 60 ? "end" : "middle";
  for (const gr of GRUPOS) {
    const x = X(gr.m), y = gr.mudou ? yMud : yEst, a = ancora(x);
    const cor = gr.mudou ? "var(--destaque)" : "var(--suave)";
    svg.append(el("line", { x1: x, x2: x, y1: gr.mudou ? y + 5 : yEixo, y2: gr.mudou ? yEixo : y - 14, stroke: cor, "stroke-width": 1 }));
    svg.append(el("circle", { cx: x, cy: yEixo, r: 6, fill: gr.mudou ? "var(--destaque)" : "var(--fundo)", stroke: gr.mudou ? "var(--fundo)" : "var(--texto)", "stroke-width": 2 }));
    svg.append(el("text", { x, y, "text-anchor": a, class: gr.mudou ? "azul" : "" }, gr.nome));
  }
  if (temVoce) {
    const x = X(Math.max(0, Math.min(2, voce))), a = ancora(x);
    svg.append(el("line", { x1: x, x2: x, y1: yVoce + 6, y2: yEixo + 10, stroke: "var(--texto)", "stroke-width": 2 }));
    svg.append(el("text", { x, y: yVoce, "text-anchor": a, class: "forte" }, "você"));
  }
}
const svgLinha = $("linha");
linha(svgLinha);
let ultimaNota = null, w0 = svgLinha.clientWidth;
new ResizeObserver(() => { if (Math.abs(svgLinha.clientWidth - w0) > 4) { w0 = svgLinha.clientWidth; linha(svgLinha, ultimaNota); } }).observe(svgLinha);

// ---------- ferramenta ----------
let progressoModelo = 0;
const prontos = carregarModelos(p => { progressoModelo = p; });
prontos.catch(e => console.error(e));

function arredondar(p) {           // inteiros que somam 100 (maiores restos)
  const base = p.map(v => Math.floor(v * 100)), resto = p.map((v, i) => v * 100 - base[i]);
  let falta = 100 - base.reduce((a, b) => a + b, 0);
  resto.map((r, i) => [r, i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (falta-- > 0) base[i]++; });
  return base;
}
function status(txt, progresso) {
  $("status").hidden = !txt; $("status-txt").textContent = txt || "";
  $("prog").hidden = progresso == null;
  if (progresso != null) $("prog").firstElementChild.style.width = Math.round(progresso * 100) + "%";
}
function erro(txt) { $("erro").hidden = !txt; $("erro").textContent = txt || ""; }

const MSG = {
  sem_rosto: "Não encontrei um rosto nessa foto. Tente uma foto de frente, com o rosto inteiro visível e boa luz.",
  varios_rostos: "Encontrei mais de um rosto. Use uma foto com uma pessoa só, ou recorte a foto antes.",
  rosto_pequeno: "O rosto está pequeno demais na foto. Use uma foto mais de perto.",
};

let vez = 0;   // se a pessoa trocar de foto no meio da análise, só a última vale
$("foto").addEventListener("change", async ev => {
  const arq = ev.target.files?.[0];
  ev.target.value = "";
  if (!arq) return;
  const minha = ++vez;
  erro("");
  const previa = $("previa");
  if (previa.src) URL.revokeObjectURL(previa.src);
  previa.src = URL.createObjectURL(arq);
  previa.hidden = false; $("vazio").hidden = true; $("retrato").hidden = true;
  $("botao-txt").textContent = "Trocar foto";
  $("resposta").hidden = true;
  try {
    status("Analisando a foto…");
    const espera = setInterval(() => { if (progressoModelo < 1) status("Preparando o modelo (só na primeira vez)…", progressoModelo); }, 300);
    try { await prontos; } finally { clearInterval(espera); }
    if (minha !== vez) return;
    status("Analisando a foto…");
    const img = await lerFoto(arq);
    const r = await analisar(img, t => { if (minha === vez) status(t); });
    if (minha !== vez) return;
    status("");
    if (r.erro) { erro(MSG[r.erro]); $("espera").hidden = false; return; }
    mostrar(r);
  } catch (e) {
    if (minha !== vez) return;
    status("");
    erro("Algo deu errado: " + (e?.message || e) + ". Tente outra foto ou outro navegador.");
    console.error(e);
  }
});

const COR = ["branca", "parda", "preta", "indígena"];
const PLURAL = ["brancos", "pardos", "pretos", "indígenas"], SING = ["branco", "pardo", "preto", "indígena"];
const gente = (k, i) => `${k} ${k === 1 ? SING[i] : PLURAL[i]}`;
const CLASSE = ["c0", "c1", "c2", "c3"];
const lista = xs => xs.length < 2 ? xs.join("") : xs.slice(0, -1).join(", ") + " e " + xs.at(-1);

function mostrar({ prob, indigena, retrato, saturacao }) {
  const ctx = $("retrato").getContext("2d");
  ctx.clearRect(0, 0, 480, 480); ctx.putImageData(retrato, 0, 0);
  $("retrato").hidden = false; $("previa").hidden = true;

  // junta as duas saídas: indígena primeiro, o resto dividido entre branca/parda/preta
  const n = arredondar([...prob.map(v => v * (1 - indigena)), indigena]);
  const ordem = [0, 1, 2, 3].filter(i => n[i] > 0).sort((a, b) => n[b] - n[a]);
  const [p1, ...resto] = ordem;
  $("frase").innerHTML = `De cada 100 candidatos com rosto parecido com o seu, <b>${n[p1]} se declararam ${PLURAL[p1]}</b>`
    + (resto.length ? `, ${lista(resto.map(i => gente(n[i], i)))}.` : ".");
  $("barra-voce").innerHTML = [0, 1, 2, 3].filter(i => n[i] > 0)
    .map(i => `<div class="${CLASSE[i]}" style="--w:${n[i]}%">${n[i] >= 8 ? n[i] : ""}</div>`).join("");
  $("legenda-voce").innerHTML = [0, 1, 2, 3].filter(i => n[i] > 0)
    .map(i => `<li><i class="${CLASSE[i]}"></i>${COR[i]}</li>`).join("");

  // leitura em palavras
  const top = n[p1];
  let txt = top >= 70 ? `A maioria respondeu igual. Mesmo assim, ${100 - top} de cada 100 escolheram outra cor.`
    : top >= 50 ? "Mais da metade escolheu a mesma cor, mas uma parte grande respondeu diferente."
    : "Ficou dividido: nenhuma cor chegou à metade. É o que acontece com muitos rostos, e é exatamente o meio-termo de que fala esta reportagem.";
  const nota = prob[1] + 2 * prob[2];
  const perto = GRUPOS.reduce((x, g) => Math.abs(g.m - nota) < Math.abs(x.m - nota) ? g : x);
  txt += ` Na linha de branca a preta, lá em cima, o seu rosto ficou mais perto de quem ${perto.mudou ? `alternou entre ${perto.nome}` : perto.nome.replace("sempre ", "sempre se declarou ")}.`;
  $("leitura").textContent = txt;

  const notas = [];
  if (saturacao < 12) notas.push("A foto parece estar em preto e branco. O modelo depende da cor, então este resultado não é confiável.");
  if (indigena >= 0.3) notas.push("A parte indígena é a mais frágil do modelo: ele reconhece só metade dos candidatos indígenas.");
  notas.push("Outra foto sua, com outra luz, pode dar números um pouco diferentes.");
  $("nota").textContent = notas.join(" ");

  $("espera").hidden = true; $("resposta").hidden = false;
  ultimaNota = nota; linha(svgLinha, nota);
  if (matchMedia("(max-width: 700px)").matches) $("resposta").scrollIntoView({ behavior: "smooth", block: "start" });
}
