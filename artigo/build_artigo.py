#!/usr/bin/env python3
"""Gera o artigo em ABNT (NBR 6022:2018) a partir de artigo/artigo.md.

Formatação igual à do artigo de marcadores de IA do autor (pesquisa-uso-de-IA/scripts/build_article.py):
A4, Times New Roman 12, entrelinha 1,5, recuo de 1,25 cm, margens 3/3/2/2 cm, número de página no
canto superior direito, resumo e abstract em espaço simples, títulos de seção em negrito, legenda
acima e fonte abaixo de tabelas e figuras, referências alinhadas à esquerda em espaço simples.

Dialeto do artigo.md
  cabeçalho      linhas "chave: valor" até a primeira linha "---" (titulo, subtitulo, titulo_en,
                 subtitulo_en, autor (repetível, "Nome | nota de rodapé"), palavras_chave, keywords)
  ## RESUMO / ## ABSTRACT     blocos do resumo
  # 1 INTRODUÇÃO              seção primária;  ## 1.1 Título  seção secundária
  [[TABELA n | título | arquivo.csv | larguras em cm separadas por vírgula | nota]]
  [[QUADRO n | ...]]          igual, alinhado à esquerda
  [[FIGURA n | título | arquivo.png | largura em cm | nota]]
  # REFERÊNCIAS               cada parágrafo seguinte é uma referência
  **negrito**  *itálico*  {{red:texto}} em vermelho (pendências do autor)
  linhas começando com "<!--" são comentários e não entram no documento

Uso:  .venv/bin/python artigo/build_artigo.py [--pdf]
"""
import argparse, csv, datetime as dt, re, subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

ART = Path(__file__).resolve().parent
FONT = "Times New Roman"
RED = RGBColor(0xC0, 0x00, 0x00)
TOK = re.compile(r"(\{\{red:.*?\}\}|\*\*.+?\*\*|\*[^*\s][^*]*?\*)")


# ---------------------------------------------------------------- texto
def _run(par, text, size=None, bold=None, italic=None, color=None):
    r = par.add_run(text)
    r.font.name = FONT
    r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    if size: r.font.size = Pt(size)
    if bold: r.font.bold = True
    if italic: r.font.italic = True
    if color is not None: r.font.color.rgb = color
    return r


def add_runs(par, text, size=None, bold=None, italic=None, color=None):
    for part in TOK.split(text):
        if not part: continue
        if part.startswith("{{red:"):
            add_runs(par, part[6:-2], size, bold, italic, RED); continue
        if part.startswith("**"):
            _run(par, part[2:-2], size, True, italic, color)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            _run(par, part[1:-1], size, bold, True, color)
        else:
            _run(par, part, size, bold, italic, color)


def fmt_par(par, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=1.25, spacing=1.5, before=0, after=0, keep=False):
    pf = par.paragraph_format
    pf.alignment = align
    pf.first_line_indent = Cm(indent) if indent else Cm(0)
    pf.line_spacing = spacing
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    if keep: pf.keep_with_next = True


# ---------------------------------------------------------------- notas de rodapé
class Footnotes:
    def __init__(self, doc):
        self.doc, self.n = doc, 0
        self.el = parse_xml(
            f'<w:footnotes {nsdecls("w")}>'
            '<w:footnote w:type="separator" w:id="-1"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:separator/></w:r></w:p></w:footnote>'
            '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:continuationSeparator/></w:r></w:p></w:footnote></w:footnotes>')

    def add(self, par, text):
        self.n += 1
        r = par.add_run()
        r._element.get_or_add_rPr().append(parse_xml(f'<w:vertAlign {nsdecls("w")} w:val="superscript"/>'))
        r._element.append(parse_xml(f'<w:footnoteReference {nsdecls("w")} w:id="{self.n}"/>'))
        runs = ""
        for seg in re.split(r"(\{\{red:.*?\}\})", text):
            if not seg: continue
            red = seg.startswith("{{red:")
            seg = seg[6:-2] if red else seg
            seg = seg.replace("&", "&amp;").replace("<", "&lt;")
            col = '<w:color w:val="C00000"/>' if red else ""
            runs += (f'<w:r><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/>{col}<w:sz w:val="20"/></w:rPr>'
                     f'<w:t xml:space="preserve">{seg}</w:t></w:r>')
        self.el.append(parse_xml(
            f'<w:footnote {nsdecls("w")} w:id="{self.n}"><w:p><w:pPr><w:jc w:val="both"/><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
            f'<w:r><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/><w:vertAlign w:val="superscript"/><w:sz w:val="20"/></w:rPr><w:footnoteRef/></w:r>'
            f'<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:t xml:space="preserve"> </w:t></w:r>{runs}</w:p></w:footnote>'))

    def finalize(self):
        from lxml import etree
        part = Part(PackURI("/word/footnotes.xml"),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml",
                    etree.tostring(self.el, xml_declaration=True, encoding="UTF-8", standalone=True), self.doc.part.package)
        self.doc.part.relate_to(part, RT.FOOTNOTES)


def page_number(section):
    hp = section.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = hp.add_run(); r.font.name = FONT; r.font.size = Pt(10)
    for xml in ('<w:fldChar %s w:fldCharType="begin"/>', '<w:instrText %s xml:space="preserve"> PAGE </w:instrText>',
                '<w:fldChar %s w:fldCharType="end"/>'):
        r._element.append(parse_xml(xml % nsdecls("w")))


# ---------------------------------------------------------------- tabelas e figuras
def set_cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = parse_xml(f'<w:tcBorders {nsdecls("w")}/>'); tcPr.append(b)
    for edge, (val, sz) in kw.items():
        old = b.find(qn(f"w:{edge}"))
        if old is not None: b.remove(old)
        b.append(parse_xml(f'<w:{edge} {nsdecls("w")} w:val="{val}" w:sz="{sz}" w:space="0" w:color="000000"/>'))


def add_table(doc, rows, widths, fs=9.5, quadro=False):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    borda = "single" if quadro else "nil"
    tblPr.append(parse_xml(f'<w:tblBorders {nsdecls("w")}><w:top w:val="nil"/><w:left w:val="{borda}" w:sz="4"/><w:bottom w:val="nil"/>'
                           f'<w:right w:val="{borda}" w:sz="4"/><w:insideH w:val="{borda}" w:sz="4"/><w:insideV w:val="{borda}" w:sz="4"/></w:tblBorders>'))
    tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
    for gc, wd in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(wd * 567)))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.width = Cm(widths[j])
            par = c.paragraphs[0]
            esq = j == 0 or quadro
            fmt_par(par, align=WD_ALIGN_PARAGRAPH.LEFT if esq else WD_ALIGN_PARAGRAPH.CENTER, indent=0, spacing=1.0,
                    keep=i < len(rows) - 1)
            add_runs(par, str(val), size=fs, bold=(i == 0))
            if i == 0: set_cell_border(c, top=("single", 8), bottom=("single", 6))
            if i == len(rows) - 1: set_cell_border(c, bottom=("single", 8))
    return t


def caption(doc, text):
    par = doc.add_paragraph()
    fmt_par(par, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, spacing=1.0, before=6, after=4, keep=True)
    add_runs(par, text, size=10)


def source(doc, note):
    par = doc.add_paragraph()
    fmt_par(par, indent=0, spacing=1.0, before=3, after=0 if note else 10)
    add_runs(par, "Fonte: elaborado pelo autor (2026).", size=10)
    if note:
        par.paragraph_format.keep_with_next = False
        p2 = doc.add_paragraph(); fmt_par(p2, indent=0, spacing=1.0, after=10)
        add_runs(p2, note, size=10)


# ---------------------------------------------------------------- leitura do .md
def ler(md: Path):
    linhas = [l.rstrip("\n") for l in md.read_text(encoding="utf-8").splitlines()]
    meta, autores, i = {}, [], 0
    while linhas[i].strip() != "---":
        if ":" in linhas[i] and not linhas[i].startswith("<!--"):
            k, v = linhas[i].split(":", 1); k, v = k.strip(), v.strip()
            if k == "autor": autores.append(v)
            else: meta[k] = v
        i += 1
    blocos, buf = [], []

    def flush():
        if buf: blocos.append(("p", " ".join(s.strip() for s in buf))); buf.clear()

    for l in linhas[i + 1:]:
        s = l.strip()
        if s.startswith("<!--"): continue
        if not s: flush(); continue
        if s.startswith("[[") and s.endswith("]]"):
            flush(); blocos.append(("obj", [x.strip() for x in s[2:-2].split("|")])); continue
        if s.startswith("## "): flush(); blocos.append(("h2", s[3:])); continue
        if s.startswith("# "): flush(); blocos.append(("h1", s[2:])); continue
        if s.startswith("- "): flush(); blocos.append(("li", s[2:])); continue
        buf.append(s)
    flush()
    return meta, autores, blocos


def construir(md: Path, saida: Path):
    meta, autores, blocos = ler(md)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.left_margin, sec.bottom_margin, sec.right_margin = Cm(3), Cm(3), Cm(2), Cm(2)
    sec.header_distance = Cm(1.5)
    page_number(sec)
    st = doc.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for hs in ("Heading 1", "Heading 2"):
        s = doc.styles[hs]
        s.font.name, s.font.size, s.font.bold, s.font.italic = FONT, Pt(12), True, False
        s.font.color.rgb = RGBColor(0, 0, 0)
        rf = s.element.get_or_add_rPr().get_or_add_rFonts()
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
        rf.set(qn("w:ascii"), FONT); rf.set(qn("w:hAnsi"), FONT)
    fn = Footnotes(doc)
    doc.core_properties.title = f'{meta.get("titulo", "")}: {meta.get("subtitulo", "")}'
    doc.core_properties.author = "; ".join(a.split("|")[0].strip() for a in autores)
    doc.core_properties.created = doc.core_properties.modified = dt.datetime.now()
    doc.core_properties.comments = ""

    # títulos
    par = doc.add_paragraph(); fmt_par(par, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, spacing=1.5)
    add_runs(par, meta["titulo"].upper() + (": " if meta.get("subtitulo") else ""), bold=True)
    if meta.get("subtitulo"): add_runs(par, meta["subtitulo"])
    par = doc.add_paragraph(); fmt_par(par, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, spacing=1.5, before=6)
    add_runs(par, meta["titulo_en"].upper() + (": " if meta.get("subtitulo_en") else ""), bold=True, italic=True)
    if meta.get("subtitulo_en"): add_runs(par, meta["subtitulo_en"], italic=True)
    for k, a in enumerate(autores):
        nome, nota = (a.split("|", 1) + [""])[:2]
        par = doc.add_paragraph(); fmt_par(par, align=WD_ALIGN_PARAGRAPH.RIGHT, indent=0, spacing=1.0, before=12 if k == 0 else 0)
        add_runs(par, nome.strip())
        if nota.strip(): fn.add(par, nota.strip())

    modo = None
    for kind, val in blocos:
        if kind == "h2" and val in ("RESUMO", "ABSTRACT"):
            modo = val
            par = doc.add_paragraph(); fmt_par(par, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, spacing=1.0, before=18, after=6, keep=True)
            add_runs(par, val, bold=True); continue
        if kind == "h1":
            modo = "REF" if val.startswith("REFERÊNCIAS") else None
            par = doc.add_paragraph(style="Heading 1")
            centro = val.startswith("REFERÊNCIAS")
            fmt_par(par, align=WD_ALIGN_PARAGRAPH.CENTER if centro else WD_ALIGN_PARAGRAPH.LEFT, indent=0, spacing=1.5,
                    before=18, after=12 if centro else 6, keep=True)
            add_runs(par, val, bold=True); continue
        if kind == "h2":
            modo = None
            par = doc.add_paragraph(style="Heading 2")
            fmt_par(par, align=WD_ALIGN_PARAGRAPH.LEFT, indent=0, spacing=1.5, before=12, after=6, keep=True)
            add_runs(par, val, bold=True); continue
        if kind == "obj":
            tipo = val[0].split()[0].upper(); n = val[0].split()[1]
            if tipo in ("TABELA", "QUADRO"):
                _, titulo, arq, larg = val[:4]; nota = val[4] if len(val) > 4 else ""
                rows = list(csv.reader((ART / "tabelas" / arq).open(encoding="utf-8")))
                caption(doc, f"{tipo.title()} {n} – {titulo}")
                add_table(doc, rows, [float(x) for x in larg.split(",")], fs=9.5 if tipo == "TABELA" else 10, quadro=tipo == "QUADRO")
                source(doc, nota)
            else:
                _, titulo, arq, larg = val[:4]; nota = val[4] if len(val) > 4 else ""
                caption(doc, f"Figura {n} – {titulo}")
                par = doc.add_paragraph(); fmt_par(par, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, spacing=1.0, keep=True)
                par.add_run().add_picture(str(ART / "figuras" / arq), width=Cm(float(larg)))
                source(doc, nota)
            continue
        if kind == "li":
            par = doc.add_paragraph(); fmt_par(par, indent=0, spacing=1.5)
            par.paragraph_format.left_indent = Cm(1.25)
            add_runs(par, val); continue
        # parágrafo
        par = doc.add_paragraph()
        if modo in ("RESUMO", "ABSTRACT"):
            if val.startswith("**Palavras-chave") or val.startswith("**Keywords"):
                fmt_par(par, indent=0, spacing=1.0, before=6, after=6)
            else:
                fmt_par(par, indent=0, spacing=1.0)
        elif modo == "REF":
            fmt_par(par, align=WD_ALIGN_PARAGRAPH.LEFT, indent=0, spacing=1.0, after=12)
        else:
            fmt_par(par)
        add_runs(par, val)

    fn.finalize()
    doc.save(saida)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pdf", action="store_true"); a = ap.parse_args()
    saida = ART / "artigo_cor_declarada_ABNT.docx"
    construir(ART / "artigo.md", saida)
    print("->", saida)
    if a.pdf:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(ART), str(saida)],
                       check=True, capture_output=True)
        print("->", saida.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
