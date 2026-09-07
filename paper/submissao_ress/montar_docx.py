# -*- coding: utf-8 -*-
"""Monta a versao ajustada do manuscrito RESS-2026-1462 (.docx).

O texto reproduz a versao submetida em 20/08/2026, com os ajustes pedidos pela
Secretaria da RESS na primeira verificacao frente as instrucoes aos autores:

 1. O simbolo "H" foi eliminado do texto, das tabelas e das figuras; a variavel
    passa a ser nomeada por extenso ("proporcao de casos internados").
 2. O quadro de aspectos eticos foi movido para logo depois das palavras-chave,
    como no modelo da revista, e recebeu a norma aplicavel.
 3. O termo "estudo ecologico" saiu do titulo e do corpo; o desenho e nomeado
    como analise de series temporais.
 4. Todo intervalo de confianca tem os limites separados por ponto e virgula.
 5. Siglas nao consagradas ou de uso raro foram removidas (SIH, AIH, OR, pp,
    ICr95%); as consagradas sao explicadas na primeira mencao de cada parte
    independente (resumo, texto, cada ilustracao).
 6. A secao de disponibilidade de dados nomeia os enderecos de acesso e a data
    de extracao das bases-fonte.
 7. Titulos de tabelas e figuras foram reescritos para dispensar o texto.
 8. A Tabela 2 foi recomposta com espacamento uniforme.
 9. Resultados e Discussao nao tem mais subtitulos.
10. Nenhuma celula carrega dado de natureza diferente da do cabecalho: a coluna
    livre "Observacao" da Tabela 2 virou nota de rodape.
11. Referencias em Vancouver, com [Internet], data de checagem e link.

Uso:  python montar_docx.py
"""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

AQUI = Path(__file__).resolve().parent
FIG = AQUI.parents[0] / "figures"
SAIDA = AQUI / "RESS-2026-1462_v2_ajustado.docx"

FONTE = "Times New Roman"
CORPO = Pt(12)
TAB = Pt(8.5)
ENTRELINHA = 1.0
ALINHAMENTO = WD_ALIGN_PARAGRAPH.LEFT


# --------------------------------------------------------------------------- #
# utilitarios de formatacao
# --------------------------------------------------------------------------- #
def _runs(par, texto):
    """Escreve `texto` no paragrafo interpretando **negrito** e *italico*."""
    import re
    for pedaco in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", texto):
        if not pedaco:
            continue
        if pedaco.startswith("**") and pedaco.endswith("**"):
            r = par.add_run(pedaco[2:-2]); r.bold = True
        elif pedaco.startswith("*") and pedaco.endswith("*"):
            r = par.add_run(pedaco[1:-1]); r.italic = True
        else:
            par.add_run(pedaco)


def p(doc, texto, alinhamento=ALINHAMENTO, espaco_antes=0,
      espaco_depois=6, entrelinha=ENTRELINHA, tamanho=None, recuo=False,
      manter=False):
    par = doc.add_paragraph()
    par.alignment = alinhamento
    pf = par.paragraph_format
    pf.space_before = Pt(espaco_antes)
    pf.space_after = Pt(espaco_depois)
    pf.line_spacing = entrelinha
    if recuo:
        pf.first_line_indent = Cm(1.25)
    if manter:
        pf.keep_with_next = True
    _runs(par, texto)
    if tamanho:
        for r in par.runs:
            r.font.size = tamanho
    return par


def titulo(doc, texto, nivel=1):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = par.paragraph_format
    pf.space_before = Pt(12 if nivel == 1 else 10)
    pf.space_after = Pt(6)
    pf.line_spacing = ENTRELINHA
    pf.keep_with_next = True
    r = par.add_run(texto)
    r.bold = True
    r.italic = (nivel == 2)
    return par


def _borda_paragrafo(par, lados=("top", "left", "bottom", "right"),
                     cor="000000", tamanho=6):
    pPr = par._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    for lado in lados:
        el = OxmlElement("w:" + lado)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(tamanho))
        el.set(qn("w:space"), "6")
        el.set(qn("w:color"), cor)
        pbdr.append(el)
    pPr.append(pbdr)


def quadro_etica(doc, cabecalho, texto):
    """Quadro de aspectos eticos: cabecalho e corpo dentro de uma moldura.

    Sao dois paragrafos, e nao um so com quebra de linha, porque a linha do
    cabecalho seria esticada pela justificacao ate as duas margens.
    """
    cab = doc.add_paragraph()
    cab.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pfc = cab.paragraph_format
    pfc.space_before = Pt(10)
    pfc.space_after = Pt(2)
    pfc.line_spacing = 1.15
    pfc.keep_with_next = True
    rc = cab.add_run(cabecalho)
    rc.bold = True
    rc.font.size = Pt(11)
    _borda_paragrafo(cab, ("top", "left", "right"))

    corpo = doc.add_paragraph()
    corpo.alignment = ALINHAMENTO
    pf = corpo.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(12)
    pf.line_spacing = 1.15
    _runs(corpo, texto)
    for r in corpo.runs:
        r.font.size = Pt(11)
    _borda_paragrafo(corpo, ("left", "bottom", "right"))
    return corpo


def _celula(cel, texto, negrito=False, alinhar=WD_ALIGN_PARAGRAPH.LEFT):
    cel.text = ""
    par = cel.paragraphs[0]
    par.alignment = alinhar
    pf = par.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(2)
    pf.line_spacing = 1.0
    _runs(par, texto)
    for r in par.runs:
        r.font.size = TAB
        r.font.name = FONTE
        if negrito:
            r.bold = True


def _regra(linha, lado, tamanho=8):
    """Desenha uma regra horizontal (estilo booktabs) em uma linha da tabela."""
    for cel in linha.cells:
        tcPr = cel._tc.get_or_add_tcPr()
        borders = tcPr.find(qn("w:tcBorders"))
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tcPr.append(borders)
        el = OxmlElement("w:" + lado)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(tamanho))
        el.set(qn("w:color"), "000000")
        borders.append(el)


def _inquebravel(texto):
    """Mantem "(1,20; 1,26)" em uma linha so."""
    return texto.replace("; ", ";\u00a0") if "(" in texto else texto


def _margens_estreitas(t, cm=0.08):
    """Reduz o recuo interno das celulas, para os intervalos caberem numa linha."""
    tblPr = t._tbl.tblPr
    marcas = OxmlElement("w:tblCellMar")
    for lado in ("left", "right"):
        el = OxmlElement("w:" + lado)
        el.set(qn("w:w"), str(int(cm * 567)))
        el.set(qn("w:type"), "dxa")
        marcas.append(el)
    tblPr.append(marcas)


def tabela(doc, cabecalhos, linhas, larguras, alinhamentos=None):
    t = doc.add_table(rows=1, cols=len(cabecalhos))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    alinhamentos = alinhamentos or [WD_ALIGN_PARAGRAPH.LEFT] * len(cabecalhos)
    for i, (h, w, a) in enumerate(zip(cabecalhos, larguras, alinhamentos)):
        t.rows[0].cells[i].width = Cm(w)
        _celula(t.rows[0].cells[i], h, negrito=True, alinhar=a)
    for dados in linhas:
        lin = t.add_row()
        for i, (v, w, a) in enumerate(zip(dados, larguras, alinhamentos)):
            lin.cells[i].width = Cm(w)
            _celula(lin.cells[i], _inquebravel(v), alinhar=a)
    for lin in t.rows:
        trPr = lin._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
        if lin is not t.rows[-1]:
            for cel in lin.cells:
                for par in cel.paragraphs:
                    par.paragraph_format.keep_with_next = True
    _margens_estreitas(t)
    _regra(t.rows[0], "top")
    _regra(t.rows[0], "bottom", 6)
    _regra(t.rows[-1], "bottom")
    return t


def nota(doc, texto):
    return p(doc, texto, espaco_antes=2, espaco_depois=12, entrelinha=1.0,
             tamanho=Pt(10))


def figura(doc, caminho, largura_cm=15.5):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(4)
    par.add_run().add_picture(str(caminho), width=Cm(largura_cm))


# --------------------------------------------------------------------------- #
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for lado in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, lado, Cm(2.5))

normal = doc.styles["Normal"]
normal.font.name = FONTE
normal.font.size = CORPO
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONTE)

# --------------------------------------------------------------------- topo --
p(doc, "**Artigo original**", WD_ALIGN_PARAGRAPH.LEFT, espaco_depois=10)
p(doc, "**Amplitude de detecção e letalidade notificada da leptospirose: "
       "análise de séries temporais, Brasil, 2007-2025**",
  WD_ALIGN_PARAGRAPH.LEFT, espaco_depois=14)

# ------------------------------------------------------------------- resumo --
titulo(doc, "Resumo")
p(doc,
  "**Objetivo.** Avaliar se a variação da letalidade notificada da leptospirose "
  "entre territórios e ao longo do tempo se associa a diferenças no espectro "
  "clínico captado pela vigilância, usando a proporção de casos confirmados "
  "internados como indicador operacional inverso dessa amplitude. "
  "**Métodos.** Análise de séries temporais de base populacional, com unidade de "
  "análise a região de saúde por ano, 2007-2025, sobre casos confirmados do "
  "Sistema de Informação de Agravos de Notificação (SINAN). A exposição foi a "
  "proporção de casos confirmados com registro de internação; valores altos "
  "indicam detecção concentrada entre hospitalizados. Ajustaram-se modelo "
  "binomial hierárquico com efeito espacial e tendência temporal e "
  "especificação com efeitos fixos de região e de ano. Compararam-se "
  "notificação, internação hospitalar e mortalidade populacional pela "
  "Classificação Internacional de Doenças, 10ª revisão, código A27. "
  "**Resultados.** Analisaram-se 66.358 casos confirmados. Entre quintis da "
  "proporção de casos internados, a letalidade subiu de 2,7% para 16,9% e a "
  "incidência caiu de 6,53 para 1,23 por 100.000 pessoas-ano; a mortalidade "
  "populacional variou entre 1,68 e 2,47 por milhão, sem gradiente monotônico. "
  "Cada 10 pontos percentuais a mais na proporção de casos internados "
  "associaram-se a razão de chances de óbito de 1,23 (intervalo de confiança de "
  "95% [IC95%] 1,20; 1,26) e, com efeitos fixos, 1,22 (IC95% 1,18; 1,27). "
  "**Conclusão.** A letalidade notificada associou-se fortemente à "
  "composição dos casos captados pela vigilância, enquanto a "
  "mortalidade populacional não apresentou gradiente monotônico com a proporção "
  "de casos internados. Comparações de letalidade devem considerar a composição "
  "dos casos detectados.")
p(doc, "**Palavras-chave:** Leptospirose; Monitoramento Epidemiológico; "
       "Sistemas de Informação em Saúde; Notificação de Doenças; Estudos de "
       "Séries Temporais.", espaco_depois=4)

quadro_etica(
    doc, "Aspectos éticos",
    "O estudo utilizou exclusivamente microdados de acesso público, "
    "disponibilizados pelo Departamento de Informática do Sistema Único de "
    "Saúde sem identificadores nominais. Não houve contato com participantes "
    "nem acesso a identificadores diretos. Por utilizar exclusivamente "
    "informações de domínio público e dados secundários anonimizados, o estudo "
    "está dispensado de apreciação pelo sistema Comitê de Ética em Pesquisa/"
    "Comissão Nacional de Ética em Pesquisa, nos termos do parágrafo único do "
    "artigo 1º da Resolução do Conselho Nacional de Saúde nº 510, de 7 de abril "
    "de 2016.")

# --------------------------------------------------------------- introducao --
titulo(doc, "Introdução")
p(doc,
  "A leptospirose causa cerca de 1,03 milhão de casos e 58.900 óbitos por ano "
  "no mundo, com maior peso em populações urbanas pobres e em trabalhadores "
  "expostos a água contaminada.(1) No Brasil é de notificação compulsória, e o "
  "Sistema de Informação de Agravos de Notificação (SINAN) é a principal fonte "
  "nacional de incidência e de letalidade.")
p(doc,
  "Um caso só entra nesse sistema após atravessar uma sequência de filtros — "
  "infecção, doença sintomática, procura por serviço, suspeita, notificação, "
  "confirmação. Essa sequência tende a selecionar casos clinicamente mais "
  "evidentes e graves, de modo que quadros leves têm menor probabilidade de "
  "alcançar a confirmação e o registro final. Coortes urbanas em Salvador "
  "mostram que a maior parte das infecções não chega ao serviço de saúde,(2,3) "
  "e inquérito sorológico em pacientes com dengue negativa em São Paulo "
  "detectou imunoglobulina M anti-*Leptospira* em 11,3% das amostras contra 64 "
  "casos oficialmente registrados no período, cerca de treze vezes a incidência "
  "notificada.(4) O denominador da letalidade notificada é, portanto, uma "
  "amostra seletiva da doença que ocorre, e a intensidade dessa seleção pode "
  "variar entre territórios.")
p(doc,
  "Um território que investiga sobretudo o caso grave produz, para a mesma "
  "doença, menos casos notificados e maior letalidade do que um território que "
  "também alcança o caso leve. A descrição nacional mais recente, para "
  "2000-2015, reportou 60.952 casos confirmados, incidência de 1,9 por 100.000 "
  "habitantes e letalidade de 10,0% entre 58.986 casos com desfecho conhecido, "
  "e nomeou a subnotificação como limitação não resolvida, sem dispor de "
  "instrumento para medi-la.(5) Uma análise estadual posterior manteve esse "
  "desenho.(6)")
p(doc,
  "Este estudo avalia se a variação da letalidade notificada entre territórios "
  "e ao longo do tempo está associada a diferenças no espectro clínico captado "
  "pela vigilância. A proporção de casos confirmados registrados como "
  "internados é avaliada como indicador operacional inverso dessa amplitude: "
  "quanto maior, mais restrita ao caso grave é a detecção. A questão não é "
  "propor um índice, mas verificar se uma variável já coletada em todo o país "
  "operacionaliza um mecanismo de seleção específico e mensurável.")

# ------------------------------------------------------------------ metodos --
titulo(doc, "Métodos")

titulo(doc, "Delineamento e período", 2)
p(doc,
  "Análise de séries temporais de base populacional, com unidade de análise a "
  "região de saúde por ano, de janeiro de 2007 a dezembro de 2025. O relato "
  "segue a iniciativa Strengthening the Reporting of Observational Studies in "
  "Epidemiology (STROBE) e sua extensão REporting of studies Conducted using "
  "Observational Routinely-collected health Data (RECORD).(7,8) A série inicia "
  "em 2007 porque o SINAN mudou de sistema naquele ano: apenas 46 dos campos do "
  "leiaute anterior sobrevivem no SINAN NET, e as variáveis de classificação, "
  "critério e evolução mudam de nome e de códigos.")

titulo(doc, "Cenário e fontes de dados", 2)
p(doc,
  "Os 5.570 municípios brasileiros foram agregados em 439 regiões de saúde por "
  "uma única configuração contemporânea, aplicada retrospectivamente a "
  "2007-2025; mudanças de limites no período não foram modeladas. Casos: "
  "arquivos anuais nacionais do SINAN. Óbitos populacionais: Sistema de "
  "Informações sobre Mortalidade, causa básica CID-10 A27, por município "
  "de residência e ano do óbito. Internações: Sistema de Informações "
  "Hospitalares, diagnóstico principal A27, contando autorizações de internação "
  "hospitalar, e não pessoas distintas, por município de residência e ano da "
  "internação, sem deduplicação de reinternações ou de autorizações de longa "
  "permanência. Denominadores: estimativas populacionais do Instituto "
  "Brasileiro de Geografia e Estatística compatíveis com a Revisão 2024 das "
  "projeções.(9) As comparações entre os três sistemas usam 2008-2024, janela "
  "comum com anos completos.")

titulo(doc, "Variáveis", 2)
p(doc,
  "A operacionalização utilizou os campos da ficha de notificação e as "
  "definições do guia de vigilância do Ministério da Saúde.(10) O caso "
  "confirmado é o registro com CLASSI_FIN = confirmado, datado pela semana de "
  "início de sintomas (SEM_PRI) e alocado ao município de residência "
  "(ID_MN_RESI); o óbito por leptospirose vem de EVOLUCAO, a internação de "
  "ATE_HOSP, a confirmação laboratorial de CRITERIO e o fenótipo grave de "
  "CLI_ICTERI, CLI_RENAL e CLI_HEMORR.")
p(doc,
  "O campo EVOLUCAO registra o desfecho após investigação (cura, óbito por "
  "leptospirose, óbito por outras causas) além de códigos de informação "
  "ausente; óbito por outras causas é desfecho conhecido que não é óbito por "
  "leptospirose. Cada campo carrega um estado de decodificação — válido, "
  "ignorado, ausente ou inválido — e os quatro não são intercambiáveis. Campo "
  "em branco nunca é lido como resposta negativa: toda proporção é calculada "
  "sobre seu próprio denominador válido, e esses denominadores são reportados.")
p(doc,
  "A exposição do estudo é a proporção de casos confirmados com registro de "
  "internação, calculada em cada região de saúde e em cada ano sobre o "
  "denominador de casos com o campo de internação válido. Valores altos indicam "
  "detecção concentrada entre hospitalizados e, portanto, amplitude de detecção "
  "mais restrita.")
p(doc,
  "A incidência notificada divide os casos confirmados pelas pessoas-ano da "
  "população residente, por 100.000; a letalidade, os óbitos por leptospirose "
  "pelos casos com desfecho registrado; a mortalidade populacional, os óbitos "
  "com causa básica A27 pelas mesmas pessoas-ano, por milhão. Nas estimativas "
  "agregadas somaram-se primeiro os eventos e os denominadores de cada unidade, "
  "calculando taxas e proporções apenas após essa agregação, evitando médias "
  "simples de razões municipais.")

titulo(doc, "Vieses", 2)
p(doc,
  "A proporção de casos internados e a letalidade são calculadas sobre "
  "subconjuntos dos mesmos casos confirmados detectados, cada uma sobre o seu "
  "denominador de campo válido, e ambas são influenciadas pela gravidade "
  "clínica, de modo que parte da associação é esperada por construção; essa "
  "parcela foi quantificada por padronização aritmética, e a mortalidade "
  "populacional foi incluída por não depender do denominador de casos "
  "notificados. A completude do campo de evolução varia entre territórios e o "
  "modelo foi reajustado sob restrição a ela. A unidade de observação é o "
  "território e as associações estimadas são entre agregados: não autorizam "
  "inferência sobre indivíduos.")

titulo(doc, "Tamanho do estudo", 2)
p(doc,
  "Todos os casos confirmados do período; não houve amostragem. O modelo "
  "hierárquico reteve as células com exposição e desfecho estimáveis, sem "
  "aplicar o limiar de 30 casos usado apenas nas descrições por quintil. Como a "
  "proporção de casos internados é estimada com maior incerteza em células "
  "pequenas, realizou-se análise de sensibilidade restrita a células com pelo "
  "menos 10 casos contribuindo para o seu denominador. O efeito do limiar "
  "descritivo é reportado em quatro valores.")

titulo(doc, "Métodos estatísticos", 2)
p(doc,
  "A extração, o processamento e a análise foram implementados em Python e R, "
  "versionados e reexecutáveis a partir das bases públicas. Utilizaram-se "
  "intervalos exatos de Poisson (Garwood) para taxas e de Clopper–Pearson(11) "
  "para proporções, sempre no nível de 95% (IC95%). A associação monotônica "
  "entre variáveis regionais foi resumida pelo coeficiente de correlação de "
  "postos de Spearman (ρ). As regiões de saúde foram ordenadas pela proporção "
  "de casos internados e agrupadas em cinco estratos, cada um contendo "
  "aproximadamente 20% do total de casos confirmados incluídos — e não 20% das "
  "regiões —, de modo que cada estrato sustente estimativa de precisão "
  "semelhante; o número de regiões, casos e pessoas-ano de cada estrato é "
  "reportado.")
p(doc,
  "O modelo principal estima a chance de óbito entre os casos com desfecho "
  "conhecido de cada região de saúde em cada ano, em função da proporção de "
  "casos internados. Como regiões vizinhas tendem a se parecer e o país mudou "
  "ao longo de dezenove anos, o modelo inclui um termo espacial, que deixa "
  "regiões contíguas compartilharem informação, e um termo de tendência anual, "
  "ambos estimados junto com a exposição em modelo hierárquico "
  "bayesiano.(12,13) A exposição entra de forma contínua e todo coeficiente é "
  "reportado como razão de chances por 10 pontos percentuais a mais na "
  "proporção de casos internados.")
p(doc,
  "Ajustou-se uma sequência de modelos para separar o que a exposição explica "
  "do que lugar e tempo já explicavam: A, apenas espaço e tempo; B, acrescenta "
  "a exposição; C, acrescenta a proporção de casos com 60 anos ou mais e a "
  "proporção de casos do sexo masculino; e D, repete C apenas onde o campo de "
  "desfecho está preenchido em ao menos 95% dos casos. Ajustou-se ainda um "
  "modelo binomial com ligação logito incluindo efeitos fixos de região e de "
  "ano e as mesmas duas covariáveis de composição, com erros-padrão robustos "
  "agrupados por região: cada região serve de controle de si mesma, e a "
  "associação vem apenas das mudanças da exposição na própria região ao longo "
  "dos anos. Essa especificação remove a confusão por características regionais "
  "estáveis no tempo, mas não por características que variem dentro da região "
  "ao longo do período.")
p(doc,
  "Para o evento de 2024 no Rio Grande do Sul, a janela do evento é maio a "
  "julho de 2024 e a linha de base é maio a julho de 2019 a 2023, sazonalmente "
  "equivalente — a leptospirose é sazonal, e uma base anual confundiria a "
  "perturbação da vigilância com a composição sazonal. Compararam-se a "
  "proporção de casos internados, a confirmação laboratorial e a letalidade "
  "entre as janelas por razão de chances com intervalo exato. A comparação "
  "entre unidades federativas usa 2024 contra 2019-2023 e restringe-se às sete "
  "com ao menos 150 casos confirmados em 2024. O painel B da Figura 3 agrega "
  "por trimestre civil, para que os denominadores sustentem intervalos "
  "interpretáveis; a janela do evento cai no segundo trimestre de 2024. Os "
  "valores foram confrontados com o relato independente do evento.(14)")

# --------------------------------------------------------------- resultados --
titulo(doc, "Resultados")
p(doc,
  "Entre 2007 e 2025 o SINAN registrou 328.984 notificações, das quais 66.667 "
  "foram confirmadas. Excluíram-se 155 registros com data de início de sintomas "
  "inválida, que não podem ser alocados a nenhum ano, 105 com início fora da "
  "janela e 49 sem município de residência, restando 66.358 casos — a população "
  "analítica. A completude dos campos foi de 96,6% para internação, 98,7% para "
  "critério de confirmação, 92,0% para evolução e 90,1% para o bloco clínico "
  "completo.")
p(doc,
  "Das 8.341 combinações região-ano possíveis, 133 correspondem às 7 regiões de "
  "saúde que não registraram nenhum caso confirmado em todo o período e por "
  "isso não compõem o painel, restando 8.208. Entram nos modelos as células com "
  "pelo menos um caso com desfecho registrado e denominador válido para a "
  "exposição, sem limiar de contagem: descartaram-se 3.242 células sem desfecho "
  "registrado e 25 com exposição indefinida, deixando 4.941 células de 426 "
  "regiões, com 66.082 casos e 5.950 óbitos sobre 60.998 desfechos conhecidos. "
  "Os modelos A, B e C usam exatamente essas mesmas células.")
p(doc,
  "A incidência notificada e a letalidade ordenam-se de modo quase inverso "
  "entre as macrorregiões, e a proporção de casos internados acompanha a "
  "letalidade (Tabela 1). O Sul apresentou a maior incidência com letalidade "
  "relativamente baixa (6,2%); o Nordeste, incidência baixa e a maior letalidade "
  "do país (14,2%); a menor incidência foi observada no Centro-Oeste (0,37 por "
  "100.000 pessoas-ano). O Norte teve a menor proporção de casos internados "
  "(51,3%) e letalidade de 5,5%; o Nordeste, com 87,1%, cerca de três vezes "
  "maior. No mapa por região de saúde (Figura 1) os padrões são geograficamente "
  "distintos: a incidência concentra-se no Sul e na Amazônia ocidental, a "
  "letalidade e a proporção de casos internados no Nordeste e no interior.")
p(doc,
  "Entre as regiões de saúde, a letalidade aumentou com a proporção de casos "
  "internados (Figura 2, painel A). Nos estratos de igual massa de casos, subiu "
  "de 2,7% no quintil de detecção mais ampla para 16,9% no de detecção mais "
  "restrita, razão de 6,32 vezes.")
p(doc,
  "No modelo hierárquico, cada 10 pontos percentuais a mais na proporção de "
  "casos internados associaram-se a razão de chances de óbito de 1,23 (IC95% "
  "1,21; 1,26) após ajuste espacial e temporal. A estimativa manteve-se "
  "praticamente inalterada com a inclusão da composição etária e por sexo — "
  "1,23 (IC95% 1,20; 1,26) — e sob restrição às regiões-ano com completude do "
  "desfecho de ao menos 95% — 1,23 (IC95% 1,20; 1,27) (Tabela 2). A inclusão da "
  "exposição reduziu em 36,2% a variância marginal do componente espacial em "
  "relação ao modelo A, ajustado sobre as mesmas células sem a exposição, de "
  "0,66 para 0,42 na escala do logito.")
p(doc,
  "Na especificação com efeitos fixos de região e de ano, em que as "
  "características territoriais invariantes no tempo são removidas por "
  "construção, a razão de chances foi 1,22 (IC95% 1,18; 1,27), com 306 das 426 "
  "regiões contribuindo. A magnitude foi semelhante à do modelo hierárquico.")
p(doc,
  "A proporção de casos internados e a letalidade são calculadas sobre "
  "subconjuntos dos mesmos casos confirmados detectados, cada uma sobre o seu "
  "próprio denominador de campo válido, e ambas são influenciadas pela "
  "gravidade clínica. Em análise de padronização aritmética, mantendo fixas as "
  "letalidades específicas de internados (13,3%) e de não internados (1,0%) nos "
  "valores nacionais e variando apenas a composição entre eles, o gradiente "
  "esperado entre os quintis extremos é de 2,01 vezes, contra 6,51 vezes "
  "observados. Trata-se de decomposição descritiva, não de fração atribuível "
  "identificada.")
p(doc,
  "Entre os mesmos quintis, a incidência notificada caiu 5,32 vezes e a taxa de "
  "internações hospitalares, 2,63 vezes. A mortalidade por A27 na população "
  "variou entre 1,68 e 2,47 por milhão de pessoas-ano, sem o gradiente "
  "monotônico observado para a letalidade notificada (ρ = 0,03) (Figura 2, "
  "painel C). Nacionalmente, a razão entre as contagens de óbitos do Sistema "
  "de Informações sobre Mortalidade e do SINAN foi 0,99; por estrato, variou entre 0,93 e 1,05. São comparações de "
  "contagens agregadas, sem relacionamento de registros. A taxa de óbitos por "
  "habitante calculada com o numerador do SINAN também não se ordenou com a "
  "exposição (ρ = -0,06).")
p(doc,
  "A incidência de casos não graves foi 13,86 vezes maior no primeiro quintil "
  "que no quinto (IC95% 13,27; 14,49), contra 2,13 vezes para a doença grave "
  "(IC95% 2,05; 2,21) (Figura 2, painel B). O padrão manteve-se nas seis "
  "definições de gravidade avaliadas. A definição primária — icterícia, "
  "insuficiência renal ou hemorragia — foi fixada por critério clínico antes das "
  "comparações e, posteriormente, mostrou o maior gradiente residual entre os "
  "casos graves, sendo a menos favorável à hipótese testada.")
p(doc,
  "O ajuste por idade e sexo atenuou, mas não eliminou, a associação. A "
  "letalidade nacional variou de 3,5% entre 0 e 14 anos a 25,4% acima de 70 "
  "anos, e a proporção de casos com 60 anos ou mais foi 10,8% no quinto quintil "
  "contra 7,6% no primeiro. A padronização direta por idade e sexo moveu a "
  "razão entre os quintis extremos de 6,44 para 5,33, e o gradiente manteve-se "
  "monotônico dentro de cada faixa etária ampla. A associação também permaneceu "
  "em análise restrita aos casos com confirmação laboratorial: a razão entre os "
  "quintis extremos foi 5,49 contra 6,45 no conjunto, e a exposição recalculada "
  "apenas sobre casos laboratoriais preservou a ordenação das regiões "
  "(ρ = 0,97).")
p(doc,
  "A completude do campo de desfecho foi menor onde a proporção de casos "
  "internados é maior: 94,3% no primeiro quintil contra 88,3% no quinto. Em "
  "análise de sensibilidade de pior caso — atribuindo óbito a todos os desfechos "
  "não registrados do primeiro quintil e sobrevivência a todos os do quinto, "
  "cenário deliberadamente implausível construído para limitar o efeito possível "
  "da perda diferencial — a razão entre os quintis extremos foi de 1,86. O "
  "modelo D, restrito às células com completude de ao menos 95%, estimou 1,23 "
  "(IC95% 1,20; 1,27). Aos limiares de 10, 20, 30 e 50 casos a razão entre os "
  "quintis extremos variou entre 6,2 e 6,6; o modelo principal já não aplica "
  "limiar de contagem, e a sensibilidade à precisão da exposição é o modelo F "
  "(Tabela 2). As 229 regiões excluídas ao limiar de 30 concentram 23,2% da "
  "população e 3,6% dos casos.")
p(doc,
  "A enchente de maio de 2024 submeteu um único território a uma mudança "
  "abrupta de esforço de busca (Figura 3). Os casos confirmados com início de "
  "sintomas em maio de 2024 superaram em cinco vezes o máximo mensal de "
  "2019-2023, e a proporção de casos internados caiu de 71,1% na linha de base "
  "para 40,7% em 2024, razão de chances de 0,28 (IC95% 0,24; 0,33). O movimento "
  "é específico: entre as sete unidades federativas com ao menos 150 casos "
  "confirmados em 2024, o Rio Grande do Sul caiu 30,4 pontos percentuais e a "
  "segunda maior queda foi de 4,4. Não se observou tendência descendente "
  "evidente da exposição antes do evento, e em 2025 ela foi de 63,9%. A "
  "extração reproduz o relato independente do evento: 950 casos confirmados "
  "contra 958 e 29 óbitos contra 30 na mesma janela.(14)")
p(doc,
  "Entre os casos com confirmação laboratorial, a proporção de casos internados "
  "também caiu — razão de chances de 0,53 (IC95% 0,31; 0,91) —, enquanto a "
  "letalidade passou de 3,4% para 5,7%, razão de chances de 1,72 (IC95% 0,41; "
  "15,40).")

# ---------------------------------------------------------------- discussao --
titulo(doc, "Discussão")
p(doc,
  "A letalidade notificada da leptospirose no Brasil variou mais de seis vezes "
  "entre territórios, e essa variação está fortemente associada à composição "
  "dos casos que entram no denominador. A comparação entre fontes reforça essa "
  "interpretação: onde a letalidade notificada é mais alta, a incidência "
  "notificada é substancialmente menor e a taxa de internações também, ao passo "
  "que a mortalidade populacional pela mesma causa não apresentou gradiente "
  "monotônico com a proporção de casos internados. A magnitude foi semelhante "
  "na especificação intrarregional, o que indica que a associação não decorre "
  "apenas de diferenças estáveis entre territórios.")
p(doc,
  "O estudo tem limitações. A unidade de observação é o território e nenhuma "
  "associação autoriza inferência sobre indivíduos. Exposição e desfecho são "
  "calculados sobre subconjuntos do mesmo conjunto de casos confirmados "
  "detectados; variando apenas a composição entre internados e não internados, "
  "a padronização aritmética reproduz contraste de 2,01 vezes, frente a 6,51 "
  "observados. A comparação com a mortalidade populacional reduz essa "
  "dependência do denominador notificado, embora o SINAN e o Sistema de "
  "Informações sobre Mortalidade possam "
  "compartilhar erros de reconhecimento diagnóstico. A proporção de casos "
  "internados é estimada a partir de contagens finitas e é imprecisa em células "
  "pequenas, motivo da análise de sensibilidade com ao menos 10 casos em seu "
  "denominador. Os efeitos fixos removem características regionais estáveis, "
  "não fatores que variam no tempo, como acesso, política de internação, "
  "prática diagnóstica, gravidade circulante ou contexto de surto. A completude "
  "do desfecho varia com a exposição, embora o resultado persista na análise de "
  "pior caso. O fenótipo grave deriva da notificação, não de prontuário. A "
  "proporção de casos internados também depende do limiar de internação e da "
  "oferta de leitos. O episódio do Rio Grande do Sul é validação de apoio, não "
  "experimento natural, pois a enchente alterou simultaneamente exposição, "
  "gravidade, acesso, capacidade laboratorial e tempo até o tratamento, além de "
  "interromper parte da notificação.(14)")
p(doc,
  "O achado converge com desenhos que alcançam denominadores mais amplos que o "
  "de casos notificados — o de infecção e o de doença sintomática, distintos "
  "entre si: coortes urbanas em Salvador mostram que a maior parte das "
  "infecções não chega ao serviço de saúde,(2,3) e o inquérito sorológico de "
  "São Paulo estima cerca de treze vezes a incidência notificada.(4) Nenhum é "
  "replicável nacionalmente, ao passo que o campo de internação existe para "
  "todos os municípios desde 2007.")
p(doc,
  "A comparação mais direta é com a descrição nacional de 2000-2015, que "
  "reportou letalidade de 10,0% e incidência de 1,9 por 100.000 e nomeou a "
  "subnotificação como limitação não resolvida, dispondo apenas do SINAN.(5) A "
  "ordenação regional é a mesma descrita há uma década: o contraste persiste, e "
  "o que se altera é sua interpretação. O mesmo vale para as análises "
  "estaduais, que descrevem incidência e mortalidade sem relacioná-las ao "
  "alcance da detecção.(6)")
p(doc,
  "Uma letalidade elevada não deve ser lida isoladamente como falha "
  "assistencial, nem uma letalidade baixa como sucesso: em territórios de "
  "detecção restrita a primeira é esperada, e em territórios de busca intensa a "
  "segunda também. Séries temporais de letalidade são igualmente afetadas, já "
  "que mudanças no esforço de busca deslocam a composição dos casos captados. O "
  "episódio do Rio Grande do Sul mostra que mudanças abruptas na vigilância "
  "podem coincidir com grandes mudanças na composição dos casos detectados, "
  "tornando a letalidade notificada difícil de interpretar isoladamente. "
  "Recomenda-se que comparações de letalidade entre unidades federativas, entre "
  "regiões de saúde ou ao longo do tempo sejam apresentadas com a proporção de "
  "casos internados ao lado, como indicador de comparabilidade — informação já "
  "coletada e disponível nos sistemas nacionais.")
p(doc,
  "A proporção de casos confirmados internados pode assim servir como indicador "
  "operacional de comparabilidade, sem ser interpretada como medida direta de "
  "desempenho da vigilância: gravidade clínica, acesso e prática de internação "
  "também a determinam.")

# ------------------------------------------------- declaracoes obrigatorias --
titulo(doc, "Conflitos de interesse")
p(doc, "Os autores declaram não possuir conflitos de interesse.")

titulo(doc, "Disponibilidade de dados")
p(doc,
  "Os dados-fonte são públicos, de livre acesso e não exigem autorização. Os "
  "arquivos anuais do SINAN, do Sistema de Informações sobre Mortalidade e do "
  "Sistema de Informações Hospitalares foram obtidos no serviço de "
  "transferência de arquivos do Departamento de Informática do Sistema Único de "
  "Saúde (ftp.datasus.gov.br/dissemin/publicos), e as estimativas populacionais "
  "na interface pública de dados agregados do Instituto Brasileiro de Geografia "
  "e Estatística (https://servicodados.ibge.gov.br/api/v3/agregados). Todas as "
  "extrações foram realizadas em 14 de agosto de 2026 e os arquivos-fonte estão "
  "conferidos por resumo criptográfico, de modo que a reexecução parte "
  "exatamente dos mesmos dados.")
p(doc,
  "O pacote de reprodutibilidade está publicamente disponível em "
  "https://github.com/LeviMelo/leptospirose-brasil-2007-2025 e reúne os "
  "códigos de extração, decodificação e análise, o banco analítico derivado no "
  "nível região-ano, o dicionário de variáveis, a lista de verificação "
  "STROBE/RECORD, o manifesto de reprodutibilidade — que declara a ordem de "
  "execução dos programas e as versões de Python, R e das bibliotecas "
  "utilizadas — e o registro de rastreabilidade de cada valor apresentado no "
  "texto, nas tabelas e nas figuras, em que toda estimativa é lida do arquivo "
  "de resultado que a produziu. O pacote não contém dados identificáveis. A "
  "versão correspondente a esta submissão está identificada pela etiqueta "
  "submissao-ress no mesmo endereço.(15)", alinhamento=WD_ALIGN_PARAGRAPH.LEFT)

titulo(doc, "Uso de inteligência artificial generativa")
p(doc,
  "Utilizou-se assistente de inteligência artificial generativa (Claude, "
  "Anthropic) no desenvolvimento e na revisão do código de extração, "
  "processamento e análise e na redação e revisão do manuscrito. Todas as "
  "estimativas foram geradas por código versionado e reexecutável a partir das "
  "bases públicas, e os valores do texto são lidos dos arquivos de resultado no "
  "momento da compilação. Os autores revisaram integralmente os procedimentos, "
  "os resultados e o texto e assumem responsabilidade pelo conteúdo.")

# --------------------------------------------------------------- referencias --
titulo(doc, "Referências")
REFS = [
 "Costa F, Hagan JE, Calcagno J, Kane M, Torgerson P, Martinez-Silveira MS, et "
 "al. Global morbidity and mortality of leptospirosis: a systematic review. "
 "PLoS Negl Trop Dis [Internet]. 2015 [citado em 7 set. 2026];9(9):e0003898. "
 "Disponível em: https://doi.org/10.1371/journal.pntd.0003898",

 "Hacker KP, Sacramento GA, Cruz JS, de Oliveira D, Nery N Jr, Lindow JC, et "
 "al. Influence of rainfall on *Leptospira* infection and disease in a tropical "
 "urban setting, Brazil. Emerg Infect Dis [Internet]. 2020 [citado em 7 set. "
 "2026];26(2):311-4. Disponível em: https://doi.org/10.3201/eid2602.190102",

 "Hagan JE, Moraga P, Costa F, Capian N, Ribeiro GS, Wunder EA Jr, et al. "
 "Spatiotemporal determinants of urban leptospirosis transmission: four-year "
 "prospective cohort study of slum residents in Brazil. PLoS Negl Trop Dis "
 "[Internet]. 2016 [citado em 7 set. 2026];10(1):e0004275. Disponível em: "
 "https://doi.org/10.1371/journal.pntd.0004275",

 "Esteves SB, Oliveira LM, Guilloux AGA, Cortez A, Masi E, Ferreira IMR, et al. "
 "Into the spotlight: a spatial study of potentially underreported leptospirosis "
 "among dengue-negative patients in São Paulo city, Brazil. PLoS Negl Trop Dis "
 "[Internet]. 2025 [citado em 7 set. 2026];19(3):e0012888. Disponível em: "
 "https://doi.org/10.1371/journal.pntd.0012888",

 "Galan DI, Roess AA, Pereira SVC, Schneider MC. Epidemiology of human "
 "leptospirosis in urban and rural areas of Brazil, 2000-2015. PLoS One "
 "[Internet]. 2021 [citado em 7 set. 2026];16(3):e0247763. Disponível em: "
 "https://doi.org/10.1371/journal.pone.0247763",

 "Camelo IM, Cabral BVB, Sousa DG, Sousa GJB, Pereira MLD. Leptospirosis "
 "incidence and mortality: a spatio-temporal analysis, Ceará, 2007-2023. "
 "Epidemiol Serv Saude [Internet]. 2026 [citado em 7 set. 2026];35:e20250108. "
 "Disponível em: https://doi.org/10.1590/S2237-96222026v35e20250108.en",

 "von Elm E, Altman DG, Egger M, Pocock SJ, Gøtzsche PC, Vandenbroucke JP. The "
 "Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) "
 "statement. Lancet [Internet]. 2007 [citado em 7 set. "
 "2026];370(9596):1453-7. Disponível em: "
 "https://doi.org/10.1016/S0140-6736(07)61602-X",

 "Benchimol EI, Smeeth L, Guttmann A, Harron K, Moher D, Petersen I, et al. The "
 "REporting of studies Conducted using Observational Routinely-collected health "
 "Data (RECORD) statement. PLoS Med [Internet]. 2015 [citado em 7 set. "
 "2026];12(10):e1001885. Disponível em: "
 "https://doi.org/10.1371/journal.pmed.1001885",

 "Instituto Brasileiro de Geografia e Estatística. Projeções da população: "
 "Brasil e unidades da federação: revisão 2024 [Internet]. Rio de Janeiro: "
 "IBGE; 2024 [citado em 7 set. 2026]. Disponível em: "
 "https://ftp.ibge.gov.br/Projecao_da_Populacao/Projecao_da_Populacao_2024/",

 "Brasil. Ministério da Saúde. Secretaria de Vigilância em Saúde e Ambiente. "
 "Guia de vigilância em saúde: volume 3 [Internet]. 6ª ed. rev. Brasília: "
 "Ministério da Saúde; 2024 [citado em 7 set. 2026]. Leptospirose; p. "
 "1053-76. Disponível em: "
 "https://www.gov.br/saude/pt-br/centrais-de-conteudo/publicacoes/svsa/"
 "vigilancia/guia-de-vigilancia-em-saude-volume-3-6a-edicao",

 "Clopper CJ, Pearson ES. The use of confidence or fiducial limits illustrated "
 "in the case of the binomial. Biometrika [Internet]. 1934 [citado em 7 set. "
 "2026];26(4):404-13. Disponível em: https://doi.org/10.1093/biomet/26.4.404",

 "Riebler A, Sørbye SH, Simpson D, Rue H. An intuitive Bayesian spatial model "
 "for disease mapping that accounts for scaling. Stat Methods Med Res "
 "[Internet]. 2016 [citado em 7 set. 2026];25(4):1145-65. Disponível em: "
 "https://doi.org/10.1177/0962280216660421",

 "Rue H, Martino S, Chopin N. Approximate Bayesian inference for latent "
 "Gaussian models by using integrated nested Laplace approximations. J R Stat "
 "Soc Series B Stat Methodol [Internet]. 2009 [citado em 7 set. "
 "2026];71(2):319-92. Disponível em: "
 "https://doi.org/10.1111/j.1467-9868.2008.00700.x",

 "Ranieri TM, Viegas da Silva E, Vallandro MJ, Oliveira MM, Barcellos RB, "
 "Lenhardt RV, et al. Leptospirosis cases during the 2024 catastrophic flood in "
 "Rio Grande do Sul, Brazil. Pathogens [Internet]. 2025 [citado em 7 set. "
 "2026];14(4):393. Disponível em: https://doi.org/10.3390/pathogens14040393",

 "Amorim LM. Amplitude de detecção e letalidade notificada da leptospirose, "
 "Brasil, 2007-2025: pacote de reprodutibilidade [conjunto de dados e "
 "códigos]. 2026 [citado em 7 set. 2026]. GitHub. Disponível em: "
 "https://github.com/LeviMelo/leptospirose-brasil-2007-2025",
]
for i, ref in enumerate(REFS, 1):
    par = p(doc, "%d. %s" % (i, ref), WD_ALIGN_PARAGRAPH.LEFT,
            espaco_depois=4, entrelinha=ENTRELINHA, tamanho=Pt(11))
    par.paragraph_format.left_indent = Cm(0.75)
    par.paragraph_format.first_line_indent = Cm(-0.75)

# ------------------------------------------------------- figuras e tabelas --
doc.add_page_break()

p(doc, "**Figura 1.** Incidência notificada por 100.000 pessoas-ano (A), "
       "letalidade (B) e proporção de casos confirmados internados (C), por "
       "região de saúde, com ampliação da faixa Sul–Sudeste (D, E e F). Brasil, "
       "2007-2025 (n = 66.358 casos confirmados; 311 regiões de saúde com "
       "estimativa)", espaco_depois=4, manter=True)
figura(doc, FIG / "figura1_geografia.png")
nota(doc, "Notas: escalas truncadas no percentil 98 das regiões estimáveis, com "
          "o limite superior marcado “+”; regiões com menos de 10 casos "
          "confirmados aparecem em cinza, sem estimativa; as molduras vermelhas "
          "marcam a janela ampliada, escolhida por concentração de casos.")

p(doc, "**Tabela 1.** Casos confirmados, incidência notificada, letalidade, "
       "proporção de casos internados, confirmação laboratorial e completude do "
       "campo de desfecho, com intervalos de confiança de 95% (IC95%), por "
       "macrorregião. Brasil, 2007-2025 (n = 66.358 casos confirmados)",
  espaco_antes=10, espaco_depois=6, manter=True)
E = WD_ALIGN_PARAGRAPH.LEFT
D = WD_ALIGN_PARAGRAPH.RIGHT
tabela(
    doc,
    ["Macrorregião", "Casos\nn (%)",
     "Incidência por 100.000 pessoas-ano (IC95%)",
     "Letalidade % (IC95%)", "Casos internados % (IC95%)",
     "Confirmação laboratorial (%)", "Completude do desfecho (%)"],
    [["Sul", "22.179 (33,4)", "3,98 (3,93; 4,04)", "6,2 (5,8; 6,5)",
      "69,4 (68,8; 70,0)", "90,9", "94,1"],
     ["Norte", "10.216 (15,4)", "3,11 (3,05; 3,17)", "5,5 (5,0; 5,9)",
      "51,3 (50,3; 52,3)", "83,9", "93,8"],
     ["Sudeste", "21.643 (32,6)", "1,34 (1,32; 1,35)", "13,3 (12,8; 13,8)",
      "76,7 (76,1; 77,3)", "87,8", "89,9"],
     ["Nordeste", "11.212 (16,9)", "1,07 (1,05; 1,09)", "14,2 (13,6; 14,9)",
      "87,1 (86,4; 87,7)", "72,4", "90,8"],
     ["Centro-Oeste", "1.108 (1,7)", "0,37 (0,35; 0,40)", "11,1 (9,2; 13,3)",
      "85,8 (83,5; 87,8)", "92,5", "84,6"],
     ["Brasil", "66.358 (100,0)", "1,73 (1,71; 1,74)", "9,8 (9,5; 10,0)",
      "72,2 (71,9; 72,6)", "85,7", "92,0"]],
    [2.0, 2.0, 3.0, 2.6, 2.9, 1.7, 1.8],
    [E, D, D, D, D, D, D])
nota(doc, "Notas: casos confirmados do Sistema de Informação de Agravos de "
          "Notificação com data de início de sintomas entre 2007 e 2025; a "
          "letalidade tem por denominador os casos com desfecho registrado; a "
          "proporção de casos internados, os casos com o campo de internação "
          "válido; intervalos exatos de Poisson para a incidência e de "
          "Clopper–Pearson para as proporções.")

p(doc, "**Figura 2.** Associação entre a proporção de casos confirmados "
       "internados e a letalidade notificada, por região de saúde (A); "
       "incidência de casos graves e não graves por quintil da proporção de "
       "casos internados (B); e comparação entre notificação, internação "
       "hospitalar e mortalidade populacional, indexada ao primeiro quintil (C). "
       "Brasil, 2007-2025 (A: n = 311 regiões de saúde; B e C: n = 165 regiões, "
       "56.332 casos, 2008-2024)", espaco_antes=10, espaco_depois=6, manter=True)
figura(doc, FIG / "figura2_associacao.png")
nota(doc, "Notas: em (A) cada ponto é uma região de saúde com ao menos 10 casos "
          "com campo de internação válido, com área proporcional ao número de "
          "casos, e a linha é um ajuste binomial com faixa de 95%; em (B) a "
          "escala vertical é logarítmica e os intervalos são exatos de Poisson; "
          "em (C) cada série é indexada ao seu próprio primeiro quintil "
          "(I = 100) porque as três têm unidades diferentes — notificação e "
          "internação por 100.000 e mortalidade por 1.000.000 de pessoas-ano. A "
          "janela comum é 2008-2024.")

p(doc, "**Figura 3.** Casos confirmados por mês de início de sintomas (A); "
       "proporção de casos internados e proporção de confirmação laboratorial, "
       "com intervalos de confiança de 95% (IC95%) (B); e variação da proporção "
       "de casos internados entre 2024 e a linha de base 2019-2023, por unidade "
       "federativa (C). Rio Grande do Sul e Brasil, 2019-2025 (A e B: n = 3.527 "
       "casos confirmados no Rio Grande do Sul; C: n = 7 unidades federativas "
       "com ao menos 150 casos confirmados em 2024)",
  espaco_antes=10, espaco_depois=6, manter=True)
figura(doc, FIG / "figura3_rs2024.png")
nota(doc, "Notas: a faixa sombreada em (A) e (B) marca maio a julho de 2024, "
          "janela da enchente; as bandas em (B) são intervalos de "
          "Clopper–Pearson; em (C) incluem-se as unidades federativas com ao "
          "menos 150 casos confirmados em 2024.")

p(doc, "**Tabela 2.** Razão de chances de óbito por leptospirose entre casos "
       "confirmados com desfecho conhecido, por 10 pontos percentuais a mais na "
       "proporção de casos internados, com intervalos de 95%, segundo "
       "especificação do modelo. Brasil, 2007-2025 (n = 4.941 células "
       "região-ano; 426 regiões de saúde)", espaco_antes=10, espaco_depois=6, manter=True)
tabela(
    doc,
    ["Modelo", "Especificação", "Razão de chances (intervalo de 95%)",
     "Células (n)"],
    [["A", "Componente espacial e tendência anual, sem a exposição", "–",
      "4.941"],
     ["B", "A, mais a proporção de casos internados (contínua)",
      "1,23 (1,21; 1,26)", "4.941"],
     ["C", "B, mais composição etária e sexo (modelo principal)",
      "1,23 (1,20; 1,26)", "4.941"],
     ["D", "C, apenas onde o desfecho está preenchido em ao menos 95% dos casos",
      "1,23 (1,20; 1,27)", "3.565"],
     ["E", "C, com desfecho redefinido como letalidade apenas entre internados",
      "1,04 (1,02; 1,07)", "4.624"],
     ["F", "C, apenas em células com ao menos 10 casos no denominador da "
           "exposição", "1,24 (1,20; 1,28)", "1.314"],
     ["G", "Efeitos fixos de região e de ano, mais composição, com "
           "erros-padrão robustos agrupados por região", "1,22 (1,18; 1,27)",
      "4.289"]],
    [1.8, 7.4, 4.0, 2.3],
    [E, E, D, D])
nota(doc, "Notas: célula = combinação de região de saúde e ano; nos modelos A a "
          "F o intervalo de 95% é de credibilidade e no modelo G é intervalo de "
          "confiança de 95% (IC95%), com erros-padrão robustos agrupados por "
          "região; a variância marginal do componente espacial foi 0,66 (0,49; "
          "0,88) no modelo A e 0,42 (0,30; 0,58) no modelo B, redução de 36,2%; "
          "no modelo C, a razão de chances por 10 pontos percentuais a mais de "
          "casos com 60 anos ou mais foi 1,08 (1,05; 1,11); o modelo D retém "
          "72,2% das células, o modelo F retém 83,0% dos casos e o modelo G "
          "reúne 306 das 426 regiões e 97,9% dos casos; os modelos A, B e C são "
          "ajustados sobre as mesmas células e são comparáveis entre si, "
          "enquanto D, E, F e G usam subconjuntos ou desfecho distinto e não o "
          "são; a unidade de observação é o território, de modo que toda "
          "estimativa é uma associação entre a composição dos casos detectados "
          "de um território e sua letalidade notificada, e não autoriza leitura "
          "individual.")

SAIDA.parent.mkdir(parents=True, exist_ok=True)
doc.save(SAIDA)
print("gravado:", SAIDA)
