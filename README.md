# Amplitude de detecção e letalidade notificada da leptospirose, Brasil, 2007–2025

Pacote de reprodutibilidade do manuscrito submetido à *Epidemiologia e
Serviços de Saúde: revista do Sistema Único de Saúde do Brasil* (RESS),
manuscrito RESS-2026-1462: códigos de extração,
decodificação e análise, banco analítico derivado no nível região de saúde por
ano, registro de rastreabilidade de cada valor publicado e o programa que monta
o documento de submissão.

**A pergunta.** A letalidade notificada da leptospirose varia dez vezes entre
territórios brasileiros. Parte dessa variação não é do organismo nem do
cuidado: é de onde, na prática, cada território desenha o limite do caso. Onde
a vigilância só alcança o doente grave, o denominador da letalidade é uma
amostra seletiva da doença que ocorre. O estudo usa a **proporção de casos
confirmados internados** — um campo já coletado em todo o país — como indicador
operacional *inverso* dessa amplitude, e testa se ela prediz a letalidade
notificada depois de removidas as diferenças persistentes entre lugares e a
tendência secular.

**O resultado principal.** Cada 10 pontos percentuais a mais na proporção de
casos internados associaram-se a uma razão de chances de óbito de 1,23
(intervalo de credibilidade de 95% 1,20; 1,26), sobre 4.941 células região-ano
de 426 regiões de saúde e 66.358 casos confirmados. A mortalidade populacional
por A27, que não depende do denominador notificado, não acompanha esse
gradiente.

## O que está aqui, e o que não está

Estão os **códigos** e os **resultados derivados** — tudo o que é preciso para
conferir qualquer número do manuscrito, e para reexecutar a análise a partir
das fontes públicas.

Não estão os **microdados**, e isso é deliberado: eles são públicos, pesados e
mantidos por seus custodiantes, de modo que republicá-los criaria uma cópia que
envelhece em silêncio. O código os busca na origem:

| Fonte | Endereço | Uso |
|---|---|---|
| SINAN-LEPT, SIM-DO, SIH-RD | `ftp.datasus.gov.br/dissemin/publicos` | casos confirmados, óbitos, internações |
| Estimativas populacionais, malha municipal | `servicodados.ibge.gov.br/api/v3/agregados` | denominadores e geografia |

As extrações que sustentam o manuscrito foram feitas em **14 de agosto de
2026**, e os arquivos-fonte são conferidos por resumo criptográfico, de modo
que uma reexecução parte exatamente dos mesmos bytes ou falha dizendo que não
está partindo.

Também não está a geometria dissolvida das regiões de saúde (82 MB):
`paper/R/prep_geo.R` a regenera a partir da malha municipal do IBGE.

```
brepi/                    pacote comum: aquisição, decodificação, denominadores,
                          geografia, agregação de painel, controle de qualidade
studies/leptospirosis/    os programas do estudo, na ordem em que rodam
paper/R/                  as três figuras do artigo e o tema gráfico
paper/artigo.qmd          fonte Quarto do manuscrito, em que a lista
                          STROBE/RECORD localiza cada item
paper/submissao_ress/     o programa que monta o .docx submetido
docs/                     dicionário de variáveis, especificação de métodos,
                          descrição das fontes, arquitetura do pacote
data/results/             os resultados derivados (ver abaixo)
```

## O registro de rastreabilidade

`data/results/reporting/` é o que responde "de onde saiu esse número":

- **`RESULTS_LEDGER.csv`** — uma linha por afirmação quantitativa do
  manuscrito, nomeando a população, a janela, o numerador e o denominador. O
  valor de cada linha é *lido* do arquivo de resultado nomeado em
  `source_object`, nunca redigitado, de modo que o registro não pode divergir
  da análise. Ele existe porque o manuscrito carrega razões parecidas mas não
  idênticas — 6,32, 6,44, 6,45, 6,51 — que um leitor poderia tomar por
  inconsistência; cada uma pertence a uma regra de elegibilidade, uma janela ou
  uma padronização diferente, e a única forma de isso ser conferível em vez de
  afirmado é uma linha por alegação.
- **`REFERENCE_LEDGER.csv`** — cada referência do manuscrito com o que
  exatamente foi extraído dela.
- **`record_strobe_checklist.csv`** — a lista STROBE/RECORD item a item, cada
  um com sua situação e o arquivo que o sustenta.
- **`reproducibility_manifest.json`** — a ordem de execução derivada por
  varredura de dependências entre os programas, com as versões de Python, R e
  bibliotecas efetivamente carregadas.
- **`supplementary_inventory.csv`** — o material suplementar S1–S9.

Os demais diretórios de `data/results/` são as saídas de cada etapa; um
diretório pertence a uma etapa e nada mais escreve nele.

## Reexecutar

Python 3.11 e R 4.4.1. As dependências estão em `pyproject.toml` e `renv.lock`.
A raiz dos dados é `./data`, ou o que estiver em `BREPI_DATA_ROOT`.

Pré-requisitos, que descem às fontes e montam o nível de linha:

```bash
python studies/leptospirosis/01_extract_sinan.py
python studies/leptospirosis/02_denominators.py
python studies/leptospirosis/05_triangulation_extract.py
python studies/leptospirosis/21_build_line_level.py
python studies/leptospirosis/31_build_atlas.py
Rscript paper/R/prep_geo.R
```

Depois, na ordem que o manifesto declara — os programas de uma mesma etapa não
leem nada que os outros escrevem e podem rodar em paralelo:

```bash
# etapa 0
python studies/leptospirosis/60_analysis_panel.py
python studies/leptospirosis/67_rs2024_event.py
# etapa 1
python studies/leptospirosis/61_denominator_coupling.py
python studies/leptospirosis/62_triangulation.py
python studies/leptospirosis/63_missingness_record.py
python studies/leptospirosis/64_exclusion_sensitivity.py
python studies/leptospirosis/65_construct_validity.py
python studies/leptospirosis/66_case_mix.py
Rscript studies/leptospirosis/68_primary_model.R
python studies/leptospirosis/69_within_region.py
python studies/leptospirosis/70_descriptives.py
# etapa 2
python studies/leptospirosis/71_manuscript_tables.py
python studies/leptospirosis/72_reporting_checklist.py
# etapa 3
python studies/leptospirosis/73_ledgers.py
```

As figuras e o documento de submissão:

```bash
Rscript paper/R/fig1_geography.R
Rscript paper/R/fig2_association.R
Rscript paper/R/fig3_rs2024.R
python paper/submissao_ress/montar_docx.py
```

Os demais programas de `studies/leptospirosis/` são as análises exploratórias e
as auditorias que precederam o desenho final. Ficam no repositório porque são o
histórico honesto do estudo, mas nenhum número do manuscrito depende deles: o
que o sustenta é a série a partir de `60_analysis_panel.py`, que é a que o
manifesto ordena e a que a lista de rastreabilidade referencia.

## Uma advertência sobre o símbolo `H`

No código, `H` é o nome da variável que guarda a proporção de casos internados,
e `B = 1 - H` a escala orientada à amplitude. No texto publicado esse símbolo
não existe: a secretaria da revista observou, com razão, que não há sigla
consagrada para essa quantidade, e a variável é nomeada por extenso em todo o
manuscrito, nas tabelas e nas figuras. Os identificadores de dados não foram
renomeados porque renomeá-los invalidaria os arquivos de resultado já gravados.

`H` é um indicador **inverso**: valor alto significa detecção mais *estreita*,
concentrada no caso grave. A primeira versão do estudo chamava `H` de
"profundidade de detecção" e em seguida descrevia um valor alto como *menor*
profundidade, o que se contradiz; a palavra foi abandonada.

## Licença

Código sob licença MIT (`LICENSE`). Os arquivos derivados em `data/results/`
sob CC BY 4.0. Os microdados de origem são públicos e permanecem sob as
condições de seus custodiantes.
