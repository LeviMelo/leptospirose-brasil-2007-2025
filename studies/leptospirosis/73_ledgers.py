"""Build the RESULTS_LEDGER and REFERENCE_LEDGER required by FEEDBACK_2 §82.

The results ledger exists because the manuscript now carries several similar but
non-identical ratios — 6.32, 6.44, 6.45, 6.51 — that a reader could reasonably
take for an inconsistency. They are not: each belongs to a different eligibility
rule, calendar window or standardisation. One row per claim, naming the
population, the window and the numerator/denominator, is what makes that
checkable rather than asserted.

Every row's `value` is read from the result file named in `source_object`, not
retyped, so the ledger cannot drift from the analysis the way a hand-maintained
table would.

Outputs to ``data/results/reporting/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.config import PATHS

OUT = PATHS.results / "reporting"
R = PATHS.results


def j(rel: str) -> dict:
    return json.loads((R / rel).read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flow = pl.read_csv(R / "analysis_panel/study_flow.csv")
    pan, mod = j("analysis_panel/panel_report.json"), j("primary_model/primary_model_report.json")
    tri, trep = pl.read_csv(R / "triangulation/three_systems_by_stratum.csv"), j("triangulation/triangulation_report.json")
    cpl, cvr = j("coupling/coupling_report.json"), j("construct_validity/construct_validity_report.json")
    cmx, exc = j("case_mix/case_mix_report.json"), j("exclusion_sensitivity/exclusion_sensitivity_report.json")
    desc = pl.read_csv(R / "descriptives/table1_by_region.csv")
    sev = cvr["strand2_severity_phenotype"]["severity_gradients"][
        "all confirmed cases | jaundice OR renal OR haemorrhage (primary)"]
    cfrd = cvr["strand1_confirmation_pathway"]["cfr_gradient_by_case_definition"]
    fl = lambda k: int(flow.filter(pl.col("step").str.strip_chars() == k)["n"][0])
    BR = desc.filter(pl.col("stratum") == "Brasil")

    W = "2007-2025"
    WT = "2008-2024 (janela comum aos tres sistemas)"
    HR = "regiao de saude"
    rows = [
        ("Resumo/Resultados", "Casos confirmados analisados", fl("ANALYTIC POPULATION"),
         "analysis_panel/study_flow.csv", "casos confirmados", "notificacoes SINAN", W, "nacional",
         "CLASSI_FIN=confirmado; inicio de sintomas na janela; municipio de residencia valido", "observado"),
        ("Resultados", "Incidencia nacional /100 mil pessoas-ano", float(BR["incidence_per_100k"][0]),
         "descriptives/table1_by_region.csv", "casos confirmados", "pessoas-ano IBGE", W, "nacional",
         "grade completa regiao-ano", "observado"),
        ("Resultados", "Letalidade nacional", float(BR["cfr"][0]),
         "descriptives/table1_by_region.csv", "obitos por leptospirose", "casos com EVOLUCAO valido", W, "nacional",
         "denominador de campo valido", "observado"),
        ("Resultados", "H nacional", float(BR["H"][0]),
         "descriptives/table1_by_region.csv", "casos com ATE_HOSP=sim", "casos com ATE_HOSP valido", W, "nacional",
         "denominador de campo valido", "observado"),
        ("Resultados/Fig2C", "Letalidade Q1 (quintis de H)", float(tri["cfr"][0]),
         "triangulation/three_systems_by_stratum.csv", "obitos", "desfechos conhecidos", WT, f"{HR}, quintil de H",
         ">=30 casos com ATE_HOSP valido; quintis de igual massa de casos", "observado"),
        ("Resultados/Fig2C", "Letalidade Q5 (quintis de H)", float(tri["cfr"][4]),
         "triangulation/three_systems_by_stratum.csv", "obitos", "desfechos conhecidos", WT, f"{HR}, quintil de H",
         ">=30 casos com ATE_HOSP valido", "observado"),
        ("Resultados", "Razao de letalidade Q5/Q1", trep["cfr_gradient_Q5_over_Q1"],
         "triangulation/triangulation_report.json", "letalidade Q5", "letalidade Q1", WT, f"{HR}, quintil de H",
         ">=30 casos com ATE_HOSP valido", "observado"),
        ("Resultados", "Razao de letalidade Q5/Q1 (janela 2007-2025, >=30 casos)", cfrd["all confirmed"]["Q5_over_Q1"],
         "construct_validity/construct_validity_report.json", "letalidade Q5", "letalidade Q1", W, f"{HR}, quintil de H",
         ">=30 casos confirmados com ATE_HOSP valido", "observado"),
        ("Resultados", "Razao de letalidade Q5/Q1, so confirmacao laboratorial",
         cfrd["laboratory-confirmed only"]["Q5_over_Q1"],
         "construct_validity/construct_validity_report.json", "letalidade Q5", "letalidade Q1", W, f"{HR}, quintil de H",
         "restrito a CRITERIO=clinico_laboratorial", "sensibilidade"),
        ("Resultados", "Razao de letalidade Q5/Q1 usada na padronizacao aritmetica",
         cpl["gradient_cfr_overall_Q5_over_Q1"], "coupling/coupling_report.json",
         "letalidade Q5", "letalidade Q1", W, f"{HR}, quintil de H",
         ">=30 casos com desfecho conhecido entre internados", "observado"),
        ("Resultados", "Gradiente esperado so por composicao internados/nao internados",
         cpl["gradient_predicted_by_mix_alone"], "coupling/coupling_report.json",
         "letalidade predita Q5", "letalidade predita Q1", W, f"{HR}, quintil de H",
         "letalidades especificas fixas nos valores nacionais", "padronizacao aritmetica"),
        ("Resultados", "Razao Q5/Q1 bruta, padronizacao por idade e sexo", cmx["standardisation"]["crude_Q5_over_Q1"],
         "case_mix/case_mix_report.json", "letalidade Q5", "letalidade Q1", W, f"{HR}, quintil de H",
         "casos com idade e sexo codificados", "bruto"),
        ("Resultados", "Razao Q5/Q1 padronizada por idade e sexo", cmx["standardisation"]["std_Q5_over_Q1"],
         "case_mix/case_mix_report.json", "letalidade padronizada Q5", "letalidade padronizada Q1", W,
         f"{HR}, quintil de H", "padrao = todos os casos confirmados nacionais", "padronizado"),
        ("Resultados/Fig2B", "Razao de incidencia Q1/Q5, casos nao graves", sev["nonsevere_Q1_over_Q5"],
         "construct_validity/construct_validity_report.json", "incidencia nao grave Q1", "incidencia nao grave Q5",
         W, f"{HR}, quintil de H", "bloco clinico decodificado", "observado"),
        ("Resultados/Fig2B", "Razao de incidencia Q1/Q5, casos graves", sev["severe_Q1_over_Q5"],
         "construct_validity/construct_validity_report.json", "incidencia grave Q1", "incidencia grave Q5",
         W, f"{HR}, quintil de H", "ictericia OU renal OU hemorragia", "observado"),
        ("Resultados/Fig2C", "Queda da incidencia notificada Q1/Q5", trep["sinan_incidence_gradient_Q1_over_Q5"],
         "triangulation/triangulation_report.json", "incidencia Q1", "incidencia Q5", WT, f"{HR}, quintil de H",
         ">=30 casos com ATE_HOSP valido", "observado"),
        ("Resultados/Fig2C", "Queda da taxa de internacao SIH Q1/Q5", trep["sih_admission_gradient_Q1_over_Q5"],
         "triangulation/triangulation_report.json", "internacoes SIH Q1", "internacoes SIH Q5", WT,
         f"{HR}, quintil de H", "CID-10 A27, municipio de residencia", "observado"),
        ("Resultados/Fig2C", "Mortalidade SIM, minimo entre quintis", float(tri["sim_mortality_per_1m"].min()),
         "triangulation/three_systems_by_stratum.csv", "obitos SIM A27", "pessoas-ano", WT, f"{HR}, quintil de H",
         "causa basica A27, municipio de residencia", "observado"),
        ("Resultados/Fig2C", "Mortalidade SIM, maximo entre quintis", float(tri["sim_mortality_per_1m"].max()),
         "triangulation/three_systems_by_stratum.csv", "obitos SIM A27", "pessoas-ano", WT, f"{HR}, quintil de H",
         "causa basica A27", "observado"),
        ("Resultados", "Spearman H vs mortalidade SIM", trep["spearman_H_vs_sim_mortality"],
         "triangulation/triangulation_report.json", "-", "regioes elegiveis", WT, HR,
         ">=30 casos com ATE_HOSP valido", "observado"),
        ("Resultados", "Razao obitos SIM / obitos SINAN, nacional",
         trep["numerator_invariance"]["sim_per_sinan_death_national"], "triangulation/triangulation_report.json",
         "obitos SIM A27", "obitos SINAN por leptospirose", WT, "nacional",
         "comparacao de contagens agregadas, sem relacionamento de registros", "observado"),
        ("Resultados", "Spearman H vs taxa de obito SINAN por habitante",
         trep["numerator_invariance"]["spearman_H_vs_sinan_death_rate_per_capita"],
         "triangulation/triangulation_report.json", "-", "regioes", WT, HR, "todas as regioes com H definido", "observado"),
        ("Resultados/Tab2", "OR de H por +10 pp, modelo B", mod["models"]["B_plus_H"]["or_H_per_10pp"]["estimate"],
         "primary_model/primary_model_report.json", "obitos", "desfechos conhecidos", W, f"{HR} x ano",
         "celulas com desfecho e H definidos; sem limiar de contagem", "modelo bayesiano"),
        ("Resultados/Tab2", "OR de H por +10 pp, modelo C (principal)",
         mod["models"]["C_plus_case_mix"]["or_H_per_10pp"]["estimate"], "primary_model/primary_model_report.json",
         "obitos", "desfechos conhecidos", W, f"{HR} x ano",
         "modelo B + proporcao 60+ e proporcao masculina", "modelo bayesiano"),
        ("Resultados/Tab2", "OR de H por +10 pp, modelo D",
         mod["models"]["D_complete_outcomes"]["or_H_per_10pp"]["estimate"], "primary_model/primary_model_report.json",
         "obitos", "desfechos conhecidos", W, f"{HR} x ano", "completude do desfecho >= 95%", "modelo bayesiano"),
        ("Resultados/Tab2", "OR de H por +10 pp, modelo E (so internados)",
         mod["threat_checks"]["E_within_hospitalised"]["or_H_per_10pp"]["estimate"],
         "primary_model/primary_model_report.json", "obitos entre internados", "internados com desfecho conhecido",
         W, f"{HR} x ano", "desfecho redefinido", "modelo bayesiano"),
        ("Resultados/Tab2", "OR de H por +10 pp, modelo F (>=10 casos no denominador de H)",
         mod["threat_checks"]["F_hosp_known_ge10"]["or_H_per_10pp"]["estimate"],
         "primary_model/primary_model_report.json", "obitos", "desfechos conhecidos", W, f"{HR} x ano",
         "hosp_known >= 10", "modelo bayesiano"),
        ("Resultados/Tab2", "OR de H por +10 pp, efeitos fixos de regiao e ano",
         mod["within_region"]["or_H_per_10pp_case_mix"]["estimate"], "primary_model/primary_model_report.json",
         "obitos", "desfechos conhecidos", W, f"{HR} x ano",
         "regioes com ao menos um obito e um sobrevivente", "modelo frequentista"),
        ("Resultados", "OR de H por +10 pp, hierarquico sem limiar de 30 casos",
         exc["c_hierarchical_binomial"]["region_year_time_varying_H_no_exclusion_or_per_10pp"][0],
         "exclusion_sensitivity/exclusion_sensitivity_report.json", "obitos", "desfechos conhecidos", W,
         f"{HR} x ano", "todas as regioes com caso; H variavel no tempo", "modelo bayesiano"),
        ("Resultados", "Reducao da variancia marginal do efeito espacial (A->B)",
         mod["spatial_variance_absorbed_by_H"]["proportion_absorbed_marginal"],
         "primary_model/primary_model_report.json", "variancia B", "variancia A", W, f"{HR} x ano",
         "A e B ajustados sobre as mesmas celulas", "modelo bayesiano"),
        ("Resultados", "Celulas regiao-ano nos modelos A/B/C", mod["data"]["analysis_cells"],
         "primary_model/primary_model_report.json", "celulas", "8.341 combinacoes possiveis", W, f"{HR} x ano",
         "desfecho conhecido > 0 e H definido", "observado"),
        ("Resultados", "Populacao excluida ao limiar de 30 casos (%)",
         exc["a_what_is_excluded_at_30"]["excluded_person_year_share_pct"],
         "exclusion_sensitivity/exclusion_sensitivity_report.json", "pessoas-ano excluidas", "pessoas-ano totais",
         W, HR, "limiar descritivo por quintil", "observado"),
    ]
    cols = ["manuscript_location", "claim", "value", "source_object", "numerator",
            "denominator", "window", "geographic_unit", "inclusion_rule", "value_type"]
    led = pl.DataFrame([dict(zip(cols, r)) for r in rows])
    led.write_csv(OUT / "RESULTS_LEDGER.csv")
    print(f"RESULTS_LEDGER.csv: {led.height} linhas")
    print(led.select("claim", "value", "value_type").to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
