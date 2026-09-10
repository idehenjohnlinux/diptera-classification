from pathlib import Path

from graphviz import Digraph


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NAME = "pipeline_computacional"


# ============================================================
# CRIAÇÃO DO DIAGRAMA
# ============================================================

pipeline = Digraph(
    name="pipeline_computacional",
    format="png",
)

pipeline.attr(
    rankdir="TB",
    bgcolor="white",
    pad="0.25",
    nodesep="0.35",
    ranksep="0.55",
    dpi="300",
)

pipeline.attr(
    "node",
    shape="box",
    style="rounded,filled",
    fillcolor="white",
    fontname="Arial",
    fontsize="14",
    fontcolor="#222222",
    penwidth="2.2",
    width="2.8",
    height="0.65",
    margin="0.15,0.10",
)

pipeline.attr(
    "edge",
    color="#B5B5B5",
    penwidth="2.0",
    arrowsize="0.85",
)


# ============================================================
# NÓS DO PIPELINE
# ============================================================

pipeline.node(
    "input",
    "Dados de Entrada",
    color="#A8CF45",
)

pipeline.node(
    "validation",
    "Validação dos Dados",
    color="#52C95A",
)

pipeline.node(
    "preparation",
    "Preparação dos Dados",
    color="#32B8D8",
)

pipeline.node(
    "cross_validation",
    "Validação Cruzada",
    color="#48C9B0",
)

pipeline.node(
    "training",
    "Treino",
    color="#D9A52E",
)

pipeline.node(
    "evaluation",
    "Avaliação",
    color="#7E57C2",
)

pipeline.node(
    "best_model",
    "Seleção do Melhor Modelo",
    color="#E67E22",
)

pipeline.node(
    "best_view",
    "Seleção da Melhor Vista",
    color="#F4B942",
)

pipeline.node(
    "prediction",
    "Classificação Hierárquica",
    color="#3498DB",
)

pipeline.node(
    "reports",
    "Resultados Finais",
    color="#16A085",
)


# ============================================================
# LIGAÇÕES ENTRE OS NÓS
# ============================================================

pipeline.edge(
    "input",
    "validation",
)

pipeline.edge(
    "validation",
    "preparation",
)

pipeline.edge(
    "preparation",
    "cross_validation",
)

pipeline.edge(
    "cross_validation",
    "training",
)

pipeline.edge(
    "training",
    "evaluation",
)

pipeline.edge(
    "evaluation",
    "best_model",
)

pipeline.edge(
    "best_model",
    "best_view",
)

pipeline.edge(
    "best_view",
    "prediction",
)

pipeline.edge(
    "prediction",
    "reports",
)




# ============================================================
# EXPORTAÇÃO
# ============================================================

output_path = pipeline.render(
    filename=OUTPUT_NAME,
    directory=str(OUTPUT_DIR),
    cleanup=True,
)

print("=" * 70)
print("PIPELINE COMPUTACIONAL CRIADO")
print("=" * 70)
print(f"Ficheiro: {output_path}")
print("=" * 70)
