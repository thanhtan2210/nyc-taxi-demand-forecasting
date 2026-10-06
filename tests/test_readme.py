"""Keeps the README diagrams in line with the code."""
import re

from src import paths

README = (paths.ROOT / "README.md").read_text(encoding="utf-8")


def mermaid_blocks():
    return re.findall(r"```mermaid\n(.*?)```", README, flags=re.DOTALL)


def test_readme_has_a_flowchart_and_an_er_diagram_without_numbers():
    blocks = mermaid_blocks()
    assert [block.split()[0] for block in blocks] == ["flowchart", "erDiagram"]
    for block in blocks:
        assert not re.search(r"\d", block), "diagrams must not contain numbers"


def test_er_diagram_tables_exist_in_the_warehouse_code():
    diagram = next(block for block in mermaid_blocks() if block.startswith("erDiagram"))
    tables = re.findall(r"^\s*(\w+) \{$", diagram, flags=re.MULTILINE)
    assert tables == ["dim_zone", "dim_service", "dim_hour", "fact_pickups_hourly", "mart_demand_hourly"]
    code = (paths.ROOT / "src" / "warehouse.py").read_text(encoding="utf-8")
    code += "".join(p.read_text(encoding="utf-8") for p in sorted(paths.SQL_DIR.glob("*.sql")))
    for table in tables:
        assert table in code, f"{table} is not created anywhere in sql/ or src/warehouse.py"
    # every table used in a relationship line is one of the declared tables
    related = set(re.findall(r"^\s*(\w+) \|\|--o\{ (\w+) :", diagram, flags=re.MULTILINE))
    assert related and {name for pair in related for name in pair} <= set(tables)


def test_er_diagram_columns_match_the_fact_table_ddl():
    diagram = next(block for block in mermaid_blocks() if block.startswith("erDiagram"))
    body = re.search(r"fact_pickups_hourly \{\n(.*?)\n\s*\}", diagram, flags=re.DOTALL).group(1)
    in_diagram = [tuple(line.split()[:2]) for line in body.splitlines()]
    ddl = (paths.SQL_DIR / "schema.sql").read_text(encoding="utf-8")
    in_ddl = re.findall(r"^\s+(\w+)\s+(TIMESTAMP|INTEGER|TINYINT)\s+NOT NULL", ddl, flags=re.MULTILINE)
    assert in_diagram == [(kind, name) for name, kind in in_ddl]


def test_every_image_shown_in_the_readme_exists():
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README)
    assert images, "the README is expected to show screenshots"
    for image in images:
        assert (paths.ROOT / image).is_file(), image
