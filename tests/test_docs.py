import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READMES = (ROOT / "README.md", ROOT / "README.zh-CN.md")
PUBLIC_MARKDOWN = (*READMES, *sorted((ROOT / "docs").glob("*.md")))
REPOSITORY = "LawrenceRiver/TextGT"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CI_BADGE = (
    f"[![CI](https://github.com/{REPOSITORY}/actions/workflows/ci.yml/badge.svg)]"
    f"(https://github.com/{REPOSITORY}/actions/workflows/ci.yml)"
)
PAPER_PATH = "paper/group2-term-paper-public.pdf"
FORBIDDEN_PUBLIC_PHRASES = (
    "## Paper\n",
    "## 论文\n",
    "preferred-citation:",
)
EXPECTED_BADGES = (
    CI_BADGE,
    (
        "![Python 3.9 and 3.10]"
        "(https://img.shields.io/badge/Python-3.9%20%7C%203.10-3776AB?"
        "logo=python&logoColor=white)"
    ),
    (
        "![PyTorch 1.12.1]"
        "(https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?"
        "logo=pytorch&logoColor=white)"
    ),
    "[![MIT software license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)",
    (
        "[![Upstream TextGT]"
        "(https://img.shields.io/badge/upstream-shuoyinn%2FTextGT-181717?logo=github)]"
        "(https://github.com/shuoyinn/TextGT)"
    ),
    (
        "[![Base paper DOI]"
        "(https://img.shields.io/badge/base%20paper-10.1609%2Faaai.v38i17.29911-2f6f9f)]"
        "(https://doi.org/10.1609/aaai.v38i17.29911)"
    ),
)
HEADING_KEYS = {
    "Upstream relationship": "upstream",
    "与上游项目的关系": "upstream",
    "Architecture": "architecture",
    "架构": "architecture",
    "Method": "method",
    "方法": "method",
    "Fact": "fact",
    "事实": "fact",
    "Comparison": "comparison",
    "比较": "comparison",
    "Intensity": "intensity",
    "强度": "intensity",
    "Composer": "composer",
    "组合器": "composer",
    "Reported Restaurant results": "results",
    "Restaurant 已报告结果": "results",
    "Installation": "installation",
    "安装": "installation",
    "Quick start": "quick-start",
    "快速开始": "quick-start",
    "Configuration": "configuration",
    "配置": "configuration",
    "Prior construction": "prior",
    "构建先验": "prior",
    "Reproducibility": "reproducibility",
    "可复现性": "reproducibility",
    "Contributors": "contributors",
    "贡献者": "contributors",
    "Citation": "citation",
    "引用": "citation",
    "License": "license",
    "许可证": "license",
}
EXPECTED_HEADING_STRUCTURE = (
    (2, "upstream"),
    (2, "architecture"),
    (2, "method"),
    (3, "fact"),
    (3, "comparison"),
    (3, "intensity"),
    (3, "composer"),
    (2, "results"),
    (2, "installation"),
    (2, "quick-start"),
    (2, "configuration"),
    (2, "prior"),
    (2, "reproducibility"),
    (2, "contributors"),
    (2, "citation"),
    (2, "license"),
)


def markdown_targets(document: Path) -> set[str]:
    text = document.read_text(encoding="utf-8")
    return set(re.findall(r"\]\(([^)]+)\)", text))


def badge_lines(document: Path) -> tuple[str, ...]:
    preamble = document.read_text(encoding="utf-8").split("\n## ", 1)[0]
    return tuple(line for line in preamble.splitlines() if line.startswith(("![", "[![")))


def normalized_heading_structure(document: Path) -> tuple[tuple[int, str], ...]:
    text = document.read_text(encoding="utf-8")
    headings = re.findall(r"^(#{2,3}) ([^#\n].*)$", text, re.MULTILINE)
    return tuple(
        (len(marks), HEADING_KEYS.get(title, f"unknown:{title}")) for marks, title in headings
    )


def test_local_markdown_links_exist() -> None:
    missing = []
    for document in PUBLIC_MARKDOWN:
        for target in markdown_targets(document):
            if target.startswith(("http://", "https://", "#")):
                continue
            path = (document.parent / target.split("#", 1)[0]).resolve()
            if not path.exists():
                missing.append(f"{document.relative_to(ROOT)}: {target}")
    assert not missing, "\n".join(missing)


def test_readmes_link_each_other() -> None:
    assert "README.zh-CN.md" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "README.md" in (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")


def test_readmes_begin_with_the_same_language_switch() -> None:
    expected = "[English](README.md) | [简体中文](README.zh-CN.md)"
    assert all(
        document.read_text(encoding="utf-8").splitlines()[0] == expected for document in READMES
    )


def test_readmes_keep_matching_section_order() -> None:
    for document in READMES:
        assert normalized_heading_structure(document) == EXPECTED_HEADING_STRUCTURE


def test_readmes_reference_public_release_artifacts() -> None:
    required = {
        "assets/architecture.png",
        "results/reported_metrics.json",
        "docs/method.md",
        "docs/results.md",
        "docs/reproducibility.md",
        "AUTHORS.md",
        "CONTRIBUTING.md",
        "CITATION.cff",
        "LICENSE",
        "NOTICE",
    }
    for document in READMES:
        assert required <= markdown_targets(document)


def test_release_attribution_files_agree() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert f'repository-code: "https://github.com/{REPOSITORY}"' in citation
    assert "https://github.com/shuoyinn/TextGT" in notice
    assert "10.1609/aaai.v38i17.29911" in notice
    assert "not endorsed by or affiliated with" in notice
    for document in READMES:
        text = document.read_text(encoding="utf-8")
        assert "shuoyinn/TextGT" in text
        assert "10.1609/aaai.v38i17.29911" in text
        assert "CITATION.cff" in text
        assert "NOTICE" in text


def test_release_is_paperless() -> None:
    assert not (ROOT / PAPER_PATH).exists()
    public_metadata = (
        *READMES,
        ROOT / "AUTHORS.md",
        ROOT / "CITATION.cff",
        ROOT / "NOTICE",
    )
    for document in public_metadata:
        text = document.read_text(encoding="utf-8")
        assert PAPER_PATH not in text
        assert not any(phrase in text for phrase in FORBIDDEN_PUBLIC_PHRASES)


def test_readmes_document_configuration() -> None:
    for document in READMES:
        text = document.read_text(encoding="utf-8")
        assert "modules" in text
        assert "prior" in text.lower()
        assert "DEEPSEEK_API_KEY" in text


def test_method_documents_exact_logit_shapes_and_errors() -> None:
    method = " ".join((ROOT / "docs" / "method.md").read_text(encoding="utf-8").split())

    assert "`apply_pragmatic_residual` accepts logits shaped `[3]`." in method
    assert "`apply_batch` accepts logits shaped `[batch, 3]`." in method
    assert "Single-sample logits with any other shape raise `ValueError`." in method
    assert "Batch logits with any other shape raise `ValueError`." in method


def test_readme_badges_are_real_and_limited_to_the_allowed_set() -> None:
    assert CI_WORKFLOW.is_file()
    for document in READMES:
        assert badge_lines(document) == EXPECTED_BADGES
