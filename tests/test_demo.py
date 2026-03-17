from __future__ import annotations

import json

from nyne_er_lab.demo import build_demo_artifacts


def test_demo_builder_generates_expected_artifacts(tmp_path) -> None:
    outputs = build_demo_artifacts(tmp_path)

    metrics_path = tmp_path / "benchmark_metrics.json"
    demo_path = tmp_path / "demo.html"
    blog_path = tmp_path / "blog.md"

    assert outputs["metrics_json"] == str(metrics_path)
    assert outputs["demo_html"] == str(demo_path)
    assert outputs["blog_md"] == str(blog_path)

    payload = json.loads(metrics_path.read_text())
    assert len(payload["models"]) == 4
    assert payload["cluster_f1"] >= 0.8
    assert len(payload["cases"]) >= 5

    html = demo_path.read_text()
    assert "Interpretable Public-Web Entity Resolution Lab" in html
    assert "Curated Cases" in html

    blog = blog_path.read_text()
    assert "# Interpretable Public-Web Entity Resolution Lab" in blog
    assert "## Benchmark" in blog
    assert "## Failure Modes" in blog
