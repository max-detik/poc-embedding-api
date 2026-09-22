import pytest

from embedder import prompts
from embedder.specs import MODEL_SPECS, get_spec

HARRIER, GEMMA = MODEL_SPECS["harrier"], MODEL_SPECS["gemma"]


def test_query_prompt_keeps_each_models_training_format():
    # Harrier's trailing space after "Query:" is deliberate.
    assert prompts.query_prompt(HARRIER, "Cari berita") == "Instruct: Cari berita\nQuery: "
    # Gemma uses a fixed prefix and ignores the task.
    assert prompts.query_prompt(GEMMA, "Cari berita") == "task: search result | query: "


@pytest.mark.parametrize("spec, title, expected", [
    (HARRIER, "Judul", "isi"),
    (HARRIER, None, "isi"),
    (GEMMA, "Judul", "title: Judul | text: isi"),
    (GEMMA, None, "title: none | text: isi"),
    (GEMMA, "", "title: none | text: isi"),
])
def test_default_document_templates(spec, title, expected):
    assert prompts.format_document(spec, spec.document_template, "isi", title) == expected


def test_custom_template_strips_empty_title_separator():
    template = "{title}\n\n{content}"
    assert prompts.format_document(HARRIER, template, "isi", "Judul") == "Judul\n\nisi"
    assert prompts.format_document(HARRIER, template, "isi", None) == "isi"


def test_braces_inside_values_are_not_reinterpreted():
    out = prompts.format_document(HARRIER, "{title}: {content}", "kode {x} {content}", "T {y}")
    assert out == "T {y}: kode {x} {content}"


@pytest.mark.parametrize("bad", ["{title} only", "{content} {author}", "{content} {", "{0}{content}"])
def test_malformed_templates_are_rejected(bad):
    with pytest.raises(ValueError):
        prompts.validate_document_template(bad)


def test_literal_braces_are_allowed():
    prompts.validate_document_template("{{json}} {content}")


def test_template_overhead_includes_title():
    assert prompts.template_overhead(GEMMA, GEMMA.document_template, "Judul") == "title: Judul | text: "


def test_pair_titles():
    assert prompts.pair_titles(["a", "b"], None) == [None, None]
    assert prompts.pair_titles(["a", "b"], ["x", None]) == ["x", None]
    with pytest.raises(ValueError, match="2 texts but 1 titles"):
        prompts.pair_titles(["a", "b"], ["x"])


def test_unknown_model_key():
    with pytest.raises(ValueError, match="Unknown model_key"):
        get_spec("qwen3")
