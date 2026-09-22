"""Rendering the strings a model actually sees: query prompts and documents.

Pure functions over a ModelSpec -- no model or tokenizer needed.
"""

from typing import List, Optional

from embedder.specs import ModelSpec


def validate_document_template(template: str) -> None:
    """Fail at construction, not mid-indexing, on a malformed template."""
    if "{content}" not in template:
        raise ValueError(f"document_template must contain {{content}}: {template!r}")
    try:
        template.format(title="", content="")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"document_template may only use {{title}} and {{content}} "
            f"(write literal braces as {{{{ }}}}): {template!r}"
        ) from exc


def query_prompt(spec: ModelSpec, task: str) -> str:
    """The prefix applied to queries."""
    if spec.instruct_template is None:
        return spec.query_prefix
    return spec.instruct_template.format(task=task)


def resolve_title(spec: ModelSpec, title: Optional[str]) -> str:
    return title if title else spec.empty_title


def format_document(
    spec: ModelSpec, template: str, content: str, title: Optional[str] = None
) -> str:
    """The exact string embedded for one document."""
    rendered = template.format(title=resolve_title(spec, title), content=content)
    # Strip so an empty {title} doesn't leave stray leading separators.
    return rendered.strip()


def template_overhead(spec: ModelSpec, template: str, title: Optional[str] = None) -> str:
    """Everything the template adds around the content (prefix, title,
    separators) -- what chunking has to leave room for."""
    return template.format(title=resolve_title(spec, title), content="")


def pair_titles(
    texts: List[str], titles: Optional[List[Optional[str]]]
) -> List[Optional[str]]:
    """One title per text; None means no titles at all."""
    if titles is None:
        return [None] * len(texts)
    if len(titles) != len(texts):
        raise ValueError(f"got {len(texts)} texts but {len(titles)} titles")
    return list(titles)
