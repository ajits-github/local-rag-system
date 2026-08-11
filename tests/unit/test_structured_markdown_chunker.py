from __future__ import annotations

from rag.chunkers.recursive_chunker import RecursiveCharacterChunker
from rag.chunkers.structured_markdown import StructuredMarkdownChunker


def test_non_markdown_source_type_never_triggers_structural_parsing():
    """A table-looking body is untouched when source_type isn't 'markdown'."""
    chunker = StructuredMarkdownChunker()
    text = "| a | b |\n|---|---|\n| 1 | 2 |"

    spans = chunker.split(text, source_type="text")

    assert len(spans) == 1
    assert spans[0].content_type is None
    assert spans[0].text == text


def test_prose_only_markdown_matches_recursive_chunker_byte_for_byte():
    """A prose-only Markdown document chunks identically to the plain recursive chunker."""
    text = "This is a prose paragraph.\n\nAnother paragraph with more words in it, quite a lot."
    structured = StructuredMarkdownChunker(chunk_size=40, chunk_overlap=5)
    plain = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=5)

    structured_texts = [s.text for s in structured.split(text, source_type="markdown")]
    plain_texts = [s.text for s in plain.split(text)]

    assert structured_texts == plain_texts


def test_table_is_kept_atomic_with_headers_and_body():
    """A whole small table becomes one span, headers and rows together."""
    chunker = StructuredMarkdownChunker()
    text = "| Name | Value |\n|---|---|\n| a | 1 |\n| b | 2 |"

    spans = chunker.split(text, source_type="markdown")

    assert len(spans) == 1
    span = spans[0]
    assert span.content_type == "table"
    assert span.table_headers == ["Name", "Value"]
    assert "| a | 1 |" in span.text
    assert "| b | 2 |" in span.text
    assert span.source_anchor is None  # only one group


def test_large_table_splits_into_row_groups_with_repeated_headers():
    """A table with more rows than table_row_group_size splits into multiple atomic groups."""
    chunker = StructuredMarkdownChunker(table_row_group_size=20)
    header = "| Name | Value |\n|---|---|"
    rows = "\n".join(f"| row{i} | {i} |" for i in range(45))
    text = f"{header}\n{rows}"

    spans = chunker.split(text, source_type="markdown")

    assert len(spans) == 3
    assert [s.content_type for s in spans] == ["table", "table", "table"]
    assert all(s.table_headers == ["Name", "Value"] for s in spans)
    assert spans[0].source_anchor == "rows 1-20"
    assert spans[1].source_anchor == "rows 21-40"
    assert spans[2].source_anchor == "rows 41-45"
    # Every group repeats the header row.
    assert all("| Name | Value |" in s.text for s in spans)


def test_fenced_code_block_is_kept_atomic_with_language_tag():
    """A fenced code block becomes one content_type='code' span, tagging its language."""
    chunker = StructuredMarkdownChunker()
    text = "Some intro text.\n\n```python\ndef f():\n    return 1\n```\n\nMore text after."

    spans = chunker.split(text, source_type="markdown")

    code_spans = [s for s in spans if s.content_type == "code"]
    assert len(code_spans) == 1
    assert code_spans[0].code_language == "python"
    assert "def f():" in code_spans[0].text


def test_json_fence_is_tagged_as_configuration():
    """A ```json fence gets content_type='configuration', not 'code'."""
    chunker = StructuredMarkdownChunker()
    text = '```json\n{"key": "value"}\n```'

    spans = chunker.split(text, source_type="markdown")

    assert len(spans) == 1
    assert spans[0].content_type == "configuration"
    assert spans[0].code_language == "json"


def test_oversized_fence_falls_back_to_prose_chunker_preserving_tags():
    """A fence exceeding max_atomic_block_chars splits, each sub-span keeping its tags."""
    chunker = StructuredMarkdownChunker(max_atomic_block_chars=200)
    body = "\n".join(f"line {i}" for i in range(80))
    text = f"```python\n{body}\n```"

    spans = chunker.split(text, source_type="markdown")

    assert len(spans) > 1
    assert all(s.content_type == "code" for s in spans)
    assert all(s.code_language == "python" for s in spans)


def test_chart_caption_pairs_with_text_fence_into_one_chart_span():
    """A ```text fence + an emphasis-wrapped paragraph right after it becomes a 'chart' span."""
    chunker = StructuredMarkdownChunker()
    text = "```text\nQ1 |####\nQ2 |######\n```\n\n*Chart caption: usage grew steadily.*"

    spans = chunker.split(text, source_type="markdown")

    assert len(spans) == 1
    assert spans[0].content_type == "chart"
    assert "Q1 |####" in spans[0].text
    assert "Chart caption: usage grew steadily." in spans[0].text


def test_text_fence_without_emphasis_caption_is_plain_code_not_chart():
    """A ```text fence followed by a plain (non-emphasis) paragraph is NOT tagged as a chart."""
    chunker = StructuredMarkdownChunker()
    text = "```text\nsome preformatted text\n```\n\nThis is a normal paragraph, not a caption."

    spans = chunker.split(text, source_type="markdown")

    assert spans[0].content_type == "code"
    assert spans[0].code_language == "text"


def test_section_path_tracks_nested_headers():
    """A structural boundary (here, a fence) flushes prose under the header stack active then."""
    chunker = StructuredMarkdownChunker()
    text = "# Top\n\nIntro under top.\n\n```python\nx = 1\n```\n\n## Sub\n\nDetail under sub."

    spans = chunker.split(text, source_type="markdown")

    paths = [s.section_path for s in spans]
    assert "Top" in paths  # prose flushed by the fence, while header_stack == ["Top"]
    assert "Top > Sub" in paths  # prose flushed at EOF, after "## Sub" was seen


def test_attachment_tagged_only_on_containing_sub_chunk():
    """Only the specific prose sub-chunk containing an asset link gets attachment_name set."""
    chunker = StructuredMarkdownChunker(chunk_size=40, chunk_overlap=0)
    text = (
        "This is an unrelated paragraph with no links in it at all whatsoever.\n\n"
        "See the diagram: ![Diagram](assets/diagram.svg) for details."
    )

    spans = chunker.split(text, source_type="markdown")

    tagged = [s for s in spans if s.attachment_name is not None]
    assert len(tagged) == 1
    assert tagged[0].attachment_name == "diagram.svg"
    assert tagged[0].source_anchor == "assets/diagram.svg"
    untagged = [s for s in spans if s.attachment_name is None]
    assert untagged  # the unrelated paragraph stays untagged


def test_mermaid_language_defaults_to_code():
    """An unrecognized fence language (e.g. mermaid) defaults to content_type='code'."""
    chunker = StructuredMarkdownChunker()
    text = "```mermaid\ngraph TD;\nA-->B;\n```"

    spans = chunker.split(text, source_type="markdown")

    assert spans[0].content_type == "code"
    assert spans[0].code_language == "mermaid"


def test_standalone_image_line_becomes_its_own_image_span():
    """A `![alt](path)` line alone on its line is a dedicated content_type='image' span."""
    chunker = StructuredMarkdownChunker()
    text = (
        "Some intro prose.\n\n"
        "![Service topology](images/topology.png)\n\n"
        "More prose after the image."
    )

    spans = chunker.split(text, source_type="markdown")

    image_spans = [s for s in spans if s.content_type == "image"]
    assert len(image_spans) == 1
    assert image_spans[0].attachment_name == "topology.png"
    assert image_spans[0].source_anchor == "images/topology.png"
    assert "![Service topology](images/topology.png)" in image_spans[0].text


def test_image_caption_folds_into_the_same_image_span():
    """An emphasis-wrapped paragraph right after an image line joins that image's own span."""
    chunker = StructuredMarkdownChunker()
    text = "![Diagram](images/x.png)\n\n*Figure 1: Components and dependencies.*\n\nNext para."

    spans = chunker.split(text, source_type="markdown")

    image_spans = [s for s in spans if s.content_type == "image"]
    assert len(image_spans) == 1
    assert "Figure 1: Components and dependencies." in image_spans[0].text
    # The caption text is consumed into the image span, not left as its own
    # separate prose span.
    assert not any("Figure 1" in s.text for s in spans if s.content_type != "image")


def test_image_without_caption_is_still_its_own_span():
    """An image line with no following emphasis-wrapped paragraph is still a bare image span."""
    chunker = StructuredMarkdownChunker()
    text = "![Diagram](images/x.png)\n\nA plain paragraph, not a caption."

    spans = chunker.split(text, source_type="markdown")

    image_spans = [s for s in spans if s.content_type == "image"]
    assert len(image_spans) == 1
    assert image_spans[0].text == "![Diagram](images/x.png)"
    prose_spans = [s for s in spans if s.content_type != "image"]
    assert any("A plain paragraph, not a caption." in s.text for s in prose_spans)


def test_image_span_section_path_matches_surrounding_headers():
    """An image span picks up section_path from the same header-tracking logic as other spans."""
    chunker = StructuredMarkdownChunker()
    text = "# Top\n\n## Sub\n\n![Diagram](images/x.png)"

    spans = chunker.split(text, source_type="markdown")

    image_spans = [s for s in spans if s.content_type == "image"]
    assert len(image_spans) == 1
    assert image_spans[0].section_path == "Top > Sub"


def test_non_asset_image_extension_is_not_treated_as_a_block():
    """A standalone image-syntax line pointing at a non-asset extension stays ordinary prose."""
    chunker = StructuredMarkdownChunker()
    text = "![Not an asset](https://example.com/tracking-pixel.gif?x=1)"

    spans = chunker.split(text, source_type="markdown")

    assert all(s.content_type != "image" for s in spans)
