from fast_rag.extract import extract_html, split_passages


def test_extract_html_prefers_article_text() -> None:
    html = """
    <html>
      <head><title>Example</title><script>ignore()</script></head>
      <body>
        <nav>navigation should not dominate</nav>
        <article>
          <h1>Real heading for the page</h1>
          <p>This is the useful body paragraph with enough detail to be retained by the extractor.</p>
        </article>
      </body>
    </html>
    """
    title, text = extract_html(html)
    assert title == "Example"
    assert "useful body paragraph" in text
    assert "ignore" not in text


def test_split_passages_keeps_overlap() -> None:
    text = "First sentence. " * 120
    passages = split_passages(text, target_chars=220, overlap_chars=40)
    assert len(passages) > 1
    assert all(passages)

