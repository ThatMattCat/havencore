"""Native webpage fetcher for the general tools module.

Replaces the retired upstream ``mcp-server-fetch`` stdio server (the 12th
server, dropped in Phase 1 of the MCP migration): an async httpx GET that
renders HTML to markdown so the LLM can read pages directly, with
``start_index`` paging for long documents (mirroring the upstream tool's
semantics). Problems come back as friendly plain-text ``Error ...`` strings,
never exceptions — consistent with the module's error contract (ordinary,
non-``isError`` content).
"""
import re

import httpx
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

USER_AGENT = "HavenCore-Selene/1.0"
FETCH_TIMEOUT_SEC = 15.0
DEFAULT_MAX_LENGTH = 10000

# Non-text/* content types that are still plain text on the wire.
_TEXTUAL_TYPES = {
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-ndjson",
    "application/rss+xml",
    "application/atom+xml",
}

_HTML_TYPES = {"text/html", "application/xhtml+xml"}


def _make_client() -> httpx.AsyncClient:
    """One client per fetch; patched out by tests (MockTransport)."""
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(FETCH_TIMEOUT_SEC),
        headers={"User-Agent": USER_AGENT},
    )


def _html_to_markdown(html: str) -> str:
    """Render HTML to markdown, dropping script/style noise entirely."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    markdown = MarkdownConverter(heading_style="ATX").convert_soup(soup)
    # Collapse the blank-line runs that stripped markup leaves behind.
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


async def fetch_webpage(
    url: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    start_index: int = 0,
) -> str:
    """Fetch ``url`` and return its readable content as markdown/plain text.

    HTML is converted to markdown; other textual content types pass through
    verbatim; binary content is refused with an informative message. The
    ``start_index``/``max_length`` window pages through long documents, and a
    truncated response says how much remains and where to resume.
    """
    if not url.lower().startswith(("http://", "https://")):
        return f"Error: only http:// and https:// URLs are supported (got: {url})"

    try:
        async with _make_client() as client:
            response = await client.get(url)
    except httpx.TimeoutException:
        return f"Error fetching {url}: request timed out after {int(FETCH_TIMEOUT_SEC)} seconds"
    except httpx.HTTPError as e:  # DNS failure, refused connection, bad redirect...
        return f"Error fetching {url}: {e}"

    if response.status_code >= 400:
        return f"Error fetching {url}: HTTP {response.status_code} {response.reason_phrase}"

    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type in _HTML_TYPES:
        content = _html_to_markdown(response.text)
    elif (
        not content_type  # no header — assume text, best effort
        or content_type.startswith("text/")
        or content_type in _TEXTUAL_TYPES
        or content_type.endswith(("+json", "+xml"))
    ):
        content = response.text
    else:
        return (
            f"Error: {url} returned non-text content (content-type: {content_type}). "
            f"fetch_webpage can only read HTML or text documents."
        )

    if not content:
        return f"The page at {url} returned no readable text content."

    total = len(content)
    if start_index >= total:
        return (
            f"No more content available: the document is {total} characters long "
            f"and start_index was {start_index}."
        )

    chunk = content[start_index:start_index + max_length]
    end = start_index + len(chunk)
    remaining = total - end
    if remaining > 0:
        chunk += (
            f"\n\n<truncated, {remaining} more chars — call fetch_webpage again "
            f"with start_index={end} to continue>"
        )
    return chunk
