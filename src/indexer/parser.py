"""
Markdown parser for WaveMaker documentation.

Extracts:
- Frontmatter metadata
- Header structure
- Content sections
- Code blocks
- Links and images
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import frontmatter
import mistune

logger = logging.getLogger(__name__)


@dataclass
class Header:
    """A header in the document."""
    level: int
    text: str
    line: int


@dataclass
class CodeBlock:
    """A code block in the document."""
    language: str
    content: str
    line: int


@dataclass
class ParsedDocument:
    """Result of parsing a markdown document."""
    file_path: str
    frontmatter: dict[str, Any]
    title: str
    description: Optional[str]
    keywords: list[str]
    headers: list[Header]
    code_blocks: list[CodeBlock]
    content: str  # Full markdown content (without frontmatter)
    raw_content: str  # Original file content


class MarkdownParser:
    """
    Parser for WaveMaker documentation markdown files.
    """

    def __init__(self):
        self._mdx_component_pattern = re.compile(
            r'<[A-Z][a-zA-Z]*[^>]*(?:/>|>[\s\S]*?</[A-Z][a-zA-Z]*>)',
            re.MULTILINE,
        )
        self._import_pattern = re.compile(
            r'^import\s+.*?(?:from\s+[\'"].*?[\'"])?;?\s*$',
            re.MULTILINE,
        )

    def parse_file(self, file_path: Path) -> Optional[ParsedDocument]:
        """
        Parse a markdown/MDX file.

        Args:
            file_path: Path to the markdown file

        Returns:
            ParsedDocument or None if parsing fails
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            return self.parse_content(content, str(file_path))
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None

    def parse_content(
        self,
        content: str,
        file_path: str = "",
    ) -> ParsedDocument:
        """
        Parse markdown content.

        Args:
            content: Raw markdown content
            file_path: Path for reference

        Returns:
            ParsedDocument
        """
        # Extract frontmatter
        post = frontmatter.loads(content)
        fm = dict(post.metadata)
        markdown_content = post.content

        # Clean MDX content
        cleaned_content = self._clean_mdx(markdown_content)

        # Extract headers
        headers = self._extract_headers(cleaned_content)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(cleaned_content)

        # Get title from frontmatter or first H1
        title = fm.get("title", "")
        if not title and headers:
            h1_headers = [h for h in headers if h.level == 1]
            if h1_headers:
                title = h1_headers[0].text

        # Get description
        description = fm.get("description", fm.get("sidebar_label", ""))

        # Get keywords
        keywords = fm.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",")]

        return ParsedDocument(
            file_path=file_path,
            frontmatter=fm,
            title=title,
            description=description,
            keywords=keywords,
            headers=headers,
            code_blocks=code_blocks,
            content=cleaned_content,
            raw_content=content,
        )

    def _clean_mdx(self, content: str) -> str:
        """
        Clean MDX content by removing/simplifying JSX components.
        """
        # Remove import statements
        content = self._import_pattern.sub("", content)

        # Simplify common MDX components
        # Convert <Tabs> content to plain text
        content = re.sub(r'<Tabs[^>]*>', '', content)
        content = re.sub(r'</Tabs>', '', content)
        content = re.sub(r'<TabItem[^>]*label="([^"]*)"[^>]*>', r'\n### \1\n', content)
        content = re.sub(r'</TabItem>', '', content)

        # Convert admonitions to markdown format
        content = self._convert_admonitions(content)

        # Remove remaining JSX components but keep their text content
        content = self._mdx_component_pattern.sub(
            lambda m: self._extract_text_from_jsx(m.group(0)),
            content,
        )

        # Clean up multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()

    def _convert_admonitions(self, content: str) -> str:
        """Convert Docusaurus admonitions to markdown."""
        # :::note, :::tip, :::warning, :::danger, :::info
        admonition_pattern = re.compile(
            r':::(note|tip|warning|danger|info|caution)(?:\[([^\]]*)\])?\s*\n([\s\S]*?):::',
            re.MULTILINE,
        )

        def replace_admonition(match):
            admonition_type = match.group(1).upper()
            title = match.group(2) or admonition_type
            content = match.group(3).strip()
            return f"**{title}**: {content}"

        return admonition_pattern.sub(replace_admonition, content)

    def _extract_text_from_jsx(self, jsx: str) -> str:
        """Extract plain text from JSX component."""
        # Remove tags but keep content
        text = re.sub(r'<[^>]+>', ' ', jsx)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_headers(self, content: str) -> list[Header]:
        """Extract all headers from markdown content."""
        headers = []
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                # Remove any trailing anchors like {#custom-id}
                text = re.sub(r'\s*\{#[^}]+\}\s*$', '', text)
                headers.append(Header(level=level, text=text, line=line_num))

        return headers

    def _extract_code_blocks(self, content: str) -> list[CodeBlock]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        pattern = re.compile(r'```(\w*)\n([\s\S]*?)```', re.MULTILINE)

        for match in pattern.finditer(content):
            language = match.group(1) or "text"
            code_content = match.group(2).strip()
            # Calculate line number
            line = content[:match.start()].count('\n') + 1

            code_blocks.append(CodeBlock(
                language=language,
                content=code_content,
                line=line,
            ))

        return code_blocks

    def should_index(self, doc: ParsedDocument) -> bool:
        """
        Check if a document should be indexed.

        Skip drafts, empty docs, etc.
        """
        # Skip drafts
        if doc.frontmatter.get("draft", False):
            return False

        # Skip if no content
        if not doc.content.strip():
            return False

        # Skip if marked to hide
        if doc.frontmatter.get("hide_table_of_contents") and not doc.content.strip():
            return False

        return True
