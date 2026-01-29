"""
Semantic chunker for WaveMaker documentation.

Chunks documents by:
- H2 (##) section boundaries
- Preserves H1 context in each chunk
- Handles size limits
- Keeps code blocks and tables intact
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.config.settings import get_settings
from src.indexer.parser import Header, ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of document content ready for indexing."""
    id: str
    content: str
    title: str
    section: Optional[str]
    url: str
    file_path: str
    doc_type: str
    has_code: bool
    code_language: Optional[str]
    word_count: int
    chunk_index: int
    metadata: dict[str, Any]


class SemanticChunker:
    """
    Chunks documents semantically based on header structure.
    """

    def __init__(self):
        self.settings = get_settings()
        self._docs_base_url = self.settings.docs_base_url.rstrip("/")

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """
        Split a document into semantic chunks.

        Args:
            doc: Parsed document

        Returns:
            List of chunks
        """
        chunks = []

        # Get the H1 title for context
        h1_title = doc.title or "Documentation"

        # Split content by H2 headers
        sections = self._split_by_h2(doc.content, doc.headers)

        # Process each section
        for idx, (section_header, section_content) in enumerate(sections):
            # Skip empty sections
            if not section_content.strip():
                continue

            # Check if section needs sub-splitting
            if self._needs_splitting(section_content):
                sub_chunks = self._split_large_section(
                    section_content,
                    doc.headers,
                    idx,
                )
                for sub_idx, sub_content in enumerate(sub_chunks):
                    chunk = self._create_chunk(
                        doc=doc,
                        h1_title=h1_title,
                        section_header=section_header,
                        content=sub_content,
                        chunk_index=len(chunks),
                    )
                    chunks.append(chunk)
            else:
                chunk = self._create_chunk(
                    doc=doc,
                    h1_title=h1_title,
                    section_header=section_header,
                    content=section_content,
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)

        # If no chunks yet (no H2 headers), create one from entire content
        if not chunks and doc.content.strip():
            chunk = self._create_chunk(
                doc=doc,
                h1_title=h1_title,
                section_header=None,
                content=doc.content,
                chunk_index=0,
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks from {doc.file_path}")
        return chunks

    def _split_by_h2(
        self,
        content: str,
        headers: list[Header],
    ) -> list[tuple[Optional[str], str]]:
        """
        Split content by H2 (##) headers.

        Returns:
            List of (section_header, section_content) tuples
        """
        sections = []
        lines = content.split('\n')

        # Find H2 header positions
        h2_positions = []
        for i, line in enumerate(lines):
            if re.match(r'^##\s+', line.strip()):
                header_text = re.sub(r'^##\s+', '', line.strip())
                header_text = re.sub(r'\s*\{#[^}]+\}\s*$', '', header_text)
                h2_positions.append((i, header_text))

        if not h2_positions:
            # No H2 headers, return entire content as one section
            return [(None, content)]

        # Handle content before first H2
        first_h2_line = h2_positions[0][0]
        if first_h2_line > 0:
            intro_content = '\n'.join(lines[:first_h2_line]).strip()
            if intro_content:
                # Remove H1 from intro (it's used as title)
                intro_content = re.sub(r'^#\s+[^\n]+\n*', '', intro_content).strip()
                if intro_content:
                    sections.append((None, intro_content))

        # Process each H2 section
        for i, (line_num, header_text) in enumerate(h2_positions):
            # Find end of this section
            if i + 1 < len(h2_positions):
                end_line = h2_positions[i + 1][0]
            else:
                end_line = len(lines)

            # Extract section content (including the H2 header)
            section_lines = lines[line_num:end_line]
            section_content = '\n'.join(section_lines).strip()

            sections.append((header_text, section_content))

        return sections

    def _needs_splitting(self, content: str) -> bool:
        """Check if content exceeds max chunk size."""
        word_count = len(content.split())
        # Approximate tokens as 0.75 * words
        approx_tokens = int(word_count * 0.75)
        return approx_tokens > self.settings.chunk_max_tokens

    def _split_large_section(
        self,
        content: str,
        headers: list[Header],
        section_index: int,
    ) -> list[str]:
        """
        Split a large section into smaller chunks.
        """
        chunks = []

        # First, try to split by H3 (###) headers
        h3_pattern = re.compile(r'^###\s+', re.MULTILINE)
        h3_matches = list(h3_pattern.finditer(content))

        if h3_matches:
            # Split by H3
            positions = [0] + [m.start() for m in h3_matches] + [len(content)]
            for i in range(len(positions) - 1):
                chunk_content = content[positions[i]:positions[i + 1]].strip()
                if chunk_content:
                    chunks.extend(self._split_by_size(chunk_content))
        else:
            # No H3 headers, split by paragraphs
            chunks = self._split_by_size(content)

        return chunks

    def _split_by_size(self, content: str) -> list[str]:
        """
        Split content by paragraph boundaries to fit size limit.
        """
        max_tokens = self.settings.chunk_max_tokens
        min_tokens = self.settings.chunk_min_tokens

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', content)

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = int(len(para.split()) * 0.75)  # Approximate tokens

            # Check if this is a code block - keep it intact
            is_code_block = para.strip().startswith('```')

            if is_code_block:
                # If current chunk exists, save it first
                if current_chunk and current_size >= min_tokens:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Add code block as its own chunk or append to current
                if para_size > max_tokens:
                    # Code block too large, keep anyway (don't split code)
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    chunks.append(para)
                else:
                    current_chunk.append(para)
                    current_size += para_size
            elif current_size + para_size > max_tokens:
                # Would exceed limit, start new chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _create_chunk(
        self,
        doc: ParsedDocument,
        h1_title: str,
        section_header: Optional[str],
        content: str,
        chunk_index: int,
    ) -> Chunk:
        """
        Create a Chunk object from content.
        """
        # Generate URL
        url = self._generate_url(doc.file_path, section_header)

        # Check for code
        has_code = '```' in content
        code_language = None
        if has_code:
            lang_match = re.search(r'```(\w+)', content)
            if lang_match:
                code_language = lang_match.group(1)

        # Determine doc type
        doc_type = self._classify_doc_type(doc.file_path, content)

        # Format content with title context
        if section_header:
            formatted_content = f"{h1_title} > {section_header}\n\n{content}"
        else:
            formatted_content = f"{h1_title}\n\n{content}"

        # Generate unique ID
        chunk_id = self._generate_chunk_id(doc.file_path, chunk_index)

        # Build metadata
        metadata = {
            "file_path": doc.file_path,
            "frontmatter": doc.frontmatter,
            "keywords": doc.keywords,
            "sidebar_position": doc.frontmatter.get("sidebar_position", 999),
        }

        return Chunk(
            id=chunk_id,
            content=formatted_content,
            title=h1_title,
            section=section_header,
            url=url,
            file_path=doc.file_path,
            doc_type=doc_type,
            has_code=has_code,
            code_language=code_language,
            word_count=len(content.split()),
            chunk_index=chunk_index,
            metadata=metadata,
        )

    def _generate_url(
        self,
        file_path: str,
        section_header: Optional[str],
    ) -> str:
        """
        Generate documentation URL from file path.

        Example:
        docs/api/rest-variables.md + "Creating" ->
        https://docs.wavemaker.com/learn/api/rest-variables#creating
        """
        # Extract relative path from docs folder
        parts = file_path.replace('\\', '/').split('/')
        try:
            docs_idx = parts.index('docs')
            # Get path after 'docs'
            relative_parts = parts[docs_idx + 1:]
        except ValueError:
            # No 'docs' folder found, use full path
            relative_parts = parts

        # Remove .md or .mdx extension
        if relative_parts:
            relative_parts[-1] = re.sub(r'\.(md|mdx)$', '', relative_parts[-1])

        # Join path
        url_path = '/'.join(relative_parts)

        # Build full URL
        url = f"{self._docs_base_url}/{url_path}"

        # Add section anchor
        if section_header:
            anchor = self._slugify(section_header)
            url = f"{url}#{anchor}"

        return url

    def _slugify(self, text: str) -> str:
        """Convert text to URL-safe slug."""
        # Lowercase
        slug = text.lower()
        # Replace spaces with hyphens
        slug = slug.replace(' ', '-')
        # Remove special characters
        slug = re.sub(r'[^a-z0-9-]', '', slug)
        # Collapse multiple hyphens
        slug = re.sub(r'-+', '-', slug)
        # Trim hyphens
        slug = slug.strip('-')
        return slug

    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """Generate unique chunk ID as UUID (required by Qdrant)."""
        import uuid
        # Create a deterministic UUID based on file path and chunk index
        namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace
        unique_string = f"{file_path}:{chunk_index}"
        return str(uuid.uuid5(namespace, unique_string))

    def _classify_doc_type(self, file_path: str, content: str) -> str:
        """
        Classify document type based on path and content.
        """
        path_lower = file_path.lower()

        # Path-based classification
        if '/api/' in path_lower or '/apis-' in path_lower:
            return 'reference'
        if '/getting-started/' in path_lower or '/tutorial' in path_lower:
            return 'tutorial'
        if '/troubleshoot' in path_lower or '/faq' in path_lower:
            return 'troubleshoot'
        if '/concept' in path_lower or '/overview' in path_lower:
            return 'concept'

        # Content-based classification
        content_lower = content.lower()
        if re.search(r'^\d+\.\s', content, re.MULTILINE):
            # Has numbered steps
            return 'how-to'
        if 'properties' in content_lower and '|' in content:
            # Has property table
            return 'reference'
        if 'error' in content_lower and 'fix' in content_lower:
            return 'troubleshoot'

        return 'general'
