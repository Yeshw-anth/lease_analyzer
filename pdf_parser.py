"""
This module is responsible for parsing PDF lease documents.

It performs the following key functions:
1.  Extracts raw text from each page of a PDF.
2.  Identifies logical document sections based on patterns like "ARTICLE X" and "Section X.X".
3.  Chunks the document by these logical sections, preserving the contextual integrity
    of each clause.
4.  Stores each chunk as a structured `LeaseChunk` object containing the text,
    page number, and any identified article/section numbers.
"""
import logging
import re
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



@dataclass
class LeaseChunk:
    """
    Represents a single logical chunk of the lease, typically a section.

    Attributes:
        chunk_id (int): A unique sequential identifier for the chunk.
        page_number (int): The page number in the PDF where the chunk begins.
        raw_text (str): The full, unprocessed text content of the chunk.
        article_number (Optional[str]): The article number (e.g., "V") if identified.
        section_number (Optional[str]): The section number (e.g., "5.1") if identified.
    """
    chunk_id: int
    page_number: int
    raw_text: str
    article_number: Optional[str] = None
    section_number: Optional[str] = None


def _extract_text_with_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extracts text from each page of a PDF document using PyMuPDF (fitz).

    Args:
        pdf_path: The file path to the PDF.

    Returns:
        A list of tuples, where each tuple contains the page number (1-indexed)
        and the full text content of that page.
    """
    logging.info(f"Starting text extraction from '{pdf_path}' using fitz.")
    text_with_pages = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text_with_pages.append((page_num, page.get_text()))
    logging.info(f"Successfully extracted {len(text_with_pages)} pages.")
    return text_with_pages


def parse_lease_pdf(pdf_path: str) -> List[LeaseChunk]:
    """
    Parses a lease PDF, chunking it by article and section.

    This function orchestrates the PDF parsing process:
    1. It extracts text and page numbers.
    2. It iterates through the text, identifying ARTICLE and Section headings.
    3. It groups text under the appropriate section, creating a `LeaseChunk` for each.

    Args:
        pdf_path: The file path to the lease PDF.

    Returns:
        A list of `LeaseChunk` objects, representing the structured,
        semantically chunked document.
    """
    logging.info(f"Starting PDF parsing and chunking for '{pdf_path}'.")
    text_with_pages = _extract_text_with_pages(pdf_path)
    full_text = "".join([text for _, text in text_with_pages])

    # Regex to find sections. This is a crucial part of the logic.
    # It looks for "ARTICLE [roman/number]" or "Section [number.number]"
    # The (?i) flag makes it case-insensitive.
    # The \s* handles optional whitespace.
    section_pattern = re.compile(
        r"(?i)(ARTICLE\s+[IVXLCDM\d]+|Section\s+\d+\.\d+)"
    )
    
    # Find all matches of the section pattern throughout the document
    matches = list(section_pattern.finditer(full_text))
    logging.info(f"Found {len(matches)} potential section headers.")
    
    chunks: List[LeaseChunk] = []
    chunk_id_counter = 0
    current_article = None

    # Handle text before the first match (e.g., cover page, TOC)
    if matches:
        first_match_start = matches[0].start()
        if first_match_start > 0:
            preamble_text = full_text[:first_match_start].strip()
            if preamble_text:
                page_num = _find_page_for_position(text_with_pages, 0)
                chunks.append(LeaseChunk(
                    chunk_id=chunk_id_counter,
                    page_number=page_num,
                    raw_text=preamble_text
                ))
                chunk_id_counter += 1
    else:
        # If no sections found, treat the whole document as one chunk
        logging.warning("No section headers found. Treating the entire document as a single chunk.")
        if full_text.strip():
            chunks.append(LeaseChunk(
                chunk_id=0,
                page_number=1,
                raw_text=full_text.strip()
            ))
        return chunks

    # Process each identified section
    for i, match in enumerate(matches):
        header = match.group(1).strip()
        
        # Determine if the header is an ARTICLE or Section
        if "article" in header.lower():
            current_article = header.split()[-1]
            is_article_header = True
        else:
            is_article_header = False

        # The text of the chunk is from the end of the current header
        # to the start of the next one.
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        
        chunk_text = full_text[start_pos:end_pos].strip()
        
        # If the header is an article title, we often don't want to create a chunk
        # for the whitespace between "ARTICLE X" and "Section X.1". We only create
        # a chunk if it contains meaningful text.
        if is_article_header and not chunk_text:
            continue

        page_num = _find_page_for_position(text_with_pages, match.start())
        
        section_num = header.split()[-1] if not is_article_header else None

        chunks.append(LeaseChunk(
            chunk_id=chunk_id_counter,
            page_number=page_num,
            article_number=current_article,
            section_number=section_num,
            raw_text=f"{header}\n\n{chunk_text}" # Prepend header for context
        ))
        chunk_id_counter += 1

    logging.info(f"Finished chunking. Created {len(chunks)} chunks.")
    return chunks


def _find_page_for_position(text_with_pages: List[Tuple[int, str]], position: int) -> int:
    """
    Finds the page number corresponding to a character position in the full text.

    Args:
        text_with_pages: A list of (page_number, page_text) tuples.
        position: The character offset in the concatenated full text.

    Returns:
        The 1-indexed page number where the character position is found.
    """
    char_count = 0
    for page_num, text in text_with_pages:
        if char_count + len(text) > position:
            return page_num
        char_count += len(text)
    return len(text_with_pages) # Default to the last page if not found