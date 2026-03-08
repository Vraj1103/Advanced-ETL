import tiktoken
import json
from typing import List, Dict, Tuple

import config


class PDFChunkingService:
    """Standalone PDF chunking service with table support"""
    
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.buffer_size = config.BUFFER_SIZE
        self.encoding_name = "cl100k_base"

    def chunk_pdf(self, extracted_data: dict, file_name: str, file_id: str) -> List[Tuple[str, Dict]]:
        """
        Chunk extracted PDF data into processable chunks
        
        Args:
            extracted_data: Output from extraction service
            file_name: Name of the PDF file
            file_id: Unique identifier for the file
            
        Returns:
            List of (chunk_text, metadata) tuples
        """
        para_data = extracted_data.get('paragraphs', [])
        tables_data = extracted_data.get('tables', [])
        pages_data = extracted_data.get('pages', [])
        
        # Create text chunks from paragraphs
        text_chunks = self._create_text_chunks(
            para_data=para_data,
            tables_data=tables_data,
            pages_data=pages_data,
            file_name=file_name,
            file_id=file_id
        )
        
        # Create table chunks
        table_chunks = self._create_table_chunks(
            tables_data=tables_data,
            para_data=para_data,
            pages_data=pages_data,
            file_name=file_name,
            file_id=file_id
        )
        
        # Combine all chunks
        all_chunks = text_chunks + table_chunks
        
        # Convert metadata to string format for storage
        final_chunks = [
            (text, {
                'page_number': str(details.get('page_number', [])),
                'bounding_box': json.dumps(details.get('bounding_box', {})),
                'file_name': str(details.get('file_name', '')),
                'file_id': str(details.get('file_id', '')),
                'page_info': str(details.get('page_info', {}))
            }) for text, details in all_chunks
        ]
        
        return final_chunks

    def _num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string"""
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def _create_text_chunks(self, para_data, tables_data, pages_data, file_name, file_id) -> List[Tuple[str, Dict]]:
        """Create chunks from paragraph data"""
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_metadata = {
            'page_number': [],
            'bounding_box': {},
            'file_name': file_name,
            'file_id': file_id,
            'page_info': {}
        }
        
        for para in para_data:
            content = para.get('content', '').strip()
            if not content:
                continue
            
            tokens = self._num_tokens_from_string(content)
            page_num = None
            
            if 'bounding_regions' in para and para['bounding_regions']:
                page_num = para['bounding_regions'][0].get('page_number')
            
            # Check if adding this paragraph would exceed chunk size
            if current_tokens + tokens > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append((chunk_text, current_metadata.copy()))
                
                # Reset for next chunk
                current_chunk = []
                current_tokens = 0
                current_metadata = {
                    'page_number': [],
                    'bounding_box': {},
                    'file_name': file_name,
                    'file_id': file_id,
                    'page_info': {}
                }
            
            # Add paragraph to current chunk
            current_chunk.append(content)
            current_tokens += tokens
            
            if page_num and page_num not in current_metadata['page_number']:
                current_metadata['page_number'].append(page_num)
            
            # Extract bounding box coordinates
            if 'bounding_regions' in para and para['bounding_regions']:
                for region in para['bounding_regions']:
                    page = region.get('page_number')
                    polygon = region.get('polygon', [])
                    if page and polygon:
                        if page not in current_metadata['bounding_box']:
                            current_metadata['bounding_box'][page] = []
                        current_metadata['bounding_box'][page].append(polygon)
        
        # Add final chunk if exists
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append((chunk_text, current_metadata))
        
        return chunks

    def _create_table_chunks(self, tables_data, para_data, pages_data, file_name, file_id) -> List[Tuple[str, Dict]]:
        """Create chunks from table data with HTML formatting"""
        chunks = []
        
        for table in tables_data:
            # Convert table to HTML
            html_content = self._table_to_html(table)
            
            # Extract metadata
            page_num = None
            bounding_boxes = {}
            if 'bounding_regions' in table and table['bounding_regions']:
                page_num = table['bounding_regions'][0].get('page_number')
                # Extract bounding box coordinates for tables
                for region in table['bounding_regions']:
                    page = region.get('page_number')
                    polygon = region.get('polygon', [])
                    if page and polygon:
                        if page not in bounding_boxes:
                            bounding_boxes[page] = []
                        bounding_boxes[page].append(polygon)
            
            # Find context (section heading, etc.)
            context = self._find_table_context(table, para_data)
            
            # Combine context and table
            full_content = f"file_name: {file_name}\n"
            if page_num:
                full_content += f"page_number: [{page_num}]\n"
            if context:
                full_content += f"{context}\n"
            full_content += html_content
            
            metadata = {
                'page_number': [page_num] if page_num else [],
                'bounding_box': bounding_boxes,
                'file_name': file_name,
                'file_id': file_id,
                'page_info': {}
            }
            
            chunks.append((full_content, metadata))
        
        return chunks

    def _table_to_html(self, table: dict) -> str:
        """Convert table data to HTML format"""
        html = '<table border="1">\n'
        
        rows = table.get('cells', [])
        if not rows:
            return '<table></table>'
        
        # Group cells by row
        max_row = max(cell.get('row_index', 0) for cell in rows) + 1
        max_col = max(cell.get('column_index', 0) for cell in rows) + 1
        
        # Create a matrix
        matrix = [[None for _ in range(max_col)] for _ in range(max_row)]
        
        for cell in rows:
            row_idx = cell.get('row_index', 0)
            col_idx = cell.get('column_index', 0)
            content = cell.get('content', '')
            is_header = cell.get('kind', '') == 'columnHeader'
            
            matrix[row_idx][col_idx] = {
                'content': content,
                'is_header': is_header
            }
        
        # Convert matrix to HTML
        for row in matrix:
            html += '<tr>'
            for cell in row:
                if cell:
                    tag = 'th' if cell['is_header'] else 'td'
                    html += f'<{tag}>{cell["content"]}</{tag}>'
                else:
                    html += '<td></td>'
            html += '</tr>\n'
        
        html += '</table>'
        return html

    def _find_table_context(self, table: dict, para_data: list) -> str:
        """Find contextual information for a table (section headings, etc.)"""
        table_offset = None
        if 'spans' in table and table['spans']:
            table_offset = table['spans'][0].get('offset')
        
        if not table_offset:
            return ""
        
        # Find closest section heading before the table
        context = ""
        for para in para_data:
            para_offset = None
            if 'spans' in para and para['spans']:
                para_offset = para['spans'][0].get('offset')
            
            if para_offset and para_offset < table_offset:
                role = para.get('role', '')
                if role in ['title', 'sectionHeading']:
                    context = para.get('content', '')
        
        return context
