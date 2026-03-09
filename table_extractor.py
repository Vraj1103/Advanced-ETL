"""
Table extraction service - converts AFR tables to structured format
Part of dual storage architecture
"""
from typing import Dict, List


class TableExtractor:
    """Extract and structure tables from AFR output"""
    
    def extract_tables(
        self, 
        extracted_data: Dict, 
        file_name: str
    ) -> List[Dict]:
        """
        Convert AFR table output to structured format for PostgreSQL
        
        Args:
            extracted_data: Output from AFR extraction service
            file_name: Name of source file
            
        Returns:
            List of structured table dictionaries
        """
        tables = extracted_data.get('tables', [])
        structured_tables = []
        
        for idx, table in enumerate(tables):
            structured_table = self._structure_table(table, idx, file_name)
            if structured_table:
                structured_tables.append(structured_table)
        
        return structured_tables
    
    def _structure_table(self, table: Dict, table_index: int, file_name: str) -> Dict:
        """
        Convert a single AFR table to structured format
        
        Args:
            table: AFR table dictionary
            table_index: Index of this table in document
            file_name: Name of source file
            
        Returns:
            Structured table dictionary
        """
        cells = table.get('cells', [])
        
        if not cells:
            return None
        
        # Get table dimensions
        max_row = max(cell.get('row_index', 0) for cell in cells) + 1
        max_col = max(cell.get('column_index', 0) for cell in cells) + 1
        
        # Create matrix
        matrix = [[None for _ in range(max_col)] for _ in range(max_row)]
        
        for cell in cells:
            row_idx = cell.get('row_index', 0)
            col_idx = cell.get('column_index', 0)
            content = cell.get('content', '')
            is_header = cell.get('kind', '') == 'columnHeader'
            
            matrix[row_idx][col_idx] = {
                'content': content,
                'is_header': is_header
            }
        
        # Extract headers (first row with is_header=True)
        headers = []
        header_row_idx = 0
        for row_idx, row in enumerate(matrix):
            if any(cell and cell.get('is_header') for cell in row if cell):
                headers = [cell['content'] if cell else f"Column_{i}" 
                          for i, cell in enumerate(row)]
                header_row_idx = row_idx
                break
        
        # If no headers found, use default names
        if not headers:
            headers = [f"Column_{i}" for i in range(max_col)]
            header_row_idx = -1
        
        # Extract data rows (skip header row)
        data_rows = []
        for row_idx in range(header_row_idx + 1, max_row):
            row = matrix[row_idx]
            row_dict = {}
            
            for col_idx, cell in enumerate(row):
                if col_idx < len(headers):
                    # Try to convert to numeric if possible
                    content = cell['content'] if cell else ""
                    value = self._parse_value(content)
                    row_dict[headers[col_idx]] = value
            
            if any(v for v in row_dict.values()):  # Only add non-empty rows
                data_rows.append(row_dict)
        
        # Get page number
        page_number = None
        if 'bounding_regions' in table and table['bounding_regions']:
            page_number = table['bounding_regions'][0].get('page_number')
        
        # Infer column types
        column_types = self._infer_column_types(data_rows, headers)
        
        # Generate table ID
        table_id = f"table_{table_index}_{file_name.replace('.', '_')}"
        
        return {
            "table_id": table_id,
            "page": page_number,
            "file_name": file_name,
            "headers": [
                {
                    "name": header,
                    "type": column_types.get(header, "string"),
                    "index": idx
                }
                for idx, header in enumerate(headers)
            ],
            "data": data_rows,
            "row_count": len(data_rows),
            "column_count": len(headers),
            "metadata": {
                "table_index": table_index,
                "has_headers": header_row_idx >= 0
            }
        }
    
    def _parse_value(self, value_str: str):
        """
        Parse string value to appropriate type (numeric or string)
        
        Args:
            value_str: String value from table cell
            
        Returns:
            Parsed value (int, float, or string)
        """
        if not value_str or not isinstance(value_str, str):
            return value_str
        
        # Remove common formatting
        cleaned = value_str.replace(',', '').replace('€', '').replace('%', '').strip()
        
        # Try to convert to number
        try:
            # Check if it's an integer
            if '.' not in cleaned:
                return int(cleaned)
            else:
                return float(cleaned)
        except ValueError:
            # Return as string if not numeric
            return value_str
    
    def _infer_column_types(self, data_rows: List[Dict], headers: List[str]) -> Dict[str, str]:
        """
        Infer data types for each column
        
        Args:
            data_rows: List of row dictionaries
            headers: List of column names
            
        Returns:
            Dictionary mapping header names to types
        """
        column_types = {}
        
        for header in headers:
            # Check first non-empty value in column
            for row in data_rows:
                value = row.get(header)
                if value is not None and value != "":
                    if isinstance(value, int):
                        column_types[header] = "integer"
                    elif isinstance(value, float):
                        column_types[header] = "numeric"
                    else:
                        column_types[header] = "string"
                    break
            
            # Default to string if no values found
            if header not in column_types:
                column_types[header] = "string"
        
        return column_types
