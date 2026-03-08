"""
Fact extraction service - extracts structured facts from documents
Part of dual storage architecture
"""
import re
from typing import Dict, List


class FactExtractor:
    """Extract structured facts from extracted document data"""
    
    def __init__(self):
        # Patterns for identifying facts
        self.numeric_patterns = [
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:jobs?|positions?|roles?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:firms?|companies?|organizations?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:percent|%)',
            r'€?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand)',
        ]
        
        self.entity_keywords = {
            "job_count": ["jobs", "positions", "roles", "employment", "workforce"],
            "firm_count": ["firms", "companies", "organizations", "enterprises"],
            "percentage": ["percent", "%", "proportion"],
            "revenue": ["revenue", "turnover", "sales", "€", "million", "billion"],
            "growth_rate": ["growth", "increase", "CAGR", "compound annual"]
        }
    
    def extract_facts(self, extracted_data: Dict, file_name: str) -> List[Dict]:
        """
        Extract structured facts from document
        
        Args:
            extracted_data: Output from extraction service
            file_name: Name of source file
            
        Returns:
            List of fact dictionaries
        """
        facts = []
        paragraphs = extracted_data.get('paragraphs', [])
        
        for para in paragraphs:
            content = para.get('content', '')
            page_number = None
            
            if 'bounding_regions' in para and para['bounding_regions']:
                page_number = para['bounding_regions'][0].get('page_number')
            
            # Extract numeric facts
            extracted_facts = self._extract_numeric_facts(content, page_number, file_name)
            facts.extend(extracted_facts)
        
        # Deduplicate and rank by confidence
        facts = self._deduplicate_facts(facts)
        
        return facts
    
    def _extract_numeric_facts(self, text: str, page: int, file_name: str) -> List[Dict]:
        """Extract numeric facts from text"""
        facts = []
        
        # Look for numbers with context
        for entity_type, keywords in self.entity_keywords.items():
            for keyword in keywords:
                # Create pattern to find numbers near keywords
                pattern = rf'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:\w+\s+){{0,3}}{re.escape(keyword)}'
                pattern_reverse = rf'{re.escape(keyword)}\s+(?:\w+\s+){{0,3}}(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value_str = match.group(1).replace(',', '')
                    try:
                        value = float(value_str)
                        
                        # Extract context (surrounding text)
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end].strip()
                        
                        facts.append({
                            "entity_type": entity_type,
                            "value": value,
                            "value_string": match.group(1),
                            "page": page,
                            "source_quote": context,
                            "confidence": 0.8,  # Base confidence
                            "context": context,
                            "file_name": file_name
                        })
                    except ValueError:
                        continue
                
                # Check reverse pattern
                for match in re.finditer(pattern_reverse, text, re.IGNORECASE):
                    value_str = match.group(1).replace(',', '')
                    try:
                        value = float(value_str)
                        
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end].strip()
                        
                        facts.append({
                            "entity_type": entity_type,
                            "value": value,
                            "value_string": match.group(1),
                            "page": page,
                            "source_quote": context,
                            "confidence": 0.75,  # Slightly lower confidence for reverse pattern
                            "context": context,
                            "file_name": file_name
                        })
                    except ValueError:
                        continue
        
        return facts
    
    def _deduplicate_facts(self, facts: List[Dict]) -> List[Dict]:
        """Remove duplicate facts, keeping highest confidence"""
        seen = {}
        
        for fact in facts:
            key = (fact['entity_type'], fact['value'], fact['page'])
            
            if key not in seen or fact['confidence'] > seen[key]['confidence']:
                seen[key] = fact
        
        return list(seen.values())
