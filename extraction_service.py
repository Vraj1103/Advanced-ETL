import asyncio
from pathlib import Path
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.formrecognizer.aio import DocumentAnalysisClient

import config
from llm_middleware import LLMMiddleware


class PDFExtractionService:
    """Standalone PDF extraction service using Azure Document Intelligence"""
    
    def __init__(self):
        self.credential = AzureKeyCredential(config.AZURE_AFR_API_KEY)
        self.document_analysis_client = DocumentAnalysisClient(
            config.AZURE_AFR_ENDPOINT, 
            self.credential
        )
        self.llm_middleware = LLMMiddleware()
        self.client = self.llm_middleware.initialize_client()

    async def extract_from_pdf(self, document_bytes: bytes) -> dict:
        """
        Extract text, tables, and layout from PDF using Azure Form Recognizer
        
        Args:
            document_bytes: PDF file content as bytes
            
        Returns:
            Extracted data dictionary with paragraphs, tables, pages
        """
        try:
            enable_timeout = config.TIMEOUT_ENABLED
            afr_timeout = config.AFR_TIMEOUT_SECS
            
            poller = await self.document_analysis_client.begin_analyze_document(
                "prebuilt-layout", 
                document=document_bytes
            )
            
            result = await asyncio.wait_for(
                poller.result(),
                timeout=afr_timeout
            ) if enable_timeout else await poller.result()
            
            extracted_data = result.to_dict()
            
            # Process and structure the data
            paragraphs, tables, page_numbers = self._extract_paragraphs_and_tables(extracted_data)
            updated_paragraphs = self._remove_duplicate_tables_from_paragraphs(paragraphs, tables)
            structured_data = self._update_json_structure(extracted_data, updated_paragraphs)
            
            return structured_data
            
        except asyncio.TimeoutError as e:
            raise Exception(f"Azure Form Recognizer timeout: {str(e)}")
        except HttpResponseError as e:
            if e.status_code == 429:
                raise Exception(f"Azure Form Recognizer rate limit: {str(e)}")
            else:
                raise Exception(f"Azure Form Recognizer HTTP error: {str(e)}")
        except Exception as e:
            raise Exception(f"Extraction error: {str(e)}")

    def _extract_paragraphs_and_tables(self, extracted_data):
        """Extract and organize paragraphs and tables from AFR output"""
        paragraphs = extracted_data.get('paragraphs', [])
        tables = extracted_data.get('tables', [])
        # Fallback: some SDK/API shapes put paragraphs and tables inside documents[]
        if not paragraphs or not tables:
            for doc in extracted_data.get('documents', []):
                if isinstance(doc, dict):
                    if not paragraphs:
                        paragraphs = doc.get('paragraphs', [])
                    if not tables:
                        tables = doc.get('tables', [])
                if paragraphs and tables:
                    break
        page_numbers = set()
        
        for para in paragraphs:
            if 'bounding_regions' in para and para['bounding_regions']:
                page_num = para['bounding_regions'][0].get('page_number')
                if page_num:
                    page_numbers.add(page_num)
        
        return paragraphs, tables, sorted(list(page_numbers))

    def _remove_duplicate_tables_from_paragraphs(self, paragraphs, tables):
        """Remove table content that appears in paragraphs"""
        updated_paragraphs = []
        
        for para in paragraphs:
            is_duplicate = False
            para_content = para.get('content', '').strip()
            
            for table in tables:
                table_content = table.get('content', '').strip()
                if para_content and table_content and para_content in table_content:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                updated_paragraphs.append(para)
        
        return updated_paragraphs

    def _update_json_structure(self, extracted_data, updated_paragraphs):
        """Update the JSON structure with processed paragraphs"""
        result = extracted_data.copy()
        result['paragraphs'] = updated_paragraphs
        return result

    async def close(self):
        """Close the document analysis client"""
        await self.document_analysis_client.close()
