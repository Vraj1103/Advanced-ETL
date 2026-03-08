import time
import uuid
from typing import Dict

from extraction_service import PDFExtractionService
from chunking_service import PDFChunkingService
from vector_service import VectorStorageService
from structured_storage_service import StructuredStorageService
from fact_extractor import FactExtractor
from table_extractor import TableExtractor


class StandaloneECEProcessor:
    """
    Standalone Extract-Chunk-Embed processor with DUAL STORAGE
    - Vector DB (Azure Search): For semantic search and discovery
    - PostgreSQL: For structured facts and tables with accurate queries
    """
    
    def __init__(self):
        self.extraction_service = None
        self.chunking_service = PDFChunkingService()
        self.vector_service = None
        self.structured_storage = None
        self.fact_extractor = FactExtractor()
        self.table_extractor = TableExtractor()

    async def process_pdf(
        self,
        file_content: bytes,
        file_name: str,
        namespace: str = "default"
    ) -> Dict:
        """
        Process a PDF file through the complete ECE pipeline
        
        Args:
            file_content: PDF file content as bytes
            file_name: Name of the PDF file
            namespace: Namespace for vector storage (e.g., project name)
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        file_id = str(uuid.uuid4())
        
        result = {
            "status": "success",
            "file_name": file_name,
            "file_id": file_id,
            "namespace": namespace,
            "stages": {},
            "errors": []
        }
        
        try:
            # Stage 1: Extract
            print(f"[EXTRACT] Starting extraction for {file_name}")
            extract_start = time.time()
            
            self.extraction_service = PDFExtractionService()
            extracted_data = await self.extraction_service.extract_from_pdf(file_content)
            
            extract_time = time.time() - extract_start
            result["stages"]["extract"] = {
                "status": "success",
                "time_seconds": round(extract_time, 2),
                "pages": len(extracted_data.get('pages', [])),
                "paragraphs": len(extracted_data.get('paragraphs', [])),
                "tables": len(extracted_data.get('tables', []))
            }
            print(f"[EXTRACT] Completed in {extract_time:.2f}s - {result['stages']['extract']['pages']} pages, {result['stages']['extract']['tables']} tables")
            
            # Stage 2: Chunk
            print(f"[CHUNK] Starting chunking for {file_name}")
            chunk_start = time.time()
            
            chunks = self.chunking_service.chunk_pdf(
                extracted_data=extracted_data,
                file_name=file_name,
                file_id=file_id
            )
            
            chunk_time = time.time() - chunk_start
            result["stages"]["chunk"] = {
                "status": "success",
                "time_seconds": round(chunk_time, 2),
                "chunk_count": len(chunks)
            }
            print(f"[CHUNK] Completed in {chunk_time:.2f}s - {len(chunks)} chunks created")
            
            # Stage 3: Embed & Store in Vector DB
            print(f"[EMBED] Starting embedding and vector storage for {file_name}")
            embed_start = time.time()
            
            self.vector_service = VectorStorageService()
            await self.vector_service.initialize()
            
            upsert_result = await self.vector_service.upsert_chunks(
                chunks=chunks,
                namespace=namespace,
                file_name=file_name,
                file_id=file_id
            )
            
            embed_time = time.time() - embed_start
            result["stages"]["embed"] = {
                "status": upsert_result["status"],
                "time_seconds": round(embed_time, 2),
                "total_chunks": upsert_result["total_chunks"],
                "success_count": upsert_result["success_count"],
                "failed_count": upsert_result.get("failed_count", 0)
            }
            print(f"[EMBED] Completed in {embed_time:.2f}s - {upsert_result['success_count']}/{upsert_result['total_chunks']} chunks stored")
            
            # Stage 4: Extract & Store Structured Data (DUAL STORAGE)
            print(f"[STRUCTURED] Extracting facts and tables for MongoDB storage")
            structured_start = time.time()
            
            self.structured_storage = StructuredStorageService()
            await self.structured_storage.initialize()
            
            # Extract facts
            facts = self.fact_extractor.extract_facts(extracted_data, file_name)
            facts_result = await self.structured_storage.store_facts(
                facts=facts,
                file_id=file_id,
                file_name=file_name,
                namespace=namespace
            )
            
            # Extract tables
            tables = self.table_extractor.extract_tables(extracted_data, file_name)
            tables_result = await self.structured_storage.store_tables(
                tables=tables,
                file_id=file_id,
                file_name=file_name,
                namespace=namespace
            )
            
            structured_time = time.time() - structured_start
            result["stages"]["structured"] = {
                "status": "success",
                "time_seconds": round(structured_time, 2),
                "facts_stored": facts_result["stored_count"],
                "tables_stored": tables_result["stored_count"]
            }
            print(f"[STRUCTURED] Completed in {structured_time:.2f}s - {facts_result['stored_count']} facts, {tables_result['stored_count']} tables stored")
            
            # Overall stats
            total_time = time.time() - start_time
            result["total_time_seconds"] = round(total_time, 2)
            result["summary"] = {
                "pages_processed": result["stages"]["extract"]["pages"],
                "tables_extracted": result["stages"]["extract"]["tables"],
                "chunks_created": result["stages"]["chunk"]["chunk_count"],
                "chunks_stored_vector_db": result["stages"]["embed"]["success_count"],
                "facts_stored_mongodb": result["stages"]["structured"]["facts_stored"],
                "tables_stored_mongodb": result["stages"]["structured"]["tables_stored"]
            }
            
            if result["stages"]["embed"]["failed_count"] > 0:
                result["status"] = "partial_success"
                result["errors"].append(f"{result['stages']['embed']['failed_count']} chunks failed to store")
            
            print(f"\n✅ Pipeline completed successfully in {total_time:.2f}s")
            print(f"📊 Summary: {result['summary']}")
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
            print(f"\n❌ Pipeline failed: {str(e)}")
            
        finally:
            # Cleanup
            if self.extraction_service:
                await self.extraction_service.close()
            if self.structured_storage:
                await self.structured_storage.close()
            if self.vector_service:
                await self.vector_service.close()
        
        return result

    async def search(
        self,
        query: str,
        namespace: str = None,
        file_id: str = None,
        top_k: int = 10
    ) -> Dict:
        """
        Search for relevant content in the vector database
        
        Args:
            query: Search query
            namespace: Optional namespace filter
            file_id: Optional file ID filter
            top_k: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            self.vector_service = VectorStorageService()
            await self.vector_service.initialize()
            
            results = await self.vector_service.search(
                query=query,
                namespace=namespace,
                file_id=file_id,
                top_k=top_k
            )
            
            await self.vector_service.close()
            
            return {
                "status": "success",
                "query": query,
                "result_count": len(results),
                "results": results
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "query": query,
                "error": str(e),
                "results": []
            }
