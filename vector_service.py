import uuid
import asyncio
import json
from typing import List, Tuple, Dict
import backoff
from azure.search.documents.aio import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchIndex,
)

import config
from llm_middleware import LLMMiddleware


class VectorStorageService:
    """Standalone vector storage service using Azure Cognitive Search"""
    
    def __init__(self, index_name: str = None):
        self.index_name = index_name or config.AZURE_INDEX_NAME
        self.azure_search_endpoint = config.AZURE_SEARCH_ENDPOINT
        self.azure_search_key = config.AZURE_SEARCH_KEY
        self.vector_dimensions = config.VECTOR_SEARCH_DIMENSIONS
        
        self.llm_middleware = LLMMiddleware()
        self.client = self.llm_middleware.initialize_client()
        
        self.search_client = None
        self.search_index_client = None
        self._initialized = False

    async def initialize(self):
        """Initialize search clients and ensure index exists"""
        if self._initialized:
            return
        
        credential = AzureKeyCredential(self.azure_search_key)
        
        self.search_client = SearchClient(
            endpoint=self.azure_search_endpoint,
            index_name=self.index_name,
            credential=credential
        )
        
        self.search_index_client = SearchIndexClient(
            endpoint=self.azure_search_endpoint,
            credential=credential
        )
        
        # Ensure index exists
        await self._ensure_index_exists()
        self._initialized = True

    async def _ensure_index_exists(self):
        """Check if index exists, create only if it doesn't"""
        try:
            await self.search_index_client.get_index(self.index_name)
            print(f"✅ Using existing index '{self.index_name}'")
        except Exception as e:
            # If index doesn't exist, try to create it
            # This will fail if quota exceeded, but that's ok if using existing index
            print(f"Index '{self.index_name}' not found, attempting to create...")
            try:
                await self._create_index()
            except Exception as create_error:
                # If creation fails but index already exists (race condition), that's ok
                if "already exists" in str(create_error).lower():
                    print(f"✅ Index '{self.index_name}' already exists")
                else:
                    # Re-raise if it's a different error
                    print(f"⚠️ Could not create index: {create_error}")
                    print(f"Attempting to use existing index '{self.index_name}'...")
                    # Try to use it anyway - may already exist
                    try:
                        await self.search_index_client.get_index(self.index_name)
                        print(f"✅ Successfully connected to existing index")
                    except:
                        raise create_error

    async def _create_index(self):
        """Create the search index with vector support - matching existing schema"""
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True
            ),
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.vector_dimensions,
                vector_search_profile_name="myHnswProfile"
            ),
            SimpleField(
                name="file_id",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SimpleField(
                name="file_name",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SimpleField(
                name="namespace",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SimpleField(
                name="bounding_box",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SimpleField(
                name="page_info",
                type=SearchFieldDataType.String,
                filterable=True
            )
        ]
        
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="hnsw-config"
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(name="hnsw-config")
            ]
        )
        
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        await self.search_index_client.create_index(index)
        print(f"Index '{self.index_name}' created successfully")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            response = await self.client.embeddings.create(
                model=self.llm_middleware.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise

    async def upsert_chunks(
        self, 
        chunks: List[Tuple[str, Dict]], 
        namespace: str,
        file_name: str,
        file_id: str
    ) -> dict:
        """
        Upsert chunks to vector database with embeddings
        
        Args:
            chunks: List of (text, metadata) tuples
            namespace: Namespace for the chunks (e.g., project name)
            file_name: Name of the source file
            file_id: Unique identifier for the file
            
        Returns:
            Status dictionary with success/failure info
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Prepare documents
            documents = []
            texts_to_embed = []
            
            for i, (text, metadata) in enumerate(chunks):
                doc_id = str(uuid.uuid4())
                # Parse page_info to add metadata if present
                page_info_str = metadata.get('page_info', '{}')
                try:
                    page_info = eval(page_info_str) if isinstance(page_info_str, str) else page_info_str
                except:
                    page_info = {}
                
                # Add metadata inside page_info (matching main app structure)
                # Meta fields are stored inside page_info, not as separate fields
                page_info_final = str(page_info)
                
                documents.append({
                    "id": doc_id,
                    "title": metadata.get('file_name', file_name),
                    "content": text,
                    "namespace": namespace,
                    "file_name": metadata.get('file_name', file_name),
                    "file_id": metadata.get('file_id', file_id),
                    "page_number": metadata.get('page_number', '[]'),
                    "bounding_box": metadata.get('bounding_box', '{}'),
                    "page_info": page_info_final
                })
                texts_to_embed.append(text)
            
            # Generate embeddings in batches
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                embeddings = await self.generate_embeddings(batch)
                all_embeddings.extend(embeddings)
                await asyncio.sleep(0.1)  # Rate limiting
            
            # Add embeddings to documents
            for doc, embedding in zip(documents, all_embeddings):
                doc["contentVector"] = embedding
            
            # Upsert to Azure Search
            result = await self.search_client.upload_documents(documents=documents)
            
            success_count = sum(1 for r in result if r.succeeded)
            failed_count = len(result) - success_count
            
            return {
                "status": "success" if failed_count == 0 else "partial",
                "total_chunks": len(chunks),
                "success_count": success_count,
                "failed_count": failed_count
            }
            
        except Exception as e:
            print(f"Error upserting chunks: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "total_chunks": len(chunks),
                "success_count": 0
            }

    async def search(
        self, 
        query: str, 
        namespace: str = None,
        file_id: str = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search for relevant chunks
        
        Args:
            query: Search query
            namespace: Optional namespace filter
            file_id: Optional file ID filter
            top_k: Number of results to return
            
        Returns:
            List of search results with content and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = (await self.generate_embeddings([query]))[0]
            
            # Build filter
            filters = []
            if namespace:
                filters.append(f"namespace eq '{namespace}'")
            if file_id:
                filters.append(f"file_id eq '{file_id}'")
            
            filter_str = " and ".join(filters) if filters else None
            
            # Perform vector search
            results = await self.search_client.search(
                search_text=None,
                vector_queries=[{
                    "kind": "vector",
                    "vector": query_embedding,
                    "fields": "contentVector",
                    "k": top_k
                }],
                filter=filter_str,
                top=top_k
            )
            
            # Format results
            formatted_results = []
            async for result in results:
                # Parse bounding_box JSON string back to dict
                bounding_box = result.get("bounding_box", "{}")
                try:
                    bounding_box = json.loads(bounding_box) if bounding_box else {}
                except (json.JSONDecodeError, TypeError):
                    bounding_box = {}
                # Parse page_number from string representation (e.g. "[1]" or "[1, 2]") to list of ints
                page_number_raw = result.get("page_number") or "[]"
                try:
                    page_numbers = json.loads(page_number_raw) if isinstance(page_number_raw, str) else page_number_raw
                    if not isinstance(page_numbers, list):
                        page_numbers = [page_numbers] if page_numbers is not None else []
                    page_numbers = [int(p) for p in page_numbers if p is not None]
                except (json.JSONDecodeError, TypeError, ValueError):
                    page_numbers = []

                formatted_results.append({
                    "content": result.get("content"),
                    "file_name": result.get("file_name"),
                    "page_number": page_numbers,
                    "bounding_box": bounding_box,
                    "score": result.get("@search.score")
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching: {str(e)}")
            return []

    async def close(self):
        """Close the search clients"""
        if self.search_client:
            await self.search_client.close()
        if self.search_index_client:
            await self.search_index_client.close()
