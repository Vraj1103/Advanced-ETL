import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

import config
from ece_processor import StandaloneECEProcessor
from langgraph_agent import LangGraphDualStorageAgent


app = FastAPI(
    title="Standalone ECE Pipeline",
    description="Independent Extract-Chunk-Embed pipeline for PDF processing",
    version="1.0.0"
)

processor = StandaloneECEProcessor()
agent = LangGraphDualStorageAgent(default_namespace="default")


class SearchRequest(BaseModel):
    query: str
    namespace: Optional[str] = None
    file_id: Optional[str] = None
    top_k: Optional[int] = 10


class AgentQueryRequest(BaseModel):
    query: str
    namespace: Optional[str] = None
    max_steps: Optional[int] = 8
    debug: Optional[bool] = False


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Standalone ECE Pipeline",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /upload": "Upload and process PDF file",
            "POST /search": "Search processed documents",
            "POST /agent/query": "Run LangGraph agent query",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "standalone-ece-pipeline"
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to process"),
    namespace: str = Form(default="default", description="Namespace for organizing documents")
):
    """
    Upload and process a PDF file through the ECE pipeline
    
    - **file**: PDF file to process
    - **namespace**: Namespace for organizing documents (e.g., project name)
    
    Returns processing results including:
    - Extraction statistics (pages, tables, paragraphs)
    - Chunking statistics (number of chunks)
    - Embedding statistics (chunks stored)
    - Processing times for each stage
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Empty file provided"
            )
        
        # Process through ECE pipeline
        result = await processor.process_pdf(
            file_content=file_content,
            file_name=file.filename,
            namespace=namespace
        )
        
        # Return appropriate status code
        status_code = 200 if result["status"] == "success" else 207 if result["status"] == "partial_success" else 500
        
        return JSONResponse(
            content=result,
            status_code=status_code
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/search")
async def search_documents(request: SearchRequest):
    """
    Search for relevant content in processed documents
    
    - **query**: Search query text
    - **namespace**: Optional namespace filter
    - **file_id**: Optional file ID filter
    - **top_k**: Number of results to return (default: 10)
    
    Returns relevant chunks with:
    - Content text
    - File name
    - Page number
    - Bounding box coordinates
    - Relevance score
    """
    try:
        result = await processor.search(
            query=request.query,
            namespace=request.namespace,
            file_id=request.file_id,
            top_k=request.top_k
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )


@app.post("/agent/query")
async def agent_query(request: AgentQueryRequest):
    """
    Query the LangGraph dual-storage agent.

    - **query**: Natural language question
    - **namespace**: Optional namespace filter
    - **max_steps**: Max tool-usage iterations (default: 8)
    - **debug**: Include tool calls and intermediate tool results in response

    Returns:
    - Final grounded answer
    - Number of graph/tool steps executed
    - Namespace used
    """
    try:
        result = await agent.ainvoke(
            user_query=request.query,
            namespace=request.namespace,
            max_steps=request.max_steps or 8,
            debug=bool(request.debug)
        )

        response = {
            "status": "success",
            "query": request.query,
            "namespace": result.get("namespace"),
            "steps": result.get("steps", 0),
            "answer": result.get("answer", "")
        }

        if request.debug:
            response["trace"] = result.get("trace", {})

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running agent query: {str(e)}"
        )


@app.get("/config")
async def get_config():
    """Get current configuration (sanitized, no secrets)"""
    return {
        "chunk_size": config.CHUNK_SIZE,
        "buffer_size": config.BUFFER_SIZE,
        "embedding_model": config.EMBEDDING_MODEL,
        "vector_dimensions": config.VECTOR_SEARCH_DIMENSIONS,
        "llm_vendor": config.ACTIVE_LLM_VENDOR,
        "index_name": config.AZURE_INDEX_NAME,
        "timeout_enabled": config.TIMEOUT_ENABLED,
        "afr_timeout_seconds": config.AFR_TIMEOUT_SECS
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Close async services on server shutdown."""
    await agent.aclose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )
