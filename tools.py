"""
LangGraph Tool Definitions for Dual-Storage RAG System

This module provides tool implementations that can be called by a LangGraph agent.
Each tool is designed to be invoked with simple input/output contracts.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from vector_service import VectorStorageService
from structured_storage_service import StructuredStorageService


# Initialize services globally (will be properly initialized in agent)
vector_service: Optional[VectorStorageService] = None
storage_service: Optional[StructuredStorageService] = None


async def initialize_services(namespace: str = "diligence-ai"):
    """Initialize both storage services"""
    global vector_service, storage_service
    
    if vector_service is None:
        vector_service = VectorStorageService()
        await vector_service.initialize()
    
    if storage_service is None:
        storage_service = StructuredStorageService()
        await storage_service.initialize()


async def close_services():
    """Close initialized services and reset globals."""
    global vector_service, storage_service

    if vector_service is not None:
        await vector_service.close()
        vector_service = None

    if storage_service is not None:
        await storage_service.close()
        storage_service = None


# ============================================================================
# TOOL 1: SEMANTIC_SEARCH
# ============================================================================

async def semantic_search(
    query: str,
    top_k: int = 10,
    file_id: Optional[str] = None,
    namespace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for relevant content chunks using semantic similarity.
    
    This tool finds chunks in the vector database that match the query's semantic meaning,
    regardless of exact keyword matches. Returns chunks with page numbers and bounding boxes.
    
    Args:
        query: The search query describing what to find
        top_k: Number of results to return (default: 10)
        file_id: Optional filter to search within a specific file
        namespace: Optional filter to search within a specific namespace
        
    Returns:
        Dictionary with:
        - status: "success" or "error"
        - results: List of chunks with content, page_number, bounding_box, score
        - result_count: Number of results returned
        - error: Error message if status is "error"
        
    Example:
        >>> results = await semantic_search("job market growth", top_k=5)
        >>> for chunk in results['results']:
        ...     print(f"Page {chunk['page_number']}: {chunk['content'][:100]}...")
    """
    if vector_service is None:
        await initialize_services(namespace or "diligence-ai")
    
    try:
        search_results = await vector_service.search(
            query=query,
            top_k=top_k,
            file_id=file_id,
            namespace=namespace
        )
        
        return {
            "status": "success",
            "results": search_results,
            "result_count": len(search_results),
            "query": query
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "results": [],
            "result_count": 0
        }


# ============================================================================
# TOOL 2: QUERY_TABLE
# ============================================================================

async def query_table(
    table_id: str,
    column_filters: Optional[Dict[str, Any]] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Query a specific table with optional column filters.
    
    This tool retrieves data from extracted tables stored in PostgreSQL.
    Supports filtering by column values and limiting result rows.
    
    Args:
        table_id: ID of the table to query
        column_filters: Dictionary of column_name -> value filters to apply
                       (e.g., {"Region": "Cork", "Year": 2022})
        limit: Maximum number of rows to return (default: 100)
        
    Returns:
        Dictionary with:
        - status: "success" or "error"
        - table_id: The queried table ID
        - headers: List of column names with types
        - rows: List of row data matching filters
        - row_count: Number of rows returned
        - metadata: Table metadata (total row count, etc)
        - error: Error message if status is "error"
        
    Example:
        >>> results = await query_table(
        ...     table_id="table_5_regional_jobs",
        ...     column_filters={"Region": "Dublin"},
        ...     limit=50
        ... )
        >>> for row in results['rows']:
        ...     print(f"{row['Region']}: {row['Job_Count']} jobs")
    """
    if storage_service is None:
        await initialize_services()
    
    try:
        table_data = await storage_service.query_table(
            table_id=table_id,
            filters=column_filters,
            limit=limit
        )
        
        if not table_data:
            return {
                "status": "not_found",
                "error": f"Table '{table_id}' not found",
                "table_id": table_id,
                "headers": [],
                "rows": [],
                "row_count": 0
            }
        
        return {
            "status": "success",
            "table_id": table_id,
            "headers": table_data.get("headers", []),
            "rows": table_data.get("data", []),
            "row_count": len(table_data.get("data", [])),
            "metadata": table_data.get("metadata", {}),
            "filters_applied": column_filters
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "table_id": table_id,
            "headers": [],
            "rows": [],
            "row_count": 0
        }


# ============================================================================
# TOOL 3: DISCOVER_TABLES
# ============================================================================

async def discover_tables(
    namespace: Optional[str] = None,
    keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Discover available tables by namespace or keywords.
    
    This tool helps the agent find which tables exist and what data they contain.
    Returns table metadata including column information and row counts.
    
    Args:
        namespace: Filter tables by namespace (e.g., project/document name)
        keywords: List of keywords to match in table descriptions/names
        
    Returns:
        Dictionary with:
        - status: "success" or "error"
        - tables: List of available tables with metadata
        - table_count: Number of tables found
        - error: Error message if status is "error"
        
    Example:
        >>> tables = await discover_tables(namespace="cyber-ireland-2022")
        >>> for table in tables['tables']:
        ...     print(f"{table['table_id']}: {table['column_count']} columns, {table['row_count']} rows")
    """
    if storage_service is None:
        await initialize_services(namespace or "diligence-ai")
    
    try:
        tables = await storage_service.list_tables_in_namespace(
            namespace=namespace
        )
        
        # Format table metadata (reflects stored schema: metadata has page_number, row_count from len(data))
        formatted_tables = []
        for table in tables or []:
            headers = table.get("headers", [])
            metadata = table.get("metadata") or {}
            row_count = metadata.get("row_count")  # from table extractor when present
            if row_count is None:
                row_count = len(table.get("data", []))

            formatted_tables.append({
                "table_id": table.get("table_id"),
                "column_count": len(headers),
                "columns": [h.get("name") for h in headers] if headers else [],
                "row_count": row_count,
                "page_number": metadata.get("page_number"),
                "data_types": [h.get("type") for h in headers] if headers else [],
                "created_at": table.get("created_at")
            })
        
        # Apply keyword filtering if provided
        if keywords:
            keywords_lower = [k.lower() for k in keywords]
            filtered_tables = []
            for table in formatted_tables:
                # Check if any keyword matches table_id or column names
                table_text = (table["table_id"] + " " + " ".join(table["columns"])).lower()
                if any(kw in table_text for kw in keywords_lower):
                    filtered_tables.append(table)
            formatted_tables = filtered_tables
        
        return {
            "status": "success",
            "tables": formatted_tables,
            "table_count": len(formatted_tables),
            "namespace": namespace,
            "keywords_filter": keywords
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "tables": [],
            "table_count": 0
        }


# ============================================================================
# TOOL 4: GET_TABLE_INFO (Convenience Tool)
# ============================================================================

async def get_table_info(table_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific table.
    
    Provides column names, data types, sample rows, and metadata for a table
    to help the agent understand its structure before querying.
    
    Args:
        table_id: ID of the table to inspect
        
    Returns:
        Dictionary with:
        - status: "success" or "error"
        - table_id: The table ID
        - columns: List of column info (name, type)
        - sample_rows: First few rows for preview
        - metadata: Row count, creation info
        - error: Error message if status is "error"
        
    Example:
        >>> info = await get_table_info("table_5_regional_jobs")
        >>> print(f"Columns: {info['columns']}")
        >>> print(f"Sample data: {info['sample_rows'][:3]}")
    """
    if storage_service is None:
        await initialize_services()
    
    try:
        table_data = await storage_service.query_table(
            table_id=table_id,
            limit=5  # Get first 5 rows as sample
        )
        
        if not table_data:
            return {
                "status": "not_found",
                "error": f"Table '{table_id}' not found",
                "table_id": table_id
            }
        
        headers = table_data.get("headers", [])
        return {
            "status": "success",
            "table_id": table_id,
            "columns": [
                {"name": h.get("name"), "type": h.get("type")}
                for h in headers
            ],
            "column_count": len(headers),
            "sample_rows": table_data.get("data", [])[:5],
            "metadata": table_data.get("metadata", {})
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "table_id": table_id
        }


# ============================================================================
# TOOL 5: CALCULATE_METRICS
# ============================================================================

async def calculate_metrics(
    metric_type: str,
    data: Optional[List[Dict]] = None,
    column: Optional[str] = None,
    start_value: Optional[float] = None,
    end_value: Optional[float] = None,
    years: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate metrics on data (sum, count, avg, CAGR, growth rate).

    This tool performs aggregations and compound annual growth rate calculations
    for forecasting and trend analysis scenarios.

    For CAGR: pass metric_type="cagr", start_value, end_value, and years. data can be omitted.

    Args:
        metric_type: Type of metric - "count", "sum", "avg", "min", "max", "cagr"
        data: List of row dictionaries to calculate on (optional for CAGR)
        column: Column name to calculate on (required for sum/avg/min/max)
        start_value: Starting value for CAGR calculation
        end_value: Ending value for CAGR calculation
        years: Number of years for CAGR calculation (e.g., 8 for 2022 to 2030)

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - metric_type: The calculated metric
        - result: The calculated value
        - data_points: Number of data points used
        - error: Error message if status is "error"

    Example:
        >>> # Calculate CAGR from 2022 to 2030 (e.g. jobs 7351 -> 17333)
        >>> result = await calculate_metrics(
        ...     metric_type="cagr",
        ...     start_value=7351,
        ...     end_value=17333,
        ...     years=8
        ... )
        >>> print(f"CAGR: {result['result']:.2%}")
    """
    data = data if data is not None else []
    if not data and metric_type != "cagr":
        return {
            "status": "error",
            "error": "No data provided",
            "metric_type": metric_type,
            "result": None,
            "data_points": 0
        }
    
    try:
        if metric_type == "count":
            result = len(data)
        
        elif metric_type == "sum":
            if not column:
                return {"status": "error", "error": "column required for sum metric"}
            values = [row.get(column, 0) for row in data if isinstance(row.get(column), (int, float))]
            result = sum(values)
        
        elif metric_type == "avg":
            if not column:
                return {"status": "error", "error": "column required for avg metric"}
            values = [row.get(column, 0) for row in data if isinstance(row.get(column), (int, float))]
            result = sum(values) / len(values) if values else 0
        
        elif metric_type == "min":
            if not column:
                return {"status": "error", "error": "column required for min metric"}
            values = [row.get(column) for row in data if isinstance(row.get(column), (int, float))]
            result = min(values) if values else None
        
        elif metric_type == "max":
            if not column:
                return {"status": "error", "error": "column required for max metric"}
            values = [row.get(column) for row in data if isinstance(row.get(column), (int, float))]
            result = max(values) if values else None
        
        elif metric_type == "cagr":
            # CAGR = (Ending Value / Beginning Value) ^ (1 / number of years) - 1
            if start_value is None or end_value is None or years is None:
                return {
                    "status": "error",
                    "error": "start_value, end_value, and years required for CAGR"
                }
            if start_value <= 0:
                return {"status": "error", "error": "start_value must be positive"}
            
            cagr = (end_value / start_value) ** (1 / years) - 1
            result = cagr
        
        else:
            return {"status": "error", "error": f"Unknown metric type: {metric_type}"}
        
        return {
            "status": "success",
            "metric_type": metric_type,
            "result": result,
            "data_points": len(data) if data else 0,
            "column": column if column else None
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "metric_type": metric_type,
            "result": None
        }


# ============================================================================
# TOOL 6: GET_SOURCE_CITATION
# ============================================================================

async def get_source_citation(
    table_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    page_number: Optional[int] = None,
    row_index: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get source citation information for a specific data point.
    
    Returns page numbers, bounding boxes, and source quotes for citations
    in verification scenarios. Helps trace data back to the original document.
    
    Args:
        table_id: Get citation for a specific table
        chunk_id: Get citation for a specific chunk (from semantic search)
        page_number: Filter citations by page number
        row_index: For tables, which row to get citation for
        
    Returns:
        Dictionary with:
        - status: "success" or "not_found" or "error"
        - page_number: Page(s) where data appears
        - source_quote: Direct quote from source
        - bounding_box: Coordinates for visual highlighting
        - context: Surrounding text for context
        - error: Error message if status is "error"
        
    Example:
        >>> citation = await get_source_citation(
        ...     table_id="table_5_regional_jobs",
        ...     page_number=10,
        ...     row_index=3
        ... )
        >>> print(f"Page {citation['page_number']}: {citation['source_quote']}")
    """
    if storage_service is None or vector_service is None:
        await initialize_services()
    
    try:
        # Handle table citation
        if table_id:
            table_data = await storage_service.query_table(table_id=table_id)
            if not table_data:
                return {
                    "status": "not_found",
                    "error": f"Table '{table_id}' not found"
                }
            
            # Get row data if specified
            if row_index is not None and row_index < len(table_data.get('data', [])):
                row = table_data['data'][row_index]
                return {
                    "status": "success",
                    "source": "table",
                    "table_id": table_id,
                    "row_index": row_index,
                    "row_data": row,
                    "page_number": table_data.get('metadata', {}).get('page_number'),
                    "source_quote": str(row),
                    "context": f"Row {row_index} from {table_id}"
                }
            else:
                # Return table-level citation
                return {
                    "status": "success",
                    "source": "table",
                    "table_id": table_id,
                    "page_number": table_data.get('metadata', {}).get('page_number'),
                    "source_quote": f"Table: {table_id}",
                    "context": f"Table with {len(table_data.get('data', []))} rows"
                }
        
        # Handle chunk citation (from semantic search results)
        if chunk_id or page_number:
            # Reconstruct from bounding box if we have page number
            return {
                "status": "success",
                "source": "chunk",
                "page_number": page_number,
                "bounding_box": {},  # Would be populated if we had the chunk
                "source_quote": "Retrieved from semantic search",
                "context": f"Page {page_number}"
            }
        
        return {
            "status": "error",
            "error": "Either table_id, chunk_id, or page_number required"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# TOOL 7: COMPARE_DATA
# ============================================================================

async def compare_data(
    dataset_1: List[Dict],
    dataset_2: List[Dict],
    comparison_type: str = "difference",
    key_column: Optional[str] = None,
    value_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare two datasets and calculate differences, ratios, or trends.
    
    This tool supports regional comparisons, year-over-year analysis,
    and data synthesis scenarios where multiple data points need comparison.
    
    Args:
        dataset_1: First dataset (list of row dicts)
        dataset_2: Second dataset (list of row dicts)
        comparison_type: "difference", "ratio", "percentage_change", "correlation"
        key_column: Column to group/match rows (e.g., "Region" for regional comparison)
        value_column: Column to compare values (e.g., "Job_Count")
        
    Returns:
        Dictionary with:
        - status: "success" or "error"
        - comparison_type: The type of comparison performed
        - summary: High-level comparison summary
        - comparisons: Detailed per-item comparisons
        - overall_metric: Overall comparison metric
        - error: Error message if status is "error"
        
    Example:
        >>> result = await compare_data(
        ...     dataset_1=cork_jobs,
        ...     dataset_2=dublin_jobs,
        ...     comparison_type="difference",
        ...     key_column="Job_Type",
        ...     value_column="Count"
        ... )
        >>> for comp in result['comparisons'][:3]:
        ...     print(f"{comp['key']}: {comp['metric']}")
    """
    if not dataset_1 or not dataset_2:
        return {
            "status": "error",
            "error": "Both datasets required with data",
            "comparison_type": comparison_type,
            "comparisons": []
        }
    
    try:
        comparisons = []
        
        if comparison_type == "difference":
            # Direct value difference
            if not value_column:
                return {"status": "error", "error": "value_column required for difference comparison"}
            
            values_1 = [row.get(value_column, 0) for row in dataset_1 if isinstance(row.get(value_column), (int, float))]
            values_2 = [row.get(value_column, 0) for row in dataset_2 if isinstance(row.get(value_column), (int, float))]
            
            sum_1 = sum(values_1) if values_1 else 0
            sum_2 = sum(values_2) if values_2 else 0
            difference = sum_2 - sum_1
            
            comparisons = [{
                "dataset_1_total": sum_1,
                "dataset_2_total": sum_2,
                "difference": difference,
                "column": value_column
            }]
            
            overall_metric = {
                "metric": "absolute_difference",
                "value": difference
            }
        
        elif comparison_type == "ratio":
            # Ratio between datasets
            if not value_column:
                return {"status": "error", "error": "value_column required for ratio comparison"}
            
            values_1 = [row.get(value_column, 0) for row in dataset_1 if isinstance(row.get(value_column), (int, float))]
            values_2 = [row.get(value_column, 0) for row in dataset_2 if isinstance(row.get(value_column), (int, float))]
            
            sum_1 = sum(values_1) if values_1 else 1  # Avoid division by zero
            sum_2 = sum(values_2) if values_2 else 0
            ratio = sum_2 / sum_1 if sum_1 > 0 else 0
            
            comparisons = [{
                "dataset_1_total": sum_1,
                "dataset_2_total": sum_2,
                "ratio": ratio,
                "column": value_column
            }]
            
            overall_metric = {
                "metric": "ratio",
                "value": ratio
            }
        
        elif comparison_type == "percentage_change":
            # Percentage change calculation
            if not value_column:
                return {"status": "error", "error": "value_column required for percentage_change"}
            
            values_1 = [row.get(value_column, 0) for row in dataset_1 if isinstance(row.get(value_column), (int, float))]
            values_2 = [row.get(value_column, 0) for row in dataset_2 if isinstance(row.get(value_column), (int, float))]
            
            sum_1 = sum(values_1) if values_1 else 1
            sum_2 = sum(values_2) if values_2 else 0
            pct_change = ((sum_2 - sum_1) / sum_1 * 100) if sum_1 > 0 else 0
            
            comparisons = [{
                "dataset_1_total": sum_1,
                "dataset_2_total": sum_2,
                "percentage_change": pct_change,
                "column": value_column
            }]
            
            overall_metric = {
                "metric": "percentage_change",
                "value": pct_change
            }
        
        elif comparison_type == "correlation":
            # Row-by-row comparison if key column specified
            if key_column and value_column:
                # Build maps for comparison
                map_1 = {row.get(key_column): row.get(value_column, 0) for row in dataset_1}
                map_2 = {row.get(key_column): row.get(value_column, 0) for row in dataset_2}
                
                common_keys = set(map_1.keys()) & set(map_2.keys())
                
                for key in common_keys:
                    val_1 = map_1[key]
                    val_2 = map_2[key]
                    if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                        comparisons.append({
                            "key": key,
                            "dataset_1_value": val_1,
                            "dataset_2_value": val_2,
                            "difference": val_2 - val_1,
                            "ratio": val_2 / val_1 if val_1 > 0 else 0
                        })
                
                overall_metric = {
                    "metric": "rows_compared",
                    "value": len(comparisons)
                }
            else:
                return {"status": "error", "error": "key_column and value_column required for correlation"}
        
        else:
            return {"status": "error", "error": f"Unknown comparison type: {comparison_type}"}
        
        return {
            "status": "success",
            "comparison_type": comparison_type,
            "summary": f"Compared {len(dataset_1)} rows from dataset_1 with {len(dataset_2)} rows from dataset_2",
            "comparisons": comparisons,
            "overall_metric": overall_metric,
            "key_column": key_column,
            "value_column": value_column
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "comparison_type": comparison_type,
            "comparisons": []
        }


# ============================================================================
# Tool Registry for LangGraph
# ============================================================================

TOOLS = {
    "semantic_search": semantic_search,
    "query_table": query_table,
    "discover_tables": discover_tables,
    "get_table_info": get_table_info,
    "calculate_metrics": calculate_metrics,
    "get_source_citation": get_source_citation,
    "compare_data": compare_data,
}


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions in OpenAI JSON schema format for LangGraph.
    
    Returns:
        List of tool definitions compatible with OpenAI function calling
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": "Search document chunks by meaning. Use to find where a number or concept is stated, narrative context, and tables with baseline/target or growth figures. Then use query_table or calculate_metrics as needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query describing what to find"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10
                        },
                        "file_id": {
                            "type": "string",
                            "description": "Optional filter to search within a specific file"
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Optional filter to search within a specific namespace"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_table",
                "description": "Get rows from a table by table_id (from discover_tables). Use for regional breakdowns, employment figures, growth data. Optional column_filters and limit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_id": {
                            "type": "string",
                            "description": "ID of the table to query"
                        },
                        "column_filters": {
                            "type": "object",
                            "description": "Column name to value filters (e.g., {\"Region\": \"Cork\", \"Year\": 2022})"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return (default: 100)",
                            "default": 100
                        }
                    },
                    "required": ["table_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "discover_tables",
                "description": "List tables in the document. Returns table_id, columns, row_count, page_number. For regional or segment comparisons, find the table whose columns match what is being compared (e.g. region/location and count or share); then call query_table with that table_id. Optional keywords to filter when many tables exist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Filter tables by namespace/document"
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords to match in table names or descriptions"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_table_info",
                "description": "Get detailed information about a table. View column names, types, and sample data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_id": {
                            "type": "string",
                            "description": "ID of the table to inspect"
                        }
                    },
                    "required": ["table_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_metrics",
                "description": "Calculate metrics: count, sum, avg, min, max, or CAGR. For CAGR use metric_type='cagr' with start_value, end_value, and years only (data is not required for CAGR).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_type": {
                            "type": "string",
                            "enum": ["count", "sum", "avg", "min", "max", "cagr"],
                            "description": "Type of metric to calculate"
                        },
                        "data": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of row dictionaries (optional for CAGR)"
                        },
                        "column": {
                            "type": "string",
                            "description": "Column name for sum/avg/min/max"
                        },
                        "start_value": {
                            "type": "number",
                            "description": "Starting value for CAGR (e.g. 2022 baseline jobs)"
                        },
                        "end_value": {
                            "type": "number",
                            "description": "Ending value for CAGR (e.g. 2030 target jobs)"
                        },
                        "years": {
                            "type": "integer",
                            "description": "Number of years for CAGR (e.g. 8 for 2022 to 2030)"
                        }
                    },
                    "required": ["metric_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_source_citation",
                "description": "Get source citation for data with page numbers, quotes, and bounding boxes. Traces data back to original document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_id": {
                            "type": "string",
                            "description": "Get citation for a specific table"
                        },
                        "chunk_id": {
                            "type": "string",
                            "description": "Get citation for a specific chunk from semantic search"
                        },
                        "page_number": {
                            "type": "integer",
                            "description": "Filter citations by page number"
                        },
                        "row_index": {
                            "type": "integer",
                            "description": "For tables, which row to get citation for"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_data",
                "description": "Compare two datasets (differences, ratios, percentage change). You must pass dataset_1 and dataset_2 as arrays of rows—e.g. from query_table. Get table data first with query_table, then call compare_data with those rows. Required: dataset_1, dataset_2, comparison_type.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_1": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "First dataset (list of row dictionaries)"
                        },
                        "dataset_2": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Second dataset (list of row dictionaries)"
                        },
                        "comparison_type": {
                            "type": "string",
                            "enum": ["difference", "ratio", "percentage_change", "correlation"],
                            "description": "Type of comparison to perform"
                        },
                        "key_column": {
                            "type": "string",
                            "description": "Column to group/match rows (e.g., 'Region' for regional comparison)"
                        },
                        "value_column": {
                            "type": "string",
                            "description": "Column to compare values (e.g., 'Job_Count')"
                        }
                    },
                    "required": ["dataset_1", "dataset_2", "comparison_type"]
                }
            }
        }
    ]
