"""
PostgreSQL service for structured storage of facts and tables
Part of dual storage architecture: Vector DB + PostgreSQL
"""
import uuid
import json
from typing import Dict, List, Optional
import asyncpg

import config


class StructuredStorageService:
    """PostgreSQL service for storing extracted facts and tables"""
    
    def __init__(self):
        self.postgres_uri = config.POSTGRES_URI
        self.connection_pool = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize PostgreSQL connection and create tables"""
        if self._initialized:
            return
        
        print(f"[POSTGRES] Connecting to PostgreSQL...")
        self.connection_pool = await asyncpg.create_pool(self.postgres_uri)
        
        # Create tables if they don't exist
        async with self.connection_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    file_id VARCHAR(255) NOT NULL,
                    file_name VARCHAR(255) NOT NULL,
                    namespace VARCHAR(255) NOT NULL,
                    entity_type VARCHAR(100) NOT NULL,
                    value VARCHAR(500) NOT NULL,
                    page INTEGER,
                    source_quote TEXT,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS tables (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    table_id VARCHAR(255) NOT NULL UNIQUE,
                    file_id VARCHAR(255) NOT NULL,
                    file_name VARCHAR(255) NOT NULL,
                    namespace VARCHAR(255) NOT NULL,
                    headers JSONB NOT NULL,
                    data JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for efficient queries
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_facts_file_id 
                ON facts(file_id)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_facts_entity_type 
                ON facts(entity_type)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_facts_namespace 
                ON facts(namespace)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_facts_namespace_entity 
                ON facts(namespace, entity_type)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_tables_file_id
                ON tables(file_id)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_tables_table_id
                ON tables(table_id)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_tables_namespace
                ON tables(namespace)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_tables_namespace_id
                ON tables(namespace, table_id)
            ''')
        
        self._initialized = True
        print(f"[POSTGRES] Connected and tables initialized")
    
    async def store_facts(
        self, 
        facts: List[Dict], 
        file_id: str, 
        file_name: str,
        namespace: str
    ) -> Dict:
        """
        Store extracted facts in PostgreSQL
        
        Args:
            facts: List of fact dictionaries
            file_id: Unique file identifier
            file_name: Name of source file
            namespace: Namespace for organization
            
        Returns:
            Status dictionary
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if not facts:
                return {
                    "status": "success",
                    "stored_count": 0,
                    "message": "No facts to store"
                }
            
            async with self.connection_pool.acquire() as conn:
                inserted_ids = []
                async with conn.transaction():
                    for fact in facts:
                        fact_id = str(uuid.uuid4())
                        # Value must be string for VARCHAR(500); avoid scientific notation for integers
                        raw_value = fact.get('value')
                        if raw_value is None:
                            value_str = ''
                        elif isinstance(raw_value, float) and raw_value == int(raw_value):
                            value_str = str(int(raw_value))
                        else:
                            value_str = str(raw_value)
                        if len(value_str) > 500:
                            value_str = value_str[:500]

                        await conn.execute('''
                            INSERT INTO facts (
                                id, file_id, file_name, namespace, entity_type,
                                value, page, source_quote, confidence
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ''',
                            fact_id,
                            file_id,
                            file_name,
                            namespace,
                            fact.get('entity_type'),
                            value_str,
                            fact.get('page'),
                            fact.get('source_quote'),
                            fact.get('confidence', 0.0)
                        )
                        inserted_ids.append(fact_id)
            
            return {
                "status": "success",
                "stored_count": len(inserted_ids),
                "fact_ids": inserted_ids
            }
            
        except Exception as e:
            print(f"[POSTGRES] Error storing facts: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "stored_count": 0
            }
    
    async def store_tables(
        self, 
        tables: List[Dict], 
        file_id: str, 
        file_name: str,
        namespace: str
    ) -> Dict:
        """
        Store structured tables in PostgreSQL
        
        Args:
            tables: List of table dictionaries
            file_id: Unique file identifier
            file_name: Name of source file
            namespace: Namespace for organization
            
        Returns:
            Status dictionary
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if not tables:
                return {
                    "status": "success",
                    "stored_count": 0,
                    "message": "No tables to store"
                }
            
            async with self.connection_pool.acquire() as conn:
                inserted_ids = []
                async with conn.transaction():
                    for table in tables:
                        table_uuid = str(uuid.uuid4())
                        table_id = table.get('table_id', f'table_{table_uuid[:8]}')
                        # Include page_number in metadata for citations (get_source_citation)
                        metadata = dict(table.get('metadata') or {})
                        if table.get('page') is not None:
                            metadata['page_number'] = table['page']

                        await conn.execute('''
                            INSERT INTO tables (
                                id, table_id, file_id, file_name, namespace,
                                headers, data, metadata
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (table_id) DO UPDATE SET
                                headers = EXCLUDED.headers,
                                data = EXCLUDED.data,
                                metadata = EXCLUDED.metadata
                        ''',
                            table_uuid,
                            table_id,
                            file_id,
                            file_name,
                            namespace,
                            json.dumps(table.get('headers', [])),
                            json.dumps(table.get('data', [])),
                            json.dumps(metadata)
                        )
                        inserted_ids.append(table_uuid)
            
            return {
                "status": "success",
                "stored_count": len(inserted_ids),
                "table_ids": inserted_ids
            }
            
        except Exception as e:
            print(f"[POSTGRES] Error storing tables: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "stored_count": 0
            }
    
    async def get_fact(
        self, 
        entity_type: Optional[str] = None, 
        filters: Optional[Dict] = None,
        namespace: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Retrieve facts from PostgreSQL with optional filtering

        Args:
            entity_type: Type of entity (e.g., 'job_count'). If None, returns all facts.
            filters: Additional filter criteria (e.g., {'page': 10, 'file_id': 'xyz'})
            namespace: Filter by namespace (recommended to avoid cross-document results)
            limit: Maximum number of facts to return (default: 50)

        Returns:
            List of fact dictionaries
        """
        if not self._initialized:
            await self.initialize()

        try:
            query = 'SELECT * FROM facts WHERE 1=1'
            params = []

            if entity_type:
                query += f' AND entity_type = ${len(params) + 1}'
                params.append(entity_type)

            if namespace:
                query += f' AND namespace = ${len(params) + 1}'
                params.append(namespace)

            if filters:
                for key, value in filters.items():
                    query += f' AND {key} = ${len(params) + 1}'
                    params.append(value)

            query += f' ORDER BY created_at DESC LIMIT ${len(params) + 1}'
            params.append(limit)
            
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            print(f"[POSTGRES] Error getting facts: {str(e)}")
            return []
    
    async def query_table(
        self,
        table_id: str,
        filters: Optional[Dict] = None,
        columns: Optional[List[str]] = None,
        limit: int = 100
    ) -> Optional[Dict]:
        """
        Query a structured table with optional filtering and column selection
        
        Args:
            table_id: ID of the table
            filters: Dictionary of column_name -> value filters (e.g., {"Region": "Cork"})
            columns: Specific columns to return (None = all columns)
            limit: Maximum rows to return (default: 100)
            
        Returns:
            Table data dictionary with headers and filtered data, or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(
                    'SELECT * FROM tables WHERE table_id = $1',
                    table_id
                )
            
            if not row:
                return None
            
            table = dict(row)
            # Deserialize JSONB columns
            table['headers'] = json.loads(table['headers']) if isinstance(table['headers'], str) else table['headers']
            table['data'] = json.loads(table['data']) if isinstance(table['data'], str) else table['data']
            table['metadata'] = json.loads(table['metadata']) if isinstance(table['metadata'], str) else table['metadata']
            
            # Apply filters if provided (dict-based: {"column_name": value})
            if filters:
                filtered_rows = []
                for row_data in table.get('data', []):
                    match = True
                    for column, filter_value in filters.items():
                        if row_data.get(column) != filter_value:
                            match = False
                            break
                    if match:
                        filtered_rows.append(row_data)
                table['data'] = filtered_rows
            
            # Apply limit
            table['data'] = table['data'][:limit]
            
            # Select specific columns if provided
            if columns:
                filtered_data = []
                for row_data in table.get('data', []):
                    filtered_row = {col: row_data.get(col) for col in columns if col in row_data}
                    filtered_data.append(filtered_row)
                table['data'] = filtered_data
            
            return table
            
        except Exception as e:
            print(f"[POSTGRES] Error querying table: {str(e)}")
            return None
    
    async def list_tables_in_namespace(
        self,
        namespace: str
    ) -> List[Dict]:
        """
        List all tables in a namespace with full metadata
        
        Args:
            namespace: Namespace to query
            
        Returns:
            List of table metadata including headers, row counts, and creation info
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(
                    'SELECT * FROM tables WHERE namespace = $1 ORDER BY created_at DESC',
                    namespace
                )
            
            tables = []
            for row in rows:
                table_dict = dict(row)
                # Deserialize JSONB columns
                headers = json.loads(table_dict['headers']) if isinstance(table_dict['headers'], str) else table_dict['headers']
                data = json.loads(table_dict['data']) if isinstance(table_dict['data'], str) else table_dict['data']
                metadata = json.loads(table_dict['metadata']) if isinstance(table_dict['metadata'], str) else table_dict['metadata']
                
                # Add deserialized data back
                table_dict['headers'] = headers
                table_dict['data'] = data
                table_dict['metadata'] = metadata
                
                tables.append(table_dict)
            
            return tables
            
        except Exception as e:
            print(f"[POSTGRES] Error listing tables: {str(e)}")
            return []
    
    async def delete_by_file_id(self, file_id: str) -> Dict:
        """
        Delete all facts and tables for a specific file
        
        Args:
            file_id: File identifier
            
        Returns:
            Status dictionary with counts
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.connection_pool.acquire() as conn:
                async with conn.transaction():
                    facts_deleted = await conn.fetchval(
                        'DELETE FROM facts WHERE file_id = $1',
                        file_id
                    )
                    tables_deleted = await conn.fetchval(
                        'DELETE FROM tables WHERE file_id = $1',
                        file_id
                    )
            
            return {
                "status": "success",
                "facts_deleted": facts_deleted,
                "tables_deleted": tables_deleted
            }
            
        except Exception as e:
            print(f"[POSTGRES] Error deleting by file_id: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "facts_deleted": 0,
                "tables_deleted": 0
            }
    
    async def close(self):
        """Close PostgreSQL connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            print("[POSTGRES] Connection pool closed")
