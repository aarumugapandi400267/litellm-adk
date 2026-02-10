# 📊 DatabaseAgent (NL2SQL / NL2Mongo)

The `DatabaseAgent` (aliased as `NL2SQLAgent`) specializes in converting natural language into database queries. It uses a pluggable **Adapter** architecture to support multiple database types.

## 🔌 Supported Databases

*   **SQL**: PostgreSQL, MySQL, SQLite, etc. (via SQLAlchemy).
*   **NoSQL**: MongoDB.

## 🚀 Usage

### Simple Initialization (Inferred)
```python
from litellm_adk import DatabaseAgent

agent = DatabaseAgent(db_url="mysql+pymysql://user:pass@host/db")
```

### Advanced Config (Explicit)
```python
agent = DatabaseAgent(
    db_config={
        "type": "mongo",
        "url": "mongodb://localhost:27017/my_app"
    },
    return_direct=True # Hidden query results from LLM context
)
```

## 🧠 Core Concepts

### 1. Pluggable Adapters
*   `SQLAdapter`: Handles schema introspection and query execution for relational DBs.
*   `MySQLAdapter`: Adds backtick escaping and MySQL-specific optimization.
*   `MongoAdapter`: Handles collection listing and aggregation pipelines.

### 2. Blind Summarization (`return_direct=True`)
For enterprise security, the LLM can trigger a tool and receive a "Blind Summary" (e.g., "Found 500 rows") instead of the full data. This prevents leaking sensitive PII into provider logs while still allowing the agent to confirm success.

### 3. Lazy Schema Loading
For databases with hundreds of tables, the agent loads a simplified list initially. It uses the `inspect_table` / `inspect_collection` tools to fetch detailed columns only for the tables it decides are relevant.

## 🛠️ Tools Provided
*   `execute_sql`: Run queries.
*   `inspect_table`: Fetch detailed schema instructions.
*   `mongo_find` / `mongo_aggregate`: Execute Mongo operations.
