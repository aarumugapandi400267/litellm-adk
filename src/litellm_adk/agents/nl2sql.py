from typing import List, Dict, Any, Optional, Union
import logging
from ..core.agent import LiteLLMAgent
from ..adapters import DatabaseAdapter, SQLAdapter, MongoAdapter

logger = logging.getLogger("litellm_adk.agents.database")

class DatabaseAgent(LiteLLMAgent):
    """
    Unified Database Agent supporting SQL and NoSQL via Pluggable Adapters.
    """
    def __init__(
        self,
        db_url: Optional[str] = None,
        db_config: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        schema_config: Optional[Dict[str, Any]] = None,
        data_dictionary: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        lazy_schema_limit: int = 15,
        return_direct: bool = False,
        **kwargs
    ):
        """
        Initialize the Database Agent with an Adapter strategy.
        
        Args:
            return_direct: If True, returns the tool execution result directly (for the UI to render) 
                           without LLM summarization context window bloat.
        """
        self.data_dictionary = data_dictionary or {}
        self.lazy_schema_limit = lazy_schema_limit
        self.return_direct = return_direct
        self.last_query_result = None # Stores the raw data from the last tool execution
        
        # 1. Initialize Adapter
        if db_url and not db_config:
            # Legacy/Simple mode inference
            if db_url.startswith("mongodb"):
                self.adapter = MongoAdapter({"url": db_url})
            else:
                 self.adapter = SQLAdapter({"url": db_url, "schema_config": schema_config})
        elif db_config:
            # Advanced Config Mode
            db_type = db_config.get("type", "sql").lower()
            if db_type == "mongo":
                self.adapter = MongoAdapter(db_config)
            else:
                if schema_config:
                    db_config["schema_config"] = schema_config
                self.adapter = SQLAdapter(db_config)
        else:
            raise ValueError("Must provide either db_url or db_config.")
            
        self.max_retries = max_retries
        
        # 2. Get Initial Schema (Lazy Logic)
        self.all_tables = self.adapter.get_table_names()
        
        self.use_lazy_schema = False
        if len(self.all_tables) > self.lazy_schema_limit:
            self.use_lazy_schema = True
            logger.info(f"Lazy Schema Enabled: {len(self.all_tables)} tables > limit {self.lazy_schema_limit}")
            self.schema = self._get_table_list_summary()
        else:
            self.schema = self.adapter.get_schema_summary()
        
        # 3. Construct System Prompt & Register Tools
        self.base_system_prompt = kwargs.get("system_prompt", 
            "You are an expert Database Analyst. Your goal is to answer user questions by querying the database.")
            
        # Get DB-specific tools from adapter
        db_tools = self.adapter.get_tools()
        
        if self.return_direct:
             # Create callback to store result
             def _store_result(data):
                 self.last_query_result = data
                 
             # Re-bind tools with callback
             # This requires adapter support
             if hasattr(self.adapter, 'set_result_callback'):
                 self.adapter.set_result_callback(_store_result)
            
        # Merge with user provided tools if any
        user_tools = kwargs.get("tools", [])
        combined_tools = db_tools + user_tools
        
        # 4. Initialize Parent Agent
        kwargs.pop("system_prompt", None)
        kwargs.pop("tools", None)
        
        system_prompt = self._construct_system_prompt(self.schema)
        
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            tools=combined_tools,
            **kwargs
        )

    def _get_table_list_summary(self) -> str:
        """Returns a simplified list of Available Tables."""
        return "Available Tables/Collections (Full schema loaded on demand):\n" + "\n".join([f"- {t}" for t in self.all_tables])

    def _construct_system_prompt(self, schema_context: str) -> str:
        data_dict_str = self._format_data_dictionary() or "No additional context provided."
        
        # Get instructions from adapter
        instructions = self.adapter.get_system_prompt_template()
        
        if self.return_direct:
            instructions += """
\n**IMPORTANT MODE: Blind Summarization**
The user cannot see the private data execution results, but you represent as if you know the data.
1. **FIRST**, you MUST call the tool (e.g., 'execute_sql' or 'mongo_find'). Do NOT generate the summary yet.
2. The tool will return a hidden summary (e.g. '500 rows found').
3. **AFTER** the tool returns, you must respond to the user confirming the action, e.g., "I have retrieved 500 orders from the database..."
4. Do NOT say "I cannot see the data". Pretend you are presenting it.
5. Do NOT hallucinate the result before calling the tool.
"""
        else:
             instructions += "\n5. If the result is empty, try a broader query.\n6. Return the final answer in natural language."

        return f"""{self.base_system_prompt}

You have access to the database.
Schema Context:
{schema_context}

**Domain Knowledge (Data Dictionary):**
{data_dict_str}

**Instructions:**
{instructions}
"""

    async def ainvoke(self, content: Union[str, List[dict]], **kwargs):
        """Override ainvoke to inject relevant schema if lazy loading is enabled."""
        if not self.use_lazy_schema:
            return await super().ainvoke(content, **kwargs)
            
        # Lazy Loading Strategy
        user_query = content if isinstance(content, str) else str(content)
        
        relevant_tables = []
        low_query = user_query.lower()
        
        for t in self.all_tables:
            if t.lower() in low_query:
                relevant_tables.append(t)
            for k in self.data_dictionary.keys():
                if k.startswith(f"{t}.") and (k.lower() in low_query or self.data_dictionary[k].lower() in low_query):
                     if t not in relevant_tables:
                         relevant_tables.append(t)

        if not relevant_tables:
             relevant_tables = self.all_tables[:5]
             
        # Use Adapter to fetch targeted schema
        targeted_schema = self.adapter.get_schema_summary(relevant_tables)
        
        schema_msg = {
            "role": "system", 
            "content": f"**Relevant Schema Context for this Query:**\n{targeted_schema}"
        }
        
        if isinstance(content, str):
            messages = [schema_msg, {"role": "user", "content": content}]
        else:
            messages = [schema_msg] + content
            
        return await super().ainvoke(messages, **kwargs)

    def refresh_schema(self):
        """Force refresh of database schema."""
        self.all_tables = self.adapter.get_table_names()
        
        if not self.use_lazy_schema:
            self.schema = self.adapter.get_schema_summary()
        else:
             self.schema = self._get_table_list_summary()

    def _format_data_dictionary(self) -> str:
        if not self.data_dictionary:
            return ""
        lines = []
        for key, description in self.data_dictionary.items():
            lines.append(f"- {key}: {description}")
        return "\n".join(lines)

# Backward compatibility alias
NL2SQLAgent = DatabaseAgent
