"""
LangGraph agent for dual-storage querying.

This implementation uses LangGraph for orchestration and OpenAI/Azure OpenAI
tool-calling directly (no LangChain agents).
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from llm_middleware import LLMMiddleware
from tools import TOOLS, close_services, get_tool_definitions, initialize_services


SYSTEM_PROMPT = """
You are a precise data analysis assistant for diligence-ai dual-storage retrieval.

Rules:
1) Prefer exact structured tools when user asks for numbers/comparisons.
2) Use semantic_search for narrative context and citations.
3) For calculations, use calculate_metrics/compare_data instead of mental math.
4) Always ground answers in tool outputs and include page/source context when available.
5) If data is missing, clearly say what is missing and suggest the next tool/query.
""".strip()


class AgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    pending_tool_calls: List[Dict[str, Any]]
    final_answer: str
    namespace: Optional[str]
    step_count: int
    max_steps: int


class LangGraphDualStorageAgent:
    """LangGraph agent using OpenAI function-calling and local async tools."""

    def __init__(self, model: Optional[str] = None, default_namespace: Optional[str] = None):
        self.default_namespace = default_namespace
        self.model = model or os.getenv("AGENT_CHAT_MODEL", "gpt-4o-mini")
        self.llm = LLMMiddleware()
        self.client = self.llm.initialize_client()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("llm", self._llm_node)
        graph.add_node("tools", self._tool_node)

        graph.set_entry_point("llm")
        graph.add_conditional_edges(
            "llm",
            self._route_after_llm,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "llm")
        return graph.compile()

    def _route_after_llm(self, state: AgentState) -> str:
        pending = state.get("pending_tool_calls", [])
        max_steps = state.get("max_steps", 8)
        step_count = state.get("step_count", 0)

        if pending and step_count < max_steps:
            return "tools"
        return "end"

    async def _llm_node(self, state: AgentState) -> AgentState:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=state["messages"],
            tools=get_tool_definitions(),
            tool_choice="auto",
            temperature=0,
        )

        msg = response.choices[0].message
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }

        pending_tool_calls: List[Dict[str, Any]] = []
        if msg.tool_calls:
            tool_calls = []
            for call in msg.tool_calls:
                payload = {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                tool_calls.append(payload)
                pending_tool_calls.append(payload)

            assistant_message["tool_calls"] = tool_calls

        updated_messages = state["messages"] + [assistant_message]

        final_answer = state.get("final_answer", "")
        if not pending_tool_calls:
            final_answer = msg.content or final_answer

        return {
            **state,
            "messages": updated_messages,
            "pending_tool_calls": pending_tool_calls,
            "final_answer": final_answer,
        }

    async def _tool_node(self, state: AgentState) -> AgentState:
        tool_messages: List[Dict[str, Any]] = []

        for call in state.get("pending_tool_calls", []):
            function_name = call["function"]["name"]
            raw_args = call["function"].get("arguments", "{}")

            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                arguments = {}

            if function_name == "semantic_search":
                namespace = arguments.get("namespace") or state.get("namespace")
                if namespace:
                    arguments["namespace"] = namespace
            elif function_name == "lookup_fact":
                namespace = arguments.get("namespace") or state.get("namespace")
                if namespace:
                    arguments["namespace"] = namespace
            elif function_name == "discover_tables":
                namespace = arguments.get("namespace") or state.get("namespace")
                if namespace:
                    arguments["namespace"] = namespace

            if function_name not in TOOLS:
                result = {"status": "error", "error": f"Unknown tool: {function_name}"}
            else:
                try:
                    result = await TOOLS[function_name](**arguments)
                except Exception as exc:
                    result = {"status": "error", "error": str(exc)}

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": function_name,
                    "content": json.dumps(result, default=str),
                }
            )

        return {
            **state,
            "messages": state["messages"] + tool_messages,
            "pending_tool_calls": [],
            "step_count": state.get("step_count", 0) + 1,
        }

    async def ainvoke(
        self,
        user_query: str,
        namespace: Optional[str] = None,
        max_steps: int = 8,
        debug: bool = False,
    ) -> Dict[str, Any]:
        active_namespace = namespace or self.default_namespace
        await initialize_services(active_namespace or "diligence-ai")

        initial_state: AgentState = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            "namespace": active_namespace,
            "step_count": 0,
            "max_steps": max_steps,
            "pending_tool_calls": [],
            "final_answer": "",
        }

        result = await self.graph.ainvoke(initial_state)
        response = {
            "answer": result.get("final_answer", ""),
            "messages": result.get("messages", []),
            "steps": result.get("step_count", 0),
            "namespace": active_namespace,
        }

        if debug:
            response["trace"] = self._build_trace(response.get("messages", []))

        return response

    def _build_trace(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build debug trace from message history (tool calls + intermediate results)."""
        events: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []

        for message in messages:
            role = message.get("role")

            if role == "assistant" and message.get("tool_calls"):
                for call in message.get("tool_calls", []):
                    raw_args = call.get("function", {}).get("arguments", "{}")
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        parsed_args = {"_raw": raw_args}

                    entry = {
                        "tool_call_id": call.get("id"),
                        "tool_name": call.get("function", {}).get("name"),
                        "arguments": parsed_args,
                    }
                    tool_calls.append(entry)
                    events.append({"type": "tool_call", **entry})

            elif role == "tool":
                raw_content = message.get("content", "")
                try:
                    parsed_result = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                except json.JSONDecodeError:
                    parsed_result = {"_raw": raw_content}

                entry = {
                    "tool_call_id": message.get("tool_call_id"),
                    "tool_name": message.get("name"),
                    "result": parsed_result,
                }
                tool_results.append(entry)
                events.append({"type": "tool_result", **entry})

            elif role == "assistant" and message.get("content"):
                events.append({
                    "type": "assistant_message",
                    "content": message.get("content"),
                })

        return {
            "events": events,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }

    async def aclose(self):
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            maybe_awaitable = close_fn()
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
        await close_services()


async def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Run LangGraph dual-storage agent")
    parser.add_argument("query", type=str, help="User query")
    parser.add_argument("--namespace", type=str, default=None, help="Optional namespace filter")
    parser.add_argument("--model", type=str, default=None, help="Override chat model")
    parser.add_argument("--debug", action="store_true", help="Include agent trace output")
    args = parser.parse_args()

    agent = LangGraphDualStorageAgent(model=args.model, default_namespace=args.namespace)
    try:
        result = await agent.ainvoke(args.query, namespace=args.namespace, debug=args.debug)
        print("\n=== Agent Response ===")
        print(result["answer"] or "(no final response text)")
        print(f"\n[steps: {result['steps']}]")
        if args.debug:
            print("\n=== Agent Trace ===")
            print(json.dumps(result.get("trace", {}), indent=2, default=str))
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(_main())
