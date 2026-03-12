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

import config
from llm_middleware import LLMMiddleware
from tools import TOOLS, close_services, get_tool_definitions, initialize_services


SYSTEM_PROMPT = """
You are a precise data analysis assistant for document-backed retrieval. You have access to semantic search (vector), structured tables (PostgreSQL), and calculation tools. Always use tools to find numbers—never ask the user to supply values that should be in the document.

## General
- Prefer structured tools (discover_tables, query_table, calculate_metrics) when the question is about numbers, comparisons, or growth rates.
- Always ground answers in tool outputs and cite page/source when available (e.g. table and page where the figure appears).
- When several figures (e.g. region and national) came from the same query_table result, cite that result's table_id and page for all of them; do not cite a different table for some figures unless those values actually came from that other table's rows.
- If a tool returns an error, read the error message and fix the call (e.g. add missing arguments); do not repeat the exact same failing call.
- Always produce a final answer: summarize what you found and cite sources; if tools failed, say what you tried and what error occurred—do not return an empty answer.

## Verification (exact number + where stated)
- Use semantic_search with queries that match the metric (e.g. total employment, headcount, revenue) to find the number and the table/section where it is stated.
- Include in your answer: the exact number and where it appears in the document (e.g. table caption and page, or section title and page).

## Regional or segment comparison (e.g. one region vs national, one category vs total)
- Do not rely only on semantic search for comparisons; tables hold the exact counts. Call discover_tables, then pick the table whose columns match what is being compared (e.g. region/location and count or share). Call query_table with that table_id to get the rows.
- For discover_tables: call it without keywords (or with keywords that appear in column headers, e.g. region, offices, dedicated). Do not use specific region names (e.g. South-West) or "National Average" as keywords—those usually appear in row values, not in column names, so they filter out the right table. Scan the full table list and pick the table that has a region/location column and count or share columns.
- Use the document's own labels for regions or segments (e.g. area names, category names as they appear in the table). Map user terms to table labels (e.g. South-West → Cork if that is how the table labels the region).
- **Concentration of a category (e.g. Pure-Play, dedicated) in a region vs national:** Concentration means *share* (that category as a fraction of the total). Pick the table that has REGION and both the category count (e.g. "NO. OF OFFICES (DEDICATED FIRMS)") and total count (e.g. "NO. OF CYBER SECURITY OFFICES" or similar). From that same table: (1) take the row for the region (e.g. Cork for South-West), (2) take the row for the whole country (e.g. Ireland). Compute region_concentration = category_count_region / total_count_region and national_concentration = category_count_national / total_count_national. Report both as percentages and compare them. Do not mix data from different tables; do not use the national row's numbers as the region's numbers.
- Only call compare_data after you have two datasets (e.g. rows from query_table). compare_data requires dataset_1, dataset_2, and comparison_type—do not call it with only column names.

## Forecasting / CAGR / growth rate (baseline to target)
- Use semantic_search to find the baseline and target values and where they are stated (e.g. growth table, projections section). Identify the table or section that contains the figures.
- Call calculate_metrics with metric_type="cagr", start_value=<baseline>, end_value=<target>, years=<number of years>. Do not pass "data" for CAGR—only start_value, end_value, and years are required.
- State the result as a percentage and cite the source (table/section and page) from the document.
""".strip()


class AgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    pending_tool_calls: List[Dict[str, Any]]
    final_answer: str
    namespace: Optional[str]
    step_count: int
    max_steps: int
    has_tool_results: bool  # True once at least one tool round has completed
    verified: bool          # True once the verify node has run


class LangGraphDualStorageAgent:
    """LangGraph agent using OpenAI function-calling and local async tools."""

    def __init__(self, model: Optional[str] = None, default_namespace: Optional[str] = None):
        self.default_namespace = default_namespace
        self.model = model or config.AGENT_CHAT_MODEL
        self.llm = LLMMiddleware()
        self.client = self.llm.initialize_client()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("llm", self._llm_node)
        graph.add_node("tools", self._tool_node)
        graph.add_node("verify", self._verify_node)

        graph.set_entry_point("llm")
        graph.add_conditional_edges(
            "llm",
            self._route_after_llm,
            {"tools": "tools", "verify": "verify", "end": END},
        )
        graph.add_edge("tools", "llm")
        # After verify, one final llm_node call synthesises the grounded answer.
        graph.add_edge("verify", "llm")
        return graph.compile()

    def _route_after_llm(self, state: AgentState) -> str:
        pending = state.get("pending_tool_calls", [])
        max_steps = state.get("max_steps", 8)
        step_count = state.get("step_count", 0)
        verified = state.get("verified", False)
        has_tool_results = state.get("has_tool_results", False)

        # Keep executing tool calls while there are pending ones, we haven't
        # verified yet, and we're within the step budget.
        if pending and step_count < max_steps and not verified:
            return "tools"

        # All tool rounds are done and evidence hasn't been verified yet —
        # run the verify phase before synthesising the final answer.
        if has_tool_results and not verified:
            return "verify"

        # Either no tools were called (direct answer) or verification is done.
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

            # Request namespace (from API) always wins so the agent cannot query wrong document
            request_ns = state.get("namespace")
            if function_name in ("semantic_search", "discover_tables") and request_ns:
                arguments["namespace"] = request_ns

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
            "has_tool_results": True,
        }

    async def _verify_node(self, state: AgentState) -> AgentState:
        """
        Verification phase (retrieve → verify → answer).

        Makes a single focused LLM call that reviews all tool results against the
        original question, then injects a concise verification note as a system
        message.  The subsequent llm_node call synthesises the final answer with
        that grounding assessment already in context.
        """
        user_question = next(
            (m["content"] for m in state["messages"] if m["role"] == "user"), ""
        )

        # Collect every tool result, capped to avoid bloating the context window.
        evidence_parts: List[str] = []
        for m in state["messages"]:
            if m["role"] == "tool":
                try:
                    parsed = json.loads(m.get("content", "{}"))
                except (json.JSONDecodeError, TypeError):
                    parsed = {"_raw": m.get("content", "")}
                snippet = json.dumps(parsed, default=str)[:800]
                evidence_parts.append(f"[{m.get('name', 'tool')}]\n{snippet}")

        evidence_block = "\n\n".join(evidence_parts) if evidence_parts else "No tool results available."

        verify_prompt = (
            f"Question: {user_question}\n\n"
            f"Retrieved evidence:\n{evidence_block}\n\n"
            "Assess the evidence in 2–3 sentences:\n"
            "1. Does it directly answer the question? State the key finding and "
            "its source (page number or table ID).\n"
            "2. Are there any contradictions or inconsistencies between results?\n"
            "3. Is any critical data missing that would affect accuracy?\n"
            "Be concise and factual."
        )

        verification_response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": verify_prompt}],
            temperature=0,
        )

        verification_note = verification_response.choices[0].message.content or ""

        inject: Dict[str, Any] = {
            "role": "system",
            "content": (
                f"[VERIFICATION ASSESSMENT]\n{verification_note}\n\n"
                "Based on this assessment, produce the final answer. "
                "Cite specific pages and sources. Address any gaps noted above."
            ),
        }

        return {
            **state,
            "messages": state["messages"] + [inject],
            "verified": True,
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
            "has_tool_results": False,
            "verified": False,
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
            response["verified"] = result.get("verified", False)

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

            elif role == "system" and message.get("content", "").startswith("[VERIFICATION ASSESSMENT]"):
                events.append({
                    "type": "verification",
                    "content": message.get("content"),
                })

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
