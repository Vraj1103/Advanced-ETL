# Execution logs (agent traces for evaluation)

This folder holds the **execution logs/traces** for the three evaluation test queries. Each file shows how the agent planned its steps, used tools, and arrived at the answer.

## How to reproduce

1. **Start the API** (from the project root):
   ```bash
   python app.py
   ```
   Server runs at `http://localhost:8001` by default.

2. **Upload the Cyber Ireland 2022 PDF** with namespace `cyber-ireland-2022`:
   ```bash
   curl -X POST "http://localhost:8001/upload" \
     -F "file=@/path/to/Cyber_Ireland_2022.pdf" \
     -F "namespace=cyber-ireland-2022"
   ```
   Wait until the response shows success and chunks/tables stored.

3. **Run the test-queries script** (with debug so traces are included):
   ```bash
   python scripts/run_test_queries_and_save_traces.py
   ```
   Optional: use a different namespace or base URL:
   ```bash
   python scripts/run_test_queries_and_save_traces.py --namespace my-namespace --base-url http://localhost:8001
   ```

4. **Generated files**
   - `test1_verification.json` — “What is the total number of jobs reported, and where exactly is this stated?”
   - `test2_data_synthesis.json` — “Compare the concentration of 'Pure-Play' cybersecurity firms in the South-West against the National Average.”
   - `test3_forecasting.json` — “Based on our 2022 baseline and the stated 2030 job target, what is the required CAGR to hit that goal?”

Each JSON file contains:
- `query`, `namespace`, `run_at`
- `response.status`, `response.answer`, `response.steps`
- `response.trace` with `events`, `tool_calls`, `tool_results` (agent thought process)

Commit the generated `*.json` files after running the script so evaluators can inspect the traces.
