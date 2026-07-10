# Dialeng Performance Benchmarks

Run the reproducible server-side baseline from the repository root:

```bash
uv run python benchmarks/performance_baseline.py
```

The default run measures notebook serialization, notebook loading, prompt
context construction, and file-explorer directory scans for 20, 100, and 250
cells. Results are JSON so they can be compared in CI or attached to a pull
request.

```bash
uv run python benchmarks/performance_baseline.py \
  --sizes 20,100,250 \
  --repeats 10 \
  --output tmp/performance-baseline.json
```

These are microbenchmarks, not release gates. Browser benchmarks for Monaco,
scrolling, rich outputs, and WebSocket recovery should be collected with the
end-to-end fixtures planned in the performance roadmap.
