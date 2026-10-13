"""Fetch a bounded public routing sample via Hugging Face's dataset viewer API."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

root = Path(__file__).resolve().parent / "artifacts" / "real_trace"
root.mkdir(parents=True, exist_ok=True)
dataset = "mistrjirka/Predicting-Experts-for-MOE-Suite"
api = "https://datasets-server.huggingface.co/rows"
sample_size = 1000
rows = []
revision = None

for offset in range(0, sample_size, 100):
    url = api + "?" + urlencode({
        "dataset": dataset, "config": "token_cube", "split": "train",
        "offset": offset, "length": 100,
    })
    for attempt in range(3):
        try:
            with urlopen(url, timeout=45) as response:
                revision = response.headers.get("x-revision", revision)
                payload = json.load(response)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    if payload.get("partial") or len(payload["rows"]) != 100:
        raise RuntimeError(f"Incomplete dataset page at offset {offset}")
    for item in payload["rows"]:
        row = item["row"]
        routes = row["route_selected"]
        if len(routes) != 75 or any(len(layer) != 8 for layer in routes):
            raise RuntimeError(f"Invalid route shape at row {item['row_idx']}")
        rows.append({
            "row_idx": item["row_idx"],
            "example_id": row["example_id"],
            "task_id": row["task_id"],
            "generated_position": row["generated_position"],
            "route_selected": routes,
        })
    print(f"Fetched {len(rows)} tokens", flush=True)

with (root / "tokens.jsonl").open("w", encoding="utf-8") as out:
    for row in rows:
        out.write(json.dumps(row, separators=(",", ":")) + "\n")

metadata = {
    "dataset": dataset,
    "dataset_url": "https://huggingface.co/datasets/" + dataset,
    "api": api,
    "config": "token_cube",
    "split": "train",
    "offset": 0,
    "length": len(rows),
    "revision": revision,
    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Router decisions from a GLM-5.2 variant on synthetic coding tasks; not DeepSeekV3 simulator routing",
}
(root / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"saved": len(rows), "revision": revision}))
