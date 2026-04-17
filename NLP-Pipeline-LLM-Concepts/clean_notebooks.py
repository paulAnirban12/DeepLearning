import json
from pathlib import Path

for nb_path in Path(".").rglob("*.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if "metadata" in nb and "widgets" in nb["metadata"]:
        del nb["metadata"]["widgets"]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Cleaned: {nb_path}")