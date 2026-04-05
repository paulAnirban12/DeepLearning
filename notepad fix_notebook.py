import json

filename = "BERT_Text_Classifier.ipynb"

with open(filename, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove broken widget metadata if present
if "metadata" in nb and "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]

with open(filename, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Cleaned notebook metadata.")
