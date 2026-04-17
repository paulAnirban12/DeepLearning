import json

filename = "NLP_Pipeline_with_Tokenization,Prompt_Classification_&_Semantic_Embeddings.ipynb"

with open(filename, "r", encoding="utf-8") as f:
    nb = json.load(f)

if "metadata" in nb and "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]

with open(filename, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Cleaned notebook metadata.")