import json

with open("shap_summary.json", "r") as f:
    data = json.load(f)

result = {}

for entry in data:
    model = entry["model"]
    class_name = entry["class"]
    genes = entry["genes"]
    
    # Initialize model key if not exist
    if "model" not in result:
        result["model"] = model
        
    # Add gene list for class
    result[class_name] = genes

# Save the new JSON
with open("shap_summary_grouped.json", "w") as f:
    json.dump(result, f, indent=2)

print("Saved grouped JSON to shap_summary_grouped.json")
