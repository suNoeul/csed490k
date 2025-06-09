import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from searcher import Searcher
import json

if __name__ == "__main__":
    cluster_path = "cluster/a6000/n1_g4.json"
    model_path = "model/gpt-1.json" # Change this to your model path
    with open(cluster_path, "r") as f:
        cluster_metadata = f.read()
    with open(model_path, "r") as f:
        model_metadata = f.read()

    cluster_metadata = json.loads(cluster_metadata)
    model_metadata = json.loads(model_metadata)
    
    result = Searcher.search(cluster_metadata, model_metadata)

    print(result)
