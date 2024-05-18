import os
order="python3 preprocess_features.py"
os.system(order)
order="python3 preprocess_text.py"
os.system(order)
order="python3 preprocess_relations.py"
os.system(order)