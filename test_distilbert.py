import os
from transformers import pipeline

model_path = os.path.join('models', 'distilbert_ticket_type')
print('Path exists:', os.path.exists(model_path))
print('Files:', os.listdir(model_path) if os.path.exists(model_path) else 'NOT FOUND')

try:
    clf = pipeline('text-classification', model=model_path, tokenizer=model_path, top_k=None)
    print('✅ Loaded OK')
    result = clf('my account is not working')
    print('Result:', result)
except Exception as e:
    print('❌ Error:', e)
