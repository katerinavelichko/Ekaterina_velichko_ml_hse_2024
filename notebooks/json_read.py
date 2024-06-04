import os
import json

output_folder = 'output_messages_yaml'
os.makedirs(output_folder, exist_ok=True)

with open('result_yaml_2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

messages = data['messages']
total_messages = len(messages)
chunk_size = 2000

for i in range(0, total_messages, chunk_size):
    chunk = messages[i:i + chunk_size]
    chunk_folder = os.path.join(output_folder, f'yaml_{i // chunk_size}')
    os.makedirs(chunk_folder, exist_ok=True)
    chunk_filename = os.path.join(chunk_folder, f"yaml_{i // chunk_size}.json")
    with open(chunk_filename, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, ensure_ascii=False, indent=4)
