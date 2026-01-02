import json

path = 'data/oracle_sft_conversations/oracle_sft_conversations_full.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    tables = set()
    for ex in data:
        if 'meta' in ex and 'tables' in ex['meta']:
            # Some entries might be string if they were parsed from single table examples
            for t in ex['meta']['tables']:
                tables.add(t.upper())
    
    print(f"Total Unique Tables in Dataset: {len(tables)}")
    with open('all_tables.txt', 'w', encoding='utf-8') as f_out:
        for t in sorted(list(tables)):
            f_out.write(t + '\n')
    print("Full list written to all_tables.txt")
