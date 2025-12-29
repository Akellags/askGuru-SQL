Perfect! Here's your complete guide to using **Defog SQLCoder-70B** for Oracle EBS NL2SQL fine-tuning:

## **1. Model Information**

**SQLCoder-70B Details:**
- **Base Model**: DeepSeek-Coder-70B-Base (but Defog is US company)
- **License**: Apache 2.0 ✅
- **HuggingFace**: `defog/sqlcoder-70b-alpha`
- **Size**: ~140GB (FP16), ~70GB (8-bit), ~35GB (4-bit)
- **Context**: 16K tokens
- **Specialization**: Trained specifically on text-to-SQL tasks

## **2. Installation & Setup**

```bash
# Install required packages
pip install torch transformers accelerate peft bitsandbytes
pip install datasets trl wandb  # For training
pip install vllm  # For inference (optional but recommended)

# Login to HuggingFace
huggingface-cli login
```

## **3. Download the Model**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "defog/sqlcoder-70b-alpha"

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download model (for LoRA fine-tuning)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically distribute across GPUs
    trust_remote_code=True
)
```

## **4. Inference (Before Fine-tuning)**

Test the base model first:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "defog/sqlcoder-70b-alpha"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

def generate_sql(question, schema, model, tokenizer):
    prompt = f"""### Task
Generate a SQL query to answer the following question: `{question}`

### Database Schema
{schema}

### SQL Query
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the SQL part
    sql = sql.split("### SQL Query")[-1].strip()
    return sql

# Example Oracle EBS query
schema = """
CREATE TABLE gl_ledgers (
    ledger_id NUMBER,
    name VARCHAR2(240),
    currency_code VARCHAR2(15)
);

CREATE TABLE gl_balances (
    ledger_id NUMBER,
    code_combination_id NUMBER,
    period_name VARCHAR2(15),
    begin_balance_dr NUMBER,
    begin_balance_cr NUMBER
);
"""

question = "What is the total balance for ledger 'US Primary Ledger' in period JAN-24?"

sql = generate_sql(question, schema, model, tokenizer)
print("Generated SQL:")
print(sql)
```

## **5. LoRA Fine-tuning Setup**

### **A. Prepare Your Training Data**

Format your Oracle EBS examples:

```python
# training_data.jsonl format
# Each line is a JSON object:
"""
{"instruction": "What is the total balance for ledger 'US Primary Ledger'?", 
 "schema": "CREATE TABLE gl_ledgers...", 
 "output": "SELECT SUM(begin_balance_dr - begin_balance_cr) FROM gl_balances..."}
"""

# Load your data
from datasets import load_dataset

dataset = load_dataset('json', data_files='training_data.jsonl')
train_test = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test['train']
eval_dataset = train_test['test']
```

### **B. LoRA Configuration**

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

# LoRA configuration for SQLCoder-70B
lora_config = LoraConfig(
    r=64,                          # Rank - higher for complex SQL
    lora_alpha=128,                # 2x rank
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Should show: trainable params: ~200M / 70B (~0.3%)
```

### **C. Training Configuration**

```python
training_args = TrainingArguments(
    output_dir="./sqlcoder-70b-oracle-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,      # Adjust based on GPU memory
    gradient_accumulation_steps=8,       # Effective batch = 2*8*8GPUs = 128
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",           # Memory efficient
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    bf16=True,                          # Use BF16 on A100
    tf32=True,                          # Enable TF32 on A100
    max_grad_norm=1.0,
    ddp_find_unused_parameters=False,
    report_to="wandb",                  # Track experiments
    run_name="sqlcoder70b-oracle-ebs"
)
```

### **D. Format Training Prompts**

```python
def format_prompt(example):
    """Format training examples to match SQLCoder's expected format"""
    prompt = f"""### Task
Generate a SQL query to answer the following question: `{example['instruction']}`

### Database Schema
{example['schema']}

### SQL Query
{example['output']}"""
    return {"text": prompt}

# Apply formatting
train_dataset = train_dataset.map(format_prompt)
eval_dataset = eval_dataset.map(format_prompt)
```

### **E. Start Training**

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,  # Keep False for SQL to avoid mixing queries
)

# Start training
trainer.train()

# Save the final LoRA adapter
trainer.save_model("./sqlcoder-70b-oracle-final")
```

## **6. Multi-GPU Training (8x A100)**

Add this to your training script:

```bash
# train.sh
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_sqlcoder.py \
    --model_name defog/sqlcoder-70b-alpha \
    --output_dir ./sqlcoder-70b-oracle-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --bf16 \
    --ddp_find_unused_parameters False
```

## **7. Inference with Fine-tuned Model**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "defog/sqlcoder-70b-alpha",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load your LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./sqlcoder-70b-oracle-final"
)

tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-70b-alpha")

# Now use generate_sql() function from earlier
sql = generate_sql(question, schema, model, tokenizer)
```

## **8. Production Deployment with vLLM**

For fast inference in production:

```bash
# Install vLLM
pip install vllm

# Merge LoRA weights (one-time)
python merge_lora.py  # Script below
```

```python
# merge_lora.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "defog/sqlcoder-70b-alpha",
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(base_model, "./sqlcoder-70b-oracle-final")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./sqlcoder-70b-oracle-merged")
tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-70b-alpha")
tokenizer.save_pretrained("./sqlcoder-70b-oracle-merged")
```

```python
# Serve with vLLM
from vllm import LLM, SamplingParams

llm = LLM(
    model="./sqlcoder-70b-oracle-merged",
    tensor_parallel_size=4,  # Use 4 GPUs for inference
    dtype="bfloat16"
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.95,
    max_tokens=512
)

# Generate
outputs = llm.generate([prompt], sampling_params)
sql = outputs[0].outputs[0].text
```

## **9. Expected Training Stats**

With your 8x A100 80GB setup:

- **Memory per GPU**: ~55-60GB
- **Training time**: 12-18 hours (depends on dataset size)
- **Throughput**: ~8-10 samples/second
- **Total adapter size**: ~300MB

## **10. Evaluation**

```python
def evaluate_sql(model, tokenizer, test_dataset):
    """Evaluate on your test set"""
    correct = 0
    total = len(test_dataset)
    
    for example in test_dataset:
        generated_sql = generate_sql(
            example['instruction'],
            example['schema'],
            model,
            tokenizer
        )
        
        # Compare with ground truth (you may need fuzzy matching)
        if normalize_sql(generated_sql) == normalize_sql(example['output']):
            correct += 1
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy

def normalize_sql(sql):
    """Normalize SQL for comparison"""
    import sqlparse
    return sqlparse.format(sql, reindent=True, keyword_case='upper').strip()
```

## **Key Advantages for Your Use Case**

1. ✅ **Apache 2.0 license** - Deploy to clients freely
2. ✅ **US company (Defog)** - Meets compliance
3. ✅ **SQL-specialized** - Better than general models
4. ✅ **Proven performance** - Tops SQL benchmarks
5. ✅ **Your 8x A100** - Perfect fit for LoRA fine-tuning

## **Next Steps**

1. Download the model and test base performance
2. Format your existing Oracle EBS training data (you mentioned 1,500+ examples)
3. Start with a small pilot (100-200 examples) to validate
4. Scale to full training
5. Deploy with vLLM for production

Want me to help you convert your existing Oracle training data to the SQLCoder format, or need the complete training script?