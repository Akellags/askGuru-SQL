"""
Multiple SQL Generation + Critic Selection
Implements the ensemble SQL generation with critic selection mechanism
"""

import os
import json
import torch
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from tqdm import tqdm
import datetime
from utils.common_utils import read_json, write_json


class MultiCandidateSQLGenerator:
    """
    Generate multiple SQL candidate queries using ensemble methods
    """
    
    def __init__(self, model_path: str, lora_path: str = "", batch_size: int = 4, 
                 device: str = 'auto', num_candidates: int = 5):
        self.model_path = model_path
        self.lora_path = lora_path
        self.batch_size = batch_size
        self.device = device
        self.num_candidates = num_candidates
        self.model = None
        self.tokenizer = None
    
    def model_init(self, use_flash_attention: bool = True):
        """Load model and optional LoRA adapter"""
        config = AutoConfig.from_pretrained(
            self.model_path,
            use_cache=True,
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attention else None,
            device_map=self.device
        )
        
        if len(self.lora_path) > 0:
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.model, self.lora_path, 
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            self.model = self.model.merge_and_unload()
        
        self.model.eval()
    
    def generate_candidates_beam_search(self, texts: List[str], 
                                       num_beams: int = None) -> List[List[str]]:
        """
        Generate multiple candidates using beam search
        
        Args:
            texts: List of input prompts
            num_beams: Number of beam candidates (default: self.num_candidates)
        
        Returns:
            List of candidate lists, one list per input text
        """
        if num_beams is None:
            num_beams = self.num_candidates
        
        model_inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                length_penalty=0.0,
                early_stopping=False,
            )
        
        torch.cuda.empty_cache()
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        candidates_list = []
        for i in range(len(texts)):
            candidates = responses[i*num_beams:(i+1)*num_beams]
            candidates_list.append(candidates)
        
        return candidates_list
    
    def generate_candidates_temperature_sampling(self, texts: List[str], 
                                                 temperature: float = 0.7) -> List[List[str]]:
        """
        Generate diverse candidates using temperature-based sampling
        
        Args:
            texts: List of input prompts
            temperature: Sampling temperature (0.7 for diversity, 1.0 for balance)
        
        Returns:
            List of candidate lists
        """
        candidates_list = [[] for _ in range(len(texts))]
        
        for _ in range(self.num_candidates):
            model_inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                )
            
            torch.cuda.empty_cache()
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for i, response in enumerate(responses):
                candidates_list[i].append(response)
        
        return candidates_list
    
    def generate_candidates_hybrid(self, texts: List[str]) -> List[List[str]]:
        """
        Hybrid approach: combine beam search + temperature sampling
        
        Args:
            texts: List of input prompts
        
        Returns:
            List of candidate lists
        """
        num_beam_candidates = max(2, self.num_candidates // 2)
        num_sample_candidates = self.num_candidates - num_beam_candidates
        
        beam_candidates = self.generate_candidates_beam_search(texts, num_beam_candidates)
        sample_candidates = self.generate_candidates_temperature_sampling(texts, 0.8)
        
        combined = []
        for i in range(len(texts)):
            candidates = beam_candidates[i] + sample_candidates[i][:num_sample_candidates]
            combined.append(candidates)
        
        return combined


class SQLCritic:
    """
    Critic model that selects the best SQL from candidates
    """
    
    def __init__(self, model_path: str, lora_path: str = "", device: str = 'auto'):
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def model_init(self, use_flash_attention: bool = True):
        """Load critic model"""
        config = AutoConfig.from_pretrained(
            self.model_path,
            use_cache=True,
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading critic model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attention else None,
            device_map=self.device
        )
        
        if len(self.lora_path) > 0:
            print("Loading critic LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.model, self.lora_path, 
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            self.model = self.model.merge_and_unload()
        
        self.model.eval()
    
    def select_best_candidate(self, question: str, db_schema: str, 
                            evidence: str, candidates: List[str], 
                            exec_results: List[str]) -> Tuple[int, str]:
        """
        Use the critic to select the best SQL candidate
        
        Args:
            question: Natural language question
            db_schema: Database schema description
            evidence: Supporting evidence
            candidates: List of candidate SQL queries
            exec_results: Execution results for each candidate
        
        Returns:
            Tuple of (selected_index, selected_sql)
        """
        candidate_text = ""
        for i, (sql, result) in enumerate(zip(candidates, exec_results)):
            candidate_text += f"\n[Candidate {i+1}】\n{sql}\n[Execution Result]\n{result}\n"
        
        prompt = f"""You are a SQL expert，need tocompare以下candidateSQL并selectoptimal/best的。

【database schema】
{db_schema}

[Reference Information]
{evidence}

[User Question]
{question}

==========
{candidate_text}

请outputselect的candidatenumber（如：candidate1、candidate2等）："""
        
        model_input = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                **model_input,
                max_new_tokens=10,
                temperature=0.1,
                num_beams=1,
            )
        
        response = self.tokenizer.decode(
            generated[0][len(model_input.input_ids[0]):],
            skip_special_tokens=True
        ).strip()
        
        try:
            selected_idx = int(''.join(filter(str.isdigit, response))) - 1
            selected_idx = max(0, min(selected_idx, len(candidates) - 1))
        except:
            selected_idx = 0
        
        return selected_idx, candidates[selected_idx]


class MultiCandidateInference:
    """
    Complete inference pipeline with multiple candidate generation and critic selection
    """
    
    def __init__(self, generator_path: str, critic_path: str = None,
                 lora_path: str = "", critic_lora_path: str = "",
                 num_candidates: int = 5, use_critic: bool = True):
        self.generator = MultiCandidateSQLGenerator(
            generator_path, lora_path, num_candidates=num_candidates
        )
        
        self.critic = None
        if use_critic and critic_path:
            self.critic = SQLCritic(critic_path, critic_lora_path)
        elif use_critic:
            self.critic = SQLCritic(generator_path, lora_path)
        
        self.num_candidates = num_candidates
        self.use_critic = use_critic
    
    def run_inference(self, test_data_path: str, output_dir: str = "output",
                     expr_version: str = "multi_candidate_v1",
                     generation_method: str = "hybrid"):
        """
        Run multi-candidate generation + critic selection on test data
        
        Args:
            test_data_path: Path to test data JSON
            output_dir: Output directory for results
            expr_version: Experiment version identifier
            generation_method: "beam", "sampling", or "hybrid"
        """
        self.generator.model_init()
        if self.critic:
            self.critic.model_init()
        
        os.makedirs(output_dir, exist_ok=True)
        today = datetime.date.today().strftime('%Y%m%d')
        save_path = os.path.join(
            output_dir, 
            f'{expr_version}_{today}_results.json'
        )
        
        final_result = []
        if os.path.exists(save_path):
            final_result = read_json(save_path)
        
        print("------Multi-Candidate Generation & Critic Selection-------")
        eval_json = read_json(test_data_path)
        print(f"{len(eval_json)} samples to process...")
        
        texts, item_temps = [], []
        
        for idx, item in tqdm(enumerate(eval_json)):
            if idx < len(final_result) and 'pred_sql' in final_result[idx]:
                continue
            
            conversations = item['conversations'][:1]
            text = self.generator.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)
            item_temps.append((idx, item))
            
            if (len(texts) % self.generator.batch_size == 0) or idx == len(eval_json) - 1:
                if generation_method == "beam":
                    candidates_list = self.generator.generate_candidates_beam_search(texts)
                elif generation_method == "sampling":
                    candidates_list = self.generator.generate_candidates_temperature_sampling(texts)
                else:
                    candidates_list = self.generator.generate_candidates_hybrid(texts)
                
                for i, (cur_idx, item_temp) in enumerate(item_temps):
                    candidates = candidates_list[i]
                    
                    selected_idx = 0
                    if self.critic:
                        try:
                            selected_idx, selected_sql = self.critic.select_best_candidate(
                                item_temp['conversations'][0]['content'],
                                item_temp.get('db_schema', ''),
                                item_temp.get('evidence', ''),
                                candidates,
                                [""] * len(candidates)
                            )
                        except Exception as e:
                            print(f"Critic failed: {e}, using first candidate")
                            selected_sql = candidates[0]
                    else:
                        selected_sql = candidates[0]
                    
                    item_temp['pred_sql'] = selected_sql
                    item_temp['all_candidates'] = candidates
                    item_temp['selected_candidate_idx'] = selected_idx
                    item_temp['sql'] = item_temp['conversations'][1]['content'] \
                        if len(item_temp['conversations']) > 1 else ""
                    
                    if cur_idx < len(final_result):
                        final_result[cur_idx] = item_temp
                    else:
                        final_result.append(item_temp)
                
                texts = []
                item_temps = []
                write_json(save_path, final_result)
        
        print(f"\nResults saved to {save_path}")
        return final_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_path", type=str, required=True)
    parser.add_argument("--critic_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--critic_lora_path", type=str, default="")
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--expr_version", type=str, default="multi_candidate_v1")
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--generation_method", type=str, default="hybrid",
                       choices=["beam", "sampling", "hybrid"])
    parser.add_argument("--use_critic", action="store_true", default=True)
    
    args = parser.parse_args()
    
    inference = MultiCandidateInference(
        generator_path=args.generator_path,
        critic_path=args.critic_path,
        lora_path=args.lora_path,
        critic_lora_path=args.critic_lora_path,
        num_candidates=args.num_candidates,
        use_critic=args.use_critic
    )
    
    inference.run_inference(
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        expr_version=args.expr_version,
        generation_method=args.generation_method
    )
