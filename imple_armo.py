import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from judger import PREFERENCE_ATTRIBUTES
import numpy as np
import copy
import random
from tqdm import tqdm
from typing import List, Dict
import time
from torch.utils.data import DataLoader, Dataset

from model import MABRewardGenerator
from judger import PreferenceCollector
from rm_eval import PersonalizationRMEvaluator

# ---------------- Training Components ----------------
class PreferenceDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if all(k in rec for k in ("y1", "y2", "label")):
                        self.samples.append((rec["y1"], rec["y2"], rec["label"]))
                except json.JSONDecodeError:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def pairwise_loss(s1: torch.Tensor, s2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    margin = s1.view(-1) - s2.view(-1)
    y = label.view(-1).float()
    return F.binary_cross_entropy_with_logits(margin, y)

def reinit_reward_model(model: nn.Module, init_state_dict: dict):
    model.load_state_dict(copy.deepcopy(init_state_dict))


def construct_prompt_with_preference(query: str, attribute: str) -> str:
    instruction = PREFERENCE_ATTRIBUTES.get(attribute, "")
    if instruction:
        return f"{query} {instruction}"
    else:
        return query

def generate_with_fixed_weight(generator: MABRewardGenerator, query: str,
                              max_new_tokens: int, pre_screen_beam_width: int, reward_weight: float,
                              nu_mab: float = 1.0, lambda_mab: float = 1.0, attribute: str = "creative") -> str:
    # Construct enhanced prompt with preference instruction
    enhanced_prompt = construct_prompt_with_preference(query, attribute)

    try:
        ids1, _ = generator.generate(
            prompt=enhanced_prompt,
            weight=reward_weight,
            max_new_tokens=max_new_tokens,
            pre_screen_beam_width=pre_screen_beam_width,
            nu_mab=nu_mab,
            lambda_mab=lambda_mab
        )

        raw_response = generator.tokenizer.decode(ids1, skip_special_tokens=True)
        # Strip prompt echo (including preference instruction)
        if raw_response.startswith(enhanced_prompt):
            response = raw_response[len(enhanced_prompt):].lstrip()
        else:
            response = raw_response

        return response
    except Exception:
        return ""

# Baseline generation functions removed - not needed for ArmoRM evaluation

def train_reward(generator: MABRewardGenerator, judge: PreferenceCollector,
                 attribute: str, training_queries: List[str], cfg) -> str:

    init_model_weights = copy.deepcopy(generator.reward_model.state_dict())

    pref_dir = os.path.join(cfg.output_dir, "preferences")
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(pref_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    pref_path = os.path.join(pref_dir, "preferences.jsonl")

    for idx, q in enumerate(tqdm(training_queries, desc=f"Training")):

        if idx >= 1:
            reinit_reward_model(generator.reward_model, init_model_weights)
            dataset = PreferenceDataset(pref_path)
            if len(dataset) > 0:
                loader = DataLoader(
                    dataset,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=0
                )

                sample_count = len(dataset)
                wd = 1.0 / (sample_count + 50)

                optimizer = torch.optim.AdamW(
                    generator.reward_model.parameters(),
                    lr=cfg.lr,
                    weight_decay=wd
                )

                generator.reward_model.train()

                print(f"   Query {idx+1}/{len(training_queries)}: wd={wd:.4f}")

                for ep in range(cfg.epochs_per_query):
                    total_loss = 0.0
                    batch_count = 0

                    for y1, y2, lab in loader:
                        lab_t = lab.to(cfg.reward_model_device).float() \
                                if isinstance(lab, torch.Tensor) else torch.tensor(lab, device=cfg.reward_model_device).float()

                        emb1 = generator.get_embedding(list(y1)).to(cfg.reward_model_device)
                        emb2 = generator.get_embedding(list(y2)).to(cfg.reward_model_device)
                        s1 = generator.reward_model(emb1)
                        s2 = generator.reward_model(emb2)

                        # Compute pairwise preference loss
                        loss = pairwise_loss(s1, s2, lab_t)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        batch_count += 1

                        # Clean up intermediate tensors to prevent memory accumulation
                        del emb1, emb2, s1, s2, loss, lab_t

                        # Clear gradients and GPU cache every few batches to prevent accumulation
                        if batch_count % 5 == 0:  # Every 5 batches
                            torch.cuda.empty_cache()
                            # Also clear BLLM internal cache if it exists
                            if hasattr(generator.BLLM, 'past_key_values'):
                                generator.BLLM.past_key_values = None

                    # Print loss for every epoch
                    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
                    print(f"     Epoch {ep+1:3d}/{cfg.epochs_per_query}: loss = {avg_loss:.6f}")

                    # Clear GPU cache after each epoch to prevent accumulation
                    torch.cuda.empty_cache()

                # Clear optimizer and dataset after training to free memory
                del optimizer, dataset, loader
                torch.cuda.empty_cache()
        else:
            pass  

        # 3. Generate two responses using current RM with specified weight
        enhanced_prompt = construct_prompt_with_preference(q, cfg.attribute)

        try:
            ids1, ids2 = generator.generate(
                prompt=enhanced_prompt,
                weight=cfg.reward_weight,
                max_new_tokens=cfg.max_new_tokens,
                pre_screen_beam_width=cfg.pre_screen_beam_width,
                nu_mab=cfg.nu_mab,
                lambda_mab=cfg.lambda_mab
            )
        except Exception:
            continue

        # Decode & strip prompt echo
        raw1 = generator.tokenizer.decode(ids1, skip_special_tokens=True)
        raw2 = generator.tokenizer.decode(ids2, skip_special_tokens=True)

        # Strip the enhanced prompt (query + preference instruction) from the response
        response_1 = raw1[len(enhanced_prompt):].lstrip() if raw1.startswith(enhanced_prompt) else raw1
        response_2 = raw2[len(enhanced_prompt):].lstrip() if raw2.startswith(enhanced_prompt) else raw2

        # Clear generation-related variables and GPU cache
        del ids1, ids2, raw1, raw2
        torch.cuda.empty_cache()

        # 4. Collect preference data
        try:
            pref = judge.collect_single({
                "attribute": attribute,
                "query": q,  # Use original query for API prompt (avoid redundancy)
                "enhanced_query": enhanced_prompt,  # Pass enhanced_prompt for training data
                "response_1": response_1,
                "response_2": response_2,
            })
            if pref:
                with open(pref_path, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps(pref, ensure_ascii=False) + "\n")

            # Clear preference collection variables
            del pref
        except Exception:
            continue

        # Clear response variables after preference collection
        del response_1, response_2, enhanced_prompt

        # Save final checkpoint only (after last query)
        if idx == len(training_queries) - 1:
            ckpt_f = os.path.join(ckpt_dir, f"final_model.pt")
            torch.save(generator.reward_model.state_dict(), ckpt_f)

    # Return final checkpoint path
    final_ckpt = os.path.join(ckpt_dir, "final_model.pt")
    return final_ckpt

def evaluate(generator: MABRewardGenerator, queries: List[str], attribute: str,
             max_new_tokens: int, pre_screen_beam_width: int, cfg) -> Dict:

    total = 0
    failed_generations = 0

    # Create directory for saving responses
    responses_dir = os.path.join(cfg.output_dir, "responses")
    os.makedirs(responses_dir, exist_ok=True)
    responses_file = os.path.join(responses_dir, f"responses_armo_rw_{cfg.reward_weight:.2f}.jsonl")

    # Generate responses for all queries
    for query in tqdm(queries, desc=f"Generating responses"):
        # Generate response with specified weight
        response = generate_with_fixed_weight(
            generator, query, max_new_tokens, pre_screen_beam_width, cfg.reward_weight,
            nu_mab=cfg.nu_mab, lambda_mab=cfg.lambda_mab, attribute=attribute
            )

        if not response:
            failed_generations += 1
            continue

        # Save generated response (only ours, no baseline needed)
        response_data = {
            "query": query,
            "ours": response
        }

        with open(responses_file, "a", encoding="utf-8") as rf:
            rf.write(json.dumps(response_data, ensure_ascii=False) + "\n")

        total += 1

        # Clear evaluation query-specific variables
        del response, response_data

        # Clear MAB generation history (each evaluation query is independent)
        if hasattr(generator, 'last_topk_logits1'):
            generator.last_topk_logits1 = None
        if hasattr(generator, 'last_rewards1'):
            generator.last_rewards1 = None
        if hasattr(generator, 'last_topk_logits2'):
            generator.last_topk_logits2 = None
        if hasattr(generator, 'last_rewards2'):
            generator.last_rewards2 = None

        # Clear model internal caches periodically during evaluation
        if total % 50 == 0:  # Every 50 evaluation queries
            # Clear LLM KV cache
            if hasattr(generator.LLM, 'past_key_values'):
                generator.LLM.past_key_values = None
            # Clear BLLM cache
            if hasattr(generator.BLLM, 'past_key_values'):
                generator.BLLM.past_key_values = None
            # Force GPU cache cleanup
            torch.cuda.empty_cache()

        # Light cleanup every 10 queries
        elif total % 10 == 0:
            torch.cuda.empty_cache()

    # Use ArmoRM to evaluate all generated responses
    print(f"\n🤖 Evaluating {total} responses with ArmoRM...")
    print(f"   💾 ArmoRM will use {cfg.llm_device} (LLM device, since generation is complete)")

    try:
        if hasattr(generator, 'LLM'):
            del generator.LLM
        if hasattr(generator, 'BLLM'):
            del generator.BLLM
        if hasattr(generator, 'bllm_tokenizer'):
            del generator.bllm_tokenizer

        torch.cuda.empty_cache()
        rm_evaluator = PersonalizationRMEvaluator(
            attribute=attribute,
            device=cfg.llm_device,  # Use LLM device since generation is complete
            rm_batch_size=cfg.rm_batch_size if hasattr(cfg, 'rm_batch_size') else 32
        )

        avg_reward = rm_evaluator.get_rm_eval(responses_file, response_key="ours")

        summary = {
            "weight": cfg.reward_weight,
            "regularization": "weight_decay",
            "total": total,
            "avg_reward": avg_reward,
            "failed_generations": failed_generations,
            "responses_file": responses_file,
            "evaluation_method": "ArmoRM"
        }

        print(f"   Average ArmoRM reward: {avg_reward:.6f}")
        print(f"   Total responses: {total}")
        print(f"   Failed generations: {failed_generations}")
        print(f"   Responses saved to: {responses_file}")

        return summary

    except Exception as e:
        return {
            "weight": cfg.reward_weight,
            "regularization": "weight_decay",
            "total": total,
            "avg_reward": 0.0,
            "failed_generations": failed_generations,
            "responses_file": responses_file,
            "evaluation_method": "ArmoRM",
            "error": str(e)
        }

def run(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    print("=" * 50)
    print(f"🧪 Starting Experiment (rw={cfg.reward_weight})")
    print("=" * 50)

    if not os.path.exists(cfg.data_file):
        raise FileNotFoundError(f"Data file not found: {cfg.data_file}")

    all_data = json.load(open(cfg.data_file, "r", encoding="utf-8"))

    if isinstance(all_data, list) and len(all_data) > 0:
        if isinstance(all_data[0], dict) and "question" in all_data[0]:
            all_queries = [item["question"] for item in all_data]
            pass  
        else:
            all_queries = all_data
    else:
        all_queries = all_data

    if cfg.train_samples > len(all_queries):
        training_queries = all_queries.copy()
    else:
        training_queries = random.sample(all_queries, cfg.train_samples)

    eval_queries = all_queries.copy()

    os.makedirs(cfg.output_dir, exist_ok=True)

    result_file = os.path.join(cfg.output_dir, "result.txt")
    with open(result_file, 'w', encoding='utf-8') as sf:
            sf.write("EXPERIMENT SETTINGS\n")
            sf.write("=" * 50 + "\n")
            sf.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            sf.write(f"Data File: {cfg.data_file}\n")
            sf.write(f"LLM: {cfg.llm_path}\n")
            sf.write(f"Target attribute: {cfg.attribute}\n")
            sf.write(f"NU_MAB: {cfg.nu_mab}\n")
            sf.write(f"Pre-screen beam width: {cfg.pre_screen_beam_width}\n")
            sf.write(f"Max new tokens: {cfg.max_new_tokens}\n")
            sf.write(f"Regularization: Dynamic weight_decay = 1/(N + 50)\n")
            sf.write(f"Learning rate: {cfg.lr}\n")
            sf.write(f"Batch size: {cfg.batch_size}\n")
            sf.write(f"Epochs per query: {cfg.epochs_per_query}\n")
            sf.write(f"Normalize sentence embeddings: {cfg.normalize_sentence_embeddings}\n")
            sf.write(f"Reward Weight: {cfg.reward_weight}\n")
            sf.write(f"Training Samples: {cfg.train_samples}\n")
            sf.write(f"Random Seed: {cfg.seed}\n")

    try:
        judge = PreferenceCollector(
            api_key=cfg.openai_api_key,
            model=cfg.preference_model_name,
            base_url=cfg.preference_api_base_url,
            sleep_time=cfg.preference_api_sleep_time,
            max_retries=cfg.preference_api_max_retries,
        )

        # Phase 1: Training
        print("Phase 1: Training reward model with dynamic weight_decay...")

        generator = MABRewardGenerator(
            llm_path=cfg.llm_path,
            reward_model_checkpoint_path=None,  # Start fresh
            embedding_model_path=cfg.embedding_model_path,
            llm_device=cfg.llm_device,
            reward_model_device=cfg.reward_model_device,
            embedding_model_device=cfg.embedding_model_device,
            torch_dtype=torch.float16,
            normalize_sentence_embeddings=cfg.normalize_sentence_embeddings,
        )

        checkpoint_path = train_reward(
            generator=generator,
            judge=judge,
            attribute=cfg.attribute,
            training_queries=training_queries,
            cfg=cfg
        )

        # Phase 2: Evaluation
        print("Phase 2: Evaluating trained model...")

        if os.path.exists(checkpoint_path):
            generator.reward_model.load_state_dict(
                torch.load(checkpoint_path, map_location=cfg.reward_model_device)
            )
            pass

        result = evaluate(
            generator=generator,
            queries=eval_queries,
            attribute=cfg.attribute,
            max_new_tokens=cfg.max_new_tokens,
            pre_screen_beam_width=cfg.pre_screen_beam_width,
            cfg=cfg
        )

        with open(result_file, 'a', encoding='utf-8') as rf:
            rf.write("\n" + "=" * 50 + "\n")
            rf.write("EXPERIMENT RESULTS\n")
            rf.write("=" * 50 + "\n")
            rf.write(f"ArmoRM Score: {result['avg_reward']:.6f}\n")
            rf.write(f"Total Responses: {result['total']}\n")
            rf.write(f"Training Queries: {len(training_queries)}\n")
            rf.write(f"Evaluation Queries: {len(eval_queries)}\n")
            if result['failed_generations'] > 0:
                rf.write(f"Failed Generations: {result['failed_generations']}\n")
            rf.write("=" * 50 + "\n")

        print(f"Completed: {result['avg_reward']:.6f} ArmoRM score ({result['total']} responses)")
        print(f"Responses: {result['responses_file']}")
        print(f"Results: {result_file}")

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Weight Decay Regularization Experiment")

    # Basic configuration
    parser.add_argument("--llm_path", required=True, 
                        help="Path to LLM model")
    parser.add_argument("--openai_api_key", required=True, 
                        help="OpenAI API key")
    parser.add_argument("--attribute", default="creative",
                       help="Target attribute for alignment (creative, verbose, formal, etc.)")
    parser.add_argument("--data_file", required=True,
                       help="Path to JSON data file")
    parser.add_argument("--train_samples", type=int, required=True,
                       help="Number of samples to randomly select for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--embedding_model_path", default="Qwen/Qwen3-Embedding-0.6B",
                       help="Path to embedding model")
    parser.add_argument("--normalize_sentence_embeddings", action="store_true")
    parser.add_argument("--reward_weight", type=float, required=True,
                       help="Reward weight for generation (required)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--rm_batch_size", type=int, default=128,
                        help="ArmoRM evaluation batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--epochs_per_query", type=int, default=100,
                        help="Training epochs per query")

    parser.add_argument("--llm_device", default="cuda:1")
    parser.add_argument("--reward_model_device", default="cuda:0")
    parser.add_argument("--embedding_model_device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--pre_screen_beam_width", type=int, default=40)
    parser.add_argument("--nu_mab", type=float, default=1.0,
                       help="Exploration parameter for MAB (nu in TTA algorithm)")
    parser.add_argument("--lambda_mab", type=float, default=1.0,
                       help="Regularization parameter for covariance matrix")

    parser.add_argument("--preference_model_name", required=True)
    parser.add_argument("--preference_api_base_url", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preference_api_sleep_time", type=float, default=1.0)
    parser.add_argument("--preference_api_max_retries", type=int, default=3)

    cfg = parser.parse_args()
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print(f"Experiment failed: {e}")
        raise