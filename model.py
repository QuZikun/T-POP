import os
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np

class Network(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 1024, depth: int = 1,
                 init_params: Optional[List[torch.Tensor]] = None):
        super().__init__()
        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for _ in range(depth - 1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))
        if init_params is None:
            for layer in self.layer_list:
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        else:
            for i, layer in enumerate(self.layer_list):
                layer.weight.data = init_params[2*i]
                layer.bias.data   = init_params[2*i + 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layer_list[:-1]:
            y = self.activate(layer(y))
        return self.layer_list[-1](y)

def create_attention_mask(seq_len: int, bsz: int = 1) -> torch.Tensor:
    return torch.ones((bsz, seq_len), dtype=torch.long)
def even_chunk(data: torch.Tensor, chunk_size: int = 10):
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:(i + chunk_size)]

class MABRewardGenerator:
    def __init__(
        self,
        llm_path: str,
        reward_model_checkpoint_path: Optional[str],
        embedding_model_path: Optional[str],
        llm_device: str = "cuda:0",
        reward_model_device: str = "cuda:1",
        embedding_model_device: Optional[str] = None,
        torch_dtype=torch.float16,
        normalize_sentence_embeddings: bool = True
    ):
        self.llm_device = llm_device
        self.reward_model_device = reward_model_device
        self.embedding_model_device = embedding_model_device if embedding_model_device is not None else reward_model_device
        self.torch_dtype = torch_dtype
        self.normalize_sentence_embeddings = normalize_sentence_embeddings

        self.LLM = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=self.torch_dtype
        ).to(self.llm_device)
        self.LLM.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        _embedding_model_path = embedding_model_path if embedding_model_path is not None else llm_path
        self.BLLM = AutoModel.from_pretrained(
            _embedding_model_path, torch_dtype=self.torch_dtype
        ).to(self.embedding_model_device)
        self.BLLM.eval()

        self.bllm_tokenizer = AutoTokenizer.from_pretrained(_embedding_model_path)
        if self.bllm_tokenizer.pad_token_id is None:
            self.bllm_tokenizer.pad_token_id = self.bllm_tokenizer.eos_token_id

        embedding_dim = self.BLLM.config.hidden_size
        self.reward_model = Network(input_dim=embedding_dim)

        if reward_model_checkpoint_path and os.path.isfile(reward_model_checkpoint_path):
            state = torch.load(reward_model_checkpoint_path, map_location=self.reward_model_device)
            self.reward_model.load_state_dict(state)
        self.reward_model.to(self.reward_model_device).eval()

        assert self.BLLM.config.hidden_size == self.reward_model.layer_list[0].in_features

    def get_input_ids(self, prompt: str, tokenizer_type: str = "llm") -> torch.Tensor:
        if tokenizer_type == "llm":
            max_len = getattr(self.LLM.config, 'max_position_embeddings', 2048)
            tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_len
            ).input_ids.to(self.llm_device)
        elif tokenizer_type == "bllm":
            max_len = getattr(self.BLLM.config, 'max_position_embeddings', 512)
            tokens = self.bllm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_len
            ).input_ids.to(self.embedding_model_device)
        else:
            raise ValueError("tokenizer_type must be 'llm' or 'bllm'")
        return tokens

    def tokens_to_text(self, tokens: torch.Tensor, tokenizer_type: str = "llm") -> List[str]:
        tokens_cpu = tokens.cpu()
        if tokenizer_type == "llm":
            return self.tokenizer.batch_decode(tokens_cpu, skip_special_tokens=True)
        elif tokenizer_type == "bllm":
            return self.bllm_tokenizer.batch_decode(tokens_cpu, skip_special_tokens=True)
        else:
            raise ValueError("tokenizer_type must be 'llm' or 'bllm'")

    def _mean_pooling(self, model_output_last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output_last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _batch_normalize_l2(self, x: np.ndarray) -> np.ndarray:

        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x
            return x / norm
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm)

    def get_sentence_embedding(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if self.BLLM is None or self.bllm_tokenizer is None:
            raise RuntimeError("BLLM model not loaded. Cannot use local sentence embedding.")

        bllm_max_len = self.BLLM.config.max_position_embeddings if hasattr(self.BLLM.config, 'max_position_embeddings') else 512
        if self.bllm_tokenizer.pad_token_id is None:
            self.bllm_tokenizer.pad_token = self.bllm_tokenizer.eos_token
            self.bllm_tokenizer.pad_token_id = self.bllm_tokenizer.eos_token_id

        encoded_input = self.bllm_tokenizer(
            texts,
            padding=True,      
            truncation=True,   
            max_length=bllm_max_len,
            return_tensors='pt',
            add_special_tokens=True 
        )
        input_ids = encoded_input['input_ids'].to(self.embedding_model_device)
        attention_mask = encoded_input['attention_mask'].to(self.embedding_model_device)

        with torch.no_grad():
            model_output = self.BLLM(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        if hasattr(model_output, 'last_hidden_state'):
            token_embeddings = model_output.last_hidden_state
        elif isinstance(model_output, tuple) and len(model_output) > 0:
            token_embeddings = model_output[0] 
        elif hasattr(model_output, 'hidden_states') and model_output.hidden_states: 
            token_embeddings = model_output.hidden_states[-1]
        else:
            raise AttributeError("BLLM output does not have 'last_hidden_state', is not a tuple "
                                 "with it as first element, or lacks 'hidden_states'. "
                                 "Ensure BLLM is loaded correctly.")

        sentence_embeddings = self._mean_pooling(token_embeddings, attention_mask)

        if self.normalize_sentence_embeddings:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def get_embedding(self, texts: Union[str, List[str]]) -> torch.Tensor:
        return self.get_sentence_embedding(texts)

    def generate(
        self,
        prompt: str,
        weight: float,
        max_new_tokens: int = 128,
        pre_screen_beam_width: int = 20,
        select_idx_history: Optional[List[Tuple[int, int]]] = None,
        lambda_mab: float = 1.0,
        nu_mab: float = 0.1
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if select_idx_history is None:
            select_idx_history = []

        if not hasattr(self, 'V_inv') or not hasattr(self, 'total_reward_params'):
            self.total_reward_params = sum(p.numel() for p in self.reward_model.parameters() if p.requires_grad)
            self.V_inv = (1.0 / lambda_mab) * torch.ones(
                self.total_reward_params,
                dtype=torch.float32,
                device=self.reward_model_device
            )

        initial_input_ids = self.get_input_ids(prompt, tokenizer_type="llm")
        current_tokens_arm1 = initial_input_ids.clone()
        current_tokens_arm2 = initial_input_ids.clone()

        for step in range(max_new_tokens):

            attention_mask_arm1 = torch.ones_like(current_tokens_arm1).to(self.llm_device)
            mout1 = self.LLM(
                input_ids=current_tokens_arm1,
                attention_mask=attention_mask_arm1,
                use_cache=True,
                output_hidden_states=False
            )
            logits1 = mout1.logits[:, -1, :]

            attention_mask_arm2 = torch.ones_like(current_tokens_arm2).to(self.llm_device)
            mout2 = self.LLM(
                input_ids=current_tokens_arm2,
                attention_mask=attention_mask_arm2,
                use_cache=True,
                output_hidden_states=False
            )
            logits2 = mout2.logits[:, -1, :]

            pre_screen_beam_width1 = pre_screen_beam_width
            topk_logits1, topk_tokens1_indices = torch.topk(logits1, pre_screen_beam_width1, dim=-1)
            candidates_s1 = topk_tokens1_indices[0, :] 

            pre_screen_beam_width2 = pre_screen_beam_width
            topk_logits2, topk_tokens2_indices = torch.topk(logits2, pre_screen_beam_width2, dim=-1)
            candidates_s2 = topk_tokens2_indices[0, :]  

            all_candidates = torch.cat([candidates_s1, candidates_s2], dim=0)
            union_candidates = torch.unique(all_candidates)

            union_beam_width = len(union_candidates)
            union_candidate_tokens = union_candidates.unsqueeze(1)  

            full_probs1 = torch.softmax(logits1, dim=-1) 
            full_probs2 = torch.softmax(logits2, dim=-1)  

            union_candidates_expanded = union_candidates.unsqueeze(0)
            probs1 = full_probs1.gather(-1, union_candidates_expanded)[0]
            probs2 = full_probs2.gather(-1, union_candidates_expanded)[0]

            cand_sequences1_ids = torch.cat([current_tokens_arm1.repeat(union_beam_width, 1), union_candidate_tokens], dim=1)
            cand_sequences2_ids = torch.cat([current_tokens_arm2.repeat(union_beam_width, 1), union_candidate_tokens], dim=1)

            cand_texts_for_bllm1 = self.tokens_to_text(cand_sequences1_ids, tokenizer_type="llm")
            cand_texts_for_bllm2 = self.tokens_to_text(cand_sequences2_ids, tokenizer_type="llm")

            sentence_embeddings_cand1 = self.get_embedding(cand_texts_for_bllm1)
            sentence_embeddings_cand2 = self.get_embedding(cand_texts_for_bllm2)

            sentence_embeddings_cand1_grad = sentence_embeddings_cand1.clone().detach().to(self.reward_model_device).float()
            sentence_embeddings_cand2_grad = sentence_embeddings_cand2.clone().detach().to(self.reward_model_device).float()
            sentence_embeddings_cand1_grad.requires_grad_(True)
            sentence_embeddings_cand2_grad.requires_grad_(True)

            rewards1_grad = self.reward_model(sentence_embeddings_cand1_grad).view(-1)
            rewards2_grad = self.reward_model(sentence_embeddings_cand2_grad).view(-1)

            grads_cand1 = []
            grads_cand2 = []

            for i in range(union_beam_width):
                self.reward_model.zero_grad()
                rewards1_grad[i].backward(retain_graph=True)
                grad_flat = torch.cat([p.grad.flatten() for p in self.reward_model.parameters() if p.grad is not None])
                grads_cand1.append(grad_flat)

                self.reward_model.zero_grad()
                rewards2_grad[i].backward(retain_graph=True)
                grad_flat = torch.cat([p.grad.flatten() for p in self.reward_model.parameters() if p.grad is not None])
                grads_cand2.append(grad_flat)

            self.reward_model.zero_grad()

            grads_cand1 = torch.stack(grads_cand1, dim=0)
            grads_cand2 = torch.stack(grads_cand2, dim=0)

            with torch.no_grad():
                rewards1 = rewards1_grad.detach().to(self.llm_device)
                rewards2 = rewards2_grad.detach().to(self.llm_device)
                rew1_01 = torch.sigmoid(rewards1)
                rew2_01 = torch.sigmoid(rewards2)

            del sentence_embeddings_cand1_grad, sentence_embeddings_cand2_grad
            del rewards1_grad, rewards2_grad

            score1 = probs1 + weight * rew1_01
            best_idx_arm1 = torch.argmax(score1).item()
            next_token_arm1 = union_candidates[best_idx_arm1].unsqueeze(0).unsqueeze(0)

            if best_idx_arm1 < len(grads_cand1):
                greedy_grad = grads_cand1[best_idx_arm1]
            else:
                error_msg = f"Index inconsistency: best_idx_arm1={best_idx_arm1} >= len(grads_cand1)={len(grads_cand1)}"
                print(f"[ERROR] {error_msg}")
                greedy_grad = grads_cand1[-1]

            exploration_term = torch.zeros(union_beam_width, device=self.reward_model_device)

            try:
                for i in range(union_beam_width):
                    if i < len(grads_cand2):
                        grad_diff = grads_cand2[i] - greedy_grad
                        if grad_diff.shape[0] == self.V_inv.shape[0]:
                            exploration_term[i] = torch.sqrt(torch.sum(grad_diff**2 * self.V_inv).clamp(min=1e-16))
                        else:
                            exploration_term[i] = 0.0
                    else:
                        exploration_term[i] = 0.0

                exploration_term = exploration_term.to(self.llm_device)
            except Exception:
                exploration_term = torch.zeros(union_beam_width, device=self.llm_device)

            score2 = probs2 + weight * (rew2_01 + nu_mab * exploration_term)
            best_idx_arm2 = torch.argmax(score2).item()

            next_token_arm2 = union_candidates[best_idx_arm2].unsqueeze(0).unsqueeze(0)

            if next_token_arm1.item() == next_token_arm2.item():
                score2_copy = score2.clone()
                score2_copy[best_idx_arm2] = float('-inf')
                best_idx_arm2_diverse = torch.argmax(score2_copy).item()
                next_token_arm2 = union_candidates[best_idx_arm2_diverse].unsqueeze(0).unsqueeze(0)
                best_idx_arm2 = best_idx_arm2_diverse

            select_idx_history.append((best_idx_arm1, best_idx_arm2))

            try:
                selected_grad1 = grads_cand1[best_idx_arm1]
                selected_grad2 = grads_cand2[best_idx_arm2]

                if selected_grad1.shape[0] == selected_grad2.shape[0] == self.V_inv.shape[0]:
                    grad_diff = selected_grad1 - selected_grad2
                    self.V_inv = (self.V_inv + grad_diff**2).clamp(min=1e-16)
            except Exception:
                pass

            current_tokens_arm1 = torch.cat([current_tokens_arm1, next_token_arm1], dim=1)
            current_tokens_arm2 = torch.cat([current_tokens_arm2, next_token_arm2], dim=1)

            if next_token_arm1.item() == self.tokenizer.eos_token_id and \
               next_token_arm2.item() == self.tokenizer.eos_token_id:
                break

            self.last_topk_logits1 = topk_logits1[0].detach().cpu()
            self.last_rewards1     = rewards1.detach().cpu()
            self.last_topk_logits2 = topk_logits2[0].detach().cpu()
            self.last_rewards2     = rewards2.detach().cpu()

            del grads_cand1, grads_cand2
            del cand_sequences1_ids, cand_sequences2_ids
            del cand_texts_for_bllm1, cand_texts_for_bllm2
            del sentence_embeddings_cand1, sentence_embeddings_cand2
            del attention_mask_arm1, attention_mask_arm2
            del mout1, mout2, logits1, logits2
            del topk_logits1, topk_tokens1_indices, topk_logits2, topk_tokens2_indices

            if step % 5 == 0:
                torch.cuda.empty_cache()
                if hasattr(self.LLM, 'past_key_values'):
                    self.LLM.past_key_values = None
        return current_tokens_arm1[0], current_tokens_arm2[0]