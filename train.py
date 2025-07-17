import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from reward import reward_function

EPS = 1e-6
LEARNING_RATE = 1e-6
MAX_ANSWER_LEN = 128
NUM_QUESTION_PER_BATCH = 16
NUM_ANSWER_PER_QUESTION = 8
MINI_BATCH_SIZE = 8
EVAL_STEP = 30
MAX_INPUT_LEN = 768
MAX_SEQ_LEN = MAX_INPUT_LEN + MAX_ANSWER_LEN

def read_data():
    with open('data/train.jsonl') as fi:
        train_data = [json.loads(line) for line in fi]
        train_data = [item for item in train_data if item['prefix_ids_len'] < MAX_INPUT_LEN]

    with open('data/test.jsonl') as fi:
        test_data = [json.loads(line) for line in fi]
        test_data = [item for item in test_data if item['prefix_ids_len'] < MAX_INPUT_LEN]
    
    return train_data, test_data

def split_into_chunk(lst: list[any], batch_size: int) -> list[list[any]]:
    chunks = []
    while len(lst) > 0:
        chunks.append(lst[:batch_size])
        lst = lst[batch_size:]
    return chunks

def generate_answer(item: dict) -> tuple[str, dict, float]:
    target = item["target"]
    prompt = item['prefix']
    prompt_ids = tokenizer(prompt)['input_ids']
    answer_ids = model.generate(
        torch.LongTensor([prompt_ids]).cuda(), 
        max_new_tokens=MAX_ANSWER_LEN,
        do_sample=True,
        temperature=1.0
    )[0]
    answer = tokenizer.decode(answer_ids[len(prompt_ids):], skip_special_tokens=True)
    reward = reward_function(answer, target)
    return answer, target, reward

def evaluate(model, test_data: list[dict]):
    test_data = sorted(test_data, key=lambda x:x['prefix_ids_len'])
    rewards = []
    model.eval()

    for item in test_data:
        answer, target, reward = generate_answer(item)
        rewards.extend(reward)
        print(f"Answer: {answer}\nTarget: {target}, reward: {reward}")

    print('Mean eval rewards:', np.mean(rewards))


def train_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.AdamW,
    batch: list[dict],
):
    model.train()

    for chunk in split_into_chunk(batch, MINI_BATCH_SIZE):
        num_target_tokens = sum(len(item["answer_ids"]) for item in chunk) + len(chunk)

        for item in chunk:
            input_ids = item["input_ids"]
            answer_ids = item["answer_ids"]
            target_ids = torch.tensor(answer_ids + [tokenizer.eos_token_id], dtype=torch.int64, device=model.device)
            sequence_ids = input_ids + answer_ids + (MAX_SEQ_LEN - len(input_ids) - len(answer_ids)) * [0]
            sequence_ids = torch.tensor([sequence_ids] , dtype=torch.int64, device=model.device)
            logits = model(sequence_ids).logits[0, len(input_ids)-1:len(input_ids)+len(answer_ids)].float()
            log_probs = torch.nn.functional.cross_entropy(logits, target_ids, reduction="none")
            loss = (log_probs * item["norm_reward"]).sum() / num_target_tokens
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

if __name__ == '__main__':
    model_path = 'Qwen/Qwen2.5-1.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )

    train_data, test_data = read_data()
    optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)
    model.eval()

    for step in tqdm(range(len(train_data)//NUM_QUESTION_PER_BATCH)):
        batch = []
        
        for item in train_data[step*NUM_QUESTION_PER_BATCH : (step+1)*NUM_QUESTION_PER_BATCH]:
            rewards = []
            for _ in range(NUM_ANSWER_PER_QUESTION): 
                answer, target, reward = generate_answer(item)
                rewards.append(reward)
                print(f"Answer: {answer}\nTarget: {target}, reward: {reward}")

                batch.append({
                    "input_ids": tokenizer(item["prefix"])['input_ids'],
                    "answer_ids": tokenizer(answer)['input_ids']
                })
                
            rewards = np.array(rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            norm_rewards = (rewards - mean_reward) / (EPS + std_reward)

            for i, gen_item in enumerate(batch[-NUM_ANSWER_PER_QUESTION:]):
                gen_item["reward"] = rewards[i]
                gen_item["norm_reward"] = norm_rewards[i]

        batch_score = np.mean([item["reward"] for item in batch])
        print(f"\rStep {step}, mean_reward: {batch_score:.2f}")
        batch = [item for item in batch if abs(item["norm_reward"]) > EPS]
        train_batch(model, tokenizer, optimizer, batch)

        if (step + 1) % EVAL_STEP == 0:
            evaluate(model, test_data)

