import re
from typing import Optional
import json
import numpy as np

def parse_int(st: Optional[str]) -> Optional[int]:
    if st is None:
        return None

    try:
        return int(st)
    except:
        return None

def parse_float(st: Optional[str]) -> Optional[float]:
    if st is None:
        return None

    try:
        return float(st)
    except:
        return None

def parse_bool(st: Optional[str]) -> Optional[bool]:
    if st is None:
        return None

    try:
        return {"true": True, "false": False}[st]
    except:
        return None

def str_sim(s1,s2):
    items1 = list(s1)
    items2 = list(s2)
    l =  max(len(items1), len(items2))
    if l == 0: return 1.0 if s1 == s2 else 0.0
    n = 0
    while True:
        while len(items1) > 0 and items1[0] not in items2:
            items1 = items1[1:]
        if len(items1) == 0:
            break
        items2 = items2[items2.index(items1[0])+1:]
        items1 = items1[1:]
        n += 1
    return n / l
    
def str_compare(s1: str, s2: str) -> float:
    if s1 == s2: return 1.0

    if s1.lower() == s2.lower():
        return 0.9
    
    return str_sim(s1, s2)

def obj_compare(src: dict, target: dict) -> float:
    if src is None or not isinstance(src, dict):
        return 0.0
    
    if not isinstance(target, dict):
        return 0.0

    if len(target) == 0:
        return 1.0 if len(src) == 0 else 0.0

    acc_score = 0.0

    for key in target:
        target_val = target[key]
        src_val = src.get(key)

        if target_val is None:
            acc_score += 1.0 if src_val is None else 0
        
        elif isinstance(target_val, int):
            if isinstance(src_val, int):
                acc_score += 1.0 if src_val == target_val else 0

            elif isinstance(src_val, str):
                acc_score += 0.75 if parse_int(src_val) == target_val else 0
        
        elif isinstance(target_val, float):
            if isinstance(src_val, int) or isinstance(src_val, float):
                acc_score += 1.0 if src_val == target_val else 0

            elif isinstance(src_val, str):
                acc_score += 0.75 if parse_float(src_val) == target_val else 0
            
        elif isinstance(target_val, bool):
            if isinstance(src_val, bool):
                acc_score += 1.0 if src_val == target_val else 0

            elif isinstance(src_val, str):
                acc_score += 0.75 if parse_bool(src_val) == target_val else 0
        
        elif isinstance(target_val, str):
            if isinstance(src_val, str):
                acc_score += str_compare(src_val, target_val)

            else:
                acc_score += 0.75 * str_compare(str(src_val), target_val)
        
        elif isinstance(target_val, list):
            if isinstance(src_val, list):
                acc_score += obj_compare(
                    src={i: val for i,val in enumerate(src_val)},
                    target={i: val for i,val in enumerate(target_val)},
                )
        
        elif isinstance(target_val, dict):
            if isinstance(src_val, dict):
                acc_score += obj_compare(src_val, target_val)
        
        else:
            raise Exception("Unknown data type:"  + str(target_val.__class__))

    if acc_score > 0:
        return acc_score / (max(len(src), len(target)))
    else:
        return acc_score / (min(len(src), len(target)))

def count_leaf(obj: dict) -> int:
    n = 0
    if not isinstance(obj, dict):
        return 0

    for k,v in obj.items():
        if isinstance(v, list):
            n += count_leaf({i:x for i,x in enumerate(v)})
        
        elif isinstance(v, dict):
            n += count_leaf(v)
        
        else:
            n += 1
    
    return n

def answer_match_score(src: dict, target: dict):
    if not isinstance(src, dict) or src.get('name') != target['name']:
        return 0 if not src.get('name') else 0.0
    
    n_args = count_leaf(target.get("arguments", {}))
    n_args = max(n_args, 1)
    n_args = max(n_args, 4)
    arg_weight = 1 - 1.0 / (1 + n_args)

    return (1 - arg_weight) + arg_weight * obj_compare(src.get("arguments"), target.get("arguments", {}))
    
def json_parse(st: str) -> Optional[dict]:
    try:
        return json.loads(st)
    except:
        return None

def reward_function(
    response: str,
    target: list[dict],
) -> float:
    """Reward function for Response output.
    """
    
    answer_regex = r"<tool_call>([^<>]*)<\/tool_call>"

    matchs = list(re.finditer(answer_regex, response, re.DOTALL))

    answers = [
        json_parse(match.group(1))
        for match in matchs
    ]
    match_len = sum([len(match.group(0)) for match in matchs])

    n_anwers = len(answers)
    answers = [answer for answer in answers if answer]

    acc_score = 0.0 

    for target_idx,target_item in enumerate(target):
        if len(answers) == 0:
            break

        scores = [answer_match_score(src_item, target_item) for src_item in answers]
        idx = np.argmax(scores)
        acc_score += scores[idx]
        if scores[idx] > 0:
            del answers[idx]

    acc_score = acc_score / (max(len(target), n_anwers))

    if len(response) > 0:
        acc_score *= match_len / (len(response))
        
    return acc_score
