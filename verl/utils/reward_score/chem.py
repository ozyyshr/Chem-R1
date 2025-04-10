# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
import evaluate

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def validate_response_structure(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': '<think>',
        'think_end': '</think>',
        'answer_start': '<answer>',
        'answer_end': '</answer>'
    }
    search_tags = {
        'search_start': '<search>',
        'search_end': '</search>',
        'info_start': '<information>',
        'info_end': '<information>'
    }

    positions = {}
    for tag_name, tag_str in list(tags.items()) + list(search_tags.items()):
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")
        # if 'answer' in tag_name and (count !=1):
        #     if do_print:
        #         print(f"  [Error] {tag_str} appears {count} times (expected 1)")
        #     validation_passed = False

    # Check if other tags are present
    allowed_tags = set(tags.values()) | set(search_tags.values())
    found_tags = set(re.findall(r'</?[^>]+>', processed_str))
    if not found_tags.issubset(allowed_tags):
        if do_print:
            print("  [Error] Found unexpected tags")
        validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end'] or
        positions['search_end'] > positions['info_start'] 
        ):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")

    # Verify search and info tags
    if search_tags['search_start'] in processed_str:
        if search_tags['search_end'] not in processed_str and search_tags['info_start'] not in processed_str:
            if do_print:
                print("  [Error] Incorrect search format: Missing </search> or <information> tag")
            validation_passed = False
    if search_tags['info_start'] in processed_str:
        if search_tags['info_end'] not in processed_str:
            if do_print:
                print("  [Error] Missing </information> tag")
            validation_passed = False
    
    return validation_passed

def cal_bleu(prediction, golden_answers, do_print):
    """Calculate BLEU score for the prediction and golden answers."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    try:
        normalized_prediction = normalize_answer(prediction)
    except:
        return 0
    
    rouge_metric = evaluate.load("bleu")

    golden_answer = normalize_answer(golden_answers[0])

    rouge_metric.add(prediction=normalized_prediction, reference=golden_answer)
    if do_print:
        print(f"[Normalized Prediction]: {normalized_prediction}")
        print(f"[Normalized Golden Answer]: {golden_answer}")

    try:
        rouge_res = rouge_metric.compute()
        # bleu_score = rouge_res['rouge1'] + rouge_res['rougeL']
        bleu_score = rouge_res['bleu']
    except:
        return 0

    # if bleu_score >= 0.7:
    #     answer_score = 5
    # elif bleu_score >= 0.5:
    #     answer_score = 4
    # elif bleu_score >= 0.4:
    #     answer_score = 3
    # elif bleu_score >= 0.3:
    #     answer_score = 1
    # elif bleu_score >= 0.1:
    #     answer_score = 0.5
    # elif bleu_score >= 0.05:
    #     answer_score = 0.1
    # else:
    #     answer_score = -3.5

    # except:
    #     answer_score = -4

    if do_print:
        print(f"[BLEU Score]: {bleu_score}")

    return bleu_score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_bleu(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    # do_print = True
    
    # validate response structure
    # validate_format_correct = validate_response_structure(solution_str, do_print)
    
    # format_score = 0.1 if validate_format_correct else 0.0

    answer_score = cal_bleu(answer, [str(ground_truth['target']),], do_print)

    # total_score = format_score + answer_score

    if do_print:
        print(f"--------------------------------")
        print(f"[Golden answers]: {ground_truth['target']}")
        print(f"[Extracted answer]: {answer}")
        print(f"[Solution string]: {solution_str}")
        # print(f"[Format score]: {format_score}")
        print(f"[Answer score]: {answer_score}")
        # print(f"[Total score]: {total_score}")
        print(f"--------------------------------")
    
    return answer_score