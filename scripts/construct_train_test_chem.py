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
"""
Preprocess the knowledge-intensive chemistry-related dataset to parquet format
Chem data source:

Training:
- College and High-school chemistry reasoning (951*0.8)
    - MMLU-Chem 300
    - SciBench 266
    - RUC Long CoT 385
- Mol-Instruct (13k)

Testing:
- ChemBench4k (4k)
- Mol-Instruct (6.1k)
- College and High-school chemistry reasoning (951*0.2)
"""

import pandas as pd
import os
from datasets import Dataset
import datasets
import json
import re
import selfies as sf
from sklearn.model_selection import train_test_split
from datasets import concatenate_datasets

#from verl.utils.hdfs_io import copy, makedirs
import argparse

def extract_answer(text):
    # Extract the answer from the text wrapped in $$\boxed{}
    # For example, given the text "after detailed analysis, the smallest value of the maximum distance is found to be: $$\boxed{\frac{\pi}{4}}$$"
    # The function should return "\frac{\pi}{4}"
    # pattern = r"\$\$\\boxed\{(.*?)\}\$\$"  
    pattern = r"\\boxed\{(.*?)\}"  
    match = re.search(pattern, text, re.DOTALL)  
    if match:
        return match.group(1).strip()  
    else:
        print("No match found!")
        return None


def make_prefix(dp, template_type):

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'molinstruct':
        question = dp['instruction'] + " " + dp['input']
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> The Reactant is CCOC(C)=O. </answer>. Question: {question}"""
    elif template_type == 'scibench':
        question = dp['problem_text']
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> +7.3 J </answer>. Question: {question}\n"""
    elif template_type == 'chembench':
        question = dp['question']
        choices = "\n".join([dp['A'], dp['B'], dp['C'], dp['D']])
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> The Reactant is CCOC(C)=O. </answer>. Question: {question} Choose from the following four options:\n{choices}"""
    elif template_type == 'mmlu_chem':
        question = dp['question']
        choices = "\n".join(dp['choices'])
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> +7.3 J </answer>. Question: {question} Choose from the following four options:\n{choices}"""
    elif template_type == 'chemcot':
        question = dp['question']
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> +7.3 J </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix

def make_map_fn(split, task_name=None):
    def selfies_to_smiles(selfies):
        return sf.decoder(selfies)
    
    def is_valid_selfies(s):
        return sf.is_valid_selfies(s)

    def process_fn(example, idx):

        if split == 'molinstruct':
            if example['input']:
                example['input'] = example['input'].strip()
                try:
                    temp = sf.decoder(example['input'])
                    if len(temp) > 0:
                        example['input'] = temp
                except sf.DecoderError:
                    pass

                if example['input'][-1] != '?':
                    example['input'] += '?'
        elif split == 'scibench':
            example['problem_text'] = example['problem_text'].strip()
            if example['problem_text'][-1] != '?':
                example['problem_text'] += '?'
        else:
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'

        question = make_prefix(example, template_type=split)

        if split == 'mmlu_chem':
            solution = {
                "target": example['choices'][example['golden_answers'][0]],
            }
        elif split == 'chembench':
            solution = {
                "target": example[example['answer']],
            }
        elif split == 'chemcot':
            solution = {
                "target": extract_answer(example['combined_text']),
            }
        elif split == 'scibench':
            solution = {
                "target": example['answer_latex'] + " " + example['unit'],
            }
        elif split == 'molinstruct':
            if task_name == 'property_prediction' or task_name == 'molecular_description_generation':
                solution = {
                    "target": example['output'],
                }
            elif task_name == 'true_or_false_question':
                if example['output'].lower().startswith('yes'):
                    solution = {
                        "target": "Yes.",
                    }
                else:
                    solution = {
                        "target": "No.",
                    }
            else:
                try:
                    solution = {
                        "target": selfies_to_smiles(example['output']),
                    }
                except Exception as e:
                    print(f"Error converting selfies to smiles: {e}")
                    print(f'example: {example["output"]}')
                    breakpoint()
                    solution = {
                        "target": example['output'],
                    }
        else:
            solution = {
                "target": example['output'],
            }
        if task_name is not None:
            data = {
                "data_source": f"{split}_{task_name}",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "chemistry-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(solution)
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
        else:
            data = {
                "data_source": split,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "chemistry-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(solution)
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
        return data

    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    print("### Constructing dataset for SciBench ###")
    scibench_dataset = datasets.load_dataset('xw27/scibench')
    dataset = scibench_dataset['train']
    scibench_classes = ["atkins", "chemmc", "quan", "matter"]
    # filter dataset to only include classes in scibench_classes
    dataset = dataset.filter(lambda x: x['source'] in scibench_classes)
    print("len of scibench:", len(dataset))
    scibench_set = dataset.map(function=make_map_fn('scibench'), with_indices=True).remove_columns(dataset.column_names)
    print("len of processed scibench:", len(scibench_set)) # 266
    print("*"*50)
    print("demonstration of the first data item:")
    print(scibench_set[0])
    print(scibench_set.shape)
    

    # print("### Constructing dataset for ChemCot ###")
    # chemcot_dataset = datasets.load_dataset('RUC-AIBOX/long_form_thought_data_5k')
    # dataset = chemcot_dataset['train']
    # scibench_classes = ["chemistry", "biology"]
    # # filter dataset to only include classes in scibench_classes
    # dataset = dataset.filter(lambda x: x['domain'] in scibench_classes)
    # print("len of chemcot:", len(dataset))
    # chemcot_set = dataset.map(function=make_map_fn('chemcot'), with_indices=True).remove_columns(dataset.column_names)
    # print("len of processed chemcot:", len(chemcot_set)) # 385
    # print("*"*50)
    # print("demonstration of the first data item:")
    # print(chemcot_set[0])
    # print(chemcot_set.shape)

    print("### Constructing dataset for ChemBench Train ###")
    chembench_dataset = datasets.load_dataset('AI4Chem/ChemBench4K')
    dataset = chembench_dataset['validation']
    chembench_train = dataset.map(function=make_map_fn('chembench'), with_indices=True).remove_columns(dataset.column_names)
    print("len of processed chemcot:", len(chembench_train)) # 1000
    print("*"*50)
    print("demonstration of the first data item:")
    print(chembench_train[0])
    print(chembench_train.shape)

    print("### Constructing dataset for ChemBench ###")
    chembench_dataset = datasets.load_dataset('AI4Chem/ChemBench4K')
    dataset = chembench_dataset['test']
    dataset = dataset.shuffle(seed=42).select(range(1000)) # randomly sample 1000 data items
    chembench_set = dataset.map(function=make_map_fn('chembench'), with_indices=True).remove_columns(dataset.column_names)
    print("len of processed chemcot:", len(chembench_set)) # 1000
    print("*"*50)
    print("demonstration of the first data item:")
    print(chembench_set[0])
    print(chembench_set.shape)

    print("### Constructing dataset for MMLU-Chem ###")
    with open('data/chem_mmlu.jsonl') as f:
        mmlu_chem = [json.loads(line) for line in f]
    train_dataset = Dataset.from_list(mmlu_chem)
    mmluchem = train_dataset.map(function=make_map_fn('mmlu_chem'), with_indices=True).remove_columns(train_dataset.column_names)
    print("len of processed mmlu_chem:", len(mmluchem)) # 303
    print("*"*50)
    print("demonstration of the first data item:")
    print(mmluchem[0])
    print(mmluchem.shape)

    # 1. Merge chemistry_reasoning files
    #chemistry_df = concatenate_datasets([mmluchem, chemcot_set, scibench_set])
    chemistry_df = concatenate_datasets([mmluchem, scibench_set])
    print(f"chemistry_reasoning merged: {len(chemistry_df)}")

    print("### Constructing dataset for Mol-Instruct ###")
    data_folder = os.path.join(args.local_dir, 'molinstruct')
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]

    molinstruct_train_df_list = []

    for json_file in json_files:
        task_name = json_file.split('.')[0]
        file_path = os.path.join(data_folder, json_file)
        with open(file_path, 'r') as f:
            content = json.load(f)
            dataset = Dataset.from_list(content)
            dataset = dataset.filter(lambda sample: sample['metadata']['split'] == 'train')
            if len(dataset) > 2000:
                dataset = dataset.shuffle(seed=42).select(range(2000))

            # Add unique ID and convert to DataFrame
            test_dataset = dataset.map(function=make_map_fn("molinstruct", task_name), with_indices=True).remove_columns(dataset.column_names)
            molinstruct_train_df_list.append(test_dataset)
            print(f"molinstruct train {task_name}:\n{test_dataset[0]}")

    molinstruct_train_df = concatenate_datasets(molinstruct_train_df_list)
    print(f"molinstruct train merged: {len(molinstruct_train_df)}")
    print("*"*50)
    print("demonstration of the first data item:")
    print(molinstruct_train_df[0])

    # 2. Split chemistry_reasoning into training (80%) and testing (20%)
    chem_reasoning = chemistry_df.train_test_split(test_size=0.2, seed=42)
    chem_reasoning_train = chem_reasoning['train']
    chem_reasoning_test = chem_reasoning['test']
    print(f"chemistry_reasoning split into training: {len(chem_reasoning_train)}, testing: {len(chem_reasoning_test)}")

    # 3. Merge molinstruct training with chemistry_reasoning training
    final_train_hf = concatenate_datasets([molinstruct_train_df, chem_reasoning_train, chembench_train])
    final_train_hf.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    print(f"final training merged: {len(final_train_hf)}")

    # Load molinstruct test data from JSON files and convert to DataFrame
    data_folder = os.path.join(args.local_dir, 'molinstruct')
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]

    molinstruct_test_df_list = []

    for json_file in json_files:
        task_name = json_file.split('.')[0]
        file_path = os.path.join(data_folder, json_file)
        with open(file_path, 'r') as f:
            content = json.load(f)
            dataset = Dataset.from_list(content)
            dataset = dataset.filter(lambda sample: sample['metadata']['split'] == 'test')

            # Add unique ID and convert to DataFrame
            test_dataset = dataset.map(function=make_map_fn("molinstruct", task_name), with_indices=True).remove_columns(dataset.column_names)
            molinstruct_test_df_list.append(test_dataset)

    # Merge all test DataFrames from molinstruct
    molinstruct_test_df = concatenate_datasets(molinstruct_test_df_list)
    print(f"molinstruct test merged: {len(molinstruct_test_df)}")

    # Combine molinstruct test set with chemistry_reasoning test set
    final_test_hf = concatenate_datasets([chem_reasoning_test, molinstruct_test_df, chembench_set])
    print(f"final test merged: {len(final_test_hf)}")
    final_test_hf.to_parquet(os.path.join(args.local_dir, 'test.parquet'))


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)