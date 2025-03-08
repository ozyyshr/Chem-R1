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
- ChemBench4k 4009
- College and High-school chemistry reasoning (951*0.2)
"""

import re
import os
from datasets import Dataset
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['instruction'] + " " + dp['input']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question.\
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> The Reactant is CCOC(C)=O. </answer>. Question: {question}"""
    else:
        raise NotImplementedError
    return prefix

def make_map_fn(split):

    def process_fn(example, idx):
        if example['input']:
            example['input'] = example['input'].strip()
            if example['input'][-1] != '?':
                example['input'] += '?'
        question = make_prefix(example, template_type=args.template_type)
        solution = {
            "target": example['output'],
        }

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "chemistry-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
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

    # for each json file in the data folder "args.local_dir/molinstruct", load the data
    data_folder = os.path.join(args.local_dir, 'molinstruct')
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(data_folder, json_file)
        with open(file_path, 'r') as f:
            data_source = json_file.split('.')[0]
            print(f"Processing {data_source}...")
            content = json.load(f)
            print(f"original data length: {len(content)}")
            dataset = Dataset.from_list(content)
            dataset = dataset.filter(lambda sample: sample['metadata']['split'] == 'train')
            print(f"train data length: {len(dataset)}")
            if len(dataset) > 2000:
                dataset = dataset.shuffle(seed=42).select(range(2000))

            # add a row to each data item that represents a unique id
            train_dataset = dataset.map(function=make_map_fn(data_source), with_indices=True).remove_columns(dataset.column_names)
            print(f"len of processed {data_source}:", len(train_dataset)) # 2000
            print("*"*50)
            print("demonstration of the first data item:")
            print(train_dataset[0])

            local_dir = args.local_dir
            hdfs_dir = args.hdfs_dir

            train_dataset.to_parquet(os.path.join(local_dir, f'{data_source}.parquet'))

            if hdfs_dir is not None:
                makedirs(hdfs_dir)

                copy(src=local_dir, dst=hdfs_dir)
        input()
