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
- MMLU-Chem
- Molecule QA
- SciBench (Chem)
- RUC Long CoT
- Mol-Instruct (Multiple Choice)
- ChemBench4k
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['problem_text']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        """For SciBench, we have to add the constraints of units for better evlauation"""
        prefix = f"""Answer the given question.\
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> +7.3 J </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'scibench'

    scibench_dataset = datasets.load_dataset('xw27/scibench')
    dataset = scibench_dataset['train']
    scibench_classes = ["atkins", "chemmc", "quan", "matter"]
    # filter dataset to only include classes in scibench_classes
    dataset = dataset.filter(lambda x: x['source'] in scibench_classes)
    print("len of scibench:", len(dataset))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['problem_text'] = example['problem_text'].strip()
            if example['problem_text'][-1] != '?':
                example['problem_text'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['answer_latex'] + " " + example['unit'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
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

    train_dataset = dataset.map(function=make_map_fn('scibench'), with_indices=True).remove_columns(dataset.column_names)
    print("len of processed scibench:", len(train_dataset)) # 266
    print("*"*50)
    print("demonstration of the first data item:")
    print(train_dataset[0])
    print(train_dataset.shape)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'scibench.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
