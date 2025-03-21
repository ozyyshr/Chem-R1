
file_path=/shared/data3/xzhong23/ChemRAG/hf_dataset/ChemRAG/uspto
index_file=/shared/data/bowenj4/ChemRAG/index/uspto/bm25
corpus_file=$file_path/all_modified.jsonl
retriever=bm25

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever
