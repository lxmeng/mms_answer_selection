# mms_answer_selection
multi-modal similarity metric learning for answer selection

1. prepare data
    
    download trec_qa from http://cs.stanford.edu/people/mengqiu/data/qg-emnlp07-data.tgz
    
    download word embeddings from http://nlp.stanford.edu/data/glove.6B.zip
    
2. run

    python examples/trec_qa_w2v_mms/do_trec_qa_clean.py --main_dir examples/trec_qa_w2v_mms --exp_name exp --trainmode train-all --make_data
    
3. result
