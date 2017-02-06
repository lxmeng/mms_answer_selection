# mms_answer_selection
multi-modal similarity metric learning for answer selection

1. install caffe

2. prepare data
    
    download trec_qa from http://cs.stanford.edu/people/mengqiu/data/qg-emnlp07-data.tgz
    
    download word embeddings from http://nlp.stanford.edu/data/glove.6B.zip
    
3. run

    ```python
    python examples/trec_qa_w2v_mms/do_trec_qa_clean.py --main_dir examples/trec_qa_w2v_mms --exp_name exp --trainmode train-all --make_data
    ```
    
4. result

    ```
    num_q          	all	68
    num_ret        	all	1442
    num_rel        	all	248
    num_rel_ret    	all	248
    map            	all	0.7793
    gm_ap          	all	0.7184
    R-prec         	all	0.7035
    bpref          	all	0.7049
    recip_rank     	all	0.8487
    ircl_prn.0.00  	all	0.8713
    ircl_prn.0.10  	all	0.8683
    ircl_prn.0.20  	all	0.8540
    ircl_prn.0.30  	all	0.8432
    ircl_prn.0.40  	all	0.8202
    ircl_prn.0.50  	all	0.8125
    ircl_prn.0.60  	all	0.7827
    ircl_prn.0.70  	all	0.7388
    ircl_prn.0.80  	all	0.7249
    ircl_prn.0.90  	all	0.7079
    ircl_prn.1.00  	all	0.7044
    P5             	all	0.5000
    P10            	all	0.3147
    P15            	all	0.2206
    P20            	all	0.1743
    P30            	all	0.1201
    P100           	all	0.0365
    P200           	all	0.0182
    P500           	all	0.0073
    P1000          	all	0.0036
    ```
