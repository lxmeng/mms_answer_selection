import os
import sys
import argparse
sys.path.insert(0, './python')
import subprocess
import caffe
import h5py
import re
from collections import defaultdict
import numpy as np
from pylab import *

from caffe import layers as L
from caffe import params as P

import matplotlib.pyplot as plt

sys.path.append('./python/caffe/proto'); import caffe_pb2
from caffe_pb2 import SolverParameter

class qa_caffe:
    def __init__(self, main_dir, exp_name, gpu_id, check_model_dir=True, trainmode='train'):
        print 'initializing...'
        self.network_name = 'qa'
        self.main_dir = main_dir
        self.main_dir2 = '%s/%s' % (main_dir, exp_name)
        if not os.path.exists(self.main_dir2):
            os.mkdir(self.main_dir2)
        self.solver_file = '%s/qa-solver' % self.main_dir2
        self.train_net_file = '%s/qa-train-net' % self.main_dir2
        self.test_net_file = '%s/qa-test-net' % self.main_dir2
        self.dev_net_file = '%s/qa-dev-net' % self.main_dir2
        self.train_data_dir = '%s/%s_h5/%s.txt' % (self.main_dir, trainmode, trainmode)
        if trainmode == 'train':
            self.train_size = 4718
        else:
            self.train_size = 53417 #4718 #
        self.dev_data_dir = '%s/dev_h5/dev.txt' % self.main_dir
        self.dev_size = 1148
        self.test_data_dir = '%s/test_h5/test.txt' % self.main_dir
        self.test_size = 1517
        self.model_dir = '%s/models' % self.main_dir2
        if check_model_dir and not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.gpu_id = gpu_id

        self.solver_base_lr = 1.0
        self.solver_weight_decay = 5e-4
        self.solver_lr_policy = 'fixed'
        self.solver_delta = 5e-7
        self.solver_gamma = 0.792
        self.solver_momentum = 0.95
        self.solver_clip_gradient = 3
        self.solver_display = 50
        self.solver_snapshot = 100
        self.solver_max_iter = 40001
        self.solver_snapshot_prefix = '%s/qa' % self.model_dir
        self.solver_random_seed = 22
        self.solver_test_interval = 100
        self.train_loss_record_interval = 20
        
        self.solver_test_iter = 1
        self.solver_dev_iter = 1

        self.train_batch_size = 50
        self.test_batch_size = self.test_size/self.solver_test_iter
        self.dev_batch_size = self.dev_size/self.solver_dev_iter

        self.h5_patch = int(1e6)
        self.question_max_word_len = 40
        self.answer_max_word_len = 40
        self.w2v_dim = 50;

    def make_data(self, trainmode):
        
        def load_stopwords(self, file):
            with open(file) as f:
                for st in f.readlines():
                    self.stop_words = self.stop_words + st[:-1].split(' ')
            print 'total %d stop-words' % len(self.stop_words)
            
        def load_wordvec_txt_fromlist(filename, wordlists):
            w2v = {}
            with open(filename) as f:
                for st in f.readlines():
                    all = st[:-1].split(' ')
                    if len(all) > 2 and all[0] in wordlists:
                        w2v[all[0]] = map(float, all[1:])
    
            print 'total loading %d words~' %len(w2v)
            return w2v
        
        def load_wordvec_bin_fromlist(filename, wordlists):
            w2v = {}
            with open(filename, "rb") as f:
                header = f.readline()
                vocab_size, layer1_size = map(int, header.split())
                binary_len = np.dtype('float32').itemsize * layer1_size
                for line in xrange(vocab_size):
                    word = []
                    while True:
                        ch = f.read(1)
                        if ch == ' ':
                            word = ''.join(word)
                            break
                        if ch != '\n':
                            word.append(ch)
                    bb = f.read(binary_len)
                    if word in wordlists:
                        w2v[word] = np.fromstring(bb, dtype='float32')
                    #print word, self.w2v[word]        
            print "total loading %d words~" %len(w2v)
            return w2v
        
        def load_data(fname):
            lines = open(fname).readlines()
            qids, questions, answers, labels = [], [], [], []
            num_skipped = 0
            prev = ''
            qid2num_answers = {}
            for i, line in enumerate(lines):
              line = line.strip()
          
              qid_match = re.match('<QApairs id=\'(.*)\'>', line)
          
              if qid_match:
                qid = qid_match.group(1)
                qid2num_answers[qid] = 0
          
              if prev and prev.startswith('<question>'):
                question = line.lower().split('\t')
          
              label = re.match('^<(positive|negative)>', prev)
              if label:
                label = label.group(1)
                label = 1 if label == 'positive' else 0
                answer = line.lower().split('\t')
                if len(answer) > 60:
                  num_skipped += 1
                  continue
                labels.append(label)
                answers.append(answer)
                questions.append(question)
                qids.append(qid)
                qid2num_answers[qid] += 1
              prev = line
            # print sorted(qid2num_answers.items(), key=lambda x: float(x[0]))
            print 'num_skipped', num_skipped
            return qids, questions, answers, labels

        def compute_dfs(docs):
            word2df = defaultdict(float)
            for doc in docs:
                for w in set(doc):
                    word2df[w] += 1.0
            num_docs = len(docs)
            for w, value in word2df.iteritems():
                word2df[w] = np.math.log(num_docs / value)
            return word2df

        def compute_overlap_features(questions, answers, word2df=None, stoplist=None):
            word2df = word2df if word2df else {}
            stoplist = stoplist if stoplist else set()
            feats_overlap = []
            for question, answer in zip(questions, answers):
                q_set = set([q for q in question if q not in stoplist])
                a_set = set([a for a in answer if a not in stoplist])
                word_overlap = q_set.intersection(a_set)            

                df_overlap = 0.0
                for w in word_overlap:
                    if w in word2df:
                        df_overlap += word2df[w]
                    else:
                        df_overlap += 11.0

                overlap_feat = [float(len(word_overlap)) / (len(q_set)+len(a_set)), df_overlap / (len(q_set)+len(a_set))] 
                                #float(len(question))/self.question_max_word_len, float(len(answer))/self.answer_max_word_len] #   #       
                feats_overlap.append(np.array(overlap_feat))
                
            return np.array(feats_overlap)

        def vocab_transform_embed(target_input, maxlen):
            def word_to_vecindex(x):
                if not (x in self.w2v_index):
                    return self.unknown_word_index
                return self.w2v_index[x]

            target_line = [word_to_vecindex(x) for x in target_input]
           
            slen = len(target_input)
            #target_line = target_line[:maxlen*self.w2v_dim] + self.zero_word*max(0,maxlen-slen)

            pad_b = max(0, int((maxlen-slen)/2))
            pad_a = max(0, maxlen-pad_b-slen)
            target_line = [self.zero_word_index]*pad_b + target_line[:maxlen] + [self.zero_word_index]*pad_a
            
            #print target_line, self.unknown_word_index, self.w2v_index_cc
            #exit(1)

            assert len(target_line) == maxlen
            return target_line
            
        def build_dataset(qst, ans, qids, labels, phase):
            ############################
            #  remove exist datafile
            ############################
            db_name = '%s/%s_h5' % (self.main_dir, phase)
            subprocess.call(['rm', '-rf', db_name])   
        
            qids_uni = list(set(qids))
            qids_new = [qids_uni.index(x) for x in qids]
        
            overlap_feats = compute_overlap_features(qst, ans, stoplist=self.stop_words, word2df=self.word2dfs)
            #overlap_feats_stop = compute_overlap_features(questions, answers, stoplist=self.stop_words, word2df=self.word2dfs)
            #overlap_feats = np.hstack([overlap_feats, overlap_feats_stop])
            assert(overlap_feats.shape[0] == len(qst))
            
            assert(len(qst) == len(ans))
            print 'Writing %s sentences' % len(qst)
            allQ = []
            allA = []
            for idx in range(len(qst)):
                allQ.append(vocab_transform_embed(qst[idx], self.question_max_word_len))
                allA.append(vocab_transform_embed(ans[idx], self.answer_max_word_len))
            
            dataQ = np.array(allQ, np.float)
            dataA = np.array(allA, np.float)
            dataL = np.array(labels, np.float)
            dataG = np.array(qids_new, np.float)
        
            ### create dataset
            os.mkdir(self.main_dir+'/'+phase + '_h5')
            file_w = open(self.main_dir+'/'+phase + '_h5/' + phase + '.txt', 'w')
            for idx in range(int(math.ceil(dataL.shape[0]*1.0/self.h5_patch))):    
                with h5py.File(self.main_dir+'/'+phase + '_h5' + '/data' + str(idx) + '.h5', 'w') as f:
                    d_begin = idx*self.h5_patch
                    d_end = min(dataL.shape[0], (idx+1)*self.h5_patch)
                    f['question'] = dataQ[d_begin:d_end,:]
                    f['answer'] = dataA[d_begin:d_end,:]
                    f['label'] = dataL[d_begin:d_end]
                    f['group'] = dataG[d_begin:d_end]
                    f['overlap_feat'] = overlap_feats[d_begin:d_end,:]
                
                file_w.write(self.main_dir+'/' + phase + '_h5/data' + str(idx) + '.h5\n')
            file_w.close()
        print 'Starting preparing data'
        qids_trn, questions_trn, answers_trn, labels_trn = load_data('./data/trec_qa/deep-qa-jacana/%s.xml' % trainmode)
        qids_dev, questions_dev, answers_dev, labels_dev = load_data('./data/trec_qa/deep-qa-jacana/dev.xml')
        qids_tst, questions_tst, answers_tst, labels_tst = load_data('./data/trec_qa/deep-qa-jacana/test.xml')
        
        all_sent = []
        all_sent.append(questions_trn)
        all_sent.append(answers_trn)
        all_sent.append(questions_tst)
        all_sent.append(answers_tst)
        all_sent.append(questions_dev)
        all_sent.append(answers_dev)
        all_words = set(flatten(all_sent))
        print 'totally %d words in the dataset~' % len(all_words)
        
        ############################
        #  construct the dictionary
        ############################
        self.stop_words = []
        #self.load_stopwords('./data/trec_qa/stoplist-web.txt')
        
        w2v=load_wordvec_txt_fromlist('./data/trec_qa/glove.6B.50d.txt', all_words) #glove.twitter.27B.50d.txt glove.6B.50d.txt
        self.w2v_dim = 50
        
        #w2v=load_wordvec_bin_fromlist('/search/data/menglingxun/wikiqa/WikiQACodePackage/data/GoogleNews-vectors-negative300.bin', all_words)
        #self.w2v_dim = 300
        
        #self.unknown_word = [uniform(-0.08, 0.08) for x in range(self.w2v_dim)]
        #self.zero_word = [0] * self.w2v_dim        
        
        self.w2v_index = {}
        self.w2v_index_cc = 0
        unknown_word_cc = 0
        unknown_word_index = len(all_words)-1
        
        with open('%s/wiki_dict.txt' % self.main_dir2, 'w') as f:
            for word in all_words:
                if word in w2v:
                    self.w2v_index[word] = self.w2v_index_cc
                    self.w2v_index_cc = self.w2v_index_cc + 1
                    the_sent = word+' '+' '.join(map(str, w2v[word]))+'\n'
                    f.write(the_sent)
                else:
                    #self.w2v_index[word] = unknown_word_index
                    #unknown_word_index = unknown_word_index - 1
                    unknown_word_cc = unknown_word_cc + 1
        print 'totally unknown words: %d ~' % unknown_word_cc
        
        # unknown index + padding index      
        self.unknown_word_index = len(self.w2v_index)
        self.zero_word_index = len(self.w2v_index)+1
        self.w2v_index_cc = len(self.w2v_index)+2
        
        print 'vocabunary size is : %d ~' % self.w2v_index_cc
        
        ############################
        #  compute idf
        ############################
        seen = set()
        unique_questions = []
        for q, qid in zip(questions_trn, qids_trn):
            if qid not in seen:
                seen.add(qid)
                unique_questions.append(q)
        docs = answers_trn+unique_questions
        self.word2dfs = compute_dfs(docs)
        
        ############################
        #  build dataset file
        ############################
        build_dataset(questions_trn, answers_trn, qids_trn, labels_trn, trainmode)
        build_dataset(questions_tst, answers_tst, qids_tst, labels_tst, 'test')
        build_dataset(questions_dev, answers_dev, qids_dev, labels_dev, 'dev')
        
    def make_solver(self):
        print 'make solver...'
        solver = SolverParameter()
        solver.train_net = self.train_net_file
        #solver.test_net.append(self.test_net_file)
        solver.test_net.append(self.dev_net_file)
        solver.test_interval = self.solver_test_interval
        #solver.test_iter.append(self.solver_test_iter)
        solver.test_iter.append(self.solver_dev_iter)
        solver.base_lr = self.solver_base_lr
        solver.weight_decay = self.solver_weight_decay
        solver.lr_policy = self.solver_lr_policy
        solver.display = self.solver_display
        solver.max_iter = self.solver_max_iter
        #solver.clip_gradients = self.solver_clip_gradients
        solver.snapshot = self.solver_snapshot
        solver.lr_policy = self.solver_lr_policy
        #solver.stepsize = self.solver_stepsize
        #solver.gamma = self.solver_gamma
        solver.momentum = self.solver_momentum
        solver.snapshot_prefix = self.solver_snapshot_prefix
        solver.random_seed = self.solver_random_seed
        #solver.solver_mode = self.solver_solver_mode
        solver.delta = self.solver_delta

        with open(self.solver_file, 'w') as f:
            f.write(str(solver))
    
    def make_net(self):
        print 'make network...'
        def conv_bn(bottom, ks_h, nout, weight_name, ks_w=1, stride=1, pad=0, group=1):
            conv = L.Convolution(bottom, kernel_h=ks_h, kernel_w=ks_w, stride=stride, num_output=nout, pad=pad,group=group,
                                        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'),
                                        param=[dict(name='conv_%s_w' % weight_name, lr_mult=1), 
                                       dict(name='conv_%s_b' % weight_name, lr_mult=2, decay_mult=0)]) 
            bn = L.BN(conv, scale_filler=dict(type='constant', value=1), shift_filler=dict(type='constant', value=1e-3),
                param=[dict(name='bn_%s_shape' % weight_name,lr_mult=1,decay_mult=0), dict(name='bn_%s_shift' % weight_name,lr_mult=1,decay_mult=0),
                       dict(name='bn_%s_mean' % weight_name,lr_mult=0,decay_mult=0), dict(name='bn_%s_variance' % weight_name,lr_mult=0,decay_mult=0)])

            return conv, bn
            
        def conv(bottom, ks_h, nout, weight_name, ks_w=1, stride=1, pad=0, group=1):
            conv = L.Convolution(bottom, kernel_h=ks_h, kernel_w=ks_w, stride=stride, num_output=nout, pad=pad,group=group,
                                        weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'),
                                        param=[dict(name='conv_%s_w' % weight_name, lr_mult=1), 
                                       dict(name='conv_%s_b' % weight_name, lr_mult=2, decay_mult=0)]) 

            return conv
        
        def pool_max(bottom, ks_h, ks_w=1, stride_h=1, stride_w=1):
            return L.Pooling(bottom, kernel_h=ks_h, kernel_w=ks_w, stride_h=stride_h, stride_w=stride_w, pool=P.Pooling.MAX)
        
        def pool_ave(bottom, ks_h, ks_w=1, stride_h=1, stride_w=1):
            return L.Pooling(bottom, kernel_h=ks_h, kernel_w=ks_w, stride_h=stride_h, stride_w=stride_w, pool=P.Pooling.AVE)
            
        def network_v3(lmdb, batch_size, include_eva=False):
            n = caffe.NetSpec()
            if include_eva:
                n.question, n.answer, n.label, n.group, n.overlap_feat = L.HDF5Data(batch_size=batch_size, source=lmdb, shuffle=0, ntop=5) # , n.overlap_feat
                w2v_source_name = ''
            else:
                n.question, n.answer, n.label, n.overlap_feat = L.HDF5Data(batch_size=batch_size, source=lmdb, shuffle=1, ntop=4) #
                w2v_source_name = '%s/wiki_dict.txt' % self.main_dir2                
                
            n.w2v_q = L.Embed(n.question, num_output=self.w2v_dim, input_dim=self.w2v_index_cc, weight_source=w2v_source_name,#bias_term=False,
                weight_filler=dict(type='uniform', min=-0.08, max=0.08), bias_filler=dict(type='constant', value=0),
                param=[dict(name='w2v-weights', decay_mult=0, lr_mult=1), dict(name='w2v-bias', decay_mult=0, lr_mult=2)])            
            n.w2v_a = L.Embed(n.answer, num_output=self.w2v_dim, input_dim=self.w2v_index_cc, #bias_term=False,
                weight_filler=dict(type='uniform', min=-0.08, max=0.08), bias_filler=dict(type='constant', value=0),
                param=[dict(name='w2v-weights', decay_mult=0, lr_mult=1), dict(name='w2v-bias', decay_mult=0, lr_mult=2)])
                
            #n.question_drop = L.Dropout(n.question, dropout_ratio=0.15)
            #n.answer_drop = L.Dropout(n.answer, dropout_ratio=0.15)
            
            #n.sim_cross = L.SimCross(n.w2v_q, n.w2v_a, dist_mode=2, mesure_count=2, param=[dict(name='embed-weights', decay_mult=1, lr_mult=1)])
            n.sim_cross = L.SimCross(n.w2v_q, n.w2v_a, dist_mode=0)
            #n.sim_drop = L.Dropout(n.sim_cross, dropout_ratio=0.2)#0.2
            #n.sim_bn = bn(n.sim_cross, 'sim-cross')
            
            n.conv0, n.bn0 = conv_bn(n.sim_cross, 5, 64, '0', ks_w=5)
            #n.conv0 = conv(n.sim_cross, 5, 64, '0', ks_w=5)
            n.pool0 = pool_max(n.bn0, 4, ks_w=4, stride_h=4, stride_w=4)            
            n.relu0 = L.TanH(n.pool0, in_place=True)           
            n.conv1, n.bn1 = conv_bn(n.relu0, 5, 64, '1', ks_w=5)
            #n.conv1 = conv(n.relu0, 5, 64, '1', ks_w=5)
            n.pool1 = pool_max(n.bn1, 5, ks_w=5)          
            n.relu1 = L.TanH(n.pool1, in_place=True)
            n.flt = L.Flatten(n.relu1)
            n.feat = L.Concat(n.flt, n.overlap_feat, concat_dim=1)
            
#            n.ques_rsp = L.Reshape(n.w2v_q, reshape_param=dict(shape=dict(dim=[0,1,self.question_max_word_len,-1])))
#            n.ans_rsp = L.Reshape(n.w2v_a, reshape_param=dict(shape=dict(dim=[0,1,self.answer_max_word_len,-1])))
#
#            n.conv0_q, n.bn0_q = conv_bn(n.ques_rsp, 5, 100, '1', ks_w=self.w2v_dim)
#            n.conv0_a, n.bn0_a = conv_bn(n.ans_rsp, 5, 100, '1', ks_w=self.w2v_dim)
#
#            n.pool0_q = pool_max(n.bn0_q, self.question_max_word_len-4)
#            n.pool0_a = pool_max(n.bn0_a, self.answer_max_word_len-4)
#            
#            n.tanh0_q = L.TanH(n.pool0_q, in_place=True)
#            n.tanh0_a = L.TanH(n.pool0_a, in_place=True)
#
#            n.sim = L.SimMatrix(n.tanh0_q, n.tanh0_a, weight_filler=dict(type='xavier')) #
#
#            n.flt_q = L.Flatten(n.tanh0_q)
#            n.flt_a = L.Flatten(n.tanh0_a)
#
#            n.feat = L.Concat(n.flt_q, n.flt_a, n.sim, n.overlap_feat, concat_dim=1) #, 

            n.fc1 = L.InnerProduct(n.feat, num_output=64, param=[dict(name='fc1-w', lr_mult=1, decay_mult=0),
                                    dict(name='fc1-b', lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
            n.relu1 = L.TanH(n.fc1, in_place=True)

            n.drop1 = L.Dropout(n.relu1, dropout_ratio=0.5)

            n.fc2 = L.InnerProduct(n.drop1, num_output=2, param=[dict(name='fc2-w',lr_mult=1, decay_mult=0),
                                    dict(name='fc2-b', lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))

            n.loss = L.SoftmaxWithLoss(n.fc2, n.label)#,loss_weight=[1])

            if include_eva:
                n.prob = L.Softmax(n.fc2)
                n.mrr = L.MRR(n.prob, n.label, n.group)
                n.map = L.MAP(n.prob, n.label, n.group)
                n.auc = L.AUC(n.prob, n.label)
            
            return n.to_proto()
            
        def network_v4(lmdb, batch_size, include_eva=False):
            n = caffe.NetSpec()
            if include_eva:
                n.question, n.answer, n.label, n.group, n.overlap_feat = L.HDF5Data(batch_size=batch_size, source=lmdb, shuffle=0, ntop=5) # , n.overlap_feat
                w2v_source_name = ''
            else:
                n.question, n.answer, n.label, n.overlap_feat = L.HDF5Data(batch_size=batch_size, source=lmdb, shuffle=1, ntop=4) #
                w2v_source_name = '%s/wiki_dict.txt' % self.main_dir2                
                
            n.w2v_q = L.Embed(n.question, num_output=self.w2v_dim, input_dim=self.w2v_index_cc, weight_source=w2v_source_name,#bias_term=False,
                weight_filler=dict(type='uniform', min=-0.08, max=0.08), bias_filler=dict(type='constant', value=0),
                param=[dict(name='w2v-weights', decay_mult=0, lr_mult=1), dict(name='w2v-bias', decay_mult=0, lr_mult=2)])            
            n.w2v_a = L.Embed(n.answer, num_output=self.w2v_dim, input_dim=self.w2v_index_cc, #bias_term=False,
                weight_filler=dict(type='uniform', min=-0.08, max=0.08), bias_filler=dict(type='constant', value=0),
                param=[dict(name='w2v-weights', decay_mult=0, lr_mult=1), dict(name='w2v-bias', decay_mult=0, lr_mult=2)])
   
            n.sim_cross = L.SimCross(n.w2v_q, n.w2v_a, dist_mode=2, mesure_count=4, bias_term=True, param=[dict(name='embed-weights', decay_mult=1, lr_mult=1)])
            #n.sim_cross = L.SimCross(n.w2v_q, n.w2v_a, dist_mode=1)
            n.sim_drop = L.Dropout(n.sim_cross, dropout_ratio=0.1)#0.2
            
            n.conv0, n.bn0 = conv_bn(n.sim_drop, 5, 32, '0', ks_w=5)
            n.pool0 = pool_ave(n.bn0, 4, ks_w=4, stride_h=4, stride_w=4)            
            n.relu0 = L.TanH(n.pool0, in_place=True)           
            n.conv1, n.bn1 = conv_bn(n.relu0, 5, 64, '1', ks_w=5)
            n.pool1 = pool_ave(n.bn1, 5, ks_w=5)          
            n.relu1 = L.TanH(n.pool1, in_place=True)
            n.flt = L.Flatten(n.relu1)
            n.feat = L.Concat(n.flt, n.overlap_feat, concat_dim=1)

            n.fc1 = L.InnerProduct(n.feat, num_output=32, param=[dict(name='fc1-w', lr_mult=1, decay_mult=0),
                                    dict(name='fc1-b', lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
            n.relu1 = L.TanH(n.fc1, in_place=True)
            n.drop1 = L.Dropout(n.relu1, dropout_ratio=0.5)
            n.fc2 = L.InnerProduct(n.drop1, num_output=2, param=[dict(name='fc2-w',lr_mult=1, decay_mult=0),
                                    dict(name='fc2-b', lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))

            n.loss = L.SoftmaxWithLoss(n.fc2, n.label)#,loss_weight=[1])

            if include_eva:
                n.prob = L.Softmax(n.fc2)
                n.mrr = L.MRR(n.prob, n.label, n.group)
                n.map = L.MAP(n.prob, n.label, n.group)
                n.auc = L.AUC(n.prob, n.label)
            
            return n.to_proto()
            
        def network_v4_2(lmdb, batch_size, include_eva=False):
            n = caffe.NetSpec()
            if include_eva:
                n.question, n.answer, n.label, n.group, n.overlap_feat = L.HDF5Data(batch_size=batch_size, source=lmdb, shuffle=0, ntop=5) # , n.overlap_feat
                w2v_source_name = ''
            else:
                n.question, n.answer, n.label, n.overlap_feat = L.HDF5Data(batch_size=batch_size, source=lmdb, shuffle=1, ntop=4) #
                w2v_source_name = '%s/wiki_dict.txt' % self.main_dir2                
                
            n.w2v_q = L.Embed(n.question, num_output=self.w2v_dim, input_dim=self.w2v_index_cc, weight_source=w2v_source_name,#bias_term=False,
                weight_filler=dict(type='uniform', min=-0.08, max=0.08), bias_filler=dict(type='constant', value=0),
                param=[dict(name='w2v-weights', decay_mult=0, lr_mult=1), dict(name='w2v-bias', decay_mult=0, lr_mult=2)])            
            n.w2v_a = L.Embed(n.answer, num_output=self.w2v_dim, input_dim=self.w2v_index_cc, #bias_term=False,
                weight_filler=dict(type='uniform', min=-0.08, max=0.08), bias_filler=dict(type='constant', value=0),
                param=[dict(name='w2v-weights', decay_mult=0, lr_mult=1), dict(name='w2v-bias', decay_mult=0, lr_mult=2)])
            
            n.sim_cross = L.SimCross(n.w2v_q, n.w2v_a, dist_mode=2, mesure_count=2, bias_term=False,
                          param=[dict(name='embed-weights', decay_mult=1, lr_mult=1)])
            #n.sim_cross = L.SimCross(n.w2v_q, n.w2v_a, dist_mode=0)
            #n.sim_drop = L.Dropout(n.sim_cross, dropout_ratio=0.1)#0.2
            #n.sim_bn = bn(n.sim_cross, 'sim-cross')
            
            n.conv0, n.bn0 = conv_bn(n.sim_cross, 5, 32, '0', ks_w=5)
            n.pool0 = pool_ave(n.bn0, 2, ks_w=2, stride_h=2, stride_w=2)            
            n.relu0 = L.TanH(n.pool0, in_place=True)           
            n.conv1, n.bn1 = conv_bn(n.relu0, 5, 32, '1', ks_w=5)
            n.pool1 = pool_ave(n.bn1, 2, ks_w=2, stride_h=2, stride_w=2)          
            n.relu1 = L.TanH(n.pool1, in_place=True)
            n.conv2, n.bn2 = conv_bn(n.relu1, 5, 32, '2', ks_w=5)
            n.pool2 = pool_ave(n.bn2, 3, ks_w=3, stride_h=3, stride_w=3)          
            n.relu2 = L.TanH(n.pool2, in_place=True)
            n.flt = L.Flatten(n.relu2)
            n.feat = L.Concat(n.flt, n.overlap_feat, concat_dim=1)

            n.fc1 = L.InnerProduct(n.feat, num_output=64, param=[dict(name='fc1-w', lr_mult=1, decay_mult=0),
                                    dict(name='fc1-b', lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
            n.relu1 = L.TanH(n.fc1, in_place=True)

            n.drop1 = L.Dropout(n.relu1, dropout_ratio=0.5)

            n.fc2 = L.InnerProduct(n.drop1, num_output=2, param=[dict(name='fc2-w',lr_mult=1, decay_mult=0),
                                    dict(name='fc2-b', lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))

            n.loss = L.SoftmaxWithLoss(n.fc2, n.label)#,loss_weight=[1])

            if include_eva:
                n.prob = L.Softmax(n.fc2)
                n.mrr = L.MRR(n.prob, n.label, n.group)
                n.map = L.MAP(n.prob, n.label, n.group)
                n.auc = L.AUC(n.prob, n.label)
            
            return n.to_proto()
            
        def network_v5(lmdb, batch_size, include_eva=False):
            n = caffe.NetSpec()
            if include_eva:
                n.question, n.answer, n.label, n.group, n.overlap_feat = L.HDF5Data(batch_size=batch_size, source=lmdb, shuffle=0, ntop=5) # , n.overlap_feat
                w2v_source_name = ''
            else:
                n.question, n.answer, n.label, n.overlap_feat = L.HDF5Data(batch_size=batch_size, source=lmdb, shuffle=1, ntop=4) #
                w2v_source_name = '%s/wiki_dict.txt' % self.main_dir2                
                
            n.w2v_q = L.Embed(n.question, num_output=self.w2v_dim, input_dim=self.w2v_index_cc, weight_source=w2v_source_name,#bias_term=False,
                weight_filler=dict(type='uniform', min=-0.08, max=0.08), bias_filler=dict(type='constant', value=0),
                param=[dict(name='w2v-weights', decay_mult=0, lr_mult=1), dict(name='w2v-bias', decay_mult=0, lr_mult=2)])            
            n.w2v_a = L.Embed(n.answer, num_output=self.w2v_dim, input_dim=self.w2v_index_cc, #bias_term=False,
                weight_filler=dict(type='uniform', min=-0.08, max=0.08), bias_filler=dict(type='constant', value=0),
                param=[dict(name='w2v-weights', decay_mult=0, lr_mult=1), dict(name='w2v-bias', decay_mult=0, lr_mult=2)])
            
            n.sim_cross = L.SimCross(n.w2v_q, n.w2v_a, dist_mode=2, mesure_count=2, param=[dict(name='embed-weights', decay_mult=1, lr_mult=1)])
            #n.sim_cross = L.SimCross(n.w2v_q, n.w2v_a, dist_mode=0)
            n.sim_drop = L.Dropout(n.sim_cross, dropout_ratio=0.2)#0.2
            #n.sim_bn = bn(n.sim_cross, 'sim-cross')
            
            n.conv0, n.bn0 = conv_bn(n.sim_drop, 3, 32, '0', ks_w=3)
            n.pool0 = pool_max(n.bn0, 2, ks_w=2, stride_h=2, stride_w=2)            
            n.relu0 = L.TanH(n.pool0, in_place=True)           
            n.conv1, n.bn1 = conv_bn(n.relu0, 4, 32, '1', ks_w=4)
            n.pool1 = pool_max(n.bn1, 2, ks_w=2, stride_h=2, stride_w=2)          
            n.relu1 = L.TanH(n.pool1, in_place=True)
            n.conv2, n.bn2 = conv_bn(n.relu1, 3, 32, '2', ks_w=3)
            n.pool2 = pool_max(n.bn2, 6, ks_w=6, stride_h=6, stride_w=6)          
            n.relu2 = L.TanH(n.pool2, in_place=True)
            n.flt = L.Flatten(n.relu2)
            n.feat = L.Concat(n.flt, n.overlap_feat, concat_dim=1)

            n.fc1 = L.InnerProduct(n.feat, num_output=32, param=[dict(name='fc1-w', lr_mult=1, decay_mult=0),
                                    dict(name='fc1-b', lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
            n.fc_relu1 = L.TanH(n.fc1, in_place=True)
            n.drop1 = L.Dropout(n.fc_relu1, dropout_ratio=0.5)
            n.fc2 = L.InnerProduct(n.drop1, num_output=2, param=[dict(name='fc2-w',lr_mult=1, decay_mult=0),
                                    dict(name='fc2-b', lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))

            n.loss = L.SoftmaxWithLoss(n.fc2, n.label)#,loss_weight=[1])

            if include_eva:
                n.prob = L.Softmax(n.fc2)
                n.mrr = L.MRR(n.prob, n.label, n.group)
                n.map = L.MAP(n.prob, n.label, n.group)
                n.auc = L.AUC(n.prob, n.label)
            
            return n.to_proto()
                               
                                                            
        with open(self.train_net_file, 'w') as f:
            f.write(str(network_v4(self.train_data_dir, self.train_batch_size)))

        with open(self.test_net_file, 'w') as f:
            f.write(str(network_v4(self.test_data_dir, self.test_batch_size, True)))
    
        with open(self.dev_net_file, 'w') as f:
            f.write(str(network_v4(self.dev_data_dir, self.dev_batch_size, True)))
    
    def test_networks(self, modelfilename, test_net_file):
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        net = caffe.Net(test_net_file, modelfilename, caffe.TEST)
        net.forward()

        label = net.blobs['label'].data.astype(int)
        group = net.blobs['group'].data.astype(float)
        prob = net.blobs['prob'].data

        print 'test_net map:'+str(net.blobs['map'].data)+' mrr:'+str(net.blobs['mrr'].data)

        data_all = {}
        for idx in range(label.shape[0]):
            if group[idx] in data_all:
                data_all[group[idx]]['prob'].append(prob[idx, 1])
                data_all[group[idx]]['label'].append(label[idx])
                data_all[group[idx]]['check'] = data_all[group[idx]]['check']+label[idx]
            else:
                data_all[group[idx]] = dict(prob=[prob[idx, 1]], label=[label[idx]], check=label[idx])

        with open('%s/truth' % self.main_dir2, 'w') as f:
            for groupid in data_all:
                if data_all[groupid]['check'] > 0 and data_all[groupid]['check'] < len(data_all[groupid]['label']):
                    for idx in range(len(data_all[groupid]['label'])):
                        f.write(' '.join(map(str,[groupid+1, 0, idx, data_all[groupid]['label'][idx], '\n'])))

        with open('%s/result' % self.main_dir2, 'w') as f:
            for groupid in data_all:
                if data_all[groupid]['check'] > 0 and data_all[groupid]['check'] < len(data_all[groupid]['label']):
                    for idx in range(len(data_all[groupid]['prob'])):
                        f.write(' '.join(map(str,[groupid+1, 0, idx, 1, data_all[groupid]['prob'][idx], 'glove', '\n'])))

        os.system('./data/trec_qa/trec_eval-8.0/trec_eval  %s/truth %s/result' %(self.main_dir2, self.main_dir2))
        #subprocess.call(['rm', 'truth'])
        #subprocess.call(['rm', 'result'])
        
    def test_networks_single(self, netfilename, modelfilename, datafile):
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        net = caffe.Net(netfilename, modelfilename, caffe.TEST)
        
        def load_data(fname):
            lines = open(fname).readlines()
            qids, questions, answers, labels = [], [], [], []
            num_skipped = 0
            prev = ''
            qid2num_answers = {}
            for i, line in enumerate(lines):
              line = line.strip()
          
              qid_match = re.match('<QApairs id=\'(.*)\'>', line)
          
              if qid_match:
                qid = qid_match.group(1)
                qid2num_answers[qid] = 0
          
              if prev and prev.startswith('<question>'):
                question = line.lower().split('\t')
          
              label = re.match('^<(positive|negative)>', prev)
              if label:
                label = label.group(1)
                label = 1 if label == 'positive' else 0
                answer = line.lower().split('\t')
                if len(answer) > 60:
                  num_skipped += 1
                  continue
                labels.append(label)
                answers.append(answer)
                questions.append(question)
                qids.append(qid)
                qid2num_answers[qid] += 1
              prev = line
            # print sorted(qid2num_answers.items(), key=lambda x: float(x[0]))
            print 'num_skipped', num_skipped
            return qids, questions, answers, labels

        def compute_overlap_features(questions, answers, word2df=None, stoplist=None):
            word2df = word2df if word2df else {}
            stoplist = stoplist if stoplist else set()
            feats_overlap = []
            for question, answer in zip(questions, answers):
                q_set = set([q for q in question if q not in stoplist])
                a_set = set([a for a in answer if a not in stoplist])
                word_overlap = q_set.intersection(a_set)            

                df_overlap = 0.0
                for w in word_overlap:
                    if w in word2df:
                        df_overlap += word2df[w]
                    else:
                        df_overlap += 11.0

                overlap_feat = [float(len(word_overlap)) / (len(q_set)+len(a_set)), df_overlap / (len(q_set)+len(a_set))] 
                                #float(len(question))/self.question_max_word_len, float(len(answer))/self.answer_max_word_len] #   #       
                feats_overlap.append(np.array(overlap_feat))
                
            return np.array(feats_overlap)

        def vocab_transform_embed(target_input, maxlen):
            def word_to_vecindex(x):
                if not (x in self.w2v_index):
                    return self.unknown_word_index
                return self.w2v_index[x]

            target_line = [word_to_vecindex(x) for x in target_input]
           
            slen = len(target_input)
            #target_line = target_line[:maxlen*self.w2v_dim] + self.zero_word*max(0,maxlen-slen)

            pad_b = max(0, int((maxlen-slen)/2))
            pad_a = max(0, maxlen-pad_b-slen)
            target_line = [self.zero_word_index]*pad_b + target_line[:maxlen] + [self.zero_word_index]*pad_a
            
            #print target_line, self.unknown_word_index, self.w2v_index_cc
            #exit(1)

            assert len(target_line) == maxlen
            return target_line

        qids, qst, ans, labels = load_data(datafile)
        
        qids_uni = list(set(qids))
        qids_new = [qids_uni.index(x) for x in qids]
    
        overlap_feats = compute_overlap_features(qst, ans, stoplist=self.stop_words, word2df=self.word2dfs)
        #overlap_feats_stop = compute_overlap_features(questions, answers, stoplist=self.stop_words, word2df=self.word2dfs)
        #overlap_feats = np.hstack([overlap_feats, overlap_feats_stop])
        assert(overlap_feats.shape[0] == len(qst))
        
        assert(len(qst) == len(ans))
        print 'Writing %s sentences' % len(qst)
        allQ = []
        allA = []
        for idx in range(len(qst)):
            allQ.append(vocab_transform_embed(qst[idx], self.question_max_word_len))
            allA.append(vocab_transform_embed(ans[idx], self.answer_max_word_len))
        
        #dataG = np.array(qids_new, np.float)
        
        net.blobs['question'].data[...] = np.array(allQ, np.float).reshape([1, self.char_max_num_per_sentence1, 1, 1])
        net.blobs['answer'].data[...] = np.array(allA, np.float).reshape([1, self.char_max_num_per_sentence2, 1, 1])
        net.blobs['label'].data[...] = np.array(labels, np.float).reshape([1, 1, 1, 1])
        
        labels = np.zeros((len(allL),2))
        labels[:, allL] = 1
        
        out = net.forward()
        bw = net.backward(end='c2v_q')
        bw1 = net.backward(end='c2v_a')
        print out.keys(), bw.keys(), out
        #bw = net.backward(**{net.outputs[0]: labels})#
        diff1 = bw['c2v_q'].reshape([self.char_max_num_per_sentence1, self.c2v_dim])
        diff2 = bw1['c2v_a'].reshape([self.char_max_num_per_sentence2, self.c2v_dim])
        
        print np.sum(np.fabs(diff1), 1), np.sum(np.fabs(diff2), 1)
        
        plt.subplot(2,1,1)
        plt.imshow(diff1, cmap=cm.gray_r)
        plt.subplot(2,1,2)
        plt.imshow(diff2, cmap=cm.gray_r)
        plt.savefig('%s/saliency.png' % self.main_dir2)
        
    
    def do_learn(self):
        print 'learning...'
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        solver = caffe.AdaDeltaSolver(self.solver_file)
        #solver.step(self.solver_max_iter)
        
        # losses will also be stored in the log
        train_loss = np.array([], 'float32')
        #train_loss_ce = np.array([], 'float32')

        test_mrr = np.array([])
        test_map = np.array([])
        test_auc = np.array([])
        dev_mrr = np.array([])
        dev_map = np.array([])
        dev_auc = np.array([])

        best_dev_acc = 0
        wait_epoch = 0
        epoch = -1
        wait_patience = 5
        
        # the main solver loop
        for it in range(self.solver_max_iter):
            solver.step(1)  # SGD by Caffe        
            
            # store the train loss
            # store the train loss
            if it % self.train_loss_record_interval == 0:
                train_loss = np.append(train_loss, solver.net.blobs['loss'].data)
            
            #solver.test_nets[0].forward(start='conv1')
            #output[it] = solver.test_nets[0].blobs['ip2'].data[:8]
                                                    
            if it % self.solver_test_interval == 0:
                #solver.net.save()
                test_mrr = np.append(test_mrr, solver.test_nets[0].blobs['mrr'].data)
                test_map = np.append(test_map, solver.test_nets[0].blobs['map'].data)
                test_auc = np.append(test_auc, solver.test_nets[0].blobs['auc'].data)
                dev_mrr = np.append(dev_mrr, solver.test_nets[-1].blobs['mrr'].data)
                dev_map = np.append(dev_map, solver.test_nets[-1].blobs['map'].data)
                dev_auc = np.append(dev_auc, solver.test_nets[-1].blobs['auc'].data)
                
                # early stopping judge
                if dev_map[-1] > best_dev_acc or best_dev_epoch == 0:
                    wait_epoch = 0
                    best_dev_epoch = it // self.solver_test_interval
                    best_dev_acc = dev_map[-1]
                
            if it*self.train_batch_size >= epoch*self.train_size:
                epoch = epoch + 1
                wait_epoch = wait_epoch + 1

                if wait_epoch > wait_patience:
                    print 'Epoch-%d [%d] early stop~' % (epoch, it)
                    break;
                    
        self.test_networks(self.solver_snapshot_prefix+'_iter_%d.caffemodel' % (best_dev_epoch*self.solver_test_interval), test_net_file=self.test_net_file)
        #self.test_networks(self.solver_snapshot_prefix+'_iter_%d.caffemodel' % (best_dev_epoch*self.solver_test_interval), test_net_file=self.dev_net_file)
        
        #midx = np.argmax(dev_map)
        midx = best_dev_epoch
        print "dev-data's max-map is %d,%f, test-data map: %f mrr: %f" % (midx, dev_map[midx], test_map[midx], test_mrr[midx])
        #print test_map, test_mrr, dev_map, dev_mrr
        fig = plt.figure()
        p1=plt.plot(self.solver_test_interval * arange(len(test_map)), test_map, 'r', label='test-data map')
        p2=plt.plot(self.solver_test_interval * arange(len(test_mrr)), test_mrr, 'g', label='test-data mrr')
        p3=plt.plot(self.solver_test_interval * arange(len(dev_map)), dev_map, 'm', label='dev-data map')
        p4=plt.plot(self.solver_test_interval * arange(len(dev_mrr)), dev_mrr, 'y', label='dev-data mrr')
        p5=plt.plot(self.solver_test_interval * arange(len(test_auc)), test_auc, 'b', label='test-data auc')
        p6=plt.plot(self.solver_test_interval * arange(len(dev_auc)), dev_auc, 'k', label='dev-data auc')
        plt.legend(loc=4, ncol=2, fontsize=10)
        plt.xlabel('iteration')
        plt.ylabel('cret')
        plt.title('figure')
        plt.savefig('%s/curve.png' % self.main_dir2)
        
        plt.figure()
        p1=plt.plot(self.train_loss_record_interval * arange(len(train_loss)), train_loss, 'r', label='train loss')
        #p2=plt.plot(arange(len(train_loss_ce)), train_loss_ce, 'g', label='cross entropy loss')
        plt.legend(loc=1, ncol=2, fontsize=10)
        plt.xlabel('iteration')
        plt.ylabel('criteria')
        plt.title('figure')
        plt.savefig('%s/loss-curve.png' % self.main_dir2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--make_data', action='store_true')
    parser.add_argument('--exp_name', help='experiment_name', type=str)
    parser.add_argument('--main_dir', help='main_directory', type=str)
    parser.add_argument('--gpu_id', help='gpu_id', type=str)
    parser.add_argument('--trainmode', help='trainmode', type=str)
    parser.add_argument('--test_model_index', help='test model index', type=str)
    args = parser.parse_args()
    if not args.exp_name or not args.main_dir:
        exit(1)
    print args.main_dir
    gpu_id = 1
    if args.gpu_id:
        gpu_id = int(args.gpu_id)
    trainmode = 'train'
    if args.trainmode:
        trainmode=args.trainmode
    if args.test_model_index:
        job = qa_caffe(args.main_dir, args.exp_name, gpu_id, check_model_dir=False, trainmode=trainmode)
        job.test_networks(job.solver_snapshot_prefix+'_iter_%d.caffemodel' % int(args.test_model_index))
    else:
        job = qa_caffe(args.main_dir, args.exp_name, gpu_id, trainmode=trainmode)
        if args.make_data:
            job.make_data(trainmode)
        subprocess.call(['cp', job.main_dir+'/do_trec_qa_clean.py', job.main_dir2+'/'])
        job.make_solver()
        job.make_net()
        job.do_learn()
