#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/embed_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EmbedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  N_ = this->layer_param_.embed_param().num_output();
  CHECK_GT(N_, 0) << "EmbedLayer num_output must be positive.";
  K_ = this->layer_param_.embed_param().input_dim();
  CHECK_GT(K_, 0) << "EmbedLayer input_dim must be positive.";
  bias_term_ = this->layer_param_.embed_param().bias_term();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights --
    // transposed from InnerProductLayer for spatial locality.
    vector<int> weight_shape(2);
    weight_shape[0] = K_;
    weight_shape[1] = N_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.embed_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());    
    
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.embed_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
    
    if(this->layer_param_.embed_param().weight_source() != "") {
      const char* wsfilename = this->layer_param_.embed_param().weight_source().c_str();
      int wsfn_len = strlen(wsfilename);
      char theword[256];
      int w_index = 0;
      Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
      if(strcmp(wsfilename+wsfn_len-3, "txt") == 0) {
        LOG(INFO) << "loading word embeddings txt";
        FILE* weight_file = fopen(wsfilename, "r");
        while(fscanf(weight_file, "%s ", theword) != EOF) {
          for(int i = 0; i < N_; i ++) {
            fscanf(weight_file, "%f ", (float*)(weight_data+w_index));
            w_index ++;
          }
        }
        fclose(weight_file);
      }
      else if(strcmp(wsfilename+wsfn_len-3, "all") == 0) {
        LOG(INFO) << "loading word embeddings all";
        FILE* weight_bias_file = fopen(wsfilename, "r");
        Dtype b1;
        int t1, t2;
        fscanf(weight_bias_file, "%f %d %d", (float*)&b1, &t1, &t2);
        CHECK_EQ(t1, K_-1);
        CHECK_EQ(t2, N_-1);
        while(fscanf(weight_bias_file, "%d ", &t1) != EOF) {
          for(int i = 0; i < N_; i ++) {
            fscanf(weight_bias_file, "%f ", (float*)(weight_data+w_index));
            w_index ++;
          }
          fscanf(weight_bias_file, "%s", theword);
        }
        fclose(weight_bias_file);
      }
      else {
        LOG(INFO) << "loading word embeddings bin";
        long long vsize_t, dim_t, a, b;
        FILE* weight_file = fopen(wsfilename, "rb");
        fscanf(weight_file, "%lld", &vsize_t);
        fscanf(weight_file, "%lld", &dim_t);
        LOG(INFO) << "vocab_size: " << vsize_t << ", vec_dim: " << dim_t;
        CHECK_EQ(dim_t, N_);
        
        for (b = 0; b < vsize_t; b++) {
          a = 0;
          while (1) {
            theword[a] = fgetc(weight_file);
            if (feof(weight_file) || (theword[a] == ' ')) break;
            if ((a < 256) && (theword[a] != '\n')) a++;
          }
          theword[a] = 0;
          for (a = 0; a < dim_t; a++) fread((float*)(weight_data+(w_index++)), sizeof(float), 1, weight_file);
          
          //LOG(INFO) << w_index/N_ << " & " << theword << " & " << *(weight_data+w_index-N_) << " & " << *(weight_data+w_index-1);
          //if(w_index/N_ > 5) {
          //  exit(1);
          //}
        }
        fclose(weight_file);       
      }
      LOG(INFO) << "loaded " << w_index/N_ << " words [" << weight_data[w_index-N_] << "," << weight_data[w_index-1] << "]";
      //LOG(INFO) << w_index/N_ << " & " << theword << " & " << *(weight_data+w_index-N_) << " & " << *(weight_data+w_index-1);
      //exit(1);
      //CHECK_EQ(w_index, (K_-2)*N_); // exclude zero_pad_symbol and unknown_word_symbol
    }
    else {
      LOG(INFO) << "Not loading word embeddings";
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void EmbedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  M_ = bottom[0]->count();
  vector<int> top_shape = bottom[0]->shape();
  top_shape.push_back(N_);
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void EmbedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int index;
  for (int n = 0; n < M_; ++n) {
    index = static_cast<int>(bottom_data[n]);
    DCHECK_GE(index, 0);
    DCHECK_LT(index, K_);
    DCHECK_EQ(static_cast<Dtype>(index), bottom_data[n]) << "non-integer input";
    caffe_copy(N_, weight + index * N_, top_data + n * N_);
  }
  if (bias_term_) {
    const Dtype* bias = this->blobs_[1]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1),
        bias_multiplier_.cpu_data(), bias, Dtype(1), top_data);
  }
}

template <typename Dtype>
void EmbedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[0]) << "Can't backpropagate to EmbedLayer input.";
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    int index;
    for (int n = 0; n < M_; ++n) {
      index = static_cast<int>(bottom_data[n]);
      DCHECK_GE(index, 0);
      DCHECK_LT(index, K_);
      DCHECK_EQ(static_cast<Dtype>(index), bottom_data[n])
          << "non-integer input";
      caffe_axpy(N_, Dtype(1), top_diff + n * N_, weight_diff + index * N_);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, Dtype(1), top_diff,
        bias_multiplier_.cpu_data(), Dtype(1), bias_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(EmbedLayer);
#endif

INSTANTIATE_CLASS(EmbedLayer);
REGISTER_LAYER_CLASS(Embed);

}  // namespace caffe
