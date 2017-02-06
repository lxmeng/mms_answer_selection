#include <cfloat>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/sim_matrix_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SimMatrixLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());

  K1_ = bottom[0]->count(1);
  K2_ = bottom[1]->count(1);
  
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = K1_;
    weight_shape[1] = K2_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.sim_matrix_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void SimMatrixLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_K1 = bottom[0]->count(1);
  const int new_K2 = bottom[1]->count(1);
  CHECK_EQ(K1_, new_K1)
      << "Input size incompatible with inner product parameters.";
  CHECK_EQ(K2_, new_K2)
      << "Input size incompatible with inner product parameters.";
  M_ = bottom[0]->count(0, 1);
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(2);
  top_shape[1] = 1;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SimMatrixLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data1 = bottom[0]->cpu_data();
  const Dtype* bottom_data2 = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* temp_data = bottom[1]->mutable_cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K2_, K1_, (Dtype)1.,
      bottom_data1, weight, (Dtype)0., temp_data);
  for(int i = 0; i < M_; i ++) {
    top_data[i] = caffe_cpu_dot<Dtype>(K2_, bottom_data2+i*K2_, temp_data+i*K2_);
  }
}

template <typename Dtype>
void SimMatrixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();  
  const Dtype* top_diff = top[0]->cpu_diff();
  
  if (this->param_propagate_down_[0]) {
    const Dtype* bottom_data1 = bottom[0]->cpu_data();
    const Dtype* bottom_data2 = bottom[1]->cpu_data();
    for(int i = 0; i < M_; i ++) {
      caffe_cpu_ger<Dtype>(K1_, K2_, top_diff[i], bottom_data1 + (i*K1_), 
        bottom_data2 + (i*K2_), this->blobs_[0]->mutable_cpu_diff());
    }
  }
  for(int i = 0; i < 2; i ++) {    
    if (propagate_down[i]) {
      const int step1 = i == 0 ? K2_:K1_;
      const int step2 = i == 1 ? K2_:K1_;
      const CBLAS_TRANSPOSE trans = i == 0 ? CblasNoTrans : CblasTrans;
      // Gradient with respect to bottom data
      for(int j = 0; j < M_; j ++) {
        caffe_cpu_gemv<Dtype>(trans, K1_, K2_, top_diff[j], weight, 
          bottom[1-i]->cpu_data()+(j*step1), Dtype(0), 
          bottom[i]->mutable_cpu_diff()+(j*step2));
      }
    }
  }
  
}

#ifdef CPU_ONLY
STUB_GPU(SimMatrixLayer);
#endif

INSTANTIATE_CLASS(SimMatrixLayer);
REGISTER_LAYER_CLASS(SimMatrix);

}  // namespace caffe
