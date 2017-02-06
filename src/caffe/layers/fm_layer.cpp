#include <cfloat>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/fm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bias_term_ = this->layer_param_.fm_param().bias_term();
  if (bias_term_) {
    this->blobs_.resize(1);
    vector<int> bias_shape(1);
    bias_shape[0] = 1;
    this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
    this->blobs_[0]->mutable_cpu_data()[0] = Dtype(0);
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void FMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(1);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void FMLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  const int num = bottom[0]->num();
  const int channel = bottom[0]->channels();
  const int dim = bottom[0]->height();
  
  for(int i = 0; i < num; i ++) {
    Dtype t1 = 0;
    for(int j = 1; j < dim; j ++) {
      Dtype t2 = 0;
      for(int k = 0; k < channel; k ++) {
        const int ind = i*channel*dim+k*dim+j;
        t2+=bottom_data[ind];
        t1-=bottom_data[ind]*bottom_data[ind];
      }
      t1+=t2*t2;
    }
    t1/=2;
    for(int k = 0; k < channel; k ++) {
      t1+=bottom_data[i*channel*dim+k*dim];
    }
    if(bias_term_) {
      t1+=this->blobs_[0]->cpu_data()[0];
    }
    top_data[i]=t1;
  }
}

template <typename Dtype>
void FMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  
  const int num = bottom[0]->num();
  const int channel = bottom[0]->channels();
  const int dim = bottom[0]->height();
  
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  if (bias_term_ && this->param_propagate_down_[0]) {
    this->blobs_[0]->mutable_cpu_diff()[0] = 0.;
    for(int i = 0; i < num; i ++) {
      this->blobs_[0]->mutable_cpu_diff()[0]+=top_diff[i];
    }
  }
  if(propagate_down[0]) {
    for(int i = 0; i < num; i ++) {
      for(int k = 0; k < channel; k ++) {
        bottom_diff[i*channel*dim+k*dim]=top_diff[i];
      }
      for(int j = 1; j < dim; j ++) {
        Dtype tt = 0;
        for(int k = 0; k < channel; k ++) {
          tt+=bottom_data[i*channel*dim+k*dim+j];
        }
        for(int k = 0; k < channel; k ++) {
          const int ind = i*channel*dim+k*dim+j;
          bottom_diff[ind]=top_diff[i]*(tt-bottom_data[ind]);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FMLayer);
#endif

INSTANTIATE_CLASS(FMLayer);
REGISTER_LAYER_CLASS(FM);

}  // namespace caffe
