#include <cfloat>
#include <vector>

#include "caffe/layers/sim_matrix_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void SMForward(const int n, const Dtype* in1, const Dtype* in2, 
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in1[index] * in2[index];
  }
}

template <typename Dtype>
void SimMatrixLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data1 = bottom[0]->gpu_data(); 
  const Dtype* bottom_data2 = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* temp_data = bottom[1]->mutable_gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K2_, K1_, (Dtype)1.,
      bottom_data1, weight, (Dtype)0., temp_data);
  
  const Dtype* temp_data2 = bottom[1]->gpu_diff();
  SMForward<Dtype><<<CAFFE_GET_BLOCKS(M_*K2_), CAFFE_CUDA_NUM_THREADS>>>(
      M_*K2_, bottom_data2, temp_data2, temp_data);
  CUDA_POST_KERNEL_CHECK;
  for(int i = 0; i< M_; i ++) {
    *(top_data+i)=0;
    for(int j = 0; j < K2_; j ++) {
      *(top_data+i)+=*(bottom[1]->cpu_diff()+i*K2_+j);
    }
  } 
}

template <typename Dtype>
void SimMatrixLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(SimMatrixLayer);

}  // namespace caffe
