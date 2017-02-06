#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pair_rank_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PairRankLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
#if 0
  Forward_cpu(bottom, top);// adopt cpu version
#else
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      ordered_diff_.mutable_gpu_data());  // a_i-b_i 
   
  caffe_copy(count, ordered_diff_.gpu_data(), similar_diff_.mutable_gpu_data()); 
  const int dim = count/bottom[0]->num();
  Dtype loss(0.0);  
  
  caffe_gpu_mul(count, ordered_diff_.gpu_data(), bottom[2]->gpu_data(), ordered_diff_.mutable_gpu_data());
  
  caffe_gpu_axpby(count, Dtype(-1), ordered_diff_.gpu_data(), Dtype(0), ordered_diff_.mutable_gpu_diff());
  
  for (int i = 0; i < bottom[0]->num(); ++i) {
    for(int j = 0; j < dim; j ++) {
      ordered_diff_.mutable_cpu_diff()[i*dim+j] += margin_;
      loss += std::max(Dtype(0), ordered_diff_.cpu_diff()[i*dim+j]) + //ordered
          std::abs((Dtype(1)-bottom[2]->cpu_data()[i*dim+j])*similar_diff_.cpu_data()[i*dim+j]);
    } 
  }
  caffe_copy(count, ordered_diff_.cpu_diff(), ordered_diff_.mutable_cpu_data());
  
  loss = loss / static_cast<Dtype>(count);
  top[0]->mutable_cpu_data()[0] = loss;
#endif
}

template <typename Dtype>
__global__ void PairRankLossBackward(const int count,const Dtype sign,
               const Dtype* label, const Dtype* ordered_diff, 
               const Dtype* similar_diff, 
               Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    Dtype ordered_t = ordered_diff[i] >= 0 ? 1 : 0;
    Dtype similar_t = (1 - label[i]) * similar_diff[i] > 0 ? Dtype(1) : Dtype(-1);
    bottom_diff[i] = sign * (ordered_t * label[i] - similar_t * (1 - label[i]));
  }
}

template <typename Dtype>
void PairRankLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
#if 0
  Backward_cpu(top, propagate_down, bottom);
#else
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[i]->count();
      const int dim = count/bottom[i]->num();
      
      Dtype sign = (i == 0) ? -1 : 1;
      sign *= top[0]->cpu_diff()[0] / static_cast<Dtype>(count);

      // NOLINT_NEXT_LINE(whitespace/operators)
      PairRankLossBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, sign, bottom[2]->gpu_data(),  // pair similarity 0 or 1
          ordered_diff_.gpu_data(),  // the cached eltwise difference between a and b
          similar_diff_.gpu_data(),
	        bottom[i]->mutable_gpu_diff());
      //CUDA_POST_KERNEL_CHECK;
    }
  }
  
#if 0
  printf("hello-backward-gpu\n");
  int k = propagate_down[0] == 1 ? 0 : 1; 
  printf("\n\n\n----------------bgn(%d)------------------\n\n\n", k);
  for(int i = 0; i < bottom[k]->count(); i ++) {
    printf("[%f,%f]  ", bottom[k]->cpu_data()[i], bottom[k]->cpu_diff()[i]);
  }
  printf("\n\n\n");
  for(int i = 0; i < top[0]->count(); i ++) {
    printf("[%f,%f]  ", top[0]->cpu_data()[i], top[0]->cpu_diff()[i]);
  }
  printf("\n\n\n----------------end(%d)------------------\n\n\n", k);
#endif
  
#endif

}

INSTANTIATE_LAYER_GPU_FUNCS(PairRankLossLayer);

}  // namespace caffe
