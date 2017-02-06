#include <algorithm>
#include <vector>

#include "caffe/layers/pair_rank_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PairRankLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(bottom[0]->count(1), bottom[2]->count(1));
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1));
  
  margin_ = (Dtype)this->layer_param_.pair_rank_loss_param().margin();
  
  ordered_diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  similar_diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void PairRankLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      ordered_diff_.mutable_cpu_data());  // a_i-b_i
  caffe_copy(count, ordered_diff_.cpu_data(), similar_diff_.mutable_cpu_data());
  
  caffe_mul(count, ordered_diff_.cpu_data(), bottom[2]->cpu_data(), ordered_diff_.mutable_cpu_data());
  caffe_cpu_axpby(count, Dtype(-1), ordered_diff_.cpu_data(), Dtype(0), ordered_diff_.mutable_cpu_data());
  caffe_add_scalar(count, margin_, ordered_diff_.mutable_cpu_data());
  
  const int dim = count/bottom[0]->num();
  Dtype loss(0.0);  
  for (int i = 0; i < bottom[0]->num(); i ++) {
    for(int j = 0; j < dim; j ++) {
      loss += std::max(Dtype(0), ordered_diff_.cpu_data()[i*dim+j]) + //ordered
      std::abs((1 - bottom[2]->cpu_data()[i*dim+j]) * similar_diff_.cpu_data()[i*dim+j]);//similar
    } 
  }
  
  //printf("loss is %f\n", loss);
  loss /= static_cast<Dtype>(bottom[0]->count());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void PairRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  for (int i = 0; i < 2; i ++) {
    if (propagate_down[i]) {
      Dtype sign = (i == 0) ? -1 : 1;
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      
      int num = bottom[i]->num();
      int dim = bottom[i]->count()/num;
      
      sign *= top[0]->cpu_diff()[0] / bottom[0]->count();
      
      for (int j = 0; j < num; j ++) {
        for (int k = 0; k < dim; k ++) {
          //Dtype tt = (ordered_diff_.cpu_data()[j*dim+k] + (1 - bottom[2]->cpu_data()[j*dim+k]) * similar_diff_.cpu_data()[j*dim+k]) >= 0 ? Dtype(1) : Dtype(0);
          //bottom_diff[j*dim+k] = tt * (sign * bottom[2]->cpu_data()[j*dim+k] - sign * (1 - bottom[2]->cpu_data()[j*dim+k]));
          Dtype ordered_t = ordered_diff_.cpu_data()[j*dim+k] > 0 ? Dtype(1) : Dtype(0);
          Dtype similar_t = (1 - bottom[2]->cpu_data()[j*dim+k]) * similar_diff_.cpu_data()[j*dim+k] > 0 ? Dtype(1) : Dtype(-1);
          
          bottom_diff[j*dim+k] = sign * (ordered_t * bottom[2]->cpu_data()[j*dim+k] - similar_t * (1 - bottom[2]->cpu_data()[j*dim+k]));
        }
      }
    }
  }
}

INSTANTIATE_CLASS(PairRankLossLayer);
REGISTER_LAYER_CLASS(PairRankLoss);

}  // namespace caffe
