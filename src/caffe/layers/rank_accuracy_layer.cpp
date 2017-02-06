#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/rank_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RankAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void RankAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(),  bottom[1]->count())
      << "two pairs have the same dimension!.";

  CHECK_EQ(bottom[0]->count(),  bottom[2]->count())
      << "pair should have the same dimension with the label!.";
  int outer_num_ = bottom[0]->count(0, 1);
  int inner_num_ = bottom[0]->count(1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[2]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RankAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype acc_value = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data_p = bottom[1]->cpu_data();
  const Dtype* bottom_label = bottom[2]->cpu_data();
  //const int dim = bottom[0]->count() / bottom[0]->num();
  const int count = bottom[0]->count();
  for(int i = 0; i < count; i ++) {
    acc_value += (bottom_label[i] * (bottom_data[i] - bottom_data_p[i])) > 0 ? 1 : 0;
  }
  //caffe_cpu_sign(count, diff_, diff_);
  //acc_value = caffe_cpu_asum(count, diff_);
  top[0]->mutable_cpu_data()[0] = acc_value/count;
}

INSTANTIATE_CLASS(RankAccuracyLayer);
REGISTER_LAYER_CLASS(RankAccuracy);

}  // namespace caffe
