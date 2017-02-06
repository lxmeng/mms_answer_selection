#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/mrr_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MRRLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  fixed_axis_ = this->layer_param_.mrr_param().fixed_axis();
}

template <typename Dtype>
void MRRLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(fixed_axis_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  int outer_num_ = bottom[0]->count(0, 1);
  int inner_num_ = bottom[0]->count(2);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  CHECK_EQ(outer_num_ * inner_num_, bottom[2]->count());
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

bool mycompare_mrr(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

template <typename Dtype>
void MRRLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* bottom_group = bottom[2]->cpu_data();
  
  map<int, vector<std::pair<float, int> > > all_data;

  //LOG(INFO) << fixed_axis_ << "  nnnnnnn";
  
  for(size_t i = 0; i < bottom[0]->num(); i ++) {
    all_data[bottom_group[i]].push_back(std::make_pair(bottom_data[i*(fixed_axis_+1)+fixed_axis_],bottom_label[i]));
  }
  
  Dtype mrr = Dtype(0);
  int effect_sample = 0;
  map<int, vector<std::pair<float, int> > >::iterator it;
  
  for(it = all_data.begin(); it != all_data.end(); it ++) {
    std::sort(it->second.begin(), it->second.end(), mycompare_mrr); //std::greater<std::pair<Dtype, int> >());
    int mrr_rank = -1;
    int neg_exist = 0;
    for(size_t i = 0; i < it->second.size(); i ++) {
      if(mrr_rank < 0 && it->second[i].second == 1) {
        mrr_rank = i;
      }
      if(neg_exist == 0 && it->second[i].second == 0) {
        neg_exist = 1;
      }
      if(neg_exist && mrr_rank > -1) {
        break;
      }
    }
    if(mrr_rank < 0 || neg_exist == 0) {
      continue;
    }
    effect_sample ++;
    mrr += 1.0/(mrr_rank+1);
  }
  //LOG(INFO) << "effective[" << effect_sample << "] vs All[" << all_data.size() << "]";
  top[0]->mutable_cpu_data()[0] = mrr / effect_sample;
}

INSTANTIATE_CLASS(MRRLayer);
REGISTER_LAYER_CLASS(MRR);

}  // namespace caffe
