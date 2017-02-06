#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/map_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MAPLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  fixed_axis_ = this->layer_param_.map_param().fixed_axis();
}

template <typename Dtype>
void MAPLayer<Dtype>::Reshape(
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

//template <typename Dtype>
bool mycompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
    //if (fabs(lhs.first - rhs.first) < 1e-10) 
    //    return caffe_rng_rand()%2 == 1; 
    return lhs.first > rhs.first;
}

template <typename Dtype>
void MAPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* bottom_group = bottom[2]->cpu_data();
  
  map<int, vector<std::pair<float, int> > > all_data;

  for(size_t i = 0; i < bottom[0]->num(); i ++) {
    all_data[bottom_group[i]].push_back(std::make_pair(bottom_data[i*(fixed_axis_+1)+fixed_axis_],bottom_label[i]));
  }

#if 0
  LOG(INFO) << "fixed_axis_" << fixed_axis_;
  FILE* fout = fopen("1-1", "w");
  for(int i = 0; i < bottom[0]->num(); i ++) {
    fprintf(fout, "%f\t%d\t%d\n", bottom_data[i*(fixed_axis_+1)+fixed_axis_], int(bottom_label[i]), int(bottom_group[i]));
  }
  fclose(fout);
  FILE* fout2 = fopen("1-2", "w");
  for(int i = 0; i < bottom[0]->count(); i ++) {
    fprintf(fout2, "%f,", bottom_data[i]);
  }
  fclose(fout2);
#endif

  int pos_cc = 0;
  int all_cc = 0;
  
  Dtype map_ = Dtype(0);
  int effect_sample = 0;
  map<int, vector<std::pair<float, int> > >::iterator it;
  
  for(it = all_data.begin(); it != all_data.end(); it ++) {
    std::sort(it->second.begin(), it->second.end(), mycompare);
    Dtype ap = 0;
    int map_rank = 0;
    int neg_exist = 0;
    for(size_t i = 0; i < it->second.size(); i ++) {
      if(it->second[i].second == 1) {
        ap += (++ map_rank) / (Dtype)(i + 1);
      }
      else {
        if(neg_exist == 0) {
            neg_exist = 1;
        }
      }
      //printf("[%d, %f],", it->first, it->second[i].first);
    }
    if(map_rank < 1 || neg_exist == 0) {
      continue;
    }
    effect_sample ++;
    map_ += ap / map_rank;
    pos_cc += map_rank;
    all_cc += int(it->second.size());
  }
  LOG(INFO) << "effective[" << effect_sample << "] vs All[" << all_data.size() << "] " << pos_cc << "," << all_cc;
  top[0]->mutable_cpu_data()[0] = map_ / effect_sample;
}

INSTANTIATE_CLASS(MAPLayer);
REGISTER_LAYER_CLASS(MAP);

}  // namespace caffe
