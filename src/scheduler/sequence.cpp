#include "scheduler/sequence.h"

#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>

#include "scheduler/scheduler.h"

namespace llm_system {

static std::ofstream* routeTrace() {
  static std::unique_ptr<std::ofstream> trace = [] {
    const char* path = std::getenv("MOE_TRACE_PATH");
    if (!path || !*path) return std::unique_ptr<std::ofstream>{};
    auto file = std::make_unique<std::ofstream>(path);
    if (*file) {
      *file << "sequence_id,token_position,layer,expert_id,source\n";
    }
    return file;
  }();
  return trace && *trace ? trace.get() : nullptr;
}
// Recorded decode routes, flattened in token/layer order. Fail on exhaustion.
static const std::vector<std::vector<int>>& recordedRoutes() {
  static const auto routes = [] {
    std::vector<std::vector<int>> rows;
    if (const char* path = std::getenv("MOE_INPUT_ROUTES")) {
      std::ifstream file(path);
      if (!file) throw std::runtime_error("Could not open MOE_INPUT_ROUTES");
      std::string line;
      while (std::getline(file, line)) {
        std::istringstream input(line);
        std::vector<int> experts;
        int expert;
        while (input >> expert) experts.push_back(expert);
        if (!input.eof() || experts.empty()) throw std::runtime_error("Invalid route row");
        rows.push_back(std::move(experts));
      }
      if (rows.empty()) throw std::runtime_error("Empty recorded route trace");
    }
    return rows;
  }();
  return routes;
}

Sequence::Sequence(int expert_seq_id, int input_len, int output_len)
    : expert_seq_id(expert_seq_id),
      input_len(input_len),
      output_len(output_len) {
  id = count_id;
  count_id++;
  num_process_token = 0;
  current_len = 0;
  total_len = input_len + output_len - 1;
  get_expert_from_list = true;

  gen_start_time = 0;

  assertTrue(input_len > 0, "input len is not valid, input len is " +
                                std::to_string(input_len));
  assertTrue(output_len > 0, "output len is not valid, output len is " +
                                 std::to_string(output_len));

  arrival_time = 0;
  first_token_time = 0;
  end_token_time = 0;
  queueing_delay = 0;
  num_sum_iter = 0;
  sum_stage = true;
  record = false;
}

void Sequence::setGenStartTime(time_ns time) { gen_start_time = time; }

void Sequence::update(time_ns time) {
  // update end_token_time
  end_token_time += time;

  if (sum_stage) {
    first_token_time += time;
    num_sum_iter++;
  }

  // update status
  current_len += num_process_token;
  num_process_token = 0;
  if (sum_stage && current_len == input_len) {
    sum_stage = false;
  }
}

void BatchedSequence::add(Sequence::Ptr seq) {
  sequence.push_back(seq);
  return;
}

std::vector<Sequence::Ptr> BatchedSequence::get_seq() { return sequence; }

std::vector<Sequence::Ptr> BatchedSequence::get_sum() {
  std::vector<Sequence::Ptr> sum;
  sum.resize(0);
  for (auto& seq : sequence) {
    if (seq->current_len < seq->input_len) {
      sum.push_back(seq);
    }
  }
  return sum;
}

std::vector<Sequence::Ptr> BatchedSequence::get_gen() {
  std::vector<Sequence::Ptr> gen;
  for (auto& seq : sequence) {
    if (seq->current_len >= seq->input_len) {
      gen.push_back(seq);
    }
  }
  return gen;
}

void BatchedSequence::update(time_ns time) {
  for (auto& seq : sequence) {
    seq->update(time);
  }
  // cur += 32; // cur += num_layer임
}

void BatchedSequence::pop(Sequence::Ptr seq) {
  for (auto target = sequence.begin(); target != sequence.end();) {
    if ((*target)->id == seq->id) {
      target = sequence.erase(target);
    } else {
      target++;
    }
  }
}

int BatchedSequence::get_num_seq() { return sequence.size(); }

int BatchedSequence::get_process_token() {
  int token = 0;

  for (auto& seq : sequence) {
    token += seq->num_process_token;
  }
  return token;
}

int BatchedSequence::get_gen_process_token() {
  int token = 0;
  std::vector<Sequence::Ptr> gen_seq = get_gen();

  for (auto& seq : gen_seq) {
    token += seq->num_process_token;
  }
  return token;
}

int BatchedSequence::get_sum_process_token() {
  int token = 0;
  std::vector<Sequence::Ptr> sum_seq = get_sum();

  for (auto& seq : sum_seq) {
    token += seq->num_process_token;
  }
  return token;
}

int BatchedSequence::get_average_sequence_length() {
  std::vector<Sequence::Ptr> gen_seq = get_gen();
  int total_length = 0;
  for (auto seq : gen_seq) {
    total_length += seq->current_len;
  }
  if (gen_seq.size()) {
    total_length /= gen_seq.size();
  } else {
    total_length = 0;
  }
  return total_length;
}

int BatchedSequence::get_total_sequence_length() {
  std::vector<Sequence::Ptr> gen_seq = get_gen();
  int total_length = 0;
  for (auto seq : gen_seq) {
    total_length += seq->current_len;
  }

  return total_length;
}

void BatchedSequence::update_expert(int num_expert, int top_k,
                                    bool need_new_expert) {
  num_token_in_expert.clear();
  int size = num_token_in_expert.size();
  const auto& model = scheduler->model_config;
  int num_layer = model.hasDensePrefix()
                      ? model.num_layers - model.first_k_dense
                      : model.num_layers / model.expert_freq;
  const int actual_layer = model.hasDensePrefix()
                               ? model.first_k_dense + cur_layer
                               : cur_layer * model.expert_freq;
  std::ofstream* trace = routeTrace();

  static int expert_id = 0;

  // init
  if (size == 0) { 
    for (int i = 0; i < num_expert; i++) {
      num_token_in_expert.push_back(0);
    }
  }
  clear_expert();
  std::vector<double> skewness_weight(num_expert); // w ~ 1/k^s
  for (int i = 0 ; i < num_expert; i++){
    skewness_weight[i] = 1.0 / pow((i + 1), scheduler->model_config.skewness);
  }

  for (auto seq : sequence) {
    if (seq->num_process_token == 0) {
      continue;
    }
    int seq_id = seq->expert_seq_id;
    int current_len = seq->current_len;

    for (int token_id = current_len;
         token_id < current_len + seq->num_process_token; token_id++) {
      const auto& recorded = recordedRoutes();
      if (!recorded.empty()) {
        if (sequence.size() != 1 || token_id < model.input_len) {
          throw std::runtime_error("Recorded routes require one decode sequence");
        }
        static const int first_token = token_id;
        if (token_id < first_token) throw std::runtime_error("Recorded token position went backwards");
        const auto& experts = recorded.at((token_id - first_token) * num_layer + cur_layer);
        if (experts.size() != top_k || std::set<int>(experts.begin(), experts.end()).size() != experts.size()) {
          throw std::runtime_error("Invalid recorded top-k");
        }
        for (int expert : experts) {
          if (expert < 0 || expert >= num_expert) throw std::runtime_error("Recorded expert out of range");
          ++num_token_in_expert.at(expert);
          if (trace) *trace << seq->id << ',' << token_id << ',' << actual_layer
                            << ',' << expert << ",recorded\n";
        }
      } else if (seq->get_expert_from_list) {
        for (int k = 0; k < top_k; k++) {
          int e_id = 0;
          e_id = scheduler->sequences_info[seq_id]
                     ->expert_list[token_id * num_layer + cur_layer][k];
          num_token_in_expert[e_id] += 1;
          if (trace) *trace << seq->id << ',' << token_id << ',' << actual_layer
                            << ',' << e_id << ",trace\n";
        }
      } else {
        std::set<int> expert_list;
        if(scheduler->model_config.skewness > 0){
          expert_list = scheduler->getZipfianRandomExpert(skewness_weight, top_k); // for skewed expert test
        }
        else{
          expert_list = scheduler->getRandomExpert(top_k);
        }
        for (auto idx : expert_list) {
          num_token_in_expert[idx] += 1;
          if (trace) *trace << seq->id << ',' << token_id << ',' << actual_layer
                            << ',' << idx << ",synthetic\n";
        }
      }
    }
  }

  cur_layer++;
  cur_layer %= num_layer;
}

void BatchedSequence::clear_expert() {
  int size = num_token_in_expert.size();
  for (int i = 0; i < size; i++) {
    num_token_in_expert.at(i) = 0;
  }
}

void BatchedSequence::add_sequence(std::vector<int> expert_seq_ids) {
  for (int seq_id : expert_seq_ids) {
    int Lin = scheduler->sequences_info.at(seq_id)->Lin;
    int Lout = scheduler->sequences_info.at(seq_id)->Lout;
    Sequence::Ptr seq = Sequence::Create(seq_id, Lin, Lout);
    add(seq);
  }
}

void BatchedSequence::add_dummy_sequence(int num_seq, int input_len,
                                         int output_len) {
  for (int i = 0; i < num_seq; i++) {
    Sequence::Ptr seq = Sequence::Create(0, input_len, output_len);
    seq->get_expert_from_list = false;
    add(seq);
  }
}

}  // namespace llm_system
