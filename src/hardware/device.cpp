#include "hardware/device.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <random>
#include <string>
#include <stdexcept>

#include "common/assert.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "dram/mmap_controller.h"
#include "dram/pimkernel/pim_kernel.h"
#include "module/module_graph.h"
#include "module/tensor.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

namespace llm_system {

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(0, 5);

Device::Device(SystemConfig config, int device_total_rank, Cluster_ptr cluster)
    : config(config),
      device_total_rank(device_total_rank),
      cluster(cluster),
      status() {
  compute_peak_flops = config.compute_peak_flops;
  memory_bandwidth = config.memory_bandwidth;
  memory_capacity = config.memory_capacity;
  device_local_rank = device_total_rank % config.num_device;

  top_module_graph = TopModuleGraph::Create(status);

  std::string dram_cfg_path;
  if(config.gpu_gen == "H100"){
    dram_cfg_path = "./dram_config_HBM3_80GB.yaml";
  }
  else if(config.gpu_gen == "B100" || config.gpu_gen == "B200"){
    dram_cfg_path = "./dram_config_HBM3E_192GB.yaml";
  }
  YAML::Node cfg = YAML::LoadFile(dram_cfg_path);

  double memory_scale_factor = 0;
  MemoryConfig memory_config = MemoryConfig(config.num_cube, config.num_logic_cube);
  if(config.gpu_gen == "H100"){
    // H100, HBM3 80GB, 5.2Gbps
    memory_scale_factor = 0.76923;
    memory_config = hbm3_80GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }
  else if((config.gpu_gen == "B100") || (config.gpu_gen == "B200")){
    // B100, HBM3E 192GB, 8Gbps
    memory_scale_factor = 0.5; // 8.0Gbps pin rate's ideal bandwidth = 8000, tCK = 2000, 1 / 2GHz = 0.5
    memory_config = hbm3e_192GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }

  dram_interface = DRAMInterface::Create(dram_cfg_path, memory_scale_factor);
  mmap_controller = MMapController::Create(memory_config);
  use_ramulator = config.use_ramulator;
  perform_execution = false;
}

void Device::set_dependency() { top_module_graph->set_dependency(); }

bool Device::check_module_graph_remain() {
  return top_module_graph->check_module_graph_remain();
};

void Device::run(std::vector<BatchedSequence::Ptr> sequences_metadata_list) {
  int dp_rank = device_total_rank / model_config.ne_tp_dg;
  top_module_graph->run(sequences_metadata_list.at(dp_rank));
}

void Device::restartGraph() { top_module_graph->restart_graph(); }

void Device::connectTopModuleGraph() {
  top_module_graph->connectDevice(get_ptr());
}

void Device::reset_status() {
  status = StatusBoard();
  previous_layer_.clear();
  pending_expert_.reset();
}
void Device::reset_timeboard() { top_module_graph->reset_timeboard(); }

void Device::add_module(std::string name, Module_ptr module) {
  cluster->add_module(device_total_rank, name, module);
}

void Device::setMemoryObject(Tensor::Ptr tensor) {
  mmap_controller->setMemoryObject(tensor);
}

void Device::addExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

void Device::addExecutionCache(ExecStatus& exec_status, LayerType layer_type,
                               ProcessorType processor_type,
                               DRAMRequestType dram_reqeust_type, long size) {
  CacheKey key =
      std::make_tuple(layer_type, processor_type, dram_reqeust_type, size);
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

bool Device::checkExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    exec_status = (*cache_iter).second;
    return true;
  } else {
    return false;
  }
}

bool Device::checkExecutionCache(CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    ExecStatus exec_status = (*cache_iter).second;
    ExecStatus& _status = dram_interface->getExecStatus();
    _status = exec_status;
    return true;
  } else {
    return false;
  }
}

void Device::setExecStatus(ExecStatus& exec_status_) {
  exec_status = exec_status_;
}

// Persistent off-chip controller. Completion callbacks, rather than admitted
// requests, determine service time. Each expert is sampled in 64-byte bursts.
class ExpertDram {
 public:
  explicit ExpertDram(int channels) : channels_(channels) {
    auto config = YAML::Load(R"(
Frontend: {impl: GEM5, clock_ratio: 1}
MemorySystem:
  impl: GenericDRAM
  clock_ratio: 1
  DRAM:
    impl: DDR4
    org: {preset: DDR4_8Gb_x8, rank: 2}
    timing: {preset: DDR4_2400R}
  Controller:
    impl: Generic
    Scheduler: {impl: FRFCFS}
    RefreshManager: {impl: AllBank}
  AddrMapper: {impl: RoBaRaCoCh}
)");
    config["MemorySystem"]["DRAM"]["org"]["channel"] = channels;
    frontend_ = Ramulator::Factory::create_frontend(config);
    memory_ = Ramulator::Factory::create_memory_system(config);
    frontend_->connect_memory_system(memory_);
    memory_->connect_frontend(frontend_);
  }

  time_ns read(const std::string& key, long long bytes, int sample_ratio, long long offset) {
    constexpr long long burst = 64;
    const auto requests = (bytes + burst * sample_ratio - 1) / (burst * sample_ratio);
    // Hash identities to aligned offsets in the sampled address space.
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : key) hash = (hash ^ byte) * 1099511628211ULL;
    const auto base = (hash % 2048) * (64ULL << 20) + offset / sample_ratio;
    long long issued = 0, completed = 0, cycles = 0;
    auto callback = [&](Ramulator::Request&) { ++completed; };
    while (completed < requests) {
      for (int lane = 0; lane < 2 * channels_ && issued < requests; ++lane) {
        if (!frontend_->receive_external_requests(
                Ramulator::Request::Type::Read, base + issued * burst, 0, callback)) break;
        ++issued;
      }
      memory_->tick();
      if (++cycles > 10000 + requests * 128) {
        throw std::runtime_error("Off-chip DRAM read completion timed out");
      }
    }
    std::cout << "EXPERT_DRAM," << key << ',' << bytes << ',' << issued << ','
              << completed << ',' << cycles << ',' << memory_->get_tCK() << '\n';
    return cycles * static_cast<double>(memory_->get_tCK()) * bytes / (requests * burst);
  }

 private:
  int channels_;
  Ramulator::IFrontEnd* frontend_;
  Ramulator::IMemorySystem* memory_;
};

time_ns Device::expertTransfer(const std::string& key, long long bytes, long long offset) {
  const auto link = bytes * 1e9 / config.offchip_bandwidth_bytes_per_s;
  if (!config.offchip_ramulator_channels) return config.offchip_latency_ns + link;
  if (!expert_dram_) {
    expert_dram_ = std::make_shared<ExpertDram>(config.offchip_ramulator_channels);
  }
  // Use enough bursts for short chunks so the first-read startup is not
  // multiplied by the same factor as a full expert-sized transfer.
  constexpr long long min_bursts = 128;
  const auto sample_ratio = static_cast<int>(std::min<long long>(
      config.offchip_sample_ratio, std::max(1LL, bytes / (64 * min_bursts))));
  const auto dram = expert_dram_->read(key, bytes, sample_ratio, offset);
  return config.offchip_latency_ns + std::max(dram, link);
}

void Device::cacheExpert(const std::string& key, long long bytes, ExpertFetch& result) {
  // Reserve one fixed expert slot when prefetching is enabled.
  const auto capacity = config.expert_cache_capacity_bytes -
                        (config.expert_prefetch ? bytes : 0);
  if (bytes > capacity) return;
  while (expert_cache_used_bytes_ + bytes > capacity) {
    auto oldest = std::prev(expert_lru_.end());
    expert_cache_used_bytes_ -= oldest->second;
    expert_index_.erase(oldest->first);
    expert_lru_.erase(oldest);
    ++result.evictions;
  }
  expert_lru_.emplace_front(key, bytes);
  expert_index_[key] = expert_lru_.begin();
  expert_cache_used_bytes_ += bytes;
}

Device::ExpertFetch Device::fetchExpert(const std::string& key,
                                       long long bytes, time_ns queued_ns) {
  ExpertFetch result;
  if (!config.offchip_experts || bytes <= 0) return result;
  auto found = expert_index_.find(key);
  if (found != expert_index_.end()) {
    result.hit = true;
    expert_lru_.splice(expert_lru_.begin(), expert_lru_, found->second);
    return result;
  }

  const auto now = get_time() + queued_ns;
  if (pending_expert_) {
    result.load_duration_ns = std::max<time_ns>(0, pending_expert_->ready_ns - now);
    if (key == pending_expert_->key) {
      const auto staged = pending_expert_->bytes;
      if (staged > bytes) throw std::logic_error("Prefetch weight size mismatch");
      std::cout << "EXPERT_PREFETCH_USED," << key << ',' << result.load_duration_ns << '\n';
      pending_expert_.reset();
      if (bytes > staged) {
        result.bytes_transferred = bytes - staged;
        result.load_duration_ns += expertTransfer(key, bytes - staged, staged);
      }
      cacheExpert(key, bytes, result);
      return result;
    }
  }
  result.bytes_transferred = bytes;
  result.load_duration_ns += expertTransfer(key, bytes);
  cacheExpert(key, bytes, result);
  return result;
}

void Device::routeExperts(const std::string& layer, const std::vector<int>& routed) {
  if (!previous_layer_.empty()) next_layer_[previous_layer_] = layer;
  previous_layer_ = layer;
  previous_routes_[layer] = routed;
  if (pending_expert_ && !pending_expert_->key.empty()) {
    const bool needed = std::any_of(routed.begin(), routed.end(), [&](int expert) {
      return pending_expert_->key == layer + ":" + std::to_string(expert);
    });
    if (!needed) {
      // Keep an unused transfer occupying the link until its completion.
      std::cout << "EXPERT_PREFETCH_UNUSED," << pending_expert_->key << '\n';
      pending_expert_->key.clear();
    }
  }
}

static const std::vector<std::vector<int>>& prefetchPlan() {
  static const auto plan = [] {
    std::vector<std::vector<int>> rows;
    if (const char* path = std::getenv("MOE_PREFETCH_PLAN")) {
      if (!std::getenv("MOE_INPUT_ROUTES")) throw std::runtime_error("Plan requires recorded routes");
      std::ifstream file(path);
      if (!file) throw std::runtime_error("Could not open MOE_PREFETCH_PLAN");
      std::string line;
      while (std::getline(file, line)) {
        std::istringstream input(line);
        std::vector<int> experts;
        int expert;
        while (input >> expert) {
          if (expert < 0 || expert >= 256) throw std::runtime_error("Invalid predicted expert");
          experts.push_back(expert);
        }
        if (!input.eof()) throw std::runtime_error("Invalid prefetch plan row");
        rows.push_back(std::move(experts));
      }
      if (rows.empty()) throw std::runtime_error("Empty prefetch plan");
    }
    return rows;
  }();
  return plan;
}

void Device::prefetchSharedExpert(const std::string& key, long long bytes,
                                  time_ns queued_ns) {
  if (expert_index_.count(key) || bytes > config.expert_cache_capacity_bytes) return;
  const auto now = get_time() + queued_ns;
  if (pending_expert_ && pending_expert_->ready_ns > now) return;
  const auto service = expertTransfer(key, bytes);
  pending_expert_ = PendingExpert{key, now + service, bytes};
  std::cout << "EXPERT_PREFETCH," << key << ',' << bytes << ',' << now
            << ',' << service << '\n';
}

void Device::prefetchExpert(const std::string& layer, long long bytes, time_ns queued_ns) {
  const auto next = next_layer_.find(layer);
  const auto& plan = prefetchPlan();
  const std::vector<int>* planned = plan.empty() ? nullptr : &plan.at(prefetch_index_++);
  if (next == next_layer_.end() || bytes > config.expert_cache_capacity_bytes) return;
  const auto now = get_time() + queued_ns;
  if (pending_expert_ && pending_expert_->ready_ns > now) return;
  pending_expert_.reset();
  const auto& candidates = planned ? *planned : previous_routes_[next->second];
  for (int expert : candidates) {
    const auto key = next->second + ":" + std::to_string(expert);
    if (expert_index_.count(key)) continue;
    const auto chunk = config.expert_prefetch_chunk_bytes;
    const auto staged = chunk > 0 ? std::min(chunk, bytes) : bytes;
    const auto service = expertTransfer(key, staged);
    pending_expert_ = PendingExpert{key, now + service, staged};
    std::cout << "EXPERT_PREFETCH," << key << ',' << staged << ',' << now
              << ',' << service << '\n';
    break;
  }
}

ExecStatus Device::getHighExecStatus() {
  ExecStatus return_status = high_exec_status;
  high_exec_status = ExecStatus();
  return return_status;
}

ExecStatus Device::getLowExecStatus() {
  ExecStatus return_status = low_exec_status;
  low_exec_status = ExecStatus();
  return return_status;
}

ExecStatus Device::getExecStatus() {
  ExecStatus return_status = exec_status;
  exec_status = ExecStatus();
  return return_status;
}

// check whether execution must be performed, and ramulator
void Device::execution(LayerType layer_type,
                       const std::vector<Tensor::Ptr>& tensor_list,
                       const BatchedSequence::Ptr sequences_metadata,
                       const LayerInfo layer_info) {
  if (perform_execution) {
    dram_interface->resetCounter();
    cluster->executor.execution(layer_type, tensor_list, sequences_metadata,
                                config.processor_type, layer_info,
                                use_ramulator, get_ptr());
  }
}

void Device::execution_ramulator(LayerType layer_type,
                                 std::vector<Tensor::Ptr> tensor_list) {
  Tensor::Ptr input = tensor_list.at(0);
  Tensor::Ptr weight = tensor_list.at(1);
  Tensor::Ptr output = tensor_list.at(2);

  std::cout << weight->name << std::endl;
}

void Device::run_ramulator(DRAMRequest_Ptr dram_request) {
  std::list<DRAMRequest::Ptr> request;
  request.push_back(dram_request);
  dram_interface->HandleRequest(request, 0);
}

void Device::run_ideal(DRAMRequestType dram_request_type, Tensor_Ptr tensor){
  long total_size = tensor->getSize(); // Byte
  if (total_size == 0) {
    return;
  }
  MemoryConfig memory_config = mmap_controller->getConfig();
  int num_cube = memory_config.num_cube;
  int num_channel = memory_config.num_channel; // 32 (not legacy, pCH)
  int num_col = memory_config.num_col;
  int granul = mmap_controller->getGranul();

  long total_read = total_size / granul;
  long rw_cmd_to_cube_0 = (total_read % num_cube == 0) ? (total_read / num_cube) : ((total_read / num_cube) + 1);

  long rw_cmd_to_pCH_0 = (rw_cmd_to_cube_0 % num_channel == 0) ? (rw_cmd_to_cube_0 / num_channel) : ((rw_cmd_to_cube_0 / num_channel) + 1);
  long rw_cmd_to_pCH_1 = (rw_cmd_to_cube_0 % num_channel == 1) ? (rw_cmd_to_cube_0 / num_channel) : ((rw_cmd_to_cube_0 / num_channel) + 1);

  dram_interface->resetCounter();
  dram_interface->getExecStatus().act_count = (((rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1) < num_col) ? 1 : ((rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1) / num_col));
  if(dram_request_type == DRAMRequestType::kRead){
    dram_interface->getExecStatus().read_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
  else if(dram_request_type == DRAMRequestType::kWrite){
    dram_interface->getExecStatus().write_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
}

void Device::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  int num_pseudo_ch = 0;
  if (ProcessorType == (int)(ProcessorType::GPU)) {
    num_pseudo_ch = mmap_controller->getConfig().num_cube * mmap_controller->getConfig().num_channel / 2;
  }
  else {
    num_pseudo_ch = mmap_controller->getConfig().num_cube * mmap_controller->getConfig().num_channel;
  }
  dramEnergy.kACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kREAD_energy_j_ *= num_pseudo_ch;
  dramEnergy.kWRITE_energy_j_ *= num_pseudo_ch;

  dramEnergy.kALL_ACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_READ_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_WRITE_energy_j_ *= num_pseudo_ch;

  top_module_graph->initializeDRAM(ProcessorType, dramEnergy);
}

};  // namespace llm_system
