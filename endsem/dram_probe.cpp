// Measure completion of a contiguous weight read through Ramulator2.
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "base/base.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: dram_probe BYTES CHANNELS MAPPER OFFSET\n";
    return 2;
  }
  const auto bytes = std::stoull(argv[1]);
  const auto channels = std::stoul(argv[2]);
  const std::string mapper = argv[3];
  const auto offset = std::stoull(argv[4]);
  constexpr std::uint64_t burst_bytes = 64;
  if (!bytes || bytes % burst_bytes || offset % burst_bytes ||
      (channels != 1 && channels != 2 && channels != 4 && channels != 8) ||
      (mapper != "RoBaRaCoCh" && mapper != "ChRaBaRoCo")) {
    throw std::invalid_argument("invalid probe parameters");
  }
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
)");
  config["MemorySystem"]["DRAM"]["org"]["channel"] = channels;
  config["MemorySystem"]["AddrMapper"]["impl"] = mapper;
  auto* frontend = Ramulator::Factory::create_frontend(config);
  auto* memory = Ramulator::Factory::create_memory_system(config);
  frontend->connect_memory_system(memory);
  memory->connect_frontend(frontend);

  const auto requests = bytes / burst_bytes;
  std::uint64_t issued = 0, completed = 0, cycles = 0, first = 0;
  auto done = [&](Ramulator::Request&) {
    if (!completed) first = cycles + 1;
    ++completed;
  };
  while (completed < requests) {
    // Provide sufficient source request bandwidth to exercise all channels.
    for (unsigned lane = 0; lane < channels * 2 && issued < requests; ++lane) {
      if (!frontend->receive_external_requests(Ramulator::Request::Type::Read,
              offset + issued * burst_bytes, 0, done)) break;
      ++issued;
    }
    memory->tick();
    if (++cycles > requests * 128 + 10000) {
      throw std::runtime_error("read completions timed out");
    }
  }
  std::cout << std::setprecision(12) << "DRAM_PROBE," << bytes << ',' << channels
            << ',' << mapper << ',' << offset << ',' << issued << ',' << completed
            << ',' << cycles << ',' << memory->get_tCK() << ',' << first << '\n';
  memory->finalize();
}
