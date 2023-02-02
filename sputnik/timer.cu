#include "sputnik/timer.h"

namespace sputnik {

Timer::Timer() {
  for (auto &event : events) {
    CUDA_CALL(cudaEventCreate(&event));
  }
}

Timer::~Timer() {
  for (auto &event : events) {
    CUDA_CALL(cudaEventDestroy(event));
  }
}

void Timer::start(cudaStream_t stream) {
  CUDA_CALL(cudaEventRecord(events[0], stream));
}

/// Records a stop event in the stream and synchronizes on the stream
void Timer::stop(cudaStream_t stream) {
  CUDA_CALL(cudaEventRecord(events[1], stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
}

/// Returns the duration in miliseconds
double Timer::duration(int iterations) const {
  float avg_ms;
  CUDA_CALL(cudaEventElapsedTime(&avg_ms, events[0], events[1]));
  return (double)avg_ms / iterations;
}

}  // namespace sputnik
