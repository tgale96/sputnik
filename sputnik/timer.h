#ifndef SPUTNIK_TIMER_H_
#define SPUTNIK_TIMER_H_

#include "sputnik/cuda_utils.h"
#include "sputnik/test_utils.h"

namespace sputnik {
  
struct Timer {

  cudaEvent_t events[2];

  Timer();
  ~Timer();

  void start(cudaStream_t stream);

  void stop(cudaStream_t stream);

  double duration(int iterations = 1) const;
};

}  // namespaces sputnik

#endif  // SPUTNIK_TIMER_H_
