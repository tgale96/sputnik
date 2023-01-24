#include "sputnik/logging.h"

#include <cstdlib>

namespace sputnik::internal {

LogMessage::LogMessage(const char* fname, int line, int severity) :
  fname_(fname), line_(line), severity_(severity) {}
  
LogMessage::~LogMessage() {
  GenerateLogMessage();
}

void LogMessage::GenerateLogMessage() {
  fprintf(stderr, "%c %s:%d] %s\n", "IWEF"[severity_], fname_, line_,
	  str().c_str());
  if (severity_ == FATAL) abort();
}

}  // namespace sputnik::internal
