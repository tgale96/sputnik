#ifndef SPUTNIK_LOGGING_H_
#define SPUTNIK_LOGGING_H_

#include <sstream>

namespace sputnik {

const int INFO = 0;
const int WARNING = 1;
const int ERROR = 2;
const int FATAL = 3;

namespace internal {

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  
  ~LogMessage();

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

}  // namespace internal
}  // namespace sputnik


#define _SPUTNIK_LOG_INFO						\
  sputnik::internal::LogMessage(__FILE__, __LINE__, sputnik::INFO)
#define _SPUTNIK_LOG_WARNING						\
  sputnik::internal::LogMessage(__FILE__, __LINE__, sputnik::WARNING)
#define _SPUTNIK_LOG_ERROR						\
  sputnik::internal::LogMessage(__FILE__, __LINE__, sputnik::ERROR)
#define _SPUTNIK_LOG_FATAL						\
  sputnik::internal::LogMessage(__FILE__, __LINE__, sputnik::FATAL)
  
#define LOG(severity) _SPUTNIK_LOG_##severity

#define CHECK(condition)				\
  if (!(condition))					\
    LOG(FATAL) << "Check failed: " #condition " "

#define CHECK_EQ(a, b) CHECK(a == b)
#define CHECK_NE(a, b) CHECK(a != b)
#define CHECK_LE(a, b) CHECK(a <= b)
#define CHECK_LT(a, b) CHECK(a < b)
#define CHECK_GE(a, b) CHECK(a >= b)
#define CHECK_GT(a, b) CHECK(a > b)

#endif  // SPUTNIK_LOGGING_H_
