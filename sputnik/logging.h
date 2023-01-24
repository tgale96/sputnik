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
  
#define SPUTNIK_LOG(severity) _SPUTNIK_LOG_##severity

#define SPUTNIK_CHECK(condition)			\
  if (!(condition))					\
    SPUTNIK_LOG(FATAL) << "Check failed: " #condition " "

#define SPUTNIK_CHECK_EQ(a, b) SPUTNIK_CHECK(a == b)
#define SPUTNIK_CHECK_NE(a, b) SPUTNIK_CHECK(a != b)
#define SPUTNIK_CHECK_LE(a, b) SPUTNIK_CHECK(a <= b)
#define SPUTNIK_CHECK_LT(a, b) SPUTNIK_CHECK(a < b)
#define SPUTNIK_CHECK_GE(a, b) SPUTNIK_CHECK(a >= b)
#define SPUTNIK_CHECK_GT(a, b) SPUTNIK_CHECK(a > b)

#endif  // SPUTNIK_LOGGING_H_
