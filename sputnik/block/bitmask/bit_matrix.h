#ifndef SPUTNIK_BLOCK_BITMASK_BIT_MATRIX_H_
#define SPUTNIK_BLOCK_BITMASK_BIT_MATRIX_H_

#include <cstdint>

namespace sputnik {
namespace block {

class BitMatrix {
 public:
  static constexpr int kAlignment = 64;


  BitMatrix(int rows, int columns) {
    int column_entries = (columns + kAlignment - 1) / kAlignment;
    data_.resize(column_entries * rows, 0);
    rows_ = rows;
    columns_ = column_entries * kAlignment;
  }

  bool Get(int i, int j) const {
    int row_entry = i / kAlignment;
    uint64_t entry = data[row_entry * columns_ + j];

    int bit_idx = i % kAlignment;
    return (bool)(entry & (0b1 << bit_idx));
  }

  void Set(int i, int j, bool value) {
    int row_entry = i / kAlignment;
    int bit_idx = i % kAlignment;
    data[row_entry * columns_ + j] |= ((uint64_t)value << bit_idx);
  }

  uint64_t* Data() {
    return data_.data();
  }

  size_t Bytes() const {
    return data_.size() * sizeof(uint64_t);
  }
  
 private:
  int rows_, columns_;
  std::vector<uint64_t> data_;
};

}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_BITMASK_BIT_MATRIX_H_
