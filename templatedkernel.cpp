////////////////////////////////////////////////////////////////////////////////
//
//  SYCL sample code Siggraph 2014 - The Unlicense (see LICENSE for details)
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "CL/sycl.hpp"
using namespace cl::sycl;

// templated kernel
template <typename T>
class subtract_functor {
  accessor<T, 1, access::read_write, access::global_buffer> m_acc;
  T m_value;

 public:
  subtract_functor(
      accessor<T, 1, access::read_write, access::global_buffer> acc, T value)
      : m_acc(acc), m_value(value) {}

  T subtract(T a, T b) {
    return a - b;
  }  // templated subtract. Although it could be anything!

  void operator()() {
    m_acc[0] = subtract(m_acc[0], m_value);  // function operator
  }
};

int main() {
  int intVal = 0;
  float floatVal = 0.0f;

  {  // This block defines the scope where the SYCL objects will live.
    default_selector selector;  // Default selector
    queue myQueue(selector);

    buffer<int, 1> intBuf(&intVal, 1);  // Buffers
    buffer<float, 1> floatBuf(&floatVal, 1);

    command_group(myQueue, [&]() {  // Kernel and it's dependencies for int
      auto intAcc = intBuf.get_access<access::read_write>();
      single_task(kernel_functor(subtract_functor<int>(intAcc, 42)));
    });

    command_group(myQueue, [&]() {  // Kernel and it's dependencies for float
      auto floatAcc = floatBuf.get_access<access::read_write>();
      single_task(kernel_functor(subtract_functor<float>(floatAcc, 42.4242f)));
    });
  }

  std::cout << "intVal	: " << intVal << std::endl;
  std::cout << "floatVal	: " << floatVal << std::endl;
  return 0;
}
