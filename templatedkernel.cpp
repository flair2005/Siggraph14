////////////////////////////////////////////////////////////////////////////////
//
//  SYCL sample code Siggraph 2014 - The Unlicense (see LICENSE for details)
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "CL/sycl.hpp"
using namespace cl::sycl;

template <typename T>
class subtract_functor {
  accessor<T, 1, access::read_write, access::global_buffer> m_acc;
  T m_value;
 public:
  subtract_functor(
      accessor<T, 1, access::read_write, access::global_buffer> acc, T value)
      : m_acc(acc), m_value(value) {}
  T subtract(T a, T b) { return a - b; }
  void operator()() {
    cl::sycl::parallel_for(range<1>(m_acc.size()),
    cl::sycl::kernel_functor<class kernel_functor>([=] (cl::sycl::id<1> i)
    {
        m_acc[i] = subtract(m_acc[i], m_value);
    }));
  }
};

int main() {
  float finalResultFloat[] = {0.0f, 1.0f, 2.0f, 3.0f};
  int finalResultInt[] = {0, 1, 2, 3};
  { // all SYCL work in that block will completed before exiting it
    queue myQueue;
    buffer<float, 1> floatBuf(finalResultFloat, 4);
    buffer<int, 1> intBuf(finalResultInt, 4);
    command_group(myQueue, [&]()
    {
      auto floatAcc = floatBuf.get_access<access::read_write>();
      // enqueue a parallel_for based on functor
      subtract_functor<float>(floatAcc, 42.42f);
    }); // end of commands for this queue
    command_group(myQueue, [&]()
    {
      auto intAcc = intBuf.get_access<access::read_write>();
      // enqueue a single task based on functor
      subtract_functor<int>(intAcc, 42);
    }); // end of commands for this queue
  } // end scope, so we wait for the queue to complete
  std::cout << "finalResultInt: " << finalResultInt[0] << std::endl;
  std::cout << "finalResultFloat: " << finalResultFloat[0] << std::endl;
  return 0;
}
