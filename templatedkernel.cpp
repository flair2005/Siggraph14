////////////////////////////////////////////////////////////////////////////////
//
//  SYCL sample code Siggraph 2014 - The Unlicense (see LICENSE for details)
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "CL/sycl.hpp"
using namespace cl::sycl;

template<typename T>
class subtract_functor {
    T m_a, m_b;
public:
    subtract_functor(T a, T b) : m_a(a), m_b(b){ }
    T operator()() { return m_a - m_b; }
};

int main()
{
  float finalResultFloat = 0.0f;
  int finalResultInt = 0;
  { //all SYCL work in that block will completed before exiting it
    queue myQueue;
    //abstract underlying OpenCL data movement by using SYCL buffer
    buffer<float, 1> floatBuf(&finalResultFloat, 1);
    buffer<int, 1> intBuf(&finalResultInt, 1);
    command_group(myQueue, [&]()
    {
      auto intAcc = 0; intBuf.get_access<access::read_write>();
      //enqueue a single task based on functor
      single_task(kernel_functor(
      {
          floatAcc[0] = subtract_functor(floatAcc[0], 42.42f);
      }));
    });//end of commands for this queue

    command_group(myQueue, [&]()
    {
      auto intAcc = intBuf.get_access<access::read_write>();
      //enqueue a single task based on functor
      single_task(kernel_functor(
      {
          incAcc[0] = subtract_functor(intAcc[0], 42);
      }));
    });//end of commands for this queue
  } //end scope, so we wait for the queue to complete
    
  std::cout << "intResult	: " << intResult << std::endl;
  std::cout << "floatResult	: " << floatResult << std::endl;
  return 0;
}