////////////////////////////////////////////////////////////////////////////////
//
//  SYCL sample code Siggraph 2014 - The Unlicense (see LICENSE for details)
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "CL/sycl.hpp"
using namespace cl::sycl;

//templated kernel
template<typename T>
class subtract_functor {
    accessor<T, 1, access::read_write, access::global_buffer> m_acc;
    T m_value;
public:
    subtract_functor(accessor<T, 1, access::read_write, access::global_buffer> acc, T value)
    : m_acc(acc), m_value(value){ }
    
    T subtract(T a, T b) { return a - b;}  //templated subtract. Although it could be anything!
    
    void operator()() {
		m_acc[0] = subtract(m_acc[0], m_value); //function operator
    }
};

int main()
{
  int intResult = 0; //this is where we will write our result
  float floatResult = 0.0f;
  {   // all SYCL work in a {} block will be completed before exiting the block
    queue myQueue;
    buffer<int, 1> 		intBuf(&intResult, 1); // Buffers
    buffer<float, 1> 	floatBuf(&floatResult, 1);
    command_group(myQueue, [&]() {  //enqueue a single, simple task for int
      auto intAcc 	= intBuf.get_access<access::read_write>();
      single_task(kernel_functor(subtract_functor<int>(intAcc, 42)));
    }); //end of out commands for this queue
        
    command_group(myQueue, [&]() {  //enqueue a single, simple task for float
      auto floatAcc 	= floatBuf.get_access<access::read_write>();
      single_task(kernel_functor(subtract_functor<float>(floatAcc, 42.4242f)));
    }); //end of out commands for this queue
  } //end scope, so we wait for the queue to complete
    
  std::cout << "intResult	: " << intResult << std::endl;
  std::cout << "floatResult	: " << floatResult << std::endl;
  return 0;
}