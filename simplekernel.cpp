////////////////////////////////////////////////////////////////////////////////
//
//  SYCL sample code Siggraph 2014 - The Unlicense (see LICENSE for details)
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "CL/sycl.hpp"

float subtract(float a, float b) { return a - b; }

int main()
{
  float finalResultFloat = 0.0f;
  { //all SYCL work in that block will completed before exiting it
    cl::sycl::queue myQueue;
    //abstract underlying OpenCL data movement by using SYCL buffer
    cl::sycl::buffer<float, 1> 	floatBuf(&finalResultFloat, 1);
    cl::sycl::command_group(myQueue, [&]()
    {
      auto floatAcc = floatBuf.get_access<access::read_write>();
      //enqueue a single task based on lambda
      cl::sycl::single_task(kernel_functor<class subtractKernel>([=]()
      {
        floatAcc[0] = subtract(floatAcc[0], 42.0f);
      }));
    }); //end of commands for this queue
  } //end scope, so we wait for the queue to complete
  std::cout << "finalResultFloat: " << finalResultFloat << std::endl;
  return 0;
}