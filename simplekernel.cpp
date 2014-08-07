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
  float finalResult = 0.0f;
  { // all SYCL work in that block will completed before exiting it
    cl::sycl::queue myQueue;
    //abstract underlying OpenCL data movement by using SYCL buffer
    cl::sycl::buffer<float, 1> 	resultBuf(&finalResult, 1);
    cl::sycl::command_group(myQueue, [&]()
    {
      auto resultAcc = resultBuf.get_access<access::read_write>();
      //enqueue a single task
      cl::sycl::single_task(kernel_functor<class subtractKernel>([=]()
      {
        //c++11 lambda kernel
        resultAcc[0] = subtract(resultAcc[0], 42.0f);
      }));
    }); //end of commands for this queue
  } //end scope, so we wait for the queue to complete
  std::cout << "result	: " << result << std::endl;
  return 0;
}