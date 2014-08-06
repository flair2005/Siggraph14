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
  float result = 0.0f; //this is where we will write our result
  { // all SYCL work in a {} block will be completed before exiting the block
    cl::sycl::queue myQueue; // create a queue to work on
    cl::sycl::buffer<float, 1> 	resultBuf(&result, 1); //wrap variable in a buffer
    cl::sycl::command_group(myQueue, [&]() { // create 'command' for our 'queue'
      //request access to our buffer
      //cl::sycl::accessor<float, 1, access::read_write, access::global_buffer>
      auto resultAcc = resultBuf.get_access<access::read_write>();
      //enqueue a single, simple task
      cl::sycl::single_task(kernel_functor<class subtractKernel>([=](){
        resultAcc[0] = subtract(resultAcc[0], 42.0f);
      }));
    }); //end of out commands for this queue
  } //end scope, so we wait for the queue to complete
  std::cout << "result	: " << result << std::endl;
  return 0;
}