
#include <iostream>
#include "CL/sycl.hpp"

float subtract(float a, float b) { return a - b; }

int main()
{
  float finalResultFloat[] = {0.0f, 1.0f, 2.0f, 3.0f};
  { // all SYCL work in that block will completed before exiting it
    cl::sycl::queue myQueue;
    // abstract underlying OpenCL data movement by using SYCL buffer
    cl::sycl::buffer<float, 1> 	floatBuf(finalResultFloat, 4 );
    cl::sycl::command_group(myQueue, [&]()
    {
      auto floatAcc = floatBuf.get_access<cl::sycl::access::read_write>();
      // enqueue a parallel_for based on lambda
      cl::sycl::parallel_for(cl::sycl::nd_range<1>(cl::sycl::range<1>(4)),
      cl::sycl::kernel_functor<class subtractKernel>([=] (cl::sycl::id<1> i)
      {
        floatAcc[i] = subtract(floatAcc[i], 42.0f);
      })); // end of commands for this queue
    } // end scope, so we wait for the queue to complete
    std::cout << "finalResultFloat: " << finalResultFloat[3] << std::endl;
    return 0;
}
