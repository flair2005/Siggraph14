////////////////////////////////////////////////////////////////////////////////
//
//  SYCL sample code Siggraph 2014 - The Unlicense (see LICENSE for details)
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "CL/sycl.hpp"

float subtract(float a, float b) //function we are going to call from SYCL kernel.
{
    return a - b;
}

int main()
{
	float 	floatVal = 0.0f;
	
    {   // This block defines the scope where the SYCL objects will live.
		cl::sycl::default_selector selector;  // Default selector
		cl::sycl::queue myQueue(selector);
		
		cl::sycl::buffer<float, 1> 	floatBuf(&floatVal, 1); //SYCL runtime data ownership

		cl::sycl::command_group(myQueue, [&]() {  // kernel and dependencies
            //auto == cl::sycl::accessor<float, 1, access::read_write, access::global_buffer>
            auto floatAcc = floatBuf.get_access<access::read_write>();
			cl::sycl::single_task(kernel_functor<class subtractKernel>([=](){
                floatAcc[0] = subtract(floatAcc[0], 42.0f);
            }));
		});
	}
    std::cout << "floatVal	: " << floatVal << std::endl;
	return 0;
}