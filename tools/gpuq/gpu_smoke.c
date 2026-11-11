#include <cuda.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static const char gpu_smoke_ptx[] =
    ".version 4.2\n"
    ".target sm_52\n"
    ".address_size 64\n"
    ".visible .entry increment_u32(\n"
    "    .param .u64 values,\n"
    "    .param .u32 count\n"
    ")\n"
    "{\n"
    "    .reg .pred %p;\n"
    "    .reg .b32 %r<4>;\n"
    "    .reg .b64 %rd<4>;\n"
    "    ld.param.u64 %rd1, [values];\n"
    "    ld.param.u32 %r1, [count];\n"
    "    mov.u32 %r2, %tid.x;\n"
    "    setp.ge.u32 %p, %r2, %r1;\n"
    "    @%p bra DONE;\n"
    "    mul.wide.u32 %rd2, %r2, 4;\n"
    "    add.u64 %rd3, %rd1, %rd2;\n"
    "    ld.global.u32 %r3, [%rd3];\n"
    "    add.u32 %r3, %r3, 1;\n"
    "    st.global.u32 [%rd3], %r3;\n"
    "DONE:\n"
    "    ret;\n"
    "}\n";

static int report_cuda_error(CUresult result, const char *expression)
{
	const char *message = "unknown CUDA error";
	cuGetErrorString(result, &message);
	fprintf(stderr, "%s failed: %s\n", expression, message);
	return 1;
}

#define CUDA_CHECK(expression) \
	do { \
		CUresult result_ = (expression); \
		if (result_ != CUDA_SUCCESS) \
			return report_cuda_error(result_, #expression); \
	} while (0)

int main(void)
{
	enum { value_count = 64 };
	uint32_t input[value_count];
	uint32_t output[value_count];
	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction increment;
	CUdeviceptr device_values;
	char device_name[256];
	unsigned int count = value_count;
	void *arguments[2];

	for (unsigned int i = 0; i < value_count; i++)
		input[i] = i * 3;

	CUDA_CHECK(cuInit(0));
	CUDA_CHECK(cuDeviceGet(&device, 0));
	CUDA_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device));
	CUDA_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
	CUDA_CHECK(cuModuleLoadData(&module, gpu_smoke_ptx));
	CUDA_CHECK(cuModuleGetFunction(&increment, module, "increment_u32"));
	CUDA_CHECK(cuMemAlloc(&device_values, sizeof(input)));
	CUDA_CHECK(cuMemcpyHtoD(device_values, input, sizeof(input)));

	arguments[0] = &device_values;
	arguments[1] = &count;
	CUDA_CHECK(cuLaunchKernel(increment, 1, 1, 1, value_count, 1, 1,
				  0, NULL, arguments, NULL));
	CUDA_CHECK(cuCtxSynchronize());
	CUDA_CHECK(cuMemcpyDtoH(output, device_values, sizeof(output)));

	for (unsigned int i = 0; i < value_count; i++) {
		if (output[i] != input[i] + 1) {
			fprintf(stderr, "GPU result mismatch at %u: got %u, expected %u\n",
				i, output[i], input[i] + 1);
			return 1;
		}
	}

	cuMemFree(device_values);
	cuModuleUnload(module);
	cuCtxDestroy(context);
	printf("GPU smoke passed on %s: %u results verified\n", device_name,
	       value_count);
	return 0;
}
