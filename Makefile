NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89 -Iinclude --ptxas-options=-v 

ifeq ($(OS),Windows_NT)
NATIVE_LIBRARY = bench/bin/minivllm_ops.dll
NATIVE_PLATFORM_FLAGS =
else
NATIVE_LIBRARY = bench/bin/libminivllm_ops.so
NATIVE_PLATFORM_FLAGS = -Xcompiler -fPIC
endif

SOURCES = $(wildcard kernels/*.cu)
PAGED_KV_SOURCES = $(wildcard kernels/paged_kv_cache/*.cu)
TARGETS = $(patsubst kernels/%.cu, bench/bin/%, $(SOURCES))
PAGED_KV_TARGETS = $(patsubst kernels/paged_kv_cache/%.cu, bench/bin/%, $(PAGED_KV_SOURCES))

all: $(TARGETS) $(PAGED_KV_TARGETS)

native: $(NATIVE_LIBRARY)

$(NATIVE_LIBRARY): runtime/minivllm_ops.cu
	@mkdir -p bench/bin
	$(NVCC) $(NVCCFLAGS) $(NATIVE_PLATFORM_FLAGS) -shared $< -o $@

bench/bin/%: kernels/%.cu
	@mkdir -p bench/bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bench/bin/%: kernels/paged_kv_cache/%.cu
	@mkdir -p bench/bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: clean native

clean:
	rm -rf bench/bin/*
	rm -rf bench/logs/*
	rm -rf flash_attention_report.ncu-rep
