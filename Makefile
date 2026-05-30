NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89 -Iinclude --ptxas-options=-v 

SOURCES = $(wildcard kernels/*.cu)
PAGED_KV_SOURCES = $(wildcard kernels/paged_kv_cache/*.cu)
TARGETS = $(patsubst kernels/%.cu, bench/bin/%, $(SOURCES))
PAGED_KV_TARGETS = $(patsubst kernels/paged_kv_cache/%.cu, bench/bin/%, $(PAGED_KV_SOURCES))

all: $(TARGETS) $(PAGED_KV_TARGETS)

bench/bin/%: kernels/%.cu
	@mkdir -p bench/bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bench/bin/%: kernels/paged_kv_cache/%.cu
	@mkdir -p bench/bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: clean

clean:
	rm -rf bench/bin/*
	rm -rf bench/logs/*
	rm -rf flash_attention_report.ncu-rep
