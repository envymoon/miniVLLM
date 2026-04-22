NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89 -Iinclude --ptxas-options=-v 

SOURCES = $(wildcard kernels/*.cu)
TARGETS = $(patsubst kernels/%.cu, bench/bin/%, $(SOURCES))

all: $(TARGETS)

bench/bin/%: kernels/%.cu
	@mkdir -p bench/bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: clean

clean:
	rm -rf bench/bin/*
	rm -rf bench/logs/*
	rm -rf flash_attention_report.ncu-rep