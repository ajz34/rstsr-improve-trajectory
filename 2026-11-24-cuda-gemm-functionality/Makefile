NVCC  ?= nvcc
ARCH  ?= sm_120
FLAGS := -O3 -std=c++17 -arch=$(ARCH) -lcublas -lcublasLt

all: gemm_demo

gemm_demo: gemm_demo.cu
	$(NVCC) $(FLAGS) $< -o $@

probes: probes/probe_algocaps probes/probe_emulation2 probes/probe_classic_emul probes/probe_lt64_fixed

probes/%: probes/%.cu
	$(NVCC) $(FLAGS) $< -o $@

run: gemm_demo
	./gemm_demo | tee results.txt

clean:
	rm -f gemm_demo probes/probe_algocaps probes/probe_emulation2 probes/probe_classic_emul probes/probe_lt64_fixed

.PHONY: all probes run clean
