ROOT := $(abspath ../..)
OUT := $(ROOT)/build/issue102-lock-repro
SOURCE := $(ROOT)/validation/issue102/lock_repro.c
GPUQ_RUNNER := $(ROOT)/tools/gpuq/gpuq.rb

CC ?= cc
CFLAGS ?= -O2

.PHONY: issue102-repro issue102-regression

$(OUT): $(SOURCE)
	mkdir -p $(dir $@)
	$(CC) -std=c11 -Wall -Wextra -Werror -pedantic $(CFLAGS) $(LDFLAGS) -o $@ $< -lsqlite3

issue102-repro: $(OUT)
	$(OUT) reproduce $(GPUQ_RUNNER) $(OUT)

issue102-regression: $(OUT)
	$(OUT) regression $(GPUQ_RUNNER) $(OUT)
