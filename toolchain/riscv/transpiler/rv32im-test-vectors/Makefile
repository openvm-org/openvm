#=======================================================================
# Makefile for riscv-tests/isa
#-----------------------------------------------------------------------

XLEN ?= 64

src_dir := .
out_dir := ./tests

include $(src_dir)/rv32i/Makefrag
include $(src_dir)/rv32m/Makefrag

default: all

#--------------------------------------------------------------------
# Build rules
#--------------------------------------------------------------------

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf-
RISCV_GCC ?= $(RISCV_PREFIX)gcc
RISCV_GCC_OPTS ?= -static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles
RISCV_OBJDUMP ?= $(RISCV_PREFIX)objdump 
RISCV_SIM ?= spike

vpath %.S $(src_dir)

#------------------------------------------------------------
# Build assembly tests

%.dump: %
	$(RISCV_OBJDUMP) -D $< > $@

%.out32: %
	$(RISCV_SIM) --isa=rv32gc_zfh_zicboz_svnapot_zicntr_zba_zbb_zbc_zbs --misaligned $< 2> $@

define compile_template

$$($(1)_p_tests): $(1)-p-%: $(1)/%.S
	$$(RISCV_GCC) $(2) $$(RISCV_GCC_OPTS) -I$(src_dir)/utils -T$(src_dir)/utils/link.ld $$< -o $(out_dir)/$$@
$(1)_tests += $$($(1)_p_tests)

$(1)_tests_dump = $$(addsuffix .dump, $$($(1)_tests))

.PHONY: $(1)

COMPILER_SUPPORTS_$(1) := $$(shell $$(RISCV_GCC) $(2) -c -x c /dev/null -o /dev/null 2> /dev/null; echo $$$$?)

ifeq ($$(COMPILER_SUPPORTS_$(1)),0)
tests += $$($(1)_tests)
endif

endef

$(eval $(call compile_template,rv32i,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32m,-march=rv32g -mabi=ilp32))

tests_dump = $(addsuffix .dump, $(tests))
tests_hex = $(addsuffix .hex, $(tests))
tests32_out = $(addsuffix .out32, $(filter rv32%, $(tests)))

run: $(tests_out) $(tests32_out)

junk += $(addprefix $(out_dir)/, $(tests) $(tests_dump) $(tests_hex) $(tests32_out))

#------------------------------------------------------------
# Default

rv32i: $(rv32i_tests)
rv32m: $(rv32m_tests)

all: $(tests)

dump: $(addprefix $(out_dir)/, $(tests_dump))

#------------------------------------------------------------
# Clean up

clean:
	rm -rf $(junk)
