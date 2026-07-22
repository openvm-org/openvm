CC ?= clang-22
LINKER ?= lld
OPT ?= -O3
DEBUG ?=
WARNINGS = -Weverything -Werror
# `bounds` adds local array checks that are not part of the `undefined` group.
# Trap mode keeps the generated shared library self-contained. Override with
# `make SANITIZERS=` only for performance-sensitive runs.
SANITIZERS ?= -fsanitize=undefined,bounds -fsanitize-trap=all

# The generated sources target C2y, not C++ or older C language revisions.
# This C-only project also does not use the libstdc++ installation selected by
# Clang's driver.
WARNINGS += -Wno-c++-compat -Wno-c++98-compat \
	-Wno-pre-c11-compat -Wno-pre-c23-compat -Wno-pre-c2y-compat \
	-Wno-declaration-after-statement \
	-Wno-gcc-install-dir-libstdcxx \
	-Wno-pedantic -Wno-unused-parameter -Wno-padded \
	-Wno-cast-align -Wno-unsafe-buffer-usage

# Clang 22 diagnoses hosts with multiple GCC installations while selecting its
# C++ include path, even though the generated runtime is C. Probe before adding
# the suppression so older supported Clang releases do not reject the flag.
WARNINGS += $(shell $(CC) -Werror -Wno-gcc-install-dir-libstdcxx \
	-x c -fsyntax-only /dev/null >/dev/null 2>&1 && \
	printf '%s' -Wno-gcc-install-dir-libstdcxx)

# Do not apply OpenVM warning flags to third-party C. Compiler errors still
# fail the build.
VENDOR_WARNINGS = -Wno-everything
# The core runtime uses raw buffers with explicit bounds checks. Clang raises
# this warning when raw pointers could be replaced with checked containers, so
# it is not useful for these low-level files.
CORE_WARNINGS = -Wno-unsafe-buffer-usage -Wno-unused-variable \
	-Wno-unused-but-set-variable
CFLAGS = -std=c2y $(OPT) $(DEBUG) $(WARNINGS) -fPIC -march=native -falign-functions=32
CFLAGS += $(SANITIZERS)
LTO ?=
CFLAGS += $(LTO)
LDFLAGS ?=
LDFLAGS += -fuse-ld=$(LINKER)
LDLIBS ?=
THINLTO_JOBS ?= 1
THINLTO_CACHE_DIR ?=
THINLTO_LDFLAGS = -Wl,--thinlto-jobs=$(THINLTO_JOBS)
ifneq ($(strip $(THINLTO_CACHE_DIR)),)
THINLTO_LDFLAGS += -Wl,--thinlto-cache-dir="$(THINLTO_CACHE_DIR)"
endif

# On ELF platforms, bind symbols when the library loads. Lazy PLT binding can
# overwrite `preserve_none` argument registers such as r11 on x86_64.
HOST_OS ?= $(shell uname -s)
ifeq ($(HOST_OS),Linux)
# Reject unresolved symbols and linker warnings while the generated artifact is
# still being built instead of deferring failures until dlopen.
LDFLAGS += -Wl,-z,now -Wl,-z,defs -Wl,--fatal-warnings
CFLAGS += -fno-plt -fno-semantic-interposition
LDLIBS += -lm
endif

LIB ?= libout.so
EXT_LIBS ?=
EXT_SRCS ?=
EXT_OBJS = $(EXT_SRCS:.c=.o)
VENDOR_SRCS ?=
VENDOR_OBJS = $(VENDOR_SRCS:.c=.o)
CORE_SRCS = $(filter-out $(EXT_SRCS) $(VENDOR_SRCS),$(wildcard *.c))
CORE_OBJS = $(CORE_SRCS:.c=.o)
ALL_OBJS = $(sort $(CORE_OBJS) $(EXT_OBJS) $(VENDOR_OBJS))
EXT_CFLAGS ?=

$(CORE_OBJS): CFLAGS += $(CORE_WARNINGS)
$(EXT_OBJS): CFLAGS += $(EXT_CFLAGS)
$(VENDOR_OBJS): CFLAGS += $(EXT_CFLAGS) $(VENDOR_WARNINGS)

shared: $(LIB)

$(LIB): $(ALL_OBJS)
	$(CC) -shared $(CFLAGS) $(LDFLAGS) $(THINLTO_LDFLAGS) -o $@ $^ $(EXT_LIBS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(ALL_OBJS) $(LIB)

.PHONY: shared clean
