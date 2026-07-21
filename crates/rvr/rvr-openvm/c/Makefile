CC ?= clang-22
LINKER ?= lld
OPT ?= -O3
DEBUG ?=
WARNINGS = -Weverything -Werror

# The generated sources target C2y, not C++ or older C language revisions.
WARNINGS += -Wno-c++-compat -Wno-c++98-compat \
	-Wno-pre-c11-compat -Wno-pre-c23-compat -Wno-pre-c2y-compat \
	-Wno-declaration-after-statement

# Vendored translation units are outside OpenVM's warning policy. Compiler
# errors remain fatal, but diagnostics from third-party C are suppressed.
VENDOR_WARNINGS = -Wno-everything
# The core runtime implements VM memory using raw buffers behind explicit
# bounds and invariant contracts. Clang's diagnostic is a migration heuristic
# for moving raw pointers to checked container APIs, which is not applicable
# to these low-level translation units.
CORE_WARNINGS = -Wno-unsafe-buffer-usage
CFLAGS = -std=c2y $(OPT) $(DEBUG) $(WARNINGS) -fPIC -march=native -falign-functions=32
LTO ?=
CFLAGS += $(LTO)
LDFLAGS ?=
LDFLAGS += -fuse-ld=$(LINKER)
LDLIBS ?=

# On ELF platforms, force eager symbol binding to prevent the PLT lazy
# resolver from clobbering preserve_none argument registers (r11 on x86_64).
HOST_OS ?= $(shell uname -s)
ifeq ($(HOST_OS),Linux)
LDFLAGS += -Wl,-z,now
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
	$(CC) -shared $(CFLAGS) $(LDFLAGS) -o $@ $^ $(EXT_LIBS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(ALL_OBJS) $(LIB)

.PHONY: shared clean
