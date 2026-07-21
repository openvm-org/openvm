CC ?= clang-22
LINKER ?= lld
OPT ?= -O3
DEBUG ?=
WARNINGS = -Weverything -Werror

# The generated sources target C2y, not C++ or older C language revisions.
WARNINGS += -Wno-c++-compat -Wno-c++98-compat \
	-Wno-pre-c11-compat -Wno-pre-c23-compat -Wno-pre-c2y-compat \
	-Wno-declaration-after-statement

# Do not apply OpenVM warning flags to third-party C. Compiler errors still
# fail the build.
VENDOR_WARNINGS = -Wno-everything
# The core runtime uses raw buffers with explicit bounds checks. Clang raises
# this warning when raw pointers could be replaced with checked containers, so
# it is not useful for these low-level files.
CORE_WARNINGS = -Wno-unsafe-buffer-usage
CFLAGS = -std=c2y $(OPT) $(DEBUG) $(WARNINGS) -fPIC -march=native -falign-functions=32
LTO ?=
CFLAGS += $(LTO)
LDFLAGS ?=
LDFLAGS += -fuse-ld=$(LINKER)
LDLIBS ?=

# On ELF platforms, bind symbols when the library loads. Lazy PLT binding can
# overwrite `preserve_none` argument registers such as r11 on x86_64.
HOST_OS ?= $(shell uname -s)
ifeq ($(HOST_OS),Linux)
LDFLAGS += -Wl,-z,now
CFLAGS += -fno-plt -fno-semantic-interposition
LDLIBS += -lm
endif

LIB ?= libout.so
EXT_LIBS ?=
EXT_LDLIBS ?=
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
	$(CC) -shared $(CFLAGS) $(LDFLAGS) -o $@ $^ $(EXT_LIBS) $(EXT_LDLIBS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(ALL_OBJS) $(LIB)

.PHONY: shared clean
