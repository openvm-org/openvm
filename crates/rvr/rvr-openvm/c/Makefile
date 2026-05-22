CC ?= clang-22
LINKER ?= lld
OPT ?= -O3
DEBUG ?=
CFLAGS = -std=c23 $(OPT) $(DEBUG) -Wall -Wno-unused-variable -Wno-unused-function -Wno-unused-parameter -fPIC -march=native -falign-functions=32
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

# include the right check_mem_bounds source file based on OPENVM_UNPROTECTED flag
OPENVM_UNPROTECTED ?=
ALL_SRCS = $(wildcard *.c)
ifeq ($(OPENVM_UNPROTECTED),1)
SRCS = $(filter-out openvm_check_mem_bounds_protected.c, $(ALL_SRCS))
else
SRCS = $(filter-out openvm_check_mem_bounds_unprotected.c, $(ALL_SRCS))
endif

OBJS = $(SRCS:.c=.o)

LIB ?= libout.so
EXT_LIBS ?=
EXT_SRCS ?=
EXT_OBJS = $(EXT_SRCS:.c=.o)
EXT_CFLAGS ?=

$(EXT_OBJS): CFLAGS += $(EXT_CFLAGS)

shared: $(LIB)

$(LIB): $(OBJS)
	$(CC) -shared $(CFLAGS) $(LDFLAGS) -o $@ $^ $(EXT_LIBS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(LIB)

.PHONY: shared clean
