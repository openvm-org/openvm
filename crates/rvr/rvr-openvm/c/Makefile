CC ?= clang-22
LINKER ?= lld
OPT ?= -O3
DEBUG ?=
CFLAGS = -std=c2y $(OPT) $(DEBUG) -Wall -Wno-unused-variable -Wno-unused-function -Wno-unused-parameter -fPIC -march=native -falign-functions=32
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

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

LIB ?= libout.so
EXT_LIBS ?=
EXT_SRCS ?=
EXT_OBJS = $(EXT_SRCS:.c=.o)
ALL_OBJS = $(sort $(OBJS) $(EXT_OBJS))
EXT_CFLAGS ?=

$(EXT_OBJS): CFLAGS += $(EXT_CFLAGS)

shared: $(LIB)

$(LIB): $(ALL_OBJS)
	$(CC) -shared $(CFLAGS) $(LDFLAGS) -Wl,--thinlto-jobs=1 -o $@ $^ $(EXT_LIBS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(ALL_OBJS) $(LIB)

.PHONY: shared clean
