    .section    __TEXT,__text,regular,pure_instructions
    .globl  _main
    .intel_syntax noprefix

_main:
    push    rbp
    mov     rbp, rsp
    call    _print_message
    pop     rbp
    ret

