OUTPUT_ARCH( "riscv" )
ENTRY(_start)

SECTIONS
{
  . = 0x0;
  .text : { *(.text) }
  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }
  . = ALIGN(0x1000);
  .text : { *(.text) }
  . = ALIGN(0x1000);
  .data : { *(.data) }
  .bss : { *(.bss) }
  _end = .;
}

