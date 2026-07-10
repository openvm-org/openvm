// This is musl-libc commit 37e18b7bf307fa4a8c745feebfcba54a0ba74f30:
//
// src/string/memcpy.c
//
// This was compiled into assembly with:
//
// clang-14 -target riscv32 -march=rv32im -O3 -S memcpy.c -nostdlib -fno-builtin -funroll-loops
//
// and labels manually updated to not conflict.
//
// musl as a whole is licensed under the following standard MIT license:
//
// ----------------------------------------------------------------------
// Copyright © 2005-2020 Rich Felker, et al.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ----------------------------------------------------------------------
//
// Authors/contributors include:
//
// A. Wilcox
// Ada Worcester
// Alex Dowad
// Alex Suykov
// Alexander Monakov
// Andre McCurdy
// Andrew Kelley
// Anthony G. Basile
// Aric Belsito
// Arvid Picciani
// Bartosz Brachaczek
// Benjamin Peterson
// Bobby Bingham
// Boris Brezillon
// Brent Cook
// Chris Spiegel
// Clément Vasseur
// Daniel Micay
// Daniel Sabogal
// Daurnimator
// David Carlier
// David Edelsohn
// Denys Vlasenko
// Dmitry Ivanov
// Dmitry V. Levin
// Drew DeVault
// Emil Renner Berthing
// Fangrui Song
// Felix Fietkau
// Felix Janda
// Gianluca Anzolin
// Hauke Mehrtens
// He X
// Hiltjo Posthuma
// Isaac Dunham
// Jaydeep Patil
// Jens Gustedt
// Jeremy Huntwork
// Jo-Philipp Wich
// Joakim Sindholt
// John Spencer
// Julien Ramseier
// Justin Cormack
// Kaarle Ritvanen
// Khem Raj
// Kylie McClain
// Leah Neukirchen
// Luca Barbato
// Luka Perkov
// M Farkas-Dyck (Strake)
// Mahesh Bodapati
// Markus Wichmann
// Masanori Ogino
// Michael Clark
// Michael Forney
// Mikhail Kremnyov
// Natanael Copa
// Nicholas J. Kain
// orc
// Pascal Cuoq
// Patrick Oppenlander
// Petr Hosek
// Petr Skocik
// Pierre Carrier
// Reini Urban
// Rich Felker
// Richard Pennington
// Ryan Fairfax
// Samuel Holland
// Segev Finer
// Shiz
// sin
// Solar Designer
// Stefan Kristiansson
// Stefan O'Rear
// Szabolcs Nagy
// Timo Teräs
// Trutz Behn
// Valentin Ochs
// Will Dietz
// William Haddon
// William Pitcock
//
// Portions of this software are derived from third-party works licensed
// under terms compatible with the above MIT license:
//
// The TRE regular expression implementation (src/regex/reg* and
// src/regex/tre*) is Copyright © 2001-2008 Ville Laurikari and licensed
// under a 2-clause BSD license (license text in the source files). The
// included version has been heavily modified by Rich Felker in 2012, in
// the interests of size, simplicity, and namespace cleanliness.
//
// Much of the math library code (src/math/* and src/complex/*) is
// Copyright © 1993,2004 Sun Microsystems or
// Copyright © 2003-2011 David Schultz or
// Copyright © 2003-2009 Steven G. Kargl or
// Copyright © 2003-2009 Bruce D. Evans or
// Copyright © 2008 Stephen L. Moshier or
// Copyright © 2017-2018 Arm Limited
// and labelled as such in comments in the individual source files. All
// have been licensed under extremely permissive terms.
//
// The ARM memcpy code (src/string/arm/memcpy.S) is Copyright © 2008
// The Android Open Source Project and is licensed under a two-clause BSD
// license. It was taken from Bionic libc, used on Android.
//
// The AArch64 memcpy and memset code (src/string/aarch64/*) are
// Copyright © 1999-2019, Arm Limited.
//
// The implementation of DES for crypt (src/crypt/crypt_des.c) is
// Copyright © 1994 David Burren. It is licensed under a BSD license.
//
// The implementation of blowfish crypt (src/crypt/crypt_blowfish.c) was
// originally written by Solar Designer and placed into the public
// domain. The code also comes with a fallback permissive license for use
// in jurisdictions that may not recognize the public domain.
//
// The smoothsort implementation (src/stdlib/qsort.c) is Copyright © 2011
// Valentin Ochs and is licensed under an MIT-style license.
//
// The x86_64 port was written by Nicholas J. Kain and is licensed under
// the standard MIT terms.
//
// The mips and microblaze ports were originally written by Richard
// Pennington for use in the ellcc project. The original code was adapted
// by Rich Felker for build system and code conventions during upstream
// integration. It is licensed under the standard MIT terms.
//
// The mips64 port was contributed by Imagination Technologies and is
// licensed under the standard MIT terms.
//
// The powerpc port was also originally written by Richard Pennington,
// and later supplemented and integrated by John Spencer. It is licensed
// under the standard MIT terms.
//
// All other files which have no copyright comments are original works
// produced specifically for use as part of this library, written either
// by Rich Felker, the main author of the library, or by one or more
// contributors listed above. Details on authorship of individual files
// can be found in the git version control history of the project. The
// omission of copyright and license comments in each file is in the
// interest of source tree size.
//
// In addition, permission is hereby granted for all public header files
// (include/* and arch/* /bits/* ) and crt files intended to be linked into
// applications (crt/*, ldso/dlstart.c, and arch/* /crt_arch.h) to omit
// the copyright notice and permission notice otherwise required by the
// license, and to use these files without any requirement of
// attribution. These files include substantial contributions from:
//
// Bobby Bingham
// John Spencer
// Nicholas J. Kain
// Rich Felker
// Richard Pennington
// Stefan Kristiansson
// Szabolcs Nagy
//
// all of whom have explicitly granted such permission.
//
// This file previously contained text expressing a belief that most of
// the files covered by the above exception were sufficiently trivial not
// to be subject to copyright, resulting in confusion over whether it
// negated the permissions granted in the license. In the spirit of
// permissive licensing, and of not having licensing issues being an
// obstacle to adoption, that text has been removed.
//
// This was compiled into assembly from musl's memcpy implementation
// (src/string/memcpy.c, musl v1.2.5) with:
//
// clang -target riscv32 -march=rv32im -O2 -fno-builtin -ffreestanding \
//     -D"__BYTE_ORDER=1234" -D"__LITTLE_ENDIAN=1234" \
//     -S musl_memcpy_no_includes.c -o src/memcpy.s
//
// (Homebrew clang 22.1.0.) musl_memcpy_no_includes.c is musl's memcpy.c with:
// * `#include <string.h>` replaced by `typedef unsigned int size_t;` and
//   `#include <endian.h>` removed; the -D flags supply the endianness macros
//   (rv32 is little-endian).
// * a small-copy fast path inserted at the top of the function body:
//   for n <= 32, a switch on n whose cases fall through in descending order,
//   each case K performing d[K-1] = s[K-1], then return dest. In C:
//
//       if n <= 32:  switch (n)
//           case 32: d[31] = s[31]; fallthrough
//           ... descending through every case ...
//           case  1: d[0]  = s[0];  fallthrough
//           case  0: ;
//       return dest
//
//   This compiles to a jump table into straight-line constant-offset byte
//   copies (~2.4 instructions/byte). Small copies dominate call counts in
//   practice, and musl's general path spends 20-90 instructions on alignment
//   dispatch before copying anything: with this path n<=8 costs ~2x fewer
//   instructions and n<=32 ~1.5x fewer, measured across all 16 src/dst
//   alignment combinations. Above 32 bytes musl's word loops win, including
//   the __GNUC__ shifted-word path for src/dst alignment mismatch.
//
// Labels were renamed to not conflict:
// * .LBB0_X -> .LBBmemcpy0_X
// * .LJTI0_0 -> .LJTImemcpy0_0
// * .Lfunc_end0 -> .Lmemcpyfunc_end0
// * .attribute 5 arch string -> "rv32im"
	.attribute	4, 16
	.attribute	5, "rv32im"
	.file	"memcpy_variant_c.c"
	.text
	.globl	memcpy                          # -- Begin function memcpy
	.p2align	2
	.type	memcpy,@function
memcpy:                                 # @memcpy
# %bb.0:
	li	a3, 33
	bgeu	a2, a3, .LBBmemcpy0_34
# %bb.1:
	addi	a2, a2, -1
	li	a3, 31
	bltu	a3, a2, .LBBmemcpy0_69
# %bb.2:
	slli	a2, a2, 2
	lui	a3, %hi(.LJTImemcpy0_0)
	addi	a3, a3, %lo(.LJTImemcpy0_0)
	add	a2, a3, a2
	lw	a2, 0(a2)
	mv	a3, a0
	jr	a2
.LBBmemcpy0_3:
	lbu	a2, 31(a1)
	sb	a2, 31(a0)
.LBBmemcpy0_4:
	lbu	a2, 30(a1)
	sb	a2, 30(a0)
.LBBmemcpy0_5:
	lbu	a2, 29(a1)
	sb	a2, 29(a0)
.LBBmemcpy0_6:
	lbu	a2, 28(a1)
	sb	a2, 28(a0)
.LBBmemcpy0_7:
	lbu	a2, 27(a1)
	sb	a2, 27(a0)
.LBBmemcpy0_8:
	lbu	a2, 26(a1)
	sb	a2, 26(a0)
.LBBmemcpy0_9:
	lbu	a2, 25(a1)
	sb	a2, 25(a0)
.LBBmemcpy0_10:
	lbu	a2, 24(a1)
	sb	a2, 24(a0)
.LBBmemcpy0_11:
	lbu	a2, 23(a1)
	sb	a2, 23(a0)
.LBBmemcpy0_12:
	lbu	a2, 22(a1)
	sb	a2, 22(a0)
.LBBmemcpy0_13:
	lbu	a2, 21(a1)
	sb	a2, 21(a0)
.LBBmemcpy0_14:
	lbu	a2, 20(a1)
	sb	a2, 20(a0)
.LBBmemcpy0_15:
	lbu	a2, 19(a1)
	sb	a2, 19(a0)
.LBBmemcpy0_16:
	lbu	a2, 18(a1)
	sb	a2, 18(a0)
.LBBmemcpy0_17:
	lbu	a2, 17(a1)
	sb	a2, 17(a0)
.LBBmemcpy0_18:
	lbu	a2, 16(a1)
	sb	a2, 16(a0)
.LBBmemcpy0_19:
	lbu	a2, 15(a1)
	sb	a2, 15(a0)
.LBBmemcpy0_20:
	lbu	a2, 14(a1)
	sb	a2, 14(a0)
.LBBmemcpy0_21:
	lbu	a2, 13(a1)
	sb	a2, 13(a0)
.LBBmemcpy0_22:
	lbu	a2, 12(a1)
	sb	a2, 12(a0)
.LBBmemcpy0_23:
	lbu	a2, 11(a1)
	sb	a2, 11(a0)
.LBBmemcpy0_24:
	lbu	a2, 10(a1)
	sb	a2, 10(a0)
.LBBmemcpy0_25:
	lbu	a2, 9(a1)
	sb	a2, 9(a0)
.LBBmemcpy0_26:
	lbu	a2, 8(a1)
	sb	a2, 8(a0)
.LBBmemcpy0_27:
	lbu	a2, 7(a1)
	sb	a2, 7(a0)
.LBBmemcpy0_28:
	lbu	a2, 6(a1)
	sb	a2, 6(a0)
.LBBmemcpy0_29:
	lbu	a2, 5(a1)
	sb	a2, 5(a0)
.LBBmemcpy0_30:
	lbu	a2, 4(a1)
	sb	a2, 4(a0)
.LBBmemcpy0_31:
	lbu	a2, 3(a1)
	sb	a2, 3(a0)
.LBBmemcpy0_32:
	lbu	a2, 2(a1)
	sb	a2, 2(a0)
.LBBmemcpy0_33:
	lbu	a2, 1(a1)
	sb	a2, 1(a0)
	mv	a3, a0
	j	.LBBmemcpy0_68
.LBBmemcpy0_34:
	andi	a3, a1, 3
	beqz	a3, .LBBmemcpy0_43
# %bb.35:
	addi	a4, a1, 1
	li	a5, 1
	mv	a3, a0
.LBBmemcpy0_36:                               # =>This Inner Loop Header: Depth=1
	lbu	a7, 0(a1)
	mv	a6, a2
	addi	a1, a1, 1
	andi	t0, a4, 3
	sb	a7, 0(a3)
	addi	a3, a3, 1
	addi	a2, a2, -1
	beqz	t0, .LBBmemcpy0_38
# %bb.37:                               #   in Loop: Header=BB0_36 Depth=1
	addi	a4, a4, 1
	bne	a6, a5, .LBBmemcpy0_36
.LBBmemcpy0_38:
	andi	a5, a3, 3
	beqz	a5, .LBBmemcpy0_44
.LBBmemcpy0_39:
	li	a4, 32
	bgeu	a2, a4, .LBBmemcpy0_51
# %bb.40:
	li	a4, 16
	bgeu	a2, a4, .LBBmemcpy0_62
.LBBmemcpy0_41:
	andi	a4, a2, 8
	bnez	a4, .LBBmemcpy0_63
.LBBmemcpy0_42:
	andi	a4, a2, 4
	bnez	a4, .LBBmemcpy0_64
	j	.LBBmemcpy0_65
.LBBmemcpy0_43:
	mv	a3, a0
	andi	a5, a0, 3
	bnez	a5, .LBBmemcpy0_39
.LBBmemcpy0_44:
	li	a4, 16
	bltu	a2, a4, .LBBmemcpy0_47
# %bb.45:
	li	a4, 15
.LBBmemcpy0_46:                               # =>This Inner Loop Header: Depth=1
	lw	a5, 0(a1)
	lw	a6, 4(a1)
	lw	a7, 8(a1)
	lw	t0, 12(a1)
	addi	a1, a1, 16
	addi	a2, a2, -16
	sw	a5, 0(a3)
	sw	a6, 4(a3)
	sw	a7, 8(a3)
	sw	t0, 12(a3)
	addi	a3, a3, 16
	bltu	a4, a2, .LBBmemcpy0_46
.LBBmemcpy0_47:
	li	a4, 8
	bltu	a2, a4, .LBBmemcpy0_49
# %bb.48:
	lw	a4, 0(a1)
	lw	a5, 4(a1)
	sw	a4, 0(a3)
	sw	a5, 4(a3)
	addi	a3, a3, 8
	addi	a1, a1, 8
.LBBmemcpy0_49:
	andi	a4, a2, 4
	beqz	a4, .LBBmemcpy0_65
# %bb.50:
	lw	a4, 0(a1)
	addi	a1, a1, 4
	sw	a4, 0(a3)
	addi	a3, a3, 4
	j	.LBBmemcpy0_65
.LBBmemcpy0_51:
	lw	a4, 0(a1)
	li	a6, 3
	beq	a5, a6, .LBBmemcpy0_56
# %bb.52:
	li	a6, 2
	bne	a5, a6, .LBBmemcpy0_59
# %bb.53:
	srli	a5, a4, 8
	addi	a2, a2, -2
	addi	a1, a1, 16
	sb	a4, 0(a3)
	sb	a5, 1(a3)
	addi	a3, a3, 2
	li	a5, 17
.LBBmemcpy0_54:                               # =>This Inner Loop Header: Depth=1
	srli	a6, a4, 16
	lw	a7, -12(a1)
	lw	t0, -8(a1)
	lw	t1, -4(a1)
	lw	a4, 0(a1)
	slli	t2, a7, 16
	srli	a7, a7, 16
	or	a6, t2, a6
	slli	t2, t0, 16
	srli	t0, t0, 16
	or	a7, t2, a7
	slli	t2, t1, 16
	srli	t1, t1, 16
	or	t0, t2, t0
	slli	t2, a4, 16
	or	t1, t2, t1
	addi	a2, a2, -16
	sw	a6, 0(a3)
	sw	a7, 4(a3)
	sw	t0, 8(a3)
	sw	t1, 12(a3)
	addi	a3, a3, 16
	addi	a1, a1, 16
	bltu	a5, a2, .LBBmemcpy0_54
# %bb.55:
	addi	a1, a1, -14
	li	a4, 16
	bltu	a2, a4, .LBBmemcpy0_41
	j	.LBBmemcpy0_62
.LBBmemcpy0_56:
	sb	a4, 0(a3)
	addi	a2, a2, -1
	addi	a3, a3, 1
	addi	a1, a1, 16
	li	a5, 18
.LBBmemcpy0_57:                               # =>This Inner Loop Header: Depth=1
	srli	a6, a4, 8
	lw	a7, -12(a1)
	lw	t0, -8(a1)
	lw	t1, -4(a1)
	lw	a4, 0(a1)
	slli	t2, a7, 24
	srli	a7, a7, 8
	or	a6, t2, a6
	slli	t2, t0, 24
	srli	t0, t0, 8
	or	a7, t2, a7
	slli	t2, t1, 24
	srli	t1, t1, 8
	or	t0, t2, t0
	slli	t2, a4, 24
	or	t1, t2, t1
	addi	a2, a2, -16
	sw	a6, 0(a3)
	sw	a7, 4(a3)
	sw	t0, 8(a3)
	sw	t1, 12(a3)
	addi	a3, a3, 16
	addi	a1, a1, 16
	bltu	a5, a2, .LBBmemcpy0_57
# %bb.58:
	addi	a1, a1, -15
	li	a4, 16
	bltu	a2, a4, .LBBmemcpy0_41
	j	.LBBmemcpy0_62
.LBBmemcpy0_59:
	srli	a5, a4, 8
	srli	a6, a4, 16
	addi	a2, a2, -3
	addi	a1, a1, 16
	sb	a4, 0(a3)
	sb	a5, 1(a3)
	sb	a6, 2(a3)
	addi	a3, a3, 3
	li	a5, 16
.LBBmemcpy0_60:                               # =>This Inner Loop Header: Depth=1
	srli	a6, a4, 24
	lw	a7, -12(a1)
	lw	t0, -8(a1)
	lw	t1, -4(a1)
	lw	a4, 0(a1)
	slli	t2, a7, 8
	srli	a7, a7, 24
	or	a6, t2, a6
	slli	t2, t0, 8
	srli	t0, t0, 24
	or	a7, t2, a7
	slli	t2, t1, 8
	srli	t1, t1, 24
	or	t0, t2, t0
	slli	t2, a4, 8
	or	t1, t2, t1
	addi	a2, a2, -16
	sw	a6, 0(a3)
	sw	a7, 4(a3)
	sw	t0, 8(a3)
	sw	t1, 12(a3)
	addi	a3, a3, 16
	addi	a1, a1, 16
	bltu	a5, a2, .LBBmemcpy0_60
# %bb.61:
	addi	a1, a1, -13
	li	a4, 16
	bltu	a2, a4, .LBBmemcpy0_41
.LBBmemcpy0_62:
	lbu	a4, 0(a1)
	lbu	a5, 1(a1)
	lbu	a6, 2(a1)
	lbu	a7, 3(a1)
	lbu	t0, 4(a1)
	lbu	t1, 5(a1)
	lbu	t2, 6(a1)
	lbu	t3, 7(a1)
	sb	a4, 0(a3)
	sb	a5, 1(a3)
	sb	a6, 2(a3)
	sb	a7, 3(a3)
	lbu	a4, 8(a1)
	lbu	a5, 9(a1)
	lbu	a6, 10(a1)
	lbu	a7, 11(a1)
	sb	t0, 4(a3)
	sb	t1, 5(a3)
	sb	t2, 6(a3)
	sb	t3, 7(a3)
	lbu	t0, 12(a1)
	lbu	t1, 13(a1)
	lbu	t2, 14(a1)
	lbu	t3, 15(a1)
	addi	a1, a1, 16
	sb	a4, 8(a3)
	sb	a5, 9(a3)
	sb	a6, 10(a3)
	sb	a7, 11(a3)
	addi	a4, a3, 16
	sb	t0, 12(a3)
	sb	t1, 13(a3)
	sb	t2, 14(a3)
	sb	t3, 15(a3)
	mv	a3, a4
	andi	a4, a2, 8
	beqz	a4, .LBBmemcpy0_42
.LBBmemcpy0_63:
	lbu	a4, 0(a1)
	lbu	a5, 1(a1)
	lbu	a6, 2(a1)
	lbu	a7, 3(a1)
	lbu	t0, 4(a1)
	lbu	t1, 5(a1)
	lbu	t2, 6(a1)
	lbu	t3, 7(a1)
	addi	a1, a1, 8
	sb	a4, 0(a3)
	sb	a5, 1(a3)
	sb	a6, 2(a3)
	sb	a7, 3(a3)
	addi	a4, a3, 8
	sb	t0, 4(a3)
	sb	t1, 5(a3)
	sb	t2, 6(a3)
	sb	t3, 7(a3)
	mv	a3, a4
	andi	a4, a2, 4
	beqz	a4, .LBBmemcpy0_65
.LBBmemcpy0_64:
	lbu	a4, 0(a1)
	lbu	a5, 1(a1)
	lbu	a6, 2(a1)
	lbu	a7, 3(a1)
	addi	a1, a1, 4
	addi	t0, a3, 4
	sb	a4, 0(a3)
	sb	a5, 1(a3)
	sb	a6, 2(a3)
	sb	a7, 3(a3)
	mv	a3, t0
.LBBmemcpy0_65:
	andi	a4, a2, 2
	beqz	a4, .LBBmemcpy0_67
# %bb.66:
	lbu	a4, 0(a1)
	lbu	a5, 1(a1)
	addi	a1, a1, 2
	addi	a6, a3, 2
	sb	a4, 0(a3)
	sb	a5, 1(a3)
	mv	a3, a6
.LBBmemcpy0_67:
	andi	a2, a2, 1
	beqz	a2, .LBBmemcpy0_69
.LBBmemcpy0_68:
	lbu	a1, 0(a1)
	sb	a1, 0(a3)
.LBBmemcpy0_69:
	ret
.Lmemcpyfunc_end0:
	.size	memcpy, .Lmemcpyfunc_end0-memcpy
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
.LJTImemcpy0_0:
	.word	.LBBmemcpy0_68
	.word	.LBBmemcpy0_33
	.word	.LBBmemcpy0_32
	.word	.LBBmemcpy0_31
	.word	.LBBmemcpy0_30
	.word	.LBBmemcpy0_29
	.word	.LBBmemcpy0_28
	.word	.LBBmemcpy0_27
	.word	.LBBmemcpy0_26
	.word	.LBBmemcpy0_25
	.word	.LBBmemcpy0_24
	.word	.LBBmemcpy0_23
	.word	.LBBmemcpy0_22
	.word	.LBBmemcpy0_21
	.word	.LBBmemcpy0_20
	.word	.LBBmemcpy0_19
	.word	.LBBmemcpy0_18
	.word	.LBBmemcpy0_17
	.word	.LBBmemcpy0_16
	.word	.LBBmemcpy0_15
	.word	.LBBmemcpy0_14
	.word	.LBBmemcpy0_13
	.word	.LBBmemcpy0_12
	.word	.LBBmemcpy0_11
	.word	.LBBmemcpy0_10
	.word	.LBBmemcpy0_9
	.word	.LBBmemcpy0_8
	.word	.LBBmemcpy0_7
	.word	.LBBmemcpy0_6
	.word	.LBBmemcpy0_5
	.word	.LBBmemcpy0_4
	.word	.LBBmemcpy0_3
                                        # -- End function
