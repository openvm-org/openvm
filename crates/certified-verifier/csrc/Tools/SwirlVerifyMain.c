// Lean compiler output
// Module: Tools.SwirlVerifyMain
// Imports: public import Init public import Swirl.Protocol.Noninteractive.VerifierBabyBearPoseidon2
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_readU32LE___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_parseThreeBlobs(lean_object*);
LEAN_EXPORT lean_object* _lean_main();
uint8_t lean_byte_array_fget(lean_object*, lean_object*);
lean_object* lean_uint32_to_nat(uint32_t);
LEAN_EXPORT uint8_t l_Tools_SwirlVerifyMain_readU32LE___lam__0(lean_object*, lean_object*);
uint32_t lean_uint8_to_uint32(uint8_t);
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_parseThreeBlobs___boxed(lean_object*);
static lean_object* l_Tools_SwirlVerifyMain_main___closed__2;
static lean_object* l_Tools_SwirlVerifyMain_main___closed__1;
lean_object* l_ByteArray_extract(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_main___boxed(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_sliceBytes___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_readU32LE(lean_object*, lean_object*);
lean_object* l_IO_FS_Stream_readBinToEnd(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_readU32LE___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_main();
lean_object* lean_get_stdin();
lean_object* lean_get_stderr();
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_sliceBytes(lean_object*, lean_object*, lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2(lean_object*, lean_object*, lean_object*);
lean_object* l_outOfBounds___redArg(lean_object*);
extern uint8_t l_instInhabitedUInt8;
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_main___boxed(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_main___boxed__const__2;
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
static lean_object* l_Tools_SwirlVerifyMain_main___closed__0;
uint32_t lean_uint32_lor(uint32_t, uint32_t);
uint32_t lean_uint32_shift_left(uint32_t, uint32_t);
uint32_t l_Swirl_Protocol_Noninteractive_exitCode(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_main___boxed__const__1;
lean_object* l_IO_FS_Stream_putStrLn(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_byte_array_size(lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_MonoError_toString(lean_object*);
LEAN_EXPORT uint8_t l_Tools_SwirlVerifyMain_readU32LE___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_byte_array_size(x_1);
x_4 = lean_nat_dec_lt(x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_readU32LE(lean_object* x_1, lean_object* x_2) {
_start:
{
uint32_t x_3; uint32_t x_4; uint32_t x_5; uint8_t x_6; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; 
x_20 = lean_unsigned_to_nat(4u);
x_21 = lean_nat_add(x_2, x_20);
x_22 = lean_byte_array_size(x_1);
x_23 = lean_nat_dec_le(x_21, x_22);
lean_dec(x_22);
lean_dec(x_21);
if (x_23 == 0)
{
lean_object* x_24; 
x_24 = lean_box(0);
return x_24;
}
else
{
uint8_t x_25; uint32_t x_26; uint32_t x_27; uint8_t x_28; uint32_t x_38; uint8_t x_39; uint8_t x_49; uint8_t x_59; 
x_25 = l_instInhabitedUInt8;
x_59 = l_Tools_SwirlVerifyMain_readU32LE___lam__0(x_1, x_2);
if (x_59 == 0)
{
lean_object* x_60; lean_object* x_61; uint8_t x_62; 
x_60 = lean_box(x_25);
x_61 = l_outOfBounds___redArg(x_60);
x_62 = lean_unbox(x_61);
x_49 = x_62;
goto block_58;
}
else
{
uint8_t x_63; 
x_63 = lean_byte_array_fget(x_1, x_2);
x_49 = x_63;
goto block_58;
}
block_37:
{
uint32_t x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_29 = lean_uint8_to_uint32(x_28);
x_30 = lean_unsigned_to_nat(3u);
x_31 = lean_nat_add(x_2, x_30);
x_32 = l_Tools_SwirlVerifyMain_readU32LE___lam__0(x_1, x_31);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; uint8_t x_35; 
lean_dec(x_31);
x_33 = lean_box(x_25);
x_34 = l_outOfBounds___redArg(x_33);
x_35 = lean_unbox(x_34);
x_3 = x_26;
x_4 = x_27;
x_5 = x_29;
x_6 = x_35;
goto block_19;
}
else
{
uint8_t x_36; 
x_36 = lean_byte_array_fget(x_1, x_31);
lean_dec(x_31);
x_3 = x_26;
x_4 = x_27;
x_5 = x_29;
x_6 = x_36;
goto block_19;
}
}
block_48:
{
uint32_t x_40; lean_object* x_41; lean_object* x_42; uint8_t x_43; 
x_40 = lean_uint8_to_uint32(x_39);
x_41 = lean_unsigned_to_nat(2u);
x_42 = lean_nat_add(x_2, x_41);
x_43 = l_Tools_SwirlVerifyMain_readU32LE___lam__0(x_1, x_42);
if (x_43 == 0)
{
lean_object* x_44; lean_object* x_45; uint8_t x_46; 
lean_dec(x_42);
x_44 = lean_box(x_25);
x_45 = l_outOfBounds___redArg(x_44);
x_46 = lean_unbox(x_45);
x_26 = x_38;
x_27 = x_40;
x_28 = x_46;
goto block_37;
}
else
{
uint8_t x_47; 
x_47 = lean_byte_array_fget(x_1, x_42);
lean_dec(x_42);
x_26 = x_38;
x_27 = x_40;
x_28 = x_47;
goto block_37;
}
}
block_58:
{
uint32_t x_50; lean_object* x_51; lean_object* x_52; uint8_t x_53; 
x_50 = lean_uint8_to_uint32(x_49);
x_51 = lean_unsigned_to_nat(1u);
x_52 = lean_nat_add(x_2, x_51);
x_53 = l_Tools_SwirlVerifyMain_readU32LE___lam__0(x_1, x_52);
if (x_53 == 0)
{
lean_object* x_54; lean_object* x_55; uint8_t x_56; 
lean_dec(x_52);
x_54 = lean_box(x_25);
x_55 = l_outOfBounds___redArg(x_54);
x_56 = lean_unbox(x_55);
x_38 = x_50;
x_39 = x_56;
goto block_48;
}
else
{
uint8_t x_57; 
x_57 = lean_byte_array_fget(x_1, x_52);
lean_dec(x_52);
x_38 = x_50;
x_39 = x_57;
goto block_48;
}
}
}
block_19:
{
uint32_t x_7; uint32_t x_8; uint32_t x_9; uint32_t x_10; uint32_t x_11; uint32_t x_12; uint32_t x_13; uint32_t x_14; uint32_t x_15; uint32_t x_16; lean_object* x_17; lean_object* x_18; 
x_7 = lean_uint8_to_uint32(x_6);
x_8 = 8;
x_9 = lean_uint32_shift_left(x_4, x_8);
x_10 = lean_uint32_lor(x_3, x_9);
x_11 = 16;
x_12 = lean_uint32_shift_left(x_5, x_11);
x_13 = lean_uint32_lor(x_10, x_12);
x_14 = 24;
x_15 = lean_uint32_shift_left(x_7, x_14);
x_16 = lean_uint32_lor(x_13, x_15);
x_17 = lean_box_uint32(x_16);
x_18 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_readU32LE___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = l_Tools_SwirlVerifyMain_readU32LE___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_readU32LE___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_Tools_SwirlVerifyMain_readU32LE(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_sliceBytes(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_nat_add(x_2, x_3);
x_5 = lean_byte_array_size(x_1);
x_6 = lean_nat_dec_le(x_4, x_5);
lean_dec(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
lean_dec(x_4);
lean_dec(x_2);
x_7 = lean_box(0);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = l_ByteArray_extract(x_1, x_2, x_4);
lean_dec(x_4);
x_9 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_sliceBytes___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = l_Tools_SwirlVerifyMain_sliceBytes(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_parseThreeBlobs(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = l_Tools_SwirlVerifyMain_readU32LE(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; uint32_t x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_unsigned_to_nat(4u);
x_7 = lean_unbox_uint32(x_5);
lean_dec(x_5);
x_8 = lean_uint32_to_nat(x_7);
x_9 = l_Tools_SwirlVerifyMain_sliceBytes(x_1, x_6, x_8);
if (lean_obj_tag(x_9) == 0)
{
lean_object* x_10; 
lean_dec(x_8);
x_10 = lean_box(0);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_ctor_get(x_9, 0);
lean_inc(x_11);
lean_dec_ref(x_9);
x_12 = lean_nat_add(x_6, x_8);
lean_dec(x_8);
x_13 = l_Tools_SwirlVerifyMain_readU32LE(x_1, x_12);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; 
lean_dec(x_12);
lean_dec(x_11);
x_14 = lean_box(0);
return x_14;
}
else
{
lean_object* x_15; lean_object* x_16; uint32_t x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_13, 0);
lean_inc(x_15);
lean_dec_ref(x_13);
x_16 = lean_nat_add(x_12, x_6);
lean_dec(x_12);
x_17 = lean_unbox_uint32(x_15);
lean_dec(x_15);
x_18 = lean_uint32_to_nat(x_17);
lean_inc(x_16);
x_19 = l_Tools_SwirlVerifyMain_sliceBytes(x_1, x_16, x_18);
if (lean_obj_tag(x_19) == 0)
{
lean_object* x_20; 
lean_dec(x_18);
lean_dec(x_16);
lean_dec(x_11);
x_20 = lean_box(0);
return x_20;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_21 = lean_ctor_get(x_19, 0);
lean_inc(x_21);
lean_dec_ref(x_19);
x_22 = lean_nat_add(x_16, x_18);
lean_dec(x_18);
lean_dec(x_16);
x_23 = l_Tools_SwirlVerifyMain_readU32LE(x_1, x_22);
if (lean_obj_tag(x_23) == 0)
{
lean_object* x_24; 
lean_dec(x_22);
lean_dec(x_21);
lean_dec(x_11);
x_24 = lean_box(0);
return x_24;
}
else
{
lean_object* x_25; lean_object* x_26; uint32_t x_27; lean_object* x_28; lean_object* x_29; 
x_25 = lean_ctor_get(x_23, 0);
lean_inc(x_25);
lean_dec_ref(x_23);
x_26 = lean_nat_add(x_22, x_6);
lean_dec(x_22);
x_27 = lean_unbox_uint32(x_25);
lean_dec(x_25);
x_28 = lean_uint32_to_nat(x_27);
x_29 = l_Tools_SwirlVerifyMain_sliceBytes(x_1, x_26, x_28);
lean_dec(x_28);
if (lean_obj_tag(x_29) == 0)
{
lean_object* x_30; 
lean_dec(x_21);
lean_dec(x_11);
x_30 = lean_box(0);
return x_30;
}
else
{
uint8_t x_31; 
x_31 = !lean_is_exclusive(x_29);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_32 = lean_ctor_get(x_29, 0);
x_33 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_33, 0, x_21);
lean_ctor_set(x_33, 1, x_32);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_11);
lean_ctor_set(x_34, 1, x_33);
lean_ctor_set(x_29, 0, x_34);
return x_29;
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_35 = lean_ctor_get(x_29, 0);
lean_inc(x_35);
lean_dec(x_29);
x_36 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_36, 0, x_21);
lean_ctor_set(x_36, 1, x_35);
x_37 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_37, 0, x_11);
lean_ctor_set(x_37, 1, x_36);
x_38 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_38, 0, x_37);
return x_38;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_parseThreeBlobs___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_Tools_SwirlVerifyMain_parseThreeBlobs(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_l_Tools_SwirlVerifyMain_main___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("swirl_verify: stdin framing error (received ", 44, 44);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlVerifyMain_main___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" bytes)", 7, 7);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlVerifyMain_main___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("swirl_verify: ", 14, 14);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlVerifyMain_main___boxed__const__1() {
_start:
{
uint32_t x_1; lean_object* x_2; 
x_1 = 20;
x_2 = lean_box_uint32(x_1);
return x_2;
}
}
static lean_object* _init_l_Tools_SwirlVerifyMain_main___boxed__const__2() {
_start:
{
uint32_t x_1; lean_object* x_2; 
x_1 = 0;
x_2 = lean_box_uint32(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_main() {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_get_stdin();
x_3 = lean_get_stderr();
x_4 = l_IO_FS_Stream_readBinToEnd(x_2);
if (lean_obj_tag(x_4) == 0)
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = l_Tools_SwirlVerifyMain_parseThreeBlobs(x_6);
if (lean_obj_tag(x_7) == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_free_object(x_4);
x_8 = l_Tools_SwirlVerifyMain_main___closed__0;
x_9 = lean_byte_array_size(x_6);
lean_dec(x_6);
x_10 = l_Nat_reprFast(x_9);
x_11 = lean_string_append(x_8, x_10);
lean_dec_ref(x_10);
x_12 = l_Tools_SwirlVerifyMain_main___closed__1;
x_13 = lean_string_append(x_11, x_12);
x_14 = l_IO_FS_Stream_putStrLn(x_3, x_13);
if (lean_obj_tag(x_14) == 0)
{
uint8_t x_15; 
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; 
x_16 = lean_ctor_get(x_14, 0);
lean_dec(x_16);
x_17 = l_Tools_SwirlVerifyMain_main___boxed__const__1;
lean_ctor_set(x_14, 0, x_17);
return x_14;
}
else
{
lean_object* x_18; lean_object* x_19; 
lean_dec(x_14);
x_18 = l_Tools_SwirlVerifyMain_main___boxed__const__1;
x_19 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
else
{
uint8_t x_20; 
x_20 = !lean_is_exclusive(x_14);
if (x_20 == 0)
{
return x_14;
}
else
{
lean_object* x_21; lean_object* x_22; 
x_21 = lean_ctor_get(x_14, 0);
lean_inc(x_21);
lean_dec(x_14);
x_22 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
}
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
lean_dec(x_6);
x_23 = lean_ctor_get(x_7, 0);
lean_inc(x_23);
lean_dec_ref(x_7);
x_24 = lean_ctor_get(x_23, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_23, 0);
lean_inc(x_25);
lean_dec(x_23);
x_26 = lean_ctor_get(x_24, 0);
lean_inc(x_26);
x_27 = lean_ctor_get(x_24, 1);
lean_inc(x_27);
lean_dec(x_24);
x_28 = l_Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2(x_25, x_26, x_27);
if (lean_obj_tag(x_28) == 0)
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; 
lean_free_object(x_4);
x_29 = lean_ctor_get(x_28, 0);
lean_inc(x_29);
lean_dec_ref(x_28);
x_30 = l_Tools_SwirlVerifyMain_main___closed__2;
lean_inc(x_29);
x_31 = l_Swirl_Protocol_Noninteractive_MonoError_toString(x_29);
x_32 = lean_string_append(x_30, x_31);
lean_dec_ref(x_31);
x_33 = l_IO_FS_Stream_putStrLn(x_3, x_32);
if (lean_obj_tag(x_33) == 0)
{
uint8_t x_34; 
x_34 = !lean_is_exclusive(x_33);
if (x_34 == 0)
{
lean_object* x_35; uint32_t x_36; lean_object* x_37; 
x_35 = lean_ctor_get(x_33, 0);
lean_dec(x_35);
x_36 = l_Swirl_Protocol_Noninteractive_exitCode(x_29);
lean_dec(x_29);
x_37 = lean_box_uint32(x_36);
lean_ctor_set(x_33, 0, x_37);
return x_33;
}
else
{
uint32_t x_38; lean_object* x_39; lean_object* x_40; 
lean_dec(x_33);
x_38 = l_Swirl_Protocol_Noninteractive_exitCode(x_29);
lean_dec(x_29);
x_39 = lean_box_uint32(x_38);
x_40 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_40, 0, x_39);
return x_40;
}
}
else
{
uint8_t x_41; 
lean_dec(x_29);
x_41 = !lean_is_exclusive(x_33);
if (x_41 == 0)
{
return x_33;
}
else
{
lean_object* x_42; lean_object* x_43; 
x_42 = lean_ctor_get(x_33, 0);
lean_inc(x_42);
lean_dec(x_33);
x_43 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_43, 0, x_42);
return x_43;
}
}
}
else
{
lean_object* x_44; 
lean_dec_ref(x_28);
lean_dec_ref(x_3);
x_44 = l_Tools_SwirlVerifyMain_main___boxed__const__2;
lean_ctor_set(x_4, 0, x_44);
return x_4;
}
}
}
else
{
lean_object* x_45; lean_object* x_46; 
x_45 = lean_ctor_get(x_4, 0);
lean_inc(x_45);
lean_dec(x_4);
x_46 = l_Tools_SwirlVerifyMain_parseThreeBlobs(x_45);
if (lean_obj_tag(x_46) == 0)
{
lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_47 = l_Tools_SwirlVerifyMain_main___closed__0;
x_48 = lean_byte_array_size(x_45);
lean_dec(x_45);
x_49 = l_Nat_reprFast(x_48);
x_50 = lean_string_append(x_47, x_49);
lean_dec_ref(x_49);
x_51 = l_Tools_SwirlVerifyMain_main___closed__1;
x_52 = lean_string_append(x_50, x_51);
x_53 = l_IO_FS_Stream_putStrLn(x_3, x_52);
if (lean_obj_tag(x_53) == 0)
{
lean_object* x_54; lean_object* x_55; lean_object* x_56; 
if (lean_is_exclusive(x_53)) {
 lean_ctor_release(x_53, 0);
 x_54 = x_53;
} else {
 lean_dec_ref(x_53);
 x_54 = lean_box(0);
}
x_55 = l_Tools_SwirlVerifyMain_main___boxed__const__1;
if (lean_is_scalar(x_54)) {
 x_56 = lean_alloc_ctor(0, 1, 0);
} else {
 x_56 = x_54;
}
lean_ctor_set(x_56, 0, x_55);
return x_56;
}
else
{
lean_object* x_57; lean_object* x_58; lean_object* x_59; 
x_57 = lean_ctor_get(x_53, 0);
lean_inc(x_57);
if (lean_is_exclusive(x_53)) {
 lean_ctor_release(x_53, 0);
 x_58 = x_53;
} else {
 lean_dec_ref(x_53);
 x_58 = lean_box(0);
}
if (lean_is_scalar(x_58)) {
 x_59 = lean_alloc_ctor(1, 1, 0);
} else {
 x_59 = x_58;
}
lean_ctor_set(x_59, 0, x_57);
return x_59;
}
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; 
lean_dec(x_45);
x_60 = lean_ctor_get(x_46, 0);
lean_inc(x_60);
lean_dec_ref(x_46);
x_61 = lean_ctor_get(x_60, 1);
lean_inc(x_61);
x_62 = lean_ctor_get(x_60, 0);
lean_inc(x_62);
lean_dec(x_60);
x_63 = lean_ctor_get(x_61, 0);
lean_inc(x_63);
x_64 = lean_ctor_get(x_61, 1);
lean_inc(x_64);
lean_dec(x_61);
x_65 = l_Swirl_Protocol_Noninteractive_verifyBabyBearPoseidon2(x_62, x_63, x_64);
if (lean_obj_tag(x_65) == 0)
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; 
x_66 = lean_ctor_get(x_65, 0);
lean_inc(x_66);
lean_dec_ref(x_65);
x_67 = l_Tools_SwirlVerifyMain_main___closed__2;
lean_inc(x_66);
x_68 = l_Swirl_Protocol_Noninteractive_MonoError_toString(x_66);
x_69 = lean_string_append(x_67, x_68);
lean_dec_ref(x_68);
x_70 = l_IO_FS_Stream_putStrLn(x_3, x_69);
if (lean_obj_tag(x_70) == 0)
{
lean_object* x_71; uint32_t x_72; lean_object* x_73; lean_object* x_74; 
if (lean_is_exclusive(x_70)) {
 lean_ctor_release(x_70, 0);
 x_71 = x_70;
} else {
 lean_dec_ref(x_70);
 x_71 = lean_box(0);
}
x_72 = l_Swirl_Protocol_Noninteractive_exitCode(x_66);
lean_dec(x_66);
x_73 = lean_box_uint32(x_72);
if (lean_is_scalar(x_71)) {
 x_74 = lean_alloc_ctor(0, 1, 0);
} else {
 x_74 = x_71;
}
lean_ctor_set(x_74, 0, x_73);
return x_74;
}
else
{
lean_object* x_75; lean_object* x_76; lean_object* x_77; 
lean_dec(x_66);
x_75 = lean_ctor_get(x_70, 0);
lean_inc(x_75);
if (lean_is_exclusive(x_70)) {
 lean_ctor_release(x_70, 0);
 x_76 = x_70;
} else {
 lean_dec_ref(x_70);
 x_76 = lean_box(0);
}
if (lean_is_scalar(x_76)) {
 x_77 = lean_alloc_ctor(1, 1, 0);
} else {
 x_77 = x_76;
}
lean_ctor_set(x_77, 0, x_75);
return x_77;
}
}
else
{
lean_object* x_78; lean_object* x_79; 
lean_dec_ref(x_65);
lean_dec_ref(x_3);
x_78 = l_Tools_SwirlVerifyMain_main___boxed__const__2;
x_79 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_79, 0, x_78);
return x_79;
}
}
}
}
else
{
uint8_t x_80; 
lean_dec_ref(x_3);
x_80 = !lean_is_exclusive(x_4);
if (x_80 == 0)
{
return x_4;
}
else
{
lean_object* x_81; lean_object* x_82; 
x_81 = lean_ctor_get(x_4, 0);
lean_inc(x_81);
lean_dec(x_4);
x_82 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_82, 0, x_81);
return x_82;
}
}
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlVerifyMain_main___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_Tools_SwirlVerifyMain_main();
return x_2;
}
}
LEAN_EXPORT lean_object* _lean_main() {
_start:
{
lean_object* x_2; 
x_2 = l_Tools_SwirlVerifyMain_main();
return x_2;
}
}
LEAN_EXPORT lean_object* l_main___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = _lean_main();
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Swirl_Protocol_Noninteractive_VerifierBabyBearPoseidon2(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Tools_SwirlVerifyMain(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Swirl_Protocol_Noninteractive_VerifierBabyBearPoseidon2(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
l_Tools_SwirlVerifyMain_main___closed__0 = _init_l_Tools_SwirlVerifyMain_main___closed__0();
lean_mark_persistent(l_Tools_SwirlVerifyMain_main___closed__0);
l_Tools_SwirlVerifyMain_main___closed__1 = _init_l_Tools_SwirlVerifyMain_main___closed__1();
lean_mark_persistent(l_Tools_SwirlVerifyMain_main___closed__1);
l_Tools_SwirlVerifyMain_main___closed__2 = _init_l_Tools_SwirlVerifyMain_main___closed__2();
lean_mark_persistent(l_Tools_SwirlVerifyMain_main___closed__2);
l_Tools_SwirlVerifyMain_main___boxed__const__1 = _init_l_Tools_SwirlVerifyMain_main___boxed__const__1();
lean_mark_persistent(l_Tools_SwirlVerifyMain_main___boxed__const__1);
l_Tools_SwirlVerifyMain_main___boxed__const__2 = _init_l_Tools_SwirlVerifyMain_main___boxed__const__2();
lean_mark_persistent(l_Tools_SwirlVerifyMain_main___boxed__const__2);
return lean_io_result_mk_ok(lean_box(0));
}
char ** lean_setup_args(int argc, char ** argv);
void lean_initialize_runtime_module();

  #if defined(WIN32) || defined(_WIN32)
  #include <windows.h>
  #endif

  int main(int argc, char ** argv) {
  #if defined(WIN32) || defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS);
  SetConsoleOutputCP(CP_UTF8);
  #endif
  lean_object* in; lean_object* res;
argv = lean_setup_args(argc, argv);
lean_initialize_runtime_module();
lean_set_panic_messages(false);
res = initialize_Tools_SwirlVerifyMain(1 /* builtin */);
lean_set_panic_messages(true);
lean_io_mark_end_initialization();
if (lean_io_result_is_ok(res)) {
lean_dec_ref(res);
lean_init_task_manager();
res = _lean_main();
}
lean_finalize_task_manager();
if (lean_io_result_is_ok(res)) {
  int ret = lean_unbox_uint32(lean_io_result_get_value(res));
  lean_dec_ref(res);
  return ret;
} else {
  lean_io_result_show_error(res);
  lean_dec_ref(res);
  return 1;
}
}
#ifdef __cplusplus
}
#endif
