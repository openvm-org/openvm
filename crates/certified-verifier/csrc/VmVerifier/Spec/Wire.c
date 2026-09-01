// Lean compiler output
// Module: VmVerifier.Spec.Wire
// Imports: public import Init public import Swirl.Protocol.Noninteractive.Wire.RawToTyped public import VmVerifier.Spec.Types
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
lean_object* l_Fundamentals_BabyBear_FBB_Raw_ofNat(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readBlobAt(lean_object*, lean_object*);
uint8_t lean_byte_array_fget(lean_object*, lean_object*);
lean_object* lean_uint32_to_nat(uint32_t);
LEAN_EXPORT uint8_t l_VmVerifier_Spec_Wire_readU32LE___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readUserPvsProofM(lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB(lean_object*);
uint32_t lean_uint8_to_uint32(uint8_t);
static lean_object* l_VmVerifier_Spec_Wire_userPvsMagic___closed__0;
lean_object* l_ByteArray_extract(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(uint8_t, uint8_t, uint8_t, uint8_t);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readBaseline(lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest(lean_object*);
lean_object* l_Array_ofFn___redArg(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
static lean_object* l_VmVerifier_Spec_Wire_ensureEnd___closed__0;
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readU32LE___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readUserPvsProof(lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_sliceBytes___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readU32LE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readDigest___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readDigest(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_ensureEnd(lean_object*);
lean_object* l_outOfBounds___redArg(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_parseFiveBlobs(lean_object*);
extern uint8_t l_instInhabitedUInt8;
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_parseFiveBlobs___boxed(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_userPvsMagic;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
static lean_object* l_VmVerifier_Spec_Wire_baselineMagic___closed__0;
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_baselineMagic;
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readBaselineM(lean_object*);
uint32_t lean_uint32_lor(uint32_t, uint32_t);
uint32_t lean_uint32_shift_left(uint32_t, uint32_t);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readUserPvsProofM___lam__0(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readVkCommit(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readU32LE___boxed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readDigest___lam__0(lean_object*, lean_object*);
lean_object* lean_byte_array_size(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readBlobAt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_sliceBytes(lean_object*, lean_object*, lean_object*);
static lean_object* _init_l_VmVerifier_Spec_Wire_baselineMagic___closed__0() {
_start:
{
uint8_t x_1; uint8_t x_2; uint8_t x_3; uint8_t x_4; lean_object* x_5; 
x_1 = 76;
x_2 = 66;
x_3 = 77;
x_4 = 86;
x_5 = l_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_l_VmVerifier_Spec_Wire_baselineMagic() {
_start:
{
lean_object* x_1; 
x_1 = l_VmVerifier_Spec_Wire_baselineMagic___closed__0;
return x_1;
}
}
static lean_object* _init_l_VmVerifier_Spec_Wire_userPvsMagic___closed__0() {
_start:
{
uint8_t x_1; uint8_t x_2; uint8_t x_3; uint8_t x_4; lean_object* x_5; 
x_1 = 83;
x_2 = 86;
x_3 = 80;
x_4 = 85;
x_5 = l_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_l_VmVerifier_Spec_Wire_userPvsMagic() {
_start:
{
lean_object* x_1; 
x_1 = l_VmVerifier_Spec_Wire_userPvsMagic___closed__0;
return x_1;
}
}
static lean_object* _init_l_VmVerifier_Spec_Wire_ensureEnd___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("trailing bytes: ", 16, 16);
return x_1;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_ensureEnd(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_byte_array_size(x_2);
x_5 = lean_nat_dec_eq(x_3, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = l_VmVerifier_Spec_Wire_ensureEnd___closed__0;
x_7 = lean_nat_sub(x_4, x_3);
lean_dec(x_4);
x_8 = l_Nat_reprFast(x_7);
x_9 = lean_string_append(x_6, x_8);
lean_dec_ref(x_8);
lean_inc(x_3);
x_10 = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(x_10, 0, x_3);
lean_ctor_set(x_10, 1, x_9);
x_11 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_1);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; 
lean_dec(x_4);
x_12 = lean_box(0);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_1);
return x_13;
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readDigest___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint32_t x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_array_fget_borrowed(x_1, x_2);
x_4 = lean_unbox_uint32(x_3);
x_5 = lean_uint32_to_nat(x_4);
x_6 = l_Fundamentals_BabyBear_FBB_Raw_ofNat(x_5);
lean_dec(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readDigest(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest(x_1);
if (lean_obj_tag(x_2) == 0)
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_alloc_closure((void*)(l_VmVerifier_Spec_Wire_readDigest___lam__0___boxed), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lean_unsigned_to_nat(8u);
x_7 = l_Array_ofFn___redArg(x_6, x_5);
lean_ctor_set(x_2, 0, x_7);
return x_2;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_8 = lean_ctor_get(x_2, 0);
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_2);
x_10 = lean_alloc_closure((void*)(l_VmVerifier_Spec_Wire_readDigest___lam__0___boxed), 2, 1);
lean_closure_set(x_10, 0, x_8);
x_11 = lean_unsigned_to_nat(8u);
x_12 = l_Array_ofFn___redArg(x_11, x_10);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_9);
return x_13;
}
}
else
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_2);
if (x_14 == 0)
{
return x_2;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_15 = lean_ctor_get(x_2, 0);
x_16 = lean_ctor_get(x_2, 1);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_2);
x_17 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
return x_17;
}
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readDigest___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_VmVerifier_Spec_Wire_readDigest___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readVkCommit(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_VmVerifier_Spec_Wire_readDigest(x_1);
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = l_VmVerifier_Spec_Wire_readDigest(x_4);
if (lean_obj_tag(x_5) == 0)
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_5, 0);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
lean_ctor_set(x_5, 0, x_8);
return x_5;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_5, 0);
x_10 = lean_ctor_get(x_5, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_5);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_3);
lean_ctor_set(x_11, 1, x_9);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_10);
return x_12;
}
}
else
{
uint8_t x_13; 
lean_dec(x_3);
x_13 = !lean_is_exclusive(x_5);
if (x_13 == 0)
{
return x_5;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_ctor_get(x_5, 0);
x_15 = lean_ctor_get(x_5, 1);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_5);
x_16 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
return x_16;
}
}
}
else
{
uint8_t x_17; 
x_17 = !lean_is_exclusive(x_2);
if (x_17 == 0)
{
return x_2;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_18 = lean_ctor_get(x_2, 0);
x_19 = lean_ctor_get(x_2, 1);
lean_inc(x_19);
lean_inc(x_18);
lean_dec(x_2);
x_20 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_20, 0, x_18);
lean_ctor_set(x_20, 1, x_19);
return x_20;
}
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readBaselineM(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = l_VmVerifier_Spec_Wire_baselineMagic;
x_3 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(x_2, x_1);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = l_VmVerifier_Spec_Wire_readDigest(x_4);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec_ref(x_5);
x_8 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(x_7);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_8, 1);
lean_inc(x_10);
lean_dec_ref(x_8);
x_11 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(x_10);
if (lean_obj_tag(x_11) == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_11, 1);
lean_inc(x_13);
lean_dec_ref(x_11);
x_14 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(x_13);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
x_16 = lean_ctor_get(x_14, 1);
lean_inc(x_16);
lean_dec_ref(x_14);
x_17 = l_VmVerifier_Spec_Wire_readVkCommit(x_16);
if (lean_obj_tag(x_17) == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
x_19 = lean_ctor_get(x_17, 1);
lean_inc(x_19);
lean_dec_ref(x_17);
x_20 = l_VmVerifier_Spec_Wire_readVkCommit(x_19);
if (lean_obj_tag(x_20) == 0)
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
x_22 = lean_ctor_get(x_20, 1);
lean_inc(x_22);
lean_dec_ref(x_20);
x_23 = l_VmVerifier_Spec_Wire_readVkCommit(x_22);
if (lean_obj_tag(x_23) == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_24 = lean_ctor_get(x_23, 0);
lean_inc(x_24);
x_25 = lean_ctor_get(x_23, 1);
lean_inc(x_25);
lean_dec_ref(x_23);
x_26 = l_VmVerifier_Spec_Wire_readVkCommit(x_25);
if (lean_obj_tag(x_26) == 0)
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
x_28 = lean_ctor_get(x_26, 1);
lean_inc(x_28);
lean_dec_ref(x_26);
x_29 = l_VmVerifier_Spec_Wire_ensureEnd(x_28);
if (lean_obj_tag(x_29) == 0)
{
uint8_t x_30; 
x_30 = !lean_is_exclusive(x_29);
if (x_30 == 0)
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; 
x_31 = lean_ctor_get(x_29, 0);
lean_dec(x_31);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_9);
lean_ctor_set(x_32, 1, x_12);
x_33 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_33, 0, x_6);
lean_ctor_set(x_33, 1, x_32);
lean_ctor_set(x_33, 2, x_15);
lean_ctor_set(x_33, 3, x_18);
lean_ctor_set(x_33, 4, x_21);
lean_ctor_set(x_33, 5, x_24);
lean_ctor_set(x_33, 6, x_27);
lean_ctor_set(x_29, 0, x_33);
return x_29;
}
else
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_34 = lean_ctor_get(x_29, 1);
lean_inc(x_34);
lean_dec(x_29);
x_35 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_35, 0, x_9);
lean_ctor_set(x_35, 1, x_12);
x_36 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_36, 0, x_6);
lean_ctor_set(x_36, 1, x_35);
lean_ctor_set(x_36, 2, x_15);
lean_ctor_set(x_36, 3, x_18);
lean_ctor_set(x_36, 4, x_21);
lean_ctor_set(x_36, 5, x_24);
lean_ctor_set(x_36, 6, x_27);
x_37 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_37, 0, x_36);
lean_ctor_set(x_37, 1, x_34);
return x_37;
}
}
else
{
uint8_t x_38; 
lean_dec(x_27);
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_18);
lean_dec(x_15);
lean_dec(x_12);
lean_dec(x_9);
lean_dec(x_6);
x_38 = !lean_is_exclusive(x_29);
if (x_38 == 0)
{
return x_29;
}
else
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; 
x_39 = lean_ctor_get(x_29, 0);
x_40 = lean_ctor_get(x_29, 1);
lean_inc(x_40);
lean_inc(x_39);
lean_dec(x_29);
x_41 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_41, 0, x_39);
lean_ctor_set(x_41, 1, x_40);
return x_41;
}
}
}
else
{
uint8_t x_42; 
lean_dec(x_24);
lean_dec(x_21);
lean_dec(x_18);
lean_dec(x_15);
lean_dec(x_12);
lean_dec(x_9);
lean_dec(x_6);
x_42 = !lean_is_exclusive(x_26);
if (x_42 == 0)
{
return x_26;
}
else
{
lean_object* x_43; lean_object* x_44; lean_object* x_45; 
x_43 = lean_ctor_get(x_26, 0);
x_44 = lean_ctor_get(x_26, 1);
lean_inc(x_44);
lean_inc(x_43);
lean_dec(x_26);
x_45 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_45, 0, x_43);
lean_ctor_set(x_45, 1, x_44);
return x_45;
}
}
}
else
{
uint8_t x_46; 
lean_dec(x_21);
lean_dec(x_18);
lean_dec(x_15);
lean_dec(x_12);
lean_dec(x_9);
lean_dec(x_6);
x_46 = !lean_is_exclusive(x_23);
if (x_46 == 0)
{
return x_23;
}
else
{
lean_object* x_47; lean_object* x_48; lean_object* x_49; 
x_47 = lean_ctor_get(x_23, 0);
x_48 = lean_ctor_get(x_23, 1);
lean_inc(x_48);
lean_inc(x_47);
lean_dec(x_23);
x_49 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_49, 0, x_47);
lean_ctor_set(x_49, 1, x_48);
return x_49;
}
}
}
else
{
uint8_t x_50; 
lean_dec(x_18);
lean_dec(x_15);
lean_dec(x_12);
lean_dec(x_9);
lean_dec(x_6);
x_50 = !lean_is_exclusive(x_20);
if (x_50 == 0)
{
return x_20;
}
else
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_51 = lean_ctor_get(x_20, 0);
x_52 = lean_ctor_get(x_20, 1);
lean_inc(x_52);
lean_inc(x_51);
lean_dec(x_20);
x_53 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_53, 0, x_51);
lean_ctor_set(x_53, 1, x_52);
return x_53;
}
}
}
else
{
uint8_t x_54; 
lean_dec(x_15);
lean_dec(x_12);
lean_dec(x_9);
lean_dec(x_6);
x_54 = !lean_is_exclusive(x_17);
if (x_54 == 0)
{
return x_17;
}
else
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; 
x_55 = lean_ctor_get(x_17, 0);
x_56 = lean_ctor_get(x_17, 1);
lean_inc(x_56);
lean_inc(x_55);
lean_dec(x_17);
x_57 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_57, 0, x_55);
lean_ctor_set(x_57, 1, x_56);
return x_57;
}
}
}
else
{
uint8_t x_58; 
lean_dec(x_12);
lean_dec(x_9);
lean_dec(x_6);
x_58 = !lean_is_exclusive(x_14);
if (x_58 == 0)
{
return x_14;
}
else
{
lean_object* x_59; lean_object* x_60; lean_object* x_61; 
x_59 = lean_ctor_get(x_14, 0);
x_60 = lean_ctor_get(x_14, 1);
lean_inc(x_60);
lean_inc(x_59);
lean_dec(x_14);
x_61 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_61, 0, x_59);
lean_ctor_set(x_61, 1, x_60);
return x_61;
}
}
}
else
{
uint8_t x_62; 
lean_dec(x_9);
lean_dec(x_6);
x_62 = !lean_is_exclusive(x_11);
if (x_62 == 0)
{
return x_11;
}
else
{
lean_object* x_63; lean_object* x_64; lean_object* x_65; 
x_63 = lean_ctor_get(x_11, 0);
x_64 = lean_ctor_get(x_11, 1);
lean_inc(x_64);
lean_inc(x_63);
lean_dec(x_11);
x_65 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_65, 0, x_63);
lean_ctor_set(x_65, 1, x_64);
return x_65;
}
}
}
else
{
uint8_t x_66; 
lean_dec(x_6);
x_66 = !lean_is_exclusive(x_8);
if (x_66 == 0)
{
return x_8;
}
else
{
lean_object* x_67; lean_object* x_68; lean_object* x_69; 
x_67 = lean_ctor_get(x_8, 0);
x_68 = lean_ctor_get(x_8, 1);
lean_inc(x_68);
lean_inc(x_67);
lean_dec(x_8);
x_69 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_69, 0, x_67);
lean_ctor_set(x_69, 1, x_68);
return x_69;
}
}
}
else
{
uint8_t x_70; 
x_70 = !lean_is_exclusive(x_5);
if (x_70 == 0)
{
return x_5;
}
else
{
lean_object* x_71; lean_object* x_72; lean_object* x_73; 
x_71 = lean_ctor_get(x_5, 0);
x_72 = lean_ctor_get(x_5, 1);
lean_inc(x_72);
lean_inc(x_71);
lean_dec(x_5);
x_73 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_73, 0, x_71);
lean_ctor_set(x_73, 1, x_72);
return x_73;
}
}
}
else
{
uint8_t x_74; 
x_74 = !lean_is_exclusive(x_3);
if (x_74 == 0)
{
return x_3;
}
else
{
lean_object* x_75; lean_object* x_76; lean_object* x_77; 
x_75 = lean_ctor_get(x_3, 0);
x_76 = lean_ctor_get(x_3, 1);
lean_inc(x_76);
lean_inc(x_75);
lean_dec(x_3);
x_77 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_77, 0, x_75);
lean_ctor_set(x_77, 1, x_76);
return x_77;
}
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readBaseline(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(l_VmVerifier_Spec_Wire_readBaselineM), 1, 0);
x_3 = l_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readUserPvsProofM___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB(x_1);
if (lean_obj_tag(x_2) == 0)
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; uint32_t x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_unbox_uint32(x_4);
lean_dec(x_4);
x_6 = lean_uint32_to_nat(x_5);
x_7 = l_Fundamentals_BabyBear_FBB_Raw_ofNat(x_6);
lean_dec(x_6);
lean_ctor_set(x_2, 0, x_7);
return x_2;
}
else
{
lean_object* x_8; lean_object* x_9; uint32_t x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_8 = lean_ctor_get(x_2, 0);
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_2);
x_10 = lean_unbox_uint32(x_8);
lean_dec(x_8);
x_11 = lean_uint32_to_nat(x_10);
x_12 = l_Fundamentals_BabyBear_FBB_Raw_ofNat(x_11);
lean_dec(x_11);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_9);
return x_13;
}
}
else
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_2);
if (x_14 == 0)
{
return x_2;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_15 = lean_ctor_get(x_2, 0);
x_16 = lean_ctor_get(x_2, 1);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_2);
x_17 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
return x_17;
}
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readUserPvsProofM(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = l_VmVerifier_Spec_Wire_userPvsMagic;
x_3 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(x_2, x_1);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_alloc_closure((void*)(l_VmVerifier_Spec_Wire_readDigest), 1, 0);
x_6 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(x_5, x_4);
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lean_alloc_closure((void*)(l_VmVerifier_Spec_Wire_readUserPvsProofM___lam__0), 1, 0);
x_10 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(x_9, x_8);
if (lean_obj_tag(x_10) == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
x_12 = lean_ctor_get(x_10, 1);
lean_inc(x_12);
lean_dec_ref(x_10);
x_13 = l_VmVerifier_Spec_Wire_readDigest(x_12);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_13, 1);
lean_inc(x_15);
lean_dec_ref(x_13);
x_16 = l_VmVerifier_Spec_Wire_ensureEnd(x_15);
if (lean_obj_tag(x_16) == 0)
{
uint8_t x_17; 
x_17 = !lean_is_exclusive(x_16);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_18 = lean_ctor_get(x_16, 0);
lean_dec(x_18);
x_19 = lean_array_to_list(x_7);
x_20 = lean_array_to_list(x_11);
x_21 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
lean_ctor_set(x_21, 2, x_14);
lean_ctor_set(x_16, 0, x_21);
return x_16;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_22 = lean_ctor_get(x_16, 1);
lean_inc(x_22);
lean_dec(x_16);
x_23 = lean_array_to_list(x_7);
x_24 = lean_array_to_list(x_11);
x_25 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_25, 0, x_23);
lean_ctor_set(x_25, 1, x_24);
lean_ctor_set(x_25, 2, x_14);
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_22);
return x_26;
}
}
else
{
uint8_t x_27; 
lean_dec(x_14);
lean_dec(x_11);
lean_dec(x_7);
x_27 = !lean_is_exclusive(x_16);
if (x_27 == 0)
{
return x_16;
}
else
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_28 = lean_ctor_get(x_16, 0);
x_29 = lean_ctor_get(x_16, 1);
lean_inc(x_29);
lean_inc(x_28);
lean_dec(x_16);
x_30 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_30, 0, x_28);
lean_ctor_set(x_30, 1, x_29);
return x_30;
}
}
}
else
{
uint8_t x_31; 
lean_dec(x_11);
lean_dec(x_7);
x_31 = !lean_is_exclusive(x_13);
if (x_31 == 0)
{
return x_13;
}
else
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_32 = lean_ctor_get(x_13, 0);
x_33 = lean_ctor_get(x_13, 1);
lean_inc(x_33);
lean_inc(x_32);
lean_dec(x_13);
x_34 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_33);
return x_34;
}
}
}
else
{
uint8_t x_35; 
lean_dec(x_7);
x_35 = !lean_is_exclusive(x_10);
if (x_35 == 0)
{
return x_10;
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_36 = lean_ctor_get(x_10, 0);
x_37 = lean_ctor_get(x_10, 1);
lean_inc(x_37);
lean_inc(x_36);
lean_dec(x_10);
x_38 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_38, 0, x_36);
lean_ctor_set(x_38, 1, x_37);
return x_38;
}
}
}
else
{
uint8_t x_39; 
x_39 = !lean_is_exclusive(x_6);
if (x_39 == 0)
{
return x_6;
}
else
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_40 = lean_ctor_get(x_6, 0);
x_41 = lean_ctor_get(x_6, 1);
lean_inc(x_41);
lean_inc(x_40);
lean_dec(x_6);
x_42 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_42, 0, x_40);
lean_ctor_set(x_42, 1, x_41);
return x_42;
}
}
}
else
{
uint8_t x_43; 
x_43 = !lean_is_exclusive(x_3);
if (x_43 == 0)
{
return x_3;
}
else
{
lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_44 = lean_ctor_get(x_3, 0);
x_45 = lean_ctor_get(x_3, 1);
lean_inc(x_45);
lean_inc(x_44);
lean_dec(x_3);
x_46 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_46, 0, x_44);
lean_ctor_set(x_46, 1, x_45);
return x_46;
}
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readUserPvsProof(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_closure((void*)(l_VmVerifier_Spec_Wire_readUserPvsProofM), 1, 0);
x_3 = l_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT uint8_t l_VmVerifier_Spec_Wire_readU32LE___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_byte_array_size(x_1);
x_4 = lean_nat_dec_lt(x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readU32LE(lean_object* x_1, lean_object* x_2) {
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
x_59 = l_VmVerifier_Spec_Wire_readU32LE___lam__0(x_1, x_2);
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
x_32 = l_VmVerifier_Spec_Wire_readU32LE___lam__0(x_1, x_31);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; uint8_t x_35; 
lean_dec(x_31);
x_33 = lean_box(x_25);
x_34 = l_outOfBounds___redArg(x_33);
x_35 = lean_unbox(x_34);
x_3 = x_26;
x_4 = x_29;
x_5 = x_27;
x_6 = x_35;
goto block_19;
}
else
{
uint8_t x_36; 
x_36 = lean_byte_array_fget(x_1, x_31);
lean_dec(x_31);
x_3 = x_26;
x_4 = x_29;
x_5 = x_27;
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
x_43 = l_VmVerifier_Spec_Wire_readU32LE___lam__0(x_1, x_42);
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
x_53 = l_VmVerifier_Spec_Wire_readU32LE___lam__0(x_1, x_52);
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
x_9 = lean_uint32_shift_left(x_5, x_8);
x_10 = lean_uint32_lor(x_3, x_9);
x_11 = 16;
x_12 = lean_uint32_shift_left(x_4, x_11);
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
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readU32LE___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = l_VmVerifier_Spec_Wire_readU32LE___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readU32LE___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_VmVerifier_Spec_Wire_readU32LE(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_sliceBytes(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
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
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_sliceBytes___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = l_VmVerifier_Spec_Wire_sliceBytes(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readBlobAt(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_VmVerifier_Spec_Wire_readU32LE(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint32_t x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_unsigned_to_nat(4u);
x_7 = lean_nat_add(x_2, x_6);
x_8 = lean_unbox_uint32(x_5);
lean_dec(x_5);
x_9 = lean_uint32_to_nat(x_8);
lean_inc(x_7);
x_10 = l_VmVerifier_Spec_Wire_sliceBytes(x_1, x_7, x_9);
if (lean_obj_tag(x_10) == 0)
{
lean_object* x_11; 
lean_dec(x_9);
lean_dec(x_7);
x_11 = lean_box(0);
return x_11;
}
else
{
uint8_t x_12; 
x_12 = !lean_is_exclusive(x_10);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_10, 0);
x_14 = lean_nat_add(x_7, x_9);
lean_dec(x_9);
lean_dec(x_7);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
lean_ctor_set(x_10, 0, x_15);
return x_10;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_16 = lean_ctor_get(x_10, 0);
lean_inc(x_16);
lean_dec(x_10);
x_17 = lean_nat_add(x_7, x_9);
lean_dec(x_9);
lean_dec(x_7);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_readBlobAt___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_parseFiveBlobs(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec(x_5);
x_8 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_7);
lean_dec(x_7);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_9; 
lean_dec(x_6);
x_9 = lean_box(0);
return x_9;
}
else
{
lean_object* x_10; uint8_t x_11; 
x_10 = lean_ctor_get(x_8, 0);
lean_inc(x_10);
lean_dec_ref(x_8);
x_11 = !lean_is_exclusive(x_10);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_10, 0);
x_13 = lean_ctor_get(x_10, 1);
x_14 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_13);
lean_dec(x_13);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; 
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_15 = lean_box(0);
return x_15;
}
else
{
lean_object* x_16; uint8_t x_17; 
x_16 = lean_ctor_get(x_14, 0);
lean_inc(x_16);
lean_dec_ref(x_14);
x_17 = !lean_is_exclusive(x_16);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_18 = lean_ctor_get(x_16, 0);
x_19 = lean_ctor_get(x_16, 1);
x_20 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_19);
lean_dec(x_19);
if (lean_obj_tag(x_20) == 0)
{
lean_object* x_21; 
lean_free_object(x_16);
lean_dec(x_18);
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_21 = lean_box(0);
return x_21;
}
else
{
lean_object* x_22; uint8_t x_23; 
x_22 = lean_ctor_get(x_20, 0);
lean_inc(x_22);
lean_dec_ref(x_20);
x_23 = !lean_is_exclusive(x_22);
if (x_23 == 0)
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_24 = lean_ctor_get(x_22, 0);
x_25 = lean_ctor_get(x_22, 1);
x_26 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_25);
lean_dec(x_25);
if (lean_obj_tag(x_26) == 0)
{
lean_object* x_27; 
lean_free_object(x_22);
lean_dec(x_24);
lean_free_object(x_16);
lean_dec(x_18);
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_27 = lean_box(0);
return x_27;
}
else
{
uint8_t x_28; 
x_28 = !lean_is_exclusive(x_26);
if (x_28 == 0)
{
lean_object* x_29; uint8_t x_30; 
x_29 = lean_ctor_get(x_26, 0);
x_30 = !lean_is_exclusive(x_29);
if (x_30 == 0)
{
lean_object* x_31; lean_object* x_32; lean_object* x_33; uint8_t x_34; 
x_31 = lean_ctor_get(x_29, 0);
x_32 = lean_ctor_get(x_29, 1);
x_33 = lean_byte_array_size(x_1);
x_34 = lean_nat_dec_eq(x_32, x_33);
lean_dec(x_33);
lean_dec(x_32);
if (x_34 == 0)
{
lean_object* x_35; 
lean_free_object(x_29);
lean_dec(x_31);
lean_free_object(x_26);
lean_free_object(x_22);
lean_dec(x_24);
lean_free_object(x_16);
lean_dec(x_18);
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_35 = lean_box(0);
return x_35;
}
else
{
lean_ctor_set(x_29, 1, x_31);
lean_ctor_set(x_29, 0, x_24);
lean_ctor_set(x_22, 1, x_29);
lean_ctor_set(x_22, 0, x_18);
lean_ctor_set(x_16, 1, x_22);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_10, 1, x_16);
lean_ctor_set(x_10, 0, x_6);
lean_ctor_set(x_26, 0, x_10);
return x_26;
}
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; uint8_t x_39; 
x_36 = lean_ctor_get(x_29, 0);
x_37 = lean_ctor_get(x_29, 1);
lean_inc(x_37);
lean_inc(x_36);
lean_dec(x_29);
x_38 = lean_byte_array_size(x_1);
x_39 = lean_nat_dec_eq(x_37, x_38);
lean_dec(x_38);
lean_dec(x_37);
if (x_39 == 0)
{
lean_object* x_40; 
lean_dec(x_36);
lean_free_object(x_26);
lean_free_object(x_22);
lean_dec(x_24);
lean_free_object(x_16);
lean_dec(x_18);
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_40 = lean_box(0);
return x_40;
}
else
{
lean_object* x_41; 
x_41 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_41, 0, x_24);
lean_ctor_set(x_41, 1, x_36);
lean_ctor_set(x_22, 1, x_41);
lean_ctor_set(x_22, 0, x_18);
lean_ctor_set(x_16, 1, x_22);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_10, 1, x_16);
lean_ctor_set(x_10, 0, x_6);
lean_ctor_set(x_26, 0, x_10);
return x_26;
}
}
}
else
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; uint8_t x_47; 
x_42 = lean_ctor_get(x_26, 0);
lean_inc(x_42);
lean_dec(x_26);
x_43 = lean_ctor_get(x_42, 0);
lean_inc(x_43);
x_44 = lean_ctor_get(x_42, 1);
lean_inc(x_44);
if (lean_is_exclusive(x_42)) {
 lean_ctor_release(x_42, 0);
 lean_ctor_release(x_42, 1);
 x_45 = x_42;
} else {
 lean_dec_ref(x_42);
 x_45 = lean_box(0);
}
x_46 = lean_byte_array_size(x_1);
x_47 = lean_nat_dec_eq(x_44, x_46);
lean_dec(x_46);
lean_dec(x_44);
if (x_47 == 0)
{
lean_object* x_48; 
lean_dec(x_45);
lean_dec(x_43);
lean_free_object(x_22);
lean_dec(x_24);
lean_free_object(x_16);
lean_dec(x_18);
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_48 = lean_box(0);
return x_48;
}
else
{
lean_object* x_49; lean_object* x_50; 
if (lean_is_scalar(x_45)) {
 x_49 = lean_alloc_ctor(0, 2, 0);
} else {
 x_49 = x_45;
}
lean_ctor_set(x_49, 0, x_24);
lean_ctor_set(x_49, 1, x_43);
lean_ctor_set(x_22, 1, x_49);
lean_ctor_set(x_22, 0, x_18);
lean_ctor_set(x_16, 1, x_22);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_10, 1, x_16);
lean_ctor_set(x_10, 0, x_6);
x_50 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_50, 0, x_10);
return x_50;
}
}
}
}
else
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_51 = lean_ctor_get(x_22, 0);
x_52 = lean_ctor_get(x_22, 1);
lean_inc(x_52);
lean_inc(x_51);
lean_dec(x_22);
x_53 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_52);
lean_dec(x_52);
if (lean_obj_tag(x_53) == 0)
{
lean_object* x_54; 
lean_dec(x_51);
lean_free_object(x_16);
lean_dec(x_18);
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_54 = lean_box(0);
return x_54;
}
else
{
lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; uint8_t x_61; 
x_55 = lean_ctor_get(x_53, 0);
lean_inc(x_55);
if (lean_is_exclusive(x_53)) {
 lean_ctor_release(x_53, 0);
 x_56 = x_53;
} else {
 lean_dec_ref(x_53);
 x_56 = lean_box(0);
}
x_57 = lean_ctor_get(x_55, 0);
lean_inc(x_57);
x_58 = lean_ctor_get(x_55, 1);
lean_inc(x_58);
if (lean_is_exclusive(x_55)) {
 lean_ctor_release(x_55, 0);
 lean_ctor_release(x_55, 1);
 x_59 = x_55;
} else {
 lean_dec_ref(x_55);
 x_59 = lean_box(0);
}
x_60 = lean_byte_array_size(x_1);
x_61 = lean_nat_dec_eq(x_58, x_60);
lean_dec(x_60);
lean_dec(x_58);
if (x_61 == 0)
{
lean_object* x_62; 
lean_dec(x_59);
lean_dec(x_57);
lean_dec(x_56);
lean_dec(x_51);
lean_free_object(x_16);
lean_dec(x_18);
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_62 = lean_box(0);
return x_62;
}
else
{
lean_object* x_63; lean_object* x_64; lean_object* x_65; 
if (lean_is_scalar(x_59)) {
 x_63 = lean_alloc_ctor(0, 2, 0);
} else {
 x_63 = x_59;
}
lean_ctor_set(x_63, 0, x_51);
lean_ctor_set(x_63, 1, x_57);
x_64 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_64, 0, x_18);
lean_ctor_set(x_64, 1, x_63);
lean_ctor_set(x_16, 1, x_64);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_10, 1, x_16);
lean_ctor_set(x_10, 0, x_6);
if (lean_is_scalar(x_56)) {
 x_65 = lean_alloc_ctor(1, 1, 0);
} else {
 x_65 = x_56;
}
lean_ctor_set(x_65, 0, x_10);
return x_65;
}
}
}
}
}
else
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; 
x_66 = lean_ctor_get(x_16, 0);
x_67 = lean_ctor_get(x_16, 1);
lean_inc(x_67);
lean_inc(x_66);
lean_dec(x_16);
x_68 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_67);
lean_dec(x_67);
if (lean_obj_tag(x_68) == 0)
{
lean_object* x_69; 
lean_dec(x_66);
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_69 = lean_box(0);
return x_69;
}
else
{
lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; 
x_70 = lean_ctor_get(x_68, 0);
lean_inc(x_70);
lean_dec_ref(x_68);
x_71 = lean_ctor_get(x_70, 0);
lean_inc(x_71);
x_72 = lean_ctor_get(x_70, 1);
lean_inc(x_72);
if (lean_is_exclusive(x_70)) {
 lean_ctor_release(x_70, 0);
 lean_ctor_release(x_70, 1);
 x_73 = x_70;
} else {
 lean_dec_ref(x_70);
 x_73 = lean_box(0);
}
x_74 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_72);
lean_dec(x_72);
if (lean_obj_tag(x_74) == 0)
{
lean_object* x_75; 
lean_dec(x_73);
lean_dec(x_71);
lean_dec(x_66);
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_75 = lean_box(0);
return x_75;
}
else
{
lean_object* x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; uint8_t x_82; 
x_76 = lean_ctor_get(x_74, 0);
lean_inc(x_76);
if (lean_is_exclusive(x_74)) {
 lean_ctor_release(x_74, 0);
 x_77 = x_74;
} else {
 lean_dec_ref(x_74);
 x_77 = lean_box(0);
}
x_78 = lean_ctor_get(x_76, 0);
lean_inc(x_78);
x_79 = lean_ctor_get(x_76, 1);
lean_inc(x_79);
if (lean_is_exclusive(x_76)) {
 lean_ctor_release(x_76, 0);
 lean_ctor_release(x_76, 1);
 x_80 = x_76;
} else {
 lean_dec_ref(x_76);
 x_80 = lean_box(0);
}
x_81 = lean_byte_array_size(x_1);
x_82 = lean_nat_dec_eq(x_79, x_81);
lean_dec(x_81);
lean_dec(x_79);
if (x_82 == 0)
{
lean_object* x_83; 
lean_dec(x_80);
lean_dec(x_78);
lean_dec(x_77);
lean_dec(x_73);
lean_dec(x_71);
lean_dec(x_66);
lean_free_object(x_10);
lean_dec(x_12);
lean_dec(x_6);
x_83 = lean_box(0);
return x_83;
}
else
{
lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; 
if (lean_is_scalar(x_80)) {
 x_84 = lean_alloc_ctor(0, 2, 0);
} else {
 x_84 = x_80;
}
lean_ctor_set(x_84, 0, x_71);
lean_ctor_set(x_84, 1, x_78);
if (lean_is_scalar(x_73)) {
 x_85 = lean_alloc_ctor(0, 2, 0);
} else {
 x_85 = x_73;
}
lean_ctor_set(x_85, 0, x_66);
lean_ctor_set(x_85, 1, x_84);
x_86 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_86, 0, x_12);
lean_ctor_set(x_86, 1, x_85);
lean_ctor_set(x_10, 1, x_86);
lean_ctor_set(x_10, 0, x_6);
if (lean_is_scalar(x_77)) {
 x_87 = lean_alloc_ctor(1, 1, 0);
} else {
 x_87 = x_77;
}
lean_ctor_set(x_87, 0, x_10);
return x_87;
}
}
}
}
}
}
else
{
lean_object* x_88; lean_object* x_89; lean_object* x_90; 
x_88 = lean_ctor_get(x_10, 0);
x_89 = lean_ctor_get(x_10, 1);
lean_inc(x_89);
lean_inc(x_88);
lean_dec(x_10);
x_90 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_89);
lean_dec(x_89);
if (lean_obj_tag(x_90) == 0)
{
lean_object* x_91; 
lean_dec(x_88);
lean_dec(x_6);
x_91 = lean_box(0);
return x_91;
}
else
{
lean_object* x_92; lean_object* x_93; lean_object* x_94; lean_object* x_95; lean_object* x_96; 
x_92 = lean_ctor_get(x_90, 0);
lean_inc(x_92);
lean_dec_ref(x_90);
x_93 = lean_ctor_get(x_92, 0);
lean_inc(x_93);
x_94 = lean_ctor_get(x_92, 1);
lean_inc(x_94);
if (lean_is_exclusive(x_92)) {
 lean_ctor_release(x_92, 0);
 lean_ctor_release(x_92, 1);
 x_95 = x_92;
} else {
 lean_dec_ref(x_92);
 x_95 = lean_box(0);
}
x_96 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_94);
lean_dec(x_94);
if (lean_obj_tag(x_96) == 0)
{
lean_object* x_97; 
lean_dec(x_95);
lean_dec(x_93);
lean_dec(x_88);
lean_dec(x_6);
x_97 = lean_box(0);
return x_97;
}
else
{
lean_object* x_98; lean_object* x_99; lean_object* x_100; lean_object* x_101; lean_object* x_102; 
x_98 = lean_ctor_get(x_96, 0);
lean_inc(x_98);
lean_dec_ref(x_96);
x_99 = lean_ctor_get(x_98, 0);
lean_inc(x_99);
x_100 = lean_ctor_get(x_98, 1);
lean_inc(x_100);
if (lean_is_exclusive(x_98)) {
 lean_ctor_release(x_98, 0);
 lean_ctor_release(x_98, 1);
 x_101 = x_98;
} else {
 lean_dec_ref(x_98);
 x_101 = lean_box(0);
}
x_102 = l_VmVerifier_Spec_Wire_readBlobAt(x_1, x_100);
lean_dec(x_100);
if (lean_obj_tag(x_102) == 0)
{
lean_object* x_103; 
lean_dec(x_101);
lean_dec(x_99);
lean_dec(x_95);
lean_dec(x_93);
lean_dec(x_88);
lean_dec(x_6);
x_103 = lean_box(0);
return x_103;
}
else
{
lean_object* x_104; lean_object* x_105; lean_object* x_106; lean_object* x_107; lean_object* x_108; lean_object* x_109; uint8_t x_110; 
x_104 = lean_ctor_get(x_102, 0);
lean_inc(x_104);
if (lean_is_exclusive(x_102)) {
 lean_ctor_release(x_102, 0);
 x_105 = x_102;
} else {
 lean_dec_ref(x_102);
 x_105 = lean_box(0);
}
x_106 = lean_ctor_get(x_104, 0);
lean_inc(x_106);
x_107 = lean_ctor_get(x_104, 1);
lean_inc(x_107);
if (lean_is_exclusive(x_104)) {
 lean_ctor_release(x_104, 0);
 lean_ctor_release(x_104, 1);
 x_108 = x_104;
} else {
 lean_dec_ref(x_104);
 x_108 = lean_box(0);
}
x_109 = lean_byte_array_size(x_1);
x_110 = lean_nat_dec_eq(x_107, x_109);
lean_dec(x_109);
lean_dec(x_107);
if (x_110 == 0)
{
lean_object* x_111; 
lean_dec(x_108);
lean_dec(x_106);
lean_dec(x_105);
lean_dec(x_101);
lean_dec(x_99);
lean_dec(x_95);
lean_dec(x_93);
lean_dec(x_88);
lean_dec(x_6);
x_111 = lean_box(0);
return x_111;
}
else
{
lean_object* x_112; lean_object* x_113; lean_object* x_114; lean_object* x_115; lean_object* x_116; 
if (lean_is_scalar(x_108)) {
 x_112 = lean_alloc_ctor(0, 2, 0);
} else {
 x_112 = x_108;
}
lean_ctor_set(x_112, 0, x_99);
lean_ctor_set(x_112, 1, x_106);
if (lean_is_scalar(x_101)) {
 x_113 = lean_alloc_ctor(0, 2, 0);
} else {
 x_113 = x_101;
}
lean_ctor_set(x_113, 0, x_93);
lean_ctor_set(x_113, 1, x_112);
if (lean_is_scalar(x_95)) {
 x_114 = lean_alloc_ctor(0, 2, 0);
} else {
 x_114 = x_95;
}
lean_ctor_set(x_114, 0, x_88);
lean_ctor_set(x_114, 1, x_113);
x_115 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_115, 0, x_6);
lean_ctor_set(x_115, 1, x_114);
if (lean_is_scalar(x_105)) {
 x_116 = lean_alloc_ctor(1, 1, 0);
} else {
 x_116 = x_105;
}
lean_ctor_set(x_116, 0, x_115);
return x_116;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Spec_Wire_parseFiveBlobs___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_VmVerifier_Spec_Wire_parseFiveBlobs(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Swirl_Protocol_Noninteractive_Wire_RawToTyped(uint8_t builtin);
lean_object* initialize_VmVerifier_Spec_Types(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_VmVerifier_Spec_Wire(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Swirl_Protocol_Noninteractive_Wire_RawToTyped(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_VmVerifier_Spec_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
l_VmVerifier_Spec_Wire_baselineMagic___closed__0 = _init_l_VmVerifier_Spec_Wire_baselineMagic___closed__0();
lean_mark_persistent(l_VmVerifier_Spec_Wire_baselineMagic___closed__0);
l_VmVerifier_Spec_Wire_baselineMagic = _init_l_VmVerifier_Spec_Wire_baselineMagic();
lean_mark_persistent(l_VmVerifier_Spec_Wire_baselineMagic);
l_VmVerifier_Spec_Wire_userPvsMagic___closed__0 = _init_l_VmVerifier_Spec_Wire_userPvsMagic___closed__0();
lean_mark_persistent(l_VmVerifier_Spec_Wire_userPvsMagic___closed__0);
l_VmVerifier_Spec_Wire_userPvsMagic = _init_l_VmVerifier_Spec_Wire_userPvsMagic();
lean_mark_persistent(l_VmVerifier_Spec_Wire_userPvsMagic);
l_VmVerifier_Spec_Wire_ensureEnd___closed__0 = _init_l_VmVerifier_Spec_Wire_ensureEnd___closed__0();
lean_mark_persistent(l_VmVerifier_Spec_Wire_ensureEnd___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
