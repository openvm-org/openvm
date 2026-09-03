// Lean compiler output
// Module: VmVerifier.DumpProof
// Imports: public import Init public import VmVerifier.Spec.Wire
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
static lean_object* l_VmVerifier_DumpProof_main___closed__6;
LEAN_EXPORT lean_object* _lean_main();
static lean_object* l_VmVerifier_DumpProof_main___closed__5;
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawPublicValues(lean_object*, lean_object*);
lean_object* lean_uint32_to_nat(uint32_t);
static lean_object* l_VmVerifier_DumpProof_main___closed__0;
static lean_object* l_VmVerifier_DumpProof_main___closed__2;
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00VmVerifier_DumpProof_publicValuesDigest_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_vkCommitDigest(lean_object*);
static lean_object* l_VmVerifier_DumpProof_rawDigestToCsv___closed__0;
lean_object* l_VmVerifier_Spec_Wire_readBaseline(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_digestToCsv(lean_object*);
LEAN_EXPORT lean_object* l_main___boxed(lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
static lean_object* l_VmVerifier_DumpProof_userPvsProofDigest___closed__0;
LEAN_EXPORT uint32_t l_VmVerifier_DumpProof_parseErrorExitCode(lean_object*);
lean_object* lean_get_stdout();
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_baselineDigest(lean_object*);
lean_object* l_VmVerifier_Spec_Wire_readUserPvsProof(lean_object*);
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00VmVerifier_DumpProof_userPvsProofDigest_spec__0(lean_object*, lean_object*);
lean_object* l_IO_FS_Stream_readBinToEnd(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_main___boxed__const__2;
static lean_object* l_VmVerifier_DumpProof_publicValuesDigest___closed__0;
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_main___boxed(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_rawDigestToCsv(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProof(lean_object*);
lean_object* lean_get_stdin();
lean_object* lean_get_stderr();
static lean_object* l_VmVerifier_DumpProof_main___closed__4;
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVk(lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_rawFieldsToCsv(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
static lean_object* l_VmVerifier_DumpProof_main___closed__10;
LEAN_EXPORT lean_object* l_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_IO_println___at___00VmVerifier_DumpProof_main_spec__0(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_vkDigest(lean_object*);
lean_object* l_VmVerifier_Spec_Wire_parseFiveBlobs(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_parseErrorExitCode___boxed(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_main();
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_publicValuesDigest(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_userPvsProofDigest(lean_object*);
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* l_IO_println___at___00VmVerifier_DumpProof_main_spec__0___boxed(lean_object*, lean_object*);
static lean_object* l_VmVerifier_DumpProof_main___closed__1;
static lean_object* l_VmVerifier_DumpProof_main___closed__8;
lean_object* l_String_intercalate(lean_object*, lean_object*);
static lean_object* l_VmVerifier_DumpProof_main___closed__9;
lean_object* l_IO_FS_Stream_putStrLn(lean_object*, lean_object*);
static lean_object* l_VmVerifier_DumpProof_main___closed__11;
lean_object* lean_string_append(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_proofDigest(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_main___boxed__const__1;
static lean_object* l_VmVerifier_DumpProof_main___closed__7;
lean_object* lean_byte_array_size(lean_object*);
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(lean_object*, lean_object*);
static lean_object* l_VmVerifier_DumpProof_main___closed__3;
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; uint32_t x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_unbox_uint32(x_5);
lean_dec(x_5);
x_8 = lean_uint32_to_nat(x_7);
x_9 = l_Nat_reprFast(x_8);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_9);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_11; lean_object* x_12; uint32_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_1, 0);
x_12 = lean_ctor_get(x_1, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_1);
x_13 = lean_unbox_uint32(x_11);
lean_dec(x_11);
x_14 = lean_uint32_to_nat(x_13);
x_15 = l_Nat_reprFast(x_14);
x_16 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_2);
x_1 = x_12;
x_2 = x_16;
goto _start;
}
}
}
}
static lean_object* _init_l_VmVerifier_DumpProof_rawDigestToCsv___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(",", 1, 1);
return x_1;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_rawDigestToCsv(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = l_VmVerifier_DumpProof_rawDigestToCsv___closed__0;
x_3 = lean_array_to_list(x_1);
x_4 = lean_box(0);
x_5 = l_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(x_3, x_4);
x_6 = l_String_intercalate(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_rawFieldsToCsv(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = l_VmVerifier_DumpProof_rawDigestToCsv___closed__0;
x_3 = lean_array_to_list(x_1);
x_4 = lean_box(0);
x_5 = l_List_mapTR_loop___at___00VmVerifier_DumpProof_rawDigestToCsv_spec__0(x_3, x_4);
x_6 = l_String_intercalate(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_vkDigest(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = l_VmVerifier_DumpProof_rawDigestToCsv(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_proofDigest(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = l_VmVerifier_DumpProof_rawDigestToCsv(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00VmVerifier_DumpProof_publicValuesDigest_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = l_VmVerifier_DumpProof_rawFieldsToCsv(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = l_VmVerifier_DumpProof_rawFieldsToCsv(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
static lean_object* _init_l_VmVerifier_DumpProof_publicValuesDigest___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("|", 1, 1);
return x_1;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_publicValuesDigest(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = l_VmVerifier_DumpProof_publicValuesDigest___closed__0;
x_3 = lean_array_to_list(x_1);
x_4 = lean_box(0);
x_5 = l_List_mapTR_loop___at___00VmVerifier_DumpProof_publicValuesDigest_spec__0(x_3, x_4);
x_6 = l_String_intercalate(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = l_Nat_reprFast(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = l_Nat_reprFast(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_digestToCsv(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = l_VmVerifier_DumpProof_rawDigestToCsv___closed__0;
x_3 = lean_array_to_list(x_1);
x_4 = lean_box(0);
x_5 = l_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(x_3, x_4);
x_6 = l_String_intercalate(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_vkCommitDigest(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = l_VmVerifier_DumpProof_digestToCsv(x_3);
x_6 = l_VmVerifier_DumpProof_digestToCsv(x_4);
x_7 = lean_box(0);
lean_ctor_set_tag(x_1, 1);
lean_ctor_set(x_1, 1, x_7);
lean_ctor_set(x_1, 0, x_6);
x_8 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_1);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = l_VmVerifier_DumpProof_digestToCsv(x_9);
x_12 = l_VmVerifier_DumpProof_digestToCsv(x_10);
x_13 = lean_box(0);
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_15, 0, x_11);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_baselineDigest(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_1, 4);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_1, 5);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_1, 6);
lean_inc_ref(x_8);
lean_dec_ref(x_1);
x_9 = !lean_is_exclusive(x_2);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
x_12 = l_VmVerifier_DumpProof_publicValuesDigest___closed__0;
x_13 = l_VmVerifier_DumpProof_digestToCsv(x_3);
x_14 = l_Nat_reprFast(x_10);
x_15 = l_Nat_reprFast(x_11);
x_16 = l_Nat_reprFast(x_4);
x_17 = lean_box(0);
lean_ctor_set_tag(x_2, 1);
lean_ctor_set(x_2, 1, x_17);
lean_ctor_set(x_2, 0, x_16);
x_18 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_18, 0, x_15);
lean_ctor_set(x_18, 1, x_2);
x_19 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_19, 0, x_14);
lean_ctor_set(x_19, 1, x_18);
x_20 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_20, 0, x_13);
lean_ctor_set(x_20, 1, x_19);
x_21 = l_VmVerifier_DumpProof_vkCommitDigest(x_5);
x_22 = l_List_appendTR___redArg(x_20, x_21);
x_23 = l_VmVerifier_DumpProof_vkCommitDigest(x_6);
x_24 = l_List_appendTR___redArg(x_22, x_23);
x_25 = l_VmVerifier_DumpProof_vkCommitDigest(x_7);
x_26 = l_List_appendTR___redArg(x_24, x_25);
x_27 = l_VmVerifier_DumpProof_vkCommitDigest(x_8);
x_28 = l_List_appendTR___redArg(x_26, x_27);
x_29 = l_String_intercalate(x_12, x_28);
return x_29;
}
else
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; 
x_30 = lean_ctor_get(x_2, 0);
x_31 = lean_ctor_get(x_2, 1);
lean_inc(x_31);
lean_inc(x_30);
lean_dec(x_2);
x_32 = l_VmVerifier_DumpProof_publicValuesDigest___closed__0;
x_33 = l_VmVerifier_DumpProof_digestToCsv(x_3);
x_34 = l_Nat_reprFast(x_30);
x_35 = l_Nat_reprFast(x_31);
x_36 = l_Nat_reprFast(x_4);
x_37 = lean_box(0);
x_38 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_38, 0, x_36);
lean_ctor_set(x_38, 1, x_37);
x_39 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_39, 0, x_35);
lean_ctor_set(x_39, 1, x_38);
x_40 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_40, 0, x_34);
lean_ctor_set(x_40, 1, x_39);
x_41 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_41, 0, x_33);
lean_ctor_set(x_41, 1, x_40);
x_42 = l_VmVerifier_DumpProof_vkCommitDigest(x_5);
x_43 = l_List_appendTR___redArg(x_41, x_42);
x_44 = l_VmVerifier_DumpProof_vkCommitDigest(x_6);
x_45 = l_List_appendTR___redArg(x_43, x_44);
x_46 = l_VmVerifier_DumpProof_vkCommitDigest(x_7);
x_47 = l_List_appendTR___redArg(x_45, x_46);
x_48 = l_VmVerifier_DumpProof_vkCommitDigest(x_8);
x_49 = l_List_appendTR___redArg(x_47, x_48);
x_50 = l_String_intercalate(x_32, x_49);
return x_50;
}
}
}
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00VmVerifier_DumpProof_userPvsProofDigest_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = l_VmVerifier_DumpProof_digestToCsv(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = l_VmVerifier_DumpProof_digestToCsv(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
static lean_object* _init_l_VmVerifier_DumpProof_userPvsProofDigest___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(";", 1, 1);
return x_1;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_userPvsProofDigest(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = l_VmVerifier_DumpProof_userPvsProofDigest___closed__0;
x_6 = lean_box(0);
x_7 = l_List_mapTR_loop___at___00VmVerifier_DumpProof_userPvsProofDigest_spec__0(x_2, x_6);
x_8 = l_String_intercalate(x_5, x_7);
x_9 = l_VmVerifier_DumpProof_rawDigestToCsv___closed__0;
x_10 = l_List_mapTR_loop___at___00VmVerifier_DumpProof_digestToCsv_spec__0(x_3, x_6);
x_11 = l_String_intercalate(x_9, x_10);
x_12 = l_VmVerifier_DumpProof_publicValuesDigest___closed__0;
x_13 = l_VmVerifier_DumpProof_digestToCsv(x_4);
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_6);
x_15 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_15, 0, x_11);
lean_ctor_set(x_15, 1, x_14);
x_16 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_16, 0, x_8);
lean_ctor_set(x_16, 1, x_15);
x_17 = l_String_intercalate(x_12, x_16);
return x_17;
}
}
LEAN_EXPORT uint32_t l_VmVerifier_DumpProof_parseErrorExitCode(lean_object* x_1) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
uint32_t x_2; 
x_2 = 10;
return x_2;
}
case 1:
{
uint32_t x_3; 
x_3 = 11;
return x_3;
}
case 2:
{
uint32_t x_4; 
x_4 = 12;
return x_4;
}
default: 
{
uint32_t x_5; 
x_5 = 13;
return x_5;
}
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_parseErrorExitCode___boxed(lean_object* x_1) {
_start:
{
uint32_t x_2; lean_object* x_3; 
x_2 = l_VmVerifier_DumpProof_parseErrorExitCode(x_1);
lean_dec_ref(x_1);
x_3 = lean_box_uint32(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* l_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_get_stdout();
x_4 = lean_ctor_get(x_3, 4);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lean_apply_2(x_4, x_1, lean_box(0));
return x_5;
}
}
LEAN_EXPORT lean_object* l_IO_println___at___00VmVerifier_DumpProof_main_spec__0(lean_object* x_1) {
_start:
{
uint32_t x_3; lean_object* x_4; lean_object* x_5; 
x_3 = 10;
x_4 = lean_string_push(x_1, x_3);
x_5 = l_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0(x_4);
return x_5;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("vm_dump_proof: stdin framing error (received ", 45, 45);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" bytes)", 7, 7);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("vm_dump_proof: vk parse error: ", 31, 31);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("vm_dump_proof: baseline parse error: ", 37, 37);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("vm_dump_proof: proof parse error: ", 34, 34);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("vm_dump_proof: public-values parse error: ", 42, 42);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("vm_dump_proof: user-PV proof parse error: ", 42, 42);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("vk: ", 4, 4);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("baseline: ", 10, 10);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__9() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("proof: ", 7, 7);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__10() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("pv: ", 4, 4);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("user-pvs: ", 10, 10);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___boxed__const__1() {
_start:
{
uint32_t x_1; lean_object* x_2; 
x_1 = 20;
x_2 = lean_box_uint32(x_1);
return x_2;
}
}
static lean_object* _init_l_VmVerifier_DumpProof_main___boxed__const__2() {
_start:
{
uint32_t x_1; lean_object* x_2; 
x_1 = 0;
x_2 = lean_box_uint32(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_main() {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_get_stdin();
x_3 = l_IO_FS_Stream_readBinToEnd(x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_get_stderr();
x_6 = l_VmVerifier_Spec_Wire_parseFiveBlobs(x_4);
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_7 = l_VmVerifier_DumpProof_main___closed__0;
x_8 = lean_byte_array_size(x_4);
lean_dec(x_4);
x_9 = l_Nat_reprFast(x_8);
x_10 = lean_string_append(x_7, x_9);
lean_dec_ref(x_9);
x_11 = l_VmVerifier_DumpProof_main___closed__1;
x_12 = lean_string_append(x_10, x_11);
x_13 = l_IO_FS_Stream_putStrLn(x_5, x_12);
if (lean_obj_tag(x_13) == 0)
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; 
x_15 = lean_ctor_get(x_13, 0);
lean_dec(x_15);
x_16 = l_VmVerifier_DumpProof_main___boxed__const__1;
lean_ctor_set(x_13, 0, x_16);
return x_13;
}
else
{
lean_object* x_17; lean_object* x_18; 
lean_dec(x_13);
x_17 = l_VmVerifier_DumpProof_main___boxed__const__1;
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
}
else
{
uint8_t x_19; 
x_19 = !lean_is_exclusive(x_13);
if (x_19 == 0)
{
return x_13;
}
else
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_ctor_get(x_13, 0);
lean_inc(x_20);
lean_dec(x_13);
x_21 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
lean_dec(x_4);
x_22 = lean_ctor_get(x_6, 0);
lean_inc(x_22);
lean_dec_ref(x_6);
x_23 = lean_ctor_get(x_22, 1);
lean_inc(x_23);
x_24 = lean_ctor_get(x_23, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_24, 1);
lean_inc(x_25);
x_26 = lean_ctor_get(x_22, 0);
lean_inc(x_26);
lean_dec(x_22);
x_27 = lean_ctor_get(x_23, 0);
lean_inc(x_27);
lean_dec(x_23);
x_28 = lean_ctor_get(x_24, 0);
lean_inc(x_28);
lean_dec(x_24);
x_29 = lean_ctor_get(x_25, 0);
lean_inc(x_29);
x_30 = lean_ctor_get(x_25, 1);
lean_inc(x_30);
lean_dec(x_25);
x_31 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVk(x_26);
if (lean_obj_tag(x_31) == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
lean_dec(x_30);
lean_dec(x_29);
lean_dec(x_28);
lean_dec(x_27);
x_32 = lean_ctor_get(x_31, 0);
lean_inc(x_32);
lean_dec_ref(x_31);
x_33 = l_VmVerifier_DumpProof_main___closed__2;
lean_inc(x_32);
x_34 = l_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(x_32);
x_35 = lean_string_append(x_33, x_34);
lean_dec_ref(x_34);
x_36 = l_IO_FS_Stream_putStrLn(x_5, x_35);
if (lean_obj_tag(x_36) == 0)
{
uint8_t x_37; 
x_37 = !lean_is_exclusive(x_36);
if (x_37 == 0)
{
lean_object* x_38; uint32_t x_39; lean_object* x_40; 
x_38 = lean_ctor_get(x_36, 0);
lean_dec(x_38);
x_39 = l_VmVerifier_DumpProof_parseErrorExitCode(x_32);
lean_dec(x_32);
x_40 = lean_box_uint32(x_39);
lean_ctor_set(x_36, 0, x_40);
return x_36;
}
else
{
uint32_t x_41; lean_object* x_42; lean_object* x_43; 
lean_dec(x_36);
x_41 = l_VmVerifier_DumpProof_parseErrorExitCode(x_32);
lean_dec(x_32);
x_42 = lean_box_uint32(x_41);
x_43 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_43, 0, x_42);
return x_43;
}
}
else
{
uint8_t x_44; 
lean_dec(x_32);
x_44 = !lean_is_exclusive(x_36);
if (x_44 == 0)
{
return x_36;
}
else
{
lean_object* x_45; lean_object* x_46; 
x_45 = lean_ctor_get(x_36, 0);
lean_inc(x_45);
lean_dec(x_36);
x_46 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_46, 0, x_45);
return x_46;
}
}
}
else
{
lean_object* x_47; lean_object* x_48; 
x_47 = lean_ctor_get(x_31, 0);
lean_inc(x_47);
lean_dec_ref(x_31);
x_48 = l_VmVerifier_Spec_Wire_readBaseline(x_27);
if (lean_obj_tag(x_48) == 0)
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; 
lean_dec(x_47);
lean_dec(x_30);
lean_dec(x_29);
lean_dec(x_28);
x_49 = lean_ctor_get(x_48, 0);
lean_inc(x_49);
lean_dec_ref(x_48);
x_50 = l_VmVerifier_DumpProof_main___closed__3;
lean_inc(x_49);
x_51 = l_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(x_49);
x_52 = lean_string_append(x_50, x_51);
lean_dec_ref(x_51);
x_53 = l_IO_FS_Stream_putStrLn(x_5, x_52);
if (lean_obj_tag(x_53) == 0)
{
uint8_t x_54; 
x_54 = !lean_is_exclusive(x_53);
if (x_54 == 0)
{
lean_object* x_55; uint32_t x_56; lean_object* x_57; 
x_55 = lean_ctor_get(x_53, 0);
lean_dec(x_55);
x_56 = l_VmVerifier_DumpProof_parseErrorExitCode(x_49);
lean_dec(x_49);
x_57 = lean_box_uint32(x_56);
lean_ctor_set(x_53, 0, x_57);
return x_53;
}
else
{
uint32_t x_58; lean_object* x_59; lean_object* x_60; 
lean_dec(x_53);
x_58 = l_VmVerifier_DumpProof_parseErrorExitCode(x_49);
lean_dec(x_49);
x_59 = lean_box_uint32(x_58);
x_60 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_60, 0, x_59);
return x_60;
}
}
else
{
uint8_t x_61; 
lean_dec(x_49);
x_61 = !lean_is_exclusive(x_53);
if (x_61 == 0)
{
return x_53;
}
else
{
lean_object* x_62; lean_object* x_63; 
x_62 = lean_ctor_get(x_53, 0);
lean_inc(x_62);
lean_dec(x_53);
x_63 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_63, 0, x_62);
return x_63;
}
}
}
else
{
lean_object* x_64; lean_object* x_65; 
x_64 = lean_ctor_get(x_48, 0);
lean_inc(x_64);
lean_dec_ref(x_48);
x_65 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProof(x_28);
if (lean_obj_tag(x_65) == 0)
{
lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; 
lean_dec(x_64);
lean_dec(x_47);
lean_dec(x_30);
lean_dec(x_29);
x_66 = lean_ctor_get(x_65, 0);
lean_inc(x_66);
lean_dec_ref(x_65);
x_67 = l_VmVerifier_DumpProof_main___closed__4;
lean_inc(x_66);
x_68 = l_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(x_66);
x_69 = lean_string_append(x_67, x_68);
lean_dec_ref(x_68);
x_70 = l_IO_FS_Stream_putStrLn(x_5, x_69);
if (lean_obj_tag(x_70) == 0)
{
uint8_t x_71; 
x_71 = !lean_is_exclusive(x_70);
if (x_71 == 0)
{
lean_object* x_72; uint32_t x_73; lean_object* x_74; 
x_72 = lean_ctor_get(x_70, 0);
lean_dec(x_72);
x_73 = l_VmVerifier_DumpProof_parseErrorExitCode(x_66);
lean_dec(x_66);
x_74 = lean_box_uint32(x_73);
lean_ctor_set(x_70, 0, x_74);
return x_70;
}
else
{
uint32_t x_75; lean_object* x_76; lean_object* x_77; 
lean_dec(x_70);
x_75 = l_VmVerifier_DumpProof_parseErrorExitCode(x_66);
lean_dec(x_66);
x_76 = lean_box_uint32(x_75);
x_77 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_77, 0, x_76);
return x_77;
}
}
else
{
uint8_t x_78; 
lean_dec(x_66);
x_78 = !lean_is_exclusive(x_70);
if (x_78 == 0)
{
return x_70;
}
else
{
lean_object* x_79; lean_object* x_80; 
x_79 = lean_ctor_get(x_70, 0);
lean_inc(x_79);
lean_dec(x_70);
x_80 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_80, 0, x_79);
return x_80;
}
}
}
else
{
lean_object* x_81; lean_object* x_82; 
x_81 = lean_ctor_get(x_65, 0);
lean_inc(x_81);
lean_dec_ref(x_65);
lean_inc(x_47);
x_82 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawPublicValues(x_47, x_29);
if (lean_obj_tag(x_82) == 0)
{
lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; lean_object* x_87; 
lean_dec(x_81);
lean_dec(x_64);
lean_dec(x_47);
lean_dec(x_30);
x_83 = lean_ctor_get(x_82, 0);
lean_inc(x_83);
lean_dec_ref(x_82);
x_84 = l_VmVerifier_DumpProof_main___closed__5;
lean_inc(x_83);
x_85 = l_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(x_83);
x_86 = lean_string_append(x_84, x_85);
lean_dec_ref(x_85);
x_87 = l_IO_FS_Stream_putStrLn(x_5, x_86);
if (lean_obj_tag(x_87) == 0)
{
uint8_t x_88; 
x_88 = !lean_is_exclusive(x_87);
if (x_88 == 0)
{
lean_object* x_89; uint32_t x_90; lean_object* x_91; 
x_89 = lean_ctor_get(x_87, 0);
lean_dec(x_89);
x_90 = l_VmVerifier_DumpProof_parseErrorExitCode(x_83);
lean_dec(x_83);
x_91 = lean_box_uint32(x_90);
lean_ctor_set(x_87, 0, x_91);
return x_87;
}
else
{
uint32_t x_92; lean_object* x_93; lean_object* x_94; 
lean_dec(x_87);
x_92 = l_VmVerifier_DumpProof_parseErrorExitCode(x_83);
lean_dec(x_83);
x_93 = lean_box_uint32(x_92);
x_94 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_94, 0, x_93);
return x_94;
}
}
else
{
uint8_t x_95; 
lean_dec(x_83);
x_95 = !lean_is_exclusive(x_87);
if (x_95 == 0)
{
return x_87;
}
else
{
lean_object* x_96; lean_object* x_97; 
x_96 = lean_ctor_get(x_87, 0);
lean_inc(x_96);
lean_dec(x_87);
x_97 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_97, 0, x_96);
return x_97;
}
}
}
else
{
lean_object* x_98; lean_object* x_99; 
x_98 = lean_ctor_get(x_82, 0);
lean_inc(x_98);
lean_dec_ref(x_82);
x_99 = l_VmVerifier_Spec_Wire_readUserPvsProof(x_30);
if (lean_obj_tag(x_99) == 0)
{
lean_object* x_100; lean_object* x_101; lean_object* x_102; lean_object* x_103; lean_object* x_104; 
lean_dec(x_98);
lean_dec(x_81);
lean_dec(x_64);
lean_dec(x_47);
x_100 = lean_ctor_get(x_99, 0);
lean_inc(x_100);
lean_dec_ref(x_99);
x_101 = l_VmVerifier_DumpProof_main___closed__6;
lean_inc(x_100);
x_102 = l_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(x_100);
x_103 = lean_string_append(x_101, x_102);
lean_dec_ref(x_102);
x_104 = l_IO_FS_Stream_putStrLn(x_5, x_103);
if (lean_obj_tag(x_104) == 0)
{
uint8_t x_105; 
x_105 = !lean_is_exclusive(x_104);
if (x_105 == 0)
{
lean_object* x_106; uint32_t x_107; lean_object* x_108; 
x_106 = lean_ctor_get(x_104, 0);
lean_dec(x_106);
x_107 = l_VmVerifier_DumpProof_parseErrorExitCode(x_100);
lean_dec(x_100);
x_108 = lean_box_uint32(x_107);
lean_ctor_set(x_104, 0, x_108);
return x_104;
}
else
{
uint32_t x_109; lean_object* x_110; lean_object* x_111; 
lean_dec(x_104);
x_109 = l_VmVerifier_DumpProof_parseErrorExitCode(x_100);
lean_dec(x_100);
x_110 = lean_box_uint32(x_109);
x_111 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_111, 0, x_110);
return x_111;
}
}
else
{
uint8_t x_112; 
lean_dec(x_100);
x_112 = !lean_is_exclusive(x_104);
if (x_112 == 0)
{
return x_104;
}
else
{
lean_object* x_113; lean_object* x_114; 
x_113 = lean_ctor_get(x_104, 0);
lean_inc(x_113);
lean_dec(x_104);
x_114 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_114, 0, x_113);
return x_114;
}
}
}
else
{
lean_object* x_115; lean_object* x_116; lean_object* x_117; lean_object* x_118; lean_object* x_119; 
lean_dec_ref(x_5);
x_115 = lean_ctor_get(x_99, 0);
lean_inc(x_115);
lean_dec_ref(x_99);
x_116 = l_VmVerifier_DumpProof_main___closed__7;
x_117 = l_VmVerifier_DumpProof_vkDigest(x_47);
x_118 = lean_string_append(x_116, x_117);
lean_dec_ref(x_117);
x_119 = l_IO_println___at___00VmVerifier_DumpProof_main_spec__0(x_118);
if (lean_obj_tag(x_119) == 0)
{
lean_object* x_120; lean_object* x_121; lean_object* x_122; lean_object* x_123; 
lean_dec_ref(x_119);
x_120 = l_VmVerifier_DumpProof_main___closed__8;
x_121 = l_VmVerifier_DumpProof_baselineDigest(x_64);
x_122 = lean_string_append(x_120, x_121);
lean_dec_ref(x_121);
x_123 = l_IO_println___at___00VmVerifier_DumpProof_main_spec__0(x_122);
if (lean_obj_tag(x_123) == 0)
{
lean_object* x_124; lean_object* x_125; lean_object* x_126; lean_object* x_127; 
lean_dec_ref(x_123);
x_124 = l_VmVerifier_DumpProof_main___closed__9;
x_125 = l_VmVerifier_DumpProof_proofDigest(x_81);
x_126 = lean_string_append(x_124, x_125);
lean_dec_ref(x_125);
x_127 = l_IO_println___at___00VmVerifier_DumpProof_main_spec__0(x_126);
if (lean_obj_tag(x_127) == 0)
{
lean_object* x_128; lean_object* x_129; lean_object* x_130; lean_object* x_131; 
lean_dec_ref(x_127);
x_128 = l_VmVerifier_DumpProof_main___closed__10;
x_129 = l_VmVerifier_DumpProof_publicValuesDigest(x_98);
x_130 = lean_string_append(x_128, x_129);
lean_dec_ref(x_129);
x_131 = l_IO_println___at___00VmVerifier_DumpProof_main_spec__0(x_130);
if (lean_obj_tag(x_131) == 0)
{
lean_object* x_132; lean_object* x_133; lean_object* x_134; lean_object* x_135; 
lean_dec_ref(x_131);
x_132 = l_VmVerifier_DumpProof_main___closed__11;
x_133 = l_VmVerifier_DumpProof_userPvsProofDigest(x_115);
x_134 = lean_string_append(x_132, x_133);
lean_dec_ref(x_133);
x_135 = l_IO_println___at___00VmVerifier_DumpProof_main_spec__0(x_134);
if (lean_obj_tag(x_135) == 0)
{
uint8_t x_136; 
x_136 = !lean_is_exclusive(x_135);
if (x_136 == 0)
{
lean_object* x_137; lean_object* x_138; 
x_137 = lean_ctor_get(x_135, 0);
lean_dec(x_137);
x_138 = l_VmVerifier_DumpProof_main___boxed__const__2;
lean_ctor_set(x_135, 0, x_138);
return x_135;
}
else
{
lean_object* x_139; lean_object* x_140; 
lean_dec(x_135);
x_139 = l_VmVerifier_DumpProof_main___boxed__const__2;
x_140 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_140, 0, x_139);
return x_140;
}
}
else
{
uint8_t x_141; 
x_141 = !lean_is_exclusive(x_135);
if (x_141 == 0)
{
return x_135;
}
else
{
lean_object* x_142; lean_object* x_143; 
x_142 = lean_ctor_get(x_135, 0);
lean_inc(x_142);
lean_dec(x_135);
x_143 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_143, 0, x_142);
return x_143;
}
}
}
else
{
uint8_t x_144; 
lean_dec(x_115);
x_144 = !lean_is_exclusive(x_131);
if (x_144 == 0)
{
return x_131;
}
else
{
lean_object* x_145; lean_object* x_146; 
x_145 = lean_ctor_get(x_131, 0);
lean_inc(x_145);
lean_dec(x_131);
x_146 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_146, 0, x_145);
return x_146;
}
}
}
else
{
uint8_t x_147; 
lean_dec(x_115);
lean_dec(x_98);
x_147 = !lean_is_exclusive(x_127);
if (x_147 == 0)
{
return x_127;
}
else
{
lean_object* x_148; lean_object* x_149; 
x_148 = lean_ctor_get(x_127, 0);
lean_inc(x_148);
lean_dec(x_127);
x_149 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_149, 0, x_148);
return x_149;
}
}
}
else
{
uint8_t x_150; 
lean_dec(x_115);
lean_dec(x_98);
lean_dec(x_81);
x_150 = !lean_is_exclusive(x_123);
if (x_150 == 0)
{
return x_123;
}
else
{
lean_object* x_151; lean_object* x_152; 
x_151 = lean_ctor_get(x_123, 0);
lean_inc(x_151);
lean_dec(x_123);
x_152 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_152, 0, x_151);
return x_152;
}
}
}
else
{
uint8_t x_153; 
lean_dec(x_115);
lean_dec(x_98);
lean_dec(x_81);
lean_dec(x_64);
x_153 = !lean_is_exclusive(x_119);
if (x_153 == 0)
{
return x_119;
}
else
{
lean_object* x_154; lean_object* x_155; 
x_154 = lean_ctor_get(x_119, 0);
lean_inc(x_154);
lean_dec(x_119);
x_155 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_155, 0, x_154);
return x_155;
}
}
}
}
}
}
}
}
}
else
{
uint8_t x_156; 
x_156 = !lean_is_exclusive(x_3);
if (x_156 == 0)
{
return x_3;
}
else
{
lean_object* x_157; lean_object* x_158; 
x_157 = lean_ctor_get(x_3, 0);
lean_inc(x_157);
lean_dec(x_3);
x_158 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_158, 0, x_157);
return x_158;
}
}
}
}
LEAN_EXPORT lean_object* l_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_IO_print___at___00IO_println___at___00VmVerifier_DumpProof_main_spec__0_spec__0(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* l_IO_println___at___00VmVerifier_DumpProof_main_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_IO_println___at___00VmVerifier_DumpProof_main_spec__0(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_DumpProof_main___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_VmVerifier_DumpProof_main();
return x_2;
}
}
LEAN_EXPORT lean_object* _lean_main() {
_start:
{
lean_object* x_2; 
x_2 = l_VmVerifier_DumpProof_main();
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
lean_object* initialize_VmVerifier_Spec_Wire(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_VmVerifier_DumpProof(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_VmVerifier_Spec_Wire(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
l_VmVerifier_DumpProof_rawDigestToCsv___closed__0 = _init_l_VmVerifier_DumpProof_rawDigestToCsv___closed__0();
lean_mark_persistent(l_VmVerifier_DumpProof_rawDigestToCsv___closed__0);
l_VmVerifier_DumpProof_publicValuesDigest___closed__0 = _init_l_VmVerifier_DumpProof_publicValuesDigest___closed__0();
lean_mark_persistent(l_VmVerifier_DumpProof_publicValuesDigest___closed__0);
l_VmVerifier_DumpProof_userPvsProofDigest___closed__0 = _init_l_VmVerifier_DumpProof_userPvsProofDigest___closed__0();
lean_mark_persistent(l_VmVerifier_DumpProof_userPvsProofDigest___closed__0);
l_VmVerifier_DumpProof_main___closed__0 = _init_l_VmVerifier_DumpProof_main___closed__0();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__0);
l_VmVerifier_DumpProof_main___closed__1 = _init_l_VmVerifier_DumpProof_main___closed__1();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__1);
l_VmVerifier_DumpProof_main___closed__2 = _init_l_VmVerifier_DumpProof_main___closed__2();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__2);
l_VmVerifier_DumpProof_main___closed__3 = _init_l_VmVerifier_DumpProof_main___closed__3();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__3);
l_VmVerifier_DumpProof_main___closed__4 = _init_l_VmVerifier_DumpProof_main___closed__4();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__4);
l_VmVerifier_DumpProof_main___closed__5 = _init_l_VmVerifier_DumpProof_main___closed__5();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__5);
l_VmVerifier_DumpProof_main___closed__6 = _init_l_VmVerifier_DumpProof_main___closed__6();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__6);
l_VmVerifier_DumpProof_main___closed__7 = _init_l_VmVerifier_DumpProof_main___closed__7();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__7);
l_VmVerifier_DumpProof_main___closed__8 = _init_l_VmVerifier_DumpProof_main___closed__8();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__8);
l_VmVerifier_DumpProof_main___closed__9 = _init_l_VmVerifier_DumpProof_main___closed__9();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__9);
l_VmVerifier_DumpProof_main___closed__10 = _init_l_VmVerifier_DumpProof_main___closed__10();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__10);
l_VmVerifier_DumpProof_main___closed__11 = _init_l_VmVerifier_DumpProof_main___closed__11();
lean_mark_persistent(l_VmVerifier_DumpProof_main___closed__11);
l_VmVerifier_DumpProof_main___boxed__const__1 = _init_l_VmVerifier_DumpProof_main___boxed__const__1();
lean_mark_persistent(l_VmVerifier_DumpProof_main___boxed__const__1);
l_VmVerifier_DumpProof_main___boxed__const__2 = _init_l_VmVerifier_DumpProof_main___boxed__const__2();
lean_mark_persistent(l_VmVerifier_DumpProof_main___boxed__const__2);
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
res = initialize_VmVerifier_DumpProof(1 /* builtin */);
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
