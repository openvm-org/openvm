// Lean compiler output
// Module: VmVerifier.Main
// Imports: public import Init public import VmVerifier.Spec.Runtime
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
LEAN_EXPORT lean_object* _lean_main();
lean_object* lean_uint32_to_nat(uint32_t);
static lean_object* l_VmVerifier_Executable_main___closed__1;
LEAN_EXPORT lean_object* l_VmVerifier_Executable_main___boxed__const__2;
static lean_object* l_VmVerifier_Executable_main___closed__2;
LEAN_EXPORT lean_object* l_main___boxed(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_IO_FS_Stream_readBinToEnd(lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Executable_main___boxed__const__1;
static lean_object* l_VmVerifier_Executable_main___closed__3;
LEAN_EXPORT lean_object* l_VmVerifier_Executable_main();
lean_object* lean_get_stdin();
lean_object* lean_get_stderr();
static lean_object* l_VmVerifier_Executable_main___closed__0;
lean_object* l_VmVerifier_verifyVmStarkProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_VmVerifier_Executable_main___boxed(lean_object*);
lean_object* l_VmVerifier_Spec_Wire_parseFiveBlobs(lean_object*);
lean_object* l_IO_FS_Stream_putStrLn(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_byte_array_size(lean_object*);
static lean_object* _init_l_VmVerifier_Executable_main___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("vm_verify: stdin framing error (received ", 41, 41);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_Executable_main___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" bytes)", 7, 7);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_Executable_main___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("vm_verify: verification failed (exit ", 37, 37);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_Executable_main___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(")", 1, 1);
return x_1;
}
}
static lean_object* _init_l_VmVerifier_Executable_main___boxed__const__1() {
_start:
{
uint32_t x_1; lean_object* x_2; 
x_1 = 20;
x_2 = lean_box_uint32(x_1);
return x_2;
}
}
static lean_object* _init_l_VmVerifier_Executable_main___boxed__const__2() {
_start:
{
uint32_t x_1; lean_object* x_2; 
x_1 = 0;
x_2 = lean_box_uint32(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Executable_main() {
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
x_7 = l_VmVerifier_Spec_Wire_parseFiveBlobs(x_6);
if (lean_obj_tag(x_7) == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
lean_free_object(x_4);
x_8 = l_VmVerifier_Executable_main___closed__0;
x_9 = lean_byte_array_size(x_6);
lean_dec(x_6);
x_10 = l_Nat_reprFast(x_9);
x_11 = lean_string_append(x_8, x_10);
lean_dec_ref(x_10);
x_12 = l_VmVerifier_Executable_main___closed__1;
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
x_17 = l_VmVerifier_Executable_main___boxed__const__1;
lean_ctor_set(x_14, 0, x_17);
return x_14;
}
else
{
lean_object* x_18; lean_object* x_19; 
lean_dec(x_14);
x_18 = l_VmVerifier_Executable_main___boxed__const__1;
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
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
lean_dec(x_6);
x_23 = lean_ctor_get(x_7, 0);
lean_inc(x_23);
lean_dec_ref(x_7);
x_24 = lean_ctor_get(x_23, 1);
lean_inc(x_24);
x_25 = lean_ctor_get(x_24, 1);
lean_inc(x_25);
x_26 = lean_ctor_get(x_25, 1);
lean_inc(x_26);
x_27 = lean_ctor_get(x_23, 0);
lean_inc(x_27);
lean_dec(x_23);
x_28 = lean_ctor_get(x_24, 0);
lean_inc(x_28);
lean_dec(x_24);
x_29 = lean_ctor_get(x_25, 0);
lean_inc(x_29);
lean_dec(x_25);
x_30 = lean_ctor_get(x_26, 0);
lean_inc(x_30);
x_31 = lean_ctor_get(x_26, 1);
lean_inc(x_31);
lean_dec(x_26);
x_32 = l_VmVerifier_verifyVmStarkProof(x_27, x_28, x_29, x_30, x_31);
if (lean_obj_tag(x_32) == 0)
{
lean_object* x_33; lean_object* x_34; uint32_t x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; 
lean_free_object(x_4);
x_33 = lean_ctor_get(x_32, 0);
lean_inc(x_33);
lean_dec_ref(x_32);
x_34 = l_VmVerifier_Executable_main___closed__2;
x_35 = lean_unbox_uint32(x_33);
x_36 = lean_uint32_to_nat(x_35);
x_37 = l_Nat_reprFast(x_36);
x_38 = lean_string_append(x_34, x_37);
lean_dec_ref(x_37);
x_39 = l_VmVerifier_Executable_main___closed__3;
x_40 = lean_string_append(x_38, x_39);
x_41 = l_IO_FS_Stream_putStrLn(x_3, x_40);
if (lean_obj_tag(x_41) == 0)
{
uint8_t x_42; 
x_42 = !lean_is_exclusive(x_41);
if (x_42 == 0)
{
lean_object* x_43; 
x_43 = lean_ctor_get(x_41, 0);
lean_dec(x_43);
lean_ctor_set(x_41, 0, x_33);
return x_41;
}
else
{
lean_object* x_44; 
lean_dec(x_41);
x_44 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_44, 0, x_33);
return x_44;
}
}
else
{
uint8_t x_45; 
lean_dec(x_33);
x_45 = !lean_is_exclusive(x_41);
if (x_45 == 0)
{
return x_41;
}
else
{
lean_object* x_46; lean_object* x_47; 
x_46 = lean_ctor_get(x_41, 0);
lean_inc(x_46);
lean_dec(x_41);
x_47 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_47, 0, x_46);
return x_47;
}
}
}
else
{
lean_object* x_48; 
lean_dec_ref(x_32);
lean_dec_ref(x_3);
x_48 = l_VmVerifier_Executable_main___boxed__const__2;
lean_ctor_set(x_4, 0, x_48);
return x_4;
}
}
}
else
{
lean_object* x_49; lean_object* x_50; 
x_49 = lean_ctor_get(x_4, 0);
lean_inc(x_49);
lean_dec(x_4);
x_50 = l_VmVerifier_Spec_Wire_parseFiveBlobs(x_49);
if (lean_obj_tag(x_50) == 0)
{
lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; 
x_51 = l_VmVerifier_Executable_main___closed__0;
x_52 = lean_byte_array_size(x_49);
lean_dec(x_49);
x_53 = l_Nat_reprFast(x_52);
x_54 = lean_string_append(x_51, x_53);
lean_dec_ref(x_53);
x_55 = l_VmVerifier_Executable_main___closed__1;
x_56 = lean_string_append(x_54, x_55);
x_57 = l_IO_FS_Stream_putStrLn(x_3, x_56);
if (lean_obj_tag(x_57) == 0)
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; 
if (lean_is_exclusive(x_57)) {
 lean_ctor_release(x_57, 0);
 x_58 = x_57;
} else {
 lean_dec_ref(x_57);
 x_58 = lean_box(0);
}
x_59 = l_VmVerifier_Executable_main___boxed__const__1;
if (lean_is_scalar(x_58)) {
 x_60 = lean_alloc_ctor(0, 1, 0);
} else {
 x_60 = x_58;
}
lean_ctor_set(x_60, 0, x_59);
return x_60;
}
else
{
lean_object* x_61; lean_object* x_62; lean_object* x_63; 
x_61 = lean_ctor_get(x_57, 0);
lean_inc(x_61);
if (lean_is_exclusive(x_57)) {
 lean_ctor_release(x_57, 0);
 x_62 = x_57;
} else {
 lean_dec_ref(x_57);
 x_62 = lean_box(0);
}
if (lean_is_scalar(x_62)) {
 x_63 = lean_alloc_ctor(1, 1, 0);
} else {
 x_63 = x_62;
}
lean_ctor_set(x_63, 0, x_61);
return x_63;
}
}
else
{
lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; 
lean_dec(x_49);
x_64 = lean_ctor_get(x_50, 0);
lean_inc(x_64);
lean_dec_ref(x_50);
x_65 = lean_ctor_get(x_64, 1);
lean_inc(x_65);
x_66 = lean_ctor_get(x_65, 1);
lean_inc(x_66);
x_67 = lean_ctor_get(x_66, 1);
lean_inc(x_67);
x_68 = lean_ctor_get(x_64, 0);
lean_inc(x_68);
lean_dec(x_64);
x_69 = lean_ctor_get(x_65, 0);
lean_inc(x_69);
lean_dec(x_65);
x_70 = lean_ctor_get(x_66, 0);
lean_inc(x_70);
lean_dec(x_66);
x_71 = lean_ctor_get(x_67, 0);
lean_inc(x_71);
x_72 = lean_ctor_get(x_67, 1);
lean_inc(x_72);
lean_dec(x_67);
x_73 = l_VmVerifier_verifyVmStarkProof(x_68, x_69, x_70, x_71, x_72);
if (lean_obj_tag(x_73) == 0)
{
lean_object* x_74; lean_object* x_75; uint32_t x_76; lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; 
x_74 = lean_ctor_get(x_73, 0);
lean_inc(x_74);
lean_dec_ref(x_73);
x_75 = l_VmVerifier_Executable_main___closed__2;
x_76 = lean_unbox_uint32(x_74);
x_77 = lean_uint32_to_nat(x_76);
x_78 = l_Nat_reprFast(x_77);
x_79 = lean_string_append(x_75, x_78);
lean_dec_ref(x_78);
x_80 = l_VmVerifier_Executable_main___closed__3;
x_81 = lean_string_append(x_79, x_80);
x_82 = l_IO_FS_Stream_putStrLn(x_3, x_81);
if (lean_obj_tag(x_82) == 0)
{
lean_object* x_83; lean_object* x_84; 
if (lean_is_exclusive(x_82)) {
 lean_ctor_release(x_82, 0);
 x_83 = x_82;
} else {
 lean_dec_ref(x_82);
 x_83 = lean_box(0);
}
if (lean_is_scalar(x_83)) {
 x_84 = lean_alloc_ctor(0, 1, 0);
} else {
 x_84 = x_83;
}
lean_ctor_set(x_84, 0, x_74);
return x_84;
}
else
{
lean_object* x_85; lean_object* x_86; lean_object* x_87; 
lean_dec(x_74);
x_85 = lean_ctor_get(x_82, 0);
lean_inc(x_85);
if (lean_is_exclusive(x_82)) {
 lean_ctor_release(x_82, 0);
 x_86 = x_82;
} else {
 lean_dec_ref(x_82);
 x_86 = lean_box(0);
}
if (lean_is_scalar(x_86)) {
 x_87 = lean_alloc_ctor(1, 1, 0);
} else {
 x_87 = x_86;
}
lean_ctor_set(x_87, 0, x_85);
return x_87;
}
}
else
{
lean_object* x_88; lean_object* x_89; 
lean_dec_ref(x_73);
lean_dec_ref(x_3);
x_88 = l_VmVerifier_Executable_main___boxed__const__2;
x_89 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_89, 0, x_88);
return x_89;
}
}
}
}
else
{
uint8_t x_90; 
lean_dec_ref(x_3);
x_90 = !lean_is_exclusive(x_4);
if (x_90 == 0)
{
return x_4;
}
else
{
lean_object* x_91; lean_object* x_92; 
x_91 = lean_ctor_get(x_4, 0);
lean_inc(x_91);
lean_dec(x_4);
x_92 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_92, 0, x_91);
return x_92;
}
}
}
}
LEAN_EXPORT lean_object* l_VmVerifier_Executable_main___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_VmVerifier_Executable_main();
return x_2;
}
}
LEAN_EXPORT lean_object* _lean_main() {
_start:
{
lean_object* x_2; 
x_2 = l_VmVerifier_Executable_main();
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
lean_object* initialize_VmVerifier_Spec_Runtime(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_VmVerifier_Main(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_VmVerifier_Spec_Runtime(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
l_VmVerifier_Executable_main___closed__0 = _init_l_VmVerifier_Executable_main___closed__0();
lean_mark_persistent(l_VmVerifier_Executable_main___closed__0);
l_VmVerifier_Executable_main___closed__1 = _init_l_VmVerifier_Executable_main___closed__1();
lean_mark_persistent(l_VmVerifier_Executable_main___closed__1);
l_VmVerifier_Executable_main___closed__2 = _init_l_VmVerifier_Executable_main___closed__2();
lean_mark_persistent(l_VmVerifier_Executable_main___closed__2);
l_VmVerifier_Executable_main___closed__3 = _init_l_VmVerifier_Executable_main___closed__3();
lean_mark_persistent(l_VmVerifier_Executable_main___closed__3);
l_VmVerifier_Executable_main___boxed__const__1 = _init_l_VmVerifier_Executable_main___boxed__const__1();
lean_mark_persistent(l_VmVerifier_Executable_main___boxed__const__1);
l_VmVerifier_Executable_main___boxed__const__2 = _init_l_VmVerifier_Executable_main___boxed__const__2();
lean_mark_persistent(l_VmVerifier_Executable_main___boxed__const__2);
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
res = initialize_VmVerifier_Main(1 /* builtin */);
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
