// Lean compiler output
// Module: VmVerifier.Main
// Imports: public import Init public meta import Init public import VmVerifier.Spec.Runtime
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
lean_object* lean_get_stdin();
lean_object* lean_get_stderr();
lean_object* l_IO_FS_Stream_readBinToEnd(lean_object*);
lean_object* lp_openvm_x2dfv_VmVerifier_Spec_Wire_parseFiveBlobs(lean_object*);
lean_object* lean_byte_array_size(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_IO_FS_Stream_putStrLn(lean_object*, lean_object*);
lean_object* lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_uint32_to_nat(uint32_t);
static const lean_string_object lp_openvm_x2dfv_VmVerifier_Executable_main___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "vm_verify: stdin framing error (received "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_Executable_main___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_Executable_main___closed__0_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_Executable_main___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " bytes)"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_Executable_main___closed__1 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_Executable_main___closed__1_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_Executable_main___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "vm_verify: verification failed (exit "};
static const lean_object* lp_openvm_x2dfv_VmVerifier_Executable_main___closed__2 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_Executable_main___closed__2_value;
static const lean_string_object lp_openvm_x2dfv_VmVerifier_Executable_main___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_openvm_x2dfv_VmVerifier_Executable_main___closed__3 = (const lean_object*)&lp_openvm_x2dfv_VmVerifier_Executable_main___closed__3_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__1;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__2;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_Executable_main();
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_Executable_main___boxed(lean_object*);
LEAN_EXPORT lean_object* _lean_main();
LEAN_EXPORT lean_object* lp_openvm_x2dfv_main___boxed(lean_object*);
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__1(void){
_start:
{
uint32_t v___x_5_; lean_object* v___x_6_; 
v___x_5_ = 20;
v___x_6_ = lean_box_uint32(v___x_5_);
return v___x_6_;
}
}
static lean_object* _init_lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__2(void){
_start:
{
uint32_t v___x_7_; lean_object* v___x_8_; 
v___x_7_ = 0;
v___x_8_ = lean_box_uint32(v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_Executable_main(){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_10_ = lean_get_stdin();
v___x_11_ = lean_get_stderr();
v___x_12_ = l_IO_FS_Stream_readBinToEnd(v___x_10_);
if (lean_obj_tag(v___x_12_) == 0)
{
lean_object* v_a_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_81_; 
v_a_13_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_81_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_81_ == 0)
{
v___x_15_ = v___x_12_;
v_isShared_16_ = v_isSharedCheck_81_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_a_13_);
lean_dec(v___x_12_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_81_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_17_; 
v___x_17_ = lp_openvm_x2dfv_VmVerifier_Spec_Wire_parseFiveBlobs(v_a_13_);
if (lean_obj_tag(v___x_17_) == 0)
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
lean_del_object(v___x_15_);
v___x_18_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_Executable_main___closed__0));
v___x_19_ = lean_byte_array_size(v_a_13_);
lean_dec(v_a_13_);
v___x_20_ = l_Nat_reprFast(v___x_19_);
v___x_21_ = lean_string_append(v___x_18_, v___x_20_);
lean_dec_ref(v___x_20_);
v___x_22_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_Executable_main___closed__1));
v___x_23_ = lean_string_append(v___x_21_, v___x_22_);
v___x_24_ = l_IO_FS_Stream_putStrLn(v___x_11_, v___x_23_);
if (lean_obj_tag(v___x_24_) == 0)
{
lean_object* v___x_26_; uint8_t v_isShared_27_; uint8_t v_isSharedCheck_32_; 
v_isSharedCheck_32_ = !lean_is_exclusive(v___x_24_);
if (v_isSharedCheck_32_ == 0)
{
lean_object* v_unused_33_; 
v_unused_33_ = lean_ctor_get(v___x_24_, 0);
lean_dec(v_unused_33_);
v___x_26_ = v___x_24_;
v_isShared_27_ = v_isSharedCheck_32_;
goto v_resetjp_25_;
}
else
{
lean_dec(v___x_24_);
v___x_26_ = lean_box(0);
v_isShared_27_ = v_isSharedCheck_32_;
goto v_resetjp_25_;
}
v_resetjp_25_:
{
lean_object* v___x_28_; lean_object* v___x_30_; 
v___x_28_ = lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__1;
if (v_isShared_27_ == 0)
{
lean_ctor_set(v___x_26_, 0, v___x_28_);
v___x_30_ = v___x_26_;
goto v_reusejp_29_;
}
else
{
lean_object* v_reuseFailAlloc_31_; 
v_reuseFailAlloc_31_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_31_, 0, v___x_28_);
v___x_30_ = v_reuseFailAlloc_31_;
goto v_reusejp_29_;
}
v_reusejp_29_:
{
return v___x_30_;
}
}
}
else
{
lean_object* v_a_34_; lean_object* v___x_36_; uint8_t v_isShared_37_; uint8_t v_isSharedCheck_41_; 
v_a_34_ = lean_ctor_get(v___x_24_, 0);
v_isSharedCheck_41_ = !lean_is_exclusive(v___x_24_);
if (v_isSharedCheck_41_ == 0)
{
v___x_36_ = v___x_24_;
v_isShared_37_ = v_isSharedCheck_41_;
goto v_resetjp_35_;
}
else
{
lean_inc(v_a_34_);
lean_dec(v___x_24_);
v___x_36_ = lean_box(0);
v_isShared_37_ = v_isSharedCheck_41_;
goto v_resetjp_35_;
}
v_resetjp_35_:
{
lean_object* v___x_39_; 
if (v_isShared_37_ == 0)
{
v___x_39_ = v___x_36_;
goto v_reusejp_38_;
}
else
{
lean_object* v_reuseFailAlloc_40_; 
v_reuseFailAlloc_40_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_40_, 0, v_a_34_);
v___x_39_ = v_reuseFailAlloc_40_;
goto v_reusejp_38_;
}
v_reusejp_38_:
{
return v___x_39_;
}
}
}
}
else
{
lean_object* v_val_42_; lean_object* v_snd_43_; lean_object* v_snd_44_; lean_object* v_snd_45_; lean_object* v_fst_46_; lean_object* v_fst_47_; lean_object* v_fst_48_; lean_object* v_fst_49_; lean_object* v_snd_50_; lean_object* v___x_51_; 
lean_dec(v_a_13_);
v_val_42_ = lean_ctor_get(v___x_17_, 0);
lean_inc(v_val_42_);
lean_dec_ref_known(v___x_17_, 1);
v_snd_43_ = lean_ctor_get(v_val_42_, 1);
lean_inc(v_snd_43_);
v_snd_44_ = lean_ctor_get(v_snd_43_, 1);
lean_inc(v_snd_44_);
v_snd_45_ = lean_ctor_get(v_snd_44_, 1);
lean_inc(v_snd_45_);
v_fst_46_ = lean_ctor_get(v_val_42_, 0);
lean_inc(v_fst_46_);
lean_dec(v_val_42_);
v_fst_47_ = lean_ctor_get(v_snd_43_, 0);
lean_inc(v_fst_47_);
lean_dec(v_snd_43_);
v_fst_48_ = lean_ctor_get(v_snd_44_, 0);
lean_inc(v_fst_48_);
lean_dec(v_snd_44_);
v_fst_49_ = lean_ctor_get(v_snd_45_, 0);
lean_inc(v_fst_49_);
v_snd_50_ = lean_ctor_get(v_snd_45_, 1);
lean_inc(v_snd_50_);
lean_dec(v_snd_45_);
v___x_51_ = lp_openvm_x2dfv_VmVerifier_verifyVmStarkProof(v_fst_46_, v_fst_47_, v_fst_48_, v_fst_49_, v_snd_50_);
if (lean_obj_tag(v___x_51_) == 0)
{
lean_object* v_a_52_; lean_object* v___x_53_; uint32_t v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
lean_del_object(v___x_15_);
v_a_52_ = lean_ctor_get(v___x_51_, 0);
lean_inc(v_a_52_);
lean_dec_ref_known(v___x_51_, 1);
v___x_53_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_Executable_main___closed__2));
v___x_54_ = lean_unbox_uint32(v_a_52_);
v___x_55_ = lean_uint32_to_nat(v___x_54_);
v___x_56_ = l_Nat_reprFast(v___x_55_);
v___x_57_ = lean_string_append(v___x_53_, v___x_56_);
lean_dec_ref(v___x_56_);
v___x_58_ = ((lean_object*)(lp_openvm_x2dfv_VmVerifier_Executable_main___closed__3));
v___x_59_ = lean_string_append(v___x_57_, v___x_58_);
v___x_60_ = l_IO_FS_Stream_putStrLn(v___x_11_, v___x_59_);
if (lean_obj_tag(v___x_60_) == 0)
{
lean_object* v___x_62_; uint8_t v_isShared_63_; uint8_t v_isSharedCheck_67_; 
v_isSharedCheck_67_ = !lean_is_exclusive(v___x_60_);
if (v_isSharedCheck_67_ == 0)
{
lean_object* v_unused_68_; 
v_unused_68_ = lean_ctor_get(v___x_60_, 0);
lean_dec(v_unused_68_);
v___x_62_ = v___x_60_;
v_isShared_63_ = v_isSharedCheck_67_;
goto v_resetjp_61_;
}
else
{
lean_dec(v___x_60_);
v___x_62_ = lean_box(0);
v_isShared_63_ = v_isSharedCheck_67_;
goto v_resetjp_61_;
}
v_resetjp_61_:
{
lean_object* v___x_65_; 
if (v_isShared_63_ == 0)
{
lean_ctor_set(v___x_62_, 0, v_a_52_);
v___x_65_ = v___x_62_;
goto v_reusejp_64_;
}
else
{
lean_object* v_reuseFailAlloc_66_; 
v_reuseFailAlloc_66_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_66_, 0, v_a_52_);
v___x_65_ = v_reuseFailAlloc_66_;
goto v_reusejp_64_;
}
v_reusejp_64_:
{
return v___x_65_;
}
}
}
else
{
lean_object* v_a_69_; lean_object* v___x_71_; uint8_t v_isShared_72_; uint8_t v_isSharedCheck_76_; 
lean_dec(v_a_52_);
v_a_69_ = lean_ctor_get(v___x_60_, 0);
v_isSharedCheck_76_ = !lean_is_exclusive(v___x_60_);
if (v_isSharedCheck_76_ == 0)
{
v___x_71_ = v___x_60_;
v_isShared_72_ = v_isSharedCheck_76_;
goto v_resetjp_70_;
}
else
{
lean_inc(v_a_69_);
lean_dec(v___x_60_);
v___x_71_ = lean_box(0);
v_isShared_72_ = v_isSharedCheck_76_;
goto v_resetjp_70_;
}
v_resetjp_70_:
{
lean_object* v___x_74_; 
if (v_isShared_72_ == 0)
{
v___x_74_ = v___x_71_;
goto v_reusejp_73_;
}
else
{
lean_object* v_reuseFailAlloc_75_; 
v_reuseFailAlloc_75_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_75_, 0, v_a_69_);
v___x_74_ = v_reuseFailAlloc_75_;
goto v_reusejp_73_;
}
v_reusejp_73_:
{
return v___x_74_;
}
}
}
}
else
{
lean_object* v___x_77_; lean_object* v___x_79_; 
lean_dec_ref_known(v___x_51_, 1);
lean_dec_ref(v___x_11_);
v___x_77_ = lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__2;
if (v_isShared_16_ == 0)
{
lean_ctor_set(v___x_15_, 0, v___x_77_);
v___x_79_ = v___x_15_;
goto v_reusejp_78_;
}
else
{
lean_object* v_reuseFailAlloc_80_; 
v_reuseFailAlloc_80_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_80_, 0, v___x_77_);
v___x_79_ = v_reuseFailAlloc_80_;
goto v_reusejp_78_;
}
v_reusejp_78_:
{
return v___x_79_;
}
}
}
}
}
else
{
lean_object* v_a_82_; lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_89_; 
lean_dec_ref(v___x_11_);
v_a_82_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_89_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_89_ == 0)
{
v___x_84_ = v___x_12_;
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
else
{
lean_inc(v_a_82_);
lean_dec(v___x_12_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_87_; 
if (v_isShared_85_ == 0)
{
v___x_87_ = v___x_84_;
goto v_reusejp_86_;
}
else
{
lean_object* v_reuseFailAlloc_88_; 
v_reuseFailAlloc_88_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_88_, 0, v_a_82_);
v___x_87_ = v_reuseFailAlloc_88_;
goto v_reusejp_86_;
}
v_reusejp_86_:
{
return v___x_87_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VmVerifier_Executable_main___boxed(lean_object* v_a_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_openvm_x2dfv_VmVerifier_Executable_main();
return v_res_91_;
}
}
LEAN_EXPORT lean_object* _lean_main(){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_openvm_x2dfv_VmVerifier_Executable_main();
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_main___boxed(lean_object* v_a_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = _lean_main();
return v_res_95_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VmVerifier_Spec_Runtime(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VmVerifier_Main(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VmVerifier_Spec_Runtime(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__1 = _init_lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__1();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__1);
lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__2 = _init_lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__2();
lean_mark_persistent(lp_openvm_x2dfv_VmVerifier_Executable_main___boxed__const__2);
return lean_io_result_mk_ok(lean_box(0));
}
char ** lean_setup_args(int argc, char ** argv);
#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#endif
lean_object* run_main(int argc, char ** argv) {
    return _lean_main();
}
int main(int argc, char ** argv) {
#if defined(WIN32) || defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS);
  SetConsoleOutputCP(CP_UTF8);
#endif
  lean_object* res;
  argv = lean_setup_args(argc, argv);
  res = initialize_openvm_x2dfv_VmVerifier_Main(1 /* builtin */);
  lean_io_mark_end_initialization();
  if (lean_io_result_is_ok(res)) {
    lean_dec_ref(res);
    lean_init_task_manager();
    res = lean_run_main(&run_main, argc, argv);
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
