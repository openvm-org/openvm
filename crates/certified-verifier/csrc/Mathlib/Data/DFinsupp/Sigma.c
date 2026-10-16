// Lean compiler output
// Module: Mathlib.Data.DFinsupp.Sigma
// Imports: public import Init public meta import Init public import Mathlib.Data.DFinsupp.Module public import Mathlib.Data.Fintype.Quotient
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
lean_object* lp_mathlib_Multiset_filterMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
lean_object* lp_mathlib_Quotient_finChoice___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_bind___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__3___boxed(lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_sigmaCurry___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_sigmaCurry___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__0(lean_object* v_i_1_, lean_object* v_toFun_2_, lean_object* v_j_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4_, 0, v_i_1_);
lean_ctor_set(v___x_4_, 1, v_j_3_);
v___x_5_ = lean_apply_1(v_toFun_2_, v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__1(lean_object* v_inst_6_, lean_object* v_i_7_, lean_object* v_x_8_){
_start:
{
lean_object* v_fst_9_; lean_object* v_snd_10_; lean_object* v___x_11_; uint8_t v___x_12_; 
v_fst_9_ = lean_ctor_get(v_x_8_, 0);
lean_inc(v_fst_9_);
v_snd_10_ = lean_ctor_get(v_x_8_, 1);
lean_inc(v_snd_10_);
lean_dec_ref(v_x_8_);
v___x_11_ = lean_apply_2(v_inst_6_, v_fst_9_, v_i_7_);
v___x_12_ = lean_unbox(v___x_11_);
if (v___x_12_ == 0)
{
lean_object* v___x_13_; 
lean_dec(v_snd_10_);
v___x_13_ = lean_box(0);
return v___x_13_;
}
else
{
lean_object* v___x_14_; 
v___x_14_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_14_, 0, v_snd_10_);
return v___x_14_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__2(lean_object* v_toFun_15_, lean_object* v_inst_16_, lean_object* v_support_x27_17_, lean_object* v_i_18_){
_start:
{
lean_object* v___f_19_; lean_object* v___f_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
lean_inc(v_i_18_);
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__0), 3, 2);
lean_closure_set(v___f_19_, 0, v_i_18_);
lean_closure_set(v___f_19_, 1, v_toFun_15_);
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__1), 3, 2);
lean_closure_set(v___f_20_, 0, v_inst_16_);
lean_closure_set(v___f_20_, 1, v_i_18_);
v___x_21_ = lp_mathlib_Multiset_filterMap___redArg(v___f_20_, v_support_x27_17_);
v___x_22_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_22_, 0, v___f_19_);
lean_ctor_set(v___x_22_, 1, v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__3(lean_object* v_self_23_){
_start:
{
lean_object* v_fst_24_; 
v_fst_24_ = lean_ctor_get(v_self_23_, 0);
lean_inc(v_fst_24_);
return v_fst_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__3___boxed(lean_object* v_self_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__3(v_self_25_);
lean_dec_ref(v_self_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___redArg(lean_object* v_inst_28_, lean_object* v_f_29_){
_start:
{
lean_object* v_toFun_30_; lean_object* v_support_x27_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_41_; 
v_toFun_30_ = lean_ctor_get(v_f_29_, 0);
v_support_x27_31_ = lean_ctor_get(v_f_29_, 1);
v_isSharedCheck_41_ = !lean_is_exclusive(v_f_29_);
if (v_isSharedCheck_41_ == 0)
{
v___x_33_ = v_f_29_;
v_isShared_34_ = v_isSharedCheck_41_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_support_x27_31_);
lean_inc(v_toFun_30_);
lean_dec(v_f_29_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_41_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___f_35_; lean_object* v___f_36_; lean_object* v___x_37_; lean_object* v___x_39_; 
lean_inc(v_support_x27_31_);
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaCurry___redArg___lam__2), 4, 3);
lean_closure_set(v___f_35_, 0, v_toFun_30_);
lean_closure_set(v___f_35_, 1, v_inst_28_);
lean_closure_set(v___f_35_, 2, v_support_x27_31_);
v___f_36_ = ((lean_object*)(lp_mathlib_DFinsupp_sigmaCurry___redArg___closed__0));
v___x_37_ = lp_mathlib_Multiset_map___redArg(v___f_36_, v_support_x27_31_);
if (v_isShared_34_ == 0)
{
lean_ctor_set(v___x_33_, 1, v___x_37_);
lean_ctor_set(v___x_33_, 0, v___f_35_);
v___x_39_ = v___x_33_;
goto v_reusejp_38_;
}
else
{
lean_object* v_reuseFailAlloc_40_; 
v_reuseFailAlloc_40_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_40_, 0, v___f_35_);
lean_ctor_set(v_reuseFailAlloc_40_, 1, v___x_37_);
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
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry(lean_object* v_00_u03b9_42_, lean_object* v_00_u03b1_43_, lean_object* v_00_u03b4_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_f_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_DFinsupp_sigmaCurry___redArg(v_inst_45_, v_f_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurry___boxed(lean_object* v_00_u03b9_49_, lean_object* v_00_u03b1_50_, lean_object* v_00_u03b4_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_f_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_DFinsupp_sigmaCurry(v_00_u03b9_49_, v_00_u03b1_50_, v_00_u03b4_51_, v_inst_52_, v_inst_53_, v_f_54_);
lean_dec(v_inst_53_);
return v_res_55_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__0(lean_object* v_inst_56_, lean_object* v_a_57_, lean_object* v_b_58_){
_start:
{
lean_object* v___x_59_; uint8_t v___x_60_; 
v___x_59_ = lean_apply_2(v_inst_56_, v_a_57_, v_b_58_);
v___x_60_ = lean_unbox(v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__0___boxed(lean_object* v_inst_61_, lean_object* v_a_62_, lean_object* v_b_63_){
_start:
{
uint8_t v_res_64_; lean_object* v_r_65_; 
v_res_64_ = lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__0(v_inst_61_, v_a_62_, v_b_63_);
v_r_65_ = lean_box(v_res_64_);
return v_r_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__1(lean_object* v_toFun_66_, lean_object* v_i_67_){
_start:
{
lean_object* v___x_68_; lean_object* v_support_x27_69_; 
v___x_68_ = lean_apply_1(v_toFun_66_, v_i_67_);
v_support_x27_69_ = lean_ctor_get(v___x_68_, 1);
lean_inc(v_support_x27_69_);
lean_dec_ref(v___x_68_);
return v_support_x27_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__2(lean_object* v_toFun_70_, lean_object* v_i_71_){
_start:
{
lean_object* v_fst_72_; lean_object* v_snd_73_; lean_object* v___x_74_; lean_object* v_toFun_75_; lean_object* v___x_76_; 
v_fst_72_ = lean_ctor_get(v_i_71_, 0);
lean_inc(v_fst_72_);
v_snd_73_ = lean_ctor_get(v_i_71_, 1);
lean_inc(v_snd_73_);
lean_dec_ref(v_i_71_);
v___x_74_ = lean_apply_1(v_toFun_70_, v_fst_72_);
v_toFun_75_ = lean_ctor_get(v___x_74_, 0);
lean_inc(v_toFun_75_);
lean_dec_ref(v___x_74_);
v___x_76_ = lean_apply_1(v_toFun_75_, v_snd_73_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__3(lean_object* v_i_77_, lean_object* v_snd_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v_i_77_);
lean_ctor_set(v___x_79_, 1, v_snd_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__4(lean_object* v___x_80_, lean_object* v___f_81_, lean_object* v___f_82_, lean_object* v_i_83_){
_start:
{
lean_object* v___f_84_; lean_object* v___x_109__overap_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
lean_inc(v_i_83_);
v___f_84_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__3), 2, 1);
lean_closure_set(v___f_84_, 0, v_i_83_);
v___x_109__overap_85_ = lp_mathlib_Quotient_finChoice___redArg(v___x_80_, v___f_81_, v___f_82_);
v___x_86_ = lean_apply_1(v___x_109__overap_85_, v_i_83_);
v___x_87_ = lp_mathlib_Multiset_map___redArg(v___f_84_, v___x_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___redArg(lean_object* v_inst_88_, lean_object* v_f_89_){
_start:
{
lean_object* v_toFun_90_; lean_object* v_support_x27_91_; lean_object* v___x_93_; uint8_t v_isShared_94_; uint8_t v_isSharedCheck_105_; 
v_toFun_90_ = lean_ctor_get(v_f_89_, 0);
v_support_x27_91_ = lean_ctor_get(v_f_89_, 1);
v_isSharedCheck_105_ = !lean_is_exclusive(v_f_89_);
if (v_isSharedCheck_105_ == 0)
{
v___x_93_ = v_f_89_;
v_isShared_94_ = v_isSharedCheck_105_;
goto v_resetjp_92_;
}
else
{
lean_inc(v_support_x27_91_);
lean_inc(v_toFun_90_);
lean_dec(v_f_89_);
v___x_93_ = lean_box(0);
v_isShared_94_ = v_isSharedCheck_105_;
goto v_resetjp_92_;
}
v_resetjp_92_:
{
lean_object* v___f_95_; lean_object* v___f_96_; lean_object* v___f_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___f_100_; lean_object* v___x_101_; lean_object* v___x_103_; 
lean_inc_ref(v_inst_88_);
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_95_, 0, v_inst_88_);
lean_inc(v_toFun_90_);
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__1), 2, 1);
lean_closure_set(v___f_96_, 0, v_toFun_90_);
v___f_97_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__2), 2, 1);
lean_closure_set(v___f_97_, 0, v_toFun_90_);
v___x_98_ = lp_mathlib_List_dedup___redArg(v_inst_88_, v_support_x27_91_);
v___x_99_ = lp_mathlib_Multiset_attach___redArg(v___x_98_);
lean_inc(v___x_99_);
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaUncurry___redArg___lam__4), 4, 3);
lean_closure_set(v___f_100_, 0, v___x_99_);
lean_closure_set(v___f_100_, 1, v___f_95_);
lean_closure_set(v___f_100_, 2, v___f_96_);
v___x_101_ = lp_mathlib_Multiset_bind___redArg(v___x_99_, v___f_100_);
if (v_isShared_94_ == 0)
{
lean_ctor_set(v___x_93_, 1, v___x_101_);
lean_ctor_set(v___x_93_, 0, v___f_97_);
v___x_103_ = v___x_93_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v___f_97_);
lean_ctor_set(v_reuseFailAlloc_104_, 1, v___x_101_);
v___x_103_ = v_reuseFailAlloc_104_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
return v___x_103_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry(lean_object* v_00_u03b9_106_, lean_object* v_00_u03b1_107_, lean_object* v_00_u03b4_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_f_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_DFinsupp_sigmaUncurry___redArg(v_inst_109_, v_f_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaUncurry___boxed(lean_object* v_00_u03b9_113_, lean_object* v_00_u03b1_114_, lean_object* v_00_u03b4_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_f_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_DFinsupp_sigmaUncurry(v_00_u03b9_113_, v_00_u03b1_114_, v_00_u03b4_115_, v_inst_116_, v_inst_117_, v_f_118_);
lean_dec(v_inst_117_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryEquiv___redArg(lean_object* v_inst_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
lean_inc(v_inst_121_);
lean_inc_ref(v_inst_120_);
v___x_122_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaCurry___boxed), 6, 5);
lean_closure_set(v___x_122_, 0, lean_box(0));
lean_closure_set(v___x_122_, 1, lean_box(0));
lean_closure_set(v___x_122_, 2, lean_box(0));
lean_closure_set(v___x_122_, 3, v_inst_120_);
lean_closure_set(v___x_122_, 4, v_inst_121_);
v___x_123_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaUncurry___boxed), 6, 5);
lean_closure_set(v___x_123_, 0, lean_box(0));
lean_closure_set(v___x_123_, 1, lean_box(0));
lean_closure_set(v___x_123_, 2, lean_box(0));
lean_closure_set(v___x_123_, 3, v_inst_120_);
lean_closure_set(v___x_123_, 4, v_inst_121_);
v___x_124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_124_, 0, v___x_122_);
lean_ctor_set(v___x_124_, 1, v___x_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryEquiv(lean_object* v_00_u03b9_125_, lean_object* v_00_u03b1_126_, lean_object* v_00_u03b4_127_, lean_object* v_inst_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_mathlib_DFinsupp_sigmaCurryEquiv___redArg(v_inst_128_, v_inst_129_);
return v___x_130_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Quotient(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Quotient(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Quotient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(builtin);
}
#ifdef __cplusplus
}
#endif
