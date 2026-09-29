// Lean compiler output
// Module: Mathlib.Algebra.DirectSum.Algebra
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Defs public import Mathlib.Algebra.DirectSum.Module public import Mathlib.Algebra.DirectSum.Ring
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
lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_toSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_of___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAlgebra___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulLHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulLHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulLHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulLHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_directSumGAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_directSumGAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_directSumGAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_directSumGAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAlgebra___redArg___lam__0(lean_object* v_inst_1_, lean_object* v___x_2_, lean_object* v___y_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lp_mathlib_OneHom_comp___redArg___lam__0(v_inst_1_, v___x_2_, v___y_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAlgebra___redArg(lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v_toZero_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_23_; 
v___x_11_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_8_);
v___x_12_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_11_);
v_toZero_13_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_23_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_23_ == 0)
{
lean_object* v_unused_24_; 
v_unused_24_ = lean_ctor_get(v___x_12_, 1);
lean_dec(v_unused_24_);
v___x_15_ = v___x_12_;
v_isShared_16_ = v_isSharedCheck_23_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_toZero_13_);
lean_dec(v___x_12_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_23_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___f_19_; lean_object* v___x_21_; 
lean_inc_ref(v_inst_6_);
v___x_17_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instSMulOfModule___aux__1___boxed), 8, 6);
lean_closure_set(v___x_17_, 0, lean_box(0));
lean_closure_set(v___x_17_, 1, lean_box(0));
lean_closure_set(v___x_17_, 2, lean_box(0));
lean_closure_set(v___x_17_, 3, v_inst_5_);
lean_closure_set(v___x_17_, 4, v_inst_6_);
lean_closure_set(v___x_17_, 5, v_inst_7_);
v___x_18_ = lp_mathlib_DirectSum_of___redArg(v_inst_6_, v_inst_10_, v_toZero_13_);
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instAlgebra___redArg___lam__0), 3, 2);
lean_closure_set(v___f_19_, 0, v_inst_9_);
lean_closure_set(v___f_19_, 1, v___x_18_);
if (v_isShared_16_ == 0)
{
lean_ctor_set(v___x_15_, 1, v___f_19_);
lean_ctor_set(v___x_15_, 0, v___x_17_);
v___x_21_ = v___x_15_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v___x_17_);
lean_ctor_set(v_reuseFailAlloc_22_, 1, v___f_19_);
v___x_21_ = v_reuseFailAlloc_22_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
return v___x_21_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAlgebra___redArg___boxed(lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_DirectSum_instAlgebra___redArg(v_inst_25_, v_inst_26_, v_inst_27_, v_inst_28_, v_inst_29_, v_inst_30_);
lean_dec_ref(v_inst_28_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAlgebra(lean_object* v_00_u03b9_32_, lean_object* v_R_33_, lean_object* v_A_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_DirectSum_instAlgebra___redArg(v_inst_35_, v_inst_36_, v_inst_37_, v_inst_38_, v_inst_40_, v_inst_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instAlgebra___boxed(lean_object* v_00_u03b9_43_, lean_object* v_R_44_, lean_object* v_A_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_DirectSum_instAlgebra(v_00_u03b9_43_, v_R_44_, v_A_45_, v_inst_46_, v_inst_47_, v_inst_48_, v_inst_49_, v_inst_50_, v_inst_51_, v_inst_52_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAlgebra___redArg___lam__0(lean_object* v_f_54_, lean_object* v_i_55_, lean_object* v___y_56_){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = lean_apply_1(v_f_54_, v_i_55_);
v___x_58_ = lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0(v___x_57_, v___y_56_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAlgebra___redArg___lam__1(lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v___f_62_, lean_object* v___y_63_){
_start:
{
lean_object* v___x_77__overap_64_; lean_object* v___x_65_; 
v___x_77__overap_64_ = lp_mathlib_DirectSum_toSemiring___redArg(v_inst_59_, v_inst_60_, v_inst_61_, v___f_62_);
v___x_65_ = lean_apply_1(v___x_77__overap_64_, v___y_63_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAlgebra___redArg(lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_f_69_){
_start:
{
lean_object* v___f_70_; lean_object* v___f_71_; 
v___f_70_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_toAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(v___f_70_, 0, v_f_69_);
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_toAlgebra___redArg___lam__1), 5, 4);
lean_closure_set(v___f_71_, 0, v_inst_68_);
lean_closure_set(v___f_71_, 1, v_inst_66_);
lean_closure_set(v___f_71_, 2, v_inst_67_);
lean_closure_set(v___f_71_, 3, v___f_70_);
return v___f_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAlgebra(lean_object* v_00_u03b9_72_, lean_object* v_R_73_, lean_object* v_A_74_, lean_object* v_B_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_f_85_, lean_object* v_hone_86_, lean_object* v_hmul_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib_DirectSum_toAlgebra___redArg(v_inst_77_, v_inst_81_, v_inst_84_, v_f_85_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toAlgebra___boxed(lean_object* v_00_u03b9_89_, lean_object* v_R_90_, lean_object* v_A_91_, lean_object* v_B_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_f_102_, lean_object* v_hone_103_, lean_object* v_hmul_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib_DirectSum_toAlgebra(v_00_u03b9_89_, v_R_90_, v_A_91_, v_B_92_, v_inst_93_, v_inst_94_, v_inst_95_, v_inst_96_, v_inst_97_, v_inst_98_, v_inst_99_, v_inst_100_, v_inst_101_, v_f_102_, v_hone_103_, v_hmul_104_);
lean_dec_ref(v_inst_100_);
lean_dec(v_inst_99_);
lean_dec_ref(v_inst_97_);
lean_dec_ref(v_inst_96_);
lean_dec(v_inst_95_);
lean_dec_ref(v_inst_93_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulLHom___redArg___lam__0(lean_object* v_toGNonUnitalNonAssocSemiring_106_, lean_object* v_i_107_, lean_object* v_j_108_, lean_object* v_a_109_, lean_object* v___y_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lean_apply_4(v_toGNonUnitalNonAssocSemiring_106_, v_i_107_, v_j_108_, v_a_109_, v___y_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulLHom___redArg(lean_object* v_inst_112_, lean_object* v_i_113_, lean_object* v_j_114_){
_start:
{
lean_object* v_toGNonUnitalNonAssocSemiring_115_; lean_object* v___f_116_; 
v_toGNonUnitalNonAssocSemiring_115_ = lean_ctor_get(v_inst_112_, 0);
lean_inc(v_toGNonUnitalNonAssocSemiring_115_);
lean_dec_ref(v_inst_112_);
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_gMulLHom___redArg___lam__0), 5, 3);
lean_closure_set(v___f_116_, 0, v_toGNonUnitalNonAssocSemiring_115_);
lean_closure_set(v___f_116_, 1, v_i_113_);
lean_closure_set(v___f_116_, 2, v_j_114_);
return v___f_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulLHom(lean_object* v_00_u03b9_117_, lean_object* v_R_118_, lean_object* v_A_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_i_126_, lean_object* v_j_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_mathlib_DirectSum_gMulLHom___redArg(v_inst_124_, v_i_126_, v_j_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulLHom___boxed(lean_object* v_00_u03b9_129_, lean_object* v_R_130_, lean_object* v_A_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_i_138_, lean_object* v_j_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_DirectSum_gMulLHom(v_00_u03b9_129_, v_R_130_, v_A_131_, v_inst_132_, v_inst_133_, v_inst_134_, v_inst_135_, v_inst_136_, v_inst_137_, v_i_138_, v_j_139_);
lean_dec(v_inst_137_);
lean_dec_ref(v_inst_135_);
lean_dec(v_inst_134_);
lean_dec_ref(v_inst_133_);
lean_dec_ref(v_inst_132_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_directSumGAlgebra___redArg(lean_object* v_inst_141_){
_start:
{
lean_object* v_algebraMap_142_; 
v_algebraMap_142_ = lean_ctor_get(v_inst_141_, 1);
lean_inc(v_algebraMap_142_);
return v_algebraMap_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_directSumGAlgebra___redArg___boxed(lean_object* v_inst_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_Algebra_directSumGAlgebra___redArg(v_inst_143_);
lean_dec_ref(v_inst_143_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_directSumGAlgebra(lean_object* v_00_u03b9_145_, lean_object* v_R_146_, lean_object* v_A_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_){
_start:
{
lean_object* v_algebraMap_152_; 
v_algebraMap_152_ = lean_ctor_get(v_inst_151_, 1);
lean_inc(v_algebraMap_152_);
return v_algebraMap_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_directSumGAlgebra___boxed(lean_object* v_00_u03b9_153_, lean_object* v_R_154_, lean_object* v_A_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_Algebra_directSumGAlgebra(v_00_u03b9_153_, v_R_154_, v_A_155_, v_inst_156_, v_inst_157_, v_inst_158_, v_inst_159_);
lean_dec_ref(v_inst_159_);
lean_dec_ref(v_inst_158_);
lean_dec_ref(v_inst_157_);
lean_dec_ref(v_inst_156_);
return v_res_160_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Module(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(builtin);
}
#ifdef __cplusplus
}
#endif
