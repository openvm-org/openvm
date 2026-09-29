// Lean compiler output
// Module: Mathlib.Algebra.DirectSum.Decomposition
// Imports: public import Init public meta import Init public import Mathlib.Algebra.DirectSum.Module public import Mathlib.Algebra.Module.Submodule.Basic
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
lean_object* lp_mathlib_DirectSum_coeAddMonoidHom___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofAddHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decompose___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decompose___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decompose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decompose___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAddEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeLinearEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofAddHom___redArg___lam__0(lean_object* v_decompose_1_, lean_object* v___y_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_decompose_1_, v___y_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofAddHom___redArg(lean_object* v_decompose_4_){
_start:
{
lean_object* v___f_5_; 
v___f_5_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_Decomposition_ofAddHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_5_, 0, v_decompose_4_);
return v___f_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofAddHom(lean_object* v_00_u03b9_6_, lean_object* v_M_7_, lean_object* v_00_u03c3_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_00_u2133_13_, lean_object* v_decompose_14_, lean_object* v_h__left__inv_15_, lean_object* v_h__right__inv_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_Decomposition_ofAddHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_17_, 0, v_decompose_14_);
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofAddHom___boxed(lean_object* v_00_u03b9_18_, lean_object* v_M_19_, lean_object* v_00_u03c3_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_00_u2133_25_, lean_object* v_decompose_26_, lean_object* v_h__left__inv_27_, lean_object* v_h__right__inv_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_DirectSum_Decomposition_ofAddHom(v_00_u03b9_18_, v_M_19_, v_00_u03c3_20_, v_inst_21_, v_inst_22_, v_inst_23_, v_inst_24_, v_00_u2133_25_, v_decompose_26_, v_h__left__inv_27_, v_h__right__inv_28_);
lean_dec(v_00_u2133_25_);
lean_dec_ref(v_inst_22_);
lean_dec_ref(v_inst_21_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decompose___redArg___lam__0(lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v___y_32_){
_start:
{
lean_object* v___x_32__overap_33_; lean_object* v___x_34_; 
v___x_32__overap_33_ = lp_mathlib_DirectSum_coeAddMonoidHom___redArg(v_inst_30_, v_inst_31_);
v___x_34_ = lean_apply_1(v___x_32__overap_33_, v___y_32_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decompose___redArg(lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___f_38_; lean_object* v___x_39_; 
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_decompose___redArg___lam__0), 3, 2);
lean_closure_set(v___f_38_, 0, v_inst_35_);
lean_closure_set(v___f_38_, 1, v_inst_36_);
v___x_39_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_39_, 0, v_inst_37_);
lean_ctor_set(v___x_39_, 1, v___f_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decompose(lean_object* v_00_u03b9_40_, lean_object* v_M_41_, lean_object* v_00_u03c3_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_00_u2133_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_DirectSum_decompose___redArg(v_inst_43_, v_inst_44_, v_inst_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decompose___boxed(lean_object* v_00_u03b9_50_, lean_object* v_M_51_, lean_object* v_00_u03c3_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_00_u2133_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_DirectSum_decompose(v_00_u03b9_50_, v_M_51_, v_00_u03c3_52_, v_inst_53_, v_inst_54_, v_inst_55_, v_inst_56_, v_00_u2133_57_, v_inst_58_);
lean_dec(v_00_u2133_57_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAddEquiv___redArg(lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_63_ = lp_mathlib_DirectSum_decompose___redArg(v_inst_60_, v_inst_61_, v_inst_62_);
v___x_64_ = lp_mathlib_Equiv_symm___redArg(v___x_63_);
v___x_65_ = lp_mathlib_Equiv_symm___redArg(v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAddEquiv(lean_object* v_00_u03b9_66_, lean_object* v_M_67_, lean_object* v_00_u03c3_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_00_u2133_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_DirectSum_decomposeAddEquiv___redArg(v_inst_69_, v_inst_70_, v_inst_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAddEquiv___boxed(lean_object* v_00_u03b9_76_, lean_object* v_M_77_, lean_object* v_00_u03c3_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_00_u2133_83_, lean_object* v_inst_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_DirectSum_decomposeAddEquiv(v_00_u03b9_76_, v_M_77_, v_00_u03c3_78_, v_inst_79_, v_inst_80_, v_inst_81_, v_inst_82_, v_00_u2133_83_, v_inst_84_);
lean_dec(v_00_u2133_83_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofLinearMap___redArg(lean_object* v_decompose_86_){
_start:
{
lean_object* v___f_87_; 
v___f_87_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_Decomposition_ofAddHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_87_, 0, v_decompose_86_);
return v___f_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofLinearMap(lean_object* v_00_u03b9_88_, lean_object* v_R_89_, lean_object* v_M_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_00_u2133_95_, lean_object* v_decompose_96_, lean_object* v_h__left__inv_97_, lean_object* v_h__right__inv_98_){
_start:
{
lean_object* v___f_99_; 
v___f_99_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_Decomposition_ofAddHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_99_, 0, v_decompose_96_);
return v___f_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_Decomposition_ofLinearMap___boxed(lean_object* v_00_u03b9_100_, lean_object* v_R_101_, lean_object* v_M_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_00_u2133_107_, lean_object* v_decompose_108_, lean_object* v_h__left__inv_109_, lean_object* v_h__right__inv_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_DirectSum_Decomposition_ofLinearMap(v_00_u03b9_100_, v_R_101_, v_M_102_, v_inst_103_, v_inst_104_, v_inst_105_, v_inst_106_, v_00_u2133_107_, v_decompose_108_, v_h__left__inv_109_, v_h__right__inv_110_);
lean_dec_ref(v_00_u2133_107_);
lean_dec(v_inst_106_);
lean_dec_ref(v_inst_105_);
lean_dec_ref(v_inst_104_);
lean_dec_ref(v_inst_103_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeLinearEquiv___redArg(lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v_toFun_117_; lean_object* v_invFun_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_126_; 
v___x_115_ = lp_mathlib_DirectSum_decomposeAddEquiv___redArg(v_inst_112_, v_inst_113_, v_inst_114_);
v___x_116_ = lp_mathlib_Equiv_symm___redArg(v___x_115_);
v_toFun_117_ = lean_ctor_get(v___x_116_, 0);
v_invFun_118_ = lean_ctor_get(v___x_116_, 1);
v_isSharedCheck_126_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_126_ == 0)
{
v___x_120_ = v___x_116_;
v_isShared_121_ = v_isSharedCheck_126_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_invFun_118_);
lean_inc(v_toFun_117_);
lean_dec(v___x_116_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_126_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_123_; 
if (v_isShared_121_ == 0)
{
v___x_123_ = v___x_120_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v_toFun_117_);
lean_ctor_set(v_reuseFailAlloc_125_, 1, v_invFun_118_);
v___x_123_ = v_reuseFailAlloc_125_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
lean_object* v___x_124_; 
v___x_124_ = lp_mathlib_LinearEquiv_symm___redArg(v___x_123_);
return v___x_124_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeLinearEquiv(lean_object* v_00_u03b9_127_, lean_object* v_R_128_, lean_object* v_M_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_00_u2133_134_, lean_object* v_inst_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lp_mathlib_DirectSum_decomposeLinearEquiv___redArg(v_inst_130_, v_inst_132_, v_inst_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeLinearEquiv___boxed(lean_object* v_00_u03b9_137_, lean_object* v_R_138_, lean_object* v_M_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_00_u2133_144_, lean_object* v_inst_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_DirectSum_decomposeLinearEquiv(v_00_u03b9_137_, v_R_138_, v_M_139_, v_inst_140_, v_inst_141_, v_inst_142_, v_inst_143_, v_00_u2133_144_, v_inst_145_);
lean_dec_ref(v_00_u2133_144_);
lean_dec(v_inst_143_);
lean_dec_ref(v_inst_141_);
return v_res_146_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Module(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(builtin);
}
#ifdef __cplusplus
}
#endif
