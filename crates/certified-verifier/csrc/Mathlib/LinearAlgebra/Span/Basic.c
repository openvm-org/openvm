// Lean compiler output
// Module: Mathlib.LinearAlgebra.Span.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.NonZeroDivisors public import Mathlib.Algebra.Module.Prod public import Mathlib.Algebra.Module.Submodule.Equiv public import Mathlib.Algebra.Module.Submodule.Pointwise public import Mathlib.LinearAlgebra.Span.Defs public import Mathlib.Order.CompactlyGenerated.Basic public import Mathlib.Order.BourbakiWitt import Mathlib.Algebra.Field.Basic import Mathlib.Algebra.Module.Submodule.EqLocus import Mathlib.Algebra.Module.Torsion.Field
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
lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_LinearMap_smulRight___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inclusionSpan___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inclusionSpan___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Submodule_inclusionSpan___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_inclusionSpan___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_inclusionSpan___closed__0 = (const lean_object*)&lp_mathlib_Submodule_inclusionSpan___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inclusionSpan(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inclusionSpan___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prodEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prodEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Submodule_prodEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_prodEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_prodEquiv___closed__0 = (const lean_object*)&lp_mathlib_Submodule_prodEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Submodule_prodEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_prodEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_prodEquiv___closed__1 = (const lean_object*)&lp_mathlib_Submodule_prodEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_Submodule_prodEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_prodEquiv___closed__0_value),((lean_object*)&lp_mathlib_Submodule_prodEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_Submodule_prodEquiv___closed__2 = (const lean_object*)&lp_mathlib_Submodule_prodEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_LinearMap_toSpanSingleton___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LinearMap_toSpanSingleton___redArg___closed__0 = (const lean_object*)&lp_mathlib_LinearMap_toSpanSingleton___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toSpanSingleton___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toSpanSingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toSpanSingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inclusionSpan___lam__0(lean_object* v_x_1_){
_start:
{
lean_inc(v_x_1_);
return v_x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inclusionSpan___lam__0___boxed(lean_object* v_x_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Submodule_inclusionSpan___lam__0(v_x_2_);
lean_dec(v_x_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inclusionSpan(lean_object* v_R_5_, lean_object* v_M_6_, lean_object* v_S_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_p_15_){
_start:
{
lean_object* v___f_16_; 
v___f_16_ = ((lean_object*)(lp_mathlib_Submodule_inclusionSpan___closed__0));
return v___f_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inclusionSpan___boxed(lean_object* v_R_17_, lean_object* v_M_18_, lean_object* v_S_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_p_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_Submodule_inclusionSpan(v_R_17_, v_M_18_, v_S_19_, v_inst_20_, v_inst_21_, v_inst_22_, v_inst_23_, v_inst_24_, v_inst_25_, v_inst_26_, v_p_27_);
lean_dec(v_inst_25_);
lean_dec(v_inst_24_);
lean_dec_ref(v_inst_23_);
lean_dec(v_inst_22_);
lean_dec_ref(v_inst_21_);
lean_dec_ref(v_inst_20_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prod(lean_object* v_R_29_, lean_object* v_M_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_p_34_, lean_object* v_M_x27_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_q_u2081_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lean_box(0);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prod___boxed(lean_object* v_R_40_, lean_object* v_M_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_p_45_, lean_object* v_M_x27_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_q_u2081_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_Submodule_prod(v_R_40_, v_M_41_, v_inst_42_, v_inst_43_, v_inst_44_, v_p_45_, v_M_x27_46_, v_inst_47_, v_inst_48_, v_q_u2081_49_);
lean_dec(v_inst_48_);
lean_dec_ref(v_inst_47_);
lean_dec(v_inst_44_);
lean_dec_ref(v_inst_43_);
lean_dec_ref(v_inst_42_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prodEquiv___lam__0(lean_object* v_x_51_){
_start:
{
lean_object* v_fst_52_; lean_object* v_snd_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_60_; 
v_fst_52_ = lean_ctor_get(v_x_51_, 0);
v_snd_53_ = lean_ctor_get(v_x_51_, 1);
v_isSharedCheck_60_ = !lean_is_exclusive(v_x_51_);
if (v_isSharedCheck_60_ == 0)
{
v___x_55_ = v_x_51_;
v_isShared_56_ = v_isSharedCheck_60_;
goto v_resetjp_54_;
}
else
{
lean_inc(v_snd_53_);
lean_inc(v_fst_52_);
lean_dec(v_x_51_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_60_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___x_58_; 
if (v_isShared_56_ == 0)
{
v___x_58_ = v___x_55_;
goto v_reusejp_57_;
}
else
{
lean_object* v_reuseFailAlloc_59_; 
v_reuseFailAlloc_59_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_59_, 0, v_fst_52_);
lean_ctor_set(v_reuseFailAlloc_59_, 1, v_snd_53_);
v___x_58_ = v_reuseFailAlloc_59_;
goto v_reusejp_57_;
}
v_reusejp_57_:
{
return v___x_58_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prodEquiv___lam__1(lean_object* v_y_61_){
_start:
{
lean_object* v_fst_62_; lean_object* v_snd_63_; lean_object* v___x_65_; uint8_t v_isShared_66_; uint8_t v_isSharedCheck_70_; 
v_fst_62_ = lean_ctor_get(v_y_61_, 0);
v_snd_63_ = lean_ctor_get(v_y_61_, 1);
v_isSharedCheck_70_ = !lean_is_exclusive(v_y_61_);
if (v_isSharedCheck_70_ == 0)
{
v___x_65_ = v_y_61_;
v_isShared_66_ = v_isSharedCheck_70_;
goto v_resetjp_64_;
}
else
{
lean_inc(v_snd_63_);
lean_inc(v_fst_62_);
lean_dec(v_y_61_);
v___x_65_ = lean_box(0);
v_isShared_66_ = v_isSharedCheck_70_;
goto v_resetjp_64_;
}
v_resetjp_64_:
{
lean_object* v___x_68_; 
if (v_isShared_66_ == 0)
{
v___x_68_ = v___x_65_;
goto v_reusejp_67_;
}
else
{
lean_object* v_reuseFailAlloc_69_; 
v_reuseFailAlloc_69_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_69_, 0, v_fst_62_);
lean_ctor_set(v_reuseFailAlloc_69_, 1, v_snd_63_);
v___x_68_ = v_reuseFailAlloc_69_;
goto v_reusejp_67_;
}
v_reusejp_67_:
{
return v___x_68_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prodEquiv(lean_object* v_R_76_, lean_object* v_M_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_M_x27_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_p_84_, lean_object* v_q_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = ((lean_object*)(lp_mathlib_Submodule_prodEquiv___closed__2));
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_prodEquiv___boxed(lean_object* v_R_87_, lean_object* v_M_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_M_x27_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_p_95_, lean_object* v_q_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_Submodule_prodEquiv(v_R_87_, v_M_88_, v_inst_89_, v_inst_90_, v_inst_91_, v_M_x27_92_, v_inst_93_, v_inst_94_, v_p_95_, v_q_96_);
lean_dec(v_inst_94_);
lean_dec_ref(v_inst_93_);
lean_dec(v_inst_91_);
lean_dec_ref(v_inst_90_);
lean_dec_ref(v_inst_89_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toSpanSingleton___redArg(lean_object* v_inst_99_, lean_object* v_x_100_){
_start:
{
lean_object* v___f_101_; lean_object* v___f_102_; 
v___f_101_ = ((lean_object*)(lp_mathlib_LinearMap_toSpanSingleton___redArg___closed__0));
v___f_102_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_smulRight___redArg___lam__0), 4, 3);
lean_closure_set(v___f_102_, 0, v___f_101_);
lean_closure_set(v___f_102_, 1, v_inst_99_);
lean_closure_set(v___f_102_, 2, v_x_100_);
return v___f_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toSpanSingleton(lean_object* v_R_103_, lean_object* v_M_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_x_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_mathlib_LinearMap_toSpanSingleton___redArg(v_inst_107_, v_x_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_toSpanSingleton___boxed(lean_object* v_R_110_, lean_object* v_M_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_x_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_LinearMap_toSpanSingleton(v_R_110_, v_M_111_, v_inst_112_, v_inst_113_, v_inst_114_, v_x_115_);
lean_dec_ref(v_inst_113_);
lean_dec_ref(v_inst_112_);
return v_res_116_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Pointwise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompactlyGenerated_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BourbakiWitt(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Field(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompactlyGenerated_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BourbakiWitt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Torsion_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompactlyGenerated_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BourbakiWitt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Torsion_Field(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompactlyGenerated_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BourbakiWitt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Torsion_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
