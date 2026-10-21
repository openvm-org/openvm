// Lean compiler output
// Module: Mathlib.Algebra.MvPolynomial.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Subalgebra.Lattice public import Mathlib.Algebra.Algebra.Tower public import Mathlib.Algebra.GroupWithZero.Divisibility public import Mathlib.Algebra.MonoidAlgebra.Basic public import Mathlib.Algebra.MonoidAlgebra.NoZeroDivisors public import Mathlib.Algebra.MonoidAlgebra.Support public import Mathlib.Algebra.Regular.Pow public import Mathlib.Data.Finsupp.Antidiagonal public import Mathlib.Data.Finsupp.Order public import Mathlib.Order.SymmDiff public import Mathlib.Tactic.Polynomial.Core
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_support___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_support___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_support(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_support___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_lcoeff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_lcoeff(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_lcoeff___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_constantCoeff___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_constantCoeff___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_constantCoeff___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MvPolynomial_constantCoeff___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MvPolynomial_constantCoeff___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MvPolynomial_constantCoeff___closed__0 = (const lean_object*)&lp_mathlib_MvPolynomial_constantCoeff___closed__0_value;
static const lean_closure_object lp_mathlib_MvPolynomial_constantCoeff___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MvPolynomial_constantCoeff___lam__1, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_MvPolynomial_constantCoeff___closed__0_value)} };
static const lean_object* lp_mathlib_MvPolynomial_constantCoeff___closed__1 = (const lean_object*)&lp_mathlib_MvPolynomial_constantCoeff___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_constantCoeff(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_constantCoeff___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffsIn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffsIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__0 = (const lean_object*)&lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__1;
static const lean_string_object lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MvPolynomial"};
static const lean_object* lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__2 = (const lean_object*)&lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 184, 86, 143, 133, 91, 98, 169)}};
static const lean_object* lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__3 = (const lean_object*)&lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___closed__0 = (const lean_object*)&lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl = (const lean_object*)&lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_support___redArg(lean_object* v_p_1_){
_start:
{
lean_object* v_support_2_; 
v_support_2_ = lean_ctor_get(v_p_1_, 0);
lean_inc(v_support_2_);
return v_support_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_support___redArg___boxed(lean_object* v_p_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_MvPolynomial_support___redArg(v_p_3_);
lean_dec_ref(v_p_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_support(lean_object* v_R_5_, lean_object* v_00_u03c3_6_, lean_object* v_inst_7_, lean_object* v_p_8_){
_start:
{
lean_object* v_support_9_; 
v_support_9_ = lean_ctor_get(v_p_8_, 0);
lean_inc(v_support_9_);
return v_support_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_support___boxed(lean_object* v_R_10_, lean_object* v_00_u03c3_11_, lean_object* v_inst_12_, lean_object* v_p_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_MvPolynomial_support(v_R_10_, v_00_u03c3_11_, v_inst_12_, v_p_13_);
lean_dec_ref(v_p_13_);
lean_dec_ref(v_inst_12_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffAddMonoidHom___redArg___lam__0(lean_object* v_m_15_, lean_object* v_x_16_){
_start:
{
lean_object* v_toFun_17_; lean_object* v___x_18_; 
v_toFun_17_ = lean_ctor_get(v_x_16_, 1);
lean_inc(v_toFun_17_);
lean_dec_ref(v_x_16_);
v___x_18_ = lean_apply_1(v_toFun_17_, v_m_15_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffAddMonoidHom___redArg(lean_object* v_m_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_coeffAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_20_, 0, v_m_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffAddMonoidHom(lean_object* v_R_21_, lean_object* v_00_u03c3_22_, lean_object* v_inst_23_, lean_object* v_m_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_coeffAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_25_, 0, v_m_24_);
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffAddMonoidHom___boxed(lean_object* v_R_26_, lean_object* v_00_u03c3_27_, lean_object* v_inst_28_, lean_object* v_m_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_MvPolynomial_coeffAddMonoidHom(v_R_26_, v_00_u03c3_27_, v_inst_28_, v_m_29_);
lean_dec_ref(v_inst_28_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_lcoeff___redArg(lean_object* v_m_31_){
_start:
{
lean_object* v___f_32_; 
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_coeffAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_32_, 0, v_m_31_);
return v___f_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_lcoeff(lean_object* v_R_33_, lean_object* v_00_u03c3_34_, lean_object* v_inst_35_, lean_object* v_m_36_){
_start:
{
lean_object* v___f_37_; 
v___f_37_ = lean_alloc_closure((void*)(lp_mathlib_MvPolynomial_coeffAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_37_, 0, v_m_36_);
return v___f_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_lcoeff___boxed(lean_object* v_R_38_, lean_object* v_00_u03c3_39_, lean_object* v_inst_40_, lean_object* v_m_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_MvPolynomial_lcoeff(v_R_38_, v_00_u03c3_39_, v_inst_40_, v_m_41_);
lean_dec_ref(v_inst_40_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_constantCoeff___lam__0(lean_object* v_x_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lean_unsigned_to_nat(0u);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_constantCoeff___lam__0___boxed(lean_object* v_x_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_MvPolynomial_constantCoeff___lam__0(v_x_45_);
lean_dec(v_x_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_constantCoeff___lam__1(lean_object* v___f_47_, lean_object* v_x_48_){
_start:
{
lean_object* v_toFun_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_58_; 
v_toFun_49_ = lean_ctor_get(v_x_48_, 1);
v_isSharedCheck_58_ = !lean_is_exclusive(v_x_48_);
if (v_isSharedCheck_58_ == 0)
{
lean_object* v_unused_59_; 
v_unused_59_ = lean_ctor_get(v_x_48_, 0);
lean_dec(v_unused_59_);
v___x_51_ = v_x_48_;
v_isShared_52_ = v_isSharedCheck_58_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_toFun_49_);
lean_dec(v_x_48_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_58_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
lean_object* v___x_53_; lean_object* v___x_55_; 
v___x_53_ = lean_box(0);
if (v_isShared_52_ == 0)
{
lean_ctor_set(v___x_51_, 1, v___f_47_);
lean_ctor_set(v___x_51_, 0, v___x_53_);
v___x_55_ = v___x_51_;
goto v_reusejp_54_;
}
else
{
lean_object* v_reuseFailAlloc_57_; 
v_reuseFailAlloc_57_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_57_, 0, v___x_53_);
lean_ctor_set(v_reuseFailAlloc_57_, 1, v___f_47_);
v___x_55_ = v_reuseFailAlloc_57_;
goto v_reusejp_54_;
}
v_reusejp_54_:
{
lean_object* v___x_56_; 
v___x_56_ = lean_apply_1(v_toFun_49_, v___x_55_);
return v___x_56_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_constantCoeff(lean_object* v_R_63_, lean_object* v_00_u03c3_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v___f_66_; 
v___f_66_ = ((lean_object*)(lp_mathlib_MvPolynomial_constantCoeff___closed__1));
return v___f_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_constantCoeff___boxed(lean_object* v_R_67_, lean_object* v_00_u03c3_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_MvPolynomial_constantCoeff(v_R_67_, v_00_u03c3_68_, v_inst_69_);
lean_dec_ref(v_inst_69_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffsIn(lean_object* v_R_71_, lean_object* v_S_72_, lean_object* v_00_u03c3_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_M_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lean_box(0);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_coeffsIn___boxed(lean_object* v_R_79_, lean_object* v_S_80_, lean_object* v_00_u03c3_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_M_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_MvPolynomial_coeffsIn(v_R_79_, v_S_80_, v_00_u03c3_81_, v_inst_82_, v_inst_83_, v_inst_84_, v_M_85_);
lean_dec(v_inst_84_);
lean_dec_ref(v_inst_83_);
lean_dec_ref(v_inst_82_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0_spec__0(lean_object* v_msgData_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v___x_93_; lean_object* v_env_94_; lean_object* v___x_95_; lean_object* v_mctx_96_; lean_object* v_lctx_97_; lean_object* v_options_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_93_ = lean_st_ref_get(v___y_91_);
v_env_94_ = lean_ctor_get(v___x_93_, 0);
lean_inc_ref(v_env_94_);
lean_dec(v___x_93_);
v___x_95_ = lean_st_ref_get(v___y_89_);
v_mctx_96_ = lean_ctor_get(v___x_95_, 0);
lean_inc_ref(v_mctx_96_);
lean_dec(v___x_95_);
v_lctx_97_ = lean_ctor_get(v___y_88_, 2);
v_options_98_ = lean_ctor_get(v___y_90_, 2);
lean_inc_ref(v_options_98_);
lean_inc_ref(v_lctx_97_);
v___x_99_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_99_, 0, v_env_94_);
lean_ctor_set(v___x_99_, 1, v_mctx_96_);
lean_ctor_set(v___x_99_, 2, v_lctx_97_);
lean_ctor_set(v___x_99_, 3, v_options_98_);
v___x_100_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
lean_ctor_set(v___x_100_, 1, v_msgData_87_);
v___x_101_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0_spec__0___boxed(lean_object* v_msgData_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0_spec__0(v_msgData_102_, v___y_103_, v___y_104_, v___y_105_, v___y_106_);
lean_dec(v___y_106_);
lean_dec_ref(v___y_105_);
lean_dec(v___y_104_);
lean_dec_ref(v___y_103_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0___redArg(lean_object* v_msg_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_){
_start:
{
lean_object* v_ref_115_; lean_object* v___x_116_; lean_object* v_a_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_125_; 
v_ref_115_ = lean_ctor_get(v___y_112_, 5);
v___x_116_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0_spec__0(v_msg_109_, v___y_110_, v___y_111_, v___y_112_, v___y_113_);
v_a_117_ = lean_ctor_get(v___x_116_, 0);
v_isSharedCheck_125_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_125_ == 0)
{
v___x_119_ = v___x_116_;
v_isShared_120_ = v_isSharedCheck_125_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_a_117_);
lean_dec(v___x_116_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_125_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
lean_object* v___x_121_; lean_object* v___x_123_; 
lean_inc(v_ref_115_);
v___x_121_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_121_, 0, v_ref_115_);
lean_ctor_set(v___x_121_, 1, v_a_117_);
if (v_isShared_120_ == 0)
{
lean_ctor_set_tag(v___x_119_, 1);
lean_ctor_set(v___x_119_, 0, v___x_121_);
v___x_123_ = v___x_119_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v___x_121_);
v___x_123_ = v_reuseFailAlloc_124_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
return v___x_123_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0___redArg___boxed(lean_object* v_msg_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0___redArg(v_msg_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_);
lean_dec(v___y_130_);
lean_dec_ref(v___y_129_);
lean_dec(v___y_128_);
lean_dec_ref(v___y_127_);
return v_res_132_;
}
}
static lean_object* _init_lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__1(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_134_ = ((lean_object*)(lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__0));
v___x_135_ = l_Lean_stringToMessageData(v___x_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0(lean_object* v_e_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_e_139_, v___y_141_);
if (lean_obj_tag(v___x_145_) == 0)
{
lean_object* v_a_146_; lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_170_; 
v_a_146_ = lean_ctor_get(v___x_145_, 0);
v_isSharedCheck_170_ = !lean_is_exclusive(v___x_145_);
if (v_isSharedCheck_170_ == 0)
{
v___x_148_ = v___x_145_;
v_isShared_149_ = v_isSharedCheck_170_;
goto v_resetjp_147_;
}
else
{
lean_inc(v_a_146_);
lean_dec(v___x_145_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_170_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
lean_object* v___y_151_; lean_object* v___y_152_; lean_object* v___y_153_; lean_object* v___y_154_; lean_object* v___x_157_; uint8_t v___x_158_; 
v___x_157_ = l_Lean_Expr_cleanupAnnotations(v_a_146_);
v___x_158_ = l_Lean_Expr_isApp(v___x_157_);
if (v___x_158_ == 0)
{
lean_dec_ref(v___x_157_);
lean_del_object(v___x_148_);
v___y_151_ = v___y_140_;
v___y_152_ = v___y_141_;
v___y_153_ = v___y_142_;
v___y_154_ = v___y_143_;
goto v___jp_150_;
}
else
{
lean_object* v___x_159_; uint8_t v___x_160_; 
v___x_159_ = l_Lean_Expr_appFnCleanup___redArg(v___x_157_);
v___x_160_ = l_Lean_Expr_isApp(v___x_159_);
if (v___x_160_ == 0)
{
lean_dec_ref(v___x_159_);
lean_del_object(v___x_148_);
v___y_151_ = v___y_140_;
v___y_152_ = v___y_141_;
v___y_153_ = v___y_142_;
v___y_154_ = v___y_143_;
goto v___jp_150_;
}
else
{
lean_object* v_arg_161_; lean_object* v___x_162_; uint8_t v___x_163_; 
v_arg_161_ = lean_ctor_get(v___x_159_, 1);
lean_inc_ref(v_arg_161_);
v___x_162_ = l_Lean_Expr_appFnCleanup___redArg(v___x_159_);
v___x_163_ = l_Lean_Expr_isApp(v___x_162_);
if (v___x_163_ == 0)
{
lean_dec_ref(v___x_162_);
lean_dec_ref(v_arg_161_);
lean_del_object(v___x_148_);
v___y_151_ = v___y_140_;
v___y_152_ = v___y_141_;
v___y_153_ = v___y_142_;
v___y_154_ = v___y_143_;
goto v___jp_150_;
}
else
{
lean_object* v___x_164_; lean_object* v___x_165_; uint8_t v___x_166_; 
v___x_164_ = l_Lean_Expr_appFnCleanup___redArg(v___x_162_);
v___x_165_ = ((lean_object*)(lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__3));
v___x_166_ = l_Lean_Expr_isConstOf(v___x_164_, v___x_165_);
lean_dec_ref(v___x_164_);
if (v___x_166_ == 0)
{
lean_dec_ref(v_arg_161_);
lean_del_object(v___x_148_);
v___y_151_ = v___y_140_;
v___y_152_ = v___y_141_;
v___y_153_ = v___y_142_;
v___y_154_ = v___y_143_;
goto v___jp_150_;
}
else
{
lean_object* v___x_168_; 
if (v_isShared_149_ == 0)
{
lean_ctor_set(v___x_148_, 0, v_arg_161_);
v___x_168_ = v___x_148_;
goto v_reusejp_167_;
}
else
{
lean_object* v_reuseFailAlloc_169_; 
v_reuseFailAlloc_169_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_169_, 0, v_arg_161_);
v___x_168_ = v_reuseFailAlloc_169_;
goto v_reusejp_167_;
}
v_reusejp_167_:
{
return v___x_168_;
}
}
}
}
}
v___jp_150_:
{
lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_155_ = lean_obj_once(&lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__1, &lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__1_once, _init_lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___closed__1);
v___x_156_ = lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0___redArg(v___x_155_, v___y_151_, v___y_152_, v___y_153_, v___y_154_);
return v___x_156_;
}
}
}
else
{
return v___x_145_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0___boxed(lean_object* v_e_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib_MvPolynomial_mvPolynomialInferBaseImpl___lam__0(v_e_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_);
lean_dec(v___y_175_);
lean_dec_ref(v___y_174_);
lean_dec(v___y_173_);
lean_dec_ref(v___y_172_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0(lean_object* v_00_u03b1_180_, lean_object* v_msg_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0___redArg(v_msg_181_, v___y_182_, v___y_183_, v___y_184_, v___y_185_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0___boxed(lean_object* v_00_u03b1_188_, lean_object* v_msg_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_Lean_throwError___at___00MvPolynomial_mvPolynomialInferBaseImpl_spec__0(v_00_u03b1_188_, v_msg_189_, v___y_190_, v___y_191_, v___y_192_, v___y_193_);
lean_dec(v___y_193_);
lean_dec_ref(v___y_192_);
lean_dec(v___y_191_);
lean_dec_ref(v___y_190_);
return v_res_195_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_NoZeroDivisors(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Regular_Pow(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Antidiagonal(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SymmDiff(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Polynomial_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_NoZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Regular_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Antidiagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Polynomial_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_NoZeroDivisors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_Pow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Antidiagonal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SymmDiff(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Polynomial_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Subalgebra_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_NoZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Support(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Antidiagonal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Polynomial_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
