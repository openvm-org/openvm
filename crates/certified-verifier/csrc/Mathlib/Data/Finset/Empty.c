// Lean compiler output
// Module: Mathlib.Data.Finset.Empty
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Defs public import Mathlib.Data.Multiset.ZeroCons public import Aesop
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_MetavarContext_getExprAssignmentCore_x3f(lean_object*, lean_object*);
uint8_t lp_mathlib_Finset_decidableExistsAndFinset___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSets(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mkLocalRuleSet(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
extern lean_object* lp_aesop_Aesop_Stats_empty;
lean_object* lp_aesop_Aesop_search(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableNonempty___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableNonempty___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Finset_decidableNonempty___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_decidableNonempty___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_decidableNonempty___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_decidableNonempty___redArg___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableNonempty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableNonempty___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableNonempty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableNonempty___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_empty(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instEmptyCollection(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_inhabitedFinset(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instOrderBot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Nonempty"};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__0_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__1_value),LEAN_SCALAR_PTR_LITERAL(194, 63, 16, 12, 129, 191, 206, 174)}};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "builtin"};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__3_value),LEAN_SCALAR_PTR_LITERAL(78, 115, 70, 234, 108, 55, 8, 53)}};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "finsetNonempty"};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__5_value),LEAN_SCALAR_PTR_LITERAL(35, 132, 109, 29, 55, 165, 216, 158)}};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__6_value;
static const lean_array_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 246}, .m_size = 2, .m_capacity = 2, .m_data = {((lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*6 + 16, .m_other = 6, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(30) << 1) | 1)),((lean_object*)(((size_t)(200) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 2, 1, 0, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 32, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100000) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 1, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 1, 1, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 1, 1, 1, 1, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableNonempty___redArg___lam__0(lean_object* v_a_1_){
_start:
{
uint8_t v___x_2_; 
v___x_2_ = 1;
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableNonempty___redArg___lam__0___boxed(lean_object* v_a_3_){
_start:
{
uint8_t v_res_4_; lean_object* v_r_5_; 
v_res_4_ = lp_mathlib_Finset_decidableNonempty___redArg___lam__0(v_a_3_);
lean_dec(v_a_3_);
v_r_5_ = lean_box(v_res_4_);
return v_r_5_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableNonempty___redArg(lean_object* v_s_7_){
_start:
{
lean_object* v___f_8_; uint8_t v___x_9_; 
v___f_8_ = ((lean_object*)(lp_mathlib_Finset_decidableNonempty___redArg___closed__0));
v___x_9_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_s_7_, v___f_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableNonempty___redArg___boxed(lean_object* v_s_10_){
_start:
{
uint8_t v_res_11_; lean_object* v_r_12_; 
v_res_11_ = lp_mathlib_Finset_decidableNonempty___redArg(v_s_10_);
v_r_12_ = lean_box(v_res_11_);
return v_r_12_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableNonempty(lean_object* v_00_u03b1_13_, lean_object* v_s_14_){
_start:
{
uint8_t v___x_15_; 
v___x_15_ = lp_mathlib_Finset_decidableNonempty___redArg(v_s_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableNonempty___boxed(lean_object* v_00_u03b1_16_, lean_object* v_s_17_){
_start:
{
uint8_t v_res_18_; lean_object* v_r_19_; 
v_res_18_ = lp_mathlib_Finset_decidableNonempty(v_00_u03b1_16_, v_s_17_);
v_r_19_ = lean_box(v_res_18_);
return v_r_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_empty(lean_object* v_00_u03b1_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_box(0);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instEmptyCollection(lean_object* v_00_u03b1_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_box(0);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_inhabitedFinset(lean_object* v_00_u03b1_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_box(0);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instOrderBot(lean_object* v_00_u03b1_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lean_box(0);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0___redArg(lean_object* v_mvarId_28_, lean_object* v___y_29_){
_start:
{
lean_object* v___x_31_; lean_object* v_mctx_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_31_ = lean_st_ref_get(v___y_29_);
v_mctx_32_ = lean_ctor_get(v___x_31_, 0);
lean_inc_ref(v_mctx_32_);
lean_dec(v___x_31_);
v___x_33_ = l_Lean_MetavarContext_getExprAssignmentCore_x3f(v_mctx_32_, v_mvarId_28_);
lean_dec_ref(v_mctx_32_);
v___x_34_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0___redArg___boxed(lean_object* v_mvarId_35_, lean_object* v___y_36_, lean_object* v___y_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0___redArg(v_mvarId_35_, v___y_36_);
lean_dec(v___y_36_);
lean_dec(v_mvarId_35_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0(lean_object* v_mvarId_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0___redArg(v_mvarId_39_, v___y_41_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0___boxed(lean_object* v_mvarId_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0(v_mvarId_46_, v___y_47_, v___y_48_, v___y_49_, v___y_50_);
lean_dec(v___y_50_);
lean_dec_ref(v___y_49_);
lean_dec(v___y_48_);
lean_dec_ref(v___y_47_);
lean_dec(v_mvarId_46_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty(lean_object* v_u_93_, lean_object* v_00_u03b1_94_, lean_object* v_s_95_, lean_object* v_a_96_, lean_object* v_a_97_, lean_object* v_a_98_, lean_object* v_a_99_){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; uint8_t v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_101_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__2));
v___x_102_ = lean_box(0);
v___x_103_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_103_, 0, v_u_93_);
lean_ctor_set(v___x_103_, 1, v___x_102_);
v___x_104_ = l_Lean_Expr_const___override(v___x_101_, v___x_103_);
v___x_105_ = l_Lean_Expr_app___override(v___x_104_, v_00_u03b1_94_);
v___x_106_ = l_Lean_Expr_app___override(v___x_105_, v_s_95_);
v___x_107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
v___x_108_ = 0;
v___x_109_ = lean_box(0);
v___x_110_ = l_Lean_Meta_mkFreshExprMVar(v___x_107_, v___x_108_, v___x_109_, v_a_96_, v_a_97_, v_a_98_, v_a_99_);
if (lean_obj_tag(v___x_110_) == 0)
{
lean_object* v_a_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v_a_111_ = lean_ctor_get(v___x_110_, 0);
lean_inc(v_a_111_);
lean_dec_ref_known(v___x_110_, 1);
v___x_112_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__7));
v___x_113_ = lp_aesop_Aesop_Frontend_getGlobalRuleSets(v___x_112_, v_a_98_, v_a_99_);
if (lean_obj_tag(v___x_113_) == 0)
{
lean_object* v_a_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v_a_114_ = lean_ctor_get(v___x_113_, 0);
lean_inc(v_a_114_);
lean_dec_ref_known(v___x_113_, 1);
v___x_115_ = lean_unsigned_to_nat(0u);
v___x_116_ = lean_box(0);
v___x_117_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__8));
v___x_118_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__9));
v___x_119_ = lp_aesop_Aesop_mkLocalRuleSet(v_a_114_, v___x_118_, v_a_98_, v_a_99_);
lean_dec(v_a_114_);
if (lean_obj_tag(v___x_119_) == 0)
{
lean_object* v_a_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v_a_120_ = lean_ctor_get(v___x_119_, 0);
lean_inc(v_a_120_);
lean_dec_ref_known(v___x_119_, 1);
v___x_121_ = l_Lean_Expr_mvarId_x21(v_a_111_);
lean_dec(v_a_111_);
v___x_122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_122_, 0, v_a_120_);
v___x_123_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_proveFinsetNonempty___closed__10));
v___x_124_ = lp_aesop_Aesop_Stats_empty;
lean_inc(v___x_121_);
v___x_125_ = lp_aesop_Aesop_search(v___x_121_, v___x_122_, v___x_117_, v___x_123_, v___x_116_, v___x_124_, v_a_96_, v_a_97_, v_a_98_, v_a_99_);
if (lean_obj_tag(v___x_125_) == 0)
{
lean_object* v_a_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_137_; 
v_a_126_ = lean_ctor_get(v___x_125_, 0);
v_isSharedCheck_137_ = !lean_is_exclusive(v___x_125_);
if (v_isSharedCheck_137_ == 0)
{
v___x_128_ = v___x_125_;
v_isShared_129_ = v_isSharedCheck_137_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_a_126_);
lean_dec(v___x_125_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_137_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v_fst_130_; lean_object* v___x_131_; uint8_t v___x_132_; 
v_fst_130_ = lean_ctor_get(v_a_126_, 0);
lean_inc(v_fst_130_);
lean_dec(v_a_126_);
v___x_131_ = lean_array_get_size(v_fst_130_);
lean_dec(v_fst_130_);
v___x_132_ = lean_nat_dec_lt(v___x_115_, v___x_131_);
if (v___x_132_ == 0)
{
lean_object* v___x_133_; 
lean_del_object(v___x_128_);
v___x_133_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Mathlib_Meta_proveFinsetNonempty_spec__0___redArg(v___x_121_, v_a_97_);
lean_dec(v___x_121_);
return v___x_133_;
}
else
{
lean_object* v___x_135_; 
lean_dec(v___x_121_);
if (v_isShared_129_ == 0)
{
lean_ctor_set(v___x_128_, 0, v___x_116_);
v___x_135_ = v___x_128_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v___x_116_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
}
}
else
{
lean_object* v_a_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_152_; 
lean_dec(v___x_121_);
v_a_138_ = lean_ctor_get(v___x_125_, 0);
v_isSharedCheck_152_ = !lean_is_exclusive(v___x_125_);
if (v_isSharedCheck_152_ == 0)
{
v___x_140_ = v___x_125_;
v_isShared_141_ = v_isSharedCheck_152_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_a_138_);
lean_dec(v___x_125_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_152_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
uint8_t v___y_143_; uint8_t v___x_150_; 
v___x_150_ = l_Lean_Exception_isInterrupt(v_a_138_);
if (v___x_150_ == 0)
{
uint8_t v___x_151_; 
lean_inc(v_a_138_);
v___x_151_ = l_Lean_Exception_isRuntime(v_a_138_);
v___y_143_ = v___x_151_;
goto v___jp_142_;
}
else
{
v___y_143_ = v___x_150_;
goto v___jp_142_;
}
v___jp_142_:
{
if (v___y_143_ == 0)
{
lean_object* v___x_145_; 
lean_dec(v_a_138_);
if (v_isShared_141_ == 0)
{
lean_ctor_set_tag(v___x_140_, 0);
lean_ctor_set(v___x_140_, 0, v___x_116_);
v___x_145_ = v___x_140_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v___x_116_);
v___x_145_ = v_reuseFailAlloc_146_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
return v___x_145_;
}
}
else
{
lean_object* v___x_148_; 
if (v_isShared_141_ == 0)
{
v___x_148_ = v___x_140_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_149_; 
v_reuseFailAlloc_149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_149_, 0, v_a_138_);
v___x_148_ = v_reuseFailAlloc_149_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
return v___x_148_;
}
}
}
}
}
}
else
{
lean_object* v_a_153_; lean_object* v___x_155_; uint8_t v_isShared_156_; uint8_t v_isSharedCheck_160_; 
lean_dec(v_a_111_);
v_a_153_ = lean_ctor_get(v___x_119_, 0);
v_isSharedCheck_160_ = !lean_is_exclusive(v___x_119_);
if (v_isSharedCheck_160_ == 0)
{
v___x_155_ = v___x_119_;
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
else
{
lean_inc(v_a_153_);
lean_dec(v___x_119_);
v___x_155_ = lean_box(0);
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
v_resetjp_154_:
{
lean_object* v___x_158_; 
if (v_isShared_156_ == 0)
{
v___x_158_ = v___x_155_;
goto v_reusejp_157_;
}
else
{
lean_object* v_reuseFailAlloc_159_; 
v_reuseFailAlloc_159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_159_, 0, v_a_153_);
v___x_158_ = v_reuseFailAlloc_159_;
goto v_reusejp_157_;
}
v_reusejp_157_:
{
return v___x_158_;
}
}
}
}
else
{
lean_object* v_a_161_; lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_168_; 
lean_dec(v_a_111_);
v_a_161_ = lean_ctor_get(v___x_113_, 0);
v_isSharedCheck_168_ = !lean_is_exclusive(v___x_113_);
if (v_isSharedCheck_168_ == 0)
{
v___x_163_ = v___x_113_;
v_isShared_164_ = v_isSharedCheck_168_;
goto v_resetjp_162_;
}
else
{
lean_inc(v_a_161_);
lean_dec(v___x_113_);
v___x_163_ = lean_box(0);
v_isShared_164_ = v_isSharedCheck_168_;
goto v_resetjp_162_;
}
v_resetjp_162_:
{
lean_object* v___x_166_; 
if (v_isShared_164_ == 0)
{
v___x_166_ = v___x_163_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v_a_161_);
v___x_166_ = v_reuseFailAlloc_167_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
return v___x_166_;
}
}
}
}
else
{
lean_object* v_a_169_; lean_object* v___x_171_; uint8_t v_isShared_172_; uint8_t v_isSharedCheck_176_; 
v_a_169_ = lean_ctor_get(v___x_110_, 0);
v_isSharedCheck_176_ = !lean_is_exclusive(v___x_110_);
if (v_isSharedCheck_176_ == 0)
{
v___x_171_ = v___x_110_;
v_isShared_172_ = v_isSharedCheck_176_;
goto v_resetjp_170_;
}
else
{
lean_inc(v_a_169_);
lean_dec(v___x_110_);
v___x_171_ = lean_box(0);
v_isShared_172_ = v_isSharedCheck_176_;
goto v_resetjp_170_;
}
v_resetjp_170_:
{
lean_object* v___x_174_; 
if (v_isShared_172_ == 0)
{
v___x_174_ = v___x_171_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v_a_169_);
v___x_174_ = v_reuseFailAlloc_175_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
return v___x_174_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_proveFinsetNonempty___boxed(lean_object* v_u_177_, lean_object* v_00_u03b1_178_, lean_object* v_s_179_, lean_object* v_a_180_, lean_object* v_a_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_Mathlib_Meta_proveFinsetNonempty(v_u_177_, v_00_u03b1_178_, v_s_179_, v_a_180_, v_a_181_, v_a_182_, v_a_183_);
lean_dec(v_a_183_);
lean_dec_ref(v_a_182_);
lean_dec(v_a_181_);
lean_dec_ref(v_a_180_);
return v_res_185_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Empty(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Empty(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(uint8_t builtin);
lean_object* initialize_aesop_Aesop(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Empty(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Empty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Empty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Empty(builtin);
}
#ifdef __cplusplus
}
#endif
