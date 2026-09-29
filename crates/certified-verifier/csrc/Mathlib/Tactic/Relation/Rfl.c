// Lean compiler output
// Module: Mathlib.Tactic.Relation.Rfl
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Meta.Tactic.Rfl
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
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Meta_DiscrTree_instInhabited(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_applyRfl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_MVarId_getType_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOf(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_Rfl_reflExt;
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_getMatch___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Expr_constName_x3f(lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "rel_of_eq_and_refl"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(52, 246, 21, 120, 84, 83, 244, 192)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "liftReflToEq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__0_value),LEAN_SCALAR_PTR_LITERAL(56, 30, 242, 207, 44, 185, 234, 108)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_liftReflToEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_liftReflToEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_rflTac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_rflTac___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_rflTac___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rflTac___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_rflTac___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_rflTac___lam__1___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rflTac___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_rflTac___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rflTac___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "HEq"};
static const lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(67, 180, 169, 191, 74, 196, 152, 188)}};
static const lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__3 = (const lean_object*)&lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1___redArg(lean_object* v_x_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_Lean_Meta_saveState___redArg(v___y_3_, v___y_5_);
if (lean_obj_tag(v___x_7_) == 0)
{
lean_object* v_a_8_; lean_object* v___x_9_; 
v_a_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc(v_a_8_);
lean_dec_ref_known(v___x_7_, 1);
lean_inc(v___y_5_);
lean_inc_ref(v___y_4_);
lean_inc(v___y_3_);
lean_inc_ref(v___y_2_);
v___x_9_ = lean_apply_5(v_x_1_, v___y_2_, v___y_3_, v___y_4_, v___y_5_, lean_box(0));
if (lean_obj_tag(v___x_9_) == 0)
{
lean_object* v_a_10_; lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_18_; 
lean_dec(v_a_8_);
v_a_10_ = lean_ctor_get(v___x_9_, 0);
v_isSharedCheck_18_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_18_ == 0)
{
v___x_12_ = v___x_9_;
v_isShared_13_ = v_isSharedCheck_18_;
goto v_resetjp_11_;
}
else
{
lean_inc(v_a_10_);
lean_dec(v___x_9_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_18_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v___x_14_; lean_object* v___x_16_; 
v___x_14_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_14_, 0, v_a_10_);
if (v_isShared_13_ == 0)
{
lean_ctor_set(v___x_12_, 0, v___x_14_);
v___x_16_ = v___x_12_;
goto v_reusejp_15_;
}
else
{
lean_object* v_reuseFailAlloc_17_; 
v_reuseFailAlloc_17_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_17_, 0, v___x_14_);
v___x_16_ = v_reuseFailAlloc_17_;
goto v_reusejp_15_;
}
v_reusejp_15_:
{
return v___x_16_;
}
}
}
else
{
lean_object* v_a_19_; lean_object* v___x_21_; uint8_t v_isShared_22_; uint8_t v_isSharedCheck_48_; 
v_a_19_ = lean_ctor_get(v___x_9_, 0);
v_isSharedCheck_48_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_48_ == 0)
{
v___x_21_ = v___x_9_;
v_isShared_22_ = v_isSharedCheck_48_;
goto v_resetjp_20_;
}
else
{
lean_inc(v_a_19_);
lean_dec(v___x_9_);
v___x_21_ = lean_box(0);
v_isShared_22_ = v_isSharedCheck_48_;
goto v_resetjp_20_;
}
v_resetjp_20_:
{
uint8_t v___y_24_; uint8_t v___x_46_; 
v___x_46_ = l_Lean_Exception_isInterrupt(v_a_19_);
if (v___x_46_ == 0)
{
uint8_t v___x_47_; 
lean_inc(v_a_19_);
v___x_47_ = l_Lean_Exception_isRuntime(v_a_19_);
v___y_24_ = v___x_47_;
goto v___jp_23_;
}
else
{
v___y_24_ = v___x_46_;
goto v___jp_23_;
}
v___jp_23_:
{
if (v___y_24_ == 0)
{
lean_object* v___x_25_; 
lean_del_object(v___x_21_);
lean_dec(v_a_19_);
v___x_25_ = l_Lean_Meta_SavedState_restore___redArg(v_a_8_, v___y_3_, v___y_5_);
lean_dec(v_a_8_);
if (lean_obj_tag(v___x_25_) == 0)
{
lean_object* v___x_27_; uint8_t v_isShared_28_; uint8_t v_isSharedCheck_33_; 
v_isSharedCheck_33_ = !lean_is_exclusive(v___x_25_);
if (v_isSharedCheck_33_ == 0)
{
lean_object* v_unused_34_; 
v_unused_34_ = lean_ctor_get(v___x_25_, 0);
lean_dec(v_unused_34_);
v___x_27_ = v___x_25_;
v_isShared_28_ = v_isSharedCheck_33_;
goto v_resetjp_26_;
}
else
{
lean_dec(v___x_25_);
v___x_27_ = lean_box(0);
v_isShared_28_ = v_isSharedCheck_33_;
goto v_resetjp_26_;
}
v_resetjp_26_:
{
lean_object* v___x_29_; lean_object* v___x_31_; 
v___x_29_ = lean_box(0);
if (v_isShared_28_ == 0)
{
lean_ctor_set(v___x_27_, 0, v___x_29_);
v___x_31_ = v___x_27_;
goto v_reusejp_30_;
}
else
{
lean_object* v_reuseFailAlloc_32_; 
v_reuseFailAlloc_32_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_32_, 0, v___x_29_);
v___x_31_ = v_reuseFailAlloc_32_;
goto v_reusejp_30_;
}
v_reusejp_30_:
{
return v___x_31_;
}
}
}
else
{
lean_object* v_a_35_; lean_object* v___x_37_; uint8_t v_isShared_38_; uint8_t v_isSharedCheck_42_; 
v_a_35_ = lean_ctor_get(v___x_25_, 0);
v_isSharedCheck_42_ = !lean_is_exclusive(v___x_25_);
if (v_isSharedCheck_42_ == 0)
{
v___x_37_ = v___x_25_;
v_isShared_38_ = v_isSharedCheck_42_;
goto v_resetjp_36_;
}
else
{
lean_inc(v_a_35_);
lean_dec(v___x_25_);
v___x_37_ = lean_box(0);
v_isShared_38_ = v_isSharedCheck_42_;
goto v_resetjp_36_;
}
v_resetjp_36_:
{
lean_object* v___x_40_; 
if (v_isShared_38_ == 0)
{
v___x_40_ = v___x_37_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v_a_35_);
v___x_40_ = v_reuseFailAlloc_41_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
return v___x_40_;
}
}
}
}
else
{
lean_object* v___x_44_; 
lean_dec(v_a_8_);
if (v_isShared_22_ == 0)
{
v___x_44_ = v___x_21_;
goto v_reusejp_43_;
}
else
{
lean_object* v_reuseFailAlloc_45_; 
v_reuseFailAlloc_45_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_45_, 0, v_a_19_);
v___x_44_ = v_reuseFailAlloc_45_;
goto v_reusejp_43_;
}
v_reusejp_43_:
{
return v___x_44_;
}
}
}
}
}
}
else
{
lean_object* v_a_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_56_; 
lean_dec_ref(v_x_1_);
v_a_49_ = lean_ctor_get(v___x_7_, 0);
v_isSharedCheck_56_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_56_ == 0)
{
v___x_51_ = v___x_7_;
v_isShared_52_ = v_isSharedCheck_56_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_a_49_);
lean_dec(v___x_7_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_56_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
lean_object* v___x_54_; 
if (v_isShared_52_ == 0)
{
v___x_54_ = v___x_51_;
goto v_reusejp_53_;
}
else
{
lean_object* v_reuseFailAlloc_55_; 
v_reuseFailAlloc_55_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_55_, 0, v_a_49_);
v___x_54_ = v_reuseFailAlloc_55_;
goto v_reusejp_53_;
}
v_reusejp_53_:
{
return v___x_54_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1___redArg___boxed(lean_object* v_x_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1___redArg(v_x_57_, v___y_58_, v___y_59_, v___y_60_, v___y_61_);
lean_dec(v___y_61_);
lean_dec_ref(v___y_60_);
lean_dec(v___y_59_);
lean_dec_ref(v___y_58_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1(lean_object* v_00_u03b1_64_, lean_object* v_x_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1___redArg(v_x_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1___boxed(lean_object* v_00_u03b1_72_, lean_object* v_x_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1(v_00_u03b1_72_, v_x_73_, v___y_74_, v___y_75_, v___y_76_, v___y_77_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0_spec__0(lean_object* v_msgData_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_){
_start:
{
lean_object* v___x_86_; lean_object* v_env_87_; lean_object* v___x_88_; lean_object* v_mctx_89_; lean_object* v_lctx_90_; lean_object* v_options_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_86_ = lean_st_ref_get(v___y_84_);
v_env_87_ = lean_ctor_get(v___x_86_, 0);
lean_inc_ref(v_env_87_);
lean_dec(v___x_86_);
v___x_88_ = lean_st_ref_get(v___y_82_);
v_mctx_89_ = lean_ctor_get(v___x_88_, 0);
lean_inc_ref(v_mctx_89_);
lean_dec(v___x_88_);
v_lctx_90_ = lean_ctor_get(v___y_81_, 2);
v_options_91_ = lean_ctor_get(v___y_83_, 2);
lean_inc_ref(v_options_91_);
lean_inc_ref(v_lctx_90_);
v___x_92_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_92_, 0, v_env_87_);
lean_ctor_set(v___x_92_, 1, v_mctx_89_);
lean_ctor_set(v___x_92_, 2, v_lctx_90_);
lean_ctor_set(v___x_92_, 3, v_options_91_);
v___x_93_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v_msgData_80_);
v___x_94_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0_spec__0___boxed(lean_object* v_msgData_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0_spec__0(v_msgData_95_, v___y_96_, v___y_97_, v___y_98_, v___y_99_);
lean_dec(v___y_99_);
lean_dec_ref(v___y_98_);
lean_dec(v___y_97_);
lean_dec_ref(v___y_96_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0___redArg(lean_object* v_msg_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_){
_start:
{
lean_object* v_ref_108_; lean_object* v___x_109_; lean_object* v_a_110_; lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_118_; 
v_ref_108_ = lean_ctor_get(v___y_105_, 5);
v___x_109_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0_spec__0(v_msg_102_, v___y_103_, v___y_104_, v___y_105_, v___y_106_);
v_a_110_ = lean_ctor_get(v___x_109_, 0);
v_isSharedCheck_118_ = !lean_is_exclusive(v___x_109_);
if (v_isSharedCheck_118_ == 0)
{
v___x_112_ = v___x_109_;
v_isShared_113_ = v_isSharedCheck_118_;
goto v_resetjp_111_;
}
else
{
lean_inc(v_a_110_);
lean_dec(v___x_109_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_118_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
lean_object* v___x_114_; lean_object* v___x_116_; 
lean_inc(v_ref_108_);
v___x_114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_114_, 0, v_ref_108_);
lean_ctor_set(v___x_114_, 1, v_a_110_);
if (v_isShared_113_ == 0)
{
lean_ctor_set_tag(v___x_112_, 1);
lean_ctor_set(v___x_112_, 0, v___x_114_);
v___x_116_ = v___x_112_;
goto v_reusejp_115_;
}
else
{
lean_object* v_reuseFailAlloc_117_; 
v_reuseFailAlloc_117_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_117_, 0, v___x_114_);
v___x_116_ = v_reuseFailAlloc_117_;
goto v_reusejp_115_;
}
v_reusejp_115_:
{
return v___x_116_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0___redArg___boxed(lean_object* v_msg_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0___redArg(v_msg_119_, v___y_120_, v___y_121_, v___y_122_, v___y_123_);
lean_dec(v___y_123_);
lean_dec_ref(v___y_122_);
lean_dec(v___y_121_);
lean_dec_ref(v___y_120_);
return v_res_125_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__1(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_127_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__0));
v___x_128_ = l_Lean_stringToMessageData(v___x_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0(lean_object* v___x_129_, uint8_t v___x_130_, uint8_t v___x_131_, lean_object* v_mvarId_132_, lean_object* v_a_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v___x_129_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
if (lean_obj_tag(v___x_139_) == 0)
{
lean_object* v_a_140_; uint8_t v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v_a_140_ = lean_ctor_get(v___x_139_, 0);
lean_inc(v_a_140_);
lean_dec_ref_known(v___x_139_, 1);
v___x_141_ = 0;
v___x_142_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_142_, 0, v___x_141_);
lean_ctor_set_uint8(v___x_142_, 1, v___x_130_);
lean_ctor_set_uint8(v___x_142_, 2, v___x_131_);
lean_ctor_set_uint8(v___x_142_, 3, v___x_130_);
v___x_143_ = lean_box(0);
lean_inc_ref(v___x_142_);
v___x_144_ = l_Lean_MVarId_apply(v_mvarId_132_, v_a_140_, v___x_142_, v___x_143_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
if (lean_obj_tag(v___x_144_) == 0)
{
lean_object* v_a_145_; lean_object* v___y_147_; lean_object* v___y_148_; lean_object* v___y_149_; lean_object* v___y_150_; 
v_a_145_ = lean_ctor_get(v___x_144_, 0);
lean_inc(v_a_145_);
lean_dec_ref_known(v___x_144_, 1);
if (lean_obj_tag(v_a_145_) == 1)
{
lean_object* v_tail_153_; 
v_tail_153_ = lean_ctor_get(v_a_145_, 1);
lean_inc(v_tail_153_);
if (lean_obj_tag(v_tail_153_) == 1)
{
lean_object* v_tail_154_; 
v_tail_154_ = lean_ctor_get(v_tail_153_, 1);
if (lean_obj_tag(v_tail_154_) == 0)
{
lean_object* v_head_155_; lean_object* v_head_156_; lean_object* v___x_157_; 
v_head_155_ = lean_ctor_get(v_a_145_, 0);
lean_inc(v_head_155_);
lean_dec_ref_known(v_a_145_, 2);
v_head_156_ = lean_ctor_get(v_tail_153_, 0);
lean_inc(v_head_156_);
lean_dec_ref_known(v_tail_153_, 2);
v___x_157_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_a_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
if (lean_obj_tag(v___x_157_) == 0)
{
lean_object* v_a_158_; lean_object* v___x_159_; 
v_a_158_ = lean_ctor_get(v___x_157_, 0);
lean_inc(v_a_158_);
lean_dec_ref_known(v___x_157_, 1);
v___x_159_ = l_Lean_MVarId_apply(v_head_156_, v_a_158_, v___x_142_, v___x_143_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
if (lean_obj_tag(v___x_159_) == 0)
{
lean_object* v_a_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_169_; 
v_a_160_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_169_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_169_ == 0)
{
v___x_162_ = v___x_159_;
v_isShared_163_ = v_isSharedCheck_169_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_a_160_);
lean_dec(v___x_159_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_169_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
if (lean_obj_tag(v_a_160_) == 0)
{
lean_object* v___x_165_; 
if (v_isShared_163_ == 0)
{
lean_ctor_set(v___x_162_, 0, v_head_155_);
v___x_165_ = v___x_162_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v_head_155_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
else
{
lean_object* v___x_167_; lean_object* v___x_168_; 
lean_del_object(v___x_162_);
lean_dec(v_a_160_);
lean_dec(v_head_155_);
v___x_167_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__1);
v___x_168_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0___redArg(v___x_167_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
return v___x_168_;
}
}
}
else
{
lean_object* v_a_170_; lean_object* v___x_172_; uint8_t v_isShared_173_; uint8_t v_isSharedCheck_177_; 
lean_dec(v_head_155_);
v_a_170_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_177_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_177_ == 0)
{
v___x_172_ = v___x_159_;
v_isShared_173_ = v_isSharedCheck_177_;
goto v_resetjp_171_;
}
else
{
lean_inc(v_a_170_);
lean_dec(v___x_159_);
v___x_172_ = lean_box(0);
v_isShared_173_ = v_isSharedCheck_177_;
goto v_resetjp_171_;
}
v_resetjp_171_:
{
lean_object* v___x_175_; 
if (v_isShared_173_ == 0)
{
v___x_175_ = v___x_172_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v_a_170_);
v___x_175_ = v_reuseFailAlloc_176_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
return v___x_175_;
}
}
}
}
else
{
lean_object* v_a_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_185_; 
lean_dec(v_head_156_);
lean_dec(v_head_155_);
lean_dec_ref_known(v___x_142_, 0);
v_a_178_ = lean_ctor_get(v___x_157_, 0);
v_isSharedCheck_185_ = !lean_is_exclusive(v___x_157_);
if (v_isSharedCheck_185_ == 0)
{
v___x_180_ = v___x_157_;
v_isShared_181_ = v_isSharedCheck_185_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_a_178_);
lean_dec(v___x_157_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_185_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v___x_183_; 
if (v_isShared_181_ == 0)
{
v___x_183_ = v___x_180_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_184_; 
v_reuseFailAlloc_184_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_184_, 0, v_a_178_);
v___x_183_ = v_reuseFailAlloc_184_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
return v___x_183_;
}
}
}
}
else
{
lean_dec_ref_known(v_tail_153_, 2);
lean_dec_ref_known(v_a_145_, 2);
lean_dec_ref_known(v___x_142_, 0);
lean_dec(v_a_133_);
v___y_147_ = v___y_134_;
v___y_148_ = v___y_135_;
v___y_149_ = v___y_136_;
v___y_150_ = v___y_137_;
goto v___jp_146_;
}
}
else
{
lean_dec_ref_known(v_a_145_, 2);
lean_dec(v_tail_153_);
lean_dec_ref_known(v___x_142_, 0);
lean_dec(v_a_133_);
v___y_147_ = v___y_134_;
v___y_148_ = v___y_135_;
v___y_149_ = v___y_136_;
v___y_150_ = v___y_137_;
goto v___jp_146_;
}
}
else
{
lean_dec(v_a_145_);
lean_dec_ref_known(v___x_142_, 0);
lean_dec(v_a_133_);
v___y_147_ = v___y_134_;
v___y_148_ = v___y_135_;
v___y_149_ = v___y_136_;
v___y_150_ = v___y_137_;
goto v___jp_146_;
}
v___jp_146_:
{
lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_151_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___closed__1);
v___x_152_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0___redArg(v___x_151_, v___y_147_, v___y_148_, v___y_149_, v___y_150_);
return v___x_152_;
}
}
else
{
lean_object* v_a_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_193_; 
lean_dec_ref_known(v___x_142_, 0);
lean_dec(v_a_133_);
v_a_186_ = lean_ctor_get(v___x_144_, 0);
v_isSharedCheck_193_ = !lean_is_exclusive(v___x_144_);
if (v_isSharedCheck_193_ == 0)
{
v___x_188_ = v___x_144_;
v_isShared_189_ = v_isSharedCheck_193_;
goto v_resetjp_187_;
}
else
{
lean_inc(v_a_186_);
lean_dec(v___x_144_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_193_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
lean_object* v___x_191_; 
if (v_isShared_189_ == 0)
{
v___x_191_ = v___x_188_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_192_; 
v_reuseFailAlloc_192_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_192_, 0, v_a_186_);
v___x_191_ = v_reuseFailAlloc_192_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
return v___x_191_;
}
}
}
}
else
{
lean_object* v_a_194_; lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_201_; 
lean_dec(v_a_133_);
lean_dec(v_mvarId_132_);
v_a_194_ = lean_ctor_get(v___x_139_, 0);
v_isSharedCheck_201_ = !lean_is_exclusive(v___x_139_);
if (v_isSharedCheck_201_ == 0)
{
v___x_196_ = v___x_139_;
v_isShared_197_ = v_isSharedCheck_201_;
goto v_resetjp_195_;
}
else
{
lean_inc(v_a_194_);
lean_dec(v___x_139_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_201_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v___x_199_; 
if (v_isShared_197_ == 0)
{
v___x_199_ = v___x_196_;
goto v_reusejp_198_;
}
else
{
lean_object* v_reuseFailAlloc_200_; 
v_reuseFailAlloc_200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_200_, 0, v_a_194_);
v___x_199_ = v_reuseFailAlloc_200_;
goto v_reusejp_198_;
}
v_reusejp_198_:
{
return v___x_199_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___boxed(lean_object* v___x_202_, lean_object* v___x_203_, lean_object* v___x_204_, lean_object* v_mvarId_205_, lean_object* v_a_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_){
_start:
{
uint8_t v___x_7072__boxed_212_; uint8_t v___x_7073__boxed_213_; lean_object* v_res_214_; 
v___x_7072__boxed_212_ = lean_unbox(v___x_203_);
v___x_7073__boxed_213_ = lean_unbox(v___x_204_);
v_res_214_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0(v___x_202_, v___x_7072__boxed_212_, v___x_7073__boxed_213_, v_mvarId_205_, v_a_206_, v___y_207_, v___y_208_, v___y_209_, v___y_210_);
lean_dec(v___y_210_);
lean_dec_ref(v___y_209_);
lean_dec(v___y_208_);
lean_dec_ref(v___y_207_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2(uint8_t v___x_225_, lean_object* v_mvarId_226_, lean_object* v_as_227_, size_t v_sz_228_, size_t v_i_229_, lean_object* v_b_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_){
_start:
{
uint8_t v___x_236_; 
v___x_236_ = lean_usize_dec_lt(v_i_229_, v_sz_228_);
if (v___x_236_ == 0)
{
lean_object* v___x_237_; 
lean_dec(v_mvarId_226_);
v___x_237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_237_, 0, v_b_230_);
return v___x_237_;
}
else
{
lean_object* v_a_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___f_242_; lean_object* v___x_243_; 
lean_dec_ref(v_b_230_);
v_a_238_ = lean_array_uget_borrowed(v_as_227_, v_i_229_);
v___x_239_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__3));
v___x_240_ = lean_box(v___x_236_);
v___x_241_ = lean_box(v___x_225_);
lean_inc(v_a_238_);
lean_inc(v_mvarId_226_);
v___f_242_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___lam__0___boxed), 10, 5);
lean_closure_set(v___f_242_, 0, v___x_239_);
lean_closure_set(v___f_242_, 1, v___x_240_);
lean_closure_set(v___f_242_, 2, v___x_241_);
lean_closure_set(v___f_242_, 3, v_mvarId_226_);
lean_closure_set(v___f_242_, 4, v_a_238_);
v___x_243_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_liftReflToEq_spec__1___redArg(v___f_242_, v___y_231_, v___y_232_, v___y_233_, v___y_234_);
if (lean_obj_tag(v___x_243_) == 0)
{
lean_object* v_a_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_257_; 
v_a_244_ = lean_ctor_get(v___x_243_, 0);
v_isSharedCheck_257_ = !lean_is_exclusive(v___x_243_);
if (v_isSharedCheck_257_ == 0)
{
v___x_246_ = v___x_243_;
v_isShared_247_ = v_isSharedCheck_257_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_a_244_);
lean_dec(v___x_243_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_257_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v___x_248_; 
v___x_248_ = lean_box(0);
if (lean_obj_tag(v_a_244_) == 1)
{
lean_object* v___x_249_; lean_object* v___x_251_; 
lean_dec(v_mvarId_226_);
v___x_249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_249_, 0, v_a_244_);
lean_ctor_set(v___x_249_, 1, v___x_248_);
if (v_isShared_247_ == 0)
{
lean_ctor_set(v___x_246_, 0, v___x_249_);
v___x_251_ = v___x_246_;
goto v_reusejp_250_;
}
else
{
lean_object* v_reuseFailAlloc_252_; 
v_reuseFailAlloc_252_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_252_, 0, v___x_249_);
v___x_251_ = v_reuseFailAlloc_252_;
goto v_reusejp_250_;
}
v_reusejp_250_:
{
return v___x_251_;
}
}
else
{
lean_object* v___x_253_; size_t v___x_254_; size_t v___x_255_; 
lean_del_object(v___x_246_);
lean_dec(v_a_244_);
v___x_253_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__4));
v___x_254_ = ((size_t)1ULL);
v___x_255_ = lean_usize_add(v_i_229_, v___x_254_);
v_i_229_ = v___x_255_;
v_b_230_ = v___x_253_;
goto _start;
}
}
}
else
{
lean_object* v_a_258_; lean_object* v___x_260_; uint8_t v_isShared_261_; uint8_t v_isSharedCheck_265_; 
lean_dec(v_mvarId_226_);
v_a_258_ = lean_ctor_get(v___x_243_, 0);
v_isSharedCheck_265_ = !lean_is_exclusive(v___x_243_);
if (v_isSharedCheck_265_ == 0)
{
v___x_260_ = v___x_243_;
v_isShared_261_ = v_isSharedCheck_265_;
goto v_resetjp_259_;
}
else
{
lean_inc(v_a_258_);
lean_dec(v___x_243_);
v___x_260_ = lean_box(0);
v_isShared_261_ = v_isSharedCheck_265_;
goto v_resetjp_259_;
}
v_resetjp_259_:
{
lean_object* v___x_263_; 
if (v_isShared_261_ == 0)
{
v___x_263_ = v___x_260_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_264_, 0, v_a_258_);
v___x_263_ = v_reuseFailAlloc_264_;
goto v_reusejp_262_;
}
v_reusejp_262_:
{
return v___x_263_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___boxed(lean_object* v___x_266_, lean_object* v_mvarId_267_, lean_object* v_as_268_, lean_object* v_sz_269_, lean_object* v_i_270_, lean_object* v_b_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_){
_start:
{
uint8_t v___x_7244__boxed_277_; size_t v_sz_boxed_278_; size_t v_i_boxed_279_; lean_object* v_res_280_; 
v___x_7244__boxed_277_ = lean_unbox(v___x_266_);
v_sz_boxed_278_ = lean_unbox_usize(v_sz_269_);
lean_dec(v_sz_269_);
v_i_boxed_279_ = lean_unbox_usize(v_i_270_);
lean_dec(v_i_270_);
v_res_280_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2(v___x_7244__boxed_277_, v_mvarId_267_, v_as_268_, v_sz_boxed_278_, v_i_boxed_279_, v_b_271_, v___y_272_, v___y_273_, v___y_274_, v___y_275_);
lean_dec(v___y_275_);
lean_dec_ref(v___y_274_);
lean_dec(v___y_273_);
lean_dec_ref(v___y_272_);
lean_dec_ref(v_as_268_);
return v_res_280_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__4(void){
_start:
{
lean_object* v___x_287_; 
v___x_287_ = l_Lean_Meta_DiscrTree_instInhabited(lean_box(0));
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_liftReflToEq(lean_object* v_mvarId_288_, lean_object* v_a_289_, lean_object* v_a_290_, lean_object* v_a_291_, lean_object* v_a_292_){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_294_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__1));
lean_inc(v_mvarId_288_);
v___x_295_ = l_Lean_MVarId_checkNotAssigned(v_mvarId_288_, v___x_294_, v_a_289_, v_a_290_, v_a_291_, v_a_292_);
if (lean_obj_tag(v___x_295_) == 0)
{
lean_object* v_keyedConfig_296_; uint8_t v_trackZetaDelta_297_; lean_object* v_zetaDeltaSet_298_; lean_object* v_lctx_299_; lean_object* v_localInstances_300_; lean_object* v_defEqCtx_x3f_301_; lean_object* v_synthPendingDepth_302_; lean_object* v_customCanUnfoldPredicate_x3f_303_; uint8_t v_univApprox_304_; uint8_t v_inTypeClassResolution_305_; uint8_t v_cacheInferType_306_; uint8_t v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; 
lean_dec_ref_known(v___x_295_, 1);
v_keyedConfig_296_ = lean_ctor_get(v_a_289_, 0);
v_trackZetaDelta_297_ = lean_ctor_get_uint8(v_a_289_, sizeof(void*)*7);
v_zetaDeltaSet_298_ = lean_ctor_get(v_a_289_, 1);
v_lctx_299_ = lean_ctor_get(v_a_289_, 2);
v_localInstances_300_ = lean_ctor_get(v_a_289_, 3);
v_defEqCtx_x3f_301_ = lean_ctor_get(v_a_289_, 4);
v_synthPendingDepth_302_ = lean_ctor_get(v_a_289_, 5);
v_customCanUnfoldPredicate_x3f_303_ = lean_ctor_get(v_a_289_, 6);
v_univApprox_304_ = lean_ctor_get_uint8(v_a_289_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_305_ = lean_ctor_get_uint8(v_a_289_, sizeof(void*)*7 + 2);
v_cacheInferType_306_ = lean_ctor_get_uint8(v_a_289_, sizeof(void*)*7 + 3);
v___x_307_ = 2;
lean_inc_ref(v_keyedConfig_296_);
v___x_308_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_307_, v_keyedConfig_296_);
lean_inc(v_customCanUnfoldPredicate_x3f_303_);
lean_inc(v_synthPendingDepth_302_);
lean_inc(v_defEqCtx_x3f_301_);
lean_inc_ref(v_localInstances_300_);
lean_inc_ref(v_lctx_299_);
lean_inc(v_zetaDeltaSet_298_);
v___x_309_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_309_, 0, v___x_308_);
lean_ctor_set(v___x_309_, 1, v_zetaDeltaSet_298_);
lean_ctor_set(v___x_309_, 2, v_lctx_299_);
lean_ctor_set(v___x_309_, 3, v_localInstances_300_);
lean_ctor_set(v___x_309_, 4, v_defEqCtx_x3f_301_);
lean_ctor_set(v___x_309_, 5, v_synthPendingDepth_302_);
lean_ctor_set(v___x_309_, 6, v_customCanUnfoldPredicate_x3f_303_);
lean_ctor_set_uint8(v___x_309_, sizeof(void*)*7, v_trackZetaDelta_297_);
lean_ctor_set_uint8(v___x_309_, sizeof(void*)*7 + 1, v_univApprox_304_);
lean_ctor_set_uint8(v___x_309_, sizeof(void*)*7 + 2, v_inTypeClassResolution_305_);
lean_ctor_set_uint8(v___x_309_, sizeof(void*)*7 + 3, v_cacheInferType_306_);
lean_inc(v_mvarId_288_);
v___x_310_ = l_Lean_MVarId_getType_x27(v_mvarId_288_, v___x_309_, v_a_290_, v_a_291_, v_a_292_);
lean_dec_ref_known(v___x_309_, 7);
if (lean_obj_tag(v___x_310_) == 0)
{
lean_object* v_a_311_; lean_object* v___x_313_; uint8_t v_isShared_314_; uint8_t v_isSharedCheck_371_; 
v_a_311_ = lean_ctor_get(v___x_310_, 0);
v_isSharedCheck_371_ = !lean_is_exclusive(v___x_310_);
if (v_isSharedCheck_371_ == 0)
{
v___x_313_ = v___x_310_;
v_isShared_314_ = v_isSharedCheck_371_;
goto v_resetjp_312_;
}
else
{
lean_inc(v_a_311_);
lean_dec(v___x_310_);
v___x_313_ = lean_box(0);
v_isShared_314_ = v_isSharedCheck_371_;
goto v_resetjp_312_;
}
v_resetjp_312_:
{
if (lean_obj_tag(v_a_311_) == 5)
{
lean_object* v_fn_315_; 
v_fn_315_ = lean_ctor_get(v_a_311_, 0);
lean_inc_ref(v_fn_315_);
lean_dec_ref_known(v_a_311_, 2);
if (lean_obj_tag(v_fn_315_) == 5)
{
lean_object* v_fn_316_; lean_object* v___x_317_; uint8_t v___x_318_; 
v_fn_316_ = lean_ctor_get(v_fn_315_, 0);
lean_inc_ref(v_fn_316_);
lean_dec_ref_known(v_fn_315_, 2);
v___x_317_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__3));
v___x_318_ = l_Lean_Expr_isAppOf(v_fn_316_, v___x_317_);
if (v___x_318_ == 0)
{
lean_object* v___x_319_; lean_object* v_env_320_; lean_object* v___x_321_; lean_object* v_ext_322_; lean_object* v_toEnvExtension_323_; lean_object* v_asyncMode_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
lean_del_object(v___x_313_);
v___x_319_ = lean_st_ref_get(v_a_292_);
v_env_320_ = lean_ctor_get(v___x_319_, 0);
lean_inc_ref(v_env_320_);
lean_dec(v___x_319_);
v___x_321_ = l_Lean_Meta_Rfl_reflExt;
v_ext_322_ = lean_ctor_get(v___x_321_, 1);
v_toEnvExtension_323_ = lean_ctor_get(v_ext_322_, 0);
v_asyncMode_324_ = lean_ctor_get(v_toEnvExtension_323_, 2);
v___x_325_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__4, &lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__4);
v___x_326_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_325_, v___x_321_, v_env_320_, v_asyncMode_324_);
v___x_327_ = l_Lean_Meta_DiscrTree_getMatch___redArg(v___x_326_, v_fn_316_, v_a_289_, v_a_290_, v_a_291_, v_a_292_);
lean_dec(v___x_326_);
if (lean_obj_tag(v___x_327_) == 0)
{
lean_object* v_a_328_; lean_object* v___x_329_; size_t v_sz_330_; size_t v___x_331_; lean_object* v___x_332_; 
v_a_328_ = lean_ctor_get(v___x_327_, 0);
lean_inc(v_a_328_);
lean_dec_ref_known(v___x_327_, 1);
v___x_329_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2___closed__4));
v_sz_330_ = lean_array_size(v_a_328_);
v___x_331_ = ((size_t)0ULL);
lean_inc(v_mvarId_288_);
v___x_332_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_liftReflToEq_spec__2(v___x_318_, v_mvarId_288_, v_a_328_, v_sz_330_, v___x_331_, v___x_329_, v_a_289_, v_a_290_, v_a_291_, v_a_292_);
lean_dec(v_a_328_);
if (lean_obj_tag(v___x_332_) == 0)
{
lean_object* v_a_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_345_; 
v_a_333_ = lean_ctor_get(v___x_332_, 0);
v_isSharedCheck_345_ = !lean_is_exclusive(v___x_332_);
if (v_isSharedCheck_345_ == 0)
{
v___x_335_ = v___x_332_;
v_isShared_336_ = v_isSharedCheck_345_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_a_333_);
lean_dec(v___x_332_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_345_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v_fst_337_; 
v_fst_337_ = lean_ctor_get(v_a_333_, 0);
lean_inc(v_fst_337_);
lean_dec(v_a_333_);
if (lean_obj_tag(v_fst_337_) == 0)
{
lean_object* v___x_339_; 
if (v_isShared_336_ == 0)
{
lean_ctor_set(v___x_335_, 0, v_mvarId_288_);
v___x_339_ = v___x_335_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v_mvarId_288_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
return v___x_339_;
}
}
else
{
lean_object* v_val_341_; lean_object* v___x_343_; 
lean_dec(v_mvarId_288_);
v_val_341_ = lean_ctor_get(v_fst_337_, 0);
lean_inc(v_val_341_);
lean_dec_ref_known(v_fst_337_, 1);
if (v_isShared_336_ == 0)
{
lean_ctor_set(v___x_335_, 0, v_val_341_);
v___x_343_ = v___x_335_;
goto v_reusejp_342_;
}
else
{
lean_object* v_reuseFailAlloc_344_; 
v_reuseFailAlloc_344_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_344_, 0, v_val_341_);
v___x_343_ = v_reuseFailAlloc_344_;
goto v_reusejp_342_;
}
v_reusejp_342_:
{
return v___x_343_;
}
}
}
}
else
{
lean_object* v_a_346_; lean_object* v___x_348_; uint8_t v_isShared_349_; uint8_t v_isSharedCheck_353_; 
lean_dec(v_mvarId_288_);
v_a_346_ = lean_ctor_get(v___x_332_, 0);
v_isSharedCheck_353_ = !lean_is_exclusive(v___x_332_);
if (v_isSharedCheck_353_ == 0)
{
v___x_348_ = v___x_332_;
v_isShared_349_ = v_isSharedCheck_353_;
goto v_resetjp_347_;
}
else
{
lean_inc(v_a_346_);
lean_dec(v___x_332_);
v___x_348_ = lean_box(0);
v_isShared_349_ = v_isSharedCheck_353_;
goto v_resetjp_347_;
}
v_resetjp_347_:
{
lean_object* v___x_351_; 
if (v_isShared_349_ == 0)
{
v___x_351_ = v___x_348_;
goto v_reusejp_350_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v_a_346_);
v___x_351_ = v_reuseFailAlloc_352_;
goto v_reusejp_350_;
}
v_reusejp_350_:
{
return v___x_351_;
}
}
}
}
else
{
lean_object* v_a_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_361_; 
lean_dec(v_mvarId_288_);
v_a_354_ = lean_ctor_get(v___x_327_, 0);
v_isSharedCheck_361_ = !lean_is_exclusive(v___x_327_);
if (v_isSharedCheck_361_ == 0)
{
v___x_356_ = v___x_327_;
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_a_354_);
lean_dec(v___x_327_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
lean_object* v___x_359_; 
if (v_isShared_357_ == 0)
{
v___x_359_ = v___x_356_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v_a_354_);
v___x_359_ = v_reuseFailAlloc_360_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
return v___x_359_;
}
}
}
}
else
{
lean_object* v___x_363_; 
lean_dec_ref(v_fn_316_);
if (v_isShared_314_ == 0)
{
lean_ctor_set(v___x_313_, 0, v_mvarId_288_);
v___x_363_ = v___x_313_;
goto v_reusejp_362_;
}
else
{
lean_object* v_reuseFailAlloc_364_; 
v_reuseFailAlloc_364_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_364_, 0, v_mvarId_288_);
v___x_363_ = v_reuseFailAlloc_364_;
goto v_reusejp_362_;
}
v_reusejp_362_:
{
return v___x_363_;
}
}
}
else
{
lean_object* v___x_366_; 
lean_dec_ref(v_fn_315_);
if (v_isShared_314_ == 0)
{
lean_ctor_set(v___x_313_, 0, v_mvarId_288_);
v___x_366_ = v___x_313_;
goto v_reusejp_365_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v_mvarId_288_);
v___x_366_ = v_reuseFailAlloc_367_;
goto v_reusejp_365_;
}
v_reusejp_365_:
{
return v___x_366_;
}
}
}
else
{
lean_object* v___x_369_; 
lean_dec(v_a_311_);
if (v_isShared_314_ == 0)
{
lean_ctor_set(v___x_313_, 0, v_mvarId_288_);
v___x_369_ = v___x_313_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_370_; 
v_reuseFailAlloc_370_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_370_, 0, v_mvarId_288_);
v___x_369_ = v_reuseFailAlloc_370_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
return v___x_369_;
}
}
}
}
else
{
lean_object* v_a_372_; lean_object* v___x_374_; uint8_t v_isShared_375_; uint8_t v_isSharedCheck_379_; 
lean_dec(v_mvarId_288_);
v_a_372_ = lean_ctor_get(v___x_310_, 0);
v_isSharedCheck_379_ = !lean_is_exclusive(v___x_310_);
if (v_isSharedCheck_379_ == 0)
{
v___x_374_ = v___x_310_;
v_isShared_375_ = v_isSharedCheck_379_;
goto v_resetjp_373_;
}
else
{
lean_inc(v_a_372_);
lean_dec(v___x_310_);
v___x_374_ = lean_box(0);
v_isShared_375_ = v_isSharedCheck_379_;
goto v_resetjp_373_;
}
v_resetjp_373_:
{
lean_object* v___x_377_; 
if (v_isShared_375_ == 0)
{
v___x_377_ = v___x_374_;
goto v_reusejp_376_;
}
else
{
lean_object* v_reuseFailAlloc_378_; 
v_reuseFailAlloc_378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_378_, 0, v_a_372_);
v___x_377_ = v_reuseFailAlloc_378_;
goto v_reusejp_376_;
}
v_reusejp_376_:
{
return v___x_377_;
}
}
}
}
else
{
lean_object* v_a_380_; lean_object* v___x_382_; uint8_t v_isShared_383_; uint8_t v_isSharedCheck_387_; 
lean_dec(v_mvarId_288_);
v_a_380_ = lean_ctor_get(v___x_295_, 0);
v_isSharedCheck_387_ = !lean_is_exclusive(v___x_295_);
if (v_isSharedCheck_387_ == 0)
{
v___x_382_ = v___x_295_;
v_isShared_383_ = v_isSharedCheck_387_;
goto v_resetjp_381_;
}
else
{
lean_inc(v_a_380_);
lean_dec(v___x_295_);
v___x_382_ = lean_box(0);
v_isShared_383_ = v_isSharedCheck_387_;
goto v_resetjp_381_;
}
v_resetjp_381_:
{
lean_object* v___x_385_; 
if (v_isShared_383_ == 0)
{
v___x_385_ = v___x_382_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v_a_380_);
v___x_385_ = v_reuseFailAlloc_386_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
return v___x_385_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_liftReflToEq___boxed(lean_object* v_mvarId_388_, lean_object* v_a_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_, lean_object* v_a_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_mathlib_Mathlib_Tactic_liftReflToEq(v_mvarId_388_, v_a_389_, v_a_390_, v_a_391_, v_a_392_);
lean_dec(v_a_392_);
lean_dec_ref(v_a_391_);
lean_dec(v_a_390_);
lean_dec_ref(v_a_389_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0(lean_object* v_00_u03b1_395_, lean_object* v_msg_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_){
_start:
{
lean_object* v___x_402_; 
v___x_402_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0___redArg(v_msg_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0___boxed(lean_object* v_00_u03b1_403_, lean_object* v_msg_404_, lean_object* v___y_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_){
_start:
{
lean_object* v_res_410_; 
v_res_410_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_liftReflToEq_spec__0(v_00_u03b1_403_, v_msg_404_, v___y_405_, v___y_406_, v___y_407_, v___y_408_);
lean_dec(v___y_408_);
lean_dec_ref(v___y_407_);
lean_dec(v___y_406_);
lean_dec_ref(v___y_405_);
return v_res_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__0(lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_412_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
if (lean_obj_tag(v___x_420_) == 0)
{
lean_object* v_a_421_; lean_object* v___x_422_; 
v_a_421_ = lean_ctor_get(v___x_420_, 0);
lean_inc(v_a_421_);
lean_dec_ref_known(v___x_420_, 1);
v___x_422_ = l_Lean_MVarId_applyRfl(v_a_421_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
if (lean_obj_tag(v___x_422_) == 0)
{
lean_object* v___x_423_; lean_object* v___x_424_; 
lean_dec_ref_known(v___x_422_, 1);
v___x_423_ = lean_box(0);
v___x_424_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_423_, v___y_412_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
if (lean_obj_tag(v___x_424_) == 0)
{
lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_432_; 
v_isSharedCheck_432_ = !lean_is_exclusive(v___x_424_);
if (v_isSharedCheck_432_ == 0)
{
lean_object* v_unused_433_; 
v_unused_433_ = lean_ctor_get(v___x_424_, 0);
lean_dec(v_unused_433_);
v___x_426_ = v___x_424_;
v_isShared_427_ = v_isSharedCheck_432_;
goto v_resetjp_425_;
}
else
{
lean_dec(v___x_424_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_432_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_428_; lean_object* v___x_430_; 
v___x_428_ = lean_box(0);
if (v_isShared_427_ == 0)
{
lean_ctor_set(v___x_426_, 0, v___x_428_);
v___x_430_ = v___x_426_;
goto v_reusejp_429_;
}
else
{
lean_object* v_reuseFailAlloc_431_; 
v_reuseFailAlloc_431_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_431_, 0, v___x_428_);
v___x_430_ = v_reuseFailAlloc_431_;
goto v_reusejp_429_;
}
v_reusejp_429_:
{
return v___x_430_;
}
}
}
else
{
return v___x_424_;
}
}
else
{
return v___x_422_;
}
}
else
{
lean_object* v_a_434_; lean_object* v___x_436_; uint8_t v_isShared_437_; uint8_t v_isSharedCheck_441_; 
v_a_434_ = lean_ctor_get(v___x_420_, 0);
v_isSharedCheck_441_ = !lean_is_exclusive(v___x_420_);
if (v_isSharedCheck_441_ == 0)
{
v___x_436_ = v___x_420_;
v_isShared_437_ = v_isSharedCheck_441_;
goto v_resetjp_435_;
}
else
{
lean_inc(v_a_434_);
lean_dec(v___x_420_);
v___x_436_ = lean_box(0);
v_isShared_437_ = v_isSharedCheck_441_;
goto v_resetjp_435_;
}
v_resetjp_435_:
{
lean_object* v___x_439_; 
if (v_isShared_437_ == 0)
{
v___x_439_ = v___x_436_;
goto v_reusejp_438_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v_a_434_);
v___x_439_ = v_reuseFailAlloc_440_;
goto v_reusejp_438_;
}
v_reusejp_438_:
{
return v___x_439_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__0___boxed(lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v_res_451_; 
v_res_451_ = lp_mathlib_Mathlib_Tactic_rflTac___lam__0(v___y_442_, v___y_443_, v___y_444_, v___y_445_, v___y_446_, v___y_447_, v___y_448_, v___y_449_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
lean_dec(v___y_447_);
lean_dec_ref(v___y_446_);
lean_dec(v___y_445_);
lean_dec_ref(v___y_444_);
lean_dec(v___y_443_);
lean_dec_ref(v___y_442_);
return v_res_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__1(lean_object* v___f_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_452_, v___y_453_, v___y_454_, v___y_455_, v___y_456_, v___y_457_, v___y_458_, v___y_459_, v___y_460_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__1___boxed(lean_object* v___f_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_mathlib_Mathlib_Tactic_rflTac___lam__1(v___f_463_, v___y_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_, v___y_469_, v___y_470_, v___y_471_);
lean_dec(v___y_471_);
lean_dec_ref(v___y_470_);
lean_dec(v___y_469_);
lean_dec_ref(v___y_468_);
lean_dec(v___y_467_);
lean_dec_ref(v___y_466_);
lean_dec(v___y_465_);
lean_dec_ref(v___y_464_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac(lean_object* v_a_477_, lean_object* v_a_478_, lean_object* v_a_479_, lean_object* v_a_480_, lean_object* v_a_481_, lean_object* v_a_482_, lean_object* v_a_483_, lean_object* v_a_484_){
_start:
{
lean_object* v___f_486_; lean_object* v___x_487_; 
v___f_486_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rflTac___closed__1));
v___x_487_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_486_, v_a_477_, v_a_478_, v_a_479_, v_a_480_, v_a_481_, v_a_482_, v_a_483_, v_a_484_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___boxed(lean_object* v_a_488_, lean_object* v_a_489_, lean_object* v_a_490_, lean_object* v_a_491_, lean_object* v_a_492_, lean_object* v_a_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_mathlib_Mathlib_Tactic_rflTac(v_a_488_, v_a_489_, v_a_490_, v_a_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_);
lean_dec(v_a_495_);
lean_dec_ref(v_a_494_);
lean_dec(v_a_493_);
lean_dec_ref(v_a_492_);
lean_dec(v_a_491_);
lean_dec_ref(v_a_490_);
lean_dec(v_a_489_);
lean_dec_ref(v_a_488_);
return v_res_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f(lean_object* v_e_504_, lean_object* v_a_505_, lean_object* v_a_506_, lean_object* v_a_507_, lean_object* v_a_508_){
_start:
{
lean_object* v___x_513_; lean_object* v___x_514_; uint8_t v___x_515_; 
v___x_513_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__3));
v___x_514_ = lean_unsigned_to_nat(3u);
v___x_515_ = l_Lean_Expr_isAppOfArity(v_e_504_, v___x_513_, v___x_514_);
if (v___x_515_ == 0)
{
lean_object* v___x_516_; lean_object* v___x_517_; uint8_t v___x_518_; 
v___x_516_ = ((lean_object*)(lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__1));
v___x_517_ = lean_unsigned_to_nat(2u);
v___x_518_ = l_Lean_Expr_isAppOfArity(v_e_504_, v___x_516_, v___x_517_);
if (v___x_518_ == 0)
{
lean_object* v___x_519_; lean_object* v___x_520_; uint8_t v___x_521_; 
v___x_519_ = ((lean_object*)(lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__3));
v___x_520_ = lean_unsigned_to_nat(4u);
v___x_521_ = l_Lean_Expr_isAppOfArity(v_e_504_, v___x_519_, v___x_520_);
if (v___x_521_ == 0)
{
if (lean_obj_tag(v_e_504_) == 5)
{
lean_object* v_fn_522_; 
v_fn_522_ = lean_ctor_get(v_e_504_, 0);
lean_inc_ref(v_fn_522_);
if (lean_obj_tag(v_fn_522_) == 5)
{
lean_object* v_arg_523_; lean_object* v_fn_524_; lean_object* v_arg_525_; lean_object* v___x_526_; lean_object* v_env_527_; lean_object* v___x_528_; lean_object* v_ext_529_; lean_object* v_toEnvExtension_530_; lean_object* v_asyncMode_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; 
v_arg_523_ = lean_ctor_get(v_e_504_, 1);
lean_inc_ref(v_arg_523_);
lean_dec_ref_known(v_e_504_, 2);
v_fn_524_ = lean_ctor_get(v_fn_522_, 0);
lean_inc_ref_n(v_fn_524_, 2);
v_arg_525_ = lean_ctor_get(v_fn_522_, 1);
lean_inc_ref(v_arg_525_);
lean_dec_ref_known(v_fn_522_, 2);
v___x_526_ = lean_st_ref_get(v_a_508_);
v_env_527_ = lean_ctor_get(v___x_526_, 0);
lean_inc_ref(v_env_527_);
lean_dec(v___x_526_);
v___x_528_ = l_Lean_Meta_Rfl_reflExt;
v_ext_529_ = lean_ctor_get(v___x_528_, 1);
v_toEnvExtension_530_ = lean_ctor_get(v_ext_529_, 0);
v_asyncMode_531_ = lean_ctor_get(v_toEnvExtension_530_, 2);
v___x_532_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__4, &lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_liftReflToEq___closed__4);
v___x_533_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_532_, v___x_528_, v_env_527_, v_asyncMode_531_);
v___x_534_ = l_Lean_Meta_DiscrTree_getMatch___redArg(v___x_533_, v_fn_524_, v_a_505_, v_a_506_, v_a_507_, v_a_508_);
lean_dec(v___x_533_);
if (lean_obj_tag(v___x_534_) == 0)
{
lean_object* v_a_535_; lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_557_; 
v_a_535_ = lean_ctor_get(v___x_534_, 0);
v_isSharedCheck_557_ = !lean_is_exclusive(v___x_534_);
if (v_isSharedCheck_557_ == 0)
{
v___x_537_ = v___x_534_;
v_isShared_538_ = v_isSharedCheck_557_;
goto v_resetjp_536_;
}
else
{
lean_inc(v_a_535_);
lean_dec(v___x_534_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_557_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v___x_539_; lean_object* v___x_540_; uint8_t v___x_541_; 
v___x_539_ = lean_array_get_size(v_a_535_);
lean_dec(v_a_535_);
v___x_540_ = lean_unsigned_to_nat(0u);
v___x_541_ = lean_nat_dec_eq(v___x_539_, v___x_540_);
if (v___x_541_ == 0)
{
lean_object* v___x_542_; lean_object* v___x_543_; 
v___x_542_ = l_Lean_Expr_getAppFn(v_fn_524_);
lean_dec_ref(v_fn_524_);
v___x_543_ = l_Lean_Expr_constName_x3f(v___x_542_);
lean_dec_ref(v___x_542_);
if (lean_obj_tag(v___x_543_) == 0)
{
lean_del_object(v___x_537_);
lean_dec_ref(v_arg_525_);
lean_dec_ref(v_arg_523_);
goto v___jp_510_;
}
else
{
lean_object* v_val_544_; lean_object* v___x_546_; uint8_t v_isShared_547_; uint8_t v_isSharedCheck_556_; 
v_val_544_ = lean_ctor_get(v___x_543_, 0);
v_isSharedCheck_556_ = !lean_is_exclusive(v___x_543_);
if (v_isSharedCheck_556_ == 0)
{
v___x_546_ = v___x_543_;
v_isShared_547_ = v_isSharedCheck_556_;
goto v_resetjp_545_;
}
else
{
lean_inc(v_val_544_);
lean_dec(v___x_543_);
v___x_546_ = lean_box(0);
v_isShared_547_ = v_isSharedCheck_556_;
goto v_resetjp_545_;
}
v_resetjp_545_:
{
lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_551_; 
v___x_548_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_548_, 0, v_arg_525_);
lean_ctor_set(v___x_548_, 1, v_arg_523_);
v___x_549_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_549_, 0, v_val_544_);
lean_ctor_set(v___x_549_, 1, v___x_548_);
if (v_isShared_547_ == 0)
{
lean_ctor_set(v___x_546_, 0, v___x_549_);
v___x_551_ = v___x_546_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v___x_549_);
v___x_551_ = v_reuseFailAlloc_555_;
goto v_reusejp_550_;
}
v_reusejp_550_:
{
lean_object* v___x_553_; 
if (v_isShared_538_ == 0)
{
lean_ctor_set(v___x_537_, 0, v___x_551_);
v___x_553_ = v___x_537_;
goto v_reusejp_552_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v___x_551_);
v___x_553_ = v_reuseFailAlloc_554_;
goto v_reusejp_552_;
}
v_reusejp_552_:
{
return v___x_553_;
}
}
}
}
}
else
{
lean_del_object(v___x_537_);
lean_dec_ref(v_arg_525_);
lean_dec_ref(v_fn_524_);
lean_dec_ref(v_arg_523_);
goto v___jp_510_;
}
}
}
else
{
lean_object* v_a_558_; lean_object* v___x_560_; uint8_t v_isShared_561_; uint8_t v_isSharedCheck_565_; 
lean_dec_ref(v_arg_525_);
lean_dec_ref(v_fn_524_);
lean_dec_ref(v_arg_523_);
v_a_558_ = lean_ctor_get(v___x_534_, 0);
v_isSharedCheck_565_ = !lean_is_exclusive(v___x_534_);
if (v_isSharedCheck_565_ == 0)
{
v___x_560_ = v___x_534_;
v_isShared_561_ = v_isSharedCheck_565_;
goto v_resetjp_559_;
}
else
{
lean_inc(v_a_558_);
lean_dec(v___x_534_);
v___x_560_ = lean_box(0);
v_isShared_561_ = v_isSharedCheck_565_;
goto v_resetjp_559_;
}
v_resetjp_559_:
{
lean_object* v___x_563_; 
if (v_isShared_561_ == 0)
{
v___x_563_ = v___x_560_;
goto v_reusejp_562_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v_a_558_);
v___x_563_ = v_reuseFailAlloc_564_;
goto v_reusejp_562_;
}
v_reusejp_562_:
{
return v___x_563_;
}
}
}
}
else
{
lean_dec_ref_known(v_e_504_, 2);
lean_dec_ref(v_fn_522_);
goto v___jp_510_;
}
}
else
{
lean_dec_ref(v_e_504_);
goto v___jp_510_;
}
}
else
{
lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; 
v___x_566_ = l_Lean_Expr_appFn_x21(v_e_504_);
v___x_567_ = l_Lean_Expr_appFn_x21(v___x_566_);
lean_dec_ref(v___x_566_);
v___x_568_ = l_Lean_Expr_appArg_x21(v___x_567_);
lean_dec_ref(v___x_567_);
v___x_569_ = l_Lean_Expr_appArg_x21(v_e_504_);
lean_dec_ref(v_e_504_);
v___x_570_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_570_, 0, v___x_568_);
lean_ctor_set(v___x_570_, 1, v___x_569_);
v___x_571_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_571_, 0, v___x_519_);
lean_ctor_set(v___x_571_, 1, v___x_570_);
v___x_572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_572_, 0, v___x_571_);
v___x_573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_573_, 0, v___x_572_);
return v___x_573_;
}
}
else
{
lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; 
v___x_574_ = l_Lean_Expr_appFn_x21(v_e_504_);
v___x_575_ = l_Lean_Expr_appArg_x21(v___x_574_);
lean_dec_ref(v___x_574_);
v___x_576_ = l_Lean_Expr_appArg_x21(v_e_504_);
lean_dec_ref(v_e_504_);
v___x_577_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_577_, 0, v___x_575_);
lean_ctor_set(v___x_577_, 1, v___x_576_);
v___x_578_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_578_, 0, v___x_516_);
lean_ctor_set(v___x_578_, 1, v___x_577_);
v___x_579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_579_, 0, v___x_578_);
v___x_580_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_580_, 0, v___x_579_);
return v___x_580_;
}
}
else
{
lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; 
v___x_581_ = l_Lean_Expr_appFn_x21(v_e_504_);
v___x_582_ = l_Lean_Expr_appArg_x21(v___x_581_);
lean_dec_ref(v___x_581_);
v___x_583_ = l_Lean_Expr_appArg_x21(v_e_504_);
lean_dec_ref(v_e_504_);
v___x_584_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_584_, 0, v___x_582_);
lean_ctor_set(v___x_584_, 1, v___x_583_);
v___x_585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_585_, 0, v___x_513_);
lean_ctor_set(v___x_585_, 1, v___x_584_);
v___x_586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_586_, 0, v___x_585_);
v___x_587_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_587_, 0, v___x_586_);
return v___x_587_;
}
v___jp_510_:
{
lean_object* v___x_511_; lean_object* v___x_512_; 
v___x_511_ = lean_box(0);
v___x_512_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_512_, 0, v___x_511_);
return v___x_512_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___boxed(lean_object* v_e_588_, lean_object* v_a_589_, lean_object* v_a_590_, lean_object* v_a_591_, lean_object* v_a_592_, lean_object* v_a_593_){
_start:
{
lean_object* v_res_594_; 
v_res_594_ = lp_mathlib_Lean_Expr_relSidesIfRefl_x3f(v_e_588_, v_a_589_, v_a_590_, v_a_591_, v_a_592_);
lean_dec(v_a_592_);
lean_dec_ref(v_a_591_);
lean_dec(v_a_590_);
lean_dec_ref(v_a_589_);
return v_res_594_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Relation_Rfl(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Rfl(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Relation_Rfl(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Rfl(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Relation_Rfl(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Relation_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Relation_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Relation_Rfl(builtin);
}
#ifdef __cplusplus
}
#endif
