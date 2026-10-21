// Lean compiler output
// Module: Mathlib.Tactic.FunProp.ToBatteries
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_FVarId_getUserName___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* l_Lean_Expr_letE___override(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_range(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Meta_mkAppM_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_eta(lean_object*);
uint8_t l_Lean_Expr_isLambda(lean_object*);
lean_object* l_Lean_Expr_bvar___override(lean_object*);
lean_object* l_Lean_Expr_looseBVarRange(lean_object*);
lean_object* lean_expr_instantiate(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
uint8_t l_Lean_Expr_isHeadBetaTargetFn(uint8_t, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__5_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__9_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_swapBVars(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_swapBVars___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Prod"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(117, 121, 37, 123, 104, 28, 189, 89)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "_inhabitedExprDummy"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 247, 56, 151, 29, 116, 116, 243)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdElem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fst"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__0_value),LEAN_SCALAR_PTR_LITERAL(170, 44, 236, 58, 247, 164, 254, 114)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "snd"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__2_value),LEAN_SCALAR_PTR_LITERAL(35, 40, 163, 84, 60, 49, 151, 224)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdProj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_mkProdSplitElem_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_mkProdSplitElem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdSplitElem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdSplitElem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___lam__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_etaExpand1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_ToBatteries_0__Mathlib_Meta_FunProp_betaThroughLetAux(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_ToBatteries_0__Mathlib_Meta_FunProp_betaThroughLetAux_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_ToBatteries_0__Mathlib_Meta_FunProp_betaThroughLetAux_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_betaThroughLet(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___lam__0(lean_object* v___x_1_, lean_object* v_inst_2_, lean_object* v_a_3_, lean_object* v_b_4_, lean_object* v_inst_5_, lean_object* v___x_6_, lean_object* v_j_7_, lean_object* v_h_x27_8_, lean_object* v_____s_9_){
_start:
{
uint8_t v___x_10_; 
v___x_10_ = lean_nat_dec_eq(v_____s_9_, v___x_1_);
if (v___x_10_ == 0)
{
lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; uint8_t v___x_14_; 
v___x_11_ = lean_array_get_borrowed(v_inst_2_, v_a_3_, v_____s_9_);
v___x_12_ = lean_array_fget_borrowed(v_b_4_, v_j_7_);
lean_inc(v___x_12_);
lean_inc(v___x_11_);
v___x_13_ = lean_apply_2(v_inst_5_, v___x_11_, v___x_12_);
v___x_14_ = lean_unbox(v___x_13_);
if (v___x_14_ == 0)
{
lean_object* v___x_15_; 
v___x_15_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_15_, 0, v_____s_9_);
return v___x_15_;
}
else
{
lean_object* v_i_16_; lean_object* v___x_17_; 
v_i_16_ = lean_nat_add(v_____s_9_, v___x_6_);
lean_dec(v_____s_9_);
v___x_17_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_17_, 0, v_i_16_);
return v___x_17_;
}
}
else
{
lean_object* v___x_18_; 
lean_dec_ref(v_inst_5_);
v___x_18_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_18_, 0, v_____s_9_);
return v___x_18_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___lam__0___boxed(lean_object* v___x_19_, lean_object* v_inst_20_, lean_object* v_a_21_, lean_object* v_b_22_, lean_object* v_inst_23_, lean_object* v___x_24_, lean_object* v_j_25_, lean_object* v_h_x27_26_, lean_object* v_____s_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___lam__0(v___x_19_, v_inst_20_, v_a_21_, v_b_22_, v_inst_23_, v___x_24_, v_j_25_, v_h_x27_26_, v_____s_27_);
lean_dec(v_j_25_);
lean_dec(v___x_24_);
lean_dec_ref(v_b_22_);
lean_dec_ref(v_a_21_);
lean_dec(v_inst_20_);
lean_dec(v___x_19_);
return v_res_28_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg(lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_a_50_, lean_object* v_b_51_){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; uint8_t v___x_54_; 
v___x_52_ = lean_array_get_size(v_b_51_);
v___x_53_ = lean_array_get_size(v_a_50_);
v___x_54_ = lean_nat_dec_lt(v___x_52_, v___x_53_);
if (v___x_54_ == 0)
{
lean_object* v_i_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___f_58_; lean_object* v___x_59_; lean_object* v___x_60_; uint8_t v___x_61_; 
v_i_55_ = lean_unsigned_to_nat(0u);
v___x_56_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___closed__9));
v___x_57_ = lean_unsigned_to_nat(1u);
v___f_58_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___lam__0___boxed), 9, 6);
lean_closure_set(v___f_58_, 0, v___x_53_);
lean_closure_set(v___f_58_, 1, v_inst_48_);
lean_closure_set(v___f_58_, 2, v_a_50_);
lean_closure_set(v___f_58_, 3, v_b_51_);
lean_closure_set(v___f_58_, 4, v_inst_49_);
lean_closure_set(v___f_58_, 5, v___x_57_);
v___x_59_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_59_, 0, v_i_55_);
lean_ctor_set(v___x_59_, 1, v___x_52_);
lean_ctor_set(v___x_59_, 2, v___x_57_);
v___x_60_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v___x_56_, v___x_59_, v___f_58_, v_i_55_, v_i_55_, lean_box(0), lean_box(0));
v___x_61_ = lean_nat_dec_eq(v___x_60_, v___x_53_);
lean_dec(v___x_60_);
return v___x_61_;
}
else
{
uint8_t v___x_62_; 
lean_dec_ref(v_b_51_);
lean_dec_ref(v_a_50_);
lean_dec_ref(v_inst_49_);
lean_dec(v_inst_48_);
v___x_62_ = 0;
return v___x_62_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg___boxed(lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_a_65_, lean_object* v_b_66_){
_start:
{
uint8_t v_res_67_; lean_object* v_r_68_; 
v_res_67_ = lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg(v_inst_63_, v_inst_64_, v_a_65_, v_b_66_);
v_r_68_ = lean_box(v_res_67_);
return v_r_68_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf(lean_object* v_00_u03b1_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_a_72_, lean_object* v_b_73_){
_start:
{
uint8_t v___x_74_; 
v___x_74_ = lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___redArg(v_inst_70_, v_inst_71_, v_a_72_, v_b_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf___boxed(lean_object* v_00_u03b1_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_a_78_, lean_object* v_b_79_){
_start:
{
uint8_t v_res_80_; lean_object* v_r_81_; 
v_res_80_ = lp_mathlib_Mathlib_Meta_FunProp_isOrderedSubsetOf(v_00_u03b1_75_, v_inst_76_, v_inst_77_, v_a_78_, v_b_79_);
v_r_81_ = lean_box(v_res_80_);
return v_r_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0___redArg(lean_object* v_i_82_, lean_object* v_j_83_, lean_object* v_range_84_, lean_object* v_b_85_, lean_object* v_i_86_){
_start:
{
lean_object* v_stop_87_; lean_object* v_step_88_; lean_object* v___y_90_; uint8_t v___x_95_; 
v_stop_87_ = lean_ctor_get(v_range_84_, 1);
v_step_88_ = lean_ctor_get(v_range_84_, 2);
v___x_95_ = lean_nat_dec_lt(v_i_86_, v_stop_87_);
if (v___x_95_ == 0)
{
lean_dec(v_i_86_);
lean_dec(v_j_83_);
lean_dec(v_i_82_);
return v_b_85_;
}
else
{
uint8_t v___x_96_; 
v___x_96_ = lean_nat_dec_eq(v_i_86_, v_i_82_);
if (v___x_96_ == 0)
{
uint8_t v___x_97_; 
v___x_97_ = lean_nat_dec_eq(v_i_86_, v_j_83_);
if (v___x_97_ == 0)
{
lean_inc(v_i_86_);
v___y_90_ = v_i_86_;
goto v___jp_89_;
}
else
{
lean_inc(v_i_82_);
v___y_90_ = v_i_82_;
goto v___jp_89_;
}
}
else
{
lean_inc(v_j_83_);
v___y_90_ = v_j_83_;
goto v___jp_89_;
}
}
v___jp_89_:
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_91_ = l_Lean_Expr_bvar___override(v___y_90_);
v___x_92_ = lean_array_push(v_b_85_, v___x_91_);
v___x_93_ = lean_nat_add(v_i_86_, v_step_88_);
lean_dec(v_i_86_);
v_b_85_ = v___x_92_;
v_i_86_ = v___x_93_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0___redArg___boxed(lean_object* v_i_98_, lean_object* v_j_99_, lean_object* v_range_100_, lean_object* v_b_101_, lean_object* v_i_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0___redArg(v_i_98_, v_j_99_, v_range_100_, v_b_101_, v_i_102_);
lean_dec_ref(v_range_100_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_swapBVars(lean_object* v_e_104_, lean_object* v_i_105_, lean_object* v_j_106_){
_start:
{
lean_object* v___x_107_; lean_object* v_a_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v_swapBVarArray_112_; lean_object* v___x_113_; 
v___x_107_ = l_Lean_Expr_looseBVarRange(v_e_104_);
v_a_108_ = lean_mk_empty_array_with_capacity(v___x_107_);
v___x_109_ = lean_unsigned_to_nat(0u);
v___x_110_ = lean_unsigned_to_nat(1u);
v___x_111_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_111_, 0, v___x_109_);
lean_ctor_set(v___x_111_, 1, v___x_107_);
lean_ctor_set(v___x_111_, 2, v___x_110_);
v_swapBVarArray_112_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0___redArg(v_i_105_, v_j_106_, v___x_111_, v_a_108_, v___x_109_);
lean_dec_ref_known(v___x_111_, 3);
v___x_113_ = lean_expr_instantiate(v_e_104_, v_swapBVarArray_112_);
lean_dec_ref(v_swapBVarArray_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_swapBVars___boxed(lean_object* v_e_114_, lean_object* v_i_115_, lean_object* v_j_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Lean_Expr_swapBVars(v_e_114_, v_i_115_, v_j_116_);
lean_dec_ref(v_e_114_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0(lean_object* v_i_118_, lean_object* v_j_119_, lean_object* v_range_120_, lean_object* v_b_121_, lean_object* v_i_122_, lean_object* v_hs_123_, lean_object* v_hl_124_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0___redArg(v_i_118_, v_j_119_, v_range_120_, v_b_121_, v_i_122_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0___boxed(lean_object* v_i_126_, lean_object* v_j_127_, lean_object* v_range_128_, lean_object* v_b_129_, lean_object* v_i_130_, lean_object* v_hs_131_, lean_object* v_hl_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Expr_swapBVars_spec__0(v_i_126_, v_j_127_, v_range_128_, v_b_129_, v_i_130_, v_hs_131_, v_hl_132_);
lean_dec_ref(v_range_128_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0(lean_object* v_as_139_, size_t v_i_140_, size_t v_stop_141_, lean_object* v_b_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_){
_start:
{
uint8_t v___x_148_; 
v___x_148_ = lean_usize_dec_eq(v_i_140_, v_stop_141_);
if (v___x_148_ == 0)
{
size_t v___x_149_; size_t v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_149_ = ((size_t)1ULL);
v___x_150_ = lean_usize_sub(v_i_140_, v___x_149_);
v___x_151_ = lean_array_uget_borrowed(v_as_139_, v___x_150_);
v___x_152_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___closed__2));
v___x_153_ = lean_unsigned_to_nat(2u);
v___x_154_ = lean_mk_empty_array_with_capacity(v___x_153_);
lean_inc(v___x_151_);
v___x_155_ = lean_array_push(v___x_154_, v___x_151_);
v___x_156_ = lean_array_push(v___x_155_, v_b_142_);
v___x_157_ = l_Lean_Meta_mkAppM(v___x_152_, v___x_156_, v___y_143_, v___y_144_, v___y_145_, v___y_146_);
if (lean_obj_tag(v___x_157_) == 0)
{
lean_object* v_a_158_; 
v_a_158_ = lean_ctor_get(v___x_157_, 0);
lean_inc(v_a_158_);
lean_dec_ref_known(v___x_157_, 1);
v_i_140_ = v___x_150_;
v_b_142_ = v_a_158_;
goto _start;
}
else
{
return v___x_157_;
}
}
else
{
lean_object* v___x_160_; 
v___x_160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_160_, 0, v_b_142_);
return v___x_160_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0___boxed(lean_object* v_as_161_, lean_object* v_i_162_, lean_object* v_stop_163_, lean_object* v_b_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_){
_start:
{
size_t v_i_boxed_170_; size_t v_stop_boxed_171_; lean_object* v_res_172_; 
v_i_boxed_170_ = lean_unbox_usize(v_i_162_);
lean_dec(v_i_162_);
v_stop_boxed_171_ = lean_unbox_usize(v_stop_163_);
lean_dec(v_stop_163_);
v_res_172_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0(v_as_161_, v_i_boxed_170_, v_stop_boxed_171_, v_b_164_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
lean_dec(v___y_168_);
lean_dec_ref(v___y_167_);
lean_dec(v___y_166_);
lean_dec_ref(v___y_165_);
lean_dec_ref(v_as_161_);
return v_res_172_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__2(void){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_176_ = lean_box(0);
v___x_177_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__1));
v___x_178_ = l_Lean_Expr_const___override(v___x_177_, v___x_176_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdElem(lean_object* v_xs_179_, lean_object* v_a_180_, lean_object* v_a_181_, lean_object* v_a_182_, lean_object* v_a_183_){
_start:
{
lean_object* v___x_185_; lean_object* v_zero_186_; uint8_t v_isZero_187_; 
v___x_185_ = lean_array_get_size(v_xs_179_);
v_zero_186_ = lean_unsigned_to_nat(0u);
v_isZero_187_ = lean_nat_dec_eq(v___x_185_, v_zero_186_);
if (v_isZero_187_ == 1)
{
lean_object* v___x_188_; lean_object* v___x_189_; 
lean_dec_ref(v_xs_179_);
v___x_188_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__2, &lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__2_once, _init_lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___closed__2);
v___x_189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_189_, 0, v___x_188_);
return v___x_189_;
}
else
{
lean_object* v_one_190_; lean_object* v_n_191_; uint8_t v___x_192_; 
v_one_190_ = lean_unsigned_to_nat(1u);
v_n_191_ = lean_nat_sub(v___x_185_, v_one_190_);
v___x_192_ = lean_nat_dec_eq(v_n_191_, v_zero_186_);
if (v___x_192_ == 0)
{
lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v_array_195_; lean_object* v_start_196_; lean_object* v_stop_197_; lean_object* v___x_198_; uint8_t v___x_199_; 
v___x_193_ = lean_array_fget(v_xs_179_, v_n_191_);
v___x_194_ = l_Array_toSubarray___redArg(v_xs_179_, v_zero_186_, v_n_191_);
v_array_195_ = lean_ctor_get(v___x_194_, 0);
lean_inc_ref(v_array_195_);
v_start_196_ = lean_ctor_get(v___x_194_, 1);
lean_inc(v_start_196_);
v_stop_197_ = lean_ctor_get(v___x_194_, 2);
lean_inc(v_stop_197_);
lean_dec_ref(v___x_194_);
v___x_198_ = lean_array_get_size(v_array_195_);
v___x_199_ = lean_nat_dec_le(v_stop_197_, v___x_198_);
if (v___x_199_ == 0)
{
uint8_t v___x_200_; 
lean_dec(v_stop_197_);
v___x_200_ = lean_nat_dec_lt(v_start_196_, v___x_198_);
if (v___x_200_ == 0)
{
lean_object* v___x_201_; 
lean_dec(v_start_196_);
lean_dec_ref(v_array_195_);
v___x_201_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_201_, 0, v___x_193_);
return v___x_201_;
}
else
{
size_t v___x_202_; size_t v___x_203_; lean_object* v___x_204_; 
v___x_202_ = lean_usize_of_nat(v___x_198_);
v___x_203_ = lean_usize_of_nat(v_start_196_);
lean_dec(v_start_196_);
v___x_204_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0(v_array_195_, v___x_202_, v___x_203_, v___x_193_, v_a_180_, v_a_181_, v_a_182_, v_a_183_);
lean_dec_ref(v_array_195_);
return v___x_204_;
}
}
else
{
uint8_t v___x_205_; 
v___x_205_ = lean_nat_dec_lt(v_start_196_, v_stop_197_);
if (v___x_205_ == 0)
{
lean_object* v___x_206_; 
lean_dec(v_stop_197_);
lean_dec(v_start_196_);
lean_dec_ref(v_array_195_);
v___x_206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_206_, 0, v___x_193_);
return v___x_206_;
}
else
{
size_t v___x_207_; size_t v___x_208_; lean_object* v___x_209_; 
v___x_207_ = lean_usize_of_nat(v_stop_197_);
lean_dec(v_stop_197_);
v___x_208_ = lean_usize_of_nat(v_start_196_);
lean_dec(v_start_196_);
v___x_209_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkProdElem_spec__0(v_array_195_, v___x_207_, v___x_208_, v___x_193_, v_a_180_, v_a_181_, v_a_182_, v_a_183_);
lean_dec_ref(v_array_195_);
return v___x_209_;
}
}
}
else
{
lean_object* v___x_210_; lean_object* v___x_211_; 
lean_dec(v_n_191_);
v___x_210_ = lean_array_fget(v_xs_179_, v_zero_186_);
lean_dec_ref(v_xs_179_);
v___x_211_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_211_, 0, v___x_210_);
return v___x_211_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdElem___boxed(lean_object* v_xs_212_, lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v_a_215_, lean_object* v_a_216_, lean_object* v_a_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_Mathlib_Meta_FunProp_mkProdElem(v_xs_212_, v_a_213_, v_a_214_, v_a_215_, v_a_216_);
lean_dec(v_a_216_);
lean_dec_ref(v_a_215_);
lean_dec(v_a_214_);
lean_dec_ref(v_a_213_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdProj(lean_object* v_x_227_, lean_object* v_i_228_, lean_object* v_n_229_, lean_object* v_a_230_, lean_object* v_a_231_, lean_object* v_a_232_, lean_object* v_a_233_){
_start:
{
lean_object* v_zero_235_; uint8_t v_isZero_236_; 
v_zero_235_ = lean_unsigned_to_nat(0u);
v_isZero_236_ = lean_nat_dec_eq(v_n_229_, v_zero_235_);
if (v_isZero_236_ == 1)
{
lean_object* v___x_237_; 
lean_dec(v_n_229_);
lean_dec(v_i_228_);
v___x_237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_237_, 0, v_x_227_);
return v___x_237_;
}
else
{
lean_object* v_one_238_; lean_object* v_n_239_; uint8_t v___x_240_; 
v_one_238_ = lean_unsigned_to_nat(1u);
v_n_239_ = lean_nat_sub(v_n_229_, v_one_238_);
lean_dec(v_n_229_);
v___x_240_ = lean_nat_dec_eq(v_n_239_, v_zero_235_);
if (v___x_240_ == 0)
{
uint8_t v_isZero_241_; 
v_isZero_241_ = lean_nat_dec_eq(v_i_228_, v_zero_235_);
if (v_isZero_241_ == 1)
{
lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; 
lean_dec(v_n_239_);
lean_dec(v_i_228_);
v___x_242_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__1));
v___x_243_ = lean_mk_empty_array_with_capacity(v_one_238_);
v___x_244_ = lean_array_push(v___x_243_, v_x_227_);
v___x_245_ = l_Lean_Meta_mkAppM(v___x_242_, v___x_244_, v_a_230_, v_a_231_, v_a_232_, v_a_233_);
return v___x_245_;
}
else
{
lean_object* v_keyedConfig_246_; uint8_t v_trackZetaDelta_247_; lean_object* v_zetaDeltaSet_248_; lean_object* v_lctx_249_; lean_object* v_localInstances_250_; lean_object* v_defEqCtx_x3f_251_; lean_object* v_synthPendingDepth_252_; lean_object* v_customCanUnfoldPredicate_x3f_253_; uint8_t v_univApprox_254_; uint8_t v_inTypeClassResolution_255_; uint8_t v_cacheInferType_256_; uint8_t v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; 
v_keyedConfig_246_ = lean_ctor_get(v_a_230_, 0);
v_trackZetaDelta_247_ = lean_ctor_get_uint8(v_a_230_, sizeof(void*)*7);
v_zetaDeltaSet_248_ = lean_ctor_get(v_a_230_, 1);
v_lctx_249_ = lean_ctor_get(v_a_230_, 2);
v_localInstances_250_ = lean_ctor_get(v_a_230_, 3);
v_defEqCtx_x3f_251_ = lean_ctor_get(v_a_230_, 4);
v_synthPendingDepth_252_ = lean_ctor_get(v_a_230_, 5);
v_customCanUnfoldPredicate_x3f_253_ = lean_ctor_get(v_a_230_, 6);
v_univApprox_254_ = lean_ctor_get_uint8(v_a_230_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_255_ = lean_ctor_get_uint8(v_a_230_, sizeof(void*)*7 + 2);
v_cacheInferType_256_ = lean_ctor_get_uint8(v_a_230_, sizeof(void*)*7 + 3);
v___x_257_ = 0;
v___x_258_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___closed__3));
v___x_259_ = lean_mk_empty_array_with_capacity(v_one_238_);
v___x_260_ = lean_array_push(v___x_259_, v_x_227_);
lean_inc_ref(v_keyedConfig_246_);
v___x_261_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_257_, v_keyedConfig_246_);
lean_inc(v_customCanUnfoldPredicate_x3f_253_);
lean_inc(v_synthPendingDepth_252_);
lean_inc(v_defEqCtx_x3f_251_);
lean_inc_ref(v_localInstances_250_);
lean_inc_ref(v_lctx_249_);
lean_inc(v_zetaDeltaSet_248_);
v___x_262_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_262_, 0, v___x_261_);
lean_ctor_set(v___x_262_, 1, v_zetaDeltaSet_248_);
lean_ctor_set(v___x_262_, 2, v_lctx_249_);
lean_ctor_set(v___x_262_, 3, v_localInstances_250_);
lean_ctor_set(v___x_262_, 4, v_defEqCtx_x3f_251_);
lean_ctor_set(v___x_262_, 5, v_synthPendingDepth_252_);
lean_ctor_set(v___x_262_, 6, v_customCanUnfoldPredicate_x3f_253_);
lean_ctor_set_uint8(v___x_262_, sizeof(void*)*7, v_trackZetaDelta_247_);
lean_ctor_set_uint8(v___x_262_, sizeof(void*)*7 + 1, v_univApprox_254_);
lean_ctor_set_uint8(v___x_262_, sizeof(void*)*7 + 2, v_inTypeClassResolution_255_);
lean_ctor_set_uint8(v___x_262_, sizeof(void*)*7 + 3, v_cacheInferType_256_);
v___x_263_ = l_Lean_Meta_mkAppM(v___x_258_, v___x_260_, v___x_262_, v_a_231_, v_a_232_, v_a_233_);
lean_dec_ref_known(v___x_262_, 7);
if (lean_obj_tag(v___x_263_) == 0)
{
lean_object* v_a_264_; lean_object* v_n_265_; 
v_a_264_ = lean_ctor_get(v___x_263_, 0);
lean_inc(v_a_264_);
lean_dec_ref_known(v___x_263_, 1);
v_n_265_ = lean_nat_sub(v_i_228_, v_one_238_);
lean_dec(v_i_228_);
v_x_227_ = v_a_264_;
v_i_228_ = v_n_265_;
v_n_229_ = v_n_239_;
goto _start;
}
else
{
lean_dec(v_n_239_);
lean_dec(v_i_228_);
return v___x_263_;
}
}
}
else
{
lean_object* v___x_267_; 
lean_dec(v_n_239_);
lean_dec(v_i_228_);
v___x_267_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_267_, 0, v_x_227_);
return v___x_267_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdProj___boxed(lean_object* v_x_268_, lean_object* v_i_269_, lean_object* v_n_270_, lean_object* v_a_271_, lean_object* v_a_272_, lean_object* v_a_273_, lean_object* v_a_274_, lean_object* v_a_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_Mathlib_Meta_FunProp_mkProdProj(v_x_268_, v_i_269_, v_n_270_, v_a_271_, v_a_272_, v_a_273_, v_a_274_);
lean_dec(v_a_274_);
lean_dec_ref(v_a_273_);
lean_dec(v_a_272_);
lean_dec_ref(v_a_271_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_mkProdSplitElem_spec__0(lean_object* v_xs_277_, lean_object* v_n_278_, size_t v_sz_279_, size_t v_i_280_, lean_object* v_bs_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_){
_start:
{
uint8_t v___x_287_; 
v___x_287_ = lean_usize_dec_lt(v_i_280_, v_sz_279_);
if (v___x_287_ == 0)
{
lean_object* v___x_288_; 
lean_dec(v_n_278_);
lean_dec_ref(v_xs_277_);
v___x_288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_288_, 0, v_bs_281_);
return v___x_288_;
}
else
{
lean_object* v_v_289_; lean_object* v___x_290_; 
v_v_289_ = lean_array_uget_borrowed(v_bs_281_, v_i_280_);
lean_inc(v_n_278_);
lean_inc(v_v_289_);
lean_inc_ref(v_xs_277_);
v___x_290_ = lp_mathlib_Mathlib_Meta_FunProp_mkProdProj(v_xs_277_, v_v_289_, v_n_278_, v___y_282_, v___y_283_, v___y_284_, v___y_285_);
if (lean_obj_tag(v___x_290_) == 0)
{
lean_object* v_a_291_; lean_object* v___x_292_; lean_object* v_bs_x27_293_; size_t v___x_294_; size_t v___x_295_; lean_object* v___x_296_; 
v_a_291_ = lean_ctor_get(v___x_290_, 0);
lean_inc(v_a_291_);
lean_dec_ref_known(v___x_290_, 1);
v___x_292_ = lean_unsigned_to_nat(0u);
v_bs_x27_293_ = lean_array_uset(v_bs_281_, v_i_280_, v___x_292_);
v___x_294_ = ((size_t)1ULL);
v___x_295_ = lean_usize_add(v_i_280_, v___x_294_);
v___x_296_ = lean_array_uset(v_bs_x27_293_, v_i_280_, v_a_291_);
v_i_280_ = v___x_295_;
v_bs_281_ = v___x_296_;
goto _start;
}
else
{
lean_object* v_a_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_305_; 
lean_dec_ref(v_bs_281_);
lean_dec(v_n_278_);
lean_dec_ref(v_xs_277_);
v_a_298_ = lean_ctor_get(v___x_290_, 0);
v_isSharedCheck_305_ = !lean_is_exclusive(v___x_290_);
if (v_isSharedCheck_305_ == 0)
{
v___x_300_ = v___x_290_;
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_a_298_);
lean_dec(v___x_290_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_303_; 
if (v_isShared_301_ == 0)
{
v___x_303_ = v___x_300_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_304_; 
v_reuseFailAlloc_304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_304_, 0, v_a_298_);
v___x_303_ = v_reuseFailAlloc_304_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
return v___x_303_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_mkProdSplitElem_spec__0___boxed(lean_object* v_xs_306_, lean_object* v_n_307_, lean_object* v_sz_308_, lean_object* v_i_309_, lean_object* v_bs_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_){
_start:
{
size_t v_sz_boxed_316_; size_t v_i_boxed_317_; lean_object* v_res_318_; 
v_sz_boxed_316_ = lean_unbox_usize(v_sz_308_);
lean_dec(v_sz_308_);
v_i_boxed_317_ = lean_unbox_usize(v_i_309_);
lean_dec(v_i_309_);
v_res_318_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_mkProdSplitElem_spec__0(v_xs_306_, v_n_307_, v_sz_boxed_316_, v_i_boxed_317_, v_bs_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_);
lean_dec(v___y_314_);
lean_dec_ref(v___y_313_);
lean_dec(v___y_312_);
lean_dec_ref(v___y_311_);
return v_res_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdSplitElem(lean_object* v_xs_319_, lean_object* v_n_320_, lean_object* v_a_321_, lean_object* v_a_322_, lean_object* v_a_323_, lean_object* v_a_324_){
_start:
{
lean_object* v___x_326_; size_t v_sz_327_; size_t v___x_328_; lean_object* v___x_329_; 
lean_inc(v_n_320_);
v___x_326_ = l_Array_range(v_n_320_);
v_sz_327_ = lean_array_size(v___x_326_);
v___x_328_ = ((size_t)0ULL);
v___x_329_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_mkProdSplitElem_spec__0(v_xs_319_, v_n_320_, v_sz_327_, v___x_328_, v___x_326_, v_a_321_, v_a_322_, v_a_323_, v_a_324_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkProdSplitElem___boxed(lean_object* v_xs_330_, lean_object* v_n_331_, lean_object* v_a_332_, lean_object* v_a_333_, lean_object* v_a_334_, lean_object* v_a_335_, lean_object* v_a_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib_Mathlib_Meta_FunProp_mkProdSplitElem(v_xs_330_, v_n_331_, v_a_332_, v_a_333_, v_a_334_, v_a_335_);
lean_dec(v_a_335_);
lean_dec_ref(v_a_334_);
lean_dec(v_a_333_);
lean_dec_ref(v_a_332_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg___lam__0(lean_object* v_k_338_, lean_object* v_b_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_){
_start:
{
lean_object* v___x_345_; 
lean_inc(v___y_343_);
lean_inc_ref(v___y_342_);
lean_inc(v___y_341_);
lean_inc_ref(v___y_340_);
v___x_345_ = lean_apply_6(v_k_338_, v_b_339_, v___y_340_, v___y_341_, v___y_342_, v___y_343_, lean_box(0));
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg___lam__0___boxed(lean_object* v_k_346_, lean_object* v_b_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg___lam__0(v_k_346_, v_b_347_, v___y_348_, v___y_349_, v___y_350_, v___y_351_);
lean_dec(v___y_351_);
lean_dec_ref(v___y_350_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg(lean_object* v_name_354_, uint8_t v_bi_355_, lean_object* v_type_356_, lean_object* v_k_357_, uint8_t v_kind_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_){
_start:
{
lean_object* v___f_364_; lean_object* v___x_365_; 
v___f_364_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_364_, 0, v_k_357_);
v___x_365_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_354_, v_bi_355_, v_type_356_, v___f_364_, v_kind_358_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
if (lean_obj_tag(v___x_365_) == 0)
{
lean_object* v_a_366_; lean_object* v___x_368_; uint8_t v_isShared_369_; uint8_t v_isSharedCheck_373_; 
v_a_366_ = lean_ctor_get(v___x_365_, 0);
v_isSharedCheck_373_ = !lean_is_exclusive(v___x_365_);
if (v_isSharedCheck_373_ == 0)
{
v___x_368_ = v___x_365_;
v_isShared_369_ = v_isSharedCheck_373_;
goto v_resetjp_367_;
}
else
{
lean_inc(v_a_366_);
lean_dec(v___x_365_);
v___x_368_ = lean_box(0);
v_isShared_369_ = v_isSharedCheck_373_;
goto v_resetjp_367_;
}
v_resetjp_367_:
{
lean_object* v___x_371_; 
if (v_isShared_369_ == 0)
{
v___x_371_ = v___x_368_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v_a_366_);
v___x_371_ = v_reuseFailAlloc_372_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
return v___x_371_;
}
}
}
else
{
lean_object* v_a_374_; lean_object* v___x_376_; uint8_t v_isShared_377_; uint8_t v_isSharedCheck_381_; 
v_a_374_ = lean_ctor_get(v___x_365_, 0);
v_isSharedCheck_381_ = !lean_is_exclusive(v___x_365_);
if (v_isSharedCheck_381_ == 0)
{
v___x_376_ = v___x_365_;
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
else
{
lean_inc(v_a_374_);
lean_dec(v___x_365_);
v___x_376_ = lean_box(0);
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
v_resetjp_375_:
{
lean_object* v___x_379_; 
if (v_isShared_377_ == 0)
{
v___x_379_ = v___x_376_;
goto v_reusejp_378_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_a_374_);
v___x_379_ = v_reuseFailAlloc_380_;
goto v_reusejp_378_;
}
v_reusejp_378_:
{
return v___x_379_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg___boxed(lean_object* v_name_382_, lean_object* v_bi_383_, lean_object* v_type_384_, lean_object* v_k_385_, lean_object* v_kind_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_){
_start:
{
uint8_t v_bi_boxed_392_; uint8_t v_kind_boxed_393_; lean_object* v_res_394_; 
v_bi_boxed_392_ = lean_unbox(v_bi_383_);
v_kind_boxed_393_ = lean_unbox(v_kind_386_);
v_res_394_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg(v_name_382_, v_bi_boxed_392_, v_type_384_, v_k_385_, v_kind_boxed_393_, v___y_387_, v___y_388_, v___y_389_, v___y_390_);
lean_dec(v___y_390_);
lean_dec_ref(v___y_389_);
lean_dec(v___y_388_);
lean_dec_ref(v___y_387_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0(lean_object* v_00_u03b1_395_, lean_object* v_name_396_, uint8_t v_bi_397_, lean_object* v_type_398_, lean_object* v_k_399_, uint8_t v_kind_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg(v_name_396_, v_bi_397_, v_type_398_, v_k_399_, v_kind_400_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___boxed(lean_object* v_00_u03b1_407_, lean_object* v_name_408_, lean_object* v_bi_409_, lean_object* v_type_410_, lean_object* v_k_411_, lean_object* v_kind_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_){
_start:
{
uint8_t v_bi_boxed_418_; uint8_t v_kind_boxed_419_; lean_object* v_res_420_; 
v_bi_boxed_418_ = lean_unbox(v_bi_409_);
v_kind_boxed_419_ = lean_unbox(v_kind_412_);
v_res_420_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0(v_00_u03b1_407_, v_name_408_, v_bi_boxed_418_, v_type_410_, v_k_411_, v_kind_boxed_419_, v___y_413_, v___y_414_, v___y_415_, v___y_416_);
lean_dec(v___y_416_);
lean_dec_ref(v___y_415_);
lean_dec(v___y_414_);
lean_dec_ref(v___y_413_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg___lam__0(lean_object* v_k_421_, lean_object* v_b_422_, lean_object* v_c_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_){
_start:
{
lean_object* v___x_429_; 
lean_inc(v___y_427_);
lean_inc_ref(v___y_426_);
lean_inc(v___y_425_);
lean_inc_ref(v___y_424_);
v___x_429_ = lean_apply_7(v_k_421_, v_b_422_, v_c_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_, lean_box(0));
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg___lam__0___boxed(lean_object* v_k_430_, lean_object* v_b_431_, lean_object* v_c_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_){
_start:
{
lean_object* v_res_438_; 
v_res_438_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg___lam__0(v_k_430_, v_b_431_, v_c_432_, v___y_433_, v___y_434_, v___y_435_, v___y_436_);
lean_dec(v___y_436_);
lean_dec_ref(v___y_435_);
lean_dec(v___y_434_);
lean_dec_ref(v___y_433_);
return v_res_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg(lean_object* v_type_439_, lean_object* v_maxFVars_x3f_440_, lean_object* v_k_441_, uint8_t v_cleanupAnnotations_442_, uint8_t v_whnfType_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_){
_start:
{
lean_object* v___f_449_; lean_object* v___x_450_; 
v___f_449_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_449_, 0, v_k_441_);
v___x_450_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_box(0), v_type_439_, v_maxFVars_x3f_440_, v___f_449_, v_cleanupAnnotations_442_, v_whnfType_443_, v___y_444_, v___y_445_, v___y_446_, v___y_447_);
if (lean_obj_tag(v___x_450_) == 0)
{
lean_object* v_a_451_; lean_object* v___x_453_; uint8_t v_isShared_454_; uint8_t v_isSharedCheck_458_; 
v_a_451_ = lean_ctor_get(v___x_450_, 0);
v_isSharedCheck_458_ = !lean_is_exclusive(v___x_450_);
if (v_isSharedCheck_458_ == 0)
{
v___x_453_ = v___x_450_;
v_isShared_454_ = v_isSharedCheck_458_;
goto v_resetjp_452_;
}
else
{
lean_inc(v_a_451_);
lean_dec(v___x_450_);
v___x_453_ = lean_box(0);
v_isShared_454_ = v_isSharedCheck_458_;
goto v_resetjp_452_;
}
v_resetjp_452_:
{
lean_object* v___x_456_; 
if (v_isShared_454_ == 0)
{
v___x_456_ = v___x_453_;
goto v_reusejp_455_;
}
else
{
lean_object* v_reuseFailAlloc_457_; 
v_reuseFailAlloc_457_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_457_, 0, v_a_451_);
v___x_456_ = v_reuseFailAlloc_457_;
goto v_reusejp_455_;
}
v_reusejp_455_:
{
return v___x_456_;
}
}
}
else
{
lean_object* v_a_459_; lean_object* v___x_461_; uint8_t v_isShared_462_; uint8_t v_isSharedCheck_466_; 
v_a_459_ = lean_ctor_get(v___x_450_, 0);
v_isSharedCheck_466_ = !lean_is_exclusive(v___x_450_);
if (v_isSharedCheck_466_ == 0)
{
v___x_461_ = v___x_450_;
v_isShared_462_ = v_isSharedCheck_466_;
goto v_resetjp_460_;
}
else
{
lean_inc(v_a_459_);
lean_dec(v___x_450_);
v___x_461_ = lean_box(0);
v_isShared_462_ = v_isSharedCheck_466_;
goto v_resetjp_460_;
}
v_resetjp_460_:
{
lean_object* v___x_464_; 
if (v_isShared_462_ == 0)
{
v___x_464_ = v___x_461_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_465_; 
v_reuseFailAlloc_465_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_465_, 0, v_a_459_);
v___x_464_ = v_reuseFailAlloc_465_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
return v___x_464_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg___boxed(lean_object* v_type_467_, lean_object* v_maxFVars_x3f_468_, lean_object* v_k_469_, lean_object* v_cleanupAnnotations_470_, lean_object* v_whnfType_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_477_; uint8_t v_whnfType_boxed_478_; lean_object* v_res_479_; 
v_cleanupAnnotations_boxed_477_ = lean_unbox(v_cleanupAnnotations_470_);
v_whnfType_boxed_478_ = lean_unbox(v_whnfType_471_);
v_res_479_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg(v_type_467_, v_maxFVars_x3f_468_, v_k_469_, v_cleanupAnnotations_boxed_477_, v_whnfType_boxed_478_, v___y_472_, v___y_473_, v___y_474_, v___y_475_);
lean_dec(v___y_475_);
lean_dec_ref(v___y_474_);
lean_dec(v___y_473_);
lean_dec_ref(v___y_472_);
return v_res_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2(lean_object* v_00_u03b1_480_, lean_object* v_type_481_, lean_object* v_maxFVars_x3f_482_, lean_object* v_k_483_, uint8_t v_cleanupAnnotations_484_, uint8_t v_whnfType_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_){
_start:
{
lean_object* v___x_491_; 
v___x_491_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg(v_type_481_, v_maxFVars_x3f_482_, v_k_483_, v_cleanupAnnotations_484_, v_whnfType_485_, v___y_486_, v___y_487_, v___y_488_, v___y_489_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___boxed(lean_object* v_00_u03b1_492_, lean_object* v_type_493_, lean_object* v_maxFVars_x3f_494_, lean_object* v_k_495_, lean_object* v_cleanupAnnotations_496_, lean_object* v_whnfType_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_503_; uint8_t v_whnfType_boxed_504_; lean_object* v_res_505_; 
v_cleanupAnnotations_boxed_503_ = lean_unbox(v_cleanupAnnotations_496_);
v_whnfType_boxed_504_ = lean_unbox(v_whnfType_497_);
v_res_505_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2(v_00_u03b1_492_, v_type_493_, v_maxFVars_x3f_494_, v_k_495_, v_cleanupAnnotations_boxed_503_, v_whnfType_boxed_504_, v___y_498_, v___y_499_, v___y_500_, v___y_501_);
lean_dec(v___y_501_);
lean_dec_ref(v___y_500_);
lean_dec(v___y_499_);
lean_dec_ref(v___y_498_);
return v_res_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__0(lean_object* v_n_506_, lean_object* v_f_507_, uint8_t v___x_508_, lean_object* v_xProd_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_){
_start:
{
lean_object* v___x_515_; 
lean_inc_ref(v_xProd_509_);
v___x_515_ = lp_mathlib_Mathlib_Meta_FunProp_mkProdSplitElem(v_xProd_509_, v_n_506_, v___y_510_, v___y_511_, v___y_512_, v___y_513_);
if (lean_obj_tag(v___x_515_) == 0)
{
lean_object* v_a_516_; lean_object* v___x_517_; 
v_a_516_ = lean_ctor_get(v___x_515_, 0);
lean_inc(v_a_516_);
lean_dec_ref_known(v___x_515_, 1);
v___x_517_ = l_Lean_Meta_mkAppM_x27(v_f_507_, v_a_516_, v___y_510_, v___y_511_, v___y_512_, v___y_513_);
if (lean_obj_tag(v___x_517_) == 0)
{
lean_object* v_a_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; uint8_t v___x_523_; uint8_t v___x_524_; lean_object* v___x_525_; 
v_a_518_ = lean_ctor_get(v___x_517_, 0);
lean_inc(v_a_518_);
lean_dec_ref_known(v___x_517_, 1);
v___x_519_ = lean_unsigned_to_nat(1u);
v___x_520_ = lean_mk_empty_array_with_capacity(v___x_519_);
v___x_521_ = lean_array_push(v___x_520_, v_xProd_509_);
v___x_522_ = l_Lean_Expr_headBeta(v_a_518_);
v___x_523_ = 1;
v___x_524_ = 1;
v___x_525_ = l_Lean_Meta_mkLambdaFVars(v___x_521_, v___x_522_, v___x_508_, v___x_523_, v___x_508_, v___x_523_, v___x_524_, v___y_510_, v___y_511_, v___y_512_, v___y_513_);
lean_dec_ref(v___x_521_);
return v___x_525_;
}
else
{
lean_dec_ref(v_xProd_509_);
return v___x_517_;
}
}
else
{
lean_object* v_a_526_; lean_object* v___x_528_; uint8_t v_isShared_529_; uint8_t v_isSharedCheck_533_; 
lean_dec_ref(v_xProd_509_);
lean_dec_ref(v_f_507_);
v_a_526_ = lean_ctor_get(v___x_515_, 0);
v_isSharedCheck_533_ = !lean_is_exclusive(v___x_515_);
if (v_isSharedCheck_533_ == 0)
{
v___x_528_ = v___x_515_;
v_isShared_529_ = v_isSharedCheck_533_;
goto v_resetjp_527_;
}
else
{
lean_inc(v_a_526_);
lean_dec(v___x_515_);
v___x_528_ = lean_box(0);
v_isShared_529_ = v_isSharedCheck_533_;
goto v_resetjp_527_;
}
v_resetjp_527_:
{
lean_object* v___x_531_; 
if (v_isShared_529_ == 0)
{
v___x_531_ = v___x_528_;
goto v_reusejp_530_;
}
else
{
lean_object* v_reuseFailAlloc_532_; 
v_reuseFailAlloc_532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_532_, 0, v_a_526_);
v___x_531_ = v_reuseFailAlloc_532_;
goto v_reusejp_530_;
}
v_reusejp_530_:
{
return v___x_531_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__0___boxed(lean_object* v_n_534_, lean_object* v_f_535_, lean_object* v___x_536_, lean_object* v_xProd_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_){
_start:
{
uint8_t v___x_2338__boxed_543_; lean_object* v_res_544_; 
v___x_2338__boxed_543_ = lean_unbox(v___x_536_);
v_res_544_ = lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__0(v_n_534_, v_f_535_, v___x_2338__boxed_543_, v_xProd_537_, v___y_538_, v___y_539_, v___y_540_, v___y_541_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
lean_dec(v___y_539_);
lean_dec_ref(v___y_538_);
return v_res_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1___redArg(lean_object* v_as_545_, size_t v_i_546_, size_t v_stop_547_, lean_object* v_b_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_){
_start:
{
uint8_t v___x_553_; 
v___x_553_ = lean_usize_dec_eq(v_i_546_, v_stop_547_);
if (v___x_553_ == 0)
{
lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; 
v___x_554_ = lean_array_uget_borrowed(v_as_545_, v_i_546_);
v___x_555_ = l_Lean_Expr_fvarId_x21(v___x_554_);
v___x_556_ = l_Lean_FVarId_getUserName___redArg(v___x_555_, v___y_549_, v___y_550_, v___y_551_);
if (lean_obj_tag(v___x_556_) == 0)
{
lean_object* v_a_557_; lean_object* v___x_558_; uint8_t v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; size_t v___x_562_; size_t v___x_563_; 
v_a_557_ = lean_ctor_get(v___x_556_, 0);
lean_inc(v_a_557_);
lean_dec_ref_known(v___x_556_, 1);
v___x_558_ = l_Lean_Name_eraseMacroScopes(v_a_557_);
lean_dec(v_a_557_);
v___x_559_ = 1;
v___x_560_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_558_, v___x_559_);
v___x_561_ = lean_string_append(v_b_548_, v___x_560_);
lean_dec_ref(v___x_560_);
v___x_562_ = ((size_t)1ULL);
v___x_563_ = lean_usize_add(v_i_546_, v___x_562_);
v_i_546_ = v___x_563_;
v_b_548_ = v___x_561_;
goto _start;
}
else
{
lean_object* v_a_565_; lean_object* v___x_567_; uint8_t v_isShared_568_; uint8_t v_isSharedCheck_572_; 
lean_dec_ref(v_b_548_);
v_a_565_ = lean_ctor_get(v___x_556_, 0);
v_isSharedCheck_572_ = !lean_is_exclusive(v___x_556_);
if (v_isSharedCheck_572_ == 0)
{
v___x_567_ = v___x_556_;
v_isShared_568_ = v_isSharedCheck_572_;
goto v_resetjp_566_;
}
else
{
lean_inc(v_a_565_);
lean_dec(v___x_556_);
v___x_567_ = lean_box(0);
v_isShared_568_ = v_isSharedCheck_572_;
goto v_resetjp_566_;
}
v_resetjp_566_:
{
lean_object* v___x_570_; 
if (v_isShared_568_ == 0)
{
v___x_570_ = v___x_567_;
goto v_reusejp_569_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v_a_565_);
v___x_570_ = v_reuseFailAlloc_571_;
goto v_reusejp_569_;
}
v_reusejp_569_:
{
return v___x_570_;
}
}
}
}
else
{
lean_object* v___x_573_; 
v___x_573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_573_, 0, v_b_548_);
return v___x_573_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1___redArg___boxed(lean_object* v_as_574_, lean_object* v_i_575_, lean_object* v_stop_576_, lean_object* v_b_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_){
_start:
{
size_t v_i_boxed_582_; size_t v_stop_boxed_583_; lean_object* v_res_584_; 
v_i_boxed_582_ = lean_unbox_usize(v_i_575_);
lean_dec(v_i_575_);
v_stop_boxed_583_ = lean_unbox_usize(v_stop_576_);
lean_dec(v_stop_576_);
v_res_584_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1___redArg(v_as_574_, v_i_boxed_582_, v_stop_boxed_583_, v_b_577_, v___y_578_, v___y_579_, v___y_580_);
lean_dec(v___y_580_);
lean_dec_ref(v___y_579_);
lean_dec_ref(v___y_578_);
lean_dec_ref(v_as_574_);
return v_res_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__1(lean_object* v___f_586_, lean_object* v_xs_587_, lean_object* v_x_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_){
_start:
{
lean_object* v_a_595_; lean_object* v___y_606_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; uint8_t v___x_619_; 
v___x_616_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__1___closed__0));
v___x_617_ = lean_unsigned_to_nat(0u);
v___x_618_ = lean_array_get_size(v_xs_587_);
v___x_619_ = lean_nat_dec_lt(v___x_617_, v___x_618_);
if (v___x_619_ == 0)
{
v_a_595_ = v___x_616_;
goto v___jp_594_;
}
else
{
uint8_t v___x_620_; 
v___x_620_ = lean_nat_dec_le(v___x_618_, v___x_618_);
if (v___x_620_ == 0)
{
if (v___x_619_ == 0)
{
v_a_595_ = v___x_616_;
goto v___jp_594_;
}
else
{
size_t v___x_621_; size_t v___x_622_; lean_object* v___x_623_; 
v___x_621_ = ((size_t)0ULL);
v___x_622_ = lean_usize_of_nat(v___x_618_);
v___x_623_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1___redArg(v_xs_587_, v___x_621_, v___x_622_, v___x_616_, v___y_589_, v___y_591_, v___y_592_);
v___y_606_ = v___x_623_;
goto v___jp_605_;
}
}
else
{
size_t v___x_624_; size_t v___x_625_; lean_object* v___x_626_; 
v___x_624_ = ((size_t)0ULL);
v___x_625_ = lean_usize_of_nat(v___x_618_);
v___x_626_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1___redArg(v_xs_587_, v___x_624_, v___x_625_, v___x_616_, v___y_589_, v___y_591_, v___y_592_);
v___y_606_ = v___x_626_;
goto v___jp_605_;
}
}
v___jp_594_:
{
lean_object* v___x_596_; 
v___x_596_ = lp_mathlib_Mathlib_Meta_FunProp_mkProdElem(v_xs_587_, v___y_589_, v___y_590_, v___y_591_, v___y_592_);
if (lean_obj_tag(v___x_596_) == 0)
{
lean_object* v_a_597_; lean_object* v___x_598_; 
v_a_597_ = lean_ctor_get(v___x_596_, 0);
lean_inc(v_a_597_);
lean_dec_ref_known(v___x_596_, 1);
lean_inc(v___y_592_);
lean_inc_ref(v___y_591_);
lean_inc(v___y_590_);
lean_inc_ref(v___y_589_);
v___x_598_ = lean_infer_type(v_a_597_, v___y_589_, v___y_590_, v___y_591_, v___y_592_);
if (lean_obj_tag(v___x_598_) == 0)
{
lean_object* v_a_599_; lean_object* v___x_600_; lean_object* v___x_601_; uint8_t v___x_602_; uint8_t v___x_603_; lean_object* v___x_604_; 
v_a_599_ = lean_ctor_get(v___x_598_, 0);
lean_inc(v_a_599_);
lean_dec_ref_known(v___x_598_, 1);
v___x_600_ = lean_box(0);
v___x_601_ = l_Lean_Name_str___override(v___x_600_, v_a_595_);
v___x_602_ = 0;
v___x_603_ = 0;
v___x_604_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__0___redArg(v___x_601_, v___x_602_, v_a_599_, v___f_586_, v___x_603_, v___y_589_, v___y_590_, v___y_591_, v___y_592_);
return v___x_604_;
}
else
{
lean_dec_ref(v_a_595_);
lean_dec_ref(v___f_586_);
return v___x_598_;
}
}
else
{
lean_dec_ref(v_a_595_);
lean_dec_ref(v___f_586_);
return v___x_596_;
}
}
v___jp_605_:
{
if (lean_obj_tag(v___y_606_) == 0)
{
lean_object* v_a_607_; 
v_a_607_ = lean_ctor_get(v___y_606_, 0);
lean_inc(v_a_607_);
lean_dec_ref_known(v___y_606_, 1);
v_a_595_ = v_a_607_;
goto v___jp_594_;
}
else
{
lean_object* v_a_608_; lean_object* v___x_610_; uint8_t v_isShared_611_; uint8_t v_isSharedCheck_615_; 
lean_dec_ref(v_xs_587_);
lean_dec_ref(v___f_586_);
v_a_608_ = lean_ctor_get(v___y_606_, 0);
v_isSharedCheck_615_ = !lean_is_exclusive(v___y_606_);
if (v_isSharedCheck_615_ == 0)
{
v___x_610_ = v___y_606_;
v_isShared_611_ = v_isSharedCheck_615_;
goto v_resetjp_609_;
}
else
{
lean_inc(v_a_608_);
lean_dec(v___y_606_);
v___x_610_ = lean_box(0);
v_isShared_611_ = v_isSharedCheck_615_;
goto v_resetjp_609_;
}
v_resetjp_609_:
{
lean_object* v___x_613_; 
if (v_isShared_611_ == 0)
{
v___x_613_ = v___x_610_;
goto v_reusejp_612_;
}
else
{
lean_object* v_reuseFailAlloc_614_; 
v_reuseFailAlloc_614_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_614_, 0, v_a_608_);
v___x_613_ = v_reuseFailAlloc_614_;
goto v_reusejp_612_;
}
v_reusejp_612_:
{
return v___x_613_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__1___boxed(lean_object* v___f_627_, lean_object* v_xs_628_, lean_object* v_x_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_){
_start:
{
lean_object* v_res_635_; 
v_res_635_ = lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__1(v___f_627_, v_xs_628_, v_x_629_, v___y_630_, v___y_631_, v___y_632_, v___y_633_);
lean_dec(v___y_633_);
lean_dec_ref(v___y_632_);
lean_dec(v___y_631_);
lean_dec_ref(v___y_630_);
lean_dec_ref(v_x_629_);
return v_res_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun(lean_object* v_n_636_, lean_object* v_f_637_, lean_object* v_a_638_, lean_object* v_a_639_, lean_object* v_a_640_, lean_object* v_a_641_){
_start:
{
lean_object* v___x_643_; uint8_t v___x_644_; 
v___x_643_ = lean_unsigned_to_nat(1u);
v___x_644_ = lean_nat_dec_le(v_n_636_, v___x_643_);
if (v___x_644_ == 0)
{
lean_object* v___x_645_; 
lean_inc(v_a_641_);
lean_inc_ref(v_a_640_);
lean_inc(v_a_639_);
lean_inc_ref(v_a_638_);
lean_inc_ref(v_f_637_);
v___x_645_ = lean_infer_type(v_f_637_, v_a_638_, v_a_639_, v_a_640_, v_a_641_);
if (lean_obj_tag(v___x_645_) == 0)
{
lean_object* v_a_646_; lean_object* v___x_648_; uint8_t v_isShared_649_; uint8_t v_isSharedCheck_657_; 
v_a_646_ = lean_ctor_get(v___x_645_, 0);
v_isSharedCheck_657_ = !lean_is_exclusive(v___x_645_);
if (v_isSharedCheck_657_ == 0)
{
v___x_648_ = v___x_645_;
v_isShared_649_ = v_isSharedCheck_657_;
goto v_resetjp_647_;
}
else
{
lean_inc(v_a_646_);
lean_dec(v___x_645_);
v___x_648_ = lean_box(0);
v_isShared_649_ = v_isSharedCheck_657_;
goto v_resetjp_647_;
}
v_resetjp_647_:
{
lean_object* v___x_650_; lean_object* v___f_651_; lean_object* v___f_652_; lean_object* v___x_654_; 
v___x_650_ = lean_box(v___x_644_);
lean_inc(v_n_636_);
v___f_651_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__0___boxed), 9, 3);
lean_closure_set(v___f_651_, 0, v_n_636_);
lean_closure_set(v___f_651_, 1, v_f_637_);
lean_closure_set(v___f_651_, 2, v___x_650_);
v___f_652_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___lam__1___boxed), 8, 1);
lean_closure_set(v___f_652_, 0, v___f_651_);
if (v_isShared_649_ == 0)
{
lean_ctor_set_tag(v___x_648_, 1);
lean_ctor_set(v___x_648_, 0, v_n_636_);
v___x_654_ = v___x_648_;
goto v_reusejp_653_;
}
else
{
lean_object* v_reuseFailAlloc_656_; 
v_reuseFailAlloc_656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_656_, 0, v_n_636_);
v___x_654_ = v_reuseFailAlloc_656_;
goto v_reusejp_653_;
}
v_reusejp_653_:
{
lean_object* v___x_655_; 
v___x_655_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg(v_a_646_, v___x_654_, v___f_652_, v___x_644_, v___x_644_, v_a_638_, v_a_639_, v_a_640_, v_a_641_);
return v___x_655_;
}
}
}
else
{
lean_dec_ref(v_f_637_);
lean_dec(v_n_636_);
return v___x_645_;
}
}
else
{
lean_object* v___x_658_; 
lean_dec(v_n_636_);
v___x_658_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_658_, 0, v_f_637_);
return v___x_658_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun___boxed(lean_object* v_n_659_, lean_object* v_f_660_, lean_object* v_a_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_){
_start:
{
lean_object* v_res_666_; 
v_res_666_ = lp_mathlib_Mathlib_Meta_FunProp_mkUncurryFun(v_n_659_, v_f_660_, v_a_661_, v_a_662_, v_a_663_, v_a_664_);
lean_dec(v_a_664_);
lean_dec_ref(v_a_663_);
lean_dec(v_a_662_);
lean_dec_ref(v_a_661_);
return v_res_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1(lean_object* v_as_667_, size_t v_i_668_, size_t v_stop_669_, lean_object* v_b_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_){
_start:
{
lean_object* v___x_676_; 
v___x_676_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1___redArg(v_as_667_, v_i_668_, v_stop_669_, v_b_670_, v___y_671_, v___y_673_, v___y_674_);
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1___boxed(lean_object* v_as_677_, lean_object* v_i_678_, lean_object* v_stop_679_, lean_object* v_b_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_){
_start:
{
size_t v_i_boxed_686_; size_t v_stop_boxed_687_; lean_object* v_res_688_; 
v_i_boxed_686_ = lean_unbox_usize(v_i_678_);
lean_dec(v_i_678_);
v_stop_boxed_687_ = lean_unbox_usize(v_stop_679_);
lean_dec(v_stop_679_);
v_res_688_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__1(v_as_677_, v_i_boxed_686_, v_stop_boxed_687_, v_b_680_, v___y_681_, v___y_682_, v___y_683_, v___y_684_);
lean_dec(v___y_684_);
lean_dec_ref(v___y_683_);
lean_dec(v___y_682_);
lean_dec_ref(v___y_681_);
lean_dec_ref(v_as_677_);
return v_res_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___lam__0(lean_object* v_f_689_, uint8_t v___x_690_, uint8_t v___x_691_, lean_object* v_xs_692_, lean_object* v_x_693_, lean_object* v___y_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_){
_start:
{
lean_object* v___x_699_; uint8_t v___x_700_; lean_object* v___x_701_; 
v___x_699_ = l_Lean_mkAppN(v_f_689_, v_xs_692_);
v___x_700_ = 1;
v___x_701_ = l_Lean_Meta_mkLambdaFVars(v_xs_692_, v___x_699_, v___x_690_, v___x_691_, v___x_690_, v___x_691_, v___x_700_, v___y_694_, v___y_695_, v___y_696_, v___y_697_);
return v___x_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___lam__0___boxed(lean_object* v_f_702_, lean_object* v___x_703_, lean_object* v___x_704_, lean_object* v_xs_705_, lean_object* v_x_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
uint8_t v___x_709__boxed_712_; uint8_t v___x_710__boxed_713_; lean_object* v_res_714_; 
v___x_709__boxed_712_ = lean_unbox(v___x_703_);
v___x_710__boxed_713_ = lean_unbox(v___x_704_);
v_res_714_ = lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___lam__0(v_f_702_, v___x_709__boxed_712_, v___x_710__boxed_713_, v_xs_705_, v_x_706_, v___y_707_, v___y_708_, v___y_709_, v___y_710_);
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec(v___y_708_);
lean_dec_ref(v___y_707_);
lean_dec_ref(v_x_706_);
lean_dec_ref(v_xs_705_);
return v_res_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_etaExpand1(lean_object* v_f_717_, lean_object* v_a_718_, lean_object* v_a_719_, lean_object* v_a_720_, lean_object* v_a_721_){
_start:
{
lean_object* v_f_723_; uint8_t v___x_724_; 
v_f_723_ = l_Lean_Expr_eta(v_f_717_);
v___x_724_ = l_Lean_Expr_isLambda(v_f_723_);
if (v___x_724_ == 0)
{
lean_object* v_keyedConfig_725_; uint8_t v_trackZetaDelta_726_; lean_object* v_zetaDeltaSet_727_; lean_object* v_lctx_728_; lean_object* v_localInstances_729_; lean_object* v_defEqCtx_x3f_730_; lean_object* v_synthPendingDepth_731_; lean_object* v_customCanUnfoldPredicate_x3f_732_; uint8_t v_univApprox_733_; uint8_t v_inTypeClassResolution_734_; uint8_t v_cacheInferType_735_; uint8_t v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
v_keyedConfig_725_ = lean_ctor_get(v_a_718_, 0);
v_trackZetaDelta_726_ = lean_ctor_get_uint8(v_a_718_, sizeof(void*)*7);
v_zetaDeltaSet_727_ = lean_ctor_get(v_a_718_, 1);
v_lctx_728_ = lean_ctor_get(v_a_718_, 2);
v_localInstances_729_ = lean_ctor_get(v_a_718_, 3);
v_defEqCtx_x3f_730_ = lean_ctor_get(v_a_718_, 4);
v_synthPendingDepth_731_ = lean_ctor_get(v_a_718_, 5);
v_customCanUnfoldPredicate_x3f_732_ = lean_ctor_get(v_a_718_, 6);
v_univApprox_733_ = lean_ctor_get_uint8(v_a_718_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_734_ = lean_ctor_get_uint8(v_a_718_, sizeof(void*)*7 + 2);
v_cacheInferType_735_ = lean_ctor_get_uint8(v_a_718_, sizeof(void*)*7 + 3);
v___x_736_ = 1;
lean_inc_ref(v_keyedConfig_725_);
v___x_737_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_736_, v_keyedConfig_725_);
lean_inc(v_customCanUnfoldPredicate_x3f_732_);
lean_inc(v_synthPendingDepth_731_);
lean_inc(v_defEqCtx_x3f_730_);
lean_inc_ref(v_localInstances_729_);
lean_inc_ref(v_lctx_728_);
lean_inc(v_zetaDeltaSet_727_);
v___x_738_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_738_, 0, v___x_737_);
lean_ctor_set(v___x_738_, 1, v_zetaDeltaSet_727_);
lean_ctor_set(v___x_738_, 2, v_lctx_728_);
lean_ctor_set(v___x_738_, 3, v_localInstances_729_);
lean_ctor_set(v___x_738_, 4, v_defEqCtx_x3f_730_);
lean_ctor_set(v___x_738_, 5, v_synthPendingDepth_731_);
lean_ctor_set(v___x_738_, 6, v_customCanUnfoldPredicate_x3f_732_);
lean_ctor_set_uint8(v___x_738_, sizeof(void*)*7, v_trackZetaDelta_726_);
lean_ctor_set_uint8(v___x_738_, sizeof(void*)*7 + 1, v_univApprox_733_);
lean_ctor_set_uint8(v___x_738_, sizeof(void*)*7 + 2, v_inTypeClassResolution_734_);
lean_ctor_set_uint8(v___x_738_, sizeof(void*)*7 + 3, v_cacheInferType_735_);
lean_inc(v_a_721_);
lean_inc_ref(v_a_720_);
lean_inc(v_a_719_);
lean_inc_ref(v___x_738_);
lean_inc_ref(v_f_723_);
v___x_739_ = lean_infer_type(v_f_723_, v___x_738_, v_a_719_, v_a_720_, v_a_721_);
if (lean_obj_tag(v___x_739_) == 0)
{
lean_object* v_a_740_; uint8_t v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___f_744_; lean_object* v___x_745_; lean_object* v___x_746_; 
v_a_740_ = lean_ctor_get(v___x_739_, 0);
lean_inc(v_a_740_);
lean_dec_ref_known(v___x_739_, 1);
v___x_741_ = 1;
v___x_742_ = lean_box(v___x_724_);
v___x_743_ = lean_box(v___x_741_);
v___f_744_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___lam__0___boxed), 10, 3);
lean_closure_set(v___f_744_, 0, v_f_723_);
lean_closure_set(v___f_744_, 1, v___x_742_);
lean_closure_set(v___f_744_, 2, v___x_743_);
v___x_745_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___closed__0));
v___x_746_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00Mathlib_Meta_FunProp_mkUncurryFun_spec__2___redArg(v_a_740_, v___x_745_, v___f_744_, v___x_724_, v___x_724_, v___x_738_, v_a_719_, v_a_720_, v_a_721_);
lean_dec_ref_known(v___x_738_, 7);
return v___x_746_;
}
else
{
lean_dec_ref_known(v___x_738_, 7);
lean_dec_ref(v_f_723_);
return v___x_739_;
}
}
else
{
lean_object* v___x_747_; 
v___x_747_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_747_, 0, v_f_723_);
return v___x_747_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_etaExpand1___boxed(lean_object* v_f_748_, lean_object* v_a_749_, lean_object* v_a_750_, lean_object* v_a_751_, lean_object* v_a_752_, lean_object* v_a_753_){
_start:
{
lean_object* v_res_754_; 
v_res_754_ = lp_mathlib_Mathlib_Meta_FunProp_etaExpand1(v_f_748_, v_a_749_, v_a_750_, v_a_751_, v_a_752_);
lean_dec(v_a_752_);
lean_dec_ref(v_a_751_);
lean_dec(v_a_750_);
lean_dec_ref(v_a_749_);
return v_res_754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_ToBatteries_0__Mathlib_Meta_FunProp_betaThroughLetAux(lean_object* v_f_755_, lean_object* v_args_756_){
_start:
{
if (lean_obj_tag(v_args_756_) == 0)
{
return v_f_755_;
}
else
{
switch(lean_obj_tag(v_f_755_))
{
case 6:
{
lean_object* v_head_757_; lean_object* v_tail_758_; lean_object* v_body_759_; lean_object* v___x_760_; 
v_head_757_ = lean_ctor_get(v_args_756_, 0);
lean_inc(v_head_757_);
v_tail_758_ = lean_ctor_get(v_args_756_, 1);
lean_inc(v_tail_758_);
lean_dec_ref_known(v_args_756_, 2);
v_body_759_ = lean_ctor_get(v_f_755_, 2);
lean_inc_ref(v_body_759_);
lean_dec_ref_known(v_f_755_, 3);
v___x_760_ = lean_expr_instantiate1(v_body_759_, v_head_757_);
lean_dec(v_head_757_);
lean_dec_ref(v_body_759_);
v_f_755_ = v___x_760_;
v_args_756_ = v_tail_758_;
goto _start;
}
case 8:
{
lean_object* v_declName_762_; lean_object* v_type_763_; lean_object* v_value_764_; lean_object* v_body_765_; uint8_t v_nondep_766_; lean_object* v___x_767_; lean_object* v___x_768_; 
v_declName_762_ = lean_ctor_get(v_f_755_, 0);
lean_inc(v_declName_762_);
v_type_763_ = lean_ctor_get(v_f_755_, 1);
lean_inc_ref(v_type_763_);
v_value_764_ = lean_ctor_get(v_f_755_, 2);
lean_inc_ref(v_value_764_);
v_body_765_ = lean_ctor_get(v_f_755_, 3);
lean_inc_ref(v_body_765_);
v_nondep_766_ = lean_ctor_get_uint8(v_f_755_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_f_755_, 4);
v___x_767_ = lp_mathlib___private_Mathlib_Tactic_FunProp_ToBatteries_0__Mathlib_Meta_FunProp_betaThroughLetAux(v_body_765_, v_args_756_);
v___x_768_ = l_Lean_Expr_letE___override(v_declName_762_, v_type_763_, v_value_764_, v___x_767_, v_nondep_766_);
return v___x_768_;
}
case 10:
{
lean_object* v_expr_769_; 
v_expr_769_ = lean_ctor_get(v_f_755_, 1);
lean_inc_ref(v_expr_769_);
lean_dec_ref_known(v_f_755_, 2);
v_f_755_ = v_expr_769_;
goto _start;
}
default: 
{
lean_object* v___x_771_; lean_object* v___x_772_; 
v___x_771_ = lean_array_mk(v_args_756_);
v___x_772_ = l_Lean_mkAppN(v_f_755_, v___x_771_);
lean_dec_ref(v___x_771_);
return v___x_772_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_ToBatteries_0__Mathlib_Meta_FunProp_betaThroughLetAux_match__1_splitter___redArg(lean_object* v_f_773_, lean_object* v_args_774_, lean_object* v_h__1_775_, lean_object* v_h__2_776_, lean_object* v_h__3_777_, lean_object* v_h__4_778_, lean_object* v_h__5_779_){
_start:
{
if (lean_obj_tag(v_args_774_) == 0)
{
lean_object* v___x_780_; 
lean_dec(v_h__5_779_);
lean_dec(v_h__4_778_);
lean_dec(v_h__3_777_);
lean_dec(v_h__2_776_);
v___x_780_ = lean_apply_1(v_h__1_775_, v_f_773_);
return v___x_780_;
}
else
{
lean_dec(v_h__1_775_);
switch(lean_obj_tag(v_f_773_))
{
case 6:
{
lean_object* v_head_781_; lean_object* v_tail_782_; lean_object* v_binderName_783_; lean_object* v_binderType_784_; lean_object* v_body_785_; uint8_t v_binderInfo_786_; lean_object* v___x_787_; lean_object* v___x_788_; 
lean_dec(v_h__5_779_);
lean_dec(v_h__4_778_);
lean_dec(v_h__3_777_);
v_head_781_ = lean_ctor_get(v_args_774_, 0);
lean_inc(v_head_781_);
v_tail_782_ = lean_ctor_get(v_args_774_, 1);
lean_inc(v_tail_782_);
lean_dec_ref_known(v_args_774_, 2);
v_binderName_783_ = lean_ctor_get(v_f_773_, 0);
lean_inc(v_binderName_783_);
v_binderType_784_ = lean_ctor_get(v_f_773_, 1);
lean_inc_ref(v_binderType_784_);
v_body_785_ = lean_ctor_get(v_f_773_, 2);
lean_inc_ref(v_body_785_);
v_binderInfo_786_ = lean_ctor_get_uint8(v_f_773_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_f_773_, 3);
v___x_787_ = lean_box(v_binderInfo_786_);
v___x_788_ = lean_apply_6(v_h__2_776_, v_binderName_783_, v_binderType_784_, v_body_785_, v___x_787_, v_head_781_, v_tail_782_);
return v___x_788_;
}
case 8:
{
lean_object* v_declName_789_; lean_object* v_type_790_; lean_object* v_value_791_; lean_object* v_body_792_; uint8_t v_nondep_793_; lean_object* v___x_794_; lean_object* v___x_795_; 
lean_dec(v_h__5_779_);
lean_dec(v_h__4_778_);
lean_dec(v_h__2_776_);
v_declName_789_ = lean_ctor_get(v_f_773_, 0);
lean_inc(v_declName_789_);
v_type_790_ = lean_ctor_get(v_f_773_, 1);
lean_inc_ref(v_type_790_);
v_value_791_ = lean_ctor_get(v_f_773_, 2);
lean_inc_ref(v_value_791_);
v_body_792_ = lean_ctor_get(v_f_773_, 3);
lean_inc_ref(v_body_792_);
v_nondep_793_ = lean_ctor_get_uint8(v_f_773_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_f_773_, 4);
v___x_794_ = lean_box(v_nondep_793_);
v___x_795_ = lean_apply_7(v_h__3_777_, v_declName_789_, v_type_790_, v_value_791_, v_body_792_, v___x_794_, v_args_774_, lean_box(0));
return v___x_795_;
}
case 10:
{
lean_object* v_data_796_; lean_object* v_expr_797_; lean_object* v___x_798_; 
lean_dec(v_h__5_779_);
lean_dec(v_h__3_777_);
lean_dec(v_h__2_776_);
v_data_796_ = lean_ctor_get(v_f_773_, 0);
lean_inc(v_data_796_);
v_expr_797_ = lean_ctor_get(v_f_773_, 1);
lean_inc_ref(v_expr_797_);
lean_dec_ref_known(v_f_773_, 2);
v___x_798_ = lean_apply_4(v_h__4_778_, v_data_796_, v_expr_797_, v_args_774_, lean_box(0));
return v___x_798_;
}
default: 
{
lean_object* v___x_799_; 
lean_dec(v_h__4_778_);
lean_dec(v_h__3_777_);
lean_dec(v_h__2_776_);
v___x_799_ = lean_apply_6(v_h__5_779_, v_f_773_, v_args_774_, lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_799_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_ToBatteries_0__Mathlib_Meta_FunProp_betaThroughLetAux_match__1_splitter(lean_object* v_motive_800_, lean_object* v_f_801_, lean_object* v_args_802_, lean_object* v_h__1_803_, lean_object* v_h__2_804_, lean_object* v_h__3_805_, lean_object* v_h__4_806_, lean_object* v_h__5_807_){
_start:
{
if (lean_obj_tag(v_args_802_) == 0)
{
lean_object* v___x_808_; 
lean_dec(v_h__5_807_);
lean_dec(v_h__4_806_);
lean_dec(v_h__3_805_);
lean_dec(v_h__2_804_);
v___x_808_ = lean_apply_1(v_h__1_803_, v_f_801_);
return v___x_808_;
}
else
{
lean_dec(v_h__1_803_);
switch(lean_obj_tag(v_f_801_))
{
case 6:
{
lean_object* v_head_809_; lean_object* v_tail_810_; lean_object* v_binderName_811_; lean_object* v_binderType_812_; lean_object* v_body_813_; uint8_t v_binderInfo_814_; lean_object* v___x_815_; lean_object* v___x_816_; 
lean_dec(v_h__5_807_);
lean_dec(v_h__4_806_);
lean_dec(v_h__3_805_);
v_head_809_ = lean_ctor_get(v_args_802_, 0);
lean_inc(v_head_809_);
v_tail_810_ = lean_ctor_get(v_args_802_, 1);
lean_inc(v_tail_810_);
lean_dec_ref_known(v_args_802_, 2);
v_binderName_811_ = lean_ctor_get(v_f_801_, 0);
lean_inc(v_binderName_811_);
v_binderType_812_ = lean_ctor_get(v_f_801_, 1);
lean_inc_ref(v_binderType_812_);
v_body_813_ = lean_ctor_get(v_f_801_, 2);
lean_inc_ref(v_body_813_);
v_binderInfo_814_ = lean_ctor_get_uint8(v_f_801_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_f_801_, 3);
v___x_815_ = lean_box(v_binderInfo_814_);
v___x_816_ = lean_apply_6(v_h__2_804_, v_binderName_811_, v_binderType_812_, v_body_813_, v___x_815_, v_head_809_, v_tail_810_);
return v___x_816_;
}
case 8:
{
lean_object* v_declName_817_; lean_object* v_type_818_; lean_object* v_value_819_; lean_object* v_body_820_; uint8_t v_nondep_821_; lean_object* v___x_822_; lean_object* v___x_823_; 
lean_dec(v_h__5_807_);
lean_dec(v_h__4_806_);
lean_dec(v_h__2_804_);
v_declName_817_ = lean_ctor_get(v_f_801_, 0);
lean_inc(v_declName_817_);
v_type_818_ = lean_ctor_get(v_f_801_, 1);
lean_inc_ref(v_type_818_);
v_value_819_ = lean_ctor_get(v_f_801_, 2);
lean_inc_ref(v_value_819_);
v_body_820_ = lean_ctor_get(v_f_801_, 3);
lean_inc_ref(v_body_820_);
v_nondep_821_ = lean_ctor_get_uint8(v_f_801_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_f_801_, 4);
v___x_822_ = lean_box(v_nondep_821_);
v___x_823_ = lean_apply_7(v_h__3_805_, v_declName_817_, v_type_818_, v_value_819_, v_body_820_, v___x_822_, v_args_802_, lean_box(0));
return v___x_823_;
}
case 10:
{
lean_object* v_data_824_; lean_object* v_expr_825_; lean_object* v___x_826_; 
lean_dec(v_h__5_807_);
lean_dec(v_h__3_805_);
lean_dec(v_h__2_804_);
v_data_824_ = lean_ctor_get(v_f_801_, 0);
lean_inc(v_data_824_);
v_expr_825_ = lean_ctor_get(v_f_801_, 1);
lean_inc_ref(v_expr_825_);
lean_dec_ref_known(v_f_801_, 2);
v___x_826_ = lean_apply_4(v_h__4_806_, v_data_824_, v_expr_825_, v_args_802_, lean_box(0));
return v___x_826_;
}
default: 
{
lean_object* v___x_827_; 
lean_dec(v_h__4_806_);
lean_dec(v_h__3_805_);
lean_dec(v_h__2_804_);
v___x_827_ = lean_apply_6(v_h__5_807_, v_f_801_, v_args_802_, lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_827_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_betaThroughLet(lean_object* v_f_828_, lean_object* v_args_829_){
_start:
{
lean_object* v___x_830_; lean_object* v___x_831_; 
v___x_830_ = lean_array_to_list(v_args_829_);
v___x_831_ = lp_mathlib___private_Mathlib_Tactic_FunProp_ToBatteries_0__Mathlib_Meta_FunProp_betaThroughLetAux(v_f_828_, v___x_830_);
return v___x_831_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet___closed__0(void){
_start:
{
lean_object* v___x_832_; lean_object* v_dummy_833_; 
v___x_832_ = lean_box(0);
v_dummy_833_ = l_Lean_Expr_sort___override(v___x_832_);
return v_dummy_833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet(lean_object* v_e_834_){
_start:
{
lean_object* v_f_835_; uint8_t v___x_836_; uint8_t v___x_837_; 
v_f_835_ = l_Lean_Expr_getAppFn(v_e_834_);
v___x_836_ = 1;
v___x_837_ = l_Lean_Expr_isHeadBetaTargetFn(v___x_836_, v_f_835_);
if (v___x_837_ == 0)
{
lean_dec_ref(v_f_835_);
return v_e_834_;
}
else
{
lean_object* v_dummy_838_; lean_object* v_nargs_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; 
v_dummy_838_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet___closed__0, &lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet___closed__0_once, _init_lp_mathlib_Mathlib_Meta_FunProp_headBetaThroughLet___closed__0);
v_nargs_839_ = l_Lean_Expr_getAppNumArgs(v_e_834_);
lean_inc(v_nargs_839_);
v___x_840_ = lean_mk_array(v_nargs_839_, v_dummy_838_);
v___x_841_ = lean_unsigned_to_nat(1u);
v___x_842_ = lean_nat_sub(v_nargs_839_, v___x_841_);
lean_dec(v_nargs_839_);
v___x_843_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_e_834_, v___x_840_, v___x_842_);
v___x_844_ = lp_mathlib_Mathlib_Meta_FunProp_betaThroughLet(v_f_835_, v___x_843_);
return v___x_844_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(uint8_t builtin) {
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
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FunProp_ToBatteries(builtin);
}
#ifdef __cplusplus
}
#endif
