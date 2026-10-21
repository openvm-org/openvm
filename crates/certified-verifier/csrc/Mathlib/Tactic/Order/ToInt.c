// Lean compiler output
// Module: Mathlib.Tactic.Order.ToInt
// Imports: public import Init public meta import Init public import Batteries.Data.List.Pairwise public import Batteries.Tactic.GeneralizeProofs public import Mathlib.Tactic.Order.CollectFacts public meta import Mathlib.Util.AtomM public meta import Mathlib.Util.Qq public meta import Std.Data.HashMap.AdditionalOperations
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_instInhabitedQuoted(lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Array_ofFn___redArg(lean_object*, lean_object*);
lean_object* l_Lean_RArray_ofArray___redArg(lean_object*);
lean_object* l_Lean_RArray_toExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_bvar___override(lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_mkNatLitQ(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Expr_betaRev(lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Fin"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__4_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "RArray"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "get"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__7_value),LEAN_SCALAR_PTR_LITERAL(94, 141, 178, 243, 105, 175, 161, 86)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__8_value),LEAN_SCALAR_PTR_LITERAL(180, 117, 254, 45, 193, 83, 35, 49)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "val"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__10_value),LEAN_SCALAR_PTR_LITERAL(165, 91, 87, 132, 175, 103, 206, 109)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__13;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "elim0"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__14_value),LEAN_SCALAR_PTR_LITERAL(238, 152, 75, 19, 63, 138, 165, 248)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5_spec__7___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Std.Data.DHashMap.Internal.AssocList.Basic"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__0 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__0_value;
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Std.DHashMap.Internal.AssocList.get!"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__1 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__1_value;
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "key is not present in hash table"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__2 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__2_value;
static lean_once_cell_t lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ToInt"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toInt"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__4 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value),LEAN_SCALAR_PTR_LITERAL(45, 4, 153, 104, 126, 202, 72, 180)}};
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5_value_aux_3),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__4_value),LEAN_SCALAR_PTR_LITERAL(249, 241, 2, 27, 107, 101, 45, 173)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__6 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__1_value),LEAN_SCALAR_PTR_LITERAL(62, 91, 162, 2, 110, 238, 123, 219)}};
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__6_value),LEAN_SCALAR_PTR_LITERAL(30, 240, 210, 97, 67, 170, 216, 80)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__7 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__8;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "of_decide_eq_true"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__9 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__9_value),LEAN_SCALAR_PTR_LITERAL(199, 143, 142, 104, 169, 34, 63, 25)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__10 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__11;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__12 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__12_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__13 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__12_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__13_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__14 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__15 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__16;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__17 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__17_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__18 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__18_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__19;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__20;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instLTNat"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__21 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__21_value),LEAN_SCALAR_PTR_LITERAL(141, 27, 201, 217, 48, 203, 85, 203)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__22 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__22_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__23;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__24;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "decLt"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__25 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__17_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__26_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__25_value),LEAN_SCALAR_PTR_LITERAL(70, 116, 195, 81, 41, 93, 3, 179)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__26 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__26_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__27;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__28 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__28_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "refl"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__29 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__29_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__28_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__30_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__29_value),LEAN_SCALAR_PTR_LITERAL(72, 6, 107, 181, 0, 125, 21, 187)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__30 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__30_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__31;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__33;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__34 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__34_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__34_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__35 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__35_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__36;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__37;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Decidable"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__38 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__38_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "decide"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__39 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__39_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__38_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__40_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__39_value),LEAN_SCALAR_PTR_LITERAL(16, 96, 65, 173, 152, 155, 4, 222)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__40 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__40_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__41;
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__28_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__2_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__6_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mpr"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__6_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__7_value),LEAN_SCALAR_PTR_LITERAL(14, 81, 9, 215, 230, 198, 87, 3)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toInt_eq_toInt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11_value_aux_2),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value),LEAN_SCALAR_PTR_LITERAL(45, 4, 153, 104, 126, 202, 72, 180)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11_value_aux_3),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__10_value),LEAN_SCALAR_PTR_LITERAL(1, 236, 228, 19, 120, 15, 86, 156)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Ne"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__12 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__12_value),LEAN_SCALAR_PTR_LITERAL(161, 247, 70, 70, 118, 145, 235, 92)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__13 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__14;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__15;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toInt_ne_toInt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__16 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17_value_aux_1),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17_value_aux_2),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value),LEAN_SCALAR_PTR_LITERAL(45, 4, 153, 104, 126, 202, 72, 180)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17_value_aux_3),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__16_value),LEAN_SCALAR_PTR_LITERAL(180, 159, 3, 61, 237, 94, 128, 207)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__18 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__18_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__19 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__18_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__20_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__19_value),LEAN_SCALAR_PTR_LITERAL(109, 14, 90, 172, 72, 170, 136, 101)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__20 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__20_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__21;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__22;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instLEInt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__23 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__2_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__24_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__23_value),LEAN_SCALAR_PTR_LITERAL(190, 143, 147, 243, 104, 145, 221, 241)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__24 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__24_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__25;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__26;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Preorder"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__27 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__27_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "toLE"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__28 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__27_value),LEAN_SCALAR_PTR_LITERAL(171, 85, 2, 192, 23, 244, 204, 242)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__29_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__28_value),LEAN_SCALAR_PTR_LITERAL(142, 143, 218, 41, 228, 103, 236, 64)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__29 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__29_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "PartialOrder"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__30 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__30_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toPreorder"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__31 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__30_value),LEAN_SCALAR_PTR_LITERAL(47, 196, 146, 225, 179, 207, 152, 76)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__32_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__31_value),LEAN_SCALAR_PTR_LITERAL(3, 6, 195, 109, 53, 169, 118, 52)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__32 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__32_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "SemilatticeInf"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__33 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__33_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toPartialOrder"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__34 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__34_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__33_value),LEAN_SCALAR_PTR_LITERAL(62, 131, 181, 193, 54, 206, 77, 137)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__35_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__34_value),LEAN_SCALAR_PTR_LITERAL(232, 130, 7, 36, 8, 188, 120, 72)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__35 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__35_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Lattice"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__36 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__36_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toSemilatticeInf"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__37 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__37_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__36_value),LEAN_SCALAR_PTR_LITERAL(58, 214, 49, 195, 61, 20, 1, 8)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__38_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__37_value),LEAN_SCALAR_PTR_LITERAL(164, 130, 80, 139, 133, 157, 146, 245)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__38 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__38_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "DistribLattice"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__39 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__39_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toLattice"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__40 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__40_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__39_value),LEAN_SCALAR_PTR_LITERAL(211, 66, 65, 127, 78, 58, 2, 133)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__40_value),LEAN_SCALAR_PTR_LITERAL(176, 217, 83, 125, 106, 180, 122, 79)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "instDistribLatticeOfLinearOrder"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__42 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__42_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__42_value),LEAN_SCALAR_PTR_LITERAL(204, 88, 224, 186, 247, 116, 10, 234)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__43 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__43_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toInt_le_toInt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__44 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__44_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45_value_aux_1),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45_value_aux_2),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value),LEAN_SCALAR_PTR_LITERAL(45, 4, 153, 104, 126, 202, 72, 180)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45_value_aux_3),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__44_value),LEAN_SCALAR_PTR_LITERAL(155, 129, 124, 243, 238, 176, 223, 18)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__46 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__46_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__46_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__47 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__47_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__48;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "toInt_nle_toInt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__49 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__49_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50_value_aux_1),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50_value_aux_2),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value),LEAN_SCALAR_PTR_LITERAL(45, 4, 153, 104, 126, 202, 72, 180)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50_value_aux_3),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__49_value),LEAN_SCALAR_PTR_LITERAL(24, 90, 20, 80, 211, 244, 135, 38)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__51;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instLTInt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__52 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__52_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__53_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__2_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__53_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__52_value),LEAN_SCALAR_PTR_LITERAL(174, 212, 102, 196, 69, 170, 149, 126)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__53 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__53_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__54;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__55_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__55;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "toLT"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__56 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__56_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__57_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__27_value),LEAN_SCALAR_PTR_LITERAL(171, 85, 2, 192, 23, 244, 204, 242)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__57_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__56_value),LEAN_SCALAR_PTR_LITERAL(213, 59, 145, 160, 110, 90, 162, 17)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__57 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__57_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toInt_lt_toInt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__58 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__58_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59_value_aux_1),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59_value_aux_2),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value),LEAN_SCALAR_PTR_LITERAL(45, 4, 153, 104, 126, 202, 72, 180)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59_value_aux_3),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__58_value),LEAN_SCALAR_PTR_LITERAL(236, 156, 230, 120, 213, 62, 208, 252)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "toInt_nlt_toInt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__60 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__60_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61_value_aux_1),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61_value_aux_2),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value),LEAN_SCALAR_PTR_LITERAL(45, 4, 153, 104, 126, 202, 72, 180)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61_value_aux_3),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__60_value),LEAN_SCALAR_PTR_LITERAL(165, 116, 212, 63, 145, 46, 185, 185)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Min"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__62 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__62_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "min"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__63 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__63_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__64_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__62_value),LEAN_SCALAR_PTR_LITERAL(132, 99, 105, 121, 176, 241, 22, 117)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__64_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__63_value),LEAN_SCALAR_PTR_LITERAL(0, 174, 129, 224, 94, 80, 42, 239)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__64 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__64_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__65;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__66;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "instMin"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__67 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__67_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__68_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__2_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__68_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__67_value),LEAN_SCALAR_PTR_LITERAL(209, 125, 169, 125, 132, 173, 60, 216)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__68 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__68_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__69_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__69;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__70_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__70;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMin"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__71 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__71_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__72_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__33_value),LEAN_SCALAR_PTR_LITERAL(62, 131, 181, 193, 54, 206, 77, 137)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__72_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__71_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 170, 40, 254, 229, 31, 101)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__72 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__72_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "toInt_inf_toInt_eq_toInt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__73 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__73_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74_value_aux_1),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74_value_aux_2),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value),LEAN_SCALAR_PTR_LITERAL(45, 4, 153, 104, 126, 202, 72, 180)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74_value_aux_3),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__73_value),LEAN_SCALAR_PTR_LITERAL(250, 242, 178, 138, 71, 33, 127, 70)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Max"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__75 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__75_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "max"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__76 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__76_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__77_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__75_value),LEAN_SCALAR_PTR_LITERAL(169, 95, 226, 81, 206, 208, 89, 76)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__77_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__76_value),LEAN_SCALAR_PTR_LITERAL(247, 27, 157, 195, 66, 157, 90, 150)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__77 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__77_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__78_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__78;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__79_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__79;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "instMax"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__80 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__80_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__81_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__2_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__81_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__80_value),LEAN_SCALAR_PTR_LITERAL(172, 71, 86, 64, 121, 239, 29, 46)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__81 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__81_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__82_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__82;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__83_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__83;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "SemilatticeSup"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__84 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__84_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMax"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__85 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__85_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__86_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__84_value),LEAN_SCALAR_PTR_LITERAL(208, 106, 75, 248, 165, 223, 103, 224)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__86_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__85_value),LEAN_SCALAR_PTR_LITERAL(200, 184, 226, 98, 165, 177, 120, 75)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__86 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__86_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toSemilatticeSup"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__87 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__87_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__88_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__36_value),LEAN_SCALAR_PTR_LITERAL(58, 214, 49, 195, 61, 20, 1, 8)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__88_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__87_value),LEAN_SCALAR_PTR_LITERAL(91, 190, 251, 233, 17, 206, 48, 18)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__88 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__88_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "toInt_sup_toInt_eq_toInt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__89 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__89_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90_value_aux_0),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90_value_aux_1),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90_value_aux_2),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__3_value),LEAN_SCALAR_PTR_LITERAL(45, 4, 153, 104, 126, 202, 72, 180)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90_value_aux_3),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__89_value),LEAN_SCALAR_PTR_LITERAL(79, 64, 100, 134, 157, 135, 139, 4)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___lam__0(lean_object* v_x_1_){
_start:
{
lean_inc_ref(v_x_1_);
return v_x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___lam__0___boxed(lean_object* v_x_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___lam__0(v_x_2_);
lean_dec_ref(v_x_2_);
return v_res_3_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__3(void){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_8_ = lean_box(0);
v___x_9_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__2));
v___x_10_ = l_Lean_Expr_const___override(v___x_9_, v___x_8_);
return v___x_10_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__12(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_25_ = lean_box(0);
v___x_26_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__11));
v___x_27_ = l_Lean_Expr_const___override(v___x_26_, v___x_25_);
return v___x_27_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__13(void){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_28_ = lean_unsigned_to_nat(0u);
v___x_29_ = l_Lean_Expr_bvar___override(v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun(lean_object* v_u_34_, lean_object* v_00_u03b1_35_, lean_object* v_atoms_36_, lean_object* v_a_37_, lean_object* v_a_38_, lean_object* v_a_39_, lean_object* v_a_40_){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; uint8_t v___x_44_; 
v___x_42_ = lean_array_get_size(v_atoms_36_);
v___x_43_ = lean_unsigned_to_nat(0u);
v___x_44_ = lean_nat_dec_eq(v___x_42_, v___x_43_);
if (v___x_44_ == 0)
{
lean_object* v___f_45_; lean_object* v_rarray_46_; lean_object* v___x_47_; 
v___f_45_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__0));
v_rarray_46_ = l_Lean_RArray_ofArray___redArg(v_atoms_36_);
lean_inc_ref(v_00_u03b1_35_);
v___x_47_ = l_Lean_RArray_toExpr___redArg(v_00_u03b1_35_, v___f_45_, v_rarray_46_, v_a_37_, v_a_38_, v_a_39_, v_a_40_);
if (lean_obj_tag(v___x_47_) == 0)
{
lean_object* v_a_48_; lean_object* v___x_50_; uint8_t v_isShared_51_; uint8_t v_isSharedCheck_72_; 
v_a_48_ = lean_ctor_get(v___x_47_, 0);
v_isSharedCheck_72_ = !lean_is_exclusive(v___x_47_);
if (v_isSharedCheck_72_ == 0)
{
v___x_50_ = v___x_47_;
v_isShared_51_ = v_isSharedCheck_72_;
goto v_resetjp_49_;
}
else
{
lean_inc(v_a_48_);
lean_dec(v___x_47_);
v___x_50_ = lean_box(0);
v_isShared_51_ = v_isSharedCheck_72_;
goto v_resetjp_49_;
}
v_resetjp_49_:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; uint8_t v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_70_; 
v___x_52_ = lean_box(0);
v___x_53_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__3, &lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__3);
v___x_54_ = l_Lean_mkNatLit(v___x_42_);
lean_inc_ref(v___x_54_);
v___x_55_ = l_Lean_Expr_app___override(v___x_53_, v___x_54_);
v___x_56_ = 0;
v___x_57_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__5));
v___x_58_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__9));
v___x_59_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_59_, 0, v_u_34_);
lean_ctor_set(v___x_59_, 1, v___x_52_);
v___x_60_ = l_Lean_Expr_const___override(v___x_58_, v___x_59_);
v___x_61_ = l_Lean_Expr_app___override(v___x_60_, v_00_u03b1_35_);
v___x_62_ = l_Lean_Expr_app___override(v___x_61_, v_a_48_);
v___x_63_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__12, &lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__12);
v___x_64_ = l_Lean_Expr_app___override(v___x_63_, v___x_54_);
v___x_65_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__13, &lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__13);
v___x_66_ = l_Lean_Expr_app___override(v___x_64_, v___x_65_);
v___x_67_ = l_Lean_Expr_app___override(v___x_62_, v___x_66_);
v___x_68_ = l_Lean_Expr_lam___override(v___x_57_, v___x_55_, v___x_67_, v___x_56_);
if (v_isShared_51_ == 0)
{
lean_ctor_set(v___x_50_, 0, v___x_68_);
v___x_70_ = v___x_50_;
goto v_reusejp_69_;
}
else
{
lean_object* v_reuseFailAlloc_71_; 
v_reuseFailAlloc_71_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_71_, 0, v___x_68_);
v___x_70_ = v_reuseFailAlloc_71_;
goto v_reusejp_69_;
}
v_reusejp_69_:
{
return v___x_70_;
}
}
}
else
{
lean_dec_ref(v_00_u03b1_35_);
lean_dec(v_u_34_);
return v___x_47_;
}
}
else
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
lean_dec_ref(v_atoms_36_);
v___x_73_ = l_Lean_Level_succ___override(v_u_34_);
v___x_74_ = lean_box(0);
v___x_75_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___closed__15));
v___x_76_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_73_);
lean_ctor_set(v___x_76_, 1, v___x_74_);
v___x_77_ = l_Lean_Expr_const___override(v___x_75_, v___x_76_);
v___x_78_ = l_Lean_Expr_app___override(v___x_77_, v_00_u03b1_35_);
v___x_79_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun___boxed(lean_object* v_u_80_, lean_object* v_00_u03b1_81_, lean_object* v_atoms_82_, lean_object* v_a_83_, lean_object* v_a_84_, lean_object* v_a_85_, lean_object* v_a_86_, lean_object* v_a_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun(v_u_80_, v_00_u03b1_81_, v_atoms_82_, v_a_83_, v_a_84_, v_a_85_, v_a_86_);
lean_dec(v_a_86_);
lean_dec_ref(v_a_85_);
lean_dec(v_a_84_);
lean_dec_ref(v_a_83_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5_spec__7(lean_object* v_type_89_, lean_object* v_msg_90_){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_91_ = lp_Qq_Qq_instInhabitedQuoted(v_type_89_);
v___x_92_ = lean_panic_fn_borrowed(v___x_91_, v_msg_90_);
lean_dec_ref(v___x_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5_spec__7___boxed(lean_object* v_type_93_, lean_object* v_msg_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5_spec__7(v_type_93_, v_msg_94_);
lean_dec_ref(v_type_93_);
return v_res_95_;
}
}
static lean_object* _init_lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__3(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_99_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__2));
v___x_100_ = lean_unsigned_to_nat(11u);
v___x_101_ = lean_unsigned_to_nat(163u);
v___x_102_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__1));
v___x_103_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__0));
v___x_104_ = l_mkPanicMessageWithDecl(v___x_103_, v___x_102_, v___x_101_, v___x_100_, v___x_99_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5(lean_object* v_type_105_, lean_object* v_a_106_, lean_object* v_x_107_){
_start:
{
if (lean_obj_tag(v_x_107_) == 0)
{
lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_108_ = lean_obj_once(&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__3, &lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__3_once, _init_lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___closed__3);
v___x_109_ = lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5_spec__7(v_type_105_, v___x_108_);
return v___x_109_;
}
else
{
lean_object* v_key_110_; lean_object* v_value_111_; lean_object* v_tail_112_; uint8_t v___x_113_; 
v_key_110_ = lean_ctor_get(v_x_107_, 0);
v_value_111_ = lean_ctor_get(v_x_107_, 1);
v_tail_112_ = lean_ctor_get(v_x_107_, 2);
v___x_113_ = lean_nat_dec_eq(v_key_110_, v_a_106_);
if (v___x_113_ == 0)
{
v_x_107_ = v_tail_112_;
goto _start;
}
else
{
lean_inc(v_value_111_);
return v_value_111_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5___boxed(lean_object* v_type_115_, lean_object* v_a_116_, lean_object* v_x_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5(v_type_115_, v_a_116_, v_x_117_);
lean_dec(v_x_117_);
lean_dec(v_a_116_);
lean_dec_ref(v_type_115_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2(lean_object* v_type_119_, lean_object* v_m_120_, lean_object* v_a_121_){
_start:
{
lean_object* v_buckets_122_; lean_object* v___x_123_; uint64_t v___x_124_; uint64_t v___x_125_; uint64_t v___x_126_; uint64_t v_fold_127_; uint64_t v___x_128_; uint64_t v___x_129_; uint64_t v___x_130_; size_t v___x_131_; size_t v___x_132_; size_t v___x_133_; size_t v___x_134_; size_t v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v_buckets_122_ = lean_ctor_get(v_m_120_, 1);
v___x_123_ = lean_array_get_size(v_buckets_122_);
v___x_124_ = lean_uint64_of_nat(v_a_121_);
v___x_125_ = 32ULL;
v___x_126_ = lean_uint64_shift_right(v___x_124_, v___x_125_);
v_fold_127_ = lean_uint64_xor(v___x_124_, v___x_126_);
v___x_128_ = 16ULL;
v___x_129_ = lean_uint64_shift_right(v_fold_127_, v___x_128_);
v___x_130_ = lean_uint64_xor(v_fold_127_, v___x_129_);
v___x_131_ = lean_uint64_to_usize(v___x_130_);
v___x_132_ = lean_usize_of_nat(v___x_123_);
v___x_133_ = ((size_t)1ULL);
v___x_134_ = lean_usize_sub(v___x_132_, v___x_133_);
v___x_135_ = lean_usize_land(v___x_131_, v___x_134_);
v___x_136_ = lean_array_uget_borrowed(v_buckets_122_, v___x_135_);
v___x_137_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2_spec__5(v_type_119_, v_a_121_, v___x_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2___boxed(lean_object* v_type_138_, lean_object* v_m_139_, lean_object* v_a_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2(v_type_138_, v_m_139_, v_a_140_);
lean_dec(v_a_140_);
lean_dec_ref(v_m_139_);
lean_dec_ref(v_type_138_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___lam__0(lean_object* v_type_142_, lean_object* v_a_143_, lean_object* v_n_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__2(v_type_142_, v_a_143_, v_n_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___lam__0___boxed(lean_object* v_type_146_, lean_object* v_a_147_, lean_object* v_n_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___lam__0(v_type_146_, v_a_147_, v_n_148_);
lean_dec(v_n_148_);
lean_dec_ref(v_a_147_);
lean_dec_ref(v_type_146_);
return v_res_149_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__8(void){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; 
v___x_165_ = lean_box(0);
v___x_166_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__7));
v___x_167_ = l_Lean_Expr_const___override(v___x_166_, v___x_165_);
return v___x_167_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__11(void){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_171_ = lean_box(0);
v___x_172_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__10));
v___x_173_ = l_Lean_Expr_const___override(v___x_172_, v___x_171_);
return v___x_173_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__16(void){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_182_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__15));
v___x_183_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__14));
v___x_184_ = l_Lean_Expr_const___override(v___x_183_, v___x_182_);
return v___x_184_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__19(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_188_ = lean_box(0);
v___x_189_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__18));
v___x_190_ = l_Lean_Expr_const___override(v___x_189_, v___x_188_);
return v___x_190_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__20(void){
_start:
{
lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v___x_191_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__19, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__19_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__19);
v___x_192_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__16, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__16_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__16);
v___x_193_ = l_Lean_Expr_app___override(v___x_192_, v___x_191_);
return v___x_193_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__23(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_197_ = lean_box(0);
v___x_198_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__22));
v___x_199_ = l_Lean_Expr_const___override(v___x_198_, v___x_197_);
return v___x_199_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__24(void){
_start:
{
lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; 
v___x_200_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__23, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__23_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__23);
v___x_201_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__20, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__20_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__20);
v___x_202_ = l_Lean_Expr_app___override(v___x_201_, v___x_200_);
return v___x_202_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__27(void){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_207_ = lean_box(0);
v___x_208_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__26));
v___x_209_ = l_Lean_Expr_const___override(v___x_208_, v___x_207_);
return v___x_209_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__31(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_215_ = lean_box(0);
v___x_216_ = l_Lean_Level_succ___override(v___x_215_);
return v___x_216_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32(void){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_217_ = lean_box(0);
v___x_218_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__31, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__31_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__31);
v___x_219_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_219_, 0, v___x_218_);
lean_ctor_set(v___x_219_, 1, v___x_217_);
return v___x_219_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__33(void){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_220_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32);
v___x_221_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__30));
v___x_222_ = l_Lean_Expr_const___override(v___x_221_, v___x_220_);
return v___x_222_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__36(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_226_ = lean_box(0);
v___x_227_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__35));
v___x_228_ = l_Lean_Expr_const___override(v___x_227_, v___x_226_);
return v___x_228_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__37(void){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_229_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__36, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__36_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__36);
v___x_230_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__33, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__33_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__33);
v___x_231_ = l_Lean_Expr_app___override(v___x_230_, v___x_229_);
return v___x_231_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__41(void){
_start:
{
lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_237_ = lean_box(0);
v___x_238_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__40));
v___x_239_ = l_Lean_Expr_const___override(v___x_238_, v___x_237_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7(lean_object* v_u_240_, lean_object* v_type_241_, lean_object* v_inst_242_, lean_object* v___x_243_, lean_object* v_a_244_, lean_object* v_acc_245_, lean_object* v_a_246_){
_start:
{
if (lean_obj_tag(v_a_246_) == 0)
{
lean_dec_ref(v_a_244_);
lean_dec(v___x_243_);
lean_dec_ref(v_inst_242_);
lean_dec_ref(v_type_241_);
lean_dec(v_u_240_);
return v_acc_245_;
}
else
{
lean_object* v_key_247_; lean_object* v_tail_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_286_; 
v_key_247_ = lean_ctor_get(v_a_246_, 0);
v_tail_248_ = lean_ctor_get(v_a_246_, 2);
v_isSharedCheck_286_ = !lean_is_exclusive(v_a_246_);
if (v_isSharedCheck_286_ == 0)
{
lean_object* v_unused_287_; 
v_unused_287_ = lean_ctor_get(v_a_246_, 1);
lean_dec(v_unused_287_);
v___x_250_ = v_a_246_;
v_isShared_251_ = v_isSharedCheck_286_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_tail_248_);
lean_inc(v_key_247_);
lean_dec(v_a_246_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_286_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_283_; 
v___x_252_ = lean_box(0);
v___x_253_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5));
lean_inc(v_u_240_);
v___x_254_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_254_, 0, v_u_240_);
lean_ctor_set(v___x_254_, 1, v___x_252_);
v___x_255_ = l_Lean_Expr_const___override(v___x_253_, v___x_254_);
lean_inc_ref(v_type_241_);
v___x_256_ = l_Lean_Expr_app___override(v___x_255_, v_type_241_);
lean_inc_ref(v_inst_242_);
v___x_257_ = l_Lean_Expr_app___override(v___x_256_, v_inst_242_);
lean_inc(v___x_243_);
v___x_258_ = lp_mathlib_Qq_mkNatLitQ(v___x_243_);
lean_inc_ref_n(v___x_258_, 3);
v___x_259_ = l_Lean_Expr_app___override(v___x_257_, v___x_258_);
lean_inc_ref(v_a_244_);
v___x_260_ = l_Lean_Expr_app___override(v___x_259_, v_a_244_);
v___x_261_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__8, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__8_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__8);
v___x_262_ = l_Lean_Expr_app___override(v___x_261_, v___x_258_);
lean_inc(v_key_247_);
v___x_263_ = lp_mathlib_Qq_mkNatLitQ(v_key_247_);
lean_inc_ref_n(v___x_263_, 2);
v___x_264_ = l_Lean_Expr_app___override(v___x_262_, v___x_263_);
v___x_265_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__11, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__11_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__11);
v___x_266_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__24, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__24_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__24);
v___x_267_ = l_Lean_Expr_app___override(v___x_266_, v___x_263_);
v___x_268_ = l_Lean_Expr_app___override(v___x_267_, v___x_258_);
lean_inc_ref(v___x_268_);
v___x_269_ = l_Lean_Expr_app___override(v___x_265_, v___x_268_);
v___x_270_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__27, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__27_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__27);
v___x_271_ = l_Lean_Expr_app___override(v___x_270_, v___x_263_);
v___x_272_ = l_Lean_Expr_app___override(v___x_271_, v___x_258_);
lean_inc_ref(v___x_272_);
v___x_273_ = l_Lean_Expr_app___override(v___x_269_, v___x_272_);
v___x_274_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__37, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__37_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__37);
v___x_275_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__41, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__41_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__41);
v___x_276_ = l_Lean_Expr_app___override(v___x_275_, v___x_268_);
v___x_277_ = l_Lean_Expr_app___override(v___x_276_, v___x_272_);
v___x_278_ = l_Lean_Expr_app___override(v___x_274_, v___x_277_);
v___x_279_ = l_Lean_Expr_app___override(v___x_273_, v___x_278_);
v___x_280_ = l_Lean_Expr_app___override(v___x_264_, v___x_279_);
v___x_281_ = l_Lean_Expr_app___override(v___x_260_, v___x_280_);
if (v_isShared_251_ == 0)
{
lean_ctor_set(v___x_250_, 2, v_acc_245_);
lean_ctor_set(v___x_250_, 1, v___x_281_);
v___x_283_ = v___x_250_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v_key_247_);
lean_ctor_set(v_reuseFailAlloc_285_, 1, v___x_281_);
lean_ctor_set(v_reuseFailAlloc_285_, 2, v_acc_245_);
v___x_283_ = v_reuseFailAlloc_285_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
v_acc_245_ = v___x_283_;
v_a_246_ = v_tail_248_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__8(lean_object* v_u_288_, lean_object* v_type_289_, lean_object* v_inst_290_, lean_object* v___x_291_, lean_object* v_a_292_, size_t v_sz_293_, size_t v_i_294_, lean_object* v_bs_295_){
_start:
{
uint8_t v___x_296_; 
v___x_296_ = lean_usize_dec_lt(v_i_294_, v_sz_293_);
if (v___x_296_ == 0)
{
lean_dec_ref(v_a_292_);
lean_dec(v___x_291_);
lean_dec_ref(v_inst_290_);
lean_dec_ref(v_type_289_);
lean_dec(v_u_288_);
return v_bs_295_;
}
else
{
lean_object* v_v_297_; lean_object* v___x_298_; lean_object* v_bs_x27_299_; lean_object* v___x_300_; lean_object* v___x_301_; size_t v___x_302_; size_t v___x_303_; lean_object* v___x_304_; 
v_v_297_ = lean_array_uget(v_bs_295_, v_i_294_);
v___x_298_ = lean_unsigned_to_nat(0u);
v_bs_x27_299_ = lean_array_uset(v_bs_295_, v_i_294_, v___x_298_);
v___x_300_ = lean_box(0);
lean_inc_ref(v_a_292_);
lean_inc(v___x_291_);
lean_inc_ref(v_inst_290_);
lean_inc_ref(v_type_289_);
lean_inc(v_u_288_);
v___x_301_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7(v_u_288_, v_type_289_, v_inst_290_, v___x_291_, v_a_292_, v___x_300_, v_v_297_);
v___x_302_ = ((size_t)1ULL);
v___x_303_ = lean_usize_add(v_i_294_, v___x_302_);
v___x_304_ = lean_array_uset(v_bs_x27_299_, v_i_294_, v___x_301_);
v_i_294_ = v___x_303_;
v_bs_295_ = v___x_304_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__8___boxed(lean_object* v_u_306_, lean_object* v_type_307_, lean_object* v_inst_308_, lean_object* v___x_309_, lean_object* v_a_310_, lean_object* v_sz_311_, lean_object* v_i_312_, lean_object* v_bs_313_){
_start:
{
size_t v_sz_boxed_314_; size_t v_i_boxed_315_; lean_object* v_res_316_; 
v_sz_boxed_314_ = lean_unbox_usize(v_sz_311_);
lean_dec(v_sz_311_);
v_i_boxed_315_ = lean_unbox_usize(v_i_312_);
lean_dec(v_i_312_);
v_res_316_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__8(v_u_306_, v_type_307_, v_inst_308_, v___x_309_, v_a_310_, v_sz_boxed_314_, v_i_boxed_315_, v_bs_313_);
return v_res_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3(lean_object* v_u_317_, lean_object* v_type_318_, lean_object* v_inst_319_, lean_object* v___x_320_, lean_object* v_a_321_, lean_object* v_m_322_){
_start:
{
lean_object* v_size_323_; lean_object* v_buckets_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_334_; 
v_size_323_ = lean_ctor_get(v_m_322_, 0);
v_buckets_324_ = lean_ctor_get(v_m_322_, 1);
v_isSharedCheck_334_ = !lean_is_exclusive(v_m_322_);
if (v_isSharedCheck_334_ == 0)
{
v___x_326_ = v_m_322_;
v_isShared_327_ = v_isSharedCheck_334_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_buckets_324_);
lean_inc(v_size_323_);
lean_dec(v_m_322_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_334_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
size_t v_sz_328_; size_t v___x_329_; lean_object* v_newBuckets_330_; lean_object* v___x_332_; 
v_sz_328_ = lean_array_size(v_buckets_324_);
v___x_329_ = ((size_t)0ULL);
v_newBuckets_330_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__8(v_u_317_, v_type_318_, v_inst_319_, v___x_320_, v_a_321_, v_sz_328_, v___x_329_, v_buckets_324_);
if (v_isShared_327_ == 0)
{
lean_ctor_set(v___x_326_, 1, v_newBuckets_330_);
v___x_332_ = v___x_326_;
goto v_reusejp_331_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v_size_323_);
lean_ctor_set(v_reuseFailAlloc_333_, 1, v_newBuckets_330_);
v___x_332_ = v_reuseFailAlloc_333_;
goto v_reusejp_331_;
}
v_reusejp_331_:
{
return v___x_332_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2_spec__7___redArg(lean_object* v_x_335_, lean_object* v_x_336_){
_start:
{
if (lean_obj_tag(v_x_336_) == 0)
{
return v_x_335_;
}
else
{
lean_object* v_key_337_; lean_object* v_value_338_; lean_object* v_tail_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_362_; 
v_key_337_ = lean_ctor_get(v_x_336_, 0);
v_value_338_ = lean_ctor_get(v_x_336_, 1);
v_tail_339_ = lean_ctor_get(v_x_336_, 2);
v_isSharedCheck_362_ = !lean_is_exclusive(v_x_336_);
if (v_isSharedCheck_362_ == 0)
{
v___x_341_ = v_x_336_;
v_isShared_342_ = v_isSharedCheck_362_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_tail_339_);
lean_inc(v_value_338_);
lean_inc(v_key_337_);
lean_dec(v_x_336_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_362_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
lean_object* v___x_343_; uint64_t v___x_344_; uint64_t v___x_345_; uint64_t v___x_346_; uint64_t v_fold_347_; uint64_t v___x_348_; uint64_t v___x_349_; uint64_t v___x_350_; size_t v___x_351_; size_t v___x_352_; size_t v___x_353_; size_t v___x_354_; size_t v___x_355_; lean_object* v___x_356_; lean_object* v___x_358_; 
v___x_343_ = lean_array_get_size(v_x_335_);
v___x_344_ = lean_uint64_of_nat(v_key_337_);
v___x_345_ = 32ULL;
v___x_346_ = lean_uint64_shift_right(v___x_344_, v___x_345_);
v_fold_347_ = lean_uint64_xor(v___x_344_, v___x_346_);
v___x_348_ = 16ULL;
v___x_349_ = lean_uint64_shift_right(v_fold_347_, v___x_348_);
v___x_350_ = lean_uint64_xor(v_fold_347_, v___x_349_);
v___x_351_ = lean_uint64_to_usize(v___x_350_);
v___x_352_ = lean_usize_of_nat(v___x_343_);
v___x_353_ = ((size_t)1ULL);
v___x_354_ = lean_usize_sub(v___x_352_, v___x_353_);
v___x_355_ = lean_usize_land(v___x_351_, v___x_354_);
v___x_356_ = lean_array_uget_borrowed(v_x_335_, v___x_355_);
lean_inc(v___x_356_);
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 2, v___x_356_);
v___x_358_ = v___x_341_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_361_; 
v_reuseFailAlloc_361_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_361_, 0, v_key_337_);
lean_ctor_set(v_reuseFailAlloc_361_, 1, v_value_338_);
lean_ctor_set(v_reuseFailAlloc_361_, 2, v___x_356_);
v___x_358_ = v_reuseFailAlloc_361_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
lean_object* v___x_359_; 
v___x_359_ = lean_array_uset(v_x_335_, v___x_355_, v___x_358_);
v_x_335_ = v___x_359_;
v_x_336_ = v_tail_339_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2___redArg(lean_object* v_i_363_, lean_object* v_source_364_, lean_object* v_target_365_){
_start:
{
lean_object* v___x_366_; uint8_t v___x_367_; 
v___x_366_ = lean_array_get_size(v_source_364_);
v___x_367_ = lean_nat_dec_lt(v_i_363_, v___x_366_);
if (v___x_367_ == 0)
{
lean_dec_ref(v_source_364_);
lean_dec(v_i_363_);
return v_target_365_;
}
else
{
lean_object* v_es_368_; lean_object* v___x_369_; lean_object* v_source_370_; lean_object* v_target_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v_es_368_ = lean_array_fget(v_source_364_, v_i_363_);
v___x_369_ = lean_box(0);
v_source_370_ = lean_array_fset(v_source_364_, v_i_363_, v___x_369_);
v_target_371_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2_spec__7___redArg(v_target_365_, v_es_368_);
v___x_372_ = lean_unsigned_to_nat(1u);
v___x_373_ = lean_nat_add(v_i_363_, v___x_372_);
lean_dec(v_i_363_);
v_i_363_ = v___x_373_;
v_source_364_ = v_source_370_;
v_target_365_ = v_target_371_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1___redArg(lean_object* v_data_375_){
_start:
{
lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v_nbuckets_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
v___x_376_ = lean_array_get_size(v_data_375_);
v___x_377_ = lean_unsigned_to_nat(2u);
v_nbuckets_378_ = lean_nat_mul(v___x_376_, v___x_377_);
v___x_379_ = lean_unsigned_to_nat(0u);
v___x_380_ = lean_box(0);
v___x_381_ = lean_mk_array(v_nbuckets_378_, v___x_380_);
v___x_382_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2___redArg(v___x_379_, v_data_375_, v___x_381_);
return v___x_382_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0___redArg(lean_object* v_a_383_, lean_object* v_x_384_){
_start:
{
if (lean_obj_tag(v_x_384_) == 0)
{
uint8_t v___x_385_; 
v___x_385_ = 0;
return v___x_385_;
}
else
{
lean_object* v_key_386_; lean_object* v_tail_387_; uint8_t v___x_388_; 
v_key_386_ = lean_ctor_get(v_x_384_, 0);
v_tail_387_ = lean_ctor_get(v_x_384_, 2);
v___x_388_ = lean_nat_dec_eq(v_key_386_, v_a_383_);
if (v___x_388_ == 0)
{
v_x_384_ = v_tail_387_;
goto _start;
}
else
{
return v___x_388_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0___redArg___boxed(lean_object* v_a_390_, lean_object* v_x_391_){
_start:
{
uint8_t v_res_392_; lean_object* v_r_393_; 
v_res_392_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0___redArg(v_a_390_, v_x_391_);
lean_dec(v_x_391_);
lean_dec(v_a_390_);
v_r_393_ = lean_box(v_res_392_);
return v_r_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__2___redArg(lean_object* v_a_394_, lean_object* v_b_395_, lean_object* v_x_396_){
_start:
{
if (lean_obj_tag(v_x_396_) == 0)
{
lean_dec(v_b_395_);
lean_dec(v_a_394_);
return v_x_396_;
}
else
{
lean_object* v_key_397_; lean_object* v_value_398_; lean_object* v_tail_399_; lean_object* v___x_401_; uint8_t v_isShared_402_; uint8_t v_isSharedCheck_411_; 
v_key_397_ = lean_ctor_get(v_x_396_, 0);
v_value_398_ = lean_ctor_get(v_x_396_, 1);
v_tail_399_ = lean_ctor_get(v_x_396_, 2);
v_isSharedCheck_411_ = !lean_is_exclusive(v_x_396_);
if (v_isSharedCheck_411_ == 0)
{
v___x_401_ = v_x_396_;
v_isShared_402_ = v_isSharedCheck_411_;
goto v_resetjp_400_;
}
else
{
lean_inc(v_tail_399_);
lean_inc(v_value_398_);
lean_inc(v_key_397_);
lean_dec(v_x_396_);
v___x_401_ = lean_box(0);
v_isShared_402_ = v_isSharedCheck_411_;
goto v_resetjp_400_;
}
v_resetjp_400_:
{
uint8_t v___x_403_; 
v___x_403_ = lean_nat_dec_eq(v_key_397_, v_a_394_);
if (v___x_403_ == 0)
{
lean_object* v___x_404_; lean_object* v___x_406_; 
v___x_404_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__2___redArg(v_a_394_, v_b_395_, v_tail_399_);
if (v_isShared_402_ == 0)
{
lean_ctor_set(v___x_401_, 2, v___x_404_);
v___x_406_ = v___x_401_;
goto v_reusejp_405_;
}
else
{
lean_object* v_reuseFailAlloc_407_; 
v_reuseFailAlloc_407_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_407_, 0, v_key_397_);
lean_ctor_set(v_reuseFailAlloc_407_, 1, v_value_398_);
lean_ctor_set(v_reuseFailAlloc_407_, 2, v___x_404_);
v___x_406_ = v_reuseFailAlloc_407_;
goto v_reusejp_405_;
}
v_reusejp_405_:
{
return v___x_406_;
}
}
else
{
lean_object* v___x_409_; 
lean_dec(v_value_398_);
lean_dec(v_key_397_);
if (v_isShared_402_ == 0)
{
lean_ctor_set(v___x_401_, 1, v_b_395_);
lean_ctor_set(v___x_401_, 0, v_a_394_);
v___x_409_ = v___x_401_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_410_; 
v_reuseFailAlloc_410_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_410_, 0, v_a_394_);
lean_ctor_set(v_reuseFailAlloc_410_, 1, v_b_395_);
lean_ctor_set(v_reuseFailAlloc_410_, 2, v_tail_399_);
v___x_409_ = v_reuseFailAlloc_410_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
return v___x_409_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0___redArg(lean_object* v_m_412_, lean_object* v_a_413_, lean_object* v_b_414_){
_start:
{
lean_object* v_size_415_; lean_object* v_buckets_416_; lean_object* v___x_418_; uint8_t v_isShared_419_; uint8_t v_isSharedCheck_459_; 
v_size_415_ = lean_ctor_get(v_m_412_, 0);
v_buckets_416_ = lean_ctor_get(v_m_412_, 1);
v_isSharedCheck_459_ = !lean_is_exclusive(v_m_412_);
if (v_isSharedCheck_459_ == 0)
{
v___x_418_ = v_m_412_;
v_isShared_419_ = v_isSharedCheck_459_;
goto v_resetjp_417_;
}
else
{
lean_inc(v_buckets_416_);
lean_inc(v_size_415_);
lean_dec(v_m_412_);
v___x_418_ = lean_box(0);
v_isShared_419_ = v_isSharedCheck_459_;
goto v_resetjp_417_;
}
v_resetjp_417_:
{
lean_object* v___x_420_; uint64_t v___x_421_; uint64_t v___x_422_; uint64_t v___x_423_; uint64_t v_fold_424_; uint64_t v___x_425_; uint64_t v___x_426_; uint64_t v___x_427_; size_t v___x_428_; size_t v___x_429_; size_t v___x_430_; size_t v___x_431_; size_t v___x_432_; lean_object* v_bkt_433_; uint8_t v___x_434_; 
v___x_420_ = lean_array_get_size(v_buckets_416_);
v___x_421_ = lean_uint64_of_nat(v_a_413_);
v___x_422_ = 32ULL;
v___x_423_ = lean_uint64_shift_right(v___x_421_, v___x_422_);
v_fold_424_ = lean_uint64_xor(v___x_421_, v___x_423_);
v___x_425_ = 16ULL;
v___x_426_ = lean_uint64_shift_right(v_fold_424_, v___x_425_);
v___x_427_ = lean_uint64_xor(v_fold_424_, v___x_426_);
v___x_428_ = lean_uint64_to_usize(v___x_427_);
v___x_429_ = lean_usize_of_nat(v___x_420_);
v___x_430_ = ((size_t)1ULL);
v___x_431_ = lean_usize_sub(v___x_429_, v___x_430_);
v___x_432_ = lean_usize_land(v___x_428_, v___x_431_);
v_bkt_433_ = lean_array_uget_borrowed(v_buckets_416_, v___x_432_);
v___x_434_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0___redArg(v_a_413_, v_bkt_433_);
if (v___x_434_ == 0)
{
lean_object* v___x_435_; lean_object* v_size_x27_436_; lean_object* v___x_437_; lean_object* v_buckets_x27_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; uint8_t v___x_444_; 
v___x_435_ = lean_unsigned_to_nat(1u);
v_size_x27_436_ = lean_nat_add(v_size_415_, v___x_435_);
lean_dec(v_size_415_);
lean_inc(v_bkt_433_);
v___x_437_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_437_, 0, v_a_413_);
lean_ctor_set(v___x_437_, 1, v_b_414_);
lean_ctor_set(v___x_437_, 2, v_bkt_433_);
v_buckets_x27_438_ = lean_array_uset(v_buckets_416_, v___x_432_, v___x_437_);
v___x_439_ = lean_unsigned_to_nat(4u);
v___x_440_ = lean_nat_mul(v_size_x27_436_, v___x_439_);
v___x_441_ = lean_unsigned_to_nat(3u);
v___x_442_ = lean_nat_div(v___x_440_, v___x_441_);
lean_dec(v___x_440_);
v___x_443_ = lean_array_get_size(v_buckets_x27_438_);
v___x_444_ = lean_nat_dec_le(v___x_442_, v___x_443_);
lean_dec(v___x_442_);
if (v___x_444_ == 0)
{
lean_object* v_val_445_; lean_object* v___x_447_; 
v_val_445_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1___redArg(v_buckets_x27_438_);
if (v_isShared_419_ == 0)
{
lean_ctor_set(v___x_418_, 1, v_val_445_);
lean_ctor_set(v___x_418_, 0, v_size_x27_436_);
v___x_447_ = v___x_418_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_448_; 
v_reuseFailAlloc_448_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_448_, 0, v_size_x27_436_);
lean_ctor_set(v_reuseFailAlloc_448_, 1, v_val_445_);
v___x_447_ = v_reuseFailAlloc_448_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
return v___x_447_;
}
}
else
{
lean_object* v___x_450_; 
if (v_isShared_419_ == 0)
{
lean_ctor_set(v___x_418_, 1, v_buckets_x27_438_);
lean_ctor_set(v___x_418_, 0, v_size_x27_436_);
v___x_450_ = v___x_418_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_size_x27_436_);
lean_ctor_set(v_reuseFailAlloc_451_, 1, v_buckets_x27_438_);
v___x_450_ = v_reuseFailAlloc_451_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
return v___x_450_;
}
}
}
else
{
lean_object* v___x_452_; lean_object* v_buckets_x27_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_457_; 
lean_inc(v_bkt_433_);
v___x_452_ = lean_box(0);
v_buckets_x27_453_ = lean_array_uset(v_buckets_416_, v___x_432_, v___x_452_);
v___x_454_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__2___redArg(v_a_413_, v_b_414_, v_bkt_433_);
v___x_455_ = lean_array_uset(v_buckets_x27_453_, v___x_432_, v___x_454_);
if (v_isShared_419_ == 0)
{
lean_ctor_set(v___x_418_, 1, v___x_455_);
v___x_457_ = v___x_418_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v_size_415_);
lean_ctor_set(v_reuseFailAlloc_458_, 1, v___x_455_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
return v___x_457_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1___redArg(lean_object* v_type_460_, lean_object* v_as_461_, size_t v_sz_462_, size_t v_i_463_, lean_object* v_b_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_){
_start:
{
lean_object* v_a_471_; uint8_t v___x_475_; 
v___x_475_ = lean_usize_dec_lt(v_i_463_, v_sz_462_);
if (v___x_475_ == 0)
{
lean_object* v___x_476_; 
lean_dec_ref(v_type_460_);
v___x_476_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_476_, 0, v_b_464_);
return v___x_476_;
}
else
{
lean_object* v_a_477_; uint8_t v_a_479_; lean_object* v___x_482_; 
v_a_477_ = lean_array_uget_borrowed(v_as_461_, v_i_463_);
lean_inc(v___y_468_);
lean_inc_ref(v___y_467_);
lean_inc(v___y_466_);
lean_inc_ref(v___y_465_);
lean_inc(v_a_477_);
v___x_482_ = lean_infer_type(v_a_477_, v___y_465_, v___y_466_, v___y_467_, v___y_468_);
if (lean_obj_tag(v___x_482_) == 0)
{
lean_object* v_a_483_; lean_object* v_keyedConfig_484_; uint8_t v_trackZetaDelta_485_; lean_object* v_zetaDeltaSet_486_; lean_object* v_lctx_487_; lean_object* v_localInstances_488_; lean_object* v_defEqCtx_x3f_489_; lean_object* v_synthPendingDepth_490_; lean_object* v_customCanUnfoldPredicate_x3f_491_; uint8_t v_univApprox_492_; uint8_t v_inTypeClassResolution_493_; uint8_t v_cacheInferType_494_; uint8_t v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; 
v_a_483_ = lean_ctor_get(v___x_482_, 0);
lean_inc(v_a_483_);
lean_dec_ref_known(v___x_482_, 1);
v_keyedConfig_484_ = lean_ctor_get(v___y_465_, 0);
v_trackZetaDelta_485_ = lean_ctor_get_uint8(v___y_465_, sizeof(void*)*7);
v_zetaDeltaSet_486_ = lean_ctor_get(v___y_465_, 1);
v_lctx_487_ = lean_ctor_get(v___y_465_, 2);
v_localInstances_488_ = lean_ctor_get(v___y_465_, 3);
v_defEqCtx_x3f_489_ = lean_ctor_get(v___y_465_, 4);
v_synthPendingDepth_490_ = lean_ctor_get(v___y_465_, 5);
v_customCanUnfoldPredicate_x3f_491_ = lean_ctor_get(v___y_465_, 6);
v_univApprox_492_ = lean_ctor_get_uint8(v___y_465_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_493_ = lean_ctor_get_uint8(v___y_465_, sizeof(void*)*7 + 2);
v_cacheInferType_494_ = lean_ctor_get_uint8(v___y_465_, sizeof(void*)*7 + 3);
v___x_495_ = 2;
lean_inc_ref(v_keyedConfig_484_);
v___x_496_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_495_, v_keyedConfig_484_);
lean_inc(v_customCanUnfoldPredicate_x3f_491_);
lean_inc(v_synthPendingDepth_490_);
lean_inc(v_defEqCtx_x3f_489_);
lean_inc_ref(v_localInstances_488_);
lean_inc_ref(v_lctx_487_);
lean_inc(v_zetaDeltaSet_486_);
v___x_497_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_497_, 0, v___x_496_);
lean_ctor_set(v___x_497_, 1, v_zetaDeltaSet_486_);
lean_ctor_set(v___x_497_, 2, v_lctx_487_);
lean_ctor_set(v___x_497_, 3, v_localInstances_488_);
lean_ctor_set(v___x_497_, 4, v_defEqCtx_x3f_489_);
lean_ctor_set(v___x_497_, 5, v_synthPendingDepth_490_);
lean_ctor_set(v___x_497_, 6, v_customCanUnfoldPredicate_x3f_491_);
lean_ctor_set_uint8(v___x_497_, sizeof(void*)*7, v_trackZetaDelta_485_);
lean_ctor_set_uint8(v___x_497_, sizeof(void*)*7 + 1, v_univApprox_492_);
lean_ctor_set_uint8(v___x_497_, sizeof(void*)*7 + 2, v_inTypeClassResolution_493_);
lean_ctor_set_uint8(v___x_497_, sizeof(void*)*7 + 3, v_cacheInferType_494_);
lean_inc_ref(v_type_460_);
v___x_498_ = l_Lean_Meta_isExprDefEq(v_type_460_, v_a_483_, v___x_497_, v___y_466_, v___y_467_, v___y_468_);
lean_dec_ref_known(v___x_497_, 7);
if (lean_obj_tag(v___x_498_) == 0)
{
lean_object* v_a_499_; uint8_t v___x_500_; 
v_a_499_ = lean_ctor_get(v___x_498_, 0);
lean_inc(v_a_499_);
lean_dec_ref_known(v___x_498_, 1);
v___x_500_ = lean_unbox(v_a_499_);
lean_dec(v_a_499_);
v_a_479_ = v___x_500_;
goto v___jp_478_;
}
else
{
if (lean_obj_tag(v___x_498_) == 0)
{
lean_object* v_a_501_; uint8_t v___x_502_; 
v_a_501_ = lean_ctor_get(v___x_498_, 0);
lean_inc(v_a_501_);
lean_dec_ref_known(v___x_498_, 1);
v___x_502_ = lean_unbox(v_a_501_);
lean_dec(v_a_501_);
v_a_479_ = v___x_502_;
goto v___jp_478_;
}
else
{
lean_object* v_a_503_; lean_object* v___x_505_; uint8_t v_isShared_506_; uint8_t v_isSharedCheck_510_; 
lean_dec_ref(v_b_464_);
lean_dec_ref(v_type_460_);
v_a_503_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_510_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_510_ == 0)
{
v___x_505_ = v___x_498_;
v_isShared_506_ = v_isSharedCheck_510_;
goto v_resetjp_504_;
}
else
{
lean_inc(v_a_503_);
lean_dec(v___x_498_);
v___x_505_ = lean_box(0);
v_isShared_506_ = v_isSharedCheck_510_;
goto v_resetjp_504_;
}
v_resetjp_504_:
{
lean_object* v___x_508_; 
if (v_isShared_506_ == 0)
{
v___x_508_ = v___x_505_;
goto v_reusejp_507_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v_a_503_);
v___x_508_ = v_reuseFailAlloc_509_;
goto v_reusejp_507_;
}
v_reusejp_507_:
{
return v___x_508_;
}
}
}
}
}
else
{
lean_object* v_a_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_518_; 
lean_dec_ref(v_b_464_);
lean_dec_ref(v_type_460_);
v_a_511_ = lean_ctor_get(v___x_482_, 0);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_482_);
if (v_isSharedCheck_518_ == 0)
{
v___x_513_ = v___x_482_;
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_a_511_);
lean_dec(v___x_482_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v___x_516_; 
if (v_isShared_514_ == 0)
{
v___x_516_ = v___x_513_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_a_511_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
v___jp_478_:
{
if (v_a_479_ == 0)
{
v_a_471_ = v_b_464_;
goto v___jp_470_;
}
else
{
lean_object* v_size_480_; lean_object* v___x_481_; 
v_size_480_ = lean_ctor_get(v_b_464_, 0);
lean_inc(v_size_480_);
lean_inc(v_a_477_);
v___x_481_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0___redArg(v_b_464_, v_size_480_, v_a_477_);
v_a_471_ = v___x_481_;
goto v___jp_470_;
}
}
}
v___jp_470_:
{
size_t v___x_472_; size_t v___x_473_; 
v___x_472_ = ((size_t)1ULL);
v___x_473_ = lean_usize_add(v_i_463_, v___x_472_);
v_i_463_ = v___x_473_;
v_b_464_ = v_a_471_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1___redArg___boxed(lean_object* v_type_519_, lean_object* v_as_520_, lean_object* v_sz_521_, lean_object* v_i_522_, lean_object* v_b_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_){
_start:
{
size_t v_sz_boxed_529_; size_t v_i_boxed_530_; lean_object* v_res_531_; 
v_sz_boxed_529_ = lean_unbox_usize(v_sz_521_);
lean_dec(v_sz_521_);
v_i_boxed_530_ = lean_unbox_usize(v_i_522_);
lean_dec(v_i_522_);
v_res_531_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1___redArg(v_type_519_, v_as_520_, v_sz_boxed_529_, v_i_boxed_530_, v_b_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_);
lean_dec(v___y_527_);
lean_dec_ref(v___y_526_);
lean_dec(v___y_525_);
lean_dec_ref(v___y_524_);
lean_dec_ref(v_as_520_);
return v_res_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(lean_object* v___x_532_, lean_object* v_k_533_){
_start:
{
lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; 
v___x_534_ = lp_mathlib_Qq_mkNatLitQ(v___x_532_);
v___x_535_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__8, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__8_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__8);
lean_inc_ref_n(v___x_534_, 2);
v___x_536_ = l_Lean_Expr_app___override(v___x_535_, v___x_534_);
v___x_537_ = lp_mathlib_Qq_mkNatLitQ(v_k_533_);
lean_inc_ref_n(v___x_537_, 2);
v___x_538_ = l_Lean_Expr_app___override(v___x_536_, v___x_537_);
v___x_539_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__11, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__11_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__11);
v___x_540_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__24, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__24_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__24);
v___x_541_ = l_Lean_Expr_app___override(v___x_540_, v___x_537_);
v___x_542_ = l_Lean_Expr_app___override(v___x_541_, v___x_534_);
lean_inc_ref(v___x_542_);
v___x_543_ = l_Lean_Expr_app___override(v___x_539_, v___x_542_);
v___x_544_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__27, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__27_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__27);
v___x_545_ = l_Lean_Expr_app___override(v___x_544_, v___x_537_);
v___x_546_ = l_Lean_Expr_app___override(v___x_545_, v___x_534_);
lean_inc_ref(v___x_546_);
v___x_547_ = l_Lean_Expr_app___override(v___x_543_, v___x_546_);
v___x_548_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__37, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__37_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__37);
v___x_549_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__41, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__41_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__41);
v___x_550_ = l_Lean_Expr_app___override(v___x_549_, v___x_542_);
v___x_551_ = l_Lean_Expr_app___override(v___x_550_, v___x_546_);
v___x_552_ = l_Lean_Expr_app___override(v___x_548_, v___x_551_);
v___x_553_ = l_Lean_Expr_app___override(v___x_547_, v___x_552_);
v___x_554_ = l_Lean_Expr_app___override(v___x_538_, v___x_553_);
return v___x_554_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__1(void){
_start:
{
lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; 
v___x_557_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32);
v___x_558_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__0));
v___x_559_ = l_Lean_Expr_const___override(v___x_558_, v___x_557_);
return v___x_559_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4(void){
_start:
{
lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; 
v___x_563_ = lean_box(0);
v___x_564_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__3));
v___x_565_ = l_Lean_Expr_const___override(v___x_564_, v___x_563_);
return v___x_565_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5(void){
_start:
{
lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; 
v___x_566_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4);
v___x_567_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__1);
v___x_568_ = l_Lean_Expr_app___override(v___x_567_, v___x_566_);
return v___x_568_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9(void){
_start:
{
lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; 
v___x_574_ = lean_box(0);
v___x_575_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__8));
v___x_576_ = l_Lean_Expr_const___override(v___x_575_, v___x_574_);
return v___x_576_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__14(void){
_start:
{
lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; 
v___x_587_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__32);
v___x_588_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__13));
v___x_589_ = l_Lean_Expr_const___override(v___x_588_, v___x_587_);
return v___x_589_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__15(void){
_start:
{
lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; 
v___x_590_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4);
v___x_591_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__14, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__14_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__14);
v___x_592_ = l_Lean_Expr_app___override(v___x_591_, v___x_590_);
return v___x_592_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__21(void){
_start:
{
lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; 
v___x_605_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__15));
v___x_606_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__20));
v___x_607_ = l_Lean_Expr_const___override(v___x_606_, v___x_605_);
return v___x_607_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__22(void){
_start:
{
lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; 
v___x_608_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4);
v___x_609_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__21, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__21_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__21);
v___x_610_ = l_Lean_Expr_app___override(v___x_609_, v___x_608_);
return v___x_610_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__25(void){
_start:
{
lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; 
v___x_615_ = lean_box(0);
v___x_616_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__24));
v___x_617_ = l_Lean_Expr_const___override(v___x_616_, v___x_615_);
return v___x_617_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__26(void){
_start:
{
lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; 
v___x_618_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__25, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__25_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__25);
v___x_619_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__22, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__22_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__22);
v___x_620_ = l_Lean_Expr_app___override(v___x_619_, v___x_618_);
return v___x_620_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__48(void){
_start:
{
lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; 
v___x_659_ = lean_box(0);
v___x_660_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__47));
v___x_661_ = l_Lean_Expr_const___override(v___x_660_, v___x_659_);
return v___x_661_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__51(void){
_start:
{
lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; 
v___x_669_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4);
v___x_670_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__16, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__16_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__16);
v___x_671_ = l_Lean_Expr_app___override(v___x_670_, v___x_669_);
return v___x_671_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__54(void){
_start:
{
lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; 
v___x_676_ = lean_box(0);
v___x_677_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__53));
v___x_678_ = l_Lean_Expr_const___override(v___x_677_, v___x_676_);
return v___x_678_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__55(void){
_start:
{
lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; 
v___x_679_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__54, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__54_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__54);
v___x_680_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__51, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__51_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__51);
v___x_681_ = l_Lean_Expr_app___override(v___x_680_, v___x_679_);
return v___x_681_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__65(void){
_start:
{
lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_705_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__15));
v___x_706_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__64));
v___x_707_ = l_Lean_Expr_const___override(v___x_706_, v___x_705_);
return v___x_707_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__66(void){
_start:
{
lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; 
v___x_708_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4);
v___x_709_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__65, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__65_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__65);
v___x_710_ = l_Lean_Expr_app___override(v___x_709_, v___x_708_);
return v___x_710_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__69(void){
_start:
{
lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; 
v___x_715_ = lean_box(0);
v___x_716_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__68));
v___x_717_ = l_Lean_Expr_const___override(v___x_716_, v___x_715_);
return v___x_717_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__70(void){
_start:
{
lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_718_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__69, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__69_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__69);
v___x_719_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__66, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__66_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__66);
v___x_720_ = l_Lean_Expr_app___override(v___x_719_, v___x_718_);
return v___x_720_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__78(void){
_start:
{
lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
v___x_737_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__15));
v___x_738_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__77));
v___x_739_ = l_Lean_Expr_const___override(v___x_738_, v___x_737_);
return v___x_739_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__79(void){
_start:
{
lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; 
v___x_740_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__4);
v___x_741_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__78, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__78_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__78);
v___x_742_ = l_Lean_Expr_app___override(v___x_741_, v___x_740_);
return v___x_742_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__82(void){
_start:
{
lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; 
v___x_747_ = lean_box(0);
v___x_748_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__81));
v___x_749_ = l_Lean_Expr_const___override(v___x_748_, v___x_747_);
return v___x_749_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__83(void){
_start:
{
lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; 
v___x_750_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__82, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__82_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__82);
v___x_751_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__79, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__79_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__79);
v___x_752_ = l_Lean_Expr_app___override(v___x_751_, v___x_750_);
return v___x_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4(lean_object* v_u_769_, lean_object* v_type_770_, lean_object* v_inst_771_, lean_object* v___x_772_, lean_object* v_a_773_, lean_object* v_as_774_, size_t v_i_775_, size_t v_stop_776_, lean_object* v_b_777_){
_start:
{
lean_object* v___y_779_; uint8_t v___x_783_; 
v___x_783_ = lean_usize_dec_eq(v_i_775_, v_stop_776_);
if (v___x_783_ == 0)
{
lean_object* v_snd_784_; lean_object* v_fst_785_; lean_object* v_fst_786_; lean_object* v_snd_787_; lean_object* v___x_789_; uint8_t v_isShared_790_; uint8_t v_isSharedCheck_1490_; 
v_snd_784_ = lean_ctor_get(v_b_777_, 1);
lean_inc(v_snd_784_);
v_fst_785_ = lean_ctor_get(v_b_777_, 0);
v_fst_786_ = lean_ctor_get(v_snd_784_, 0);
v_snd_787_ = lean_ctor_get(v_snd_784_, 1);
v_isSharedCheck_1490_ = !lean_is_exclusive(v_snd_784_);
if (v_isSharedCheck_1490_ == 0)
{
v___x_789_ = v_snd_784_;
v_isShared_790_ = v_isSharedCheck_1490_;
goto v_resetjp_788_;
}
else
{
lean_inc(v_snd_787_);
lean_inc(v_fst_786_);
lean_dec(v_snd_784_);
v___x_789_ = lean_box(0);
v_isShared_790_ = v_isSharedCheck_1490_;
goto v_resetjp_788_;
}
v_resetjp_788_:
{
lean_object* v___x_791_; 
v___x_791_ = lean_array_uget(v_as_774_, v_i_775_);
switch(lean_obj_tag(v___x_791_))
{
case 0:
{
lean_object* v___x_793_; uint8_t v_isShared_794_; uint8_t v_isSharedCheck_854_; 
lean_inc(v_fst_785_);
v_isSharedCheck_854_ = !lean_is_exclusive(v_b_777_);
if (v_isSharedCheck_854_ == 0)
{
lean_object* v_unused_855_; lean_object* v_unused_856_; 
v_unused_855_ = lean_ctor_get(v_b_777_, 1);
lean_dec(v_unused_855_);
v_unused_856_ = lean_ctor_get(v_b_777_, 0);
lean_dec(v_unused_856_);
v___x_793_ = v_b_777_;
v_isShared_794_ = v_isSharedCheck_854_;
goto v_resetjp_792_;
}
else
{
lean_dec(v_b_777_);
v___x_793_ = lean_box(0);
v_isShared_794_ = v_isSharedCheck_854_;
goto v_resetjp_792_;
}
v_resetjp_792_:
{
lean_object* v_lhs_795_; lean_object* v_rhs_796_; lean_object* v_proof_797_; lean_object* v___x_799_; uint8_t v_isShared_800_; uint8_t v_isSharedCheck_853_; 
v_lhs_795_ = lean_ctor_get(v___x_791_, 0);
v_rhs_796_ = lean_ctor_get(v___x_791_, 1);
v_proof_797_ = lean_ctor_get(v___x_791_, 2);
v_isSharedCheck_853_ = !lean_is_exclusive(v___x_791_);
if (v_isSharedCheck_853_ == 0)
{
v___x_799_ = v___x_791_;
v_isShared_800_ = v_isSharedCheck_853_;
goto v_resetjp_798_;
}
else
{
lean_inc(v_proof_797_);
lean_inc(v_rhs_796_);
lean_inc(v_lhs_795_);
lean_dec(v___x_791_);
v___x_799_ = lean_box(0);
v_isShared_800_ = v_isSharedCheck_853_;
goto v_resetjp_798_;
}
v_resetjp_798_:
{
lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_844_; 
v___x_801_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__0));
v___x_802_ = lean_box(0);
v___x_803_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5);
v___x_804_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5));
lean_inc_n(v_u_769_, 2);
v___x_805_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_805_, 0, v_u_769_);
lean_ctor_set(v___x_805_, 1, v___x_802_);
lean_inc_ref(v___x_805_);
v___x_806_ = l_Lean_Expr_const___override(v___x_804_, v___x_805_);
lean_inc_ref_n(v_type_770_, 3);
v___x_807_ = l_Lean_Expr_app___override(v___x_806_, v_type_770_);
lean_inc_ref_n(v_inst_771_, 2);
v___x_808_ = l_Lean_Expr_app___override(v___x_807_, v_inst_771_);
lean_inc_n(v___x_772_, 3);
v___x_809_ = lp_mathlib_Qq_mkNatLitQ(v___x_772_);
lean_inc_ref(v___x_809_);
v___x_810_ = l_Lean_Expr_app___override(v___x_808_, v___x_809_);
lean_inc_ref_n(v_a_773_, 4);
v___x_811_ = l_Lean_Expr_app___override(v___x_810_, v_a_773_);
lean_inc(v_lhs_795_);
v___x_812_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_lhs_795_);
lean_inc_ref_n(v___x_812_, 2);
lean_inc_ref(v___x_811_);
v___x_813_ = l_Lean_Expr_app___override(v___x_811_, v___x_812_);
v___x_814_ = l_Lean_Expr_app___override(v___x_803_, v___x_813_);
lean_inc(v_rhs_796_);
v___x_815_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_rhs_796_);
lean_inc_ref_n(v___x_815_, 2);
v___x_816_ = l_Lean_Expr_app___override(v___x_811_, v___x_815_);
v___x_817_ = l_Lean_Expr_app___override(v___x_814_, v___x_816_);
v___x_818_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9);
v___x_819_ = l_Lean_Expr_app___override(v___x_818_, v___x_817_);
v___x_820_ = l_Lean_Level_succ___override(v_u_769_);
v___x_821_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_821_, 0, v___x_820_);
lean_ctor_set(v___x_821_, 1, v___x_802_);
v___x_822_ = l_Lean_Expr_const___override(v___x_801_, v___x_821_);
v___x_823_ = l_Lean_Expr_app___override(v___x_822_, v_type_770_);
v___x_824_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_824_, 0, v___x_812_);
lean_ctor_set(v___x_824_, 1, v___x_802_);
v___x_825_ = lean_array_mk(v___x_824_);
v___x_826_ = l_Lean_Expr_betaRev(v_a_773_, v___x_825_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_825_);
v___x_827_ = l_Lean_Expr_app___override(v___x_823_, v___x_826_);
v___x_828_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_828_, 0, v___x_815_);
lean_ctor_set(v___x_828_, 1, v___x_802_);
v___x_829_ = lean_array_mk(v___x_828_);
v___x_830_ = l_Lean_Expr_betaRev(v_a_773_, v___x_829_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_829_);
v___x_831_ = l_Lean_Expr_app___override(v___x_827_, v___x_830_);
v___x_832_ = l_Lean_Expr_app___override(v___x_819_, v___x_831_);
v___x_833_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__11));
v___x_834_ = l_Lean_Expr_const___override(v___x_833_, v___x_805_);
v___x_835_ = l_Lean_Expr_app___override(v___x_834_, v_type_770_);
v___x_836_ = l_Lean_Expr_app___override(v___x_835_, v_inst_771_);
v___x_837_ = l_Lean_Expr_app___override(v___x_836_, v___x_809_);
v___x_838_ = l_Lean_Expr_app___override(v___x_837_, v_a_773_);
v___x_839_ = l_Lean_Expr_app___override(v___x_838_, v___x_812_);
v___x_840_ = l_Lean_Expr_app___override(v___x_839_, v___x_815_);
v___x_841_ = l_Lean_Expr_app___override(v___x_832_, v___x_840_);
v___x_842_ = l_Lean_Expr_app___override(v___x_841_, v_proof_797_);
if (v_isShared_800_ == 0)
{
lean_ctor_set(v___x_799_, 2, v___x_842_);
v___x_844_ = v___x_799_;
goto v_reusejp_843_;
}
else
{
lean_object* v_reuseFailAlloc_852_; 
v_reuseFailAlloc_852_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_852_, 0, v_lhs_795_);
lean_ctor_set(v_reuseFailAlloc_852_, 1, v_rhs_796_);
lean_ctor_set(v_reuseFailAlloc_852_, 2, v___x_842_);
v___x_844_ = v_reuseFailAlloc_852_;
goto v_reusejp_843_;
}
v_reusejp_843_:
{
lean_object* v___x_845_; lean_object* v___x_847_; 
v___x_845_ = lean_array_push(v_snd_787_, v___x_844_);
if (v_isShared_790_ == 0)
{
lean_ctor_set(v___x_789_, 1, v___x_845_);
v___x_847_ = v___x_789_;
goto v_reusejp_846_;
}
else
{
lean_object* v_reuseFailAlloc_851_; 
v_reuseFailAlloc_851_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_851_, 0, v_fst_786_);
lean_ctor_set(v_reuseFailAlloc_851_, 1, v___x_845_);
v___x_847_ = v_reuseFailAlloc_851_;
goto v_reusejp_846_;
}
v_reusejp_846_:
{
lean_object* v___x_849_; 
if (v_isShared_794_ == 0)
{
lean_ctor_set(v___x_793_, 1, v___x_847_);
v___x_849_ = v___x_793_;
goto v_reusejp_848_;
}
else
{
lean_object* v_reuseFailAlloc_850_; 
v_reuseFailAlloc_850_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_850_, 0, v_fst_785_);
lean_ctor_set(v_reuseFailAlloc_850_, 1, v___x_847_);
v___x_849_ = v_reuseFailAlloc_850_;
goto v_reusejp_848_;
}
v_reusejp_848_:
{
v___y_779_ = v___x_849_;
goto v___jp_778_;
}
}
}
}
}
}
case 1:
{
lean_object* v___x_858_; uint8_t v_isShared_859_; uint8_t v_isSharedCheck_919_; 
lean_inc(v_fst_785_);
v_isSharedCheck_919_ = !lean_is_exclusive(v_b_777_);
if (v_isSharedCheck_919_ == 0)
{
lean_object* v_unused_920_; lean_object* v_unused_921_; 
v_unused_920_ = lean_ctor_get(v_b_777_, 1);
lean_dec(v_unused_920_);
v_unused_921_ = lean_ctor_get(v_b_777_, 0);
lean_dec(v_unused_921_);
v___x_858_ = v_b_777_;
v_isShared_859_ = v_isSharedCheck_919_;
goto v_resetjp_857_;
}
else
{
lean_dec(v_b_777_);
v___x_858_ = lean_box(0);
v_isShared_859_ = v_isSharedCheck_919_;
goto v_resetjp_857_;
}
v_resetjp_857_:
{
lean_object* v_lhs_860_; lean_object* v_rhs_861_; lean_object* v_proof_862_; lean_object* v___x_864_; uint8_t v_isShared_865_; uint8_t v_isSharedCheck_918_; 
v_lhs_860_ = lean_ctor_get(v___x_791_, 0);
v_rhs_861_ = lean_ctor_get(v___x_791_, 1);
v_proof_862_ = lean_ctor_get(v___x_791_, 2);
v_isSharedCheck_918_ = !lean_is_exclusive(v___x_791_);
if (v_isSharedCheck_918_ == 0)
{
v___x_864_ = v___x_791_;
v_isShared_865_ = v_isSharedCheck_918_;
goto v_resetjp_863_;
}
else
{
lean_inc(v_proof_862_);
lean_inc(v_rhs_861_);
lean_inc(v_lhs_860_);
lean_dec(v___x_791_);
v___x_864_ = lean_box(0);
v_isShared_865_ = v_isSharedCheck_918_;
goto v_resetjp_863_;
}
v_resetjp_863_:
{
lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_909_; 
v___x_866_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__13));
v___x_867_ = lean_box(0);
v___x_868_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__15, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__15_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__15);
v___x_869_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5));
lean_inc_n(v_u_769_, 2);
v___x_870_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_870_, 0, v_u_769_);
lean_ctor_set(v___x_870_, 1, v___x_867_);
lean_inc_ref(v___x_870_);
v___x_871_ = l_Lean_Expr_const___override(v___x_869_, v___x_870_);
lean_inc_ref_n(v_type_770_, 3);
v___x_872_ = l_Lean_Expr_app___override(v___x_871_, v_type_770_);
lean_inc_ref_n(v_inst_771_, 2);
v___x_873_ = l_Lean_Expr_app___override(v___x_872_, v_inst_771_);
lean_inc_n(v___x_772_, 3);
v___x_874_ = lp_mathlib_Qq_mkNatLitQ(v___x_772_);
lean_inc_ref(v___x_874_);
v___x_875_ = l_Lean_Expr_app___override(v___x_873_, v___x_874_);
lean_inc_ref_n(v_a_773_, 4);
v___x_876_ = l_Lean_Expr_app___override(v___x_875_, v_a_773_);
lean_inc(v_lhs_860_);
v___x_877_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_lhs_860_);
lean_inc_ref_n(v___x_877_, 2);
lean_inc_ref(v___x_876_);
v___x_878_ = l_Lean_Expr_app___override(v___x_876_, v___x_877_);
v___x_879_ = l_Lean_Expr_app___override(v___x_868_, v___x_878_);
lean_inc(v_rhs_861_);
v___x_880_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_rhs_861_);
lean_inc_ref_n(v___x_880_, 2);
v___x_881_ = l_Lean_Expr_app___override(v___x_876_, v___x_880_);
v___x_882_ = l_Lean_Expr_app___override(v___x_879_, v___x_881_);
v___x_883_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9);
v___x_884_ = l_Lean_Expr_app___override(v___x_883_, v___x_882_);
v___x_885_ = l_Lean_Level_succ___override(v_u_769_);
v___x_886_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_886_, 0, v___x_885_);
lean_ctor_set(v___x_886_, 1, v___x_867_);
v___x_887_ = l_Lean_Expr_const___override(v___x_866_, v___x_886_);
v___x_888_ = l_Lean_Expr_app___override(v___x_887_, v_type_770_);
v___x_889_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_889_, 0, v___x_877_);
lean_ctor_set(v___x_889_, 1, v___x_867_);
v___x_890_ = lean_array_mk(v___x_889_);
v___x_891_ = l_Lean_Expr_betaRev(v_a_773_, v___x_890_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_890_);
v___x_892_ = l_Lean_Expr_app___override(v___x_888_, v___x_891_);
v___x_893_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_893_, 0, v___x_880_);
lean_ctor_set(v___x_893_, 1, v___x_867_);
v___x_894_ = lean_array_mk(v___x_893_);
v___x_895_ = l_Lean_Expr_betaRev(v_a_773_, v___x_894_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_894_);
v___x_896_ = l_Lean_Expr_app___override(v___x_892_, v___x_895_);
v___x_897_ = l_Lean_Expr_app___override(v___x_884_, v___x_896_);
v___x_898_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__17));
v___x_899_ = l_Lean_Expr_const___override(v___x_898_, v___x_870_);
v___x_900_ = l_Lean_Expr_app___override(v___x_899_, v_type_770_);
v___x_901_ = l_Lean_Expr_app___override(v___x_900_, v_inst_771_);
v___x_902_ = l_Lean_Expr_app___override(v___x_901_, v___x_874_);
v___x_903_ = l_Lean_Expr_app___override(v___x_902_, v_a_773_);
v___x_904_ = l_Lean_Expr_app___override(v___x_903_, v___x_877_);
v___x_905_ = l_Lean_Expr_app___override(v___x_904_, v___x_880_);
v___x_906_ = l_Lean_Expr_app___override(v___x_897_, v___x_905_);
v___x_907_ = l_Lean_Expr_app___override(v___x_906_, v_proof_862_);
if (v_isShared_865_ == 0)
{
lean_ctor_set(v___x_864_, 2, v___x_907_);
v___x_909_ = v___x_864_;
goto v_reusejp_908_;
}
else
{
lean_object* v_reuseFailAlloc_917_; 
v_reuseFailAlloc_917_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_917_, 0, v_lhs_860_);
lean_ctor_set(v_reuseFailAlloc_917_, 1, v_rhs_861_);
lean_ctor_set(v_reuseFailAlloc_917_, 2, v___x_907_);
v___x_909_ = v_reuseFailAlloc_917_;
goto v_reusejp_908_;
}
v_reusejp_908_:
{
lean_object* v___x_910_; lean_object* v___x_912_; 
v___x_910_ = lean_array_push(v_snd_787_, v___x_909_);
if (v_isShared_790_ == 0)
{
lean_ctor_set(v___x_789_, 1, v___x_910_);
v___x_912_ = v___x_789_;
goto v_reusejp_911_;
}
else
{
lean_object* v_reuseFailAlloc_916_; 
v_reuseFailAlloc_916_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_916_, 0, v_fst_786_);
lean_ctor_set(v_reuseFailAlloc_916_, 1, v___x_910_);
v___x_912_ = v_reuseFailAlloc_916_;
goto v_reusejp_911_;
}
v_reusejp_911_:
{
lean_object* v___x_914_; 
if (v_isShared_859_ == 0)
{
lean_ctor_set(v___x_858_, 1, v___x_912_);
v___x_914_ = v___x_858_;
goto v_reusejp_913_;
}
else
{
lean_object* v_reuseFailAlloc_915_; 
v_reuseFailAlloc_915_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_915_, 0, v_fst_785_);
lean_ctor_set(v_reuseFailAlloc_915_, 1, v___x_912_);
v___x_914_ = v_reuseFailAlloc_915_;
goto v_reusejp_913_;
}
v_reusejp_913_:
{
v___y_779_ = v___x_914_;
goto v___jp_778_;
}
}
}
}
}
}
case 2:
{
lean_object* v___x_923_; uint8_t v_isShared_924_; uint8_t v_isSharedCheck_1007_; 
lean_inc(v_fst_785_);
v_isSharedCheck_1007_ = !lean_is_exclusive(v_b_777_);
if (v_isSharedCheck_1007_ == 0)
{
lean_object* v_unused_1008_; lean_object* v_unused_1009_; 
v_unused_1008_ = lean_ctor_get(v_b_777_, 1);
lean_dec(v_unused_1008_);
v_unused_1009_ = lean_ctor_get(v_b_777_, 0);
lean_dec(v_unused_1009_);
v___x_923_ = v_b_777_;
v_isShared_924_ = v_isSharedCheck_1007_;
goto v_resetjp_922_;
}
else
{
lean_dec(v_b_777_);
v___x_923_ = lean_box(0);
v_isShared_924_ = v_isSharedCheck_1007_;
goto v_resetjp_922_;
}
v_resetjp_922_:
{
lean_object* v_lhs_925_; lean_object* v_rhs_926_; lean_object* v_proof_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_1006_; 
v_lhs_925_ = lean_ctor_get(v___x_791_, 0);
v_rhs_926_ = lean_ctor_get(v___x_791_, 1);
v_proof_927_ = lean_ctor_get(v___x_791_, 2);
v_isSharedCheck_1006_ = !lean_is_exclusive(v___x_791_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_929_ = v___x_791_;
v_isShared_930_ = v_isSharedCheck_1006_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_proof_927_);
lean_inc(v_rhs_926_);
lean_inc(v_lhs_925_);
lean_dec(v___x_791_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_1006_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_997_; 
v___x_931_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__20));
v___x_932_ = lean_box(0);
v___x_933_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__26, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__26_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__26);
v___x_934_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5));
lean_inc(v_u_769_);
v___x_935_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_935_, 0, v_u_769_);
lean_ctor_set(v___x_935_, 1, v___x_932_);
lean_inc_ref_n(v___x_935_, 8);
v___x_936_ = l_Lean_Expr_const___override(v___x_934_, v___x_935_);
lean_inc_ref_n(v_type_770_, 9);
v___x_937_ = l_Lean_Expr_app___override(v___x_936_, v_type_770_);
lean_inc_ref_n(v_inst_771_, 3);
v___x_938_ = l_Lean_Expr_app___override(v___x_937_, v_inst_771_);
lean_inc_n(v___x_772_, 3);
v___x_939_ = lp_mathlib_Qq_mkNatLitQ(v___x_772_);
lean_inc_ref(v___x_939_);
v___x_940_ = l_Lean_Expr_app___override(v___x_938_, v___x_939_);
lean_inc_ref_n(v_a_773_, 4);
v___x_941_ = l_Lean_Expr_app___override(v___x_940_, v_a_773_);
lean_inc(v_lhs_925_);
v___x_942_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_lhs_925_);
lean_inc_ref_n(v___x_942_, 2);
lean_inc_ref(v___x_941_);
v___x_943_ = l_Lean_Expr_app___override(v___x_941_, v___x_942_);
v___x_944_ = l_Lean_Expr_app___override(v___x_933_, v___x_943_);
lean_inc(v_rhs_926_);
v___x_945_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_rhs_926_);
lean_inc_ref_n(v___x_945_, 2);
v___x_946_ = l_Lean_Expr_app___override(v___x_941_, v___x_945_);
v___x_947_ = l_Lean_Expr_app___override(v___x_944_, v___x_946_);
v___x_948_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9);
v___x_949_ = l_Lean_Expr_app___override(v___x_948_, v___x_947_);
v___x_950_ = l_Lean_Expr_const___override(v___x_931_, v___x_935_);
v___x_951_ = l_Lean_Expr_app___override(v___x_950_, v_type_770_);
v___x_952_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__29));
v___x_953_ = l_Lean_Expr_const___override(v___x_952_, v___x_935_);
v___x_954_ = l_Lean_Expr_app___override(v___x_953_, v_type_770_);
v___x_955_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__32));
v___x_956_ = l_Lean_Expr_const___override(v___x_955_, v___x_935_);
v___x_957_ = l_Lean_Expr_app___override(v___x_956_, v_type_770_);
v___x_958_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__35));
v___x_959_ = l_Lean_Expr_const___override(v___x_958_, v___x_935_);
v___x_960_ = l_Lean_Expr_app___override(v___x_959_, v_type_770_);
v___x_961_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__38));
v___x_962_ = l_Lean_Expr_const___override(v___x_961_, v___x_935_);
v___x_963_ = l_Lean_Expr_app___override(v___x_962_, v_type_770_);
v___x_964_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41));
v___x_965_ = l_Lean_Expr_const___override(v___x_964_, v___x_935_);
v___x_966_ = l_Lean_Expr_app___override(v___x_965_, v_type_770_);
v___x_967_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__43));
v___x_968_ = l_Lean_Expr_const___override(v___x_967_, v___x_935_);
v___x_969_ = l_Lean_Expr_app___override(v___x_968_, v_type_770_);
v___x_970_ = l_Lean_Expr_app___override(v___x_969_, v_inst_771_);
v___x_971_ = l_Lean_Expr_app___override(v___x_966_, v___x_970_);
v___x_972_ = l_Lean_Expr_app___override(v___x_963_, v___x_971_);
v___x_973_ = l_Lean_Expr_app___override(v___x_960_, v___x_972_);
v___x_974_ = l_Lean_Expr_app___override(v___x_957_, v___x_973_);
v___x_975_ = l_Lean_Expr_app___override(v___x_954_, v___x_974_);
v___x_976_ = l_Lean_Expr_app___override(v___x_951_, v___x_975_);
v___x_977_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_977_, 0, v___x_942_);
lean_ctor_set(v___x_977_, 1, v___x_932_);
v___x_978_ = lean_array_mk(v___x_977_);
v___x_979_ = l_Lean_Expr_betaRev(v_a_773_, v___x_978_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_978_);
v___x_980_ = l_Lean_Expr_app___override(v___x_976_, v___x_979_);
v___x_981_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_981_, 0, v___x_945_);
lean_ctor_set(v___x_981_, 1, v___x_932_);
v___x_982_ = lean_array_mk(v___x_981_);
v___x_983_ = l_Lean_Expr_betaRev(v_a_773_, v___x_982_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_982_);
v___x_984_ = l_Lean_Expr_app___override(v___x_980_, v___x_983_);
v___x_985_ = l_Lean_Expr_app___override(v___x_949_, v___x_984_);
v___x_986_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__45));
v___x_987_ = l_Lean_Expr_const___override(v___x_986_, v___x_935_);
v___x_988_ = l_Lean_Expr_app___override(v___x_987_, v_type_770_);
v___x_989_ = l_Lean_Expr_app___override(v___x_988_, v_inst_771_);
v___x_990_ = l_Lean_Expr_app___override(v___x_989_, v___x_939_);
v___x_991_ = l_Lean_Expr_app___override(v___x_990_, v_a_773_);
v___x_992_ = l_Lean_Expr_app___override(v___x_991_, v___x_942_);
v___x_993_ = l_Lean_Expr_app___override(v___x_992_, v___x_945_);
v___x_994_ = l_Lean_Expr_app___override(v___x_985_, v___x_993_);
v___x_995_ = l_Lean_Expr_app___override(v___x_994_, v_proof_927_);
if (v_isShared_930_ == 0)
{
lean_ctor_set(v___x_929_, 2, v___x_995_);
v___x_997_ = v___x_929_;
goto v_reusejp_996_;
}
else
{
lean_object* v_reuseFailAlloc_1005_; 
v_reuseFailAlloc_1005_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1005_, 0, v_lhs_925_);
lean_ctor_set(v_reuseFailAlloc_1005_, 1, v_rhs_926_);
lean_ctor_set(v_reuseFailAlloc_1005_, 2, v___x_995_);
v___x_997_ = v_reuseFailAlloc_1005_;
goto v_reusejp_996_;
}
v_reusejp_996_:
{
lean_object* v___x_998_; lean_object* v___x_1000_; 
v___x_998_ = lean_array_push(v_snd_787_, v___x_997_);
if (v_isShared_790_ == 0)
{
lean_ctor_set(v___x_789_, 1, v___x_998_);
v___x_1000_ = v___x_789_;
goto v_reusejp_999_;
}
else
{
lean_object* v_reuseFailAlloc_1004_; 
v_reuseFailAlloc_1004_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1004_, 0, v_fst_786_);
lean_ctor_set(v_reuseFailAlloc_1004_, 1, v___x_998_);
v___x_1000_ = v_reuseFailAlloc_1004_;
goto v_reusejp_999_;
}
v_reusejp_999_:
{
lean_object* v___x_1002_; 
if (v_isShared_924_ == 0)
{
lean_ctor_set(v___x_923_, 1, v___x_1000_);
v___x_1002_ = v___x_923_;
goto v_reusejp_1001_;
}
else
{
lean_object* v_reuseFailAlloc_1003_; 
v_reuseFailAlloc_1003_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1003_, 0, v_fst_785_);
lean_ctor_set(v_reuseFailAlloc_1003_, 1, v___x_1000_);
v___x_1002_ = v_reuseFailAlloc_1003_;
goto v_reusejp_1001_;
}
v_reusejp_1001_:
{
v___y_779_ = v___x_1002_;
goto v___jp_778_;
}
}
}
}
}
}
case 3:
{
lean_object* v___x_1011_; uint8_t v_isShared_1012_; uint8_t v_isSharedCheck_1098_; 
lean_inc(v_fst_785_);
v_isSharedCheck_1098_ = !lean_is_exclusive(v_b_777_);
if (v_isSharedCheck_1098_ == 0)
{
lean_object* v_unused_1099_; lean_object* v_unused_1100_; 
v_unused_1099_ = lean_ctor_get(v_b_777_, 1);
lean_dec(v_unused_1099_);
v_unused_1100_ = lean_ctor_get(v_b_777_, 0);
lean_dec(v_unused_1100_);
v___x_1011_ = v_b_777_;
v_isShared_1012_ = v_isSharedCheck_1098_;
goto v_resetjp_1010_;
}
else
{
lean_dec(v_b_777_);
v___x_1011_ = lean_box(0);
v_isShared_1012_ = v_isSharedCheck_1098_;
goto v_resetjp_1010_;
}
v_resetjp_1010_:
{
lean_object* v_lhs_1013_; lean_object* v_rhs_1014_; lean_object* v_proof_1015_; lean_object* v___x_1017_; uint8_t v_isShared_1018_; uint8_t v_isSharedCheck_1097_; 
v_lhs_1013_ = lean_ctor_get(v___x_791_, 0);
v_rhs_1014_ = lean_ctor_get(v___x_791_, 1);
v_proof_1015_ = lean_ctor_get(v___x_791_, 2);
v_isSharedCheck_1097_ = !lean_is_exclusive(v___x_791_);
if (v_isSharedCheck_1097_ == 0)
{
v___x_1017_ = v___x_791_;
v_isShared_1018_ = v_isSharedCheck_1097_;
goto v_resetjp_1016_;
}
else
{
lean_inc(v_proof_1015_);
lean_inc(v_rhs_1014_);
lean_inc(v_lhs_1013_);
lean_dec(v___x_791_);
v___x_1017_ = lean_box(0);
v_isShared_1018_ = v_isSharedCheck_1097_;
goto v_resetjp_1016_;
}
v_resetjp_1016_:
{
lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1088_; 
v___x_1019_ = lean_box(0);
v___x_1020_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__48, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__48_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__48);
v___x_1021_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__20));
v___x_1022_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__26, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__26_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__26);
v___x_1023_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5));
lean_inc(v_u_769_);
v___x_1024_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1024_, 0, v_u_769_);
lean_ctor_set(v___x_1024_, 1, v___x_1019_);
lean_inc_ref_n(v___x_1024_, 8);
v___x_1025_ = l_Lean_Expr_const___override(v___x_1023_, v___x_1024_);
lean_inc_ref_n(v_type_770_, 9);
v___x_1026_ = l_Lean_Expr_app___override(v___x_1025_, v_type_770_);
lean_inc_ref_n(v_inst_771_, 3);
v___x_1027_ = l_Lean_Expr_app___override(v___x_1026_, v_inst_771_);
lean_inc_n(v___x_772_, 3);
v___x_1028_ = lp_mathlib_Qq_mkNatLitQ(v___x_772_);
lean_inc_ref(v___x_1028_);
v___x_1029_ = l_Lean_Expr_app___override(v___x_1027_, v___x_1028_);
lean_inc_ref_n(v_a_773_, 4);
v___x_1030_ = l_Lean_Expr_app___override(v___x_1029_, v_a_773_);
lean_inc(v_lhs_1013_);
v___x_1031_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_lhs_1013_);
lean_inc_ref_n(v___x_1031_, 2);
lean_inc_ref(v___x_1030_);
v___x_1032_ = l_Lean_Expr_app___override(v___x_1030_, v___x_1031_);
v___x_1033_ = l_Lean_Expr_app___override(v___x_1022_, v___x_1032_);
lean_inc(v_rhs_1014_);
v___x_1034_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_rhs_1014_);
lean_inc_ref_n(v___x_1034_, 2);
v___x_1035_ = l_Lean_Expr_app___override(v___x_1030_, v___x_1034_);
v___x_1036_ = l_Lean_Expr_app___override(v___x_1033_, v___x_1035_);
v___x_1037_ = l_Lean_Expr_app___override(v___x_1020_, v___x_1036_);
v___x_1038_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9);
v___x_1039_ = l_Lean_Expr_app___override(v___x_1038_, v___x_1037_);
v___x_1040_ = l_Lean_Expr_const___override(v___x_1021_, v___x_1024_);
v___x_1041_ = l_Lean_Expr_app___override(v___x_1040_, v_type_770_);
v___x_1042_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__29));
v___x_1043_ = l_Lean_Expr_const___override(v___x_1042_, v___x_1024_);
v___x_1044_ = l_Lean_Expr_app___override(v___x_1043_, v_type_770_);
v___x_1045_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__32));
v___x_1046_ = l_Lean_Expr_const___override(v___x_1045_, v___x_1024_);
v___x_1047_ = l_Lean_Expr_app___override(v___x_1046_, v_type_770_);
v___x_1048_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__35));
v___x_1049_ = l_Lean_Expr_const___override(v___x_1048_, v___x_1024_);
v___x_1050_ = l_Lean_Expr_app___override(v___x_1049_, v_type_770_);
v___x_1051_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__38));
v___x_1052_ = l_Lean_Expr_const___override(v___x_1051_, v___x_1024_);
v___x_1053_ = l_Lean_Expr_app___override(v___x_1052_, v_type_770_);
v___x_1054_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41));
v___x_1055_ = l_Lean_Expr_const___override(v___x_1054_, v___x_1024_);
v___x_1056_ = l_Lean_Expr_app___override(v___x_1055_, v_type_770_);
v___x_1057_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__43));
v___x_1058_ = l_Lean_Expr_const___override(v___x_1057_, v___x_1024_);
v___x_1059_ = l_Lean_Expr_app___override(v___x_1058_, v_type_770_);
v___x_1060_ = l_Lean_Expr_app___override(v___x_1059_, v_inst_771_);
v___x_1061_ = l_Lean_Expr_app___override(v___x_1056_, v___x_1060_);
v___x_1062_ = l_Lean_Expr_app___override(v___x_1053_, v___x_1061_);
v___x_1063_ = l_Lean_Expr_app___override(v___x_1050_, v___x_1062_);
v___x_1064_ = l_Lean_Expr_app___override(v___x_1047_, v___x_1063_);
v___x_1065_ = l_Lean_Expr_app___override(v___x_1044_, v___x_1064_);
v___x_1066_ = l_Lean_Expr_app___override(v___x_1041_, v___x_1065_);
v___x_1067_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1067_, 0, v___x_1031_);
lean_ctor_set(v___x_1067_, 1, v___x_1019_);
v___x_1068_ = lean_array_mk(v___x_1067_);
v___x_1069_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1068_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1068_);
v___x_1070_ = l_Lean_Expr_app___override(v___x_1066_, v___x_1069_);
v___x_1071_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1071_, 0, v___x_1034_);
lean_ctor_set(v___x_1071_, 1, v___x_1019_);
v___x_1072_ = lean_array_mk(v___x_1071_);
v___x_1073_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1072_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1072_);
v___x_1074_ = l_Lean_Expr_app___override(v___x_1070_, v___x_1073_);
v___x_1075_ = l_Lean_Expr_app___override(v___x_1020_, v___x_1074_);
v___x_1076_ = l_Lean_Expr_app___override(v___x_1039_, v___x_1075_);
v___x_1077_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__50));
v___x_1078_ = l_Lean_Expr_const___override(v___x_1077_, v___x_1024_);
v___x_1079_ = l_Lean_Expr_app___override(v___x_1078_, v_type_770_);
v___x_1080_ = l_Lean_Expr_app___override(v___x_1079_, v_inst_771_);
v___x_1081_ = l_Lean_Expr_app___override(v___x_1080_, v___x_1028_);
v___x_1082_ = l_Lean_Expr_app___override(v___x_1081_, v_a_773_);
v___x_1083_ = l_Lean_Expr_app___override(v___x_1082_, v___x_1031_);
v___x_1084_ = l_Lean_Expr_app___override(v___x_1083_, v___x_1034_);
v___x_1085_ = l_Lean_Expr_app___override(v___x_1076_, v___x_1084_);
v___x_1086_ = l_Lean_Expr_app___override(v___x_1085_, v_proof_1015_);
if (v_isShared_1018_ == 0)
{
lean_ctor_set(v___x_1017_, 2, v___x_1086_);
v___x_1088_ = v___x_1017_;
goto v_reusejp_1087_;
}
else
{
lean_object* v_reuseFailAlloc_1096_; 
v_reuseFailAlloc_1096_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1096_, 0, v_lhs_1013_);
lean_ctor_set(v_reuseFailAlloc_1096_, 1, v_rhs_1014_);
lean_ctor_set(v_reuseFailAlloc_1096_, 2, v___x_1086_);
v___x_1088_ = v_reuseFailAlloc_1096_;
goto v_reusejp_1087_;
}
v_reusejp_1087_:
{
lean_object* v___x_1089_; lean_object* v___x_1091_; 
v___x_1089_ = lean_array_push(v_snd_787_, v___x_1088_);
if (v_isShared_790_ == 0)
{
lean_ctor_set(v___x_789_, 1, v___x_1089_);
v___x_1091_ = v___x_789_;
goto v_reusejp_1090_;
}
else
{
lean_object* v_reuseFailAlloc_1095_; 
v_reuseFailAlloc_1095_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1095_, 0, v_fst_786_);
lean_ctor_set(v_reuseFailAlloc_1095_, 1, v___x_1089_);
v___x_1091_ = v_reuseFailAlloc_1095_;
goto v_reusejp_1090_;
}
v_reusejp_1090_:
{
lean_object* v___x_1093_; 
if (v_isShared_1012_ == 0)
{
lean_ctor_set(v___x_1011_, 1, v___x_1091_);
v___x_1093_ = v___x_1011_;
goto v_reusejp_1092_;
}
else
{
lean_object* v_reuseFailAlloc_1094_; 
v_reuseFailAlloc_1094_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1094_, 0, v_fst_785_);
lean_ctor_set(v_reuseFailAlloc_1094_, 1, v___x_1091_);
v___x_1093_ = v_reuseFailAlloc_1094_;
goto v_reusejp_1092_;
}
v_reusejp_1092_:
{
v___y_779_ = v___x_1093_;
goto v___jp_778_;
}
}
}
}
}
}
case 4:
{
lean_object* v___x_1102_; uint8_t v_isShared_1103_; uint8_t v_isSharedCheck_1186_; 
lean_inc(v_fst_785_);
v_isSharedCheck_1186_ = !lean_is_exclusive(v_b_777_);
if (v_isSharedCheck_1186_ == 0)
{
lean_object* v_unused_1187_; lean_object* v_unused_1188_; 
v_unused_1187_ = lean_ctor_get(v_b_777_, 1);
lean_dec(v_unused_1187_);
v_unused_1188_ = lean_ctor_get(v_b_777_, 0);
lean_dec(v_unused_1188_);
v___x_1102_ = v_b_777_;
v_isShared_1103_ = v_isSharedCheck_1186_;
goto v_resetjp_1101_;
}
else
{
lean_dec(v_b_777_);
v___x_1102_ = lean_box(0);
v_isShared_1103_ = v_isSharedCheck_1186_;
goto v_resetjp_1101_;
}
v_resetjp_1101_:
{
lean_object* v_lhs_1104_; lean_object* v_rhs_1105_; lean_object* v_proof_1106_; lean_object* v___x_1108_; uint8_t v_isShared_1109_; uint8_t v_isSharedCheck_1185_; 
v_lhs_1104_ = lean_ctor_get(v___x_791_, 0);
v_rhs_1105_ = lean_ctor_get(v___x_791_, 1);
v_proof_1106_ = lean_ctor_get(v___x_791_, 2);
v_isSharedCheck_1185_ = !lean_is_exclusive(v___x_791_);
if (v_isSharedCheck_1185_ == 0)
{
v___x_1108_ = v___x_791_;
v_isShared_1109_ = v_isSharedCheck_1185_;
goto v_resetjp_1107_;
}
else
{
lean_inc(v_proof_1106_);
lean_inc(v_rhs_1105_);
lean_inc(v_lhs_1104_);
lean_dec(v___x_791_);
v___x_1108_ = lean_box(0);
v_isShared_1109_ = v_isSharedCheck_1185_;
goto v_resetjp_1107_;
}
v_resetjp_1107_:
{
lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1176_; 
v___x_1110_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__14));
v___x_1111_ = lean_box(0);
v___x_1112_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__55, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__55_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__55);
v___x_1113_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5));
lean_inc(v_u_769_);
v___x_1114_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1114_, 0, v_u_769_);
lean_ctor_set(v___x_1114_, 1, v___x_1111_);
lean_inc_ref_n(v___x_1114_, 8);
v___x_1115_ = l_Lean_Expr_const___override(v___x_1113_, v___x_1114_);
lean_inc_ref_n(v_type_770_, 9);
v___x_1116_ = l_Lean_Expr_app___override(v___x_1115_, v_type_770_);
lean_inc_ref_n(v_inst_771_, 3);
v___x_1117_ = l_Lean_Expr_app___override(v___x_1116_, v_inst_771_);
lean_inc_n(v___x_772_, 3);
v___x_1118_ = lp_mathlib_Qq_mkNatLitQ(v___x_772_);
lean_inc_ref(v___x_1118_);
v___x_1119_ = l_Lean_Expr_app___override(v___x_1117_, v___x_1118_);
lean_inc_ref_n(v_a_773_, 4);
v___x_1120_ = l_Lean_Expr_app___override(v___x_1119_, v_a_773_);
lean_inc(v_lhs_1104_);
v___x_1121_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_lhs_1104_);
lean_inc_ref_n(v___x_1121_, 2);
lean_inc_ref(v___x_1120_);
v___x_1122_ = l_Lean_Expr_app___override(v___x_1120_, v___x_1121_);
v___x_1123_ = l_Lean_Expr_app___override(v___x_1112_, v___x_1122_);
lean_inc(v_rhs_1105_);
v___x_1124_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_rhs_1105_);
lean_inc_ref_n(v___x_1124_, 2);
v___x_1125_ = l_Lean_Expr_app___override(v___x_1120_, v___x_1124_);
v___x_1126_ = l_Lean_Expr_app___override(v___x_1123_, v___x_1125_);
v___x_1127_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9);
v___x_1128_ = l_Lean_Expr_app___override(v___x_1127_, v___x_1126_);
v___x_1129_ = l_Lean_Expr_const___override(v___x_1110_, v___x_1114_);
v___x_1130_ = l_Lean_Expr_app___override(v___x_1129_, v_type_770_);
v___x_1131_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__57));
v___x_1132_ = l_Lean_Expr_const___override(v___x_1131_, v___x_1114_);
v___x_1133_ = l_Lean_Expr_app___override(v___x_1132_, v_type_770_);
v___x_1134_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__32));
v___x_1135_ = l_Lean_Expr_const___override(v___x_1134_, v___x_1114_);
v___x_1136_ = l_Lean_Expr_app___override(v___x_1135_, v_type_770_);
v___x_1137_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__35));
v___x_1138_ = l_Lean_Expr_const___override(v___x_1137_, v___x_1114_);
v___x_1139_ = l_Lean_Expr_app___override(v___x_1138_, v_type_770_);
v___x_1140_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__38));
v___x_1141_ = l_Lean_Expr_const___override(v___x_1140_, v___x_1114_);
v___x_1142_ = l_Lean_Expr_app___override(v___x_1141_, v_type_770_);
v___x_1143_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41));
v___x_1144_ = l_Lean_Expr_const___override(v___x_1143_, v___x_1114_);
v___x_1145_ = l_Lean_Expr_app___override(v___x_1144_, v_type_770_);
v___x_1146_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__43));
v___x_1147_ = l_Lean_Expr_const___override(v___x_1146_, v___x_1114_);
v___x_1148_ = l_Lean_Expr_app___override(v___x_1147_, v_type_770_);
v___x_1149_ = l_Lean_Expr_app___override(v___x_1148_, v_inst_771_);
v___x_1150_ = l_Lean_Expr_app___override(v___x_1145_, v___x_1149_);
v___x_1151_ = l_Lean_Expr_app___override(v___x_1142_, v___x_1150_);
v___x_1152_ = l_Lean_Expr_app___override(v___x_1139_, v___x_1151_);
v___x_1153_ = l_Lean_Expr_app___override(v___x_1136_, v___x_1152_);
v___x_1154_ = l_Lean_Expr_app___override(v___x_1133_, v___x_1153_);
v___x_1155_ = l_Lean_Expr_app___override(v___x_1130_, v___x_1154_);
v___x_1156_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1156_, 0, v___x_1121_);
lean_ctor_set(v___x_1156_, 1, v___x_1111_);
v___x_1157_ = lean_array_mk(v___x_1156_);
v___x_1158_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1157_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1157_);
v___x_1159_ = l_Lean_Expr_app___override(v___x_1155_, v___x_1158_);
v___x_1160_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1160_, 0, v___x_1124_);
lean_ctor_set(v___x_1160_, 1, v___x_1111_);
v___x_1161_ = lean_array_mk(v___x_1160_);
v___x_1162_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1161_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1161_);
v___x_1163_ = l_Lean_Expr_app___override(v___x_1159_, v___x_1162_);
v___x_1164_ = l_Lean_Expr_app___override(v___x_1128_, v___x_1163_);
v___x_1165_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__59));
v___x_1166_ = l_Lean_Expr_const___override(v___x_1165_, v___x_1114_);
v___x_1167_ = l_Lean_Expr_app___override(v___x_1166_, v_type_770_);
v___x_1168_ = l_Lean_Expr_app___override(v___x_1167_, v_inst_771_);
v___x_1169_ = l_Lean_Expr_app___override(v___x_1168_, v___x_1118_);
v___x_1170_ = l_Lean_Expr_app___override(v___x_1169_, v_a_773_);
v___x_1171_ = l_Lean_Expr_app___override(v___x_1170_, v___x_1121_);
v___x_1172_ = l_Lean_Expr_app___override(v___x_1171_, v___x_1124_);
v___x_1173_ = l_Lean_Expr_app___override(v___x_1164_, v___x_1172_);
v___x_1174_ = l_Lean_Expr_app___override(v___x_1173_, v_proof_1106_);
if (v_isShared_1109_ == 0)
{
lean_ctor_set(v___x_1108_, 2, v___x_1174_);
v___x_1176_ = v___x_1108_;
goto v_reusejp_1175_;
}
else
{
lean_object* v_reuseFailAlloc_1184_; 
v_reuseFailAlloc_1184_ = lean_alloc_ctor(4, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1184_, 0, v_lhs_1104_);
lean_ctor_set(v_reuseFailAlloc_1184_, 1, v_rhs_1105_);
lean_ctor_set(v_reuseFailAlloc_1184_, 2, v___x_1174_);
v___x_1176_ = v_reuseFailAlloc_1184_;
goto v_reusejp_1175_;
}
v_reusejp_1175_:
{
lean_object* v___x_1177_; lean_object* v___x_1179_; 
v___x_1177_ = lean_array_push(v_snd_787_, v___x_1176_);
if (v_isShared_790_ == 0)
{
lean_ctor_set(v___x_789_, 1, v___x_1177_);
v___x_1179_ = v___x_789_;
goto v_reusejp_1178_;
}
else
{
lean_object* v_reuseFailAlloc_1183_; 
v_reuseFailAlloc_1183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1183_, 0, v_fst_786_);
lean_ctor_set(v_reuseFailAlloc_1183_, 1, v___x_1177_);
v___x_1179_ = v_reuseFailAlloc_1183_;
goto v_reusejp_1178_;
}
v_reusejp_1178_:
{
lean_object* v___x_1181_; 
if (v_isShared_1103_ == 0)
{
lean_ctor_set(v___x_1102_, 1, v___x_1179_);
v___x_1181_ = v___x_1102_;
goto v_reusejp_1180_;
}
else
{
lean_object* v_reuseFailAlloc_1182_; 
v_reuseFailAlloc_1182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1182_, 0, v_fst_785_);
lean_ctor_set(v_reuseFailAlloc_1182_, 1, v___x_1179_);
v___x_1181_ = v_reuseFailAlloc_1182_;
goto v_reusejp_1180_;
}
v_reusejp_1180_:
{
v___y_779_ = v___x_1181_;
goto v___jp_778_;
}
}
}
}
}
}
case 5:
{
lean_object* v___x_1190_; uint8_t v_isShared_1191_; uint8_t v_isSharedCheck_1277_; 
lean_inc(v_fst_785_);
v_isSharedCheck_1277_ = !lean_is_exclusive(v_b_777_);
if (v_isSharedCheck_1277_ == 0)
{
lean_object* v_unused_1278_; lean_object* v_unused_1279_; 
v_unused_1278_ = lean_ctor_get(v_b_777_, 1);
lean_dec(v_unused_1278_);
v_unused_1279_ = lean_ctor_get(v_b_777_, 0);
lean_dec(v_unused_1279_);
v___x_1190_ = v_b_777_;
v_isShared_1191_ = v_isSharedCheck_1277_;
goto v_resetjp_1189_;
}
else
{
lean_dec(v_b_777_);
v___x_1190_ = lean_box(0);
v_isShared_1191_ = v_isSharedCheck_1277_;
goto v_resetjp_1189_;
}
v_resetjp_1189_:
{
lean_object* v_lhs_1192_; lean_object* v_rhs_1193_; lean_object* v_proof_1194_; lean_object* v___x_1196_; uint8_t v_isShared_1197_; uint8_t v_isSharedCheck_1276_; 
v_lhs_1192_ = lean_ctor_get(v___x_791_, 0);
v_rhs_1193_ = lean_ctor_get(v___x_791_, 1);
v_proof_1194_ = lean_ctor_get(v___x_791_, 2);
v_isSharedCheck_1276_ = !lean_is_exclusive(v___x_791_);
if (v_isSharedCheck_1276_ == 0)
{
v___x_1196_ = v___x_791_;
v_isShared_1197_ = v_isSharedCheck_1276_;
goto v_resetjp_1195_;
}
else
{
lean_inc(v_proof_1194_);
lean_inc(v_rhs_1193_);
lean_inc(v_lhs_1192_);
lean_dec(v___x_791_);
v___x_1196_ = lean_box(0);
v_isShared_1197_ = v_isSharedCheck_1276_;
goto v_resetjp_1195_;
}
v_resetjp_1195_:
{
lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1267_; 
v___x_1198_ = lean_box(0);
v___x_1199_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__48, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__48_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__48);
v___x_1200_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__14));
v___x_1201_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__55, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__55_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__55);
v___x_1202_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5));
lean_inc(v_u_769_);
v___x_1203_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1203_, 0, v_u_769_);
lean_ctor_set(v___x_1203_, 1, v___x_1198_);
lean_inc_ref_n(v___x_1203_, 8);
v___x_1204_ = l_Lean_Expr_const___override(v___x_1202_, v___x_1203_);
lean_inc_ref_n(v_type_770_, 9);
v___x_1205_ = l_Lean_Expr_app___override(v___x_1204_, v_type_770_);
lean_inc_ref_n(v_inst_771_, 3);
v___x_1206_ = l_Lean_Expr_app___override(v___x_1205_, v_inst_771_);
lean_inc_n(v___x_772_, 3);
v___x_1207_ = lp_mathlib_Qq_mkNatLitQ(v___x_772_);
lean_inc_ref(v___x_1207_);
v___x_1208_ = l_Lean_Expr_app___override(v___x_1206_, v___x_1207_);
lean_inc_ref_n(v_a_773_, 4);
v___x_1209_ = l_Lean_Expr_app___override(v___x_1208_, v_a_773_);
lean_inc(v_lhs_1192_);
v___x_1210_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_lhs_1192_);
lean_inc_ref_n(v___x_1210_, 2);
lean_inc_ref(v___x_1209_);
v___x_1211_ = l_Lean_Expr_app___override(v___x_1209_, v___x_1210_);
v___x_1212_ = l_Lean_Expr_app___override(v___x_1201_, v___x_1211_);
lean_inc(v_rhs_1193_);
v___x_1213_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_rhs_1193_);
lean_inc_ref_n(v___x_1213_, 2);
v___x_1214_ = l_Lean_Expr_app___override(v___x_1209_, v___x_1213_);
v___x_1215_ = l_Lean_Expr_app___override(v___x_1212_, v___x_1214_);
v___x_1216_ = l_Lean_Expr_app___override(v___x_1199_, v___x_1215_);
v___x_1217_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9);
v___x_1218_ = l_Lean_Expr_app___override(v___x_1217_, v___x_1216_);
v___x_1219_ = l_Lean_Expr_const___override(v___x_1200_, v___x_1203_);
v___x_1220_ = l_Lean_Expr_app___override(v___x_1219_, v_type_770_);
v___x_1221_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__57));
v___x_1222_ = l_Lean_Expr_const___override(v___x_1221_, v___x_1203_);
v___x_1223_ = l_Lean_Expr_app___override(v___x_1222_, v_type_770_);
v___x_1224_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__32));
v___x_1225_ = l_Lean_Expr_const___override(v___x_1224_, v___x_1203_);
v___x_1226_ = l_Lean_Expr_app___override(v___x_1225_, v_type_770_);
v___x_1227_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__35));
v___x_1228_ = l_Lean_Expr_const___override(v___x_1227_, v___x_1203_);
v___x_1229_ = l_Lean_Expr_app___override(v___x_1228_, v_type_770_);
v___x_1230_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__38));
v___x_1231_ = l_Lean_Expr_const___override(v___x_1230_, v___x_1203_);
v___x_1232_ = l_Lean_Expr_app___override(v___x_1231_, v_type_770_);
v___x_1233_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41));
v___x_1234_ = l_Lean_Expr_const___override(v___x_1233_, v___x_1203_);
v___x_1235_ = l_Lean_Expr_app___override(v___x_1234_, v_type_770_);
v___x_1236_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__43));
v___x_1237_ = l_Lean_Expr_const___override(v___x_1236_, v___x_1203_);
v___x_1238_ = l_Lean_Expr_app___override(v___x_1237_, v_type_770_);
v___x_1239_ = l_Lean_Expr_app___override(v___x_1238_, v_inst_771_);
v___x_1240_ = l_Lean_Expr_app___override(v___x_1235_, v___x_1239_);
v___x_1241_ = l_Lean_Expr_app___override(v___x_1232_, v___x_1240_);
v___x_1242_ = l_Lean_Expr_app___override(v___x_1229_, v___x_1241_);
v___x_1243_ = l_Lean_Expr_app___override(v___x_1226_, v___x_1242_);
v___x_1244_ = l_Lean_Expr_app___override(v___x_1223_, v___x_1243_);
v___x_1245_ = l_Lean_Expr_app___override(v___x_1220_, v___x_1244_);
v___x_1246_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1246_, 0, v___x_1210_);
lean_ctor_set(v___x_1246_, 1, v___x_1198_);
v___x_1247_ = lean_array_mk(v___x_1246_);
v___x_1248_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1247_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1247_);
v___x_1249_ = l_Lean_Expr_app___override(v___x_1245_, v___x_1248_);
v___x_1250_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1250_, 0, v___x_1213_);
lean_ctor_set(v___x_1250_, 1, v___x_1198_);
v___x_1251_ = lean_array_mk(v___x_1250_);
v___x_1252_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1251_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1251_);
v___x_1253_ = l_Lean_Expr_app___override(v___x_1249_, v___x_1252_);
v___x_1254_ = l_Lean_Expr_app___override(v___x_1199_, v___x_1253_);
v___x_1255_ = l_Lean_Expr_app___override(v___x_1218_, v___x_1254_);
v___x_1256_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__61));
v___x_1257_ = l_Lean_Expr_const___override(v___x_1256_, v___x_1203_);
v___x_1258_ = l_Lean_Expr_app___override(v___x_1257_, v_type_770_);
v___x_1259_ = l_Lean_Expr_app___override(v___x_1258_, v_inst_771_);
v___x_1260_ = l_Lean_Expr_app___override(v___x_1259_, v___x_1207_);
v___x_1261_ = l_Lean_Expr_app___override(v___x_1260_, v_a_773_);
v___x_1262_ = l_Lean_Expr_app___override(v___x_1261_, v___x_1210_);
v___x_1263_ = l_Lean_Expr_app___override(v___x_1262_, v___x_1213_);
v___x_1264_ = l_Lean_Expr_app___override(v___x_1255_, v___x_1263_);
v___x_1265_ = l_Lean_Expr_app___override(v___x_1264_, v_proof_1194_);
if (v_isShared_1197_ == 0)
{
lean_ctor_set(v___x_1196_, 2, v___x_1265_);
v___x_1267_ = v___x_1196_;
goto v_reusejp_1266_;
}
else
{
lean_object* v_reuseFailAlloc_1275_; 
v_reuseFailAlloc_1275_ = lean_alloc_ctor(5, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1275_, 0, v_lhs_1192_);
lean_ctor_set(v_reuseFailAlloc_1275_, 1, v_rhs_1193_);
lean_ctor_set(v_reuseFailAlloc_1275_, 2, v___x_1265_);
v___x_1267_ = v_reuseFailAlloc_1275_;
goto v_reusejp_1266_;
}
v_reusejp_1266_:
{
lean_object* v___x_1268_; lean_object* v___x_1270_; 
v___x_1268_ = lean_array_push(v_snd_787_, v___x_1267_);
if (v_isShared_790_ == 0)
{
lean_ctor_set(v___x_789_, 1, v___x_1268_);
v___x_1270_ = v___x_789_;
goto v_reusejp_1269_;
}
else
{
lean_object* v_reuseFailAlloc_1274_; 
v_reuseFailAlloc_1274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1274_, 0, v_fst_786_);
lean_ctor_set(v_reuseFailAlloc_1274_, 1, v___x_1268_);
v___x_1270_ = v_reuseFailAlloc_1274_;
goto v_reusejp_1269_;
}
v_reusejp_1269_:
{
lean_object* v___x_1272_; 
if (v_isShared_1191_ == 0)
{
lean_ctor_set(v___x_1190_, 1, v___x_1270_);
v___x_1272_ = v___x_1190_;
goto v_reusejp_1271_;
}
else
{
lean_object* v_reuseFailAlloc_1273_; 
v_reuseFailAlloc_1273_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1273_, 0, v_fst_785_);
lean_ctor_set(v_reuseFailAlloc_1273_, 1, v___x_1270_);
v___x_1272_ = v_reuseFailAlloc_1273_;
goto v_reusejp_1271_;
}
v_reusejp_1271_:
{
v___y_779_ = v___x_1272_;
goto v___jp_778_;
}
}
}
}
}
}
case 8:
{
lean_object* v___x_1281_; uint8_t v_isShared_1282_; uint8_t v_isSharedCheck_1382_; 
lean_inc(v_fst_785_);
v_isSharedCheck_1382_ = !lean_is_exclusive(v_b_777_);
if (v_isSharedCheck_1382_ == 0)
{
lean_object* v_unused_1383_; lean_object* v_unused_1384_; 
v_unused_1383_ = lean_ctor_get(v_b_777_, 1);
lean_dec(v_unused_1383_);
v_unused_1384_ = lean_ctor_get(v_b_777_, 0);
lean_dec(v_unused_1384_);
v___x_1281_ = v_b_777_;
v_isShared_1282_ = v_isSharedCheck_1382_;
goto v_resetjp_1280_;
}
else
{
lean_dec(v_b_777_);
v___x_1281_ = lean_box(0);
v_isShared_1282_ = v_isSharedCheck_1382_;
goto v_resetjp_1280_;
}
v_resetjp_1280_:
{
lean_object* v_lhs_1283_; lean_object* v_rhs_1284_; lean_object* v_res_1285_; lean_object* v___x_1287_; uint8_t v_isShared_1288_; uint8_t v_isSharedCheck_1381_; 
v_lhs_1283_ = lean_ctor_get(v___x_791_, 0);
v_rhs_1284_ = lean_ctor_get(v___x_791_, 1);
v_res_1285_ = lean_ctor_get(v___x_791_, 2);
v_isSharedCheck_1381_ = !lean_is_exclusive(v___x_791_);
if (v_isSharedCheck_1381_ == 0)
{
v___x_1287_ = v___x_791_;
v_isShared_1288_ = v_isSharedCheck_1381_;
goto v_resetjp_1286_;
}
else
{
lean_inc(v_res_1285_);
lean_inc(v_rhs_1284_);
lean_inc(v_lhs_1283_);
lean_dec(v___x_791_);
v___x_1287_ = lean_box(0);
v_isShared_1288_ = v_isSharedCheck_1381_;
goto v_resetjp_1286_;
}
v_resetjp_1286_:
{
lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1308_; lean_object* v___x_1310_; 
v___x_1289_ = lean_unsigned_to_nat(1u);
v___x_1290_ = lean_nat_add(v_fst_785_, v___x_1289_);
v___x_1291_ = lean_box(0);
v___x_1292_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__64));
v___x_1293_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__70, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__70_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__70);
v___x_1294_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5));
lean_inc(v_u_769_);
v___x_1295_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1295_, 0, v_u_769_);
lean_ctor_set(v___x_1295_, 1, v___x_1291_);
lean_inc_ref(v___x_1295_);
v___x_1296_ = l_Lean_Expr_const___override(v___x_1294_, v___x_1295_);
lean_inc_ref(v_type_770_);
v___x_1297_ = l_Lean_Expr_app___override(v___x_1296_, v_type_770_);
lean_inc_ref(v_inst_771_);
v___x_1298_ = l_Lean_Expr_app___override(v___x_1297_, v_inst_771_);
lean_inc_n(v___x_772_, 3);
v___x_1299_ = lp_mathlib_Qq_mkNatLitQ(v___x_772_);
lean_inc_ref(v___x_1299_);
v___x_1300_ = l_Lean_Expr_app___override(v___x_1298_, v___x_1299_);
lean_inc_ref(v_a_773_);
v___x_1301_ = l_Lean_Expr_app___override(v___x_1300_, v_a_773_);
lean_inc(v_lhs_1283_);
v___x_1302_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_lhs_1283_);
lean_inc_ref(v___x_1302_);
lean_inc_ref_n(v___x_1301_, 2);
v___x_1303_ = l_Lean_Expr_app___override(v___x_1301_, v___x_1302_);
v___x_1304_ = l_Lean_Expr_app___override(v___x_1293_, v___x_1303_);
lean_inc(v_rhs_1284_);
v___x_1305_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_rhs_1284_);
lean_inc_ref(v___x_1305_);
v___x_1306_ = l_Lean_Expr_app___override(v___x_1301_, v___x_1305_);
v___x_1307_ = l_Lean_Expr_app___override(v___x_1304_, v___x_1306_);
lean_inc_ref(v___x_1307_);
lean_inc_n(v_fst_785_, 2);
v___x_1308_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0___redArg(v_fst_786_, v_fst_785_, v___x_1307_);
if (v_isShared_1288_ == 0)
{
lean_ctor_set(v___x_1287_, 2, v_fst_785_);
v___x_1310_ = v___x_1287_;
goto v_reusejp_1309_;
}
else
{
lean_object* v_reuseFailAlloc_1380_; 
v_reuseFailAlloc_1380_ = lean_alloc_ctor(8, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1380_, 0, v_lhs_1283_);
lean_ctor_set(v_reuseFailAlloc_1380_, 1, v_rhs_1284_);
lean_ctor_set(v_reuseFailAlloc_1380_, 2, v_fst_785_);
v___x_1310_ = v_reuseFailAlloc_1380_;
goto v_reusejp_1309_;
}
v_reusejp_1309_:
{
lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; lean_object* v___x_1344_; lean_object* v___x_1345_; lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1375_; 
v___x_1311_ = lean_array_push(v_snd_787_, v___x_1310_);
v___x_1312_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__0));
v___x_1313_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5);
v___x_1314_ = l_Lean_Expr_app___override(v___x_1313_, v___x_1307_);
lean_inc(v_res_1285_);
lean_inc(v___x_772_);
v___x_1315_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_res_1285_);
lean_inc_ref_n(v___x_1315_, 2);
v___x_1316_ = l_Lean_Expr_app___override(v___x_1301_, v___x_1315_);
v___x_1317_ = l_Lean_Expr_app___override(v___x_1314_, v___x_1316_);
v___x_1318_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9);
v___x_1319_ = l_Lean_Expr_app___override(v___x_1318_, v___x_1317_);
lean_inc(v_u_769_);
v___x_1320_ = l_Lean_Level_succ___override(v_u_769_);
v___x_1321_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1321_, 0, v___x_1320_);
lean_ctor_set(v___x_1321_, 1, v___x_1291_);
lean_inc_ref(v___x_1321_);
v___x_1322_ = l_Lean_Expr_const___override(v___x_1312_, v___x_1321_);
lean_inc_ref_n(v_type_770_, 8);
v___x_1323_ = l_Lean_Expr_app___override(v___x_1322_, v_type_770_);
lean_inc_ref_n(v___x_1295_, 5);
v___x_1324_ = l_Lean_Expr_const___override(v___x_1292_, v___x_1295_);
v___x_1325_ = l_Lean_Expr_app___override(v___x_1324_, v_type_770_);
v___x_1326_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__72));
v___x_1327_ = l_Lean_Expr_const___override(v___x_1326_, v___x_1295_);
v___x_1328_ = l_Lean_Expr_app___override(v___x_1327_, v_type_770_);
v___x_1329_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__38));
v___x_1330_ = l_Lean_Expr_const___override(v___x_1329_, v___x_1295_);
v___x_1331_ = l_Lean_Expr_app___override(v___x_1330_, v_type_770_);
v___x_1332_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41));
v___x_1333_ = l_Lean_Expr_const___override(v___x_1332_, v___x_1295_);
v___x_1334_ = l_Lean_Expr_app___override(v___x_1333_, v_type_770_);
v___x_1335_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__43));
v___x_1336_ = l_Lean_Expr_const___override(v___x_1335_, v___x_1295_);
v___x_1337_ = l_Lean_Expr_app___override(v___x_1336_, v_type_770_);
lean_inc_ref_n(v_inst_771_, 2);
v___x_1338_ = l_Lean_Expr_app___override(v___x_1337_, v_inst_771_);
v___x_1339_ = l_Lean_Expr_app___override(v___x_1334_, v___x_1338_);
v___x_1340_ = l_Lean_Expr_app___override(v___x_1331_, v___x_1339_);
v___x_1341_ = l_Lean_Expr_app___override(v___x_1328_, v___x_1340_);
v___x_1342_ = l_Lean_Expr_app___override(v___x_1325_, v___x_1341_);
lean_inc_ref(v___x_1302_);
v___x_1343_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1343_, 0, v___x_1302_);
lean_ctor_set(v___x_1343_, 1, v___x_1291_);
v___x_1344_ = lean_array_mk(v___x_1343_);
lean_inc_ref_n(v_a_773_, 4);
v___x_1345_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1344_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1344_);
v___x_1346_ = l_Lean_Expr_app___override(v___x_1342_, v___x_1345_);
lean_inc_ref(v___x_1305_);
v___x_1347_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1347_, 0, v___x_1305_);
lean_ctor_set(v___x_1347_, 1, v___x_1291_);
v___x_1348_ = lean_array_mk(v___x_1347_);
v___x_1349_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1348_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1348_);
v___x_1350_ = l_Lean_Expr_app___override(v___x_1346_, v___x_1349_);
lean_inc_ref(v___x_1350_);
v___x_1351_ = l_Lean_Expr_app___override(v___x_1323_, v___x_1350_);
v___x_1352_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1352_, 0, v___x_1315_);
lean_ctor_set(v___x_1352_, 1, v___x_1291_);
v___x_1353_ = lean_array_mk(v___x_1352_);
v___x_1354_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1353_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1353_);
v___x_1355_ = l_Lean_Expr_app___override(v___x_1351_, v___x_1354_);
v___x_1356_ = l_Lean_Expr_app___override(v___x_1319_, v___x_1355_);
v___x_1357_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__74));
v___x_1358_ = l_Lean_Expr_const___override(v___x_1357_, v___x_1295_);
v___x_1359_ = l_Lean_Expr_app___override(v___x_1358_, v_type_770_);
v___x_1360_ = l_Lean_Expr_app___override(v___x_1359_, v_inst_771_);
v___x_1361_ = l_Lean_Expr_app___override(v___x_1360_, v___x_1299_);
v___x_1362_ = l_Lean_Expr_app___override(v___x_1361_, v_a_773_);
v___x_1363_ = l_Lean_Expr_app___override(v___x_1362_, v___x_1302_);
v___x_1364_ = l_Lean_Expr_app___override(v___x_1363_, v___x_1305_);
v___x_1365_ = l_Lean_Expr_app___override(v___x_1364_, v___x_1315_);
v___x_1366_ = l_Lean_Expr_app___override(v___x_1356_, v___x_1365_);
v___x_1367_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__30));
v___x_1368_ = l_Lean_Expr_const___override(v___x_1367_, v___x_1321_);
v___x_1369_ = l_Lean_Expr_app___override(v___x_1368_, v_type_770_);
v___x_1370_ = l_Lean_Expr_app___override(v___x_1369_, v___x_1350_);
v___x_1371_ = l_Lean_Expr_app___override(v___x_1366_, v___x_1370_);
v___x_1372_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1372_, 0, v_fst_785_);
lean_ctor_set(v___x_1372_, 1, v_res_1285_);
lean_ctor_set(v___x_1372_, 2, v___x_1371_);
v___x_1373_ = lean_array_push(v___x_1311_, v___x_1372_);
if (v_isShared_790_ == 0)
{
lean_ctor_set(v___x_789_, 1, v___x_1373_);
lean_ctor_set(v___x_789_, 0, v___x_1308_);
v___x_1375_ = v___x_789_;
goto v_reusejp_1374_;
}
else
{
lean_object* v_reuseFailAlloc_1379_; 
v_reuseFailAlloc_1379_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1379_, 0, v___x_1308_);
lean_ctor_set(v_reuseFailAlloc_1379_, 1, v___x_1373_);
v___x_1375_ = v_reuseFailAlloc_1379_;
goto v_reusejp_1374_;
}
v_reusejp_1374_:
{
lean_object* v___x_1377_; 
if (v_isShared_1282_ == 0)
{
lean_ctor_set(v___x_1281_, 1, v___x_1375_);
lean_ctor_set(v___x_1281_, 0, v___x_1290_);
v___x_1377_ = v___x_1281_;
goto v_reusejp_1376_;
}
else
{
lean_object* v_reuseFailAlloc_1378_; 
v_reuseFailAlloc_1378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1378_, 0, v___x_1290_);
lean_ctor_set(v_reuseFailAlloc_1378_, 1, v___x_1375_);
v___x_1377_ = v_reuseFailAlloc_1378_;
goto v_reusejp_1376_;
}
v_reusejp_1376_:
{
v___y_779_ = v___x_1377_;
goto v___jp_778_;
}
}
}
}
}
}
case 9:
{
lean_object* v___x_1386_; uint8_t v_isShared_1387_; uint8_t v_isSharedCheck_1487_; 
lean_inc(v_fst_785_);
v_isSharedCheck_1487_ = !lean_is_exclusive(v_b_777_);
if (v_isSharedCheck_1487_ == 0)
{
lean_object* v_unused_1488_; lean_object* v_unused_1489_; 
v_unused_1488_ = lean_ctor_get(v_b_777_, 1);
lean_dec(v_unused_1488_);
v_unused_1489_ = lean_ctor_get(v_b_777_, 0);
lean_dec(v_unused_1489_);
v___x_1386_ = v_b_777_;
v_isShared_1387_ = v_isSharedCheck_1487_;
goto v_resetjp_1385_;
}
else
{
lean_dec(v_b_777_);
v___x_1386_ = lean_box(0);
v_isShared_1387_ = v_isSharedCheck_1487_;
goto v_resetjp_1385_;
}
v_resetjp_1385_:
{
lean_object* v_lhs_1388_; lean_object* v_rhs_1389_; lean_object* v_res_1390_; lean_object* v___x_1392_; uint8_t v_isShared_1393_; uint8_t v_isSharedCheck_1486_; 
v_lhs_1388_ = lean_ctor_get(v___x_791_, 0);
v_rhs_1389_ = lean_ctor_get(v___x_791_, 1);
v_res_1390_ = lean_ctor_get(v___x_791_, 2);
v_isSharedCheck_1486_ = !lean_is_exclusive(v___x_791_);
if (v_isSharedCheck_1486_ == 0)
{
v___x_1392_ = v___x_791_;
v_isShared_1393_ = v_isSharedCheck_1486_;
goto v_resetjp_1391_;
}
else
{
lean_inc(v_res_1390_);
lean_inc(v_rhs_1389_);
lean_inc(v_lhs_1388_);
lean_dec(v___x_791_);
v___x_1392_ = lean_box(0);
v_isShared_1393_ = v_isSharedCheck_1486_;
goto v_resetjp_1391_;
}
v_resetjp_1391_:
{
lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___x_1415_; 
v___x_1394_ = lean_unsigned_to_nat(1u);
v___x_1395_ = lean_nat_add(v_fst_785_, v___x_1394_);
v___x_1396_ = lean_box(0);
v___x_1397_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__77));
v___x_1398_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__83, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__83_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__83);
v___x_1399_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__5));
lean_inc(v_u_769_);
v___x_1400_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1400_, 0, v_u_769_);
lean_ctor_set(v___x_1400_, 1, v___x_1396_);
lean_inc_ref(v___x_1400_);
v___x_1401_ = l_Lean_Expr_const___override(v___x_1399_, v___x_1400_);
lean_inc_ref(v_type_770_);
v___x_1402_ = l_Lean_Expr_app___override(v___x_1401_, v_type_770_);
lean_inc_ref(v_inst_771_);
v___x_1403_ = l_Lean_Expr_app___override(v___x_1402_, v_inst_771_);
lean_inc_n(v___x_772_, 3);
v___x_1404_ = lp_mathlib_Qq_mkNatLitQ(v___x_772_);
lean_inc_ref(v___x_1404_);
v___x_1405_ = l_Lean_Expr_app___override(v___x_1403_, v___x_1404_);
lean_inc_ref(v_a_773_);
v___x_1406_ = l_Lean_Expr_app___override(v___x_1405_, v_a_773_);
lean_inc(v_lhs_1388_);
v___x_1407_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_lhs_1388_);
lean_inc_ref(v___x_1407_);
lean_inc_ref_n(v___x_1406_, 2);
v___x_1408_ = l_Lean_Expr_app___override(v___x_1406_, v___x_1407_);
v___x_1409_ = l_Lean_Expr_app___override(v___x_1398_, v___x_1408_);
lean_inc(v_rhs_1389_);
v___x_1410_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_rhs_1389_);
lean_inc_ref(v___x_1410_);
v___x_1411_ = l_Lean_Expr_app___override(v___x_1406_, v___x_1410_);
v___x_1412_ = l_Lean_Expr_app___override(v___x_1409_, v___x_1411_);
lean_inc_ref(v___x_1412_);
lean_inc_n(v_fst_785_, 2);
v___x_1413_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0___redArg(v_fst_786_, v_fst_785_, v___x_1412_);
if (v_isShared_1393_ == 0)
{
lean_ctor_set(v___x_1392_, 2, v_fst_785_);
v___x_1415_ = v___x_1392_;
goto v_reusejp_1414_;
}
else
{
lean_object* v_reuseFailAlloc_1485_; 
v_reuseFailAlloc_1485_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1485_, 0, v_lhs_1388_);
lean_ctor_set(v_reuseFailAlloc_1485_, 1, v_rhs_1389_);
lean_ctor_set(v_reuseFailAlloc_1485_, 2, v_fst_785_);
v___x_1415_ = v_reuseFailAlloc_1485_;
goto v_reusejp_1414_;
}
v_reusejp_1414_:
{
lean_object* v___x_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1480_; 
v___x_1416_ = lean_array_push(v_snd_787_, v___x_1415_);
v___x_1417_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__0));
v___x_1418_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__5);
v___x_1419_ = l_Lean_Expr_app___override(v___x_1418_, v___x_1412_);
lean_inc(v_res_1390_);
lean_inc(v___x_772_);
v___x_1420_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___lam__0(v___x_772_, v_res_1390_);
lean_inc_ref_n(v___x_1420_, 2);
v___x_1421_ = l_Lean_Expr_app___override(v___x_1406_, v___x_1420_);
v___x_1422_ = l_Lean_Expr_app___override(v___x_1419_, v___x_1421_);
v___x_1423_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__9);
v___x_1424_ = l_Lean_Expr_app___override(v___x_1423_, v___x_1422_);
lean_inc(v_u_769_);
v___x_1425_ = l_Lean_Level_succ___override(v_u_769_);
v___x_1426_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1426_, 0, v___x_1425_);
lean_ctor_set(v___x_1426_, 1, v___x_1396_);
lean_inc_ref(v___x_1426_);
v___x_1427_ = l_Lean_Expr_const___override(v___x_1417_, v___x_1426_);
lean_inc_ref_n(v_type_770_, 8);
v___x_1428_ = l_Lean_Expr_app___override(v___x_1427_, v_type_770_);
lean_inc_ref_n(v___x_1400_, 5);
v___x_1429_ = l_Lean_Expr_const___override(v___x_1397_, v___x_1400_);
v___x_1430_ = l_Lean_Expr_app___override(v___x_1429_, v_type_770_);
v___x_1431_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__86));
v___x_1432_ = l_Lean_Expr_const___override(v___x_1431_, v___x_1400_);
v___x_1433_ = l_Lean_Expr_app___override(v___x_1432_, v_type_770_);
v___x_1434_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__88));
v___x_1435_ = l_Lean_Expr_const___override(v___x_1434_, v___x_1400_);
v___x_1436_ = l_Lean_Expr_app___override(v___x_1435_, v_type_770_);
v___x_1437_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__41));
v___x_1438_ = l_Lean_Expr_const___override(v___x_1437_, v___x_1400_);
v___x_1439_ = l_Lean_Expr_app___override(v___x_1438_, v_type_770_);
v___x_1440_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__43));
v___x_1441_ = l_Lean_Expr_const___override(v___x_1440_, v___x_1400_);
v___x_1442_ = l_Lean_Expr_app___override(v___x_1441_, v_type_770_);
lean_inc_ref_n(v_inst_771_, 2);
v___x_1443_ = l_Lean_Expr_app___override(v___x_1442_, v_inst_771_);
v___x_1444_ = l_Lean_Expr_app___override(v___x_1439_, v___x_1443_);
v___x_1445_ = l_Lean_Expr_app___override(v___x_1436_, v___x_1444_);
v___x_1446_ = l_Lean_Expr_app___override(v___x_1433_, v___x_1445_);
v___x_1447_ = l_Lean_Expr_app___override(v___x_1430_, v___x_1446_);
lean_inc_ref(v___x_1407_);
v___x_1448_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1448_, 0, v___x_1407_);
lean_ctor_set(v___x_1448_, 1, v___x_1396_);
v___x_1449_ = lean_array_mk(v___x_1448_);
lean_inc_ref_n(v_a_773_, 4);
v___x_1450_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1449_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1449_);
v___x_1451_ = l_Lean_Expr_app___override(v___x_1447_, v___x_1450_);
lean_inc_ref(v___x_1410_);
v___x_1452_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1452_, 0, v___x_1410_);
lean_ctor_set(v___x_1452_, 1, v___x_1396_);
v___x_1453_ = lean_array_mk(v___x_1452_);
v___x_1454_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1453_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1453_);
v___x_1455_ = l_Lean_Expr_app___override(v___x_1451_, v___x_1454_);
lean_inc_ref(v___x_1455_);
v___x_1456_ = l_Lean_Expr_app___override(v___x_1428_, v___x_1455_);
v___x_1457_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1457_, 0, v___x_1420_);
lean_ctor_set(v___x_1457_, 1, v___x_1396_);
v___x_1458_ = lean_array_mk(v___x_1457_);
v___x_1459_ = l_Lean_Expr_betaRev(v_a_773_, v___x_1458_, v___x_783_, v___x_783_);
lean_dec_ref(v___x_1458_);
v___x_1460_ = l_Lean_Expr_app___override(v___x_1456_, v___x_1459_);
v___x_1461_ = l_Lean_Expr_app___override(v___x_1424_, v___x_1460_);
v___x_1462_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___closed__90));
v___x_1463_ = l_Lean_Expr_const___override(v___x_1462_, v___x_1400_);
v___x_1464_ = l_Lean_Expr_app___override(v___x_1463_, v_type_770_);
v___x_1465_ = l_Lean_Expr_app___override(v___x_1464_, v_inst_771_);
v___x_1466_ = l_Lean_Expr_app___override(v___x_1465_, v___x_1404_);
v___x_1467_ = l_Lean_Expr_app___override(v___x_1466_, v_a_773_);
v___x_1468_ = l_Lean_Expr_app___override(v___x_1467_, v___x_1407_);
v___x_1469_ = l_Lean_Expr_app___override(v___x_1468_, v___x_1410_);
v___x_1470_ = l_Lean_Expr_app___override(v___x_1469_, v___x_1420_);
v___x_1471_ = l_Lean_Expr_app___override(v___x_1461_, v___x_1470_);
v___x_1472_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_map_go___at___00Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3_spec__7___closed__30));
v___x_1473_ = l_Lean_Expr_const___override(v___x_1472_, v___x_1426_);
v___x_1474_ = l_Lean_Expr_app___override(v___x_1473_, v_type_770_);
v___x_1475_ = l_Lean_Expr_app___override(v___x_1474_, v___x_1455_);
v___x_1476_ = l_Lean_Expr_app___override(v___x_1471_, v___x_1475_);
v___x_1477_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1477_, 0, v_fst_785_);
lean_ctor_set(v___x_1477_, 1, v_res_1390_);
lean_ctor_set(v___x_1477_, 2, v___x_1476_);
v___x_1478_ = lean_array_push(v___x_1416_, v___x_1477_);
if (v_isShared_790_ == 0)
{
lean_ctor_set(v___x_789_, 1, v___x_1478_);
lean_ctor_set(v___x_789_, 0, v___x_1413_);
v___x_1480_ = v___x_789_;
goto v_reusejp_1479_;
}
else
{
lean_object* v_reuseFailAlloc_1484_; 
v_reuseFailAlloc_1484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1484_, 0, v___x_1413_);
lean_ctor_set(v_reuseFailAlloc_1484_, 1, v___x_1478_);
v___x_1480_ = v_reuseFailAlloc_1484_;
goto v_reusejp_1479_;
}
v_reusejp_1479_:
{
lean_object* v___x_1482_; 
if (v_isShared_1387_ == 0)
{
lean_ctor_set(v___x_1386_, 1, v___x_1480_);
lean_ctor_set(v___x_1386_, 0, v___x_1395_);
v___x_1482_ = v___x_1386_;
goto v_reusejp_1481_;
}
else
{
lean_object* v_reuseFailAlloc_1483_; 
v_reuseFailAlloc_1483_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1483_, 0, v___x_1395_);
lean_ctor_set(v_reuseFailAlloc_1483_, 1, v___x_1480_);
v___x_1482_ = v_reuseFailAlloc_1483_;
goto v_reusejp_1481_;
}
v_reusejp_1481_:
{
v___y_779_ = v___x_1482_;
goto v___jp_778_;
}
}
}
}
}
}
default: 
{
lean_dec(v___x_791_);
lean_del_object(v___x_789_);
lean_dec(v_snd_787_);
lean_dec(v_fst_786_);
v___y_779_ = v_b_777_;
goto v___jp_778_;
}
}
}
}
else
{
lean_dec_ref(v_a_773_);
lean_dec(v___x_772_);
lean_dec_ref(v_inst_771_);
lean_dec_ref(v_type_770_);
lean_dec(v_u_769_);
return v_b_777_;
}
v___jp_778_:
{
size_t v___x_780_; size_t v___x_781_; 
v___x_780_ = ((size_t)1ULL);
v___x_781_ = lean_usize_add(v_i_775_, v___x_780_);
v_i_775_ = v___x_781_;
v_b_777_ = v___y_779_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4___boxed(lean_object* v_u_1491_, lean_object* v_type_1492_, lean_object* v_inst_1493_, lean_object* v___x_1494_, lean_object* v_a_1495_, lean_object* v_as_1496_, lean_object* v_i_1497_, lean_object* v_stop_1498_, lean_object* v_b_1499_){
_start:
{
size_t v_i_boxed_1500_; size_t v_stop_boxed_1501_; lean_object* v_res_1502_; 
v_i_boxed_1500_ = lean_unbox_usize(v_i_1497_);
lean_dec(v_i_1497_);
v_stop_boxed_1501_ = lean_unbox_usize(v_stop_1498_);
lean_dec(v_stop_1498_);
v_res_1502_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4(v_u_1491_, v_type_1492_, v_inst_1493_, v___x_1494_, v_a_1495_, v_as_1496_, v_i_boxed_1500_, v_stop_boxed_1501_, v_b_1499_);
lean_dec_ref(v_as_1496_);
return v_res_1502_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__0(void){
_start:
{
lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; 
v___x_1503_ = lean_box(0);
v___x_1504_ = lean_unsigned_to_nat(16u);
v___x_1505_ = lean_mk_array(v___x_1504_, v___x_1503_);
return v___x_1505_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__1(void){
_start:
{
lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v_idxToAtom_1508_; 
v___x_1506_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__0, &lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__0);
v___x_1507_ = lean_unsigned_to_nat(0u);
v_idxToAtom_1508_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_idxToAtom_1508_, 0, v___x_1507_);
lean_ctor_set(v_idxToAtom_1508_, 1, v___x_1506_);
return v_idxToAtom_1508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt(lean_object* v_u_1509_, lean_object* v_type_1510_, lean_object* v_inst_1511_, lean_object* v_facts_1512_, lean_object* v_a_1513_, lean_object* v_a_1514_, lean_object* v_a_1515_, lean_object* v_a_1516_, lean_object* v_a_1517_, lean_object* v_a_1518_){
_start:
{
lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v_idxToAtom_1522_; size_t v_sz_1523_; size_t v___x_1524_; lean_object* v___x_1525_; 
v___x_1520_ = lean_st_ref_get(v_a_1514_);
v___x_1521_ = lean_unsigned_to_nat(0u);
v_idxToAtom_1522_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__1, &lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___closed__1);
v_sz_1523_ = lean_array_size(v___x_1520_);
v___x_1524_ = ((size_t)0ULL);
lean_inc_ref(v_type_1510_);
v___x_1525_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1___redArg(v_type_1510_, v___x_1520_, v_sz_1523_, v___x_1524_, v_idxToAtom_1522_, v_a_1515_, v_a_1516_, v_a_1517_, v_a_1518_);
lean_dec(v___x_1520_);
if (lean_obj_tag(v___x_1525_) == 0)
{
lean_object* v_a_1526_; lean_object* v___x_1528_; uint8_t v_isShared_1529_; uint8_t v_isSharedCheck_1570_; 
v_a_1526_ = lean_ctor_get(v___x_1525_, 0);
v_isSharedCheck_1570_ = !lean_is_exclusive(v___x_1525_);
if (v_isSharedCheck_1570_ == 0)
{
v___x_1528_ = v___x_1525_;
v_isShared_1529_ = v_isSharedCheck_1570_;
goto v_resetjp_1527_;
}
else
{
lean_inc(v_a_1526_);
lean_dec(v___x_1525_);
v___x_1528_ = lean_box(0);
v_isShared_1529_ = v_isSharedCheck_1570_;
goto v_resetjp_1527_;
}
v_resetjp_1527_:
{
lean_object* v_size_1530_; lean_object* v___f_1531_; lean_object* v___x_1532_; lean_object* v___x_1533_; 
v_size_1530_ = lean_ctor_get(v_a_1526_, 0);
lean_inc_n(v_size_1530_, 2);
lean_inc(v_a_1526_);
lean_inc_ref_n(v_type_1510_, 2);
v___f_1531_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1531_, 0, v_type_1510_);
lean_closure_set(v___f_1531_, 1, v_a_1526_);
v___x_1532_ = l_Array_ofFn___redArg(v_size_1530_, v___f_1531_);
lean_inc(v_u_1509_);
v___x_1533_ = lp_mathlib_Mathlib_Tactic_Order_ToInt_mkFinFun(v_u_1509_, v_type_1510_, v___x_1532_, v_a_1515_, v_a_1516_, v_a_1517_, v_a_1518_);
if (lean_obj_tag(v___x_1533_) == 0)
{
lean_object* v_a_1534_; lean_object* v___x_1536_; uint8_t v_isShared_1537_; uint8_t v_isSharedCheck_1561_; 
v_a_1534_ = lean_ctor_get(v___x_1533_, 0);
v_isSharedCheck_1561_ = !lean_is_exclusive(v___x_1533_);
if (v_isSharedCheck_1561_ == 0)
{
v___x_1536_ = v___x_1533_;
v_isShared_1537_ = v_isSharedCheck_1561_;
goto v_resetjp_1535_;
}
else
{
lean_inc(v_a_1534_);
lean_dec(v___x_1533_);
v___x_1536_ = lean_box(0);
v_isShared_1537_ = v_isSharedCheck_1561_;
goto v_resetjp_1535_;
}
v_resetjp_1535_:
{
lean_object* v___y_1539_; lean_object* v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; uint8_t v___x_1548_; 
lean_inc(v_a_1534_);
lean_inc(v_size_1530_);
lean_inc_ref(v_inst_1511_);
lean_inc_ref(v_type_1510_);
lean_inc(v_u_1509_);
v___x_1544_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_map___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__3(v_u_1509_, v_type_1510_, v_inst_1511_, v_size_1530_, v_a_1534_, v_a_1526_);
v___x_1545_ = lean_mk_empty_array_with_capacity(v_size_1530_);
v___x_1546_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1546_, 0, v___x_1544_);
lean_ctor_set(v___x_1546_, 1, v___x_1545_);
v___x_1547_ = lean_array_get_size(v_facts_1512_);
v___x_1548_ = lean_nat_dec_lt(v___x_1521_, v___x_1547_);
if (v___x_1548_ == 0)
{
lean_object* v___x_1550_; 
lean_del_object(v___x_1536_);
lean_dec(v_a_1534_);
lean_dec(v_size_1530_);
lean_dec_ref(v_inst_1511_);
lean_dec_ref(v_type_1510_);
lean_dec(v_u_1509_);
if (v_isShared_1529_ == 0)
{
lean_ctor_set(v___x_1528_, 0, v___x_1546_);
v___x_1550_ = v___x_1528_;
goto v_reusejp_1549_;
}
else
{
lean_object* v_reuseFailAlloc_1551_; 
v_reuseFailAlloc_1551_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1551_, 0, v___x_1546_);
v___x_1550_ = v_reuseFailAlloc_1551_;
goto v_reusejp_1549_;
}
v_reusejp_1549_:
{
return v___x_1550_;
}
}
else
{
lean_object* v___x_1552_; uint8_t v___x_1553_; 
lean_inc_ref(v___x_1546_);
lean_inc(v_size_1530_);
v___x_1552_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1552_, 0, v_size_1530_);
lean_ctor_set(v___x_1552_, 1, v___x_1546_);
v___x_1553_ = lean_nat_dec_le(v___x_1547_, v___x_1547_);
if (v___x_1553_ == 0)
{
if (v___x_1548_ == 0)
{
lean_object* v___x_1555_; 
lean_dec_ref_known(v___x_1552_, 2);
lean_del_object(v___x_1536_);
lean_dec(v_a_1534_);
lean_dec(v_size_1530_);
lean_dec_ref(v_inst_1511_);
lean_dec_ref(v_type_1510_);
lean_dec(v_u_1509_);
if (v_isShared_1529_ == 0)
{
lean_ctor_set(v___x_1528_, 0, v___x_1546_);
v___x_1555_ = v___x_1528_;
goto v_reusejp_1554_;
}
else
{
lean_object* v_reuseFailAlloc_1556_; 
v_reuseFailAlloc_1556_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1556_, 0, v___x_1546_);
v___x_1555_ = v_reuseFailAlloc_1556_;
goto v_reusejp_1554_;
}
v_reusejp_1554_:
{
return v___x_1555_;
}
}
else
{
size_t v___x_1557_; lean_object* v___x_1558_; 
lean_dec_ref_known(v___x_1546_, 2);
lean_del_object(v___x_1528_);
v___x_1557_ = lean_usize_of_nat(v___x_1547_);
v___x_1558_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4(v_u_1509_, v_type_1510_, v_inst_1511_, v_size_1530_, v_a_1534_, v_facts_1512_, v___x_1524_, v___x_1557_, v___x_1552_);
v___y_1539_ = v___x_1558_;
goto v___jp_1538_;
}
}
else
{
size_t v___x_1559_; lean_object* v___x_1560_; 
lean_dec_ref_known(v___x_1546_, 2);
lean_del_object(v___x_1528_);
v___x_1559_ = lean_usize_of_nat(v___x_1547_);
v___x_1560_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__4(v_u_1509_, v_type_1510_, v_inst_1511_, v_size_1530_, v_a_1534_, v_facts_1512_, v___x_1524_, v___x_1559_, v___x_1552_);
v___y_1539_ = v___x_1560_;
goto v___jp_1538_;
}
}
v___jp_1538_:
{
lean_object* v_snd_1540_; lean_object* v___x_1542_; 
v_snd_1540_ = lean_ctor_get(v___y_1539_, 1);
lean_inc(v_snd_1540_);
lean_dec_ref(v___y_1539_);
if (v_isShared_1537_ == 0)
{
lean_ctor_set(v___x_1536_, 0, v_snd_1540_);
v___x_1542_ = v___x_1536_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1543_; 
v_reuseFailAlloc_1543_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1543_, 0, v_snd_1540_);
v___x_1542_ = v_reuseFailAlloc_1543_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
return v___x_1542_;
}
}
}
}
else
{
lean_object* v_a_1562_; lean_object* v___x_1564_; uint8_t v_isShared_1565_; uint8_t v_isSharedCheck_1569_; 
lean_dec(v_size_1530_);
lean_del_object(v___x_1528_);
lean_dec(v_a_1526_);
lean_dec_ref(v_inst_1511_);
lean_dec_ref(v_type_1510_);
lean_dec(v_u_1509_);
v_a_1562_ = lean_ctor_get(v___x_1533_, 0);
v_isSharedCheck_1569_ = !lean_is_exclusive(v___x_1533_);
if (v_isSharedCheck_1569_ == 0)
{
v___x_1564_ = v___x_1533_;
v_isShared_1565_ = v_isSharedCheck_1569_;
goto v_resetjp_1563_;
}
else
{
lean_inc(v_a_1562_);
lean_dec(v___x_1533_);
v___x_1564_ = lean_box(0);
v_isShared_1565_ = v_isSharedCheck_1569_;
goto v_resetjp_1563_;
}
v_resetjp_1563_:
{
lean_object* v___x_1567_; 
if (v_isShared_1565_ == 0)
{
v___x_1567_ = v___x_1564_;
goto v_reusejp_1566_;
}
else
{
lean_object* v_reuseFailAlloc_1568_; 
v_reuseFailAlloc_1568_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1568_, 0, v_a_1562_);
v___x_1567_ = v_reuseFailAlloc_1568_;
goto v_reusejp_1566_;
}
v_reusejp_1566_:
{
return v___x_1567_;
}
}
}
}
}
else
{
lean_object* v_a_1571_; lean_object* v___x_1573_; uint8_t v_isShared_1574_; uint8_t v_isSharedCheck_1578_; 
lean_dec_ref(v_inst_1511_);
lean_dec_ref(v_type_1510_);
lean_dec(v_u_1509_);
v_a_1571_ = lean_ctor_get(v___x_1525_, 0);
v_isSharedCheck_1578_ = !lean_is_exclusive(v___x_1525_);
if (v_isSharedCheck_1578_ == 0)
{
v___x_1573_ = v___x_1525_;
v_isShared_1574_ = v_isSharedCheck_1578_;
goto v_resetjp_1572_;
}
else
{
lean_inc(v_a_1571_);
lean_dec(v___x_1525_);
v___x_1573_ = lean_box(0);
v_isShared_1574_ = v_isSharedCheck_1578_;
goto v_resetjp_1572_;
}
v_resetjp_1572_:
{
lean_object* v___x_1576_; 
if (v_isShared_1574_ == 0)
{
v___x_1576_ = v___x_1573_;
goto v_reusejp_1575_;
}
else
{
lean_object* v_reuseFailAlloc_1577_; 
v_reuseFailAlloc_1577_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1577_, 0, v_a_1571_);
v___x_1576_ = v_reuseFailAlloc_1577_;
goto v_reusejp_1575_;
}
v_reusejp_1575_:
{
return v___x_1576_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt___boxed(lean_object* v_u_1579_, lean_object* v_type_1580_, lean_object* v_inst_1581_, lean_object* v_facts_1582_, lean_object* v_a_1583_, lean_object* v_a_1584_, lean_object* v_a_1585_, lean_object* v_a_1586_, lean_object* v_a_1587_, lean_object* v_a_1588_, lean_object* v_a_1589_){
_start:
{
lean_object* v_res_1590_; 
v_res_1590_ = lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt(v_u_1579_, v_type_1580_, v_inst_1581_, v_facts_1582_, v_a_1583_, v_a_1584_, v_a_1585_, v_a_1586_, v_a_1587_, v_a_1588_);
lean_dec(v_a_1588_);
lean_dec_ref(v_a_1587_);
lean_dec(v_a_1586_);
lean_dec_ref(v_a_1585_);
lean_dec(v_a_1584_);
lean_dec_ref(v_a_1583_);
lean_dec_ref(v_facts_1582_);
return v_res_1590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0(lean_object* v_00_u03b2_1591_, lean_object* v_m_1592_, lean_object* v_a_1593_, lean_object* v_b_1594_){
_start:
{
lean_object* v___x_1595_; 
v___x_1595_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0___redArg(v_m_1592_, v_a_1593_, v_b_1594_);
return v___x_1595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1(lean_object* v_type_1596_, lean_object* v_as_1597_, size_t v_sz_1598_, size_t v_i_1599_, lean_object* v_b_1600_, lean_object* v___y_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_){
_start:
{
lean_object* v___x_1608_; 
v___x_1608_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1___redArg(v_type_1596_, v_as_1597_, v_sz_1598_, v_i_1599_, v_b_1600_, v___y_1603_, v___y_1604_, v___y_1605_, v___y_1606_);
return v___x_1608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1___boxed(lean_object* v_type_1609_, lean_object* v_as_1610_, lean_object* v_sz_1611_, lean_object* v_i_1612_, lean_object* v_b_1613_, lean_object* v___y_1614_, lean_object* v___y_1615_, lean_object* v___y_1616_, lean_object* v___y_1617_, lean_object* v___y_1618_, lean_object* v___y_1619_, lean_object* v___y_1620_){
_start:
{
size_t v_sz_boxed_1621_; size_t v_i_boxed_1622_; lean_object* v_res_1623_; 
v_sz_boxed_1621_ = lean_unbox_usize(v_sz_1611_);
lean_dec(v_sz_1611_);
v_i_boxed_1622_ = lean_unbox_usize(v_i_1612_);
lean_dec(v_i_1612_);
v_res_1623_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__1(v_type_1609_, v_as_1610_, v_sz_boxed_1621_, v_i_boxed_1622_, v_b_1613_, v___y_1614_, v___y_1615_, v___y_1616_, v___y_1617_, v___y_1618_, v___y_1619_);
lean_dec(v___y_1619_);
lean_dec_ref(v___y_1618_);
lean_dec(v___y_1617_);
lean_dec_ref(v___y_1616_);
lean_dec(v___y_1615_);
lean_dec_ref(v___y_1614_);
lean_dec_ref(v_as_1610_);
return v_res_1623_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0(lean_object* v_00_u03b2_1624_, lean_object* v_a_1625_, lean_object* v_x_1626_){
_start:
{
uint8_t v___x_1627_; 
v___x_1627_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0___redArg(v_a_1625_, v_x_1626_);
return v___x_1627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1628_, lean_object* v_a_1629_, lean_object* v_x_1630_){
_start:
{
uint8_t v_res_1631_; lean_object* v_r_1632_; 
v_res_1631_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__0(v_00_u03b2_1628_, v_a_1629_, v_x_1630_);
lean_dec(v_x_1630_);
lean_dec(v_a_1629_);
v_r_1632_ = lean_box(v_res_1631_);
return v_r_1632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1(lean_object* v_00_u03b2_1633_, lean_object* v_data_1634_){
_start:
{
lean_object* v___x_1635_; 
v___x_1635_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1___redArg(v_data_1634_);
return v___x_1635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__2(lean_object* v_00_u03b2_1636_, lean_object* v_a_1637_, lean_object* v_b_1638_, lean_object* v_x_1639_){
_start:
{
lean_object* v___x_1640_; 
v___x_1640_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__2___redArg(v_a_1637_, v_b_1638_, v_x_1639_);
return v___x_1640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_1641_, lean_object* v_i_1642_, lean_object* v_source_1643_, lean_object* v_target_1644_){
_start:
{
lean_object* v___x_1645_; 
v___x_1645_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2___redArg(v_i_1642_, v_source_1643_, v_target_1644_);
return v___x_1645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2_spec__7(lean_object* v_00_u03b2_1646_, lean_object* v_x_1647_, lean_object* v_x_1648_){
_start:
{
lean_object* v___x_1649_; 
v___x_1649_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_ToInt_translateToInt_spec__0_spec__1_spec__2_spec__7___redArg(v_x_1647_, v_x_1648_);
return v___x_1649_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Pairwise(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_GeneralizeProofs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_ToInt(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_GeneralizeProofs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
lean_object* runtime_initialize_Std_Data_HashMap_AdditionalOperations(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Order_ToInt(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Data_HashMap_AdditionalOperations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Pairwise(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_GeneralizeProofs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin);
lean_object* initialize_Std_Data_HashMap_AdditionalOperations(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Order_ToInt(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_GeneralizeProofs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Data_HashMap_AdditionalOperations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_ToInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Order_ToInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Order_ToInt(builtin);
}
#ifdef __cplusplus
}
#endif
