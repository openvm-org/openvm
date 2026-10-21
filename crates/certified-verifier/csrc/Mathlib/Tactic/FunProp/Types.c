// Lean compiler output
// Module: Mathlib.Tactic.FunProp.Types
// Imports: public import Init public meta import Init public meta import Mathlib.Lean.Meta.RefinedDiscrTree.Basic public import Mathlib.Tactic.FunProp.FunctionData public import Lean.Meta.Tactic.Simp public import Mathlib.Lean.Meta.RefinedDiscrTree.Basic public meta import Mathlib.Tactic.FunProp.FunctionData
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
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_mkConstWithLevelParams___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fun_prop"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(211, 174, 49, 251, 64, 24, 251, 1)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(194, 95, 140, 15, 16, 100, 236, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(186, 231, 153, 117, 210, 184, 65, 240)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__7_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "FunProp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__10_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__8_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(142, 123, 70, 114, 125, 89, 248, 238)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__10_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__10_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__11_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Types"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__11_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__11_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__12_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__10_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__11_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(178, 56, 70, 123, 209, 249, 1, 42)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__12_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__12_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__13_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__12_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(179, 58, 144, 195, 214, 41, 54, 156)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__13_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__13_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__14_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__13_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(206, 220, 136, 59, 19, 224, 209, 157)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__14_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__14_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__15_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__14_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(154, 23, 220, 112, 117, 8, 121, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__15_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__15_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__16_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__15_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(111, 76, 48, 17, 137, 128, 198, 68)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__16_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__16_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__17_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__17_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__17_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__18_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__16_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__17_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(110, 203, 130, 13, 3, 97, 21, 59)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__18_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__18_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__19_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__19_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__19_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__20_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__18_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__19_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(239, 151, 231, 25, 27, 63, 239, 195)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__20_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__20_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__21_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__20_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__6_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(18, 192, 241, 203, 204, 201, 141, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__21_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__21_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__22_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__21_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(23, 151, 185, 230, 75, 179, 204, 14)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__22_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__22_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__23_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__22_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__9_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(86, 11, 172, 121, 3, 94, 193, 81)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__23_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__23_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__24_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__23_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__11_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(154, 58, 135, 250, 63, 98, 42, 112)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__24_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__24_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__25_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__24_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)(((size_t)(33604414) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(159, 74, 123, 76, 61, 204, 247, 193)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__25_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__25_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__26_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__26_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__26_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__27_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__25_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__26_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(244, 204, 64, 127, 26, 20, 189, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__27_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__27_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__28_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__28_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__28_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__29_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__27_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__28_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(32, 206, 104, 182, 26, 49, 35, 200)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__29_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__29_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__30_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__29_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(145, 98, 136, 213, 245, 246, 93, 141)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__30_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__30_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "attr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(211, 174, 49, 251, 64, 24, 251, 1)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(194, 95, 140, 15, 16, 100, 236, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(186, 231, 153, 117, 210, 184, 65, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(176, 68, 102, 141, 146, 150, 56, 47)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__24_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1258255133) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(220, 133, 230, 188, 251, 121, 55, 243)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__26_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(243, 193, 178, 124, 89, 143, 104, 122)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__28_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(51, 160, 99, 89, 202, 65, 5, 68)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(118, 223, 179, 169, 93, 182, 148, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Debug"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(167, 248, 27, 31, 3, 126, 142, 13)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__0_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(119, 140, 6, 58, 231, 192, 8, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(246, 39, 251, 153, 6, 255, 160, 132)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(94, 207, 73, 80, 221, 75, 157, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_decl_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_decl_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_fvar_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_fvar_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instInhabitedOrigin_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedOrigin_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedOrigin_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedOrigin_default = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedOrigin_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedOrigin = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedOrigin_default___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_name(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_name___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_getValue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_getValue___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_ppOrigin_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin_x27___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_ppOrigin_x27___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnOrigin(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnOrigin___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__0_value),LEAN_SCALAR_PTR_LITERAL(223, 78, 141, 85, 50, 255, 216, 83)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "comp"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__2_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__3_value),LEAN_SCALAR_PTR_LITERAL(38, 235, 97, 97, 37, 43, 137, 69)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "const"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__2_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__5_value),LEAN_SCALAR_PTR_LITERAL(231, 33, 22, 82, 100, 121, 126, 178)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "HasUncurry"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "uncurry"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__2_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__7_value),LEAN_SCALAR_PTR_LITERAL(114, 26, 20, 28, 208, 159, 153, 136)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__8_value),LEAN_SCALAR_PTR_LITERAL(3, 56, 213, 201, 95, 118, 112, 151)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__2_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__8_value),LEAN_SCALAR_PTR_LITERAL(116, 38, 84, 18, 42, 97, 149, 205)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__10_value;
static const lean_array_object lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 246}, .m_size = 5, .m_capacity = 5, .m_data = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instInhabitedConfig_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(100000) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedConfig_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedConfig_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedConfig_default = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedConfig_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedConfig = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedConfig_default___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorem_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1000) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorem_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorem_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorem_default = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorem_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorem = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorem_default___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__1;
static const lean_array_object lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Context_increaseTransitionDepth(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_defaultUnfoldPred(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultUnfoldPred___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_increaseSteps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "fun_prop failed, maximum number("};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_increaseSteps___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_increaseSteps___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_increaseSteps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = ") of steps exceeded"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_increaseSteps___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_increaseSteps___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_increaseSteps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_increaseSteps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "maximum transition depth ("};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 120, .m_capacity = 120, .m_length = 119, .m_data = ") reached\n    if you want `fun_prop` to continue then increase the maximum depth with `fun_prop (maxTransitionDepth := "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ")`"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00Mathlib_Meta_FunProp_logError_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00Mathlib_Meta_FunProp_logError_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_logError___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_logError___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_logError(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_logError___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_74_; uint8_t v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_74_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_));
v___x_75_ = 0;
v___x_76_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__30_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_));
v___x_77_ = l_Lean_registerTraceClass(v___x_74_, v___x_75_, v___x_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2____boxed(lean_object* v_a_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_();
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_99_; uint8_t v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_99_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_));
v___x_100_ = 0;
v___x_101_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_));
v___x_102_ = l_Lean_registerTraceClass(v___x_99_, v___x_100_, v___x_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2____boxed(lean_object* v_a_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_();
return v_res_104_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_111_ = lean_unsigned_to_nat(4092581443u);
v___x_112_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__24_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_));
v___x_113_ = l_Lean_Name_num___override(v___x_112_, v___x_111_);
return v___x_113_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_114_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__26_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_));
v___x_115_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__2_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_);
v___x_116_ = l_Lean_Name_str___override(v___x_115_, v___x_114_);
return v___x_116_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_117_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__28_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_));
v___x_118_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_);
v___x_119_ = l_Lean_Name_str___override(v___x_118_, v___x_117_);
return v___x_119_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_120_ = lean_unsigned_to_nat(2u);
v___x_121_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__4_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_);
v___x_122_ = l_Lean_Name_num___override(v___x_121_, v___x_120_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_124_; uint8_t v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_124_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__1_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_));
v___x_125_ = 0;
v___x_126_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__5_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_);
v___x_127_ = l_Lean_registerTraceClass(v___x_124_, v___x_125_, v___x_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2____boxed(lean_object* v_a_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_();
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorIdx(lean_object* v_x_130_){
_start:
{
if (lean_obj_tag(v_x_130_) == 0)
{
lean_object* v___x_131_; 
v___x_131_ = lean_unsigned_to_nat(0u);
return v___x_131_;
}
else
{
lean_object* v___x_132_; 
v___x_132_ = lean_unsigned_to_nat(1u);
return v___x_132_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorIdx___boxed(lean_object* v_x_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorIdx(v_x_133_);
lean_dec_ref(v_x_133_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim___redArg(lean_object* v_t_135_, lean_object* v_k_136_){
_start:
{
lean_object* v_name_137_; lean_object* v___x_138_; 
v_name_137_ = lean_ctor_get(v_t_135_, 0);
lean_inc(v_name_137_);
lean_dec_ref(v_t_135_);
v___x_138_ = lean_apply_1(v_k_136_, v_name_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim(lean_object* v_motive_139_, lean_object* v_ctorIdx_140_, lean_object* v_t_141_, lean_object* v_h_142_, lean_object* v_k_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim___redArg(v_t_141_, v_k_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim___boxed(lean_object* v_motive_145_, lean_object* v_ctorIdx_146_, lean_object* v_t_147_, lean_object* v_h_148_, lean_object* v_k_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim(v_motive_145_, v_ctorIdx_146_, v_t_147_, v_h_148_, v_k_149_);
lean_dec(v_ctorIdx_146_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_decl_elim___redArg(lean_object* v_t_151_, lean_object* v_decl_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim___redArg(v_t_151_, v_decl_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_decl_elim(lean_object* v_motive_154_, lean_object* v_t_155_, lean_object* v_h_156_, lean_object* v_decl_157_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim___redArg(v_t_155_, v_decl_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_fvar_elim___redArg(lean_object* v_t_159_, lean_object* v_fvar_160_){
_start:
{
lean_object* v___x_161_; 
v___x_161_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim___redArg(v_t_159_, v_fvar_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_fvar_elim(lean_object* v_motive_162_, lean_object* v_t_163_, lean_object* v_h_164_, lean_object* v_fvar_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_ctorElim___redArg(v_t_163_, v_fvar_165_);
return v___x_166_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin_beq(lean_object* v_x_171_, lean_object* v_x_172_){
_start:
{
if (lean_obj_tag(v_x_171_) == 0)
{
if (lean_obj_tag(v_x_172_) == 0)
{
lean_object* v_name_173_; lean_object* v_name_174_; uint8_t v___x_175_; 
v_name_173_ = lean_ctor_get(v_x_171_, 0);
v_name_174_ = lean_ctor_get(v_x_172_, 0);
v___x_175_ = lean_name_eq(v_name_173_, v_name_174_);
return v___x_175_;
}
else
{
uint8_t v___x_176_; 
v___x_176_ = 0;
return v___x_176_;
}
}
else
{
if (lean_obj_tag(v_x_172_) == 1)
{
lean_object* v_fvarId_177_; lean_object* v_fvarId_178_; uint8_t v___x_179_; 
v_fvarId_177_ = lean_ctor_get(v_x_171_, 0);
v_fvarId_178_ = lean_ctor_get(v_x_172_, 0);
v___x_179_ = l_Lean_instBEqFVarId_beq(v_fvarId_177_, v_fvarId_178_);
return v___x_179_;
}
else
{
uint8_t v___x_180_; 
v___x_180_ = 0;
return v___x_180_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin_beq___boxed(lean_object* v_x_181_, lean_object* v_x_182_){
_start:
{
uint8_t v_res_183_; lean_object* v_r_184_; 
v_res_183_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqOrigin_beq(v_x_181_, v_x_182_);
lean_dec_ref(v_x_182_);
lean_dec_ref(v_x_181_);
v_r_184_ = lean_box(v_res_183_);
return v_r_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_name(lean_object* v_origin_187_){
_start:
{
lean_object* v_name_188_; 
v_name_188_ = lean_ctor_get(v_origin_187_, 0);
lean_inc(v_name_188_);
return v_name_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_name___boxed(lean_object* v_origin_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_name(v_origin_189_);
lean_dec_ref(v_origin_189_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_getValue(lean_object* v_origin_191_, lean_object* v_a_192_, lean_object* v_a_193_, lean_object* v_a_194_, lean_object* v_a_195_){
_start:
{
if (lean_obj_tag(v_origin_191_) == 0)
{
lean_object* v_name_197_; lean_object* v___x_198_; 
v_name_197_ = lean_ctor_get(v_origin_191_, 0);
lean_inc(v_name_197_);
lean_dec_ref_known(v_origin_191_, 1);
v___x_198_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_name_197_, v_a_192_, v_a_193_, v_a_194_, v_a_195_);
return v___x_198_;
}
else
{
lean_object* v_fvarId_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_207_; 
v_fvarId_199_ = lean_ctor_get(v_origin_191_, 0);
v_isSharedCheck_207_ = !lean_is_exclusive(v_origin_191_);
if (v_isSharedCheck_207_ == 0)
{
v___x_201_ = v_origin_191_;
v_isShared_202_ = v_isSharedCheck_207_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_fvarId_199_);
lean_dec(v_origin_191_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_207_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v___x_203_; lean_object* v___x_205_; 
v___x_203_ = l_Lean_Expr_fvar___override(v_fvarId_199_);
if (v_isShared_202_ == 0)
{
lean_ctor_set_tag(v___x_201_, 0);
lean_ctor_set(v___x_201_, 0, v___x_203_);
v___x_205_ = v___x_201_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v___x_203_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_getValue___boxed(lean_object* v_origin_208_, lean_object* v_a_209_, lean_object* v_a_210_, lean_object* v_a_211_, lean_object* v_a_212_, lean_object* v_a_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_getValue(v_origin_208_, v_a_209_, v_a_210_, v_a_211_, v_a_212_);
lean_dec(v_a_212_);
lean_dec_ref(v_a_211_);
lean_dec(v_a_210_);
lean_dec_ref(v_a_209_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin___redArg___lam__0(lean_object* v_toPure_215_, lean_object* v_____do__lift_216_){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_217_ = l_Lean_MessageData_ofExpr(v_____do__lift_216_);
v___x_218_ = lean_apply_2(v_toPure_215_, lean_box(0), v___x_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin___redArg(lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_x_222_){
_start:
{
if (lean_obj_tag(v_x_222_) == 0)
{
lean_object* v_toApplicative_223_; lean_object* v_toBind_224_; lean_object* v_toPure_225_; lean_object* v_name_226_; lean_object* v___f_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v_toApplicative_223_ = lean_ctor_get(v_inst_219_, 0);
v_toBind_224_ = lean_ctor_get(v_inst_219_, 1);
lean_inc(v_toBind_224_);
v_toPure_225_ = lean_ctor_get(v_toApplicative_223_, 1);
v_name_226_ = lean_ctor_get(v_x_222_, 0);
lean_inc(v_name_226_);
lean_dec_ref_known(v_x_222_, 1);
lean_inc(v_toPure_225_);
v___f_227_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_ppOrigin___redArg___lam__0), 2, 1);
lean_closure_set(v___f_227_, 0, v_toPure_225_);
v___x_228_ = l_Lean_mkConstWithLevelParams___redArg(v_inst_219_, v_inst_220_, v_inst_221_, v_name_226_);
v___x_229_ = lean_apply_4(v_toBind_224_, lean_box(0), lean_box(0), v___x_228_, v___f_227_);
return v___x_229_;
}
else
{
lean_object* v_toApplicative_230_; lean_object* v_toPure_231_; lean_object* v_fvarId_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v_toApplicative_230_ = lean_ctor_get(v_inst_219_, 0);
lean_inc_ref(v_toApplicative_230_);
lean_dec_ref(v_inst_221_);
lean_dec_ref(v_inst_220_);
lean_dec_ref(v_inst_219_);
v_toPure_231_ = lean_ctor_get(v_toApplicative_230_, 1);
lean_inc(v_toPure_231_);
lean_dec_ref(v_toApplicative_230_);
v_fvarId_232_ = lean_ctor_get(v_x_222_, 0);
lean_inc(v_fvarId_232_);
lean_dec_ref_known(v_x_222_, 1);
v___x_233_ = l_Lean_mkFVar(v_fvarId_232_);
v___x_234_ = l_Lean_MessageData_ofExpr(v___x_233_);
v___x_235_ = lean_apply_2(v_toPure_231_, lean_box(0), v___x_234_);
return v___x_235_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin(lean_object* v_m_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_x_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_mathlib_Mathlib_Meta_FunProp_ppOrigin___redArg(v_inst_237_, v_inst_238_, v_inst_239_, v_x_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin_x27(lean_object* v_origin_243_, lean_object* v_a_244_, lean_object* v_a_245_, lean_object* v_a_246_, lean_object* v_a_247_){
_start:
{
lean_object* v___y_250_; 
if (lean_obj_tag(v_origin_243_) == 1)
{
lean_object* v_fvarId_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v_fvarId_254_ = lean_ctor_get(v_origin_243_, 0);
lean_inc(v_fvarId_254_);
lean_dec_ref_known(v_origin_243_, 1);
v___x_255_ = l_Lean_Expr_fvar___override(v_fvarId_254_);
lean_inc_ref(v___x_255_);
v___x_256_ = l_Lean_Meta_ppExpr(v___x_255_, v_a_244_, v_a_245_, v_a_246_, v_a_247_);
if (lean_obj_tag(v___x_256_) == 0)
{
lean_object* v_a_257_; lean_object* v___x_258_; 
v_a_257_ = lean_ctor_get(v___x_256_, 0);
lean_inc(v_a_257_);
lean_dec_ref_known(v___x_256_, 1);
lean_inc(v_a_247_);
lean_inc_ref(v_a_246_);
lean_inc(v_a_245_);
lean_inc_ref(v_a_244_);
v___x_258_ = lean_infer_type(v___x_255_, v_a_244_, v_a_245_, v_a_246_, v_a_247_);
if (lean_obj_tag(v___x_258_) == 0)
{
lean_object* v_a_259_; lean_object* v___x_260_; 
v_a_259_ = lean_ctor_get(v___x_258_, 0);
lean_inc(v_a_259_);
lean_dec_ref_known(v___x_258_, 1);
v___x_260_ = l_Lean_Meta_ppExpr(v_a_259_, v_a_244_, v_a_245_, v_a_246_, v_a_247_);
if (lean_obj_tag(v___x_260_) == 0)
{
lean_object* v_a_261_; lean_object* v___x_263_; uint8_t v_isShared_264_; uint8_t v_isSharedCheck_275_; 
v_a_261_ = lean_ctor_get(v___x_260_, 0);
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_260_);
if (v_isSharedCheck_275_ == 0)
{
v___x_263_ = v___x_260_;
v_isShared_264_ = v_isSharedCheck_275_;
goto v_resetjp_262_;
}
else
{
lean_inc(v_a_261_);
lean_dec(v___x_260_);
v___x_263_ = lean_box(0);
v_isShared_264_ = v_isSharedCheck_275_;
goto v_resetjp_262_;
}
v_resetjp_262_:
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_273_; 
v___x_265_ = l_Std_Format_defWidth;
v___x_266_ = lean_unsigned_to_nat(0u);
v___x_267_ = l_Std_Format_pretty(v_a_257_, v___x_265_, v___x_266_, v___x_266_);
v___x_268_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_ppOrigin_x27___closed__0));
v___x_269_ = lean_string_append(v___x_267_, v___x_268_);
v___x_270_ = l_Std_Format_pretty(v_a_261_, v___x_265_, v___x_266_, v___x_266_);
v___x_271_ = lean_string_append(v___x_269_, v___x_270_);
lean_dec_ref(v___x_270_);
if (v_isShared_264_ == 0)
{
lean_ctor_set(v___x_263_, 0, v___x_271_);
v___x_273_ = v___x_263_;
goto v_reusejp_272_;
}
else
{
lean_object* v_reuseFailAlloc_274_; 
v_reuseFailAlloc_274_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_274_, 0, v___x_271_);
v___x_273_ = v_reuseFailAlloc_274_;
goto v_reusejp_272_;
}
v_reusejp_272_:
{
return v___x_273_;
}
}
}
else
{
lean_object* v_a_276_; lean_object* v___x_278_; uint8_t v_isShared_279_; uint8_t v_isSharedCheck_283_; 
lean_dec(v_a_257_);
v_a_276_ = lean_ctor_get(v___x_260_, 0);
v_isSharedCheck_283_ = !lean_is_exclusive(v___x_260_);
if (v_isSharedCheck_283_ == 0)
{
v___x_278_ = v___x_260_;
v_isShared_279_ = v_isSharedCheck_283_;
goto v_resetjp_277_;
}
else
{
lean_inc(v_a_276_);
lean_dec(v___x_260_);
v___x_278_ = lean_box(0);
v_isShared_279_ = v_isSharedCheck_283_;
goto v_resetjp_277_;
}
v_resetjp_277_:
{
lean_object* v___x_281_; 
if (v_isShared_279_ == 0)
{
v___x_281_ = v___x_278_;
goto v_reusejp_280_;
}
else
{
lean_object* v_reuseFailAlloc_282_; 
v_reuseFailAlloc_282_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_282_, 0, v_a_276_);
v___x_281_ = v_reuseFailAlloc_282_;
goto v_reusejp_280_;
}
v_reusejp_280_:
{
return v___x_281_;
}
}
}
}
else
{
lean_object* v_a_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_291_; 
lean_dec(v_a_257_);
v_a_284_ = lean_ctor_get(v___x_258_, 0);
v_isSharedCheck_291_ = !lean_is_exclusive(v___x_258_);
if (v_isSharedCheck_291_ == 0)
{
v___x_286_ = v___x_258_;
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_a_284_);
lean_dec(v___x_258_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___x_289_; 
if (v_isShared_287_ == 0)
{
v___x_289_ = v___x_286_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v_a_284_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
}
else
{
lean_object* v_a_292_; lean_object* v___x_294_; uint8_t v_isShared_295_; uint8_t v_isSharedCheck_299_; 
lean_dec_ref(v___x_255_);
v_a_292_ = lean_ctor_get(v___x_256_, 0);
v_isSharedCheck_299_ = !lean_is_exclusive(v___x_256_);
if (v_isSharedCheck_299_ == 0)
{
v___x_294_ = v___x_256_;
v_isShared_295_ = v_isSharedCheck_299_;
goto v_resetjp_293_;
}
else
{
lean_inc(v_a_292_);
lean_dec(v___x_256_);
v___x_294_ = lean_box(0);
v_isShared_295_ = v_isSharedCheck_299_;
goto v_resetjp_293_;
}
v_resetjp_293_:
{
lean_object* v___x_297_; 
if (v_isShared_295_ == 0)
{
v___x_297_ = v___x_294_;
goto v_reusejp_296_;
}
else
{
lean_object* v_reuseFailAlloc_298_; 
v_reuseFailAlloc_298_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_298_, 0, v_a_292_);
v___x_297_ = v_reuseFailAlloc_298_;
goto v_reusejp_296_;
}
v_reusejp_296_:
{
return v___x_297_;
}
}
}
}
else
{
lean_object* v_name_300_; 
v_name_300_ = lean_ctor_get(v_origin_243_, 0);
lean_inc(v_name_300_);
lean_dec_ref(v_origin_243_);
v___y_250_ = v_name_300_;
goto v___jp_249_;
}
v___jp_249_:
{
uint8_t v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; 
v___x_251_ = 1;
v___x_252_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___y_250_, v___x_251_);
v___x_253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_253_, 0, v___x_252_);
return v___x_253_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_ppOrigin_x27___boxed(lean_object* v_origin_301_, lean_object* v_a_302_, lean_object* v_a_303_, lean_object* v_a_304_, lean_object* v_a_305_, lean_object* v_a_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_mathlib_Mathlib_Meta_FunProp_ppOrigin_x27(v_origin_301_, v_a_302_, v_a_303_, v_a_304_, v_a_305_);
lean_dec(v_a_305_);
lean_dec_ref(v_a_304_);
lean_dec(v_a_303_);
lean_dec_ref(v_a_302_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnOrigin(lean_object* v_fData_308_){
_start:
{
lean_object* v_fn_309_; 
v_fn_309_ = lean_ctor_get(v_fData_308_, 2);
switch(lean_obj_tag(v_fn_309_))
{
case 1:
{
lean_object* v_fvarId_310_; lean_object* v___x_311_; 
v_fvarId_310_ = lean_ctor_get(v_fn_309_, 0);
lean_inc(v_fvarId_310_);
v___x_311_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_311_, 0, v_fvarId_310_);
return v___x_311_;
}
case 4:
{
lean_object* v_declName_312_; lean_object* v___x_313_; 
v_declName_312_ = lean_ctor_get(v_fn_309_, 0);
lean_inc(v_declName_312_);
v___x_313_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_313_, 0, v_declName_312_);
return v___x_313_;
}
default: 
{
lean_object* v___x_314_; 
v___x_314_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedOrigin_default___closed__0));
return v___x_314_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnOrigin___boxed(lean_object* v_fData_315_){
_start:
{
lean_object* v_res_316_; 
v_res_316_ = lp_mathlib_Mathlib_Meta_FunProp_FunctionData_getFnOrigin(v_fData_315_);
lean_dec_ref(v_fData_315_);
return v_res_316_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig_beq(lean_object* v_x_356_, lean_object* v_x_357_){
_start:
{
lean_object* v_maxTransitionDepth_358_; lean_object* v_maxSteps_359_; lean_object* v_maxTransitionDepth_360_; lean_object* v_maxSteps_361_; uint8_t v___x_362_; 
v_maxTransitionDepth_358_ = lean_ctor_get(v_x_356_, 0);
v_maxSteps_359_ = lean_ctor_get(v_x_356_, 1);
v_maxTransitionDepth_360_ = lean_ctor_get(v_x_357_, 0);
v_maxSteps_361_ = lean_ctor_get(v_x_357_, 1);
v___x_362_ = lean_nat_dec_eq(v_maxTransitionDepth_358_, v_maxTransitionDepth_360_);
if (v___x_362_ == 0)
{
return v___x_362_;
}
else
{
uint8_t v___x_363_; 
v___x_363_ = lean_nat_dec_eq(v_maxSteps_359_, v_maxSteps_361_);
return v___x_363_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig_beq___boxed(lean_object* v_x_364_, lean_object* v_x_365_){
_start:
{
uint8_t v_res_366_; lean_object* v_r_367_; 
v_res_366_ = lp_mathlib_Mathlib_Meta_FunProp_instBEqConfig_beq(v_x_364_, v_x_365_);
lean_dec_ref(v_x_365_);
lean_dec_ref(v_x_364_);
v_r_367_ = lean_box(v_res_366_);
return v_r_367_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__0(void){
_start:
{
lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; 
v___x_376_ = lean_box(0);
v___x_377_ = lean_unsigned_to_nat(16u);
v___x_378_ = lean_mk_array(v___x_377_, v___x_376_);
return v___x_378_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__1(void){
_start:
{
lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; 
v___x_379_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__0, &lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__0_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__0);
v___x_380_ = lean_unsigned_to_nat(0u);
v___x_381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
lean_ctor_set(v___x_381_, 1, v___x_379_);
return v___x_381_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__3(void){
_start:
{
lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; 
v___x_384_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__2));
v___x_385_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__1, &lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__1_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__1);
v___x_386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_386_, 0, v___x_385_);
lean_ctor_set(v___x_386_, 1, v___x_384_);
return v___x_386_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default(void){
_start:
{
lean_object* v___x_387_; 
v___x_387_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__3, &lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__3_once, _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default___closed__3);
return v___x_387_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems(void){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default;
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Context_increaseTransitionDepth(lean_object* v_ctx_389_){
_start:
{
lean_object* v_config_390_; lean_object* v_constToUnfold_391_; lean_object* v_disch_392_; lean_object* v_transitionDepth_393_; lean_object* v___x_395_; uint8_t v_isShared_396_; uint8_t v_isSharedCheck_402_; 
v_config_390_ = lean_ctor_get(v_ctx_389_, 0);
v_constToUnfold_391_ = lean_ctor_get(v_ctx_389_, 1);
v_disch_392_ = lean_ctor_get(v_ctx_389_, 2);
v_transitionDepth_393_ = lean_ctor_get(v_ctx_389_, 3);
v_isSharedCheck_402_ = !lean_is_exclusive(v_ctx_389_);
if (v_isSharedCheck_402_ == 0)
{
v___x_395_ = v_ctx_389_;
v_isShared_396_ = v_isSharedCheck_402_;
goto v_resetjp_394_;
}
else
{
lean_inc(v_transitionDepth_393_);
lean_inc(v_disch_392_);
lean_inc(v_constToUnfold_391_);
lean_inc(v_config_390_);
lean_dec(v_ctx_389_);
v___x_395_ = lean_box(0);
v_isShared_396_ = v_isSharedCheck_402_;
goto v_resetjp_394_;
}
v_resetjp_394_:
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_400_; 
v___x_397_ = lean_unsigned_to_nat(1u);
v___x_398_ = lean_nat_add(v_transitionDepth_393_, v___x_397_);
lean_dec(v_transitionDepth_393_);
if (v_isShared_396_ == 0)
{
lean_ctor_set(v___x_395_, 3, v___x_398_);
v___x_400_ = v___x_395_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v_config_390_);
lean_ctor_set(v_reuseFailAlloc_401_, 1, v_constToUnfold_391_);
lean_ctor_set(v_reuseFailAlloc_401_, 2, v_disch_392_);
lean_ctor_set(v_reuseFailAlloc_401_, 3, v___x_398_);
v___x_400_ = v_reuseFailAlloc_401_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
return v___x_400_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0_spec__0(lean_object* v_a_403_, lean_object* v_as_404_, size_t v_i_405_, size_t v_stop_406_){
_start:
{
uint8_t v___x_407_; 
v___x_407_ = lean_usize_dec_eq(v_i_405_, v_stop_406_);
if (v___x_407_ == 0)
{
lean_object* v___x_408_; uint8_t v___x_409_; 
v___x_408_ = lean_array_uget_borrowed(v_as_404_, v_i_405_);
v___x_409_ = lean_name_eq(v_a_403_, v___x_408_);
if (v___x_409_ == 0)
{
size_t v___x_410_; size_t v___x_411_; 
v___x_410_ = ((size_t)1ULL);
v___x_411_ = lean_usize_add(v_i_405_, v___x_410_);
v_i_405_ = v___x_411_;
goto _start;
}
else
{
return v___x_409_;
}
}
else
{
uint8_t v___x_413_; 
v___x_413_ = 0;
return v___x_413_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0_spec__0___boxed(lean_object* v_a_414_, lean_object* v_as_415_, lean_object* v_i_416_, lean_object* v_stop_417_){
_start:
{
size_t v_i_boxed_418_; size_t v_stop_boxed_419_; uint8_t v_res_420_; lean_object* v_r_421_; 
v_i_boxed_418_ = lean_unbox_usize(v_i_416_);
lean_dec(v_i_416_);
v_stop_boxed_419_ = lean_unbox_usize(v_stop_417_);
lean_dec(v_stop_417_);
v_res_420_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0_spec__0(v_a_414_, v_as_415_, v_i_boxed_418_, v_stop_boxed_419_);
lean_dec_ref(v_as_415_);
lean_dec(v_a_414_);
v_r_421_ = lean_box(v_res_420_);
return v_r_421_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0(lean_object* v_as_422_, lean_object* v_a_423_){
_start:
{
lean_object* v___x_424_; lean_object* v___x_425_; uint8_t v___x_426_; 
v___x_424_ = lean_unsigned_to_nat(0u);
v___x_425_ = lean_array_get_size(v_as_422_);
v___x_426_ = lean_nat_dec_lt(v___x_424_, v___x_425_);
if (v___x_426_ == 0)
{
return v___x_426_;
}
else
{
if (v___x_426_ == 0)
{
return v___x_426_;
}
else
{
size_t v___x_427_; size_t v___x_428_; uint8_t v___x_429_; 
v___x_427_ = ((size_t)0ULL);
v___x_428_ = lean_usize_of_nat(v___x_425_);
v___x_429_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0_spec__0(v_a_423_, v_as_422_, v___x_427_, v___x_428_);
return v___x_429_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0___boxed(lean_object* v_as_430_, lean_object* v_a_431_){
_start:
{
uint8_t v_res_432_; lean_object* v_r_433_; 
v_res_432_ = lp_mathlib_Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0(v_as_430_, v_a_431_);
lean_dec(v_a_431_);
lean_dec_ref(v_as_430_);
v_r_433_ = lean_box(v_res_432_);
return v_r_433_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_defaultUnfoldPred(lean_object* v_a_434_){
_start:
{
lean_object* v___x_435_; uint8_t v___x_436_; 
v___x_435_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold));
v___x_436_ = lp_mathlib_Array_contains___at___00Mathlib_Meta_FunProp_defaultUnfoldPred_spec__0(v___x_435_, v_a_434_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultUnfoldPred___boxed(lean_object* v_a_437_){
_start:
{
uint8_t v_res_438_; lean_object* v_r_439_; 
v_res_438_ = lp_mathlib_Mathlib_Meta_FunProp_defaultUnfoldPred(v_a_437_);
lean_dec(v_a_437_);
v_r_439_ = lean_box(v_res_438_);
return v_r_439_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0___redArg(lean_object* v_k_440_, lean_object* v_t_441_){
_start:
{
if (lean_obj_tag(v_t_441_) == 0)
{
lean_object* v_k_442_; lean_object* v_l_443_; lean_object* v_r_444_; uint8_t v___x_445_; 
v_k_442_ = lean_ctor_get(v_t_441_, 1);
v_l_443_ = lean_ctor_get(v_t_441_, 3);
v_r_444_ = lean_ctor_get(v_t_441_, 4);
v___x_445_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_440_, v_k_442_);
switch(v___x_445_)
{
case 0:
{
v_t_441_ = v_l_443_;
goto _start;
}
case 1:
{
uint8_t v___x_447_; 
v___x_447_ = 1;
return v___x_447_;
}
default: 
{
v_t_441_ = v_r_444_;
goto _start;
}
}
}
else
{
uint8_t v___x_449_; 
v___x_449_ = 0;
return v___x_449_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0___redArg___boxed(lean_object* v_k_450_, lean_object* v_t_451_){
_start:
{
uint8_t v_res_452_; lean_object* v_r_453_; 
v_res_452_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0___redArg(v_k_450_, v_t_451_);
lean_dec(v_t_451_);
lean_dec(v_k_450_);
v_r_453_ = lean_box(v_res_452_);
return v_r_453_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg___lam__0(lean_object* v_constToUnfold_454_, lean_object* v_n_455_){
_start:
{
uint8_t v___x_456_; 
v___x_456_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0___redArg(v_n_455_, v_constToUnfold_454_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg___lam__0___boxed(lean_object* v_constToUnfold_457_, lean_object* v_n_458_){
_start:
{
uint8_t v_res_459_; lean_object* v_r_460_; 
v_res_459_ = lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg___lam__0(v_constToUnfold_457_, v_n_458_);
lean_dec(v_n_458_);
lean_dec(v_constToUnfold_457_);
v_r_460_ = lean_box(v_res_459_);
return v_r_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg(lean_object* v_a_461_, lean_object* v_a_462_){
_start:
{
lean_object* v_constToUnfold_464_; lean_object* v___f_465_; lean_object* v___x_466_; lean_object* v___x_467_; 
v_constToUnfold_464_ = lean_ctor_get(v_a_461_, 1);
lean_inc(v_constToUnfold_464_);
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_465_, 0, v_constToUnfold_464_);
v___x_466_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_466_, 0, v___f_465_);
lean_ctor_set(v___x_466_, 1, v_a_462_);
v___x_467_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_467_, 0, v___x_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg___boxed(lean_object* v_a_468_, lean_object* v_a_469_, lean_object* v_a_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg(v_a_468_, v_a_469_);
lean_dec_ref(v_a_468_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred(lean_object* v_a_472_, lean_object* v_a_473_, lean_object* v_a_474_, lean_object* v_a_475_, lean_object* v_a_476_, lean_object* v_a_477_){
_start:
{
lean_object* v___x_479_; 
v___x_479_ = lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___redArg(v_a_472_, v_a_473_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred___boxed(lean_object* v_a_480_, lean_object* v_a_481_, lean_object* v_a_482_, lean_object* v_a_483_, lean_object* v_a_484_, lean_object* v_a_485_, lean_object* v_a_486_){
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_mathlib_Mathlib_Meta_FunProp_unfoldNamePred(v_a_480_, v_a_481_, v_a_482_, v_a_483_, v_a_484_, v_a_485_);
lean_dec(v_a_485_);
lean_dec_ref(v_a_484_);
lean_dec(v_a_483_);
lean_dec_ref(v_a_482_);
lean_dec_ref(v_a_480_);
return v_res_487_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0(lean_object* v_00_u03b2_488_, lean_object* v_k_489_, lean_object* v_t_490_){
_start:
{
uint8_t v___x_491_; 
v___x_491_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0___redArg(v_k_489_, v_t_490_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0___boxed(lean_object* v_00_u03b2_492_, lean_object* v_k_493_, lean_object* v_t_494_){
_start:
{
uint8_t v_res_495_; lean_object* v_r_496_; 
v_res_495_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Mathlib_Meta_FunProp_unfoldNamePred_spec__0(v_00_u03b2_492_, v_k_493_, v_t_494_);
lean_dec(v_t_494_);
lean_dec(v_k_493_);
v_r_496_ = lean_box(v_res_495_);
return v_r_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0_spec__0(lean_object* v_msgData_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_){
_start:
{
lean_object* v___x_503_; lean_object* v_env_504_; lean_object* v___x_505_; lean_object* v_mctx_506_; lean_object* v_lctx_507_; lean_object* v_options_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; 
v___x_503_ = lean_st_ref_get(v___y_501_);
v_env_504_ = lean_ctor_get(v___x_503_, 0);
lean_inc_ref(v_env_504_);
lean_dec(v___x_503_);
v___x_505_ = lean_st_ref_get(v___y_499_);
v_mctx_506_ = lean_ctor_get(v___x_505_, 0);
lean_inc_ref(v_mctx_506_);
lean_dec(v___x_505_);
v_lctx_507_ = lean_ctor_get(v___y_498_, 2);
v_options_508_ = lean_ctor_get(v___y_500_, 2);
lean_inc_ref(v_options_508_);
lean_inc_ref(v_lctx_507_);
v___x_509_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_509_, 0, v_env_504_);
lean_ctor_set(v___x_509_, 1, v_mctx_506_);
lean_ctor_set(v___x_509_, 2, v_lctx_507_);
lean_ctor_set(v___x_509_, 3, v_options_508_);
v___x_510_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_510_, 0, v___x_509_);
lean_ctor_set(v___x_510_, 1, v_msgData_497_);
v___x_511_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_511_, 0, v___x_510_);
return v___x_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0_spec__0___boxed(lean_object* v_msgData_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_){
_start:
{
lean_object* v_res_518_; 
v_res_518_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0_spec__0(v_msgData_512_, v___y_513_, v___y_514_, v___y_515_, v___y_516_);
lean_dec(v___y_516_);
lean_dec_ref(v___y_515_);
lean_dec(v___y_514_);
lean_dec_ref(v___y_513_);
return v_res_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0___redArg(lean_object* v_msg_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_){
_start:
{
lean_object* v_ref_525_; lean_object* v___x_526_; lean_object* v_a_527_; lean_object* v___x_529_; uint8_t v_isShared_530_; uint8_t v_isSharedCheck_535_; 
v_ref_525_ = lean_ctor_get(v___y_522_, 5);
v___x_526_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0_spec__0(v_msg_519_, v___y_520_, v___y_521_, v___y_522_, v___y_523_);
v_a_527_ = lean_ctor_get(v___x_526_, 0);
v_isSharedCheck_535_ = !lean_is_exclusive(v___x_526_);
if (v_isSharedCheck_535_ == 0)
{
v___x_529_ = v___x_526_;
v_isShared_530_ = v_isSharedCheck_535_;
goto v_resetjp_528_;
}
else
{
lean_inc(v_a_527_);
lean_dec(v___x_526_);
v___x_529_ = lean_box(0);
v_isShared_530_ = v_isSharedCheck_535_;
goto v_resetjp_528_;
}
v_resetjp_528_:
{
lean_object* v___x_531_; lean_object* v___x_533_; 
lean_inc(v_ref_525_);
v___x_531_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_531_, 0, v_ref_525_);
lean_ctor_set(v___x_531_, 1, v_a_527_);
if (v_isShared_530_ == 0)
{
lean_ctor_set_tag(v___x_529_, 1);
lean_ctor_set(v___x_529_, 0, v___x_531_);
v___x_533_ = v___x_529_;
goto v_reusejp_532_;
}
else
{
lean_object* v_reuseFailAlloc_534_; 
v_reuseFailAlloc_534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_534_, 0, v___x_531_);
v___x_533_ = v_reuseFailAlloc_534_;
goto v_reusejp_532_;
}
v_reusejp_532_:
{
return v___x_533_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0___redArg___boxed(lean_object* v_msg_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_){
_start:
{
lean_object* v_res_542_; 
v_res_542_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0___redArg(v_msg_536_, v___y_537_, v___y_538_, v___y_539_, v___y_540_);
lean_dec(v___y_540_);
lean_dec_ref(v___y_539_);
lean_dec(v___y_538_);
lean_dec_ref(v___y_537_);
return v_res_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_increaseSteps(lean_object* v_a_545_, lean_object* v_a_546_, lean_object* v_a_547_, lean_object* v_a_548_, lean_object* v_a_549_, lean_object* v_a_550_){
_start:
{
lean_object* v_cache_553_; lean_object* v_failureCache_554_; lean_object* v_numSteps_555_; lean_object* v_msgLog_556_; lean_object* v_morTheorems_557_; lean_object* v_transitionTheorems_558_; lean_object* v_config_565_; lean_object* v_cache_566_; lean_object* v_failureCache_567_; lean_object* v_numSteps_568_; lean_object* v_msgLog_569_; lean_object* v_morTheorems_570_; lean_object* v_transitionTheorems_571_; lean_object* v_maxSteps_572_; uint8_t v___x_573_; 
v_config_565_ = lean_ctor_get(v_a_545_, 0);
v_cache_566_ = lean_ctor_get(v_a_546_, 0);
lean_inc_ref(v_cache_566_);
v_failureCache_567_ = lean_ctor_get(v_a_546_, 1);
lean_inc_ref(v_failureCache_567_);
v_numSteps_568_ = lean_ctor_get(v_a_546_, 2);
lean_inc(v_numSteps_568_);
v_msgLog_569_ = lean_ctor_get(v_a_546_, 3);
lean_inc(v_msgLog_569_);
v_morTheorems_570_ = lean_ctor_get(v_a_546_, 4);
lean_inc_ref(v_morTheorems_570_);
v_transitionTheorems_571_ = lean_ctor_get(v_a_546_, 5);
lean_inc_ref(v_transitionTheorems_571_);
lean_dec_ref(v_a_546_);
v_maxSteps_572_ = lean_ctor_get(v_config_565_, 1);
v___x_573_ = lean_nat_dec_lt(v_maxSteps_572_, v_numSteps_568_);
if (v___x_573_ == 0)
{
v_cache_553_ = v_cache_566_;
v_failureCache_554_ = v_failureCache_567_;
v_numSteps_555_ = v_numSteps_568_;
v_msgLog_556_ = v_msgLog_569_;
v_morTheorems_557_ = v_morTheorems_570_;
v_transitionTheorems_558_ = v_transitionTheorems_571_;
goto v___jp_552_;
}
else
{
lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; 
lean_dec_ref(v_transitionTheorems_571_);
lean_dec_ref(v_morTheorems_570_);
lean_dec(v_msgLog_569_);
lean_dec(v_numSteps_568_);
lean_dec_ref(v_failureCache_567_);
lean_dec_ref(v_cache_566_);
v___x_574_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_increaseSteps___closed__0));
lean_inc(v_maxSteps_572_);
v___x_575_ = l_Nat_reprFast(v_maxSteps_572_);
v___x_576_ = lean_string_append(v___x_574_, v___x_575_);
lean_dec_ref(v___x_575_);
v___x_577_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_increaseSteps___closed__1));
v___x_578_ = lean_string_append(v___x_576_, v___x_577_);
v___x_579_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_579_, 0, v___x_578_);
v___x_580_ = l_Lean_MessageData_ofFormat(v___x_579_);
v___x_581_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0___redArg(v___x_580_, v_a_547_, v_a_548_, v_a_549_, v_a_550_);
return v___x_581_;
}
v___jp_552_:
{
lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; 
v___x_559_ = lean_box(0);
v___x_560_ = lean_unsigned_to_nat(1u);
v___x_561_ = lean_nat_add(v_numSteps_555_, v___x_560_);
lean_dec(v_numSteps_555_);
v___x_562_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_562_, 0, v_cache_553_);
lean_ctor_set(v___x_562_, 1, v_failureCache_554_);
lean_ctor_set(v___x_562_, 2, v___x_561_);
lean_ctor_set(v___x_562_, 3, v_msgLog_556_);
lean_ctor_set(v___x_562_, 4, v_morTheorems_557_);
lean_ctor_set(v___x_562_, 5, v_transitionTheorems_558_);
v___x_563_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_563_, 0, v___x_559_);
lean_ctor_set(v___x_563_, 1, v___x_562_);
v___x_564_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_564_, 0, v___x_563_);
return v___x_564_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_increaseSteps___boxed(lean_object* v_a_582_, lean_object* v_a_583_, lean_object* v_a_584_, lean_object* v_a_585_, lean_object* v_a_586_, lean_object* v_a_587_, lean_object* v_a_588_){
_start:
{
lean_object* v_res_589_; 
v_res_589_ = lp_mathlib_Mathlib_Meta_FunProp_increaseSteps(v_a_582_, v_a_583_, v_a_584_, v_a_585_, v_a_586_, v_a_587_);
lean_dec(v_a_587_);
lean_dec_ref(v_a_586_);
lean_dec(v_a_585_);
lean_dec_ref(v_a_584_);
lean_dec_ref(v_a_582_);
return v_res_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0(lean_object* v_00_u03b1_590_, lean_object* v_msg_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_){
_start:
{
lean_object* v___x_599_; 
v___x_599_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0___redArg(v_msg_591_, v___y_594_, v___y_595_, v___y_596_, v___y_597_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0___boxed(lean_object* v_00_u03b1_600_, lean_object* v_msg_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0(v_00_u03b1_600_, v_msg_601_, v___y_602_, v___y_603_, v___y_604_, v___y_605_, v___y_606_, v___y_607_);
lean_dec(v___y_607_);
lean_dec_ref(v___y_606_);
lean_dec(v___y_605_);
lean_dec_ref(v___y_604_);
lean_dec_ref(v___y_603_);
lean_dec_ref(v___y_602_);
return v_res_609_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_610_; double v___x_611_; 
v___x_610_ = lean_unsigned_to_nat(0u);
v___x_611_ = lean_float_of_nat(v___x_610_);
return v___x_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg(lean_object* v_cls_615_, lean_object* v_msg_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_){
_start:
{
lean_object* v_ref_623_; lean_object* v___x_624_; lean_object* v_a_625_; lean_object* v___x_627_; uint8_t v_isShared_628_; uint8_t v_isSharedCheck_670_; 
v_ref_623_ = lean_ctor_get(v___y_620_, 5);
v___x_624_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Meta_FunProp_increaseSteps_spec__0_spec__0(v_msg_616_, v___y_618_, v___y_619_, v___y_620_, v___y_621_);
v_a_625_ = lean_ctor_get(v___x_624_, 0);
v_isSharedCheck_670_ = !lean_is_exclusive(v___x_624_);
if (v_isSharedCheck_670_ == 0)
{
v___x_627_ = v___x_624_;
v_isShared_628_ = v_isSharedCheck_670_;
goto v_resetjp_626_;
}
else
{
lean_inc(v_a_625_);
lean_dec(v___x_624_);
v___x_627_ = lean_box(0);
v_isShared_628_ = v_isSharedCheck_670_;
goto v_resetjp_626_;
}
v_resetjp_626_:
{
lean_object* v___x_629_; lean_object* v_traceState_630_; lean_object* v_env_631_; lean_object* v_nextMacroScope_632_; lean_object* v_ngen_633_; lean_object* v_auxDeclNGen_634_; lean_object* v_cache_635_; lean_object* v_messages_636_; lean_object* v_infoState_637_; lean_object* v_snapshotTasks_638_; lean_object* v___x_640_; uint8_t v_isShared_641_; uint8_t v_isSharedCheck_669_; 
v___x_629_ = lean_st_ref_take(v___y_621_);
v_traceState_630_ = lean_ctor_get(v___x_629_, 4);
v_env_631_ = lean_ctor_get(v___x_629_, 0);
v_nextMacroScope_632_ = lean_ctor_get(v___x_629_, 1);
v_ngen_633_ = lean_ctor_get(v___x_629_, 2);
v_auxDeclNGen_634_ = lean_ctor_get(v___x_629_, 3);
v_cache_635_ = lean_ctor_get(v___x_629_, 5);
v_messages_636_ = lean_ctor_get(v___x_629_, 6);
v_infoState_637_ = lean_ctor_get(v___x_629_, 7);
v_snapshotTasks_638_ = lean_ctor_get(v___x_629_, 8);
v_isSharedCheck_669_ = !lean_is_exclusive(v___x_629_);
if (v_isSharedCheck_669_ == 0)
{
v___x_640_ = v___x_629_;
v_isShared_641_ = v_isSharedCheck_669_;
goto v_resetjp_639_;
}
else
{
lean_inc(v_snapshotTasks_638_);
lean_inc(v_infoState_637_);
lean_inc(v_messages_636_);
lean_inc(v_cache_635_);
lean_inc(v_traceState_630_);
lean_inc(v_auxDeclNGen_634_);
lean_inc(v_ngen_633_);
lean_inc(v_nextMacroScope_632_);
lean_inc(v_env_631_);
lean_dec(v___x_629_);
v___x_640_ = lean_box(0);
v_isShared_641_ = v_isSharedCheck_669_;
goto v_resetjp_639_;
}
v_resetjp_639_:
{
uint64_t v_tid_642_; lean_object* v_traces_643_; lean_object* v___x_645_; uint8_t v_isShared_646_; uint8_t v_isSharedCheck_668_; 
v_tid_642_ = lean_ctor_get_uint64(v_traceState_630_, sizeof(void*)*1);
v_traces_643_ = lean_ctor_get(v_traceState_630_, 0);
v_isSharedCheck_668_ = !lean_is_exclusive(v_traceState_630_);
if (v_isSharedCheck_668_ == 0)
{
v___x_645_ = v_traceState_630_;
v_isShared_646_ = v_isSharedCheck_668_;
goto v_resetjp_644_;
}
else
{
lean_inc(v_traces_643_);
lean_dec(v_traceState_630_);
v___x_645_ = lean_box(0);
v_isShared_646_ = v_isSharedCheck_668_;
goto v_resetjp_644_;
}
v_resetjp_644_:
{
lean_object* v___x_647_; double v___x_648_; uint8_t v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_657_; 
v___x_647_ = lean_box(0);
v___x_648_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__0);
v___x_649_ = 0;
v___x_650_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__1));
v___x_651_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_651_, 0, v_cls_615_);
lean_ctor_set(v___x_651_, 1, v___x_647_);
lean_ctor_set(v___x_651_, 2, v___x_650_);
lean_ctor_set_float(v___x_651_, sizeof(void*)*3, v___x_648_);
lean_ctor_set_float(v___x_651_, sizeof(void*)*3 + 8, v___x_648_);
lean_ctor_set_uint8(v___x_651_, sizeof(void*)*3 + 16, v___x_649_);
v___x_652_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___closed__2));
v___x_653_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_653_, 0, v___x_651_);
lean_ctor_set(v___x_653_, 1, v_a_625_);
lean_ctor_set(v___x_653_, 2, v___x_652_);
lean_inc(v_ref_623_);
v___x_654_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_654_, 0, v_ref_623_);
lean_ctor_set(v___x_654_, 1, v___x_653_);
v___x_655_ = l_Lean_PersistentArray_push___redArg(v_traces_643_, v___x_654_);
if (v_isShared_646_ == 0)
{
lean_ctor_set(v___x_645_, 0, v___x_655_);
v___x_657_ = v___x_645_;
goto v_reusejp_656_;
}
else
{
lean_object* v_reuseFailAlloc_667_; 
v_reuseFailAlloc_667_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_667_, 0, v___x_655_);
lean_ctor_set_uint64(v_reuseFailAlloc_667_, sizeof(void*)*1, v_tid_642_);
v___x_657_ = v_reuseFailAlloc_667_;
goto v_reusejp_656_;
}
v_reusejp_656_:
{
lean_object* v___x_659_; 
if (v_isShared_641_ == 0)
{
lean_ctor_set(v___x_640_, 4, v___x_657_);
v___x_659_ = v___x_640_;
goto v_reusejp_658_;
}
else
{
lean_object* v_reuseFailAlloc_666_; 
v_reuseFailAlloc_666_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_666_, 0, v_env_631_);
lean_ctor_set(v_reuseFailAlloc_666_, 1, v_nextMacroScope_632_);
lean_ctor_set(v_reuseFailAlloc_666_, 2, v_ngen_633_);
lean_ctor_set(v_reuseFailAlloc_666_, 3, v_auxDeclNGen_634_);
lean_ctor_set(v_reuseFailAlloc_666_, 4, v___x_657_);
lean_ctor_set(v_reuseFailAlloc_666_, 5, v_cache_635_);
lean_ctor_set(v_reuseFailAlloc_666_, 6, v_messages_636_);
lean_ctor_set(v_reuseFailAlloc_666_, 7, v_infoState_637_);
lean_ctor_set(v_reuseFailAlloc_666_, 8, v_snapshotTasks_638_);
v___x_659_ = v_reuseFailAlloc_666_;
goto v_reusejp_658_;
}
v_reusejp_658_:
{
lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_664_; 
v___x_660_ = lean_st_ref_set(v___y_621_, v___x_659_);
v___x_661_ = lean_box(0);
v___x_662_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_662_, 0, v___x_661_);
lean_ctor_set(v___x_662_, 1, v___y_617_);
if (v_isShared_628_ == 0)
{
lean_ctor_set(v___x_627_, 0, v___x_662_);
v___x_664_ = v___x_627_;
goto v_reusejp_663_;
}
else
{
lean_object* v_reuseFailAlloc_665_; 
v_reuseFailAlloc_665_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_665_, 0, v___x_662_);
v___x_664_ = v_reuseFailAlloc_665_;
goto v_reusejp_663_;
}
v_reusejp_663_:
{
return v___x_664_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg___boxed(lean_object* v_cls_671_, lean_object* v_msg_672_, lean_object* v___y_673_, lean_object* v___y_674_, lean_object* v___y_675_, lean_object* v___y_676_, lean_object* v___y_677_, lean_object* v___y_678_){
_start:
{
lean_object* v_res_679_; 
v_res_679_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg(v_cls_671_, v_msg_672_, v___y_673_, v___y_674_, v___y_675_, v___y_676_, v___y_677_);
lean_dec(v___y_677_);
lean_dec_ref(v___y_676_);
lean_dec(v___y_675_);
lean_dec_ref(v___y_674_);
return v_res_679_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__2(void){
_start:
{
lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; 
v___x_683_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_));
v___x_684_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__1));
v___x_685_ = l_Lean_Name_append(v___x_684_, v___x_683_);
return v___x_685_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__4(void){
_start:
{
lean_object* v___x_687_; lean_object* v___x_688_; 
v___x_687_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__3));
v___x_688_ = l_Lean_stringToMessageData(v___x_687_);
return v___x_688_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__6(void){
_start:
{
lean_object* v___x_690_; lean_object* v___x_691_; 
v___x_690_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__5));
v___x_691_ = l_Lean_stringToMessageData(v___x_690_);
return v___x_691_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__8(void){
_start:
{
lean_object* v___x_693_; lean_object* v___x_694_; 
v___x_693_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__7));
v___x_694_ = l_Lean_stringToMessageData(v___x_693_);
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg(lean_object* v_go_695_, lean_object* v_a_696_, lean_object* v_a_697_, lean_object* v_a_698_, lean_object* v_a_699_, lean_object* v_a_700_, lean_object* v_a_701_){
_start:
{
lean_object* v___y_704_; lean_object* v_config_708_; lean_object* v_constToUnfold_709_; lean_object* v_disch_710_; lean_object* v_transitionDepth_711_; lean_object* v_maxTransitionDepth_712_; lean_object* v___x_713_; lean_object* v___x_714_; uint8_t v___x_715_; 
v_config_708_ = lean_ctor_get(v_a_696_, 0);
v_constToUnfold_709_ = lean_ctor_get(v_a_696_, 1);
v_disch_710_ = lean_ctor_get(v_a_696_, 2);
v_transitionDepth_711_ = lean_ctor_get(v_a_696_, 3);
v_maxTransitionDepth_712_ = lean_ctor_get(v_config_708_, 0);
v___x_713_ = lean_unsigned_to_nat(1u);
v___x_714_ = lean_nat_add(v_transitionDepth_711_, v___x_713_);
v___x_715_ = lean_nat_dec_lt(v_maxTransitionDepth_712_, v___x_714_);
if (v___x_715_ == 0)
{
lean_object* v___x_716_; lean_object* v___x_717_; 
lean_inc_ref(v_disch_710_);
lean_inc(v_constToUnfold_709_);
lean_inc_ref(v_config_708_);
v___x_716_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_716_, 0, v_config_708_);
lean_ctor_set(v___x_716_, 1, v_constToUnfold_709_);
lean_ctor_set(v___x_716_, 2, v_disch_710_);
lean_ctor_set(v___x_716_, 3, v___x_714_);
lean_inc(v_a_701_);
lean_inc_ref(v_a_700_);
lean_inc(v_a_699_);
lean_inc_ref(v_a_698_);
v___x_717_ = lean_apply_7(v_go_695_, v___x_716_, v_a_697_, v_a_698_, v_a_699_, v_a_700_, v_a_701_, lean_box(0));
return v___x_717_;
}
else
{
lean_object* v_options_718_; uint8_t v_hasTrace_719_; 
lean_dec_ref(v_go_695_);
v_options_718_ = lean_ctor_get(v_a_700_, 2);
v_hasTrace_719_ = lean_ctor_get_uint8(v_options_718_, sizeof(void*)*1);
if (v_hasTrace_719_ == 0)
{
lean_dec(v___x_714_);
v___y_704_ = v_a_697_;
goto v___jp_703_;
}
else
{
lean_object* v_inheritedTraceOptions_720_; lean_object* v___x_721_; lean_object* v___x_722_; uint8_t v___x_723_; 
v_inheritedTraceOptions_720_ = lean_ctor_get(v_a_700_, 13);
v___x_721_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn___closed__3_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_));
v___x_722_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__2, &lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__2_once, _init_lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__2);
v___x_723_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_720_, v_options_718_, v___x_722_);
if (v___x_723_ == 0)
{
lean_dec(v___x_714_);
v___y_704_ = v_a_697_;
goto v___jp_703_;
}
else
{
lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; 
v___x_724_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__4, &lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__4_once, _init_lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__4);
lean_inc(v_maxTransitionDepth_712_);
v___x_725_ = l_Nat_reprFast(v_maxTransitionDepth_712_);
v___x_726_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_726_, 0, v___x_725_);
v___x_727_ = l_Lean_MessageData_ofFormat(v___x_726_);
v___x_728_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_728_, 0, v___x_724_);
lean_ctor_set(v___x_728_, 1, v___x_727_);
v___x_729_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__6);
v___x_730_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_730_, 0, v___x_728_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
v___x_731_ = l_Nat_reprFast(v___x_714_);
v___x_732_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_732_, 0, v___x_731_);
v___x_733_ = l_Lean_MessageData_ofFormat(v___x_732_);
v___x_734_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_734_, 0, v___x_730_);
lean_ctor_set(v___x_734_, 1, v___x_733_);
v___x_735_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__8, &lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__8_once, _init_lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___closed__8);
v___x_736_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_736_, 0, v___x_734_);
lean_ctor_set(v___x_736_, 1, v___x_735_);
v___x_737_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg(v___x_721_, v___x_736_, v_a_697_, v_a_698_, v_a_699_, v_a_700_, v_a_701_);
if (lean_obj_tag(v___x_737_) == 0)
{
lean_object* v_a_738_; lean_object* v_snd_739_; 
v_a_738_ = lean_ctor_get(v___x_737_, 0);
lean_inc(v_a_738_);
lean_dec_ref_known(v___x_737_, 1);
v_snd_739_ = lean_ctor_get(v_a_738_, 1);
lean_inc(v_snd_739_);
lean_dec(v_a_738_);
v___y_704_ = v_snd_739_;
goto v___jp_703_;
}
else
{
lean_object* v_a_740_; lean_object* v___x_742_; uint8_t v_isShared_743_; uint8_t v_isSharedCheck_747_; 
v_a_740_ = lean_ctor_get(v___x_737_, 0);
v_isSharedCheck_747_ = !lean_is_exclusive(v___x_737_);
if (v_isSharedCheck_747_ == 0)
{
v___x_742_ = v___x_737_;
v_isShared_743_ = v_isSharedCheck_747_;
goto v_resetjp_741_;
}
else
{
lean_inc(v_a_740_);
lean_dec(v___x_737_);
v___x_742_ = lean_box(0);
v_isShared_743_ = v_isSharedCheck_747_;
goto v_resetjp_741_;
}
v_resetjp_741_:
{
lean_object* v___x_745_; 
if (v_isShared_743_ == 0)
{
v___x_745_ = v___x_742_;
goto v_reusejp_744_;
}
else
{
lean_object* v_reuseFailAlloc_746_; 
v_reuseFailAlloc_746_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_746_, 0, v_a_740_);
v___x_745_ = v_reuseFailAlloc_746_;
goto v_reusejp_744_;
}
v_reusejp_744_:
{
return v___x_745_;
}
}
}
}
}
}
v___jp_703_:
{
lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_705_ = lean_box(0);
v___x_706_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_706_, 0, v___x_705_);
lean_ctor_set(v___x_706_, 1, v___y_704_);
v___x_707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_707_, 0, v___x_706_);
return v___x_707_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg___boxed(lean_object* v_go_748_, lean_object* v_a_749_, lean_object* v_a_750_, lean_object* v_a_751_, lean_object* v_a_752_, lean_object* v_a_753_, lean_object* v_a_754_, lean_object* v_a_755_){
_start:
{
lean_object* v_res_756_; 
v_res_756_ = lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg(v_go_748_, v_a_749_, v_a_750_, v_a_751_, v_a_752_, v_a_753_, v_a_754_);
lean_dec(v_a_754_);
lean_dec_ref(v_a_753_);
lean_dec(v_a_752_);
lean_dec_ref(v_a_751_);
lean_dec_ref(v_a_749_);
return v_res_756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth(lean_object* v_00_u03b1_757_, lean_object* v_go_758_, lean_object* v_a_759_, lean_object* v_a_760_, lean_object* v_a_761_, lean_object* v_a_762_, lean_object* v_a_763_, lean_object* v_a_764_){
_start:
{
lean_object* v___x_766_; 
v___x_766_ = lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___redArg(v_go_758_, v_a_759_, v_a_760_, v_a_761_, v_a_762_, v_a_763_, v_a_764_);
return v___x_766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth___boxed(lean_object* v_00_u03b1_767_, lean_object* v_go_768_, lean_object* v_a_769_, lean_object* v_a_770_, lean_object* v_a_771_, lean_object* v_a_772_, lean_object* v_a_773_, lean_object* v_a_774_, lean_object* v_a_775_){
_start:
{
lean_object* v_res_776_; 
v_res_776_ = lp_mathlib_Mathlib_Meta_FunProp_withIncreasedTransitionDepth(v_00_u03b1_767_, v_go_768_, v_a_769_, v_a_770_, v_a_771_, v_a_772_, v_a_773_, v_a_774_);
lean_dec(v_a_774_);
lean_dec_ref(v_a_773_);
lean_dec(v_a_772_);
lean_dec_ref(v_a_771_);
lean_dec_ref(v_a_769_);
return v_res_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0(lean_object* v_cls_777_, lean_object* v_msg_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_){
_start:
{
lean_object* v___x_786_; 
v___x_786_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___redArg(v_cls_777_, v_msg_778_, v___y_780_, v___y_781_, v___y_782_, v___y_783_, v___y_784_);
return v___x_786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0___boxed(lean_object* v_cls_787_, lean_object* v_msg_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_){
_start:
{
lean_object* v_res_796_; 
v_res_796_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Meta_FunProp_withIncreasedTransitionDepth_spec__0(v_cls_787_, v_msg_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
lean_dec(v___y_794_);
lean_dec_ref(v___y_793_);
lean_dec(v___y_792_);
lean_dec_ref(v___y_791_);
lean_dec_ref(v___y_789_);
return v_res_796_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00Mathlib_Meta_FunProp_logError_spec__0(lean_object* v_a_797_, lean_object* v_x_798_){
_start:
{
if (lean_obj_tag(v_x_798_) == 0)
{
uint8_t v___x_799_; 
v___x_799_ = 0;
return v___x_799_;
}
else
{
lean_object* v_head_800_; lean_object* v_tail_801_; uint8_t v___x_802_; 
v_head_800_ = lean_ctor_get(v_x_798_, 0);
v_tail_801_ = lean_ctor_get(v_x_798_, 1);
v___x_802_ = lean_string_dec_eq(v_a_797_, v_head_800_);
if (v___x_802_ == 0)
{
v_x_798_ = v_tail_801_;
goto _start;
}
else
{
return v___x_802_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00Mathlib_Meta_FunProp_logError_spec__0___boxed(lean_object* v_a_804_, lean_object* v_x_805_){
_start:
{
uint8_t v_res_806_; lean_object* v_r_807_; 
v_res_806_ = lp_mathlib_List_elem___at___00Mathlib_Meta_FunProp_logError_spec__0(v_a_804_, v_x_805_);
lean_dec(v_x_805_);
lean_dec_ref(v_a_804_);
v_r_807_ = lean_box(v_res_806_);
return v_r_807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_logError___redArg(lean_object* v_msg_808_, lean_object* v_a_809_, lean_object* v_a_810_){
_start:
{
lean_object* v_transitionDepth_812_; lean_object* v___x_813_; uint8_t v___x_814_; 
v_transitionDepth_812_ = lean_ctor_get(v_a_809_, 3);
v___x_813_ = lean_unsigned_to_nat(0u);
v___x_814_ = lean_nat_dec_eq(v_transitionDepth_812_, v___x_813_);
if (v___x_814_ == 0)
{
lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; 
lean_dec_ref(v_msg_808_);
v___x_815_ = lean_box(0);
v___x_816_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_816_, 0, v___x_815_);
lean_ctor_set(v___x_816_, 1, v_a_810_);
v___x_817_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_817_, 0, v___x_816_);
return v___x_817_;
}
else
{
lean_object* v_cache_818_; lean_object* v_failureCache_819_; lean_object* v_numSteps_820_; lean_object* v_msgLog_821_; lean_object* v_morTheorems_822_; lean_object* v_transitionTheorems_823_; lean_object* v___x_825_; uint8_t v_isShared_826_; uint8_t v_isSharedCheck_837_; 
v_cache_818_ = lean_ctor_get(v_a_810_, 0);
v_failureCache_819_ = lean_ctor_get(v_a_810_, 1);
v_numSteps_820_ = lean_ctor_get(v_a_810_, 2);
v_msgLog_821_ = lean_ctor_get(v_a_810_, 3);
v_morTheorems_822_ = lean_ctor_get(v_a_810_, 4);
v_transitionTheorems_823_ = lean_ctor_get(v_a_810_, 5);
v_isSharedCheck_837_ = !lean_is_exclusive(v_a_810_);
if (v_isSharedCheck_837_ == 0)
{
v___x_825_ = v_a_810_;
v_isShared_826_ = v_isSharedCheck_837_;
goto v_resetjp_824_;
}
else
{
lean_inc(v_transitionTheorems_823_);
lean_inc(v_morTheorems_822_);
lean_inc(v_msgLog_821_);
lean_inc(v_numSteps_820_);
lean_inc(v_failureCache_819_);
lean_inc(v_cache_818_);
lean_dec(v_a_810_);
v___x_825_ = lean_box(0);
v_isShared_826_ = v_isSharedCheck_837_;
goto v_resetjp_824_;
}
v_resetjp_824_:
{
lean_object* v___x_827_; lean_object* v___y_829_; uint8_t v___x_835_; 
v___x_827_ = lean_box(0);
v___x_835_ = lp_mathlib_List_elem___at___00Mathlib_Meta_FunProp_logError_spec__0(v_msg_808_, v_msgLog_821_);
if (v___x_835_ == 0)
{
lean_object* v___x_836_; 
v___x_836_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_836_, 0, v_msg_808_);
lean_ctor_set(v___x_836_, 1, v_msgLog_821_);
v___y_829_ = v___x_836_;
goto v___jp_828_;
}
else
{
lean_dec_ref(v_msg_808_);
v___y_829_ = v_msgLog_821_;
goto v___jp_828_;
}
v___jp_828_:
{
lean_object* v___x_831_; 
if (v_isShared_826_ == 0)
{
lean_ctor_set(v___x_825_, 3, v___y_829_);
v___x_831_ = v___x_825_;
goto v_reusejp_830_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v_cache_818_);
lean_ctor_set(v_reuseFailAlloc_834_, 1, v_failureCache_819_);
lean_ctor_set(v_reuseFailAlloc_834_, 2, v_numSteps_820_);
lean_ctor_set(v_reuseFailAlloc_834_, 3, v___y_829_);
lean_ctor_set(v_reuseFailAlloc_834_, 4, v_morTheorems_822_);
lean_ctor_set(v_reuseFailAlloc_834_, 5, v_transitionTheorems_823_);
v___x_831_ = v_reuseFailAlloc_834_;
goto v_reusejp_830_;
}
v_reusejp_830_:
{
lean_object* v___x_832_; lean_object* v___x_833_; 
v___x_832_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_832_, 0, v___x_827_);
lean_ctor_set(v___x_832_, 1, v___x_831_);
v___x_833_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_833_, 0, v___x_832_);
return v___x_833_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_logError___redArg___boxed(lean_object* v_msg_838_, lean_object* v_a_839_, lean_object* v_a_840_, lean_object* v_a_841_){
_start:
{
lean_object* v_res_842_; 
v_res_842_ = lp_mathlib_Mathlib_Meta_FunProp_logError___redArg(v_msg_838_, v_a_839_, v_a_840_);
lean_dec_ref(v_a_839_);
return v_res_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_logError(lean_object* v_msg_843_, lean_object* v_a_844_, lean_object* v_a_845_, lean_object* v_a_846_, lean_object* v_a_847_, lean_object* v_a_848_, lean_object* v_a_849_){
_start:
{
lean_object* v___x_851_; 
v___x_851_ = lp_mathlib_Mathlib_Meta_FunProp_logError___redArg(v_msg_843_, v_a_844_, v_a_845_);
return v___x_851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_logError___boxed(lean_object* v_msg_852_, lean_object* v_a_853_, lean_object* v_a_854_, lean_object* v_a_855_, lean_object* v_a_856_, lean_object* v_a_857_, lean_object* v_a_858_, lean_object* v_a_859_){
_start:
{
lean_object* v_res_860_; 
v_res_860_ = lp_mathlib_Mathlib_Meta_FunProp_logError(v_msg_852_, v_a_853_, v_a_854_, v_a_855_, v_a_856_, v_a_857_, v_a_858_);
lean_dec(v_a_858_);
lean_dec_ref(v_a_857_);
lean_dec(v_a_856_);
lean_dec_ref(v_a_855_);
lean_dec_ref(v_a_853_);
return v_res_860_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Types(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FunProp_Types(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_33604414____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_1258255133____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_FunProp_Types_0__Mathlib_Meta_FunProp_initFn_00___x40_Mathlib_Tactic_FunProp_Types_4092581443____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default);
lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems = _init_lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Types(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_FunctionData(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FunProp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FunProp_Types(builtin);
}
#ifdef __cplusplus
}
#endif
