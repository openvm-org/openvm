// Lean compiler output
// Module: Mathlib.Tactic.SwapVar
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
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Meta_getLocalDeclFromUserName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_LocalContext_setUserName(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_swapRule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "swapRule"};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_swapRule___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_swapRule___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 50, 218, 132, 164, 129, 224, 124)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_swapRule___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_swapRule___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__6_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_swapRule___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__9_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_swapRule___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = " ↔"};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_swapRule___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__15_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_swapRule___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_swapRule___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__20_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_swapRule = (const lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSwap_var__,,"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 115, 87, 127, 138, 34, 160, 62)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "swap_var "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__4_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__10_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_swapRule___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__13_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__13_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_83_ = lean_box(0);
v___x_84_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_85_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set(v___x_85_, 1, v___x_83_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg(){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_87_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg___closed__0);
v___x_88_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg___boxed(lean_object* v___y_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg();
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0(lean_object* v_00_u03b1_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg();
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___boxed(lean_object* v_00_u03b1_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0(v_00_u03b1_102_, v___y_103_, v___y_104_, v___y_105_, v___y_106_, v___y_107_, v___y_108_, v___y_109_, v___y_110_);
lean_dec(v___y_110_);
lean_dec_ref(v___y_109_);
lean_dec(v___y_108_);
lean_dec_ref(v___y_107_);
lean_dec(v___y_106_);
lean_dec_ref(v___y_105_);
lean_dec(v___y_104_);
lean_dec_ref(v___y_103_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg___lam__0(lean_object* v_x_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
lean_object* v___x_123_; 
lean_inc(v___y_117_);
lean_inc_ref(v___y_116_);
lean_inc(v___y_115_);
lean_inc_ref(v___y_114_);
v___x_123_ = lean_apply_9(v_x_113_, v___y_114_, v___y_115_, v___y_116_, v___y_117_, v___y_118_, v___y_119_, v___y_120_, v___y_121_, lean_box(0));
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg___lam__0___boxed(lean_object* v_x_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg___lam__0(v_x_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_, v___y_131_, v___y_132_);
lean_dec(v___y_128_);
lean_dec_ref(v___y_127_);
lean_dec(v___y_126_);
lean_dec_ref(v___y_125_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg(lean_object* v_lctx_135_, lean_object* v_localInsts_136_, lean_object* v_x_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
lean_object* v___f_147_; lean_object* v___x_148_; 
lean_inc(v___y_141_);
lean_inc_ref(v___y_140_);
lean_inc(v___y_139_);
lean_inc_ref(v___y_138_);
v___f_147_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_147_, 0, v_x_137_);
lean_closure_set(v___f_147_, 1, v___y_138_);
lean_closure_set(v___f_147_, 2, v___y_139_);
lean_closure_set(v___f_147_, 3, v___y_140_);
lean_closure_set(v___f_147_, 4, v___y_141_);
v___x_148_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_135_, v_localInsts_136_, v___f_147_, v___y_142_, v___y_143_, v___y_144_, v___y_145_);
if (lean_obj_tag(v___x_148_) == 0)
{
return v___x_148_;
}
else
{
lean_object* v_a_149_; lean_object* v___x_151_; uint8_t v_isShared_152_; uint8_t v_isSharedCheck_156_; 
v_a_149_ = lean_ctor_get(v___x_148_, 0);
v_isSharedCheck_156_ = !lean_is_exclusive(v___x_148_);
if (v_isSharedCheck_156_ == 0)
{
v___x_151_ = v___x_148_;
v_isShared_152_ = v_isSharedCheck_156_;
goto v_resetjp_150_;
}
else
{
lean_inc(v_a_149_);
lean_dec(v___x_148_);
v___x_151_ = lean_box(0);
v_isShared_152_ = v_isSharedCheck_156_;
goto v_resetjp_150_;
}
v_resetjp_150_:
{
lean_object* v___x_154_; 
if (v_isShared_152_ == 0)
{
v___x_154_ = v___x_151_;
goto v_reusejp_153_;
}
else
{
lean_object* v_reuseFailAlloc_155_; 
v_reuseFailAlloc_155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_155_, 0, v_a_149_);
v___x_154_ = v_reuseFailAlloc_155_;
goto v_reusejp_153_;
}
v_reusejp_153_:
{
return v___x_154_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg___boxed(lean_object* v_lctx_157_, lean_object* v_localInsts_158_, lean_object* v_x_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg(v_lctx_157_, v_localInsts_158_, v_x_159_, v___y_160_, v___y_161_, v___y_162_, v___y_163_, v___y_164_, v___y_165_, v___y_166_, v___y_167_);
lean_dec(v___y_167_);
lean_dec_ref(v___y_166_);
lean_dec(v___y_165_);
lean_dec_ref(v___y_164_);
lean_dec(v___y_163_);
lean_dec_ref(v___y_162_);
lean_dec(v___y_161_);
lean_dec_ref(v___y_160_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1(lean_object* v_00_u03b1_170_, lean_object* v_lctx_171_, lean_object* v_localInsts_172_, lean_object* v_x_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg(v_lctx_171_, v_localInsts_172_, v_x_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_, v___y_181_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___boxed(lean_object* v_00_u03b1_184_, lean_object* v_lctx_185_, lean_object* v_localInsts_186_, lean_object* v_x_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1(v_00_u03b1_184_, v_lctx_185_, v_localInsts_186_, v_x_187_, v___y_188_, v___y_189_, v___y_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_, v___y_195_);
lean_dec(v___y_195_);
lean_dec_ref(v___y_194_);
lean_dec(v___y_193_);
lean_dec_ref(v___y_192_);
lean_dec(v___y_191_);
lean_dec_ref(v___y_190_);
lean_dec(v___y_189_);
lean_dec_ref(v___y_188_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3_spec__5___redArg(lean_object* v_x_198_, lean_object* v_x_199_, lean_object* v_x_200_, lean_object* v_x_201_){
_start:
{
lean_object* v_ks_202_; lean_object* v_vs_203_; lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_227_; 
v_ks_202_ = lean_ctor_get(v_x_198_, 0);
v_vs_203_ = lean_ctor_get(v_x_198_, 1);
v_isSharedCheck_227_ = !lean_is_exclusive(v_x_198_);
if (v_isSharedCheck_227_ == 0)
{
v___x_205_ = v_x_198_;
v_isShared_206_ = v_isSharedCheck_227_;
goto v_resetjp_204_;
}
else
{
lean_inc(v_vs_203_);
lean_inc(v_ks_202_);
lean_dec(v_x_198_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_227_;
goto v_resetjp_204_;
}
v_resetjp_204_:
{
lean_object* v___x_207_; uint8_t v___x_208_; 
v___x_207_ = lean_array_get_size(v_ks_202_);
v___x_208_ = lean_nat_dec_lt(v_x_199_, v___x_207_);
if (v___x_208_ == 0)
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_212_; 
lean_dec(v_x_199_);
v___x_209_ = lean_array_push(v_ks_202_, v_x_200_);
v___x_210_ = lean_array_push(v_vs_203_, v_x_201_);
if (v_isShared_206_ == 0)
{
lean_ctor_set(v___x_205_, 1, v___x_210_);
lean_ctor_set(v___x_205_, 0, v___x_209_);
v___x_212_ = v___x_205_;
goto v_reusejp_211_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v___x_209_);
lean_ctor_set(v_reuseFailAlloc_213_, 1, v___x_210_);
v___x_212_ = v_reuseFailAlloc_213_;
goto v_reusejp_211_;
}
v_reusejp_211_:
{
return v___x_212_;
}
}
else
{
lean_object* v_k_x27_214_; uint8_t v___x_215_; 
v_k_x27_214_ = lean_array_fget_borrowed(v_ks_202_, v_x_199_);
v___x_215_ = l_Lean_instBEqMVarId_beq(v_x_200_, v_k_x27_214_);
if (v___x_215_ == 0)
{
lean_object* v___x_217_; 
if (v_isShared_206_ == 0)
{
v___x_217_ = v___x_205_;
goto v_reusejp_216_;
}
else
{
lean_object* v_reuseFailAlloc_221_; 
v_reuseFailAlloc_221_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_221_, 0, v_ks_202_);
lean_ctor_set(v_reuseFailAlloc_221_, 1, v_vs_203_);
v___x_217_ = v_reuseFailAlloc_221_;
goto v_reusejp_216_;
}
v_reusejp_216_:
{
lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_218_ = lean_unsigned_to_nat(1u);
v___x_219_ = lean_nat_add(v_x_199_, v___x_218_);
lean_dec(v_x_199_);
v_x_198_ = v___x_217_;
v_x_199_ = v___x_219_;
goto _start;
}
}
else
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_225_; 
v___x_222_ = lean_array_fset(v_ks_202_, v_x_199_, v_x_200_);
v___x_223_ = lean_array_fset(v_vs_203_, v_x_199_, v_x_201_);
lean_dec(v_x_199_);
if (v_isShared_206_ == 0)
{
lean_ctor_set(v___x_205_, 1, v___x_223_);
lean_ctor_set(v___x_205_, 0, v___x_222_);
v___x_225_ = v___x_205_;
goto v_reusejp_224_;
}
else
{
lean_object* v_reuseFailAlloc_226_; 
v_reuseFailAlloc_226_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_226_, 0, v___x_222_);
lean_ctor_set(v_reuseFailAlloc_226_, 1, v___x_223_);
v___x_225_ = v_reuseFailAlloc_226_;
goto v_reusejp_224_;
}
v_reusejp_224_:
{
return v___x_225_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3___redArg(lean_object* v_n_228_, lean_object* v_k_229_, lean_object* v_v_230_){
_start:
{
lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_231_ = lean_unsigned_to_nat(0u);
v___x_232_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3_spec__5___redArg(v_n_228_, v___x_231_, v_k_229_, v_v_230_);
return v___x_232_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg(lean_object* v_x_234_, size_t v_x_235_, size_t v_x_236_, lean_object* v_x_237_, lean_object* v_x_238_){
_start:
{
if (lean_obj_tag(v_x_234_) == 0)
{
lean_object* v_es_239_; size_t v___x_240_; size_t v___x_241_; lean_object* v_j_242_; lean_object* v___x_243_; uint8_t v___x_244_; 
v_es_239_ = lean_ctor_get(v_x_234_, 0);
v___x_240_ = ((size_t)31ULL);
v___x_241_ = lean_usize_land(v_x_235_, v___x_240_);
v_j_242_ = lean_usize_to_nat(v___x_241_);
v___x_243_ = lean_array_get_size(v_es_239_);
v___x_244_ = lean_nat_dec_lt(v_j_242_, v___x_243_);
if (v___x_244_ == 0)
{
lean_dec(v_j_242_);
lean_dec(v_x_238_);
lean_dec(v_x_237_);
return v_x_234_;
}
else
{
lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_283_; 
lean_inc_ref(v_es_239_);
v_isSharedCheck_283_ = !lean_is_exclusive(v_x_234_);
if (v_isSharedCheck_283_ == 0)
{
lean_object* v_unused_284_; 
v_unused_284_ = lean_ctor_get(v_x_234_, 0);
lean_dec(v_unused_284_);
v___x_246_ = v_x_234_;
v_isShared_247_ = v_isSharedCheck_283_;
goto v_resetjp_245_;
}
else
{
lean_dec(v_x_234_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_283_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v_v_248_; lean_object* v___x_249_; lean_object* v_xs_x27_250_; lean_object* v___y_252_; 
v_v_248_ = lean_array_fget(v_es_239_, v_j_242_);
v___x_249_ = lean_box(0);
v_xs_x27_250_ = lean_array_fset(v_es_239_, v_j_242_, v___x_249_);
switch(lean_obj_tag(v_v_248_))
{
case 0:
{
lean_object* v_key_257_; lean_object* v_val_258_; lean_object* v___x_260_; uint8_t v_isShared_261_; uint8_t v_isSharedCheck_268_; 
v_key_257_ = lean_ctor_get(v_v_248_, 0);
v_val_258_ = lean_ctor_get(v_v_248_, 1);
v_isSharedCheck_268_ = !lean_is_exclusive(v_v_248_);
if (v_isSharedCheck_268_ == 0)
{
v___x_260_ = v_v_248_;
v_isShared_261_ = v_isSharedCheck_268_;
goto v_resetjp_259_;
}
else
{
lean_inc(v_val_258_);
lean_inc(v_key_257_);
lean_dec(v_v_248_);
v___x_260_ = lean_box(0);
v_isShared_261_ = v_isSharedCheck_268_;
goto v_resetjp_259_;
}
v_resetjp_259_:
{
uint8_t v___x_262_; 
v___x_262_ = l_Lean_instBEqMVarId_beq(v_x_237_, v_key_257_);
if (v___x_262_ == 0)
{
lean_object* v___x_263_; lean_object* v___x_264_; 
lean_del_object(v___x_260_);
v___x_263_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_257_, v_val_258_, v_x_237_, v_x_238_);
v___x_264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_264_, 0, v___x_263_);
v___y_252_ = v___x_264_;
goto v___jp_251_;
}
else
{
lean_object* v___x_266_; 
lean_dec(v_val_258_);
lean_dec(v_key_257_);
if (v_isShared_261_ == 0)
{
lean_ctor_set(v___x_260_, 1, v_x_238_);
lean_ctor_set(v___x_260_, 0, v_x_237_);
v___x_266_ = v___x_260_;
goto v_reusejp_265_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v_x_237_);
lean_ctor_set(v_reuseFailAlloc_267_, 1, v_x_238_);
v___x_266_ = v_reuseFailAlloc_267_;
goto v_reusejp_265_;
}
v_reusejp_265_:
{
v___y_252_ = v___x_266_;
goto v___jp_251_;
}
}
}
}
case 1:
{
lean_object* v_node_269_; lean_object* v___x_271_; uint8_t v_isShared_272_; uint8_t v_isSharedCheck_281_; 
v_node_269_ = lean_ctor_get(v_v_248_, 0);
v_isSharedCheck_281_ = !lean_is_exclusive(v_v_248_);
if (v_isSharedCheck_281_ == 0)
{
v___x_271_ = v_v_248_;
v_isShared_272_ = v_isSharedCheck_281_;
goto v_resetjp_270_;
}
else
{
lean_inc(v_node_269_);
lean_dec(v_v_248_);
v___x_271_ = lean_box(0);
v_isShared_272_ = v_isSharedCheck_281_;
goto v_resetjp_270_;
}
v_resetjp_270_:
{
size_t v___x_273_; size_t v___x_274_; size_t v___x_275_; size_t v___x_276_; lean_object* v___x_277_; lean_object* v___x_279_; 
v___x_273_ = ((size_t)5ULL);
v___x_274_ = lean_usize_shift_right(v_x_235_, v___x_273_);
v___x_275_ = ((size_t)1ULL);
v___x_276_ = lean_usize_add(v_x_236_, v___x_275_);
v___x_277_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg(v_node_269_, v___x_274_, v___x_276_, v_x_237_, v_x_238_);
if (v_isShared_272_ == 0)
{
lean_ctor_set(v___x_271_, 0, v___x_277_);
v___x_279_ = v___x_271_;
goto v_reusejp_278_;
}
else
{
lean_object* v_reuseFailAlloc_280_; 
v_reuseFailAlloc_280_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_280_, 0, v___x_277_);
v___x_279_ = v_reuseFailAlloc_280_;
goto v_reusejp_278_;
}
v_reusejp_278_:
{
v___y_252_ = v___x_279_;
goto v___jp_251_;
}
}
}
default: 
{
lean_object* v___x_282_; 
v___x_282_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_282_, 0, v_x_237_);
lean_ctor_set(v___x_282_, 1, v_x_238_);
v___y_252_ = v___x_282_;
goto v___jp_251_;
}
}
v___jp_251_:
{
lean_object* v___x_253_; lean_object* v___x_255_; 
v___x_253_ = lean_array_fset(v_xs_x27_250_, v_j_242_, v___y_252_);
lean_dec(v_j_242_);
if (v_isShared_247_ == 0)
{
lean_ctor_set(v___x_246_, 0, v___x_253_);
v___x_255_ = v___x_246_;
goto v_reusejp_254_;
}
else
{
lean_object* v_reuseFailAlloc_256_; 
v_reuseFailAlloc_256_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_256_, 0, v___x_253_);
v___x_255_ = v_reuseFailAlloc_256_;
goto v_reusejp_254_;
}
v_reusejp_254_:
{
return v___x_255_;
}
}
}
}
}
else
{
lean_object* v_ks_285_; lean_object* v_vs_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_306_; 
v_ks_285_ = lean_ctor_get(v_x_234_, 0);
v_vs_286_ = lean_ctor_get(v_x_234_, 1);
v_isSharedCheck_306_ = !lean_is_exclusive(v_x_234_);
if (v_isSharedCheck_306_ == 0)
{
v___x_288_ = v_x_234_;
v_isShared_289_ = v_isSharedCheck_306_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_vs_286_);
lean_inc(v_ks_285_);
lean_dec(v_x_234_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_306_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_291_; 
if (v_isShared_289_ == 0)
{
v___x_291_ = v___x_288_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_305_; 
v_reuseFailAlloc_305_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_305_, 0, v_ks_285_);
lean_ctor_set(v_reuseFailAlloc_305_, 1, v_vs_286_);
v___x_291_ = v_reuseFailAlloc_305_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
lean_object* v_newNode_292_; uint8_t v___y_294_; size_t v___x_300_; uint8_t v___x_301_; 
v_newNode_292_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3___redArg(v___x_291_, v_x_237_, v_x_238_);
v___x_300_ = ((size_t)7ULL);
v___x_301_ = lean_usize_dec_le(v___x_300_, v_x_236_);
if (v___x_301_ == 0)
{
lean_object* v___x_302_; lean_object* v___x_303_; uint8_t v___x_304_; 
v___x_302_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_292_);
v___x_303_ = lean_unsigned_to_nat(4u);
v___x_304_ = lean_nat_dec_lt(v___x_302_, v___x_303_);
lean_dec(v___x_302_);
v___y_294_ = v___x_304_;
goto v___jp_293_;
}
else
{
v___y_294_ = v___x_301_;
goto v___jp_293_;
}
v___jp_293_:
{
if (v___y_294_ == 0)
{
lean_object* v_ks_295_; lean_object* v_vs_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; 
v_ks_295_ = lean_ctor_get(v_newNode_292_, 0);
lean_inc_ref(v_ks_295_);
v_vs_296_ = lean_ctor_get(v_newNode_292_, 1);
lean_inc_ref(v_vs_296_);
lean_dec_ref(v_newNode_292_);
v___x_297_ = lean_unsigned_to_nat(0u);
v___x_298_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg___closed__0);
v___x_299_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4___redArg(v_x_236_, v_ks_295_, v_vs_296_, v___x_297_, v___x_298_);
lean_dec_ref(v_vs_296_);
lean_dec_ref(v_ks_295_);
return v___x_299_;
}
else
{
return v_newNode_292_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4___redArg(size_t v_depth_307_, lean_object* v_keys_308_, lean_object* v_vals_309_, lean_object* v_i_310_, lean_object* v_entries_311_){
_start:
{
lean_object* v___x_312_; uint8_t v___x_313_; 
v___x_312_ = lean_array_get_size(v_keys_308_);
v___x_313_ = lean_nat_dec_lt(v_i_310_, v___x_312_);
if (v___x_313_ == 0)
{
lean_dec(v_i_310_);
return v_entries_311_;
}
else
{
lean_object* v_k_314_; lean_object* v_v_315_; uint64_t v___x_316_; size_t v_h_317_; size_t v___x_318_; lean_object* v___x_319_; size_t v___x_320_; size_t v___x_321_; size_t v___x_322_; size_t v_h_323_; lean_object* v___x_324_; lean_object* v___x_325_; 
v_k_314_ = lean_array_fget_borrowed(v_keys_308_, v_i_310_);
v_v_315_ = lean_array_fget_borrowed(v_vals_309_, v_i_310_);
v___x_316_ = l_Lean_instHashableMVarId_hash(v_k_314_);
v_h_317_ = lean_uint64_to_usize(v___x_316_);
v___x_318_ = ((size_t)5ULL);
v___x_319_ = lean_unsigned_to_nat(1u);
v___x_320_ = ((size_t)1ULL);
v___x_321_ = lean_usize_sub(v_depth_307_, v___x_320_);
v___x_322_ = lean_usize_mul(v___x_318_, v___x_321_);
v_h_323_ = lean_usize_shift_right(v_h_317_, v___x_322_);
v___x_324_ = lean_nat_add(v_i_310_, v___x_319_);
lean_dec(v_i_310_);
lean_inc(v_v_315_);
lean_inc(v_k_314_);
v___x_325_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg(v_entries_311_, v_h_323_, v_depth_307_, v_k_314_, v_v_315_);
v_i_310_ = v___x_324_;
v_entries_311_ = v___x_325_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4___redArg___boxed(lean_object* v_depth_327_, lean_object* v_keys_328_, lean_object* v_vals_329_, lean_object* v_i_330_, lean_object* v_entries_331_){
_start:
{
size_t v_depth_boxed_332_; lean_object* v_res_333_; 
v_depth_boxed_332_ = lean_unbox_usize(v_depth_327_);
lean_dec(v_depth_327_);
v_res_333_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4___redArg(v_depth_boxed_332_, v_keys_328_, v_vals_329_, v_i_330_, v_entries_331_);
lean_dec_ref(v_vals_329_);
lean_dec_ref(v_keys_328_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg___boxed(lean_object* v_x_334_, lean_object* v_x_335_, lean_object* v_x_336_, lean_object* v_x_337_, lean_object* v_x_338_){
_start:
{
size_t v_x_6482__boxed_339_; size_t v_x_6483__boxed_340_; lean_object* v_res_341_; 
v_x_6482__boxed_339_ = lean_unbox_usize(v_x_335_);
lean_dec(v_x_335_);
v_x_6483__boxed_340_ = lean_unbox_usize(v_x_336_);
lean_dec(v_x_336_);
v_res_341_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg(v_x_334_, v_x_6482__boxed_339_, v_x_6483__boxed_340_, v_x_337_, v_x_338_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2___redArg(lean_object* v_x_342_, lean_object* v_x_343_, lean_object* v_x_344_){
_start:
{
uint64_t v___x_345_; size_t v___x_346_; size_t v___x_347_; lean_object* v___x_348_; 
v___x_345_ = l_Lean_instHashableMVarId_hash(v_x_343_);
v___x_346_ = lean_uint64_to_usize(v___x_345_);
v___x_347_ = ((size_t)1ULL);
v___x_348_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg(v_x_342_, v___x_346_, v___x_347_, v_x_343_, v_x_344_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3___lam__0(uint8_t v___x_349_, lean_object* v___x_350_, lean_object* v___x_351_, lean_object* v_b_352_, lean_object* v___x_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_){
_start:
{
if (v___x_349_ == 0)
{
lean_object* v___x_363_; 
lean_dec_ref(v_b_352_);
v___x_363_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg();
return v___x_363_;
}
else
{
lean_object* v___x_364_; lean_object* v___y_366_; lean_object* v___y_367_; lean_object* v___y_368_; lean_object* v___y_369_; lean_object* v___y_370_; lean_object* v___y_371_; lean_object* v___y_372_; lean_object* v___y_373_; lean_object* v___x_412_; uint8_t v___x_413_; 
v___x_364_ = l_Lean_Syntax_getArg(v___x_350_, v___x_351_);
v___x_412_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_swapRule___closed__7));
lean_inc(v___x_364_);
v___x_413_ = l_Lean_Syntax_isOfKind(v___x_364_, v___x_412_);
if (v___x_413_ == 0)
{
lean_object* v___x_414_; 
lean_dec(v___x_364_);
lean_dec_ref(v_b_352_);
v___x_414_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg();
return v___x_414_;
}
else
{
lean_object* v___x_415_; uint8_t v___x_416_; 
v___x_415_ = l_Lean_Syntax_getArg(v___x_350_, v___x_353_);
v___x_416_ = l_Lean_Syntax_isNone(v___x_415_);
if (v___x_416_ == 0)
{
uint8_t v___x_417_; 
v___x_417_ = l_Lean_Syntax_matchesNull(v___x_415_, v___x_353_);
if (v___x_417_ == 0)
{
lean_object* v___x_418_; 
lean_dec(v___x_364_);
lean_dec_ref(v_b_352_);
v___x_418_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg();
return v___x_418_;
}
else
{
v___y_366_ = v___y_354_;
v___y_367_ = v___y_355_;
v___y_368_ = v___y_356_;
v___y_369_ = v___y_357_;
v___y_370_ = v___y_358_;
v___y_371_ = v___y_359_;
v___y_372_ = v___y_360_;
v___y_373_ = v___y_361_;
goto v___jp_365_;
}
}
else
{
lean_dec(v___x_415_);
v___y_366_ = v___y_354_;
v___y_367_ = v___y_355_;
v___y_368_ = v___y_356_;
v___y_369_ = v___y_357_;
v___y_370_ = v___y_358_;
v___y_371_ = v___y_359_;
v___y_372_ = v___y_360_;
v___y_373_ = v___y_361_;
goto v___jp_365_;
}
}
v___jp_365_:
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; uint8_t v___x_377_; 
v___x_374_ = lean_unsigned_to_nat(2u);
v___x_375_ = l_Lean_Syntax_getArg(v___x_350_, v___x_374_);
v___x_376_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_swapRule___closed__7));
lean_inc(v___x_375_);
v___x_377_ = l_Lean_Syntax_isOfKind(v___x_375_, v___x_376_);
if (v___x_377_ == 0)
{
lean_object* v___x_378_; 
lean_dec(v___x_375_);
lean_dec(v___x_364_);
lean_dec_ref(v_b_352_);
v___x_378_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg();
return v___x_378_;
}
else
{
lean_object* v___x_379_; lean_object* v___x_380_; 
v___x_379_ = l_Lean_TSyntax_getId(v___x_364_);
lean_dec(v___x_364_);
lean_inc(v___x_379_);
v___x_380_ = l_Lean_Meta_getLocalDeclFromUserName(v___x_379_, v___y_370_, v___y_371_, v___y_372_, v___y_373_);
if (lean_obj_tag(v___x_380_) == 0)
{
lean_object* v_a_381_; lean_object* v___x_382_; lean_object* v___x_383_; 
v_a_381_ = lean_ctor_get(v___x_380_, 0);
lean_inc(v_a_381_);
lean_dec_ref_known(v___x_380_, 1);
v___x_382_ = l_Lean_TSyntax_getId(v___x_375_);
lean_dec(v___x_375_);
lean_inc(v___x_382_);
v___x_383_ = l_Lean_Meta_getLocalDeclFromUserName(v___x_382_, v___y_370_, v___y_371_, v___y_372_, v___y_373_);
if (lean_obj_tag(v___x_383_) == 0)
{
lean_object* v_a_384_; lean_object* v___x_386_; uint8_t v_isShared_387_; uint8_t v_isSharedCheck_395_; 
v_a_384_ = lean_ctor_get(v___x_383_, 0);
v_isSharedCheck_395_ = !lean_is_exclusive(v___x_383_);
if (v_isSharedCheck_395_ == 0)
{
v___x_386_ = v___x_383_;
v_isShared_387_ = v_isSharedCheck_395_;
goto v_resetjp_385_;
}
else
{
lean_inc(v_a_384_);
lean_dec(v___x_383_);
v___x_386_ = lean_box(0);
v_isShared_387_ = v_isSharedCheck_395_;
goto v_resetjp_385_;
}
v_resetjp_385_:
{
lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_393_; 
v___x_388_ = l_Lean_LocalDecl_fvarId(v_a_381_);
lean_dec(v_a_381_);
v___x_389_ = l_Lean_LocalDecl_fvarId(v_a_384_);
lean_dec(v_a_384_);
v___x_390_ = l_Lean_LocalContext_setUserName(v_b_352_, v___x_388_, v___x_382_);
v___x_391_ = l_Lean_LocalContext_setUserName(v___x_390_, v___x_389_, v___x_379_);
if (v_isShared_387_ == 0)
{
lean_ctor_set(v___x_386_, 0, v___x_391_);
v___x_393_ = v___x_386_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v___x_391_);
v___x_393_ = v_reuseFailAlloc_394_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
return v___x_393_;
}
}
}
else
{
lean_object* v_a_396_; lean_object* v___x_398_; uint8_t v_isShared_399_; uint8_t v_isSharedCheck_403_; 
lean_dec(v___x_382_);
lean_dec(v_a_381_);
lean_dec(v___x_379_);
lean_dec_ref(v_b_352_);
v_a_396_ = lean_ctor_get(v___x_383_, 0);
v_isSharedCheck_403_ = !lean_is_exclusive(v___x_383_);
if (v_isSharedCheck_403_ == 0)
{
v___x_398_ = v___x_383_;
v_isShared_399_ = v_isSharedCheck_403_;
goto v_resetjp_397_;
}
else
{
lean_inc(v_a_396_);
lean_dec(v___x_383_);
v___x_398_ = lean_box(0);
v_isShared_399_ = v_isSharedCheck_403_;
goto v_resetjp_397_;
}
v_resetjp_397_:
{
lean_object* v___x_401_; 
if (v_isShared_399_ == 0)
{
v___x_401_ = v___x_398_;
goto v_reusejp_400_;
}
else
{
lean_object* v_reuseFailAlloc_402_; 
v_reuseFailAlloc_402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_402_, 0, v_a_396_);
v___x_401_ = v_reuseFailAlloc_402_;
goto v_reusejp_400_;
}
v_reusejp_400_:
{
return v___x_401_;
}
}
}
}
else
{
lean_object* v_a_404_; lean_object* v___x_406_; uint8_t v_isShared_407_; uint8_t v_isSharedCheck_411_; 
lean_dec(v___x_379_);
lean_dec(v___x_375_);
lean_dec_ref(v_b_352_);
v_a_404_ = lean_ctor_get(v___x_380_, 0);
v_isSharedCheck_411_ = !lean_is_exclusive(v___x_380_);
if (v_isSharedCheck_411_ == 0)
{
v___x_406_ = v___x_380_;
v_isShared_407_ = v_isSharedCheck_411_;
goto v_resetjp_405_;
}
else
{
lean_inc(v_a_404_);
lean_dec(v___x_380_);
v___x_406_ = lean_box(0);
v_isShared_407_ = v_isSharedCheck_411_;
goto v_resetjp_405_;
}
v_resetjp_405_:
{
lean_object* v___x_409_; 
if (v_isShared_407_ == 0)
{
v___x_409_ = v___x_406_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_410_; 
v_reuseFailAlloc_410_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_410_, 0, v_a_404_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3___lam__0___boxed(lean_object* v___x_419_, lean_object* v___x_420_, lean_object* v___x_421_, lean_object* v_b_422_, lean_object* v___x_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_){
_start:
{
uint8_t v___x_6657__boxed_433_; lean_object* v_res_434_; 
v___x_6657__boxed_433_ = lean_unbox(v___x_419_);
v_res_434_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3___lam__0(v___x_6657__boxed_433_, v___x_420_, v___x_421_, v_b_422_, v___x_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_, v___y_429_, v___y_430_, v___y_431_);
lean_dec(v___y_431_);
lean_dec_ref(v___y_430_);
lean_dec(v___y_429_);
lean_dec_ref(v___y_428_);
lean_dec(v___y_427_);
lean_dec_ref(v___y_426_);
lean_dec(v___y_425_);
lean_dec_ref(v___y_424_);
lean_dec(v___x_423_);
lean_dec(v___x_421_);
lean_dec(v___x_420_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3(lean_object* v___x_435_, lean_object* v_as_436_, size_t v_i_437_, size_t v_stop_438_, lean_object* v_b_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_){
_start:
{
uint8_t v___x_449_; 
v___x_449_ = lean_usize_dec_eq(v_i_437_, v_stop_438_);
if (v___x_449_ == 0)
{
lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; uint8_t v___x_454_; lean_object* v___x_455_; lean_object* v___y_456_; lean_object* v___x_457_; 
v___x_450_ = lean_unsigned_to_nat(0u);
v___x_451_ = lean_unsigned_to_nat(1u);
v___x_452_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_swapRule___closed__3));
v___x_453_ = lean_array_uget_borrowed(v_as_436_, v_i_437_);
lean_inc_n(v___x_453_, 2);
v___x_454_ = l_Lean_Syntax_isOfKind(v___x_453_, v___x_452_);
v___x_455_ = lean_box(v___x_454_);
lean_inc_ref(v_b_439_);
v___y_456_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3___lam__0___boxed), 14, 5);
lean_closure_set(v___y_456_, 0, v___x_455_);
lean_closure_set(v___y_456_, 1, v___x_453_);
lean_closure_set(v___y_456_, 2, v___x_450_);
lean_closure_set(v___y_456_, 3, v_b_439_);
lean_closure_set(v___y_456_, 4, v___x_451_);
lean_inc_ref(v___x_435_);
v___x_457_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__1___redArg(v_b_439_, v___x_435_, v___y_456_, v___y_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_, v___y_445_, v___y_446_, v___y_447_);
if (lean_obj_tag(v___x_457_) == 0)
{
lean_object* v_a_458_; size_t v___x_459_; size_t v___x_460_; 
v_a_458_ = lean_ctor_get(v___x_457_, 0);
lean_inc(v_a_458_);
lean_dec_ref_known(v___x_457_, 1);
v___x_459_ = ((size_t)1ULL);
v___x_460_ = lean_usize_add(v_i_437_, v___x_459_);
v_i_437_ = v___x_460_;
v_b_439_ = v_a_458_;
goto _start;
}
else
{
lean_dec_ref(v___x_435_);
return v___x_457_;
}
}
else
{
lean_object* v___x_462_; 
lean_dec_ref(v___x_435_);
v___x_462_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_462_, 0, v_b_439_);
return v___x_462_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3___boxed(lean_object* v___x_463_, lean_object* v_as_464_, lean_object* v_i_465_, lean_object* v_stop_466_, lean_object* v_b_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_){
_start:
{
size_t v_i_boxed_477_; size_t v_stop_boxed_478_; lean_object* v_res_479_; 
v_i_boxed_477_ = lean_unbox_usize(v_i_465_);
lean_dec(v_i_465_);
v_stop_boxed_478_ = lean_unbox_usize(v_stop_466_);
lean_dec(v_stop_466_);
v_res_479_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3(v___x_463_, v_as_464_, v_i_boxed_477_, v_stop_boxed_478_, v_b_467_, v___y_468_, v___y_469_, v___y_470_, v___y_471_, v___y_472_, v___y_473_, v___y_474_, v___y_475_);
lean_dec(v___y_475_);
lean_dec_ref(v___y_474_);
lean_dec(v___y_473_);
lean_dec_ref(v___y_472_);
lean_dec(v___y_471_);
lean_dec_ref(v___y_470_);
lean_dec(v___y_469_);
lean_dec_ref(v___y_468_);
lean_dec_ref(v_as_464_);
return v_res_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1(lean_object* v_x_480_, lean_object* v_a_481_, lean_object* v_a_482_, lean_object* v_a_483_, lean_object* v_a_484_, lean_object* v_a_485_, lean_object* v_a_486_, lean_object* v_a_487_, lean_object* v_a_488_){
_start:
{
lean_object* v___x_490_; uint8_t v___x_491_; 
v___x_490_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSwap__var_____x2c_x2c___closed__1));
lean_inc(v_x_480_);
v___x_491_ = l_Lean_Syntax_isOfKind(v_x_480_, v___x_490_);
if (v___x_491_ == 0)
{
lean_object* v___x_492_; 
lean_dec(v_x_480_);
v___x_492_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__0___redArg();
return v___x_492_;
}
else
{
lean_object* v___x_493_; 
v___x_493_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v_a_482_, v_a_485_, v_a_486_, v_a_487_, v_a_488_);
if (lean_obj_tag(v___x_493_) == 0)
{
lean_object* v_a_494_; lean_object* v___x_495_; 
v_a_494_ = lean_ctor_get(v___x_493_, 0);
lean_inc_n(v_a_494_, 2);
lean_dec_ref_known(v___x_493_, 1);
v___x_495_ = l_Lean_MVarId_getDecl(v_a_494_, v_a_485_, v_a_486_, v_a_487_, v_a_488_);
if (lean_obj_tag(v___x_495_) == 0)
{
lean_object* v_a_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_578_; 
v_a_496_ = lean_ctor_get(v___x_495_, 0);
v_isSharedCheck_578_ = !lean_is_exclusive(v___x_495_);
if (v_isSharedCheck_578_ == 0)
{
v___x_498_ = v___x_495_;
v_isShared_499_ = v_isSharedCheck_578_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_a_496_);
lean_dec(v___x_495_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_578_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v_userName_500_; lean_object* v_lctx_501_; lean_object* v_type_502_; lean_object* v_depth_503_; lean_object* v_localInstances_504_; uint8_t v_kind_505_; lean_object* v_numScopeArgs_506_; lean_object* v_index_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_577_; 
v_userName_500_ = lean_ctor_get(v_a_496_, 0);
v_lctx_501_ = lean_ctor_get(v_a_496_, 1);
v_type_502_ = lean_ctor_get(v_a_496_, 2);
v_depth_503_ = lean_ctor_get(v_a_496_, 3);
v_localInstances_504_ = lean_ctor_get(v_a_496_, 4);
v_kind_505_ = lean_ctor_get_uint8(v_a_496_, sizeof(void*)*7);
v_numScopeArgs_506_ = lean_ctor_get(v_a_496_, 5);
v_index_507_ = lean_ctor_get(v_a_496_, 6);
v_isSharedCheck_577_ = !lean_is_exclusive(v_a_496_);
if (v_isSharedCheck_577_ == 0)
{
v___x_509_ = v_a_496_;
v_isShared_510_ = v_isSharedCheck_577_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_index_507_);
lean_inc(v_numScopeArgs_506_);
lean_inc(v_localInstances_504_);
lean_inc(v_depth_503_);
lean_inc(v_type_502_);
lean_inc(v_lctx_501_);
lean_inc(v_userName_500_);
lean_dec(v_a_496_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_577_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v_a_512_; lean_object* v___y_553_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v_swapRules_566_; lean_object* v___x_567_; lean_object* v___x_568_; uint8_t v___x_569_; 
v___x_563_ = lean_unsigned_to_nat(0u);
v___x_564_ = lean_unsigned_to_nat(1u);
v___x_565_ = l_Lean_Syntax_getArg(v_x_480_, v___x_564_);
lean_dec(v_x_480_);
v_swapRules_566_ = l_Lean_Syntax_getArgs(v___x_565_);
lean_dec(v___x_565_);
v___x_567_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_swapRules_566_);
lean_dec_ref(v_swapRules_566_);
v___x_568_ = lean_array_get_size(v___x_567_);
v___x_569_ = lean_nat_dec_lt(v___x_563_, v___x_568_);
if (v___x_569_ == 0)
{
lean_dec_ref(v___x_567_);
v_a_512_ = v_lctx_501_;
goto v___jp_511_;
}
else
{
uint8_t v___x_570_; 
v___x_570_ = lean_nat_dec_le(v___x_568_, v___x_568_);
if (v___x_570_ == 0)
{
if (v___x_569_ == 0)
{
lean_dec_ref(v___x_567_);
v_a_512_ = v_lctx_501_;
goto v___jp_511_;
}
else
{
size_t v___x_571_; size_t v___x_572_; lean_object* v___x_573_; 
v___x_571_ = ((size_t)0ULL);
v___x_572_ = lean_usize_of_nat(v___x_568_);
lean_inc_ref(v_localInstances_504_);
v___x_573_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3(v_localInstances_504_, v___x_567_, v___x_571_, v___x_572_, v_lctx_501_, v_a_481_, v_a_482_, v_a_483_, v_a_484_, v_a_485_, v_a_486_, v_a_487_, v_a_488_);
lean_dec_ref(v___x_567_);
v___y_553_ = v___x_573_;
goto v___jp_552_;
}
}
else
{
size_t v___x_574_; size_t v___x_575_; lean_object* v___x_576_; 
v___x_574_ = ((size_t)0ULL);
v___x_575_ = lean_usize_of_nat(v___x_568_);
lean_inc_ref(v_localInstances_504_);
v___x_576_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__3(v_localInstances_504_, v___x_567_, v___x_574_, v___x_575_, v_lctx_501_, v_a_481_, v_a_482_, v_a_483_, v_a_484_, v_a_485_, v_a_486_, v_a_487_, v_a_488_);
lean_dec_ref(v___x_567_);
v___y_553_ = v___x_576_;
goto v___jp_552_;
}
}
v___jp_511_:
{
lean_object* v___x_513_; lean_object* v_mctx_514_; lean_object* v_cache_515_; lean_object* v_zetaDeltaFVarIds_516_; lean_object* v_postponed_517_; lean_object* v_diag_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_551_; 
v___x_513_ = lean_st_ref_take(v_a_486_);
v_mctx_514_ = lean_ctor_get(v___x_513_, 0);
v_cache_515_ = lean_ctor_get(v___x_513_, 1);
v_zetaDeltaFVarIds_516_ = lean_ctor_get(v___x_513_, 2);
v_postponed_517_ = lean_ctor_get(v___x_513_, 3);
v_diag_518_ = lean_ctor_get(v___x_513_, 4);
v_isSharedCheck_551_ = !lean_is_exclusive(v___x_513_);
if (v_isSharedCheck_551_ == 0)
{
v___x_520_ = v___x_513_;
v_isShared_521_ = v_isSharedCheck_551_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_diag_518_);
lean_inc(v_postponed_517_);
lean_inc(v_zetaDeltaFVarIds_516_);
lean_inc(v_cache_515_);
lean_inc(v_mctx_514_);
lean_dec(v___x_513_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_551_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v_depth_522_; lean_object* v_levelAssignDepth_523_; lean_object* v_lmvarCounter_524_; lean_object* v_mvarCounter_525_; lean_object* v_lDecls_526_; lean_object* v_decls_527_; lean_object* v_userNames_528_; lean_object* v_lAssignment_529_; lean_object* v_eAssignment_530_; lean_object* v_dAssignment_531_; lean_object* v___x_533_; uint8_t v_isShared_534_; uint8_t v_isSharedCheck_550_; 
v_depth_522_ = lean_ctor_get(v_mctx_514_, 0);
v_levelAssignDepth_523_ = lean_ctor_get(v_mctx_514_, 1);
v_lmvarCounter_524_ = lean_ctor_get(v_mctx_514_, 2);
v_mvarCounter_525_ = lean_ctor_get(v_mctx_514_, 3);
v_lDecls_526_ = lean_ctor_get(v_mctx_514_, 4);
v_decls_527_ = lean_ctor_get(v_mctx_514_, 5);
v_userNames_528_ = lean_ctor_get(v_mctx_514_, 6);
v_lAssignment_529_ = lean_ctor_get(v_mctx_514_, 7);
v_eAssignment_530_ = lean_ctor_get(v_mctx_514_, 8);
v_dAssignment_531_ = lean_ctor_get(v_mctx_514_, 9);
v_isSharedCheck_550_ = !lean_is_exclusive(v_mctx_514_);
if (v_isSharedCheck_550_ == 0)
{
v___x_533_ = v_mctx_514_;
v_isShared_534_ = v_isSharedCheck_550_;
goto v_resetjp_532_;
}
else
{
lean_inc(v_dAssignment_531_);
lean_inc(v_eAssignment_530_);
lean_inc(v_lAssignment_529_);
lean_inc(v_userNames_528_);
lean_inc(v_decls_527_);
lean_inc(v_lDecls_526_);
lean_inc(v_mvarCounter_525_);
lean_inc(v_lmvarCounter_524_);
lean_inc(v_levelAssignDepth_523_);
lean_inc(v_depth_522_);
lean_dec(v_mctx_514_);
v___x_533_ = lean_box(0);
v_isShared_534_ = v_isSharedCheck_550_;
goto v_resetjp_532_;
}
v_resetjp_532_:
{
lean_object* v___x_536_; 
if (v_isShared_510_ == 0)
{
lean_ctor_set(v___x_509_, 1, v_a_512_);
v___x_536_ = v___x_509_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_549_; 
v_reuseFailAlloc_549_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_reuseFailAlloc_549_, 0, v_userName_500_);
lean_ctor_set(v_reuseFailAlloc_549_, 1, v_a_512_);
lean_ctor_set(v_reuseFailAlloc_549_, 2, v_type_502_);
lean_ctor_set(v_reuseFailAlloc_549_, 3, v_depth_503_);
lean_ctor_set(v_reuseFailAlloc_549_, 4, v_localInstances_504_);
lean_ctor_set(v_reuseFailAlloc_549_, 5, v_numScopeArgs_506_);
lean_ctor_set(v_reuseFailAlloc_549_, 6, v_index_507_);
lean_ctor_set_uint8(v_reuseFailAlloc_549_, sizeof(void*)*7, v_kind_505_);
v___x_536_ = v_reuseFailAlloc_549_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
lean_object* v___x_537_; lean_object* v___x_539_; 
v___x_537_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2___redArg(v_decls_527_, v_a_494_, v___x_536_);
if (v_isShared_534_ == 0)
{
lean_ctor_set(v___x_533_, 5, v___x_537_);
v___x_539_ = v___x_533_;
goto v_reusejp_538_;
}
else
{
lean_object* v_reuseFailAlloc_548_; 
v_reuseFailAlloc_548_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_548_, 0, v_depth_522_);
lean_ctor_set(v_reuseFailAlloc_548_, 1, v_levelAssignDepth_523_);
lean_ctor_set(v_reuseFailAlloc_548_, 2, v_lmvarCounter_524_);
lean_ctor_set(v_reuseFailAlloc_548_, 3, v_mvarCounter_525_);
lean_ctor_set(v_reuseFailAlloc_548_, 4, v_lDecls_526_);
lean_ctor_set(v_reuseFailAlloc_548_, 5, v___x_537_);
lean_ctor_set(v_reuseFailAlloc_548_, 6, v_userNames_528_);
lean_ctor_set(v_reuseFailAlloc_548_, 7, v_lAssignment_529_);
lean_ctor_set(v_reuseFailAlloc_548_, 8, v_eAssignment_530_);
lean_ctor_set(v_reuseFailAlloc_548_, 9, v_dAssignment_531_);
v___x_539_ = v_reuseFailAlloc_548_;
goto v_reusejp_538_;
}
v_reusejp_538_:
{
lean_object* v___x_541_; 
if (v_isShared_521_ == 0)
{
lean_ctor_set(v___x_520_, 0, v___x_539_);
v___x_541_ = v___x_520_;
goto v_reusejp_540_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v___x_539_);
lean_ctor_set(v_reuseFailAlloc_547_, 1, v_cache_515_);
lean_ctor_set(v_reuseFailAlloc_547_, 2, v_zetaDeltaFVarIds_516_);
lean_ctor_set(v_reuseFailAlloc_547_, 3, v_postponed_517_);
lean_ctor_set(v_reuseFailAlloc_547_, 4, v_diag_518_);
v___x_541_ = v_reuseFailAlloc_547_;
goto v_reusejp_540_;
}
v_reusejp_540_:
{
lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_545_; 
v___x_542_ = lean_st_ref_set(v_a_486_, v___x_541_);
v___x_543_ = lean_box(0);
if (v_isShared_499_ == 0)
{
lean_ctor_set(v___x_498_, 0, v___x_543_);
v___x_545_ = v___x_498_;
goto v_reusejp_544_;
}
else
{
lean_object* v_reuseFailAlloc_546_; 
v_reuseFailAlloc_546_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_546_, 0, v___x_543_);
v___x_545_ = v_reuseFailAlloc_546_;
goto v_reusejp_544_;
}
v_reusejp_544_:
{
return v___x_545_;
}
}
}
}
}
}
}
v___jp_552_:
{
if (lean_obj_tag(v___y_553_) == 0)
{
lean_object* v_a_554_; 
v_a_554_ = lean_ctor_get(v___y_553_, 0);
lean_inc(v_a_554_);
lean_dec_ref_known(v___y_553_, 1);
v_a_512_ = v_a_554_;
goto v___jp_511_;
}
else
{
lean_object* v_a_555_; lean_object* v___x_557_; uint8_t v_isShared_558_; uint8_t v_isSharedCheck_562_; 
lean_del_object(v___x_509_);
lean_dec(v_index_507_);
lean_dec(v_numScopeArgs_506_);
lean_dec_ref(v_localInstances_504_);
lean_dec(v_depth_503_);
lean_dec_ref(v_type_502_);
lean_dec(v_userName_500_);
lean_del_object(v___x_498_);
lean_dec(v_a_494_);
v_a_555_ = lean_ctor_get(v___y_553_, 0);
v_isSharedCheck_562_ = !lean_is_exclusive(v___y_553_);
if (v_isSharedCheck_562_ == 0)
{
v___x_557_ = v___y_553_;
v_isShared_558_ = v_isSharedCheck_562_;
goto v_resetjp_556_;
}
else
{
lean_inc(v_a_555_);
lean_dec(v___y_553_);
v___x_557_ = lean_box(0);
v_isShared_558_ = v_isSharedCheck_562_;
goto v_resetjp_556_;
}
v_resetjp_556_:
{
lean_object* v___x_560_; 
if (v_isShared_558_ == 0)
{
v___x_560_ = v___x_557_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v_a_555_);
v___x_560_ = v_reuseFailAlloc_561_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
return v___x_560_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_579_; lean_object* v___x_581_; uint8_t v_isShared_582_; uint8_t v_isSharedCheck_586_; 
lean_dec(v_a_494_);
lean_dec(v_x_480_);
v_a_579_ = lean_ctor_get(v___x_495_, 0);
v_isSharedCheck_586_ = !lean_is_exclusive(v___x_495_);
if (v_isSharedCheck_586_ == 0)
{
v___x_581_ = v___x_495_;
v_isShared_582_ = v_isSharedCheck_586_;
goto v_resetjp_580_;
}
else
{
lean_inc(v_a_579_);
lean_dec(v___x_495_);
v___x_581_ = lean_box(0);
v_isShared_582_ = v_isSharedCheck_586_;
goto v_resetjp_580_;
}
v_resetjp_580_:
{
lean_object* v___x_584_; 
if (v_isShared_582_ == 0)
{
v___x_584_ = v___x_581_;
goto v_reusejp_583_;
}
else
{
lean_object* v_reuseFailAlloc_585_; 
v_reuseFailAlloc_585_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_585_, 0, v_a_579_);
v___x_584_ = v_reuseFailAlloc_585_;
goto v_reusejp_583_;
}
v_reusejp_583_:
{
return v___x_584_;
}
}
}
}
else
{
lean_object* v_a_587_; lean_object* v___x_589_; uint8_t v_isShared_590_; uint8_t v_isSharedCheck_594_; 
lean_dec(v_x_480_);
v_a_587_ = lean_ctor_get(v___x_493_, 0);
v_isSharedCheck_594_ = !lean_is_exclusive(v___x_493_);
if (v_isSharedCheck_594_ == 0)
{
v___x_589_ = v___x_493_;
v_isShared_590_ = v_isSharedCheck_594_;
goto v_resetjp_588_;
}
else
{
lean_inc(v_a_587_);
lean_dec(v___x_493_);
v___x_589_ = lean_box(0);
v_isShared_590_ = v_isSharedCheck_594_;
goto v_resetjp_588_;
}
v_resetjp_588_:
{
lean_object* v___x_592_; 
if (v_isShared_590_ == 0)
{
v___x_592_ = v___x_589_;
goto v_reusejp_591_;
}
else
{
lean_object* v_reuseFailAlloc_593_; 
v_reuseFailAlloc_593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_593_, 0, v_a_587_);
v___x_592_ = v_reuseFailAlloc_593_;
goto v_reusejp_591_;
}
v_reusejp_591_:
{
return v___x_592_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1___boxed(lean_object* v_x_595_, lean_object* v_a_596_, lean_object* v_a_597_, lean_object* v_a_598_, lean_object* v_a_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_){
_start:
{
lean_object* v_res_605_; 
v_res_605_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1(v_x_595_, v_a_596_, v_a_597_, v_a_598_, v_a_599_, v_a_600_, v_a_601_, v_a_602_, v_a_603_);
lean_dec(v_a_603_);
lean_dec_ref(v_a_602_);
lean_dec(v_a_601_);
lean_dec_ref(v_a_600_);
lean_dec(v_a_599_);
lean_dec_ref(v_a_598_);
lean_dec(v_a_597_);
lean_dec_ref(v_a_596_);
return v_res_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2(lean_object* v_00_u03b2_606_, lean_object* v_x_607_, lean_object* v_x_608_, lean_object* v_x_609_){
_start:
{
lean_object* v___x_610_; 
v___x_610_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2___redArg(v_x_607_, v_x_608_, v_x_609_);
return v___x_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2(lean_object* v_00_u03b2_611_, lean_object* v_x_612_, size_t v_x_613_, size_t v_x_614_, lean_object* v_x_615_, lean_object* v_x_616_){
_start:
{
lean_object* v___x_617_; 
v___x_617_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___redArg(v_x_612_, v_x_613_, v_x_614_, v_x_615_, v_x_616_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2___boxed(lean_object* v_00_u03b2_618_, lean_object* v_x_619_, lean_object* v_x_620_, lean_object* v_x_621_, lean_object* v_x_622_, lean_object* v_x_623_){
_start:
{
size_t v_x_7066__boxed_624_; size_t v_x_7067__boxed_625_; lean_object* v_res_626_; 
v_x_7066__boxed_624_ = lean_unbox_usize(v_x_620_);
lean_dec(v_x_620_);
v_x_7067__boxed_625_ = lean_unbox_usize(v_x_621_);
lean_dec(v_x_621_);
v_res_626_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2(v_00_u03b2_618_, v_x_619_, v_x_7066__boxed_624_, v_x_7067__boxed_625_, v_x_622_, v_x_623_);
return v_res_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3(lean_object* v_00_u03b2_627_, lean_object* v_n_628_, lean_object* v_k_629_, lean_object* v_v_630_){
_start:
{
lean_object* v___x_631_; 
v___x_631_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3___redArg(v_n_628_, v_k_629_, v_v_630_);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4(lean_object* v_00_u03b2_632_, size_t v_depth_633_, lean_object* v_keys_634_, lean_object* v_vals_635_, lean_object* v_heq_636_, lean_object* v_i_637_, lean_object* v_entries_638_){
_start:
{
lean_object* v___x_639_; 
v___x_639_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4___redArg(v_depth_633_, v_keys_634_, v_vals_635_, v_i_637_, v_entries_638_);
return v___x_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4___boxed(lean_object* v_00_u03b2_640_, lean_object* v_depth_641_, lean_object* v_keys_642_, lean_object* v_vals_643_, lean_object* v_heq_644_, lean_object* v_i_645_, lean_object* v_entries_646_){
_start:
{
size_t v_depth_boxed_647_; lean_object* v_res_648_; 
v_depth_boxed_647_ = lean_unbox_usize(v_depth_641_);
lean_dec(v_depth_641_);
v_res_648_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__4(v_00_u03b2_640_, v_depth_boxed_647_, v_keys_642_, v_vals_643_, v_heq_644_, v_i_645_, v_entries_646_);
lean_dec_ref(v_vals_643_);
lean_dec_ref(v_keys_642_);
return v_res_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_649_, lean_object* v_x_650_, lean_object* v_x_651_, lean_object* v_x_652_, lean_object* v_x_653_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SwapVar______elabRules__Mathlib__Tactic__tacticSwap__var_____x2c_x2c__1_spec__2_spec__2_spec__3_spec__5___redArg(v_x_650_, v_x_651_, v_x_652_, v_x_653_);
return v___x_654_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SwapVar(uint8_t builtin) {
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
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_SwapVar(uint8_t builtin) {
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
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_SwapVar(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_SwapVar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_SwapVar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_SwapVar(builtin);
}
#ifdef __cplusplus
}
#endif
