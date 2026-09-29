// Lean compiler output
// Module: Mathlib.Tactic.Rename
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.ElabTerm public import Mathlib.Init
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_LocalContext_setUserName(lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getTag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVarAt(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Elab_Tactic_getFVarIds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* l_Lean_Elab_Term_addTermInfo_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
static const lean_string_object lp_mathlib_Mathlib_Tactic_renameArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "renameArg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_renameArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_renameArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 50, 67, 200, 2, 42, 93, 218)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_renameArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_renameArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_renameArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_renameArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_renameArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_renameArg___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_renameArg = (const lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "rename'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(184, 99, 177, 5, 23, 233, 25, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "rename' "};
static const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_renameArg___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rename_x27___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_rename_x27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rename_x27___closed__9_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__0(lean_object*, lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__7(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___boxed__const__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_65_ = lean_box(0);
v___x_66_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_67_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v___x_65_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg(){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg___closed__0);
v___x_70_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg___boxed(lean_object* v___y_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg();
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0(lean_object* v_00_u03b1_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg();
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___boxed(lean_object* v_00_u03b1_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0(v_00_u03b1_84_, v___y_85_, v___y_86_, v___y_87_, v___y_88_, v___y_89_, v___y_90_, v___y_91_, v___y_92_);
lean_dec(v___y_92_);
lean_dec_ref(v___y_91_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
lean_dec(v___y_88_);
lean_dec_ref(v___y_87_);
lean_dec(v___y_86_);
lean_dec_ref(v___y_85_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4___redArg(lean_object* v_as_95_, size_t v_sz_96_, size_t v_i_97_, lean_object* v_b_98_){
_start:
{
uint8_t v___x_100_; 
v___x_100_ = lean_usize_dec_lt(v_i_97_, v_sz_96_);
if (v___x_100_ == 0)
{
lean_object* v___x_101_; 
v___x_101_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_101_, 0, v_b_98_);
return v___x_101_;
}
else
{
lean_object* v_snd_102_; lean_object* v_fst_103_; lean_object* v___x_105_; uint8_t v_isShared_106_; uint8_t v_isSharedCheck_137_; 
v_snd_102_ = lean_ctor_get(v_b_98_, 1);
v_fst_103_ = lean_ctor_get(v_b_98_, 0);
v_isSharedCheck_137_ = !lean_is_exclusive(v_b_98_);
if (v_isSharedCheck_137_ == 0)
{
v___x_105_ = v_b_98_;
v_isShared_106_ = v_isSharedCheck_137_;
goto v_resetjp_104_;
}
else
{
lean_inc(v_snd_102_);
lean_inc(v_fst_103_);
lean_dec(v_b_98_);
v___x_105_ = lean_box(0);
v_isShared_106_ = v_isSharedCheck_137_;
goto v_resetjp_104_;
}
v_resetjp_104_:
{
lean_object* v_array_107_; lean_object* v_start_108_; lean_object* v_stop_109_; uint8_t v___x_110_; 
v_array_107_ = lean_ctor_get(v_snd_102_, 0);
v_start_108_ = lean_ctor_get(v_snd_102_, 1);
v_stop_109_ = lean_ctor_get(v_snd_102_, 2);
v___x_110_ = lean_nat_dec_lt(v_start_108_, v_stop_109_);
if (v___x_110_ == 0)
{
lean_object* v___x_112_; 
if (v_isShared_106_ == 0)
{
v___x_112_ = v___x_105_;
goto v_reusejp_111_;
}
else
{
lean_object* v_reuseFailAlloc_114_; 
v_reuseFailAlloc_114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_114_, 0, v_fst_103_);
lean_ctor_set(v_reuseFailAlloc_114_, 1, v_snd_102_);
v___x_112_ = v_reuseFailAlloc_114_;
goto v_reusejp_111_;
}
v_reusejp_111_:
{
lean_object* v___x_113_; 
v___x_113_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
return v___x_113_;
}
}
else
{
lean_object* v___x_116_; uint8_t v_isShared_117_; uint8_t v_isSharedCheck_133_; 
lean_inc(v_stop_109_);
lean_inc(v_start_108_);
lean_inc_ref(v_array_107_);
v_isSharedCheck_133_ = !lean_is_exclusive(v_snd_102_);
if (v_isSharedCheck_133_ == 0)
{
lean_object* v_unused_134_; lean_object* v_unused_135_; lean_object* v_unused_136_; 
v_unused_134_ = lean_ctor_get(v_snd_102_, 2);
lean_dec(v_unused_134_);
v_unused_135_ = lean_ctor_get(v_snd_102_, 1);
lean_dec(v_unused_135_);
v_unused_136_ = lean_ctor_get(v_snd_102_, 0);
lean_dec(v_unused_136_);
v___x_116_ = v_snd_102_;
v_isShared_117_ = v_isSharedCheck_133_;
goto v_resetjp_115_;
}
else
{
lean_dec(v_snd_102_);
v___x_116_ = lean_box(0);
v_isShared_117_ = v_isSharedCheck_133_;
goto v_resetjp_115_;
}
v_resetjp_115_:
{
lean_object* v_a_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_123_; 
v_a_118_ = lean_array_uget_borrowed(v_as_95_, v_i_97_);
v___x_119_ = lean_array_fget(v_array_107_, v_start_108_);
v___x_120_ = lean_unsigned_to_nat(1u);
v___x_121_ = lean_nat_add(v_start_108_, v___x_120_);
lean_dec(v_start_108_);
if (v_isShared_117_ == 0)
{
lean_ctor_set(v___x_116_, 1, v___x_121_);
v___x_123_ = v___x_116_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_132_; 
v_reuseFailAlloc_132_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_132_, 0, v_array_107_);
lean_ctor_set(v_reuseFailAlloc_132_, 1, v___x_121_);
lean_ctor_set(v_reuseFailAlloc_132_, 2, v_stop_109_);
v___x_123_ = v_reuseFailAlloc_132_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_127_; 
v___x_124_ = l_Lean_Syntax_getId(v___x_119_);
lean_dec(v___x_119_);
lean_inc(v_a_118_);
v___x_125_ = l_Lean_LocalContext_setUserName(v_fst_103_, v_a_118_, v___x_124_);
if (v_isShared_106_ == 0)
{
lean_ctor_set(v___x_105_, 1, v___x_123_);
lean_ctor_set(v___x_105_, 0, v___x_125_);
v___x_127_ = v___x_105_;
goto v_reusejp_126_;
}
else
{
lean_object* v_reuseFailAlloc_131_; 
v_reuseFailAlloc_131_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_131_, 0, v___x_125_);
lean_ctor_set(v_reuseFailAlloc_131_, 1, v___x_123_);
v___x_127_ = v_reuseFailAlloc_131_;
goto v_reusejp_126_;
}
v_reusejp_126_:
{
size_t v___x_128_; size_t v___x_129_; 
v___x_128_ = ((size_t)1ULL);
v___x_129_ = lean_usize_add(v_i_97_, v___x_128_);
v_i_97_ = v___x_129_;
v_b_98_ = v___x_127_;
goto _start;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4___redArg___boxed(lean_object* v_as_138_, lean_object* v_sz_139_, lean_object* v_i_140_, lean_object* v_b_141_, lean_object* v___y_142_){
_start:
{
size_t v_sz_boxed_143_; size_t v_i_boxed_144_; lean_object* v_res_145_; 
v_sz_boxed_143_ = lean_unbox_usize(v_sz_139_);
lean_dec(v_sz_139_);
v_i_boxed_144_ = lean_unbox_usize(v_i_140_);
lean_dec(v_i_140_);
v_res_145_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4___redArg(v_as_138_, v_sz_boxed_143_, v_i_boxed_144_, v_b_141_);
lean_dec_ref(v_as_138_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9_spec__10___redArg(lean_object* v_x_146_, lean_object* v_x_147_, lean_object* v_x_148_, lean_object* v_x_149_){
_start:
{
lean_object* v_ks_150_; lean_object* v_vs_151_; lean_object* v___x_153_; uint8_t v_isShared_154_; uint8_t v_isSharedCheck_175_; 
v_ks_150_ = lean_ctor_get(v_x_146_, 0);
v_vs_151_ = lean_ctor_get(v_x_146_, 1);
v_isSharedCheck_175_ = !lean_is_exclusive(v_x_146_);
if (v_isSharedCheck_175_ == 0)
{
v___x_153_ = v_x_146_;
v_isShared_154_ = v_isSharedCheck_175_;
goto v_resetjp_152_;
}
else
{
lean_inc(v_vs_151_);
lean_inc(v_ks_150_);
lean_dec(v_x_146_);
v___x_153_ = lean_box(0);
v_isShared_154_ = v_isSharedCheck_175_;
goto v_resetjp_152_;
}
v_resetjp_152_:
{
lean_object* v___x_155_; uint8_t v___x_156_; 
v___x_155_ = lean_array_get_size(v_ks_150_);
v___x_156_ = lean_nat_dec_lt(v_x_147_, v___x_155_);
if (v___x_156_ == 0)
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_160_; 
lean_dec(v_x_147_);
v___x_157_ = lean_array_push(v_ks_150_, v_x_148_);
v___x_158_ = lean_array_push(v_vs_151_, v_x_149_);
if (v_isShared_154_ == 0)
{
lean_ctor_set(v___x_153_, 1, v___x_158_);
lean_ctor_set(v___x_153_, 0, v___x_157_);
v___x_160_ = v___x_153_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v___x_157_);
lean_ctor_set(v_reuseFailAlloc_161_, 1, v___x_158_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
else
{
lean_object* v_k_x27_162_; uint8_t v___x_163_; 
v_k_x27_162_ = lean_array_fget_borrowed(v_ks_150_, v_x_147_);
v___x_163_ = l_Lean_instBEqMVarId_beq(v_x_148_, v_k_x27_162_);
if (v___x_163_ == 0)
{
lean_object* v___x_165_; 
if (v_isShared_154_ == 0)
{
v___x_165_ = v___x_153_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_169_; 
v_reuseFailAlloc_169_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_169_, 0, v_ks_150_);
lean_ctor_set(v_reuseFailAlloc_169_, 1, v_vs_151_);
v___x_165_ = v_reuseFailAlloc_169_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
lean_object* v___x_166_; lean_object* v___x_167_; 
v___x_166_ = lean_unsigned_to_nat(1u);
v___x_167_ = lean_nat_add(v_x_147_, v___x_166_);
lean_dec(v_x_147_);
v_x_146_ = v___x_165_;
v_x_147_ = v___x_167_;
goto _start;
}
}
else
{
lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_173_; 
v___x_170_ = lean_array_fset(v_ks_150_, v_x_147_, v_x_148_);
v___x_171_ = lean_array_fset(v_vs_151_, v_x_147_, v_x_149_);
lean_dec(v_x_147_);
if (v_isShared_154_ == 0)
{
lean_ctor_set(v___x_153_, 1, v___x_171_);
lean_ctor_set(v___x_153_, 0, v___x_170_);
v___x_173_ = v___x_153_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v___x_170_);
lean_ctor_set(v_reuseFailAlloc_174_, 1, v___x_171_);
v___x_173_ = v_reuseFailAlloc_174_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
return v___x_173_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9___redArg(lean_object* v_n_176_, lean_object* v_k_177_, lean_object* v_v_178_){
_start:
{
lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_179_ = lean_unsigned_to_nat(0u);
v___x_180_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9_spec__10___redArg(v_n_176_, v___x_179_, v_k_177_, v_v_178_);
return v___x_180_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg(lean_object* v_x_182_, size_t v_x_183_, size_t v_x_184_, lean_object* v_x_185_, lean_object* v_x_186_){
_start:
{
if (lean_obj_tag(v_x_182_) == 0)
{
lean_object* v_es_187_; size_t v___x_188_; size_t v___x_189_; lean_object* v_j_190_; lean_object* v___x_191_; uint8_t v___x_192_; 
v_es_187_ = lean_ctor_get(v_x_182_, 0);
v___x_188_ = ((size_t)31ULL);
v___x_189_ = lean_usize_land(v_x_183_, v___x_188_);
v_j_190_ = lean_usize_to_nat(v___x_189_);
v___x_191_ = lean_array_get_size(v_es_187_);
v___x_192_ = lean_nat_dec_lt(v_j_190_, v___x_191_);
if (v___x_192_ == 0)
{
lean_dec(v_j_190_);
lean_dec(v_x_186_);
lean_dec(v_x_185_);
return v_x_182_;
}
else
{
lean_object* v___x_194_; uint8_t v_isShared_195_; uint8_t v_isSharedCheck_231_; 
lean_inc_ref(v_es_187_);
v_isSharedCheck_231_ = !lean_is_exclusive(v_x_182_);
if (v_isSharedCheck_231_ == 0)
{
lean_object* v_unused_232_; 
v_unused_232_ = lean_ctor_get(v_x_182_, 0);
lean_dec(v_unused_232_);
v___x_194_ = v_x_182_;
v_isShared_195_ = v_isSharedCheck_231_;
goto v_resetjp_193_;
}
else
{
lean_dec(v_x_182_);
v___x_194_ = lean_box(0);
v_isShared_195_ = v_isSharedCheck_231_;
goto v_resetjp_193_;
}
v_resetjp_193_:
{
lean_object* v_v_196_; lean_object* v___x_197_; lean_object* v_xs_x27_198_; lean_object* v___y_200_; 
v_v_196_ = lean_array_fget(v_es_187_, v_j_190_);
v___x_197_ = lean_box(0);
v_xs_x27_198_ = lean_array_fset(v_es_187_, v_j_190_, v___x_197_);
switch(lean_obj_tag(v_v_196_))
{
case 0:
{
lean_object* v_key_205_; lean_object* v_val_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_216_; 
v_key_205_ = lean_ctor_get(v_v_196_, 0);
v_val_206_ = lean_ctor_get(v_v_196_, 1);
v_isSharedCheck_216_ = !lean_is_exclusive(v_v_196_);
if (v_isSharedCheck_216_ == 0)
{
v___x_208_ = v_v_196_;
v_isShared_209_ = v_isSharedCheck_216_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_val_206_);
lean_inc(v_key_205_);
lean_dec(v_v_196_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_216_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
uint8_t v___x_210_; 
v___x_210_ = l_Lean_instBEqMVarId_beq(v_x_185_, v_key_205_);
if (v___x_210_ == 0)
{
lean_object* v___x_211_; lean_object* v___x_212_; 
lean_del_object(v___x_208_);
v___x_211_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_205_, v_val_206_, v_x_185_, v_x_186_);
v___x_212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_212_, 0, v___x_211_);
v___y_200_ = v___x_212_;
goto v___jp_199_;
}
else
{
lean_object* v___x_214_; 
lean_dec(v_val_206_);
lean_dec(v_key_205_);
if (v_isShared_209_ == 0)
{
lean_ctor_set(v___x_208_, 1, v_x_186_);
lean_ctor_set(v___x_208_, 0, v_x_185_);
v___x_214_ = v___x_208_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v_x_185_);
lean_ctor_set(v_reuseFailAlloc_215_, 1, v_x_186_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
v___y_200_ = v___x_214_;
goto v___jp_199_;
}
}
}
}
case 1:
{
lean_object* v_node_217_; lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_229_; 
v_node_217_ = lean_ctor_get(v_v_196_, 0);
v_isSharedCheck_229_ = !lean_is_exclusive(v_v_196_);
if (v_isSharedCheck_229_ == 0)
{
v___x_219_ = v_v_196_;
v_isShared_220_ = v_isSharedCheck_229_;
goto v_resetjp_218_;
}
else
{
lean_inc(v_node_217_);
lean_dec(v_v_196_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_229_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
size_t v___x_221_; size_t v___x_222_; size_t v___x_223_; size_t v___x_224_; lean_object* v___x_225_; lean_object* v___x_227_; 
v___x_221_ = ((size_t)5ULL);
v___x_222_ = lean_usize_shift_right(v_x_183_, v___x_221_);
v___x_223_ = ((size_t)1ULL);
v___x_224_ = lean_usize_add(v_x_184_, v___x_223_);
v___x_225_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg(v_node_217_, v___x_222_, v___x_224_, v_x_185_, v_x_186_);
if (v_isShared_220_ == 0)
{
lean_ctor_set(v___x_219_, 0, v___x_225_);
v___x_227_ = v___x_219_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v___x_225_);
v___x_227_ = v_reuseFailAlloc_228_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
v___y_200_ = v___x_227_;
goto v___jp_199_;
}
}
}
default: 
{
lean_object* v___x_230_; 
v___x_230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_230_, 0, v_x_185_);
lean_ctor_set(v___x_230_, 1, v_x_186_);
v___y_200_ = v___x_230_;
goto v___jp_199_;
}
}
v___jp_199_:
{
lean_object* v___x_201_; lean_object* v___x_203_; 
v___x_201_ = lean_array_fset(v_xs_x27_198_, v_j_190_, v___y_200_);
lean_dec(v_j_190_);
if (v_isShared_195_ == 0)
{
lean_ctor_set(v___x_194_, 0, v___x_201_);
v___x_203_ = v___x_194_;
goto v_reusejp_202_;
}
else
{
lean_object* v_reuseFailAlloc_204_; 
v_reuseFailAlloc_204_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_204_, 0, v___x_201_);
v___x_203_ = v_reuseFailAlloc_204_;
goto v_reusejp_202_;
}
v_reusejp_202_:
{
return v___x_203_;
}
}
}
}
}
else
{
lean_object* v_ks_233_; lean_object* v_vs_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_254_; 
v_ks_233_ = lean_ctor_get(v_x_182_, 0);
v_vs_234_ = lean_ctor_get(v_x_182_, 1);
v_isSharedCheck_254_ = !lean_is_exclusive(v_x_182_);
if (v_isSharedCheck_254_ == 0)
{
v___x_236_ = v_x_182_;
v_isShared_237_ = v_isSharedCheck_254_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_vs_234_);
lean_inc(v_ks_233_);
lean_dec(v_x_182_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_254_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
lean_object* v___x_239_; 
if (v_isShared_237_ == 0)
{
v___x_239_ = v___x_236_;
goto v_reusejp_238_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v_ks_233_);
lean_ctor_set(v_reuseFailAlloc_253_, 1, v_vs_234_);
v___x_239_ = v_reuseFailAlloc_253_;
goto v_reusejp_238_;
}
v_reusejp_238_:
{
lean_object* v_newNode_240_; uint8_t v___y_242_; size_t v___x_248_; uint8_t v___x_249_; 
v_newNode_240_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9___redArg(v___x_239_, v_x_185_, v_x_186_);
v___x_248_ = ((size_t)7ULL);
v___x_249_ = lean_usize_dec_le(v___x_248_, v_x_184_);
if (v___x_249_ == 0)
{
lean_object* v___x_250_; lean_object* v___x_251_; uint8_t v___x_252_; 
v___x_250_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_240_);
v___x_251_ = lean_unsigned_to_nat(4u);
v___x_252_ = lean_nat_dec_lt(v___x_250_, v___x_251_);
lean_dec(v___x_250_);
v___y_242_ = v___x_252_;
goto v___jp_241_;
}
else
{
v___y_242_ = v___x_249_;
goto v___jp_241_;
}
v___jp_241_:
{
if (v___y_242_ == 0)
{
lean_object* v_ks_243_; lean_object* v_vs_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; 
v_ks_243_ = lean_ctor_get(v_newNode_240_, 0);
lean_inc_ref(v_ks_243_);
v_vs_244_ = lean_ctor_get(v_newNode_240_, 1);
lean_inc_ref(v_vs_244_);
lean_dec_ref(v_newNode_240_);
v___x_245_ = lean_unsigned_to_nat(0u);
v___x_246_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg___closed__0);
v___x_247_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10___redArg(v_x_184_, v_ks_243_, v_vs_244_, v___x_245_, v___x_246_);
lean_dec_ref(v_vs_244_);
lean_dec_ref(v_ks_243_);
return v___x_247_;
}
else
{
return v_newNode_240_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10___redArg(size_t v_depth_255_, lean_object* v_keys_256_, lean_object* v_vals_257_, lean_object* v_i_258_, lean_object* v_entries_259_){
_start:
{
lean_object* v___x_260_; uint8_t v___x_261_; 
v___x_260_ = lean_array_get_size(v_keys_256_);
v___x_261_ = lean_nat_dec_lt(v_i_258_, v___x_260_);
if (v___x_261_ == 0)
{
lean_dec(v_i_258_);
return v_entries_259_;
}
else
{
lean_object* v_k_262_; lean_object* v_v_263_; uint64_t v___x_264_; size_t v_h_265_; size_t v___x_266_; lean_object* v___x_267_; size_t v___x_268_; size_t v___x_269_; size_t v___x_270_; size_t v_h_271_; lean_object* v___x_272_; lean_object* v___x_273_; 
v_k_262_ = lean_array_fget_borrowed(v_keys_256_, v_i_258_);
v_v_263_ = lean_array_fget_borrowed(v_vals_257_, v_i_258_);
v___x_264_ = l_Lean_instHashableMVarId_hash(v_k_262_);
v_h_265_ = lean_uint64_to_usize(v___x_264_);
v___x_266_ = ((size_t)5ULL);
v___x_267_ = lean_unsigned_to_nat(1u);
v___x_268_ = ((size_t)1ULL);
v___x_269_ = lean_usize_sub(v_depth_255_, v___x_268_);
v___x_270_ = lean_usize_mul(v___x_266_, v___x_269_);
v_h_271_ = lean_usize_shift_right(v_h_265_, v___x_270_);
v___x_272_ = lean_nat_add(v_i_258_, v___x_267_);
lean_dec(v_i_258_);
lean_inc(v_v_263_);
lean_inc(v_k_262_);
v___x_273_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg(v_entries_259_, v_h_271_, v_depth_255_, v_k_262_, v_v_263_);
v_i_258_ = v___x_272_;
v_entries_259_ = v___x_273_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10___redArg___boxed(lean_object* v_depth_275_, lean_object* v_keys_276_, lean_object* v_vals_277_, lean_object* v_i_278_, lean_object* v_entries_279_){
_start:
{
size_t v_depth_boxed_280_; lean_object* v_res_281_; 
v_depth_boxed_280_ = lean_unbox_usize(v_depth_275_);
lean_dec(v_depth_275_);
v_res_281_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10___redArg(v_depth_boxed_280_, v_keys_276_, v_vals_277_, v_i_278_, v_entries_279_);
lean_dec_ref(v_vals_277_);
lean_dec_ref(v_keys_276_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg___boxed(lean_object* v_x_282_, lean_object* v_x_283_, lean_object* v_x_284_, lean_object* v_x_285_, lean_object* v_x_286_){
_start:
{
size_t v_x_6672__boxed_287_; size_t v_x_6673__boxed_288_; lean_object* v_res_289_; 
v_x_6672__boxed_287_ = lean_unbox_usize(v_x_283_);
lean_dec(v_x_283_);
v_x_6673__boxed_288_ = lean_unbox_usize(v_x_284_);
lean_dec(v_x_284_);
v_res_289_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg(v_x_282_, v_x_6672__boxed_287_, v_x_6673__boxed_288_, v_x_285_, v_x_286_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5___redArg(lean_object* v_x_290_, lean_object* v_x_291_, lean_object* v_x_292_){
_start:
{
uint64_t v___x_293_; size_t v___x_294_; size_t v___x_295_; lean_object* v___x_296_; 
v___x_293_ = l_Lean_instHashableMVarId_hash(v_x_291_);
v___x_294_ = lean_uint64_to_usize(v___x_293_);
v___x_295_ = ((size_t)1ULL);
v___x_296_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg(v_x_290_, v___x_294_, v___x_295_, v_x_291_, v_x_292_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5___redArg(lean_object* v_mvarId_297_, lean_object* v_val_298_, lean_object* v___y_299_){
_start:
{
lean_object* v___x_301_; lean_object* v_mctx_302_; lean_object* v_cache_303_; lean_object* v_zetaDeltaFVarIds_304_; lean_object* v_postponed_305_; lean_object* v_diag_306_; lean_object* v___x_308_; uint8_t v_isShared_309_; uint8_t v_isSharedCheck_334_; 
v___x_301_ = lean_st_ref_take(v___y_299_);
v_mctx_302_ = lean_ctor_get(v___x_301_, 0);
v_cache_303_ = lean_ctor_get(v___x_301_, 1);
v_zetaDeltaFVarIds_304_ = lean_ctor_get(v___x_301_, 2);
v_postponed_305_ = lean_ctor_get(v___x_301_, 3);
v_diag_306_ = lean_ctor_get(v___x_301_, 4);
v_isSharedCheck_334_ = !lean_is_exclusive(v___x_301_);
if (v_isSharedCheck_334_ == 0)
{
v___x_308_ = v___x_301_;
v_isShared_309_ = v_isSharedCheck_334_;
goto v_resetjp_307_;
}
else
{
lean_inc(v_diag_306_);
lean_inc(v_postponed_305_);
lean_inc(v_zetaDeltaFVarIds_304_);
lean_inc(v_cache_303_);
lean_inc(v_mctx_302_);
lean_dec(v___x_301_);
v___x_308_ = lean_box(0);
v_isShared_309_ = v_isSharedCheck_334_;
goto v_resetjp_307_;
}
v_resetjp_307_:
{
lean_object* v_depth_310_; lean_object* v_levelAssignDepth_311_; lean_object* v_lmvarCounter_312_; lean_object* v_mvarCounter_313_; lean_object* v_lDecls_314_; lean_object* v_decls_315_; lean_object* v_userNames_316_; lean_object* v_lAssignment_317_; lean_object* v_eAssignment_318_; lean_object* v_dAssignment_319_; lean_object* v___x_321_; uint8_t v_isShared_322_; uint8_t v_isSharedCheck_333_; 
v_depth_310_ = lean_ctor_get(v_mctx_302_, 0);
v_levelAssignDepth_311_ = lean_ctor_get(v_mctx_302_, 1);
v_lmvarCounter_312_ = lean_ctor_get(v_mctx_302_, 2);
v_mvarCounter_313_ = lean_ctor_get(v_mctx_302_, 3);
v_lDecls_314_ = lean_ctor_get(v_mctx_302_, 4);
v_decls_315_ = lean_ctor_get(v_mctx_302_, 5);
v_userNames_316_ = lean_ctor_get(v_mctx_302_, 6);
v_lAssignment_317_ = lean_ctor_get(v_mctx_302_, 7);
v_eAssignment_318_ = lean_ctor_get(v_mctx_302_, 8);
v_dAssignment_319_ = lean_ctor_get(v_mctx_302_, 9);
v_isSharedCheck_333_ = !lean_is_exclusive(v_mctx_302_);
if (v_isSharedCheck_333_ == 0)
{
v___x_321_ = v_mctx_302_;
v_isShared_322_ = v_isSharedCheck_333_;
goto v_resetjp_320_;
}
else
{
lean_inc(v_dAssignment_319_);
lean_inc(v_eAssignment_318_);
lean_inc(v_lAssignment_317_);
lean_inc(v_userNames_316_);
lean_inc(v_decls_315_);
lean_inc(v_lDecls_314_);
lean_inc(v_mvarCounter_313_);
lean_inc(v_lmvarCounter_312_);
lean_inc(v_levelAssignDepth_311_);
lean_inc(v_depth_310_);
lean_dec(v_mctx_302_);
v___x_321_ = lean_box(0);
v_isShared_322_ = v_isSharedCheck_333_;
goto v_resetjp_320_;
}
v_resetjp_320_:
{
lean_object* v___x_323_; lean_object* v___x_325_; 
v___x_323_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5___redArg(v_eAssignment_318_, v_mvarId_297_, v_val_298_);
if (v_isShared_322_ == 0)
{
lean_ctor_set(v___x_321_, 8, v___x_323_);
v___x_325_ = v___x_321_;
goto v_reusejp_324_;
}
else
{
lean_object* v_reuseFailAlloc_332_; 
v_reuseFailAlloc_332_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_332_, 0, v_depth_310_);
lean_ctor_set(v_reuseFailAlloc_332_, 1, v_levelAssignDepth_311_);
lean_ctor_set(v_reuseFailAlloc_332_, 2, v_lmvarCounter_312_);
lean_ctor_set(v_reuseFailAlloc_332_, 3, v_mvarCounter_313_);
lean_ctor_set(v_reuseFailAlloc_332_, 4, v_lDecls_314_);
lean_ctor_set(v_reuseFailAlloc_332_, 5, v_decls_315_);
lean_ctor_set(v_reuseFailAlloc_332_, 6, v_userNames_316_);
lean_ctor_set(v_reuseFailAlloc_332_, 7, v_lAssignment_317_);
lean_ctor_set(v_reuseFailAlloc_332_, 8, v___x_323_);
lean_ctor_set(v_reuseFailAlloc_332_, 9, v_dAssignment_319_);
v___x_325_ = v_reuseFailAlloc_332_;
goto v_reusejp_324_;
}
v_reusejp_324_:
{
lean_object* v___x_327_; 
if (v_isShared_309_ == 0)
{
lean_ctor_set(v___x_308_, 0, v___x_325_);
v___x_327_ = v___x_308_;
goto v_reusejp_326_;
}
else
{
lean_object* v_reuseFailAlloc_331_; 
v_reuseFailAlloc_331_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_331_, 0, v___x_325_);
lean_ctor_set(v_reuseFailAlloc_331_, 1, v_cache_303_);
lean_ctor_set(v_reuseFailAlloc_331_, 2, v_zetaDeltaFVarIds_304_);
lean_ctor_set(v_reuseFailAlloc_331_, 3, v_postponed_305_);
lean_ctor_set(v_reuseFailAlloc_331_, 4, v_diag_306_);
v___x_327_ = v_reuseFailAlloc_331_;
goto v_reusejp_326_;
}
v_reusejp_326_:
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_328_ = lean_st_ref_set(v___y_299_, v___x_327_);
v___x_329_ = lean_box(0);
v___x_330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_330_, 0, v___x_329_);
return v___x_330_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5___redArg___boxed(lean_object* v_mvarId_335_, lean_object* v_val_336_, lean_object* v___y_337_, lean_object* v___y_338_){
_start:
{
lean_object* v_res_339_; 
v_res_339_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5___redArg(v_mvarId_335_, v_val_336_, v___y_337_);
lean_dec(v___y_337_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__0(lean_object* v_bs_340_, lean_object* v___x_341_, lean_object* v_a_342_, size_t v___x_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_345_, v___y_348_, v___y_349_, v___y_350_, v___y_351_);
if (lean_obj_tag(v___x_353_) == 0)
{
lean_object* v_a_354_; lean_object* v_lctx_355_; lean_object* v_localInstances_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; size_t v_sz_360_; lean_object* v___x_361_; 
v_a_354_ = lean_ctor_get(v___x_353_, 0);
lean_inc(v_a_354_);
lean_dec_ref_known(v___x_353_, 1);
v_lctx_355_ = lean_ctor_get(v___y_348_, 2);
v_localInstances_356_ = lean_ctor_get(v___y_348_, 3);
v___x_357_ = lean_array_get_size(v_bs_340_);
lean_inc(v___x_341_);
v___x_358_ = l_Array_toSubarray___redArg(v_bs_340_, v___x_341_, v___x_357_);
lean_inc_ref(v_lctx_355_);
v___x_359_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_359_, 0, v_lctx_355_);
lean_ctor_set(v___x_359_, 1, v___x_358_);
v_sz_360_ = lean_array_size(v_a_342_);
v___x_361_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4___redArg(v_a_342_, v_sz_360_, v___x_343_, v___x_359_);
if (lean_obj_tag(v___x_361_) == 0)
{
lean_object* v_a_362_; lean_object* v___x_363_; 
v_a_362_ = lean_ctor_get(v___x_361_, 0);
lean_inc(v_a_362_);
lean_dec_ref_known(v___x_361_, 1);
lean_inc(v_a_354_);
v___x_363_ = l_Lean_MVarId_getType(v_a_354_, v___y_348_, v___y_349_, v___y_350_, v___y_351_);
if (lean_obj_tag(v___x_363_) == 0)
{
lean_object* v_a_364_; lean_object* v___x_365_; 
v_a_364_ = lean_ctor_get(v___x_363_, 0);
lean_inc(v_a_364_);
lean_dec_ref_known(v___x_363_, 1);
lean_inc(v_a_354_);
v___x_365_ = l_Lean_MVarId_getTag(v_a_354_, v___y_348_, v___y_349_, v___y_350_, v___y_351_);
if (lean_obj_tag(v___x_365_) == 0)
{
lean_object* v_a_366_; lean_object* v_fst_367_; lean_object* v___x_369_; uint8_t v_isShared_370_; uint8_t v_isSharedCheck_389_; 
v_a_366_ = lean_ctor_get(v___x_365_, 0);
lean_inc(v_a_366_);
lean_dec_ref_known(v___x_365_, 1);
v_fst_367_ = lean_ctor_get(v_a_362_, 0);
v_isSharedCheck_389_ = !lean_is_exclusive(v_a_362_);
if (v_isSharedCheck_389_ == 0)
{
lean_object* v_unused_390_; 
v_unused_390_ = lean_ctor_get(v_a_362_, 1);
lean_dec(v_unused_390_);
v___x_369_ = v_a_362_;
v_isShared_370_ = v_isSharedCheck_389_;
goto v_resetjp_368_;
}
else
{
lean_inc(v_fst_367_);
lean_dec(v_a_362_);
v___x_369_ = lean_box(0);
v_isShared_370_ = v_isSharedCheck_389_;
goto v_resetjp_368_;
}
v_resetjp_368_:
{
uint8_t v___x_371_; lean_object* v___x_372_; 
v___x_371_ = 2;
lean_inc_ref(v_localInstances_356_);
v___x_372_ = l_Lean_Meta_mkFreshExprMVarAt(v_fst_367_, v_localInstances_356_, v_a_364_, v___x_371_, v_a_366_, v___x_341_, v___y_348_, v___y_349_, v___y_350_, v___y_351_);
if (lean_obj_tag(v___x_372_) == 0)
{
lean_object* v_a_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_378_; 
v_a_373_ = lean_ctor_get(v___x_372_, 0);
lean_inc_n(v_a_373_, 2);
lean_dec_ref_known(v___x_372_, 1);
v___x_374_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5___redArg(v_a_354_, v_a_373_, v___y_349_);
lean_dec_ref(v___x_374_);
v___x_375_ = l_Lean_Expr_mvarId_x21(v_a_373_);
lean_dec(v_a_373_);
v___x_376_ = lean_box(0);
if (v_isShared_370_ == 0)
{
lean_ctor_set_tag(v___x_369_, 1);
lean_ctor_set(v___x_369_, 1, v___x_376_);
lean_ctor_set(v___x_369_, 0, v___x_375_);
v___x_378_ = v___x_369_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v___x_375_);
lean_ctor_set(v_reuseFailAlloc_380_, 1, v___x_376_);
v___x_378_ = v_reuseFailAlloc_380_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
lean_object* v___x_379_; 
v___x_379_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_378_, v___y_345_, v___y_348_, v___y_349_, v___y_350_, v___y_351_);
lean_dec_ref(v___y_348_);
return v___x_379_;
}
}
else
{
lean_object* v_a_381_; lean_object* v___x_383_; uint8_t v_isShared_384_; uint8_t v_isSharedCheck_388_; 
lean_del_object(v___x_369_);
lean_dec(v_a_354_);
lean_dec_ref(v___y_348_);
v_a_381_ = lean_ctor_get(v___x_372_, 0);
v_isSharedCheck_388_ = !lean_is_exclusive(v___x_372_);
if (v_isSharedCheck_388_ == 0)
{
v___x_383_ = v___x_372_;
v_isShared_384_ = v_isSharedCheck_388_;
goto v_resetjp_382_;
}
else
{
lean_inc(v_a_381_);
lean_dec(v___x_372_);
v___x_383_ = lean_box(0);
v_isShared_384_ = v_isSharedCheck_388_;
goto v_resetjp_382_;
}
v_resetjp_382_:
{
lean_object* v___x_386_; 
if (v_isShared_384_ == 0)
{
v___x_386_ = v___x_383_;
goto v_reusejp_385_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v_a_381_);
v___x_386_ = v_reuseFailAlloc_387_;
goto v_reusejp_385_;
}
v_reusejp_385_:
{
return v___x_386_;
}
}
}
}
}
else
{
lean_object* v_a_391_; lean_object* v___x_393_; uint8_t v_isShared_394_; uint8_t v_isSharedCheck_398_; 
lean_dec(v_a_364_);
lean_dec(v_a_362_);
lean_dec(v_a_354_);
lean_dec_ref(v___y_348_);
lean_dec(v___x_341_);
v_a_391_ = lean_ctor_get(v___x_365_, 0);
v_isSharedCheck_398_ = !lean_is_exclusive(v___x_365_);
if (v_isSharedCheck_398_ == 0)
{
v___x_393_ = v___x_365_;
v_isShared_394_ = v_isSharedCheck_398_;
goto v_resetjp_392_;
}
else
{
lean_inc(v_a_391_);
lean_dec(v___x_365_);
v___x_393_ = lean_box(0);
v_isShared_394_ = v_isSharedCheck_398_;
goto v_resetjp_392_;
}
v_resetjp_392_:
{
lean_object* v___x_396_; 
if (v_isShared_394_ == 0)
{
v___x_396_ = v___x_393_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_397_; 
v_reuseFailAlloc_397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_397_, 0, v_a_391_);
v___x_396_ = v_reuseFailAlloc_397_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
return v___x_396_;
}
}
}
}
else
{
lean_object* v_a_399_; lean_object* v___x_401_; uint8_t v_isShared_402_; uint8_t v_isSharedCheck_406_; 
lean_dec(v_a_362_);
lean_dec(v_a_354_);
lean_dec_ref(v___y_348_);
lean_dec(v___x_341_);
v_a_399_ = lean_ctor_get(v___x_363_, 0);
v_isSharedCheck_406_ = !lean_is_exclusive(v___x_363_);
if (v_isSharedCheck_406_ == 0)
{
v___x_401_ = v___x_363_;
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
else
{
lean_inc(v_a_399_);
lean_dec(v___x_363_);
v___x_401_ = lean_box(0);
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
v_resetjp_400_:
{
lean_object* v___x_404_; 
if (v_isShared_402_ == 0)
{
v___x_404_ = v___x_401_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v_a_399_);
v___x_404_ = v_reuseFailAlloc_405_;
goto v_reusejp_403_;
}
v_reusejp_403_:
{
return v___x_404_;
}
}
}
}
else
{
lean_object* v_a_407_; lean_object* v___x_409_; uint8_t v_isShared_410_; uint8_t v_isSharedCheck_414_; 
lean_dec(v_a_354_);
lean_dec_ref(v___y_348_);
lean_dec(v___x_341_);
v_a_407_ = lean_ctor_get(v___x_361_, 0);
v_isSharedCheck_414_ = !lean_is_exclusive(v___x_361_);
if (v_isSharedCheck_414_ == 0)
{
v___x_409_ = v___x_361_;
v_isShared_410_ = v_isSharedCheck_414_;
goto v_resetjp_408_;
}
else
{
lean_inc(v_a_407_);
lean_dec(v___x_361_);
v___x_409_ = lean_box(0);
v_isShared_410_ = v_isSharedCheck_414_;
goto v_resetjp_408_;
}
v_resetjp_408_:
{
lean_object* v___x_412_; 
if (v_isShared_410_ == 0)
{
v___x_412_ = v___x_409_;
goto v_reusejp_411_;
}
else
{
lean_object* v_reuseFailAlloc_413_; 
v_reuseFailAlloc_413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_413_, 0, v_a_407_);
v___x_412_ = v_reuseFailAlloc_413_;
goto v_reusejp_411_;
}
v_reusejp_411_:
{
return v___x_412_;
}
}
}
}
else
{
lean_object* v_a_415_; lean_object* v___x_417_; uint8_t v_isShared_418_; uint8_t v_isSharedCheck_422_; 
lean_dec_ref(v___y_348_);
lean_dec(v___x_341_);
lean_dec_ref(v_bs_340_);
v_a_415_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_422_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_422_ == 0)
{
v___x_417_ = v___x_353_;
v_isShared_418_ = v_isSharedCheck_422_;
goto v_resetjp_416_;
}
else
{
lean_inc(v_a_415_);
lean_dec(v___x_353_);
v___x_417_ = lean_box(0);
v_isShared_418_ = v_isSharedCheck_422_;
goto v_resetjp_416_;
}
v_resetjp_416_:
{
lean_object* v___x_420_; 
if (v_isShared_418_ == 0)
{
v___x_420_ = v___x_417_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v_a_415_);
v___x_420_ = v_reuseFailAlloc_421_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
return v___x_420_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__0___boxed(lean_object* v_bs_423_, lean_object* v___x_424_, lean_object* v_a_425_, lean_object* v___x_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_){
_start:
{
size_t v___x_6887__boxed_436_; lean_object* v_res_437_; 
v___x_6887__boxed_436_ = lean_unbox_usize(v___x_426_);
lean_dec(v___x_426_);
v_res_437_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__0(v_bs_423_, v___x_424_, v_a_425_, v___x_6887__boxed_436_, v___y_427_, v___y_428_, v___y_429_, v___y_430_, v___y_431_, v___y_432_, v___y_433_, v___y_434_);
lean_dec(v___y_434_);
lean_dec_ref(v___y_433_);
lean_dec(v___y_432_);
lean_dec(v___y_430_);
lean_dec_ref(v___y_429_);
lean_dec(v___y_428_);
lean_dec_ref(v___y_427_);
lean_dec_ref(v_a_425_);
return v_res_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6___redArg(lean_object* v_as_438_, size_t v_sz_439_, size_t v_i_440_, lean_object* v_b_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_){
_start:
{
uint8_t v___x_449_; 
v___x_449_ = lean_usize_dec_lt(v_i_440_, v_sz_439_);
if (v___x_449_ == 0)
{
lean_object* v___x_450_; 
v___x_450_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_450_, 0, v_b_441_);
return v___x_450_;
}
else
{
lean_object* v_array_451_; lean_object* v_start_452_; lean_object* v_stop_453_; uint8_t v___x_454_; 
v_array_451_ = lean_ctor_get(v_b_441_, 0);
v_start_452_ = lean_ctor_get(v_b_441_, 1);
v_stop_453_ = lean_ctor_get(v_b_441_, 2);
v___x_454_ = lean_nat_dec_lt(v_start_452_, v_stop_453_);
if (v___x_454_ == 0)
{
lean_object* v___x_455_; 
v___x_455_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_455_, 0, v_b_441_);
return v___x_455_;
}
else
{
lean_object* v___x_457_; uint8_t v_isShared_458_; uint8_t v_isSharedCheck_482_; 
lean_inc(v_stop_453_);
lean_inc(v_start_452_);
lean_inc_ref(v_array_451_);
v_isSharedCheck_482_ = !lean_is_exclusive(v_b_441_);
if (v_isSharedCheck_482_ == 0)
{
lean_object* v_unused_483_; lean_object* v_unused_484_; lean_object* v_unused_485_; 
v_unused_483_ = lean_ctor_get(v_b_441_, 2);
lean_dec(v_unused_483_);
v_unused_484_ = lean_ctor_get(v_b_441_, 1);
lean_dec(v_unused_484_);
v_unused_485_ = lean_ctor_get(v_b_441_, 0);
lean_dec(v_unused_485_);
v___x_457_ = v_b_441_;
v_isShared_458_ = v_isSharedCheck_482_;
goto v_resetjp_456_;
}
else
{
lean_dec(v_b_441_);
v___x_457_ = lean_box(0);
v_isShared_458_ = v_isSharedCheck_482_;
goto v_resetjp_456_;
}
v_resetjp_456_:
{
lean_object* v_a_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; uint8_t v___x_464_; lean_object* v___x_465_; 
v_a_459_ = lean_array_uget_borrowed(v_as_438_, v_i_440_);
v___x_460_ = lean_array_fget_borrowed(v_array_451_, v_start_452_);
lean_inc(v_a_459_);
v___x_461_ = l_Lean_mkFVar(v_a_459_);
v___x_462_ = lean_box(0);
v___x_463_ = lean_box(0);
v___x_464_ = 0;
lean_inc(v___x_460_);
v___x_465_ = l_Lean_Elab_Term_addTermInfo_x27(v___x_460_, v___x_461_, v___x_462_, v___x_462_, v___x_463_, v___x_464_, v___x_464_, v___y_442_, v___y_443_, v___y_444_, v___y_445_, v___y_446_, v___y_447_);
if (lean_obj_tag(v___x_465_) == 0)
{
lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_469_; 
lean_dec_ref_known(v___x_465_, 1);
v___x_466_ = lean_unsigned_to_nat(1u);
v___x_467_ = lean_nat_add(v_start_452_, v___x_466_);
lean_dec(v_start_452_);
if (v_isShared_458_ == 0)
{
lean_ctor_set(v___x_457_, 1, v___x_467_);
v___x_469_ = v___x_457_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_473_; 
v_reuseFailAlloc_473_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_473_, 0, v_array_451_);
lean_ctor_set(v_reuseFailAlloc_473_, 1, v___x_467_);
lean_ctor_set(v_reuseFailAlloc_473_, 2, v_stop_453_);
v___x_469_ = v_reuseFailAlloc_473_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
size_t v___x_470_; size_t v___x_471_; 
v___x_470_ = ((size_t)1ULL);
v___x_471_ = lean_usize_add(v_i_440_, v___x_470_);
v_i_440_ = v___x_471_;
v_b_441_ = v___x_469_;
goto _start;
}
}
else
{
lean_object* v_a_474_; lean_object* v___x_476_; uint8_t v_isShared_477_; uint8_t v_isSharedCheck_481_; 
lean_del_object(v___x_457_);
lean_dec(v_stop_453_);
lean_dec(v_start_452_);
lean_dec_ref(v_array_451_);
v_a_474_ = lean_ctor_get(v___x_465_, 0);
v_isSharedCheck_481_ = !lean_is_exclusive(v___x_465_);
if (v_isSharedCheck_481_ == 0)
{
v___x_476_ = v___x_465_;
v_isShared_477_ = v_isSharedCheck_481_;
goto v_resetjp_475_;
}
else
{
lean_inc(v_a_474_);
lean_dec(v___x_465_);
v___x_476_ = lean_box(0);
v_isShared_477_ = v_isSharedCheck_481_;
goto v_resetjp_475_;
}
v_resetjp_475_:
{
lean_object* v___x_479_; 
if (v_isShared_477_ == 0)
{
v___x_479_ = v___x_476_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v_a_474_);
v___x_479_ = v_reuseFailAlloc_480_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
return v___x_479_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6___redArg___boxed(lean_object* v_as_486_, lean_object* v_sz_487_, lean_object* v_i_488_, lean_object* v_b_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_){
_start:
{
size_t v_sz_boxed_497_; size_t v_i_boxed_498_; lean_object* v_res_499_; 
v_sz_boxed_497_ = lean_unbox_usize(v_sz_487_);
lean_dec(v_sz_487_);
v_i_boxed_498_ = lean_unbox_usize(v_i_488_);
lean_dec(v_i_488_);
v_res_499_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6___redArg(v_as_486_, v_sz_boxed_497_, v_i_boxed_498_, v_b_489_, v___y_490_, v___y_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_);
lean_dec(v___y_495_);
lean_dec_ref(v___y_494_);
lean_dec(v___y_493_);
lean_dec_ref(v___y_492_);
lean_dec(v___y_491_);
lean_dec_ref(v___y_490_);
lean_dec_ref(v_as_486_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__1(lean_object* v_a_500_, size_t v_sz_501_, size_t v___x_502_, lean_object* v___x_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_){
_start:
{
lean_object* v___x_513_; 
v___x_513_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6___redArg(v_a_500_, v_sz_501_, v___x_502_, v___x_503_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_);
if (lean_obj_tag(v___x_513_) == 0)
{
lean_object* v___x_515_; uint8_t v_isShared_516_; uint8_t v_isSharedCheck_521_; 
v_isSharedCheck_521_ = !lean_is_exclusive(v___x_513_);
if (v_isSharedCheck_521_ == 0)
{
lean_object* v_unused_522_; 
v_unused_522_ = lean_ctor_get(v___x_513_, 0);
lean_dec(v_unused_522_);
v___x_515_ = v___x_513_;
v_isShared_516_ = v_isSharedCheck_521_;
goto v_resetjp_514_;
}
else
{
lean_dec(v___x_513_);
v___x_515_ = lean_box(0);
v_isShared_516_ = v_isSharedCheck_521_;
goto v_resetjp_514_;
}
v_resetjp_514_:
{
lean_object* v___x_517_; lean_object* v___x_519_; 
v___x_517_ = lean_box(0);
if (v_isShared_516_ == 0)
{
lean_ctor_set(v___x_515_, 0, v___x_517_);
v___x_519_ = v___x_515_;
goto v_reusejp_518_;
}
else
{
lean_object* v_reuseFailAlloc_520_; 
v_reuseFailAlloc_520_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_520_, 0, v___x_517_);
v___x_519_ = v_reuseFailAlloc_520_;
goto v_reusejp_518_;
}
v_reusejp_518_:
{
return v___x_519_;
}
}
}
else
{
lean_object* v_a_523_; lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_530_; 
v_a_523_ = lean_ctor_get(v___x_513_, 0);
v_isSharedCheck_530_ = !lean_is_exclusive(v___x_513_);
if (v_isSharedCheck_530_ == 0)
{
v___x_525_ = v___x_513_;
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
else
{
lean_inc(v_a_523_);
lean_dec(v___x_513_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v___x_528_; 
if (v_isShared_526_ == 0)
{
v___x_528_ = v___x_525_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v_a_523_);
v___x_528_ = v_reuseFailAlloc_529_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
return v___x_528_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__1___boxed(lean_object* v_a_531_, lean_object* v_sz_532_, lean_object* v___x_533_, lean_object* v___x_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_){
_start:
{
size_t v_sz_boxed_544_; size_t v___x_7144__boxed_545_; lean_object* v_res_546_; 
v_sz_boxed_544_ = lean_unbox_usize(v_sz_532_);
lean_dec(v_sz_532_);
v___x_7144__boxed_545_ = lean_unbox_usize(v___x_533_);
lean_dec(v___x_533_);
v_res_546_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__1(v_a_531_, v_sz_boxed_544_, v___x_7144__boxed_545_, v___x_534_, v___y_535_, v___y_536_, v___y_537_, v___y_538_, v___y_539_, v___y_540_, v___y_541_, v___y_542_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
lean_dec(v___y_540_);
lean_dec_ref(v___y_539_);
lean_dec(v___y_538_);
lean_dec_ref(v___y_537_);
lean_dec(v___y_536_);
lean_dec_ref(v___y_535_);
lean_dec_ref(v_a_531_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__2(size_t v_sz_547_, size_t v_i_548_, lean_object* v_bs_549_){
_start:
{
uint8_t v___x_550_; 
v___x_550_ = lean_usize_dec_lt(v_i_548_, v_sz_547_);
if (v___x_550_ == 0)
{
return v_bs_549_;
}
else
{
lean_object* v_v_551_; lean_object* v_snd_552_; lean_object* v___x_553_; lean_object* v_bs_x27_554_; size_t v___x_555_; size_t v___x_556_; lean_object* v___x_557_; 
v_v_551_ = lean_array_uget_borrowed(v_bs_549_, v_i_548_);
v_snd_552_ = lean_ctor_get(v_v_551_, 1);
lean_inc(v_snd_552_);
v___x_553_ = lean_unsigned_to_nat(0u);
v_bs_x27_554_ = lean_array_uset(v_bs_549_, v_i_548_, v___x_553_);
v___x_555_ = ((size_t)1ULL);
v___x_556_ = lean_usize_add(v_i_548_, v___x_555_);
v___x_557_ = lean_array_uset(v_bs_x27_554_, v_i_548_, v_snd_552_);
v_i_548_ = v___x_556_;
v_bs_549_ = v___x_557_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__2___boxed(lean_object* v_sz_559_, lean_object* v_i_560_, lean_object* v_bs_561_){
_start:
{
size_t v_sz_boxed_562_; size_t v_i_boxed_563_; lean_object* v_res_564_; 
v_sz_boxed_562_ = lean_unbox_usize(v_sz_559_);
lean_dec(v_sz_559_);
v_i_boxed_563_ = lean_unbox_usize(v_i_560_);
lean_dec(v_i_560_);
v_res_564_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__2(v_sz_boxed_562_, v_i_boxed_563_, v_bs_561_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__1(size_t v_sz_565_, size_t v_i_566_, lean_object* v_bs_567_){
_start:
{
uint8_t v___x_568_; 
v___x_568_ = lean_usize_dec_lt(v_i_566_, v_sz_565_);
if (v___x_568_ == 0)
{
lean_object* v___x_569_; 
v___x_569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_569_, 0, v_bs_567_);
return v___x_569_;
}
else
{
lean_object* v_v_570_; lean_object* v___x_571_; uint8_t v___x_572_; 
v_v_570_ = lean_array_uget(v_bs_567_, v_i_566_);
v___x_571_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_renameArg___closed__3));
lean_inc(v_v_570_);
v___x_572_ = l_Lean_Syntax_isOfKind(v_v_570_, v___x_571_);
if (v___x_572_ == 0)
{
lean_object* v___x_573_; 
lean_dec(v_v_570_);
lean_dec_ref(v_bs_567_);
v___x_573_ = lean_box(0);
return v___x_573_;
}
else
{
lean_object* v___x_574_; lean_object* v_bs_575_; lean_object* v___x_576_; uint8_t v___x_577_; 
v___x_574_ = lean_unsigned_to_nat(2u);
v_bs_575_ = l_Lean_Syntax_getArg(v_v_570_, v___x_574_);
v___x_576_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_renameArg___closed__13));
lean_inc(v_bs_575_);
v___x_577_ = l_Lean_Syntax_isOfKind(v_bs_575_, v___x_576_);
if (v___x_577_ == 0)
{
lean_object* v___x_578_; 
lean_dec(v_bs_575_);
lean_dec(v_v_570_);
lean_dec_ref(v_bs_567_);
v___x_578_ = lean_box(0);
return v___x_578_;
}
else
{
lean_object* v___x_579_; lean_object* v_bs_x27_580_; lean_object* v_as_581_; lean_object* v___x_582_; size_t v___x_583_; size_t v___x_584_; lean_object* v___x_585_; 
v___x_579_ = lean_unsigned_to_nat(0u);
v_bs_x27_580_ = lean_array_uset(v_bs_567_, v_i_566_, v___x_579_);
v_as_581_ = l_Lean_Syntax_getArg(v_v_570_, v___x_579_);
lean_dec(v_v_570_);
v___x_582_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_582_, 0, v_as_581_);
lean_ctor_set(v___x_582_, 1, v_bs_575_);
v___x_583_ = ((size_t)1ULL);
v___x_584_ = lean_usize_add(v_i_566_, v___x_583_);
v___x_585_ = lean_array_uset(v_bs_x27_580_, v_i_566_, v___x_582_);
v_i_566_ = v___x_584_;
v_bs_567_ = v___x_585_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__1___boxed(lean_object* v_sz_587_, lean_object* v_i_588_, lean_object* v_bs_589_){
_start:
{
size_t v_sz_boxed_590_; size_t v_i_boxed_591_; lean_object* v_res_592_; 
v_sz_boxed_590_ = lean_unbox_usize(v_sz_587_);
lean_dec(v_sz_587_);
v_i_boxed_591_ = lean_unbox_usize(v_i_588_);
lean_dec(v_i_588_);
v_res_592_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__1(v_sz_boxed_590_, v_i_boxed_591_, v_bs_589_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__7(uint8_t v___x_593_, lean_object* v_as_594_, size_t v_i_595_, size_t v_stop_596_, lean_object* v_b_597_){
_start:
{
lean_object* v___y_599_; uint8_t v___x_603_; 
v___x_603_ = lean_usize_dec_eq(v_i_595_, v_stop_596_);
if (v___x_603_ == 0)
{
lean_object* v_fst_604_; uint8_t v___x_605_; 
v_fst_604_ = lean_ctor_get(v_b_597_, 0);
v___x_605_ = lean_unbox(v_fst_604_);
if (v___x_605_ == 0)
{
lean_object* v_snd_606_; lean_object* v___x_608_; uint8_t v_isShared_609_; uint8_t v_isSharedCheck_614_; 
v_snd_606_ = lean_ctor_get(v_b_597_, 1);
v_isSharedCheck_614_ = !lean_is_exclusive(v_b_597_);
if (v_isSharedCheck_614_ == 0)
{
lean_object* v_unused_615_; 
v_unused_615_ = lean_ctor_get(v_b_597_, 0);
lean_dec(v_unused_615_);
v___x_608_ = v_b_597_;
v_isShared_609_ = v_isSharedCheck_614_;
goto v_resetjp_607_;
}
else
{
lean_inc(v_snd_606_);
lean_dec(v_b_597_);
v___x_608_ = lean_box(0);
v_isShared_609_ = v_isSharedCheck_614_;
goto v_resetjp_607_;
}
v_resetjp_607_:
{
lean_object* v___x_610_; lean_object* v___x_612_; 
v___x_610_ = lean_box(v___x_593_);
if (v_isShared_609_ == 0)
{
lean_ctor_set(v___x_608_, 0, v___x_610_);
v___x_612_ = v___x_608_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v___x_610_);
lean_ctor_set(v_reuseFailAlloc_613_, 1, v_snd_606_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
v___y_599_ = v___x_612_;
goto v___jp_598_;
}
}
}
else
{
lean_object* v_snd_616_; lean_object* v___x_618_; uint8_t v_isShared_619_; uint8_t v_isSharedCheck_626_; 
v_snd_616_ = lean_ctor_get(v_b_597_, 1);
v_isSharedCheck_626_ = !lean_is_exclusive(v_b_597_);
if (v_isSharedCheck_626_ == 0)
{
lean_object* v_unused_627_; 
v_unused_627_ = lean_ctor_get(v_b_597_, 0);
lean_dec(v_unused_627_);
v___x_618_ = v_b_597_;
v_isShared_619_ = v_isSharedCheck_626_;
goto v_resetjp_617_;
}
else
{
lean_inc(v_snd_616_);
lean_dec(v_b_597_);
v___x_618_ = lean_box(0);
v_isShared_619_ = v_isSharedCheck_626_;
goto v_resetjp_617_;
}
v_resetjp_617_:
{
lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_624_; 
v___x_620_ = lean_array_uget_borrowed(v_as_594_, v_i_595_);
lean_inc(v___x_620_);
v___x_621_ = lean_array_push(v_snd_616_, v___x_620_);
v___x_622_ = lean_box(v___x_603_);
if (v_isShared_619_ == 0)
{
lean_ctor_set(v___x_618_, 1, v___x_621_);
lean_ctor_set(v___x_618_, 0, v___x_622_);
v___x_624_ = v___x_618_;
goto v_reusejp_623_;
}
else
{
lean_object* v_reuseFailAlloc_625_; 
v_reuseFailAlloc_625_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_625_, 0, v___x_622_);
lean_ctor_set(v_reuseFailAlloc_625_, 1, v___x_621_);
v___x_624_ = v_reuseFailAlloc_625_;
goto v_reusejp_623_;
}
v_reusejp_623_:
{
v___y_599_ = v___x_624_;
goto v___jp_598_;
}
}
}
}
else
{
return v_b_597_;
}
v___jp_598_:
{
size_t v___x_600_; size_t v___x_601_; 
v___x_600_ = ((size_t)1ULL);
v___x_601_ = lean_usize_add(v_i_595_, v___x_600_);
v_i_595_ = v___x_601_;
v_b_597_ = v___y_599_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__7___boxed(lean_object* v___x_628_, lean_object* v_as_629_, lean_object* v_i_630_, lean_object* v_stop_631_, lean_object* v_b_632_){
_start:
{
uint8_t v___x_7275__boxed_633_; size_t v_i_boxed_634_; size_t v_stop_boxed_635_; lean_object* v_res_636_; 
v___x_7275__boxed_633_ = lean_unbox(v___x_628_);
v_i_boxed_634_ = lean_unbox_usize(v_i_630_);
lean_dec(v_i_630_);
v_stop_boxed_635_ = lean_unbox_usize(v_stop_631_);
lean_dec(v_stop_631_);
v_res_636_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__7(v___x_7275__boxed_633_, v_as_629_, v_i_boxed_634_, v_stop_boxed_635_, v_b_632_);
lean_dec_ref(v_as_629_);
return v_res_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__3(size_t v_sz_637_, size_t v_i_638_, lean_object* v_bs_639_){
_start:
{
uint8_t v___x_640_; 
v___x_640_ = lean_usize_dec_lt(v_i_638_, v_sz_637_);
if (v___x_640_ == 0)
{
return v_bs_639_;
}
else
{
lean_object* v_v_641_; lean_object* v_fst_642_; lean_object* v___x_643_; lean_object* v_bs_x27_644_; size_t v___x_645_; size_t v___x_646_; lean_object* v___x_647_; 
v_v_641_ = lean_array_uget_borrowed(v_bs_639_, v_i_638_);
v_fst_642_ = lean_ctor_get(v_v_641_, 0);
lean_inc(v_fst_642_);
v___x_643_ = lean_unsigned_to_nat(0u);
v_bs_x27_644_ = lean_array_uset(v_bs_639_, v_i_638_, v___x_643_);
v___x_645_ = ((size_t)1ULL);
v___x_646_ = lean_usize_add(v_i_638_, v___x_645_);
v___x_647_ = lean_array_uset(v_bs_x27_644_, v_i_638_, v_fst_642_);
v_i_638_ = v___x_646_;
v_bs_639_ = v___x_647_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__3___boxed(lean_object* v_sz_649_, lean_object* v_i_650_, lean_object* v_bs_651_){
_start:
{
size_t v_sz_boxed_652_; size_t v_i_boxed_653_; lean_object* v_res_654_; 
v_sz_boxed_652_ = lean_unbox_usize(v_sz_649_);
lean_dec(v_sz_649_);
v_i_boxed_653_ = lean_unbox_usize(v_i_650_);
lean_dec(v_i_650_);
v_res_654_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__3(v_sz_boxed_652_, v_i_boxed_653_, v_bs_651_);
return v_res_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1(lean_object* v_x_659_, lean_object* v_a_660_, lean_object* v_a_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_, lean_object* v_a_666_, lean_object* v_a_667_){
_start:
{
lean_object* v___x_669_; uint8_t v___x_670_; 
v___x_669_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rename_x27___closed__1));
lean_inc(v_x_659_);
v___x_670_ = l_Lean_Syntax_isOfKind(v_x_659_, v___x_669_);
if (v___x_670_ == 0)
{
lean_object* v___x_671_; 
lean_dec(v_x_659_);
v___x_671_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg();
return v___x_671_;
}
else
{
lean_object* v___x_672_; lean_object* v___y_674_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; uint8_t v___x_708_; 
v___x_672_ = lean_unsigned_to_nat(0u);
v___x_703_ = lean_unsigned_to_nat(1u);
v___x_704_ = l_Lean_Syntax_getArg(v_x_659_, v___x_703_);
lean_dec(v_x_659_);
v___x_705_ = l_Lean_Syntax_getArgs(v___x_704_);
lean_dec(v___x_704_);
v___x_706_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___closed__0));
v___x_707_ = lean_array_get_size(v___x_705_);
v___x_708_ = lean_nat_dec_lt(v___x_672_, v___x_707_);
if (v___x_708_ == 0)
{
lean_dec_ref(v___x_705_);
v___y_674_ = v___x_706_;
goto v___jp_673_;
}
else
{
lean_object* v___x_709_; lean_object* v___x_710_; uint8_t v___x_711_; 
v___x_709_ = lean_box(v___x_670_);
v___x_710_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_710_, 0, v___x_709_);
lean_ctor_set(v___x_710_, 1, v___x_706_);
v___x_711_ = lean_nat_dec_le(v___x_707_, v___x_707_);
if (v___x_711_ == 0)
{
if (v___x_708_ == 0)
{
lean_dec_ref_known(v___x_710_, 2);
lean_dec_ref(v___x_705_);
v___y_674_ = v___x_706_;
goto v___jp_673_;
}
else
{
size_t v___x_712_; size_t v___x_713_; lean_object* v___x_714_; lean_object* v_snd_715_; 
v___x_712_ = ((size_t)0ULL);
v___x_713_ = lean_usize_of_nat(v___x_707_);
v___x_714_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__7(v___x_670_, v___x_705_, v___x_712_, v___x_713_, v___x_710_);
lean_dec_ref(v___x_705_);
v_snd_715_ = lean_ctor_get(v___x_714_, 1);
lean_inc(v_snd_715_);
lean_dec_ref(v___x_714_);
v___y_674_ = v_snd_715_;
goto v___jp_673_;
}
}
else
{
size_t v___x_716_; size_t v___x_717_; lean_object* v___x_718_; lean_object* v_snd_719_; 
v___x_716_ = ((size_t)0ULL);
v___x_717_ = lean_usize_of_nat(v___x_707_);
v___x_718_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__7(v___x_670_, v___x_705_, v___x_716_, v___x_717_, v___x_710_);
lean_dec_ref(v___x_705_);
v_snd_719_ = lean_ctor_get(v___x_718_, 1);
lean_inc(v_snd_719_);
lean_dec_ref(v___x_718_);
v___y_674_ = v_snd_719_;
goto v___jp_673_;
}
}
v___jp_673_:
{
size_t v_sz_675_; size_t v___x_676_; lean_object* v___x_677_; 
v_sz_675_ = lean_array_size(v___y_674_);
v___x_676_ = ((size_t)0ULL);
v___x_677_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__1(v_sz_675_, v___x_676_, v___y_674_);
if (lean_obj_tag(v___x_677_) == 0)
{
lean_object* v___x_678_; 
v___x_678_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__0___redArg();
return v___x_678_;
}
else
{
lean_object* v_val_679_; size_t v_sz_680_; lean_object* v_bs_681_; lean_object* v_as_682_; lean_object* v___x_683_; 
v_val_679_ = lean_ctor_get(v___x_677_, 0);
lean_inc_n(v_val_679_, 2);
lean_dec_ref_known(v___x_677_, 1);
v_sz_680_ = lean_array_size(v_val_679_);
v_bs_681_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__2(v_sz_680_, v___x_676_, v_val_679_);
v_as_682_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__3(v_sz_680_, v___x_676_, v_val_679_);
v___x_683_ = l_Lean_Elab_Tactic_getFVarIds(v_as_682_, v_a_660_, v_a_661_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
if (lean_obj_tag(v___x_683_) == 0)
{
lean_object* v_a_684_; lean_object* v___x_685_; lean_object* v___f_686_; lean_object* v___x_687_; 
v_a_684_ = lean_ctor_get(v___x_683_, 0);
lean_inc_n(v_a_684_, 2);
lean_dec_ref_known(v___x_683_, 1);
v___x_685_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___boxed__const__1));
lean_inc_ref(v_bs_681_);
v___f_686_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__0___boxed), 13, 4);
lean_closure_set(v___f_686_, 0, v_bs_681_);
lean_closure_set(v___f_686_, 1, v___x_672_);
lean_closure_set(v___f_686_, 2, v_a_684_);
lean_closure_set(v___f_686_, 3, v___x_685_);
v___x_687_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_686_, v_a_660_, v_a_661_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
if (lean_obj_tag(v___x_687_) == 0)
{
lean_object* v___x_688_; lean_object* v___x_689_; size_t v_sz_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___f_693_; lean_object* v___x_694_; 
lean_dec_ref_known(v___x_687_, 1);
v___x_688_ = lean_array_get_size(v_bs_681_);
v___x_689_ = l_Array_toSubarray___redArg(v_bs_681_, v___x_672_, v___x_688_);
v_sz_690_ = lean_array_size(v_a_684_);
v___x_691_ = lean_box_usize(v_sz_690_);
v___x_692_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___boxed__const__1));
v___f_693_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___lam__1___boxed), 13, 4);
lean_closure_set(v___f_693_, 0, v_a_684_);
lean_closure_set(v___f_693_, 1, v___x_691_);
lean_closure_set(v___f_693_, 2, v___x_692_);
lean_closure_set(v___f_693_, 3, v___x_689_);
v___x_694_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_693_, v_a_660_, v_a_661_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
return v___x_694_;
}
else
{
lean_dec(v_a_684_);
lean_dec_ref(v_bs_681_);
return v___x_687_;
}
}
else
{
lean_object* v_a_695_; lean_object* v___x_697_; uint8_t v_isShared_698_; uint8_t v_isSharedCheck_702_; 
lean_dec_ref(v_bs_681_);
v_a_695_ = lean_ctor_get(v___x_683_, 0);
v_isSharedCheck_702_ = !lean_is_exclusive(v___x_683_);
if (v_isSharedCheck_702_ == 0)
{
v___x_697_ = v___x_683_;
v_isShared_698_ = v_isSharedCheck_702_;
goto v_resetjp_696_;
}
else
{
lean_inc(v_a_695_);
lean_dec(v___x_683_);
v___x_697_ = lean_box(0);
v_isShared_698_ = v_isSharedCheck_702_;
goto v_resetjp_696_;
}
v_resetjp_696_:
{
lean_object* v___x_700_; 
if (v_isShared_698_ == 0)
{
v___x_700_ = v___x_697_;
goto v_reusejp_699_;
}
else
{
lean_object* v_reuseFailAlloc_701_; 
v_reuseFailAlloc_701_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_701_, 0, v_a_695_);
v___x_700_ = v_reuseFailAlloc_701_;
goto v_reusejp_699_;
}
v_reusejp_699_:
{
return v___x_700_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1___boxed(lean_object* v_x_720_, lean_object* v_a_721_, lean_object* v_a_722_, lean_object* v_a_723_, lean_object* v_a_724_, lean_object* v_a_725_, lean_object* v_a_726_, lean_object* v_a_727_, lean_object* v_a_728_, lean_object* v_a_729_){
_start:
{
lean_object* v_res_730_; 
v_res_730_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1(v_x_720_, v_a_721_, v_a_722_, v_a_723_, v_a_724_, v_a_725_, v_a_726_, v_a_727_, v_a_728_);
lean_dec(v_a_728_);
lean_dec_ref(v_a_727_);
lean_dec(v_a_726_);
lean_dec_ref(v_a_725_);
lean_dec(v_a_724_);
lean_dec_ref(v_a_723_);
lean_dec(v_a_722_);
lean_dec_ref(v_a_721_);
return v_res_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4(lean_object* v_as_731_, size_t v_sz_732_, size_t v_i_733_, lean_object* v_b_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
lean_object* v___x_740_; 
v___x_740_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4___redArg(v_as_731_, v_sz_732_, v_i_733_, v_b_734_);
return v___x_740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4___boxed(lean_object* v_as_741_, lean_object* v_sz_742_, lean_object* v_i_743_, lean_object* v_b_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_){
_start:
{
size_t v_sz_boxed_750_; size_t v_i_boxed_751_; lean_object* v_res_752_; 
v_sz_boxed_750_ = lean_unbox_usize(v_sz_742_);
lean_dec(v_sz_742_);
v_i_boxed_751_ = lean_unbox_usize(v_i_743_);
lean_dec(v_i_743_);
v_res_752_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__4(v_as_741_, v_sz_boxed_750_, v_i_boxed_751_, v_b_744_, v___y_745_, v___y_746_, v___y_747_, v___y_748_);
lean_dec(v___y_748_);
lean_dec_ref(v___y_747_);
lean_dec(v___y_746_);
lean_dec_ref(v___y_745_);
lean_dec_ref(v_as_741_);
return v_res_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5(lean_object* v_mvarId_753_, lean_object* v_val_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_){
_start:
{
lean_object* v___x_760_; 
v___x_760_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5___redArg(v_mvarId_753_, v_val_754_, v___y_756_);
return v___x_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5___boxed(lean_object* v_mvarId_761_, lean_object* v_val_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_){
_start:
{
lean_object* v_res_768_; 
v_res_768_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5(v_mvarId_761_, v_val_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_);
lean_dec(v___y_766_);
lean_dec_ref(v___y_765_);
lean_dec(v___y_764_);
lean_dec_ref(v___y_763_);
return v_res_768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6(lean_object* v_as_769_, size_t v_sz_770_, size_t v_i_771_, lean_object* v_b_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_){
_start:
{
lean_object* v___x_782_; 
v___x_782_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6___redArg(v_as_769_, v_sz_770_, v_i_771_, v_b_772_, v___y_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_, v___y_780_);
return v___x_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6___boxed(lean_object* v_as_783_, lean_object* v_sz_784_, lean_object* v_i_785_, lean_object* v_b_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_){
_start:
{
size_t v_sz_boxed_796_; size_t v_i_boxed_797_; lean_object* v_res_798_; 
v_sz_boxed_796_ = lean_unbox_usize(v_sz_784_);
lean_dec(v_sz_784_);
v_i_boxed_797_ = lean_unbox_usize(v_i_785_);
lean_dec(v_i_785_);
v_res_798_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__6(v_as_783_, v_sz_boxed_796_, v_i_boxed_797_, v_b_786_, v___y_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
lean_dec(v___y_794_);
lean_dec_ref(v___y_793_);
lean_dec(v___y_792_);
lean_dec_ref(v___y_791_);
lean_dec(v___y_790_);
lean_dec_ref(v___y_789_);
lean_dec(v___y_788_);
lean_dec_ref(v___y_787_);
lean_dec_ref(v_as_783_);
return v_res_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5(lean_object* v_00_u03b2_799_, lean_object* v_x_800_, lean_object* v_x_801_, lean_object* v_x_802_){
_start:
{
lean_object* v___x_803_; 
v___x_803_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5___redArg(v_x_800_, v_x_801_, v_x_802_);
return v___x_803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6(lean_object* v_00_u03b2_804_, lean_object* v_x_805_, size_t v_x_806_, size_t v_x_807_, lean_object* v_x_808_, lean_object* v_x_809_){
_start:
{
lean_object* v___x_810_; 
v___x_810_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___redArg(v_x_805_, v_x_806_, v_x_807_, v_x_808_, v_x_809_);
return v___x_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6___boxed(lean_object* v_00_u03b2_811_, lean_object* v_x_812_, lean_object* v_x_813_, lean_object* v_x_814_, lean_object* v_x_815_, lean_object* v_x_816_){
_start:
{
size_t v_x_7540__boxed_817_; size_t v_x_7541__boxed_818_; lean_object* v_res_819_; 
v_x_7540__boxed_817_ = lean_unbox_usize(v_x_813_);
lean_dec(v_x_813_);
v_x_7541__boxed_818_ = lean_unbox_usize(v_x_814_);
lean_dec(v_x_814_);
v_res_819_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6(v_00_u03b2_811_, v_x_812_, v_x_7540__boxed_817_, v_x_7541__boxed_818_, v_x_815_, v_x_816_);
return v_res_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9(lean_object* v_00_u03b2_820_, lean_object* v_n_821_, lean_object* v_k_822_, lean_object* v_v_823_){
_start:
{
lean_object* v___x_824_; 
v___x_824_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9___redArg(v_n_821_, v_k_822_, v_v_823_);
return v___x_824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10(lean_object* v_00_u03b2_825_, size_t v_depth_826_, lean_object* v_keys_827_, lean_object* v_vals_828_, lean_object* v_heq_829_, lean_object* v_i_830_, lean_object* v_entries_831_){
_start:
{
lean_object* v___x_832_; 
v___x_832_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10___redArg(v_depth_826_, v_keys_827_, v_vals_828_, v_i_830_, v_entries_831_);
return v___x_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10___boxed(lean_object* v_00_u03b2_833_, lean_object* v_depth_834_, lean_object* v_keys_835_, lean_object* v_vals_836_, lean_object* v_heq_837_, lean_object* v_i_838_, lean_object* v_entries_839_){
_start:
{
size_t v_depth_boxed_840_; lean_object* v_res_841_; 
v_depth_boxed_840_ = lean_unbox_usize(v_depth_834_);
lean_dec(v_depth_834_);
v_res_841_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__10(v_00_u03b2_833_, v_depth_boxed_840_, v_keys_835_, v_vals_836_, v_heq_837_, v_i_838_, v_entries_839_);
lean_dec_ref(v_vals_836_);
lean_dec_ref(v_keys_835_);
return v_res_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9_spec__10(lean_object* v_00_u03b2_842_, lean_object* v_x_843_, lean_object* v_x_844_, lean_object* v_x_845_, lean_object* v_x_846_){
_start:
{
lean_object* v___x_847_; 
v___x_847_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Rename______elabRules__Mathlib__Tactic__rename_x27__1_spec__5_spec__5_spec__6_spec__9_spec__10___redArg(v_x_843_, v_x_844_, v_x_845_, v_x_846_);
return v___x_847_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Rename(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Rename(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Rename(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Rename(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Rename(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Rename(builtin);
}
#ifdef __cplusplus
}
#endif
