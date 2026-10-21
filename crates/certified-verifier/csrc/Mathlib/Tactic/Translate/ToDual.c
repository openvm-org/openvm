// Lean compiler output
// Module: Mathlib.Tactic.Translate.ToDual
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Translate.TagUnfoldBoundary
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
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_Translate_attrArgs;
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_batteries_Lean_registerNameMapExtension___redArg(lean_object*);
lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_string_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabTranslationAttr(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt();
lean_object* lean_array_to_list(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Translate_addTranslationAttr(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_profileitIOUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabInsertCastFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabInsertCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ToDual"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "to_dual_ignore_args"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__3_value),LEAN_SCALAR_PTR_LITERAL(132, 222, 235, 135, 106, 122, 79, 145)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__8_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__10_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__13_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "to_dual_do_translate"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(220, 53, 153, 229, 53, 125, 121, 7)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "to_dual_dont_translate"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(6, 14, 173, 103, 48, 163, 138, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "to_dual"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 229, 230, 230, 155, 148, 191, 66)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__3_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual_to__dual;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "attrTo_dual\?_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(92, 220, 140, 109, 234, 199, 88, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "to_dual\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f__;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__3_value),LEAN_SCALAR_PTR_LITERAL(153, 16, 234, 224, 21, 208, 38, 60)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2____boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__3_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ignoreArgsAttr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(148, 252, 212, 200, 21, 153, 204, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 83, .m_capacity = 83, .m_length = 82, .m_data = "Auxiliary attribute for `to_dual` stating that certain arguments are not dualized."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual_ignoreArgsAttr;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_678359338____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_678359338____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual_unfoldBoundaries;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "doTranslateAttr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(83, 228, 12, 173, 113, 31, 115, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual_doTranslateAttr;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Already exists entry for "};
static const lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Translate"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(104, 225, 249, 99, 81, 122, 117, 142)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(133, 224, 52, 50, 80, 88, 16, 129)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(80, 214, 157, 45, 131, 58, 253, 35)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__9_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 113, 18, 179, 130, 193, 162, 240)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__9_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__9_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__10_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__9_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(48, 200, 53, 155, 253, 17, 102, 1)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__10_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__10_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__11_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__10_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(29, 28, 251, 51, 234, 82, 25, 124)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__11_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__11_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__12_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__12_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__12_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__13_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__11_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__12_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(148, 161, 218, 64, 202, 228, 205, 44)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__13_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__13_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__14_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__14_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__14_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__15_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__13_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__14_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(189, 8, 59, 21, 117, 232, 202, 4)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__15_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__15_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__16_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__15_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(248, 53, 29, 114, 49, 245, 138, 19)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__16_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__16_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__17_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__16_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(229, 45, 34, 7, 103, 131, 56, 84)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__17_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__17_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__18_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__17_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(250, 184, 58, 120, 85, 86, 11, 4)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__18_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__18_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__19_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__18_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(223, 151, 25, 22, 47, 57, 245, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__19_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__19_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__21_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__21_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__21_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__23_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__23_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__23_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__26_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__do__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(241, 199, 148, 212, 28, 230, 163, 126)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__26_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__26_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__27_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__26_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__27_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__27_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__28_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 97, .m_capacity = 97, .m_length = 96, .m_data = "Auxiliary attribute for `to_dual` stating that the operations on this type should be translated."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__28_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__28_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__31_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__31_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__31_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__32_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__dont__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(27, 68, 8, 109, 0, 121, 219, 86)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__32_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__32_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__33_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__32_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__33_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__33_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__34_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 101, .m_capacity = 101, .m_length = 100, .m_data = "Auxiliary attribute for `to_dual` stating that the operations on this type should not be translated."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__34_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__34_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "translations"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(38, 134, 81, 40, 107, 82, 88, 203)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual_translations;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "comonad"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Monad"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "monadic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Comonadic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "comonadic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Monadic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "section"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Retraction"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "retraction"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Section"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 8, .m_data = "functorπ"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 8, .m_data = "Functorι"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__20_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 8, .m_data = "functorι"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 8, .m_data = "Functorπ"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__25_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__24_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__27_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__23_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__19_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__33_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cokernel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Kernel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__36_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__35_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__37_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "kernels"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Cokernels"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__40_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__39_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__41_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "cokernels"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__43_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Kernels"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__44_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__43_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__45_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__46_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "unit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Counit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__48_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__48_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__47_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__49_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__50_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "counit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__51_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Unit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__52_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__52_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__53_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__51_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__53_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__54_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "monad"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__55_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Comonad"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__56_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__56_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__57_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__55_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__57_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__58_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__58_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__34_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__59_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__54_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__59_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__60_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__50_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__60_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__61_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__46_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__61_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__62_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__42_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__62_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__63_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__38_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__63_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__64_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "pullback"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__65_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Pushout"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__66_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__66_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__67_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__65_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__67_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__68_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "pushouts"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__69_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Pullbacks"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__70_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__70_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__71_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__69_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__71_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__72_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "pullbacks"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__73_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Pushouts"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__74_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__74_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__75_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__73_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__75_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__76_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "span"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__77_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Cospan"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__78_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__78_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__79_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__77_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__79_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__80 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__80_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cospan"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__81 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__81_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Span"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__82 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__82_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__82_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__83 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__83_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__81_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__83_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__84_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "kernel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__85_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Cokernel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__86_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__86_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__87 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__87_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__85_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__87_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__88 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__88_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__88_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__64_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__89 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__89_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__84_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__89_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__90 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__90_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__80_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__90_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__91 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__91_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__76_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__91_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__92 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__92_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__72_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__92_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__93 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__93_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__94_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__68_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__93_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__94 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__94_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "colimits"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__95 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__95_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Limits"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__96 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__96_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__96_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__97 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__97_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__95_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__97_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__98 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__98_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "product"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__99 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__99_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Coproduct"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__100 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__100_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__100_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__101 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__101_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__99_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__101_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__102 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__102_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "coproduct"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__103 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__103_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__104_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Product"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__104 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__104_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__105_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__104_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__105 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__105_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__106_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__103_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__105_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__106 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__106_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__107_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "products"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__107 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__107_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__108_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Coproducts"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__108 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__108_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__109_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__108_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__109 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__109_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__110_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__107_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__109_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__110 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__110_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__111_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "coproducts"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__111 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__111_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__112_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Products"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__112 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__112_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__113_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__112_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__113 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__113_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__114_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__111_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__113_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__114 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__114_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__115_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "pushout"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__115 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__115_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__116_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Pullback"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__116 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__116_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__117_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__116_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__117 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__117_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__118_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__115_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__117_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__118 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__118_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__119_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__118_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__94_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__119 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__119_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__120_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__114_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__119_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__120 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__120_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__121_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__110_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__120_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__121 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__121_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__122_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__106_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__121_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__122 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__122_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__123_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__102_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__122_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__123 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__123_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__124_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__98_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__123_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__124 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__124_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__125_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fan"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__125 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__125_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__126_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Cofan"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__126 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__126_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__127_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__126_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__127 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__127_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__128_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__125_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__127_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__128 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__128_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__129_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cofan"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__129 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__129_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__130_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Fan"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__130 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__130_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__131_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__130_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__131 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__131_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__132_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__129_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__131_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__132 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__132_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__133_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "limit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__133 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__133_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__134_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Colimit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__134 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__134_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__135_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__134_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__135 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__135_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__136_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__133_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__135_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__136 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__136_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__137_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "colimit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__137 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__137_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__138_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Limit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__138 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__138_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__139_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__138_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__139 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__139_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__140_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__137_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__139_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__140 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__140_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__141_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lim"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__141 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__141_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__142_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Colim"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__142 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__142_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__143_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__142_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__143 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__143_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__144_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__141_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__143_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__144 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__144_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__145_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colim"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__145 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__145_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__146_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Lim"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__146 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__146_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__147_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__146_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__147 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__147_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__148_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__145_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__147_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__148 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__148_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__149_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "limits"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__149 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__149_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__150_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Colimits"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__150 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__150_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__151_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__150_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__151 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__151_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__152_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__149_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__151_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__152 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__152_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__153_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__152_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__124_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__153 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__153_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__154_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__148_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__153_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__154 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__154_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__155_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__144_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__154_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__155 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__155_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__156_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__140_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__155_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__156 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__156_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__157_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__136_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__156_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__157 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__157_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__158_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__132_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__157_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__158 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__158_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__159_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__128_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__158_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__159 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__159_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__160_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "precompose"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__160 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__160_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__161_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Postcompose"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__161 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__161_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__162_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__161_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__162 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__162_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__163_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__160_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__162_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__163 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__163_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__164_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "postcompose"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__164 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__164_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__165_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Precompose"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__165 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__165_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__166_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__165_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__166 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__166_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__167_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__164_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__166_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__167 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__167_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__168_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cone"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__168 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__168_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__169_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Cocone"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__169 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__169_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__170_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__169_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__170 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__170_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__171_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__168_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__170_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__171 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__171_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__172_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cocone"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__172 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__172_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__173_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Cone"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__173 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__173_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__174_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__173_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__174 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__174_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__175_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__172_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__174_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__175 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__175_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__176_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cones"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__176 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__176_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__177_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Cocones"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__177 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__177_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__178_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__177_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__178 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__178_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__179_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__176_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__178_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__179 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__179_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__180_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "cocones"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__180 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__180_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__181_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Cones"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__181 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__181_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__182_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__181_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__182 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__182_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__183_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__180_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__182_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__183 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__183_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__184_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__183_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__159_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__184 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__184_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__185_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__179_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__184_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__185 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__185_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__186_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__175_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__185_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__186 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__186_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__187_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__171_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__186_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__187 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__187_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__188_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__167_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__187_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__188 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__188_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__189_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__163_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__188_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__189 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__189_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__190_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "hypograph"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__190 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__190_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__191_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Epigraph"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__191 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__191_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__192_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__191_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__192 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__192_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__193_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__190_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__192_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__193 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__193_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__194_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "epi"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__194 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__194_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__195_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Mono"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__195 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__195_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__196_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__195_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__196 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__196_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__197_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__194_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__196_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__197 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__197_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__198_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "epimorphisms"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__198 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__198_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__199_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Monomorphisms"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__199 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__199_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__200_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__199_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__200 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__200_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__201_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__198_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__200_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__201 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__201_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__202_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "monomorphisms"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__202 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__202_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__203_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Epimorphisms"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__203 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__203_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__204_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__203_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__204 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__204_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__205_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__202_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__204_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__205 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__205_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__206_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "terminal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__206 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__206_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__207_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Initial"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__207 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__207_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__208_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__207_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__208 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__208_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__209_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__206_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__208_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__209 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__209_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__210_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "initial"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__210 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__210_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__211_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Terminal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__211 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__211_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__212_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__211_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__212 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__212_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__213_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__210_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__212_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__213 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__213_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__214_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__213_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__189_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__214 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__214_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__215_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__209_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__214_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__215 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__215_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__216_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__205_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__215_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__216 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__216_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__217_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__201_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__216_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__217 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__217_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__218_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__197_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__217_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__218 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__218_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__219_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__193_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__218_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__219 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__219_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__220_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "prev"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__220 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__220_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__221_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Next"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__221 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__221_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__222_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__221_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__222 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__222_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__223_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__220_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__222_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__223 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__223_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__224_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "heyting"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__224 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__224_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__225_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Coheyting"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__225 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__225_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__226_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__225_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__226 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__226_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__227_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__224_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__226_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__227 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__227_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__228_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "coheyting"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__228 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__228_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__229_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Heyting"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__229 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__229_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__230_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__229_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__230 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__230_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__231_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__228_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__230_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__231 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__231_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__232_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "frame"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__232 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__232_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__233_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Coframe"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__233 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__233_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__234_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__233_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__234 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__234_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__235_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__232_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__234_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__235 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__235_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__236_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "coframe"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__236 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__236_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__237_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Frame"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__237 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__237_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__238_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__237_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__238 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__238_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__239_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__236_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__238_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__239 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__239_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__240_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "epigraph"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__240 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__240_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__241_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Hypograph"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__241 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__241_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__242_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__241_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__242 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__242_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__243_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__240_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__242_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__243 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__243_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__244_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__243_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__219_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__244 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__244_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__245_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__239_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__244_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__245 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__245_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__246_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__235_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__245_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__246 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__246_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__247_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__231_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__246_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__247 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__247_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__248_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__227_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__247_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__248 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__248_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__249_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__223_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__248_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__249 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__249_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__250_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ioi"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__250 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__250_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__251_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iio"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__251 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__251_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__252_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__251_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__252 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__252_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__253_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__250_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__252_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__253 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__253_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__254_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "iio"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__254 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__254_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__255_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ioi"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__255 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__255_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__256_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__255_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__256 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__256_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__257_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__254_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__256_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__257 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__257_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__258_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ici"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__258 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__258_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__259_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__259 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__259_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__260_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__259_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__260 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__260_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__261_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__258_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__260_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__261 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__261_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__262_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "iic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__262 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__262_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__263_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ici"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__263 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__263_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__264_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__263_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__264 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__264_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__265_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__262_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__264_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__265 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__265_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__266_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ioc"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__266 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__266_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__267_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ico"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__267 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__267_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__268_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__267_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__268 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__268_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__269_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__266_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__268_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__269 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__269_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__270_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ico"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__270 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__270_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__271_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ioc"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__271 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__271_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__272_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__271_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__272 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__272_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__273_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__270_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__272_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__273 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__273_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__274_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "next"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__274 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__274_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__275_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Prev"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__275 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__275_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__276_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__275_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__276 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__276_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__277_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__274_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__276_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__277 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__277_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__278_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__277_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__249_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__278 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__278_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__279_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__273_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__278_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__279 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__279_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__280_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__269_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__279_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__280 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__280_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__281_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__265_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__280_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__281 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__281_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__282_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__261_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__281_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__282 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__282_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__283_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__257_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__282_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__283 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__283_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__284_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__253_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__283_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__284 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__284_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__285_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "disjoint"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__285 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__285_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__286_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Codisjoint"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__286 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__286_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__287_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__286_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__287 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__287_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__288_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__285_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__287_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__288 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__288_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__289_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "codisjoint"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__289 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__289_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__290_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Disjoint"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__290 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__290_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__291_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__290_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__291 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__291_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__292_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__289_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__291_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__292 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__292_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__293_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "atom"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__293 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__293_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__294_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Coatom"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__294 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__294_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__295_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__294_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__295 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__295_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__296_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__293_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__295_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__296 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__296_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__297_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "coatom"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__297 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__297_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__298_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Atom"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__298 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__298_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__299_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__298_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__299 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__299_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__300_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__297_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__299_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__300 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__300_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__301_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lfp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__301 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__301_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__302_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Gfp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__302 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__302_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__303_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__302_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__303 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__303_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__304_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__301_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__303_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__304 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__304_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__305_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "gfp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__305 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__305_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__306_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Lfp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__306 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__306_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__307_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__306_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__307 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__307_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__308_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__305_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__307_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__308 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__308_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__309_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__308_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__284_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__309 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__309_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__310_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__304_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__309_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__310 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__310_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__311_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__300_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__310_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__311 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__311_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__312_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__296_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__311_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__312 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__312_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__313_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__292_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__312_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__313 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__313_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__314_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__288_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__313_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__314 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__314_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__315_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "glb"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__315 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__315_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__316_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "LUB"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__316 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__316_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__317_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__316_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__317 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__317_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__318_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__315_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__317_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__318 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__318_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__319_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__319 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__319_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__320_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "GLB"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__320 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__320_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__321_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__320_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__321 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__321_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__322_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__319_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__321_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__322 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__322_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__323_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "cofinal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__323 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__323_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__324_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Coinitial"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__324 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__324_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__325_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__324_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__325 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__325_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__326_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__323_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__325_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__326 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__326_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__327_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "coinitial"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__327 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__327_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__328_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Cofinal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__328 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__328_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__329_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__328_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__329 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__329_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__330_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__327_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__329_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__330 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__330_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__331_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "succ"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__331 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__331_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__332_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Pred"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__332 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__332_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__333_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__332_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__333 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__333_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__334_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__331_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__333_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__334 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__334_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__335_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "pred"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__335 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__335_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__336_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Succ"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__336 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__336_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__337_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__336_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__337 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__337_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__338_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__335_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__337_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__338 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__338_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__339_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__338_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__314_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__339 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__339_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__340_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__334_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__339_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__340 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__340_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__341_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__330_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__340_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__341 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__341_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__342_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__326_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__341_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__342 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__342_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__343_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__322_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__342_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__343 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__343_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__344_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__318_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__343_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__344 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__344_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__345_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lower"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__345 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__345_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__346_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Upper"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__346 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__346_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__347_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__346_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__347 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__347_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__348_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__345_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__347_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__348 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__348_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__349_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "upper"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__349 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__349_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__350_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Lower"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__350 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__350_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__351_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__350_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__351 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__351_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__352_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__349_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__351_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__352 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__352_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__353_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "below"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__353 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__353_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__354_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Above"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__354 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__354_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__355_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__354_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__355 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__355_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__356_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__353_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__355_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__356 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__356_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__357_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "above"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__357 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__357_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__358_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Below"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__358 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__358_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__359_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__358_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__359 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__359_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__360_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__357_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__359_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__360 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__360_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__361_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "least"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__361 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__361_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__362_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Greatest"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__362 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__362_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__363_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__362_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__363 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__363_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__364_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__361_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__363_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__364 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__364_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__365_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "greatest"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__365 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__365_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__366_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Least"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__366 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__366_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__367_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__366_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__367 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__367_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__368_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__365_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__367_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__368 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__368_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__369_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__368_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__344_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__369 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__369_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__370_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__364_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__369_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__370 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__370_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__371_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__360_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__370_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__371 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__371_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__372_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__356_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__371_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__372 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__372_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__373_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__352_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__372_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__373 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__373_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__374_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__348_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__373_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__374 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__374_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__375_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "argmin"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__375 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__375_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__376_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Argmax"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__376 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__376_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__377_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__376_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__377 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__377_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__378_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__375_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__377_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__378 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__378_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__379_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "argmax"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__379 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__379_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__380_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Argmin"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__380 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__380_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__381_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__380_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__381 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__381_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__382_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__379_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__381_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__382 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__382_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__383_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "minimum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__383 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__383_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__384_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Maximum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__384 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__384_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__385_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__384_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__385 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__385_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__386_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__383_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__385_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__386 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__386_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__387_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "maximum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__387 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__387_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__388_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Minimum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__388 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__388_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__389_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__388_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__389 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__389_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__390_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__387_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__389_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__390 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__390_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__391_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "minimal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__391 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__391_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__392_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Maximal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__392 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__392_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__393_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__392_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__393 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__393_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__394_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__391_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__393_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__394 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__394_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__395_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "maximal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__395 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__395_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__396_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Minimal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__396 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__396_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__397_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__396_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__397 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__397_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__398_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__395_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__397_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__398 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__398_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__399_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__398_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__374_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__399 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__399_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__400_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__394_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__399_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__400 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__400_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__401_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__390_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__400_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__401 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__401_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__402_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__386_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__401_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__402 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__402_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__403_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__382_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__402_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__403 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__403_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__404_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__378_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__403_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__404 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__404_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__405_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "bliminf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__405 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__405_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__406_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Blimsup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__406 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__406_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__407_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__406_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__407 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__407_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__408_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__405_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__407_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__408 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__408_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__409_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "blimsup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__409 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__409_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__410_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Bliminf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__410 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__410_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__411_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__410_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__411 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__411_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__412_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__409_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__411_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__412 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__412_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__413_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "min"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__413 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__413_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__414_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Max"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__414 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__414_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__415_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__414_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__415 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__415_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__416_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__413_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__415_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__416 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__416_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__417_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "max"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__417 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__417_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__418_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Min"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__418 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__418_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__419_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__418_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__419 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__419_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__420_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__417_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__419_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__420 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__420_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__421_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "min\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__421 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__421_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__422_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Max\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__422 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__422_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__423_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__422_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__423 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__423_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__424_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__421_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__423_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__424 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__424_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__425_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "max\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__425 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__425_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__426_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Min\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__426 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__426_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__427_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__426_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__427 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__427_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__428_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__425_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__427_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__428 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__428_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__429_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__428_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__404_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__429 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__429_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__430_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__424_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__429_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__430 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__430_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__431_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__420_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__430_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__431 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__431_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__432_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__416_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__431_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__432 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__432_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__433_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__412_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__432_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__433 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__433_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__434_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__408_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__433_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__434 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__434_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__435_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = "inf₂"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__435 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__435_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__436_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = "Sup₂"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__436 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__436_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__437_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__436_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__437 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__437_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__438_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__435_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__437_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__438 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__438_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__439_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = "sup₂"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__439 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__439_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__440_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = "Inf₂"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__440 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__440_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__441_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__440_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__441 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__441_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__442_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__439_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__441_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__442 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__442_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__443_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "sinf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__443 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__443_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__444_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "SSup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__444 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__444_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__445_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__444_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__445 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__445_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__446_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__443_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__445_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__446 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__446_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__447_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "ssup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__447 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__447_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__448_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "SInf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__448 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__448_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__449_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__448_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__449 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__449_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__450_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__447_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__449_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__450 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__450_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__451_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "liminf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__451 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__451_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__452_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Limsup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__452 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__452_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__453_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__452_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__453 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__453_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__454_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__451_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__453_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__454 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__454_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__455_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "limsup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__455 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__455_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__456_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Liminf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__456 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__456_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__457_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__456_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__457 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__457_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__458_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__455_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__457_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__458 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__458_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__459_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__458_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__434_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__459 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__459_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__460_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__454_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__459_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__460 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__460_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__461_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__450_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__460_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__461 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__461_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__462_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__446_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__461_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__462 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__462_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__463_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__442_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__462_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__463 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__463_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__464_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__438_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__463_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__464 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__464_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__465_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "top"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__465 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__465_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__466_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Bot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__466 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__466_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__467_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__466_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__467 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__467_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__468_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__465_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__467_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__468 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__468_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__469_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "bot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__469 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__469_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__470_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Top"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__470 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__470_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__471_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__470_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__471 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__471_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__472_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__469_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__471_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__472 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__472_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__473_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "untop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__473 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__473_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__474_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Unbot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__474 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__474_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__475_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__474_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__475 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__475_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__476_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__473_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__475_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__476 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__476_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__477_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "unbot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__477 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__477_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__478_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Untop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__478 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__478_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__479_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__478_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__479 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__479_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__480_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__477_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__479_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__480 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__480_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__481_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__481 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__481_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__482_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Sup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__482 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__482_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__483_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__482_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__483 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__483_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__484_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__481_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__483_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__484 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__484_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__485_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "sup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__485 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__485_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__486_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Inf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__486 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__486_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__487_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__486_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__487 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__487_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__488_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__485_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__487_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__488 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__488_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__489_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__488_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__464_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__489 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__489_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__490_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__484_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__489_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__490 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__490_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__491_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__480_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__490_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__491 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__491_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__492_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__476_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__491_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__492 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__492_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__493_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__472_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__492_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__493 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__493_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__494_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__468_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__493_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__494 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__494_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__495_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__495;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__496_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__496;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__497_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__497;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual_nameDict;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "wellFoundedLT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "WellFoundedGT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "wellFoundedGT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "WellFoundedLT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "nhdsLT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "NhdsGT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "nhdsGT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "NhdsLT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "nhdsLE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "NhdsGE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "nhdsGE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "NhdsLE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "relIsoLT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "RelIsoGT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "relIsoGT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "RelIsoLT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__21_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "succColimit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "SuccLimit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__24_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "predColimit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "PredLimit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__27_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "codirectedOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "DirectedOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__30_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "directedOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "CodirectedOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__33_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__34_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "galoisInsertion"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "GaloisCoinsertion"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__36_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__37_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "galoisCoinsertion"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "GaloisInsertion"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__39_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__40_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__41_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "leftOrdContinuous"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "RightOrdContinuous"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__42_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__43_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__44_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "rightOrdContinuous"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__45_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "LeftOrdContinuous"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__45_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__46_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "bihimp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__48_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "SymmDiff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__48_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__49_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__50_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "symmDiff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__51_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Bihimp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__52_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__51_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__52_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__53_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "isRightContinuous"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__54_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "IsLeftContinuous"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__55_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__54_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__55_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__56_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "isLeftContinuous"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__57_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "IsRightContinuous"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__58_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__57_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__58_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__59_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "isCadlag"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__60_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "IsCaglad"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__61_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__60_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__61_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__62_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "isCaglad"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__63_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "IsCadlag"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__64_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__63_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__64_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__65_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "neTop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__66_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "NeBot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__67_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__66_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__67_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__68_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "decidableSucc"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__69_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "DecidablePred"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__70_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__69_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__70_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__71_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ofSucc"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__72_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "OfPred"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__73_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__72_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__73_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__74_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "maximalAxioms"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__75_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "MinimalAxioms"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__76_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__75_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__76_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__77_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__77_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__78_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__74_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__78_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__79_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__71_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__79_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__80 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__80_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__68_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__80_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__81 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__81_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__65_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__81_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__82 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__82_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__62_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__82_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__83 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__83_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__59_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__83_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__84_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__56_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__84_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__85_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__53_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__85_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__86_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__50_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__86_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__87 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__87_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__47_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__87_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__88 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__88_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__44_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__88_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__89 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__89_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__41_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__89_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__90 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__90_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__38_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__90_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__91 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__91_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__35_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__91_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__92 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__92_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__32_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__92_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__93 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__93_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__94_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__29_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__93_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__94 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__94_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__26_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__94_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__95 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__95_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__23_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__95_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__96 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__96_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__20_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__96_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__97 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__97_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__97_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__98 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__98_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__98_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__99 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__99_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__99_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__100 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__100_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__100_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__101 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__101_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__101_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__102 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__102_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__102_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__103 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__103_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__104_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__104;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual_guessNameExt;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToDual_data___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToDual_data___closed__0;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_data___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__0_value),LEAN_SCALAR_PTR_LITERAL(55, 203, 172, 87, 157, 224, 154, 102)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_data___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_data___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToDual_data___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToDual_data___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual_data;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "commandTo_dual_insert_cast_:=_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(184, 37, 29, 107, 19, 180, 43, 83)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "to_dual_insert_cast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__15_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "commandTo_dual_insert_cast_fun_:=_,_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(192, 181, 115, 24, 63, 181, 145, 77)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "to_dual_insert_cast_fun"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast__fun___x3a_x3d___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast__fun___x3a_x3d___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__19_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1319075495) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(150, 8, 210, 235, 237, 115, 250, 21)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__21_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(49, 109, 13, 66, 42, 162, 198, 192)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__23_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(201, 131, 66, 202, 207, 95, 131, 187)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(180, 231, 50, 66, 144, 162, 54, 146)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed, .m_arity = 8, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_data___closed__1_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Transport to dual"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_data___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "commandInsert_to_dual_translation__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(67, 104, 1, 141, 246, 230, 38, 12)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "insert_to_dual_translation"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation____ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "commandTo_dual_name_hint__,,"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(58, 28, 218, 253, 153, 242, 68, 251)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__0_value),LEAN_SCALAR_PTR_LITERAL(76, 200, 84, 111, 223, 184, 34, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "to_dual_name_hint"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__4_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__9(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_95_ = lp_mathlib_Mathlib_Tactic_Translate_attrArgs;
v___x_96_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__8));
v___x_97_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6));
v___x_98_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v___x_96_);
lean_ctor_set(v___x_98_, 2, v___x_95_);
return v___x_98_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__10(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_99_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__9, &lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__9);
v___x_100_ = lean_unsigned_to_nat(1022u);
v___x_101_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1));
v___x_102_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v___x_100_);
lean_ctor_set(v___x_102_, 2, v___x_99_);
return v___x_102_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_to__dual(void){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__10, &lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__10);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__4(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_114_ = lp_mathlib_Mathlib_Tactic_Translate_attrArgs;
v___x_115_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__3));
v___x_116_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__6));
v___x_117_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v___x_115_);
lean_ctor_set(v___x_117_, 2, v___x_114_);
return v___x_117_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__5(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_118_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__4, &lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__4);
v___x_119_ = lean_unsigned_to_nat(1022u);
v___x_120_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1));
v___x_121_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v___x_119_);
lean_ctor_set(v___x_121_, 2, v___x_118_);
return v___x_121_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f__(void){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__5, &lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__5);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1(lean_object* v_x_126_, lean_object* v_a_127_, lean_object* v_a_128_){
_start:
{
lean_object* v___x_129_; uint8_t v___x_130_; 
v___x_129_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f___00__closed__1));
lean_inc(v_x_126_);
v___x_130_ = l_Lean_Syntax_isOfKind(v_x_126_, v___x_129_);
if (v___x_130_ == 0)
{
lean_object* v___x_131_; lean_object* v___x_132_; 
lean_dec(v_x_126_);
v___x_131_ = lean_box(1);
v___x_132_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
lean_ctor_set(v___x_132_, 1, v_a_128_);
return v___x_132_;
}
else
{
lean_object* v_ref_133_; lean_object* v___x_134_; lean_object* v___x_135_; uint8_t v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v_ref_133_ = lean_ctor_get(v_a_127_, 5);
v___x_134_ = lean_unsigned_to_nat(1u);
v___x_135_ = l_Lean_Syntax_getArg(v_x_126_, v___x_134_);
lean_dec(v_x_126_);
v___x_136_ = 0;
v___x_137_ = l_Lean_SourceInfo_fromRef(v_ref_133_, v___x_136_);
v___x_138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__0));
v___x_139_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__1));
lean_inc_n(v___x_137_, 3);
v___x_140_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_137_);
lean_ctor_set(v___x_140_, 1, v___x_138_);
v___x_141_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1___closed__1));
v___x_142_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_to__dual___closed__5));
v___x_143_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_143_, 0, v___x_137_);
lean_ctor_set(v___x_143_, 1, v___x_142_);
v___x_144_ = l_Lean_Syntax_node1(v___x_137_, v___x_141_, v___x_143_);
v___x_145_ = l_Lean_Syntax_node3(v___x_137_, v___x_139_, v___x_140_, v___x_144_, v___x_135_);
v___x_146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_145_);
lean_ctor_set(v___x_146_, 1, v_a_128_);
return v___x_146_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1___boxed(lean_object* v_x_147_, lean_object* v_a_148_, lean_object* v_a_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______macroRules__Mathlib__Tactic__ToDual__attrTo__dual_x3f____1(v_x_147_, v_a_148_, v_a_149_);
lean_dec_ref(v_a_148_);
return v_res_150_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_151_ = lean_box(0);
v___x_152_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_153_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v___x_151_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg(){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_155_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___closed__0);
v___x_156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_156_, 0, v___x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v___y_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg();
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_159_, lean_object* v___y_160_, lean_object* v___y_161_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg();
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0(v_00_u03b1_164_, v___y_165_, v___y_166_);
lean_dec(v___y_166_);
lean_dec_ref(v___y_165_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__2(size_t v_sz_169_, size_t v_i_170_, lean_object* v_bs_171_){
_start:
{
uint8_t v___x_172_; 
v___x_172_ = lean_usize_dec_lt(v_i_170_, v_sz_169_);
if (v___x_172_ == 0)
{
return v_bs_171_;
}
else
{
lean_object* v___x_173_; lean_object* v_v_174_; lean_object* v___x_175_; lean_object* v_bs_x27_176_; lean_object* v___x_177_; lean_object* v___x_178_; size_t v___x_179_; size_t v___x_180_; lean_object* v___x_181_; 
v___x_173_ = lean_unsigned_to_nat(1u);
v_v_174_ = lean_array_uget(v_bs_171_, v_i_170_);
v___x_175_ = lean_unsigned_to_nat(0u);
v_bs_x27_176_ = lean_array_uset(v_bs_171_, v_i_170_, v___x_175_);
v___x_177_ = l_Lean_TSyntax_getNat(v_v_174_);
lean_dec(v_v_174_);
v___x_178_ = lean_nat_sub(v___x_177_, v___x_173_);
lean_dec(v___x_177_);
v___x_179_ = ((size_t)1ULL);
v___x_180_ = lean_usize_add(v_i_170_, v___x_179_);
v___x_181_ = lean_array_uset(v_bs_x27_176_, v_i_170_, v___x_178_);
v_i_170_ = v___x_180_;
v_bs_171_ = v___x_181_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__2___boxed(lean_object* v_sz_183_, lean_object* v_i_184_, lean_object* v_bs_185_){
_start:
{
size_t v_sz_boxed_186_; size_t v_i_boxed_187_; lean_object* v_res_188_; 
v_sz_boxed_186_ = lean_unbox_usize(v_sz_183_);
lean_dec(v_sz_183_);
v_i_boxed_187_ = lean_unbox_usize(v_i_184_);
lean_dec(v_i_184_);
v_res_188_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__2(v_sz_boxed_186_, v_i_boxed_187_, v_bs_185_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__1(size_t v_sz_189_, size_t v_i_190_, lean_object* v_bs_191_){
_start:
{
uint8_t v___x_192_; 
v___x_192_ = lean_usize_dec_lt(v_i_190_, v_sz_189_);
if (v___x_192_ == 0)
{
lean_object* v___x_193_; 
v___x_193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_193_, 0, v_bs_191_);
return v___x_193_;
}
else
{
lean_object* v_v_194_; lean_object* v___x_195_; uint8_t v___x_196_; 
v_v_194_ = lean_array_uget(v_bs_191_, v_i_190_);
v___x_195_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_to__dual__ignore__args___closed__14));
lean_inc(v_v_194_);
v___x_196_ = l_Lean_Syntax_isOfKind(v_v_194_, v___x_195_);
if (v___x_196_ == 0)
{
lean_object* v___x_197_; 
lean_dec(v_v_194_);
lean_dec_ref(v_bs_191_);
v___x_197_ = lean_box(0);
return v___x_197_;
}
else
{
lean_object* v___x_198_; lean_object* v_bs_x27_199_; size_t v___x_200_; size_t v___x_201_; lean_object* v___x_202_; 
v___x_198_ = lean_unsigned_to_nat(0u);
v_bs_x27_199_ = lean_array_uset(v_bs_191_, v_i_190_, v___x_198_);
v___x_200_ = ((size_t)1ULL);
v___x_201_ = lean_usize_add(v_i_190_, v___x_200_);
v___x_202_ = lean_array_uset(v_bs_x27_199_, v_i_190_, v_v_194_);
v_i_190_ = v___x_201_;
v_bs_191_ = v___x_202_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__1___boxed(lean_object* v_sz_204_, lean_object* v_i_205_, lean_object* v_bs_206_){
_start:
{
size_t v_sz_boxed_207_; size_t v_i_boxed_208_; lean_object* v_res_209_; 
v_sz_boxed_207_ = lean_unbox_usize(v_sz_204_);
lean_dec(v_sz_204_);
v_i_boxed_208_ = lean_unbox_usize(v_i_205_);
lean_dec(v_i_205_);
v_res_209_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__1(v_sz_boxed_207_, v_i_boxed_208_, v_bs_206_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_(lean_object* v___x_210_, lean_object* v___x_211_, lean_object* v___x_212_, lean_object* v___x_213_, lean_object* v_x_214_, lean_object* v_stx_215_, lean_object* v___y_216_, lean_object* v___y_217_){
_start:
{
lean_object* v_ids_220_; lean_object* v___x_223_; uint8_t v___x_224_; 
v___x_223_ = l_Lean_Name_mkStr4(v___x_210_, v___x_211_, v___x_212_, v___x_213_);
lean_inc(v_stx_215_);
v___x_224_ = l_Lean_Syntax_isOfKind(v_stx_215_, v___x_223_);
lean_dec(v___x_223_);
if (v___x_224_ == 0)
{
lean_object* v___x_225_; lean_object* v_a_226_; lean_object* v___x_228_; uint8_t v_isShared_229_; uint8_t v_isSharedCheck_233_; 
lean_dec(v_stx_215_);
v___x_225_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg();
v_a_226_ = lean_ctor_get(v___x_225_, 0);
v_isSharedCheck_233_ = !lean_is_exclusive(v___x_225_);
if (v_isSharedCheck_233_ == 0)
{
v___x_228_ = v___x_225_;
v_isShared_229_ = v_isSharedCheck_233_;
goto v_resetjp_227_;
}
else
{
lean_inc(v_a_226_);
lean_dec(v___x_225_);
v___x_228_ = lean_box(0);
v_isShared_229_ = v_isSharedCheck_233_;
goto v_resetjp_227_;
}
v_resetjp_227_:
{
lean_object* v___x_231_; 
if (v_isShared_229_ == 0)
{
v___x_231_ = v___x_228_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v_a_226_);
v___x_231_ = v_reuseFailAlloc_232_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
return v___x_231_;
}
}
}
else
{
lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; size_t v_sz_237_; size_t v___x_238_; lean_object* v___x_239_; 
v___x_234_ = lean_unsigned_to_nat(1u);
v___x_235_ = l_Lean_Syntax_getArg(v_stx_215_, v___x_234_);
lean_dec(v_stx_215_);
v___x_236_ = l_Lean_Syntax_getArgs(v___x_235_);
lean_dec(v___x_235_);
v_sz_237_ = lean_array_size(v___x_236_);
v___x_238_ = ((size_t)0ULL);
v___x_239_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__1(v_sz_237_, v___x_238_, v___x_236_);
if (lean_obj_tag(v___x_239_) == 0)
{
lean_object* v___x_240_; lean_object* v_a_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_248_; 
v___x_240_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg();
v_a_241_ = lean_ctor_get(v___x_240_, 0);
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_240_);
if (v_isSharedCheck_248_ == 0)
{
v___x_243_ = v___x_240_;
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
else
{
lean_inc(v_a_241_);
lean_dec(v___x_240_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v___x_246_; 
if (v_isShared_244_ == 0)
{
v___x_246_ = v___x_243_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_a_241_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
return v___x_246_;
}
}
}
else
{
lean_object* v_val_249_; size_t v_sz_250_; lean_object* v___x_251_; 
v_val_249_ = lean_ctor_get(v___x_239_, 0);
lean_inc(v_val_249_);
lean_dec_ref_known(v___x_239_, 1);
v_sz_250_ = lean_array_size(v_val_249_);
v___x_251_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__2(v_sz_250_, v___x_238_, v_val_249_);
v_ids_220_ = v___x_251_;
goto v___jp_219_;
}
}
v___jp_219_:
{
lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_221_ = lean_array_to_list(v_ids_220_);
v___x_222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_222_, 0, v___x_221_);
return v___x_222_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2____boxed(lean_object* v___x_252_, lean_object* v___x_253_, lean_object* v___x_254_, lean_object* v___x_255_, lean_object* v_x_256_, lean_object* v_stx_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_(v___x_252_, v___x_253_, v___x_254_, v___x_255_, v_x_256_, v_stx_257_, v___y_258_, v___y_259_);
lean_dec(v___y_259_);
lean_dec_ref(v___y_258_);
lean_dec(v_x_256_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; 
v___x_282_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_));
v___x_283_ = lp_batteries_Lean_registerNameMapAttribute___redArg(v___x_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2____boxed(lean_object* v_a_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_();
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_678359338____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_287_; 
v___x_287_ = lp_mathlib_Mathlib_Tactic_UnfoldBoundary_registerUnfoldBoundaryExt();
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_678359338____hygCtx___hyg_2____boxed(lean_object* v_a_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_678359338____hygCtx___hyg_2_();
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_297_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2_));
v___x_298_ = lp_batteries_Lean_registerNameMapExtension___redArg(v___x_297_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2____boxed(lean_object* v_a_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2_();
return v_res_300_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__0(void){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_301_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_302_; lean_object* v___x_303_; 
v___x_302_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__0);
v___x_303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_303_, 0, v___x_302_);
return v___x_303_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__2(void){
_start:
{
lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; 
v___x_304_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__1);
v___x_305_ = lean_unsigned_to_nat(0u);
v___x_306_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_306_, 0, v___x_305_);
lean_ctor_set(v___x_306_, 1, v___x_305_);
lean_ctor_set(v___x_306_, 2, v___x_305_);
lean_ctor_set(v___x_306_, 3, v___x_305_);
lean_ctor_set(v___x_306_, 4, v___x_304_);
lean_ctor_set(v___x_306_, 5, v___x_304_);
lean_ctor_set(v___x_306_, 6, v___x_304_);
lean_ctor_set(v___x_306_, 7, v___x_304_);
lean_ctor_set(v___x_306_, 8, v___x_304_);
lean_ctor_set(v___x_306_, 9, v___x_304_);
return v___x_306_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__3(void){
_start:
{
lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
v___x_307_ = lean_unsigned_to_nat(32u);
v___x_308_ = lean_mk_empty_array_with_capacity(v___x_307_);
v___x_309_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_309_, 0, v___x_308_);
return v___x_309_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__4(void){
_start:
{
size_t v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; 
v___x_310_ = ((size_t)5ULL);
v___x_311_ = lean_unsigned_to_nat(0u);
v___x_312_ = lean_unsigned_to_nat(32u);
v___x_313_ = lean_mk_empty_array_with_capacity(v___x_312_);
v___x_314_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__3);
v___x_315_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_315_, 0, v___x_314_);
lean_ctor_set(v___x_315_, 1, v___x_313_);
lean_ctor_set(v___x_315_, 2, v___x_311_);
lean_ctor_set(v___x_315_, 3, v___x_311_);
lean_ctor_set_usize(v___x_315_, 4, v___x_310_);
return v___x_315_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__5(void){
_start:
{
lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_316_ = lean_box(1);
v___x_317_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__4);
v___x_318_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__1);
v___x_319_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_319_, 0, v___x_318_);
lean_ctor_set(v___x_319_, 1, v___x_317_);
lean_ctor_set(v___x_319_, 2, v___x_316_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2(lean_object* v_msgData_320_, lean_object* v___y_321_, lean_object* v___y_322_){
_start:
{
lean_object* v___x_324_; lean_object* v_env_325_; lean_object* v_options_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_324_ = lean_st_ref_get(v___y_322_);
v_env_325_ = lean_ctor_get(v___x_324_, 0);
lean_inc_ref(v_env_325_);
lean_dec(v___x_324_);
v_options_326_ = lean_ctor_get(v___y_321_, 2);
v___x_327_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__2);
v___x_328_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__5);
lean_inc_ref(v_options_326_);
v___x_329_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_329_, 0, v_env_325_);
lean_ctor_set(v___x_329_, 1, v___x_327_);
lean_ctor_set(v___x_329_, 2, v___x_328_);
lean_ctor_set(v___x_329_, 3, v_options_326_);
v___x_330_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_330_, 0, v___x_329_);
lean_ctor_set(v___x_330_, 1, v_msgData_320_);
v___x_331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_331_, 0, v___x_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object* v_msgData_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2(v_msgData_332_, v___y_333_, v___y_334_);
lean_dec(v___y_334_);
lean_dec_ref(v___y_333_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___redArg(lean_object* v_msg_337_, lean_object* v___y_338_, lean_object* v___y_339_){
_start:
{
lean_object* v_ref_341_; lean_object* v___x_342_; lean_object* v_a_343_; lean_object* v___x_345_; uint8_t v_isShared_346_; uint8_t v_isSharedCheck_351_; 
v_ref_341_ = lean_ctor_get(v___y_338_, 5);
v___x_342_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2(v_msg_337_, v___y_338_, v___y_339_);
v_a_343_ = lean_ctor_get(v___x_342_, 0);
v_isSharedCheck_351_ = !lean_is_exclusive(v___x_342_);
if (v_isSharedCheck_351_ == 0)
{
v___x_345_ = v___x_342_;
v_isShared_346_ = v_isSharedCheck_351_;
goto v_resetjp_344_;
}
else
{
lean_inc(v_a_343_);
lean_dec(v___x_342_);
v___x_345_ = lean_box(0);
v_isShared_346_ = v_isSharedCheck_351_;
goto v_resetjp_344_;
}
v_resetjp_344_:
{
lean_object* v___x_347_; lean_object* v___x_349_; 
lean_inc(v_ref_341_);
v___x_347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_347_, 0, v_ref_341_);
lean_ctor_set(v___x_347_, 1, v_a_343_);
if (v_isShared_346_ == 0)
{
lean_ctor_set_tag(v___x_345_, 1);
lean_ctor_set(v___x_345_, 0, v___x_347_);
v___x_349_ = v___x_345_;
goto v_reusejp_348_;
}
else
{
lean_object* v_reuseFailAlloc_350_; 
v_reuseFailAlloc_350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_350_, 0, v___x_347_);
v___x_349_ = v_reuseFailAlloc_350_;
goto v_reusejp_348_;
}
v_reusejp_348_:
{
return v___x_349_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object* v_msg_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_){
_start:
{
lean_object* v_res_356_; 
v_res_356_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___redArg(v_msg_352_, v___y_353_, v___y_354_);
lean_dec(v___y_354_);
lean_dec_ref(v___y_353_);
return v_res_356_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_357_; 
v___x_357_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_357_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; 
v___x_358_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0, &lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0);
v___x_359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_359_, 0, v___x_358_);
return v___x_359_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_360_; lean_object* v___x_361_; 
v___x_360_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1, &lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1);
v___x_361_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_361_, 0, v___x_360_);
lean_ctor_set(v___x_361_, 1, v___x_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object* v_env_362_, lean_object* v___y_363_){
_start:
{
lean_object* v___x_365_; lean_object* v_nextMacroScope_366_; lean_object* v_ngen_367_; lean_object* v_auxDeclNGen_368_; lean_object* v_traceState_369_; lean_object* v_messages_370_; lean_object* v_infoState_371_; lean_object* v_snapshotTasks_372_; lean_object* v___x_374_; uint8_t v_isShared_375_; uint8_t v_isSharedCheck_383_; 
v___x_365_ = lean_st_ref_take(v___y_363_);
v_nextMacroScope_366_ = lean_ctor_get(v___x_365_, 1);
v_ngen_367_ = lean_ctor_get(v___x_365_, 2);
v_auxDeclNGen_368_ = lean_ctor_get(v___x_365_, 3);
v_traceState_369_ = lean_ctor_get(v___x_365_, 4);
v_messages_370_ = lean_ctor_get(v___x_365_, 6);
v_infoState_371_ = lean_ctor_get(v___x_365_, 7);
v_snapshotTasks_372_ = lean_ctor_get(v___x_365_, 8);
v_isSharedCheck_383_ = !lean_is_exclusive(v___x_365_);
if (v_isSharedCheck_383_ == 0)
{
lean_object* v_unused_384_; lean_object* v_unused_385_; 
v_unused_384_ = lean_ctor_get(v___x_365_, 5);
lean_dec(v_unused_384_);
v_unused_385_ = lean_ctor_get(v___x_365_, 0);
lean_dec(v_unused_385_);
v___x_374_ = v___x_365_;
v_isShared_375_ = v_isSharedCheck_383_;
goto v_resetjp_373_;
}
else
{
lean_inc(v_snapshotTasks_372_);
lean_inc(v_infoState_371_);
lean_inc(v_messages_370_);
lean_inc(v_traceState_369_);
lean_inc(v_auxDeclNGen_368_);
lean_inc(v_ngen_367_);
lean_inc(v_nextMacroScope_366_);
lean_dec(v___x_365_);
v___x_374_ = lean_box(0);
v_isShared_375_ = v_isSharedCheck_383_;
goto v_resetjp_373_;
}
v_resetjp_373_:
{
lean_object* v___x_376_; lean_object* v___x_378_; 
v___x_376_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2, &lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2);
if (v_isShared_375_ == 0)
{
lean_ctor_set(v___x_374_, 5, v___x_376_);
lean_ctor_set(v___x_374_, 0, v_env_362_);
v___x_378_ = v___x_374_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v_env_362_);
lean_ctor_set(v_reuseFailAlloc_382_, 1, v_nextMacroScope_366_);
lean_ctor_set(v_reuseFailAlloc_382_, 2, v_ngen_367_);
lean_ctor_set(v_reuseFailAlloc_382_, 3, v_auxDeclNGen_368_);
lean_ctor_set(v_reuseFailAlloc_382_, 4, v_traceState_369_);
lean_ctor_set(v_reuseFailAlloc_382_, 5, v___x_376_);
lean_ctor_set(v_reuseFailAlloc_382_, 6, v_messages_370_);
lean_ctor_set(v_reuseFailAlloc_382_, 7, v_infoState_371_);
lean_ctor_set(v_reuseFailAlloc_382_, 8, v_snapshotTasks_372_);
v___x_378_ = v_reuseFailAlloc_382_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; 
v___x_379_ = lean_st_ref_set(v___y_363_, v___x_378_);
v___x_380_ = lean_box(0);
v___x_381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
return v___x_381_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object* v_env_386_, lean_object* v___y_387_, lean_object* v___y_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg(v_env_386_, v___y_387_);
lean_dec(v___y_387_);
return v_res_389_;
}
}
static lean_object* _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_391_; lean_object* v___x_392_; 
v___x_391_ = ((lean_object*)(lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__0));
v___x_392_ = l_Lean_stringToMessageData(v___x_391_);
return v___x_392_;
}
}
static lean_object* _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_394_; lean_object* v___x_395_; 
v___x_394_ = ((lean_object*)(lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__2));
v___x_395_ = l_Lean_stringToMessageData(v___x_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg(lean_object* v_ext_396_, lean_object* v_k_397_, lean_object* v_v_398_, lean_object* v___y_399_, lean_object* v___y_400_){
_start:
{
lean_object* v___x_402_; lean_object* v_env_403_; lean_object* v___x_404_; 
v___x_402_ = lean_st_ref_get(v___y_400_);
v_env_403_ = lean_ctor_get(v___x_402_, 0);
lean_inc_ref(v_env_403_);
lean_dec(v___x_402_);
v___x_404_ = lp_batteries_Lean_NameMapExtension_find_x3f___redArg(v_ext_396_, v_env_403_, v_k_397_);
if (lean_obj_tag(v___x_404_) == 1)
{
lean_object* v_name_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; 
lean_dec_ref_known(v___x_404_, 1);
lean_dec(v_v_398_);
v_name_405_ = lean_ctor_get(v_ext_396_, 1);
lean_inc(v_name_405_);
lean_dec_ref(v_ext_396_);
v___x_406_ = lean_obj_once(&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__1, &lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__1);
v___x_407_ = l_Lean_MessageData_ofName(v_name_405_);
v___x_408_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_406_);
lean_ctor_set(v___x_408_, 1, v___x_407_);
v___x_409_ = lean_obj_once(&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__3, &lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__3_once, _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__3);
v___x_410_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_410_, 0, v___x_408_);
lean_ctor_set(v___x_410_, 1, v___x_409_);
v___x_411_ = l_Lean_MessageData_ofName(v_k_397_);
v___x_412_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_412_, 0, v___x_410_);
lean_ctor_set(v___x_412_, 1, v___x_411_);
v___x_413_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___redArg(v___x_412_, v___y_399_, v___y_400_);
return v___x_413_;
}
else
{
lean_object* v___x_414_; lean_object* v_toEnvExtension_415_; lean_object* v_env_416_; lean_object* v_asyncMode_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; 
lean_dec(v___x_404_);
v___x_414_ = lean_st_ref_get(v___y_400_);
v_toEnvExtension_415_ = lean_ctor_get(v_ext_396_, 0);
v_env_416_ = lean_ctor_get(v___x_414_, 0);
lean_inc_ref(v_env_416_);
lean_dec(v___x_414_);
v_asyncMode_417_ = lean_ctor_get(v_toEnvExtension_415_, 2);
lean_inc(v_asyncMode_417_);
v___x_418_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_418_, 0, v_k_397_);
lean_ctor_set(v___x_418_, 1, v_v_398_);
v___x_419_ = lean_box(0);
v___x_420_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v_ext_396_, v_env_416_, v___x_418_, v_asyncMode_417_, v___x_419_);
lean_dec(v_asyncMode_417_);
v___x_421_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_420_, v___y_400_);
return v___x_421_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v_ext_422_, lean_object* v_k_423_, lean_object* v_v_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_){
_start:
{
lean_object* v_res_428_; 
v_res_428_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg(v_ext_422_, v_k_423_, v_v_424_, v___y_425_, v___y_426_);
lean_dec(v___y_426_);
lean_dec_ref(v___y_425_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(lean_object* v_name_429_, lean_object* v_x_430_, uint8_t v_x_431_, lean_object* v___y_432_, lean_object* v___y_433_){
_start:
{
lean_object* v___x_435_; uint8_t v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; 
v___x_435_ = lp_mathlib_Mathlib_Tactic_ToDual_doTranslateAttr;
v___x_436_ = 1;
v___x_437_ = lean_box(v___x_436_);
v___x_438_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg(v___x_435_, v_name_429_, v___x_437_, v___y_432_, v___y_433_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed(lean_object* v_name_439_, lean_object* v_x_440_, lean_object* v_x_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_){
_start:
{
uint8_t v_x_1766__boxed_445_; lean_object* v_res_446_; 
v_x_1766__boxed_445_ = lean_unbox(v_x_441_);
v_res_446_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(v_name_439_, v_x_440_, v_x_1766__boxed_445_, v___y_442_, v___y_443_);
lean_dec(v___y_443_);
lean_dec_ref(v___y_442_);
lean_dec(v_x_440_);
return v_res_446_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_448_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_449_ = l_Lean_stringToMessageData(v___x_448_);
return v___x_449_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_451_; lean_object* v___x_452_; 
v___x_451_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_452_ = l_Lean_stringToMessageData(v___x_451_);
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(lean_object* v___x_453_, lean_object* v_decl_454_, lean_object* v___y_455_, lean_object* v___y_456_){
_start:
{
lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; 
v___x_458_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_459_ = l_Lean_MessageData_ofName(v___x_453_);
v___x_460_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_460_, 0, v___x_458_);
lean_ctor_set(v___x_460_, 1, v___x_459_);
v___x_461_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_462_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_462_, 0, v___x_460_);
lean_ctor_set(v___x_462_, 1, v___x_461_);
v___x_463_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___redArg(v___x_462_, v___y_455_, v___y_456_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed(lean_object* v___x_464_, lean_object* v_decl_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_){
_start:
{
lean_object* v_res_469_; 
v_res_469_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(v___x_464_, v_decl_465_, v___y_466_, v___y_467_);
lean_dec(v___y_467_);
lean_dec_ref(v___y_466_);
lean_dec(v_decl_465_);
return v_res_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(lean_object* v_name_470_, lean_object* v_x_471_, uint8_t v_x_472_, lean_object* v___y_473_, lean_object* v___y_474_){
_start:
{
lean_object* v___x_476_; uint8_t v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; 
v___x_476_ = lp_mathlib_Mathlib_Tactic_ToDual_doTranslateAttr;
v___x_477_ = 0;
v___x_478_ = lean_box(v___x_477_);
v___x_479_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg(v___x_476_, v_name_470_, v___x_478_, v___y_473_, v___y_474_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed(lean_object* v_name_480_, lean_object* v_x_481_, lean_object* v_x_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_){
_start:
{
uint8_t v_x_1831__boxed_486_; lean_object* v_res_487_; 
v_x_1831__boxed_486_ = lean_unbox(v_x_482_);
v_res_487_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(v_name_480_, v_x_481_, v_x_1831__boxed_486_, v___y_483_, v___y_484_);
lean_dec(v___y_484_);
lean_dec_ref(v___y_483_);
lean_dec(v_x_481_);
return v_res_487_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; 
v___x_538_ = lean_unsigned_to_nat(4009335259u);
v___x_539_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__19_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_540_ = l_Lean_Name_num___override(v___x_539_, v___x_538_);
return v___x_540_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_542_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__21_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_543_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_544_ = l_Lean_Name_str___override(v___x_543_, v___x_542_);
return v___x_544_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; 
v___x_546_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__23_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_547_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_548_ = l_Lean_Name_str___override(v___x_547_, v___x_546_);
return v___x_548_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; 
v___x_549_ = lean_unsigned_to_nat(2u);
v___x_550_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_551_ = l_Lean_Name_num___override(v___x_550_, v___x_549_);
return v___x_551_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; 
v___x_557_ = 0;
v___x_558_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__28_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_559_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__26_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_560_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_561_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_561_, 0, v___x_560_);
lean_ctor_set(v___x_561_, 1, v___x_559_);
lean_ctor_set(v___x_561_, 2, v___x_558_);
lean_ctor_set_uint8(v___x_561_, sizeof(void*)*3, v___x_557_);
return v___x_561_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_562_; lean_object* v___f_563_; lean_object* v___x_564_; lean_object* v___x_565_; 
v___f_562_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__27_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___f_563_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_564_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_565_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_565_, 0, v___x_564_);
lean_ctor_set(v___x_565_, 1, v___f_563_);
lean_ctor_set(v___x_565_, 2, v___f_562_);
return v___x_565_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; 
v___x_572_ = 0;
v___x_573_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__34_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_574_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__32_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_575_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_576_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_576_, 0, v___x_575_);
lean_ctor_set(v___x_576_, 1, v___x_574_);
lean_ctor_set(v___x_576_, 2, v___x_573_);
lean_ctor_set_uint8(v___x_576_, sizeof(void*)*3, v___x_572_);
return v___x_576_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_577_; lean_object* v___f_578_; lean_object* v___x_579_; lean_object* v___x_580_; 
v___f_577_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__33_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___f_578_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__31_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_));
v___x_579_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_580_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_580_, 0, v___x_579_);
lean_ctor_set(v___x_580_, 1, v___f_578_);
lean_ctor_set(v___x_580_, 2, v___f_577_);
return v___x_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_582_; lean_object* v___x_583_; 
v___x_582_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_583_ = l_Lean_registerBuiltinAttribute(v___x_582_);
if (lean_obj_tag(v___x_583_) == 0)
{
lean_object* v___x_584_; lean_object* v___x_585_; 
lean_dec_ref_known(v___x_583_, 1);
v___x_584_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_585_ = l_Lean_registerBuiltinAttribute(v___x_584_);
return v___x_585_;
}
else
{
return v___x_583_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2____boxed(lean_object* v_a_586_){
_start:
{
lean_object* v_res_587_; 
v_res_587_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_();
return v_res_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_env_588_, lean_object* v___y_589_, lean_object* v___y_590_){
_start:
{
lean_object* v___x_592_; 
v___x_592_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___redArg(v_env_588_, v___y_590_);
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_env_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0_spec__0(v_env_593_, v___y_594_, v___y_595_);
lean_dec(v___y_595_);
lean_dec_ref(v___y_594_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_598_, lean_object* v_ext_599_, lean_object* v_k_600_, lean_object* v_v_601_, lean_object* v___y_602_, lean_object* v___y_603_){
_start:
{
lean_object* v___x_605_; 
v___x_605_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg(v_ext_599_, v_k_600_, v_v_601_, v___y_602_, v___y_603_);
return v___x_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_606_, lean_object* v_ext_607_, lean_object* v_k_608_, lean_object* v_v_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0(v_00_u03b1_606_, v_ext_607_, v_k_608_, v_v_609_, v___y_610_, v___y_611_);
lean_dec(v___y_611_);
lean_dec_ref(v___y_610_);
return v_res_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b1_614_, lean_object* v_msg_615_, lean_object* v___y_616_, lean_object* v___y_617_){
_start:
{
lean_object* v___x_619_; 
v___x_619_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___redArg(v_msg_615_, v___y_616_, v___y_617_);
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___boxed(lean_object* v_00_u03b1_620_, lean_object* v_msg_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_){
_start:
{
lean_object* v_res_625_; 
v_res_625_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1(v_00_u03b1_620_, v_msg_621_, v___y_622_, v___y_623_);
lean_dec(v___y_623_);
lean_dec_ref(v___y_622_);
return v_res_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_633_; lean_object* v___x_634_; 
v___x_633_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2_));
v___x_634_ = lp_batteries_Lean_registerNameMapExtension___redArg(v___x_633_);
return v___x_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2____boxed(lean_object* v_a_635_){
_start:
{
lean_object* v_res_636_; 
v_res_636_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2_();
return v_res_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5___redArg(lean_object* v_x_637_, lean_object* v_x_638_){
_start:
{
if (lean_obj_tag(v_x_638_) == 0)
{
return v_x_637_;
}
else
{
lean_object* v_key_639_; lean_object* v_value_640_; lean_object* v_tail_641_; lean_object* v___x_643_; uint8_t v_isShared_644_; uint8_t v_isSharedCheck_664_; 
v_key_639_ = lean_ctor_get(v_x_638_, 0);
v_value_640_ = lean_ctor_get(v_x_638_, 1);
v_tail_641_ = lean_ctor_get(v_x_638_, 2);
v_isSharedCheck_664_ = !lean_is_exclusive(v_x_638_);
if (v_isSharedCheck_664_ == 0)
{
v___x_643_ = v_x_638_;
v_isShared_644_ = v_isSharedCheck_664_;
goto v_resetjp_642_;
}
else
{
lean_inc(v_tail_641_);
lean_inc(v_value_640_);
lean_inc(v_key_639_);
lean_dec(v_x_638_);
v___x_643_ = lean_box(0);
v_isShared_644_ = v_isSharedCheck_664_;
goto v_resetjp_642_;
}
v_resetjp_642_:
{
lean_object* v___x_645_; uint64_t v___x_646_; uint64_t v___x_647_; uint64_t v___x_648_; uint64_t v_fold_649_; uint64_t v___x_650_; uint64_t v___x_651_; uint64_t v___x_652_; size_t v___x_653_; size_t v___x_654_; size_t v___x_655_; size_t v___x_656_; size_t v___x_657_; lean_object* v___x_658_; lean_object* v___x_660_; 
v___x_645_ = lean_array_get_size(v_x_637_);
v___x_646_ = lean_string_hash(v_key_639_);
v___x_647_ = 32ULL;
v___x_648_ = lean_uint64_shift_right(v___x_646_, v___x_647_);
v_fold_649_ = lean_uint64_xor(v___x_646_, v___x_648_);
v___x_650_ = 16ULL;
v___x_651_ = lean_uint64_shift_right(v_fold_649_, v___x_650_);
v___x_652_ = lean_uint64_xor(v_fold_649_, v___x_651_);
v___x_653_ = lean_uint64_to_usize(v___x_652_);
v___x_654_ = lean_usize_of_nat(v___x_645_);
v___x_655_ = ((size_t)1ULL);
v___x_656_ = lean_usize_sub(v___x_654_, v___x_655_);
v___x_657_ = lean_usize_land(v___x_653_, v___x_656_);
v___x_658_ = lean_array_uget_borrowed(v_x_637_, v___x_657_);
lean_inc(v___x_658_);
if (v_isShared_644_ == 0)
{
lean_ctor_set(v___x_643_, 2, v___x_658_);
v___x_660_ = v___x_643_;
goto v_reusejp_659_;
}
else
{
lean_object* v_reuseFailAlloc_663_; 
v_reuseFailAlloc_663_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_663_, 0, v_key_639_);
lean_ctor_set(v_reuseFailAlloc_663_, 1, v_value_640_);
lean_ctor_set(v_reuseFailAlloc_663_, 2, v___x_658_);
v___x_660_ = v_reuseFailAlloc_663_;
goto v_reusejp_659_;
}
v_reusejp_659_:
{
lean_object* v___x_661_; 
v___x_661_ = lean_array_uset(v_x_637_, v___x_657_, v___x_660_);
v_x_637_ = v___x_661_;
v_x_638_ = v_tail_641_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3___redArg(lean_object* v_i_665_, lean_object* v_source_666_, lean_object* v_target_667_){
_start:
{
lean_object* v___x_668_; uint8_t v___x_669_; 
v___x_668_ = lean_array_get_size(v_source_666_);
v___x_669_ = lean_nat_dec_lt(v_i_665_, v___x_668_);
if (v___x_669_ == 0)
{
lean_dec_ref(v_source_666_);
lean_dec(v_i_665_);
return v_target_667_;
}
else
{
lean_object* v_es_670_; lean_object* v___x_671_; lean_object* v_source_672_; lean_object* v_target_673_; lean_object* v___x_674_; lean_object* v___x_675_; 
v_es_670_ = lean_array_fget(v_source_666_, v_i_665_);
v___x_671_ = lean_box(0);
v_source_672_ = lean_array_fset(v_source_666_, v_i_665_, v___x_671_);
v_target_673_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5___redArg(v_target_667_, v_es_670_);
v___x_674_ = lean_unsigned_to_nat(1u);
v___x_675_ = lean_nat_add(v_i_665_, v___x_674_);
lean_dec(v_i_665_);
v_i_665_ = v___x_675_;
v_source_666_ = v_source_672_;
v_target_667_ = v_target_673_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2___redArg(lean_object* v_data_677_){
_start:
{
lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v_nbuckets_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; 
v___x_678_ = lean_array_get_size(v_data_677_);
v___x_679_ = lean_unsigned_to_nat(2u);
v_nbuckets_680_ = lean_nat_mul(v___x_678_, v___x_679_);
v___x_681_ = lean_unsigned_to_nat(0u);
v___x_682_ = lean_box(0);
v___x_683_ = lean_mk_array(v_nbuckets_680_, v___x_682_);
v___x_684_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3___redArg(v___x_681_, v_data_677_, v___x_683_);
return v___x_684_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1___redArg(lean_object* v_a_685_, lean_object* v_x_686_){
_start:
{
if (lean_obj_tag(v_x_686_) == 0)
{
uint8_t v___x_687_; 
v___x_687_ = 0;
return v___x_687_;
}
else
{
lean_object* v_key_688_; lean_object* v_tail_689_; uint8_t v___x_690_; 
v_key_688_ = lean_ctor_get(v_x_686_, 0);
v_tail_689_ = lean_ctor_get(v_x_686_, 2);
v___x_690_ = lean_string_dec_eq(v_key_688_, v_a_685_);
if (v___x_690_ == 0)
{
v_x_686_ = v_tail_689_;
goto _start;
}
else
{
return v___x_690_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_a_692_, lean_object* v_x_693_){
_start:
{
uint8_t v_res_694_; lean_object* v_r_695_; 
v_res_694_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1___redArg(v_a_692_, v_x_693_);
lean_dec(v_x_693_);
lean_dec_ref(v_a_692_);
v_r_695_ = lean_box(v_res_694_);
return v_r_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__3___redArg(lean_object* v_a_696_, lean_object* v_b_697_, lean_object* v_x_698_){
_start:
{
if (lean_obj_tag(v_x_698_) == 0)
{
lean_dec(v_b_697_);
lean_dec_ref(v_a_696_);
return v_x_698_;
}
else
{
lean_object* v_key_699_; lean_object* v_value_700_; lean_object* v_tail_701_; lean_object* v___x_703_; uint8_t v_isShared_704_; uint8_t v_isSharedCheck_713_; 
v_key_699_ = lean_ctor_get(v_x_698_, 0);
v_value_700_ = lean_ctor_get(v_x_698_, 1);
v_tail_701_ = lean_ctor_get(v_x_698_, 2);
v_isSharedCheck_713_ = !lean_is_exclusive(v_x_698_);
if (v_isSharedCheck_713_ == 0)
{
v___x_703_ = v_x_698_;
v_isShared_704_ = v_isSharedCheck_713_;
goto v_resetjp_702_;
}
else
{
lean_inc(v_tail_701_);
lean_inc(v_value_700_);
lean_inc(v_key_699_);
lean_dec(v_x_698_);
v___x_703_ = lean_box(0);
v_isShared_704_ = v_isSharedCheck_713_;
goto v_resetjp_702_;
}
v_resetjp_702_:
{
uint8_t v___x_705_; 
v___x_705_ = lean_string_dec_eq(v_key_699_, v_a_696_);
if (v___x_705_ == 0)
{
lean_object* v___x_706_; lean_object* v___x_708_; 
v___x_706_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__3___redArg(v_a_696_, v_b_697_, v_tail_701_);
if (v_isShared_704_ == 0)
{
lean_ctor_set(v___x_703_, 2, v___x_706_);
v___x_708_ = v___x_703_;
goto v_reusejp_707_;
}
else
{
lean_object* v_reuseFailAlloc_709_; 
v_reuseFailAlloc_709_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_709_, 0, v_key_699_);
lean_ctor_set(v_reuseFailAlloc_709_, 1, v_value_700_);
lean_ctor_set(v_reuseFailAlloc_709_, 2, v___x_706_);
v___x_708_ = v_reuseFailAlloc_709_;
goto v_reusejp_707_;
}
v_reusejp_707_:
{
return v___x_708_;
}
}
else
{
lean_object* v___x_711_; 
lean_dec(v_value_700_);
lean_dec(v_key_699_);
if (v_isShared_704_ == 0)
{
lean_ctor_set(v___x_703_, 1, v_b_697_);
lean_ctor_set(v___x_703_, 0, v_a_696_);
v___x_711_ = v___x_703_;
goto v_reusejp_710_;
}
else
{
lean_object* v_reuseFailAlloc_712_; 
v_reuseFailAlloc_712_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_712_, 0, v_a_696_);
lean_ctor_set(v_reuseFailAlloc_712_, 1, v_b_697_);
lean_ctor_set(v_reuseFailAlloc_712_, 2, v_tail_701_);
v___x_711_ = v_reuseFailAlloc_712_;
goto v_reusejp_710_;
}
v_reusejp_710_:
{
return v___x_711_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0___redArg(lean_object* v_m_714_, lean_object* v_a_715_, lean_object* v_b_716_){
_start:
{
lean_object* v_size_717_; lean_object* v_buckets_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_761_; 
v_size_717_ = lean_ctor_get(v_m_714_, 0);
v_buckets_718_ = lean_ctor_get(v_m_714_, 1);
v_isSharedCheck_761_ = !lean_is_exclusive(v_m_714_);
if (v_isSharedCheck_761_ == 0)
{
v___x_720_ = v_m_714_;
v_isShared_721_ = v_isSharedCheck_761_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_buckets_718_);
lean_inc(v_size_717_);
lean_dec(v_m_714_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_761_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v___x_722_; uint64_t v___x_723_; uint64_t v___x_724_; uint64_t v___x_725_; uint64_t v_fold_726_; uint64_t v___x_727_; uint64_t v___x_728_; uint64_t v___x_729_; size_t v___x_730_; size_t v___x_731_; size_t v___x_732_; size_t v___x_733_; size_t v___x_734_; lean_object* v_bkt_735_; uint8_t v___x_736_; 
v___x_722_ = lean_array_get_size(v_buckets_718_);
v___x_723_ = lean_string_hash(v_a_715_);
v___x_724_ = 32ULL;
v___x_725_ = lean_uint64_shift_right(v___x_723_, v___x_724_);
v_fold_726_ = lean_uint64_xor(v___x_723_, v___x_725_);
v___x_727_ = 16ULL;
v___x_728_ = lean_uint64_shift_right(v_fold_726_, v___x_727_);
v___x_729_ = lean_uint64_xor(v_fold_726_, v___x_728_);
v___x_730_ = lean_uint64_to_usize(v___x_729_);
v___x_731_ = lean_usize_of_nat(v___x_722_);
v___x_732_ = ((size_t)1ULL);
v___x_733_ = lean_usize_sub(v___x_731_, v___x_732_);
v___x_734_ = lean_usize_land(v___x_730_, v___x_733_);
v_bkt_735_ = lean_array_uget_borrowed(v_buckets_718_, v___x_734_);
v___x_736_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1___redArg(v_a_715_, v_bkt_735_);
if (v___x_736_ == 0)
{
lean_object* v___x_737_; lean_object* v_size_x27_738_; lean_object* v___x_739_; lean_object* v_buckets_x27_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; uint8_t v___x_746_; 
v___x_737_ = lean_unsigned_to_nat(1u);
v_size_x27_738_ = lean_nat_add(v_size_717_, v___x_737_);
lean_dec(v_size_717_);
lean_inc(v_bkt_735_);
v___x_739_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_739_, 0, v_a_715_);
lean_ctor_set(v___x_739_, 1, v_b_716_);
lean_ctor_set(v___x_739_, 2, v_bkt_735_);
v_buckets_x27_740_ = lean_array_uset(v_buckets_718_, v___x_734_, v___x_739_);
v___x_741_ = lean_unsigned_to_nat(4u);
v___x_742_ = lean_nat_mul(v_size_x27_738_, v___x_741_);
v___x_743_ = lean_unsigned_to_nat(3u);
v___x_744_ = lean_nat_div(v___x_742_, v___x_743_);
lean_dec(v___x_742_);
v___x_745_ = lean_array_get_size(v_buckets_x27_740_);
v___x_746_ = lean_nat_dec_le(v___x_744_, v___x_745_);
lean_dec(v___x_744_);
if (v___x_746_ == 0)
{
lean_object* v_val_747_; lean_object* v___x_749_; 
v_val_747_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2___redArg(v_buckets_x27_740_);
if (v_isShared_721_ == 0)
{
lean_ctor_set(v___x_720_, 1, v_val_747_);
lean_ctor_set(v___x_720_, 0, v_size_x27_738_);
v___x_749_ = v___x_720_;
goto v_reusejp_748_;
}
else
{
lean_object* v_reuseFailAlloc_750_; 
v_reuseFailAlloc_750_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_750_, 0, v_size_x27_738_);
lean_ctor_set(v_reuseFailAlloc_750_, 1, v_val_747_);
v___x_749_ = v_reuseFailAlloc_750_;
goto v_reusejp_748_;
}
v_reusejp_748_:
{
return v___x_749_;
}
}
else
{
lean_object* v___x_752_; 
if (v_isShared_721_ == 0)
{
lean_ctor_set(v___x_720_, 1, v_buckets_x27_740_);
lean_ctor_set(v___x_720_, 0, v_size_x27_738_);
v___x_752_ = v___x_720_;
goto v_reusejp_751_;
}
else
{
lean_object* v_reuseFailAlloc_753_; 
v_reuseFailAlloc_753_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_753_, 0, v_size_x27_738_);
lean_ctor_set(v_reuseFailAlloc_753_, 1, v_buckets_x27_740_);
v___x_752_ = v_reuseFailAlloc_753_;
goto v_reusejp_751_;
}
v_reusejp_751_:
{
return v___x_752_;
}
}
}
else
{
lean_object* v___x_754_; lean_object* v_buckets_x27_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_759_; 
lean_inc(v_bkt_735_);
v___x_754_ = lean_box(0);
v_buckets_x27_755_ = lean_array_uset(v_buckets_718_, v___x_734_, v___x_754_);
v___x_756_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__3___redArg(v_a_715_, v_b_716_, v_bkt_735_);
v___x_757_ = lean_array_uset(v_buckets_x27_755_, v___x_734_, v___x_756_);
if (v_isShared_721_ == 0)
{
lean_ctor_set(v___x_720_, 1, v___x_757_);
v___x_759_ = v___x_720_;
goto v_reusejp_758_;
}
else
{
lean_object* v_reuseFailAlloc_760_; 
v_reuseFailAlloc_760_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_760_, 0, v_size_717_);
lean_ctor_set(v_reuseFailAlloc_760_, 1, v___x_757_);
v___x_759_ = v_reuseFailAlloc_760_;
goto v_reusejp_758_;
}
v_reusejp_758_:
{
return v___x_759_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1___redArg(lean_object* v_as_x27_762_, lean_object* v_b_763_){
_start:
{
if (lean_obj_tag(v_as_x27_762_) == 0)
{
return v_b_763_;
}
else
{
lean_object* v_head_764_; lean_object* v_tail_765_; lean_object* v_fst_766_; lean_object* v_snd_767_; lean_object* v_r_768_; 
v_head_764_ = lean_ctor_get(v_as_x27_762_, 0);
v_tail_765_ = lean_ctor_get(v_as_x27_762_, 1);
v_fst_766_ = lean_ctor_get(v_head_764_, 0);
v_snd_767_ = lean_ctor_get(v_head_764_, 1);
lean_inc(v_snd_767_);
lean_inc(v_fst_766_);
v_r_768_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0___redArg(v_b_763_, v_fst_766_, v_snd_767_);
v_as_x27_762_ = v_tail_765_;
v_b_763_ = v_r_768_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1___redArg___boxed(lean_object* v_as_x27_770_, lean_object* v_b_771_){
_start:
{
lean_object* v_res_772_; 
v_res_772_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1___redArg(v_as_x27_770_, v_b_771_);
lean_dec(v_as_x27_770_);
return v_res_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0(lean_object* v_m_773_, lean_object* v_l_774_){
_start:
{
lean_object* v___x_775_; 
v___x_775_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1___redArg(v_l_774_, v_m_773_);
return v___x_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0___boxed(lean_object* v_m_776_, lean_object* v_l_777_){
_start:
{
lean_object* v_res_778_; 
v_res_778_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0(v_m_776_, v_l_777_);
lean_dec(v_l_777_);
return v_res_778_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__495(void){
_start:
{
lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; 
v___x_1868_ = lean_box(0);
v___x_1869_ = lean_unsigned_to_nat(16u);
v___x_1870_ = lean_mk_array(v___x_1869_, v___x_1868_);
return v___x_1870_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__496(void){
_start:
{
lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; 
v___x_1871_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__495, &lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__495_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__495);
v___x_1872_ = lean_unsigned_to_nat(0u);
v___x_1873_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1873_, 0, v___x_1872_);
lean_ctor_set(v___x_1873_, 1, v___x_1871_);
return v___x_1873_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__497(void){
_start:
{
lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; 
v___x_1874_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__496, &lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__496_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__496);
v___x_1875_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__494));
v___x_1876_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1___redArg(v___x_1875_, v___x_1874_);
return v___x_1876_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_nameDict(void){
_start:
{
lean_object* v___x_1877_; 
v___x_1877_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__497, &lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__497_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__497);
return v___x_1877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0(lean_object* v_00_u03b2_1878_, lean_object* v_m_1879_, lean_object* v_a_1880_, lean_object* v_b_1881_){
_start:
{
lean_object* v___x_1882_; 
v___x_1882_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0___redArg(v_m_1879_, v_a_1880_, v_b_1881_);
return v___x_1882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1(lean_object* v_as_1883_, lean_object* v_as_x27_1884_, lean_object* v_b_1885_, lean_object* v_a_1886_){
_start:
{
lean_object* v___x_1887_; 
v___x_1887_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1___redArg(v_as_x27_1884_, v_b_1885_);
return v___x_1887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1___boxed(lean_object* v_as_1888_, lean_object* v_as_x27_1889_, lean_object* v_b_1890_, lean_object* v_a_1891_){
_start:
{
lean_object* v_res_1892_; 
v_res_1892_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__1(v_as_1888_, v_as_x27_1889_, v_b_1890_, v_a_1891_);
lean_dec(v_as_x27_1889_);
lean_dec(v_as_1888_);
return v_res_1892_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_1893_, lean_object* v_a_1894_, lean_object* v_x_1895_){
_start:
{
uint8_t v___x_1896_; 
v___x_1896_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1___redArg(v_a_1894_, v_x_1895_);
return v___x_1896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_1897_, lean_object* v_a_1898_, lean_object* v_x_1899_){
_start:
{
uint8_t v_res_1900_; lean_object* v_r_1901_; 
v_res_1900_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__1(v_00_u03b2_1897_, v_a_1898_, v_x_1899_);
lean_dec(v_x_1899_);
lean_dec_ref(v_a_1898_);
v_r_1901_ = lean_box(v_res_1900_);
return v_r_1901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_1902_, lean_object* v_data_1903_){
_start:
{
lean_object* v___x_1904_; 
v___x_1904_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2___redArg(v_data_1903_);
return v___x_1904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__3(lean_object* v_00_u03b2_1905_, lean_object* v_a_1906_, lean_object* v_b_1907_, lean_object* v_x_1908_){
_start:
{
lean_object* v___x_1909_; 
v___x_1909_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__3___redArg(v_a_1906_, v_b_1907_, v_x_1908_);
return v___x_1909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3(lean_object* v_00_u03b2_1910_, lean_object* v_i_1911_, lean_object* v_source_1912_, lean_object* v_target_1913_){
_start:
{
lean_object* v___x_1914_; 
v___x_1914_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3___redArg(v_i_1911_, v_source_1912_, v_target_1913_);
return v___x_1914_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_1915_, lean_object* v_x_1916_, lean_object* v_x_1917_){
_start:
{
lean_object* v___x_1918_; 
v___x_1918_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5___redArg(v_x_1916_, v_x_1917_);
return v___x_1918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0___redArg(lean_object* v_as_x27_1919_, lean_object* v_b_1920_){
_start:
{
if (lean_obj_tag(v_as_x27_1919_) == 0)
{
return v_b_1920_;
}
else
{
lean_object* v_head_1921_; lean_object* v_tail_1922_; lean_object* v_fst_1923_; lean_object* v_snd_1924_; lean_object* v_r_1925_; 
v_head_1921_ = lean_ctor_get(v_as_x27_1919_, 0);
v_tail_1922_ = lean_ctor_get(v_as_x27_1919_, 1);
v_fst_1923_ = lean_ctor_get(v_head_1921_, 0);
v_snd_1924_ = lean_ctor_get(v_head_1921_, 1);
lean_inc(v_snd_1924_);
lean_inc(v_fst_1923_);
v_r_1925_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_nameDict_spec__0_spec__0___redArg(v_b_1920_, v_fst_1923_, v_snd_1924_);
v_as_x27_1919_ = v_tail_1922_;
v_b_1920_ = v_r_1925_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0___redArg___boxed(lean_object* v_as_x27_1927_, lean_object* v_b_1928_){
_start:
{
lean_object* v_res_1929_; 
v_res_1929_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0___redArg(v_as_x27_1927_, v_b_1928_);
lean_dec(v_as_x27_1927_);
return v_res_1929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0(lean_object* v_m_1930_, lean_object* v_l_1931_){
_start:
{
lean_object* v___x_1932_; 
v___x_1932_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0___redArg(v_l_1931_, v_m_1930_);
return v___x_1932_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0___boxed(lean_object* v_m_1933_, lean_object* v_l_1934_){
_start:
{
lean_object* v_res_1935_; 
v_res_1935_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0(v_m_1933_, v_l_1934_);
lean_dec(v_l_1934_);
return v_res_1935_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__104(void){
_start:
{
lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; 
v___x_2144_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__496, &lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__496_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_nameDict___closed__496);
v___x_2145_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__103));
v___x_2146_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0___redArg(v___x_2145_, v___x_2144_);
return v___x_2146_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict(void){
_start:
{
lean_object* v___x_2147_; 
v___x_2147_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__104, &lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__104_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict___closed__104);
return v___x_2147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0(lean_object* v_as_2148_, lean_object* v_as_x27_2149_, lean_object* v_b_2150_, lean_object* v_a_2151_){
_start:
{
lean_object* v___x_2152_; 
v___x_2152_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0___redArg(v_as_x27_2149_, v_b_2150_);
return v___x_2152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0___boxed(lean_object* v_as_2153_, lean_object* v_as_x27_2154_, lean_object* v_b_2155_, lean_object* v_a_2156_){
_start:
{
lean_object* v_res_2157_; 
v_res_2157_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToDual_abbreviationDict_spec__0_spec__0(v_as_2153_, v_as_x27_2154_, v_b_2155_, v_a_2156_);
lean_dec(v_as_x27_2154_);
lean_dec(v_as_2153_);
return v_res_2157_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; 
v___x_2158_ = lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict;
v___x_2159_ = lp_mathlib_Mathlib_Tactic_ToDual_nameDict;
v___x_2160_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2160_, 0, v___x_2159_);
lean_ctor_set(v___x_2160_, 1, v___x_2158_);
return v___x_2160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2162_; lean_object* v___x_2163_; 
v___x_2162_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2_);
v___x_2163_ = lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt(v___x_2162_);
return v___x_2163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2____boxed(lean_object* v_a_2164_){
_start:
{
lean_object* v_res_2165_; 
v_res_2165_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2_();
return v_res_2165_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_data___closed__0(void){
_start:
{
lean_object* v___x_2166_; lean_object* v___x_2167_; 
v___x_2166_ = lp_mathlib_Mathlib_Tactic_ToDual_unfoldBoundaries;
v___x_2167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2167_, 0, v___x_2166_);
return v___x_2167_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_data___closed__2(void){
_start:
{
lean_object* v___x_2170_; uint8_t v___x_2171_; uint8_t v___x_2172_; lean_object* v___x_2173_; lean_object* v___x_2174_; lean_object* v___x_2175_; lean_object* v___x_2176_; lean_object* v___x_2177_; lean_object* v___x_2178_; 
v___x_2170_ = lp_mathlib_Mathlib_Tactic_ToDual_guessNameExt;
v___x_2171_ = 1;
v___x_2172_ = 0;
v___x_2173_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_data___closed__1));
v___x_2174_ = lp_mathlib_Mathlib_Tactic_ToDual_translations;
v___x_2175_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_data___closed__0, &lp_mathlib_Mathlib_Tactic_ToDual_data___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_data___closed__0);
v___x_2176_ = lp_mathlib_Mathlib_Tactic_ToDual_doTranslateAttr;
v___x_2177_ = lp_mathlib_Mathlib_Tactic_ToDual_ignoreArgsAttr;
v___x_2178_ = lean_alloc_ctor(0, 6, 2);
lean_ctor_set(v___x_2178_, 0, v___x_2177_);
lean_ctor_set(v___x_2178_, 1, v___x_2176_);
lean_ctor_set(v___x_2178_, 2, v___x_2175_);
lean_ctor_set(v___x_2178_, 3, v___x_2174_);
lean_ctor_set(v___x_2178_, 4, v___x_2173_);
lean_ctor_set(v___x_2178_, 5, v___x_2170_);
lean_ctor_set_uint8(v___x_2178_, sizeof(void*)*6, v___x_2172_);
lean_ctor_set_uint8(v___x_2178_, sizeof(void*)*6 + 1, v___x_2171_);
return v___x_2178_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToDual_data(void){
_start:
{
lean_object* v___x_2179_; 
v___x_2179_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToDual_data___closed__2, &lp_mathlib_Mathlib_Tactic_ToDual_data___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ToDual_data___closed__2);
return v___x_2179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___redArg(){
_start:
{
lean_object* v___x_2221_; lean_object* v___x_2222_; 
v___x_2221_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2__spec__0___redArg___closed__0);
v___x_2222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2222_, 0, v___x_2221_);
return v___x_2222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___redArg___boxed(lean_object* v___y_2223_){
_start:
{
lean_object* v_res_2224_; 
v_res_2224_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___redArg();
return v_res_2224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0(lean_object* v_00_u03b1_2225_, lean_object* v___y_2226_, lean_object* v___y_2227_){
_start:
{
lean_object* v___x_2229_; 
v___x_2229_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___redArg();
return v___x_2229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___boxed(lean_object* v_00_u03b1_2230_, lean_object* v___y_2231_, lean_object* v___y_2232_, lean_object* v___y_2233_){
_start:
{
lean_object* v_res_2234_; 
v_res_2234_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0(v_00_u03b1_2230_, v___y_2231_, v___y_2232_);
lean_dec(v___y_2232_);
lean_dec_ref(v___y_2231_);
return v_res_2234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1(lean_object* v_x_2235_, lean_object* v_a_2236_, lean_object* v_a_2237_){
_start:
{
lean_object* v___x_2239_; uint8_t v___x_2240_; 
v___x_2239_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast___x3a_x3d___00__closed__1));
lean_inc(v_x_2235_);
v___x_2240_ = l_Lean_Syntax_isOfKind(v_x_2235_, v___x_2239_);
if (v___x_2240_ == 0)
{
lean_object* v___x_2241_; 
lean_dec(v_x_2235_);
v___x_2241_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___redArg();
return v___x_2241_;
}
else
{
lean_object* v___x_2242_; lean_object* v_declName_2243_; lean_object* v___x_2244_; lean_object* v_valStx_2245_; lean_object* v___x_2246_; lean_object* v___x_2247_; 
v___x_2242_ = lean_unsigned_to_nat(1u);
v_declName_2243_ = l_Lean_Syntax_getArg(v_x_2235_, v___x_2242_);
v___x_2244_ = lean_unsigned_to_nat(3u);
v_valStx_2245_ = l_Lean_Syntax_getArg(v_x_2235_, v___x_2244_);
lean_dec(v_x_2235_);
v___x_2246_ = lp_mathlib_Mathlib_Tactic_ToDual_data;
v___x_2247_ = lp_mathlib_Mathlib_Tactic_Translate_elabInsertCast(v_declName_2243_, v_valStx_2245_, v___x_2246_, v_a_2236_, v_a_2237_);
return v___x_2247_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1___boxed(lean_object* v_x_2248_, lean_object* v_a_2249_, lean_object* v_a_2250_, lean_object* v_a_2251_){
_start:
{
lean_object* v_res_2252_; 
v_res_2252_ = lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1(v_x_2248_, v_a_2249_, v_a_2250_);
lean_dec(v_a_2250_);
lean_dec_ref(v_a_2249_);
return v_res_2252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast__fun___x3a_x3d___x2c____1(lean_object* v_x_2290_, lean_object* v_a_2291_, lean_object* v_a_2292_){
_start:
{
lean_object* v___x_2294_; uint8_t v___x_2295_; 
v___x_2294_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__insert__cast__fun___x3a_x3d___x2c___00__closed__1));
lean_inc(v_x_2290_);
v___x_2295_ = l_Lean_Syntax_isOfKind(v_x_2290_, v___x_2294_);
if (v___x_2295_ == 0)
{
lean_object* v___x_2296_; 
lean_dec(v_x_2290_);
v___x_2296_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___redArg();
return v___x_2296_;
}
else
{
lean_object* v___x_2297_; lean_object* v_declName_2298_; lean_object* v___x_2299_; lean_object* v_valStx_u2081_2300_; lean_object* v___x_2301_; lean_object* v_valStx_u2082_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; 
v___x_2297_ = lean_unsigned_to_nat(1u);
v_declName_2298_ = l_Lean_Syntax_getArg(v_x_2290_, v___x_2297_);
v___x_2299_ = lean_unsigned_to_nat(3u);
v_valStx_u2081_2300_ = l_Lean_Syntax_getArg(v_x_2290_, v___x_2299_);
v___x_2301_ = lean_unsigned_to_nat(5u);
v_valStx_u2082_2302_ = l_Lean_Syntax_getArg(v_x_2290_, v___x_2301_);
lean_dec(v_x_2290_);
v___x_2303_ = lp_mathlib_Mathlib_Tactic_ToDual_data;
v___x_2304_ = lp_mathlib_Mathlib_Tactic_Translate_elabInsertCastFun(v_declName_2298_, v_valStx_u2081_2300_, v_valStx_u2082_2302_, v___x_2303_, v_a_2291_, v_a_2292_);
return v___x_2304_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast__fun___x3a_x3d___x2c____1___boxed(lean_object* v_x_2305_, lean_object* v_a_2306_, lean_object* v_a_2307_, lean_object* v_a_2308_){
_start:
{
lean_object* v_res_2309_; 
v_res_2309_ = lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast__fun___x3a_x3d___x2c____1(v_x_2305_, v_a_2306_, v_a_2307_);
lean_dec(v_a_2307_);
lean_dec_ref(v_a_2306_);
return v_res_2309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0___redArg(lean_object* v_category_2310_, lean_object* v_opts_2311_, lean_object* v_act_2312_, lean_object* v_decl_2313_, lean_object* v___y_2314_, lean_object* v___y_2315_){
_start:
{
lean_object* v___x_2317_; lean_object* v___x_2318_; 
lean_inc(v___y_2315_);
lean_inc_ref(v___y_2314_);
v___x_2317_ = lean_apply_2(v_act_2312_, v___y_2314_, v___y_2315_);
v___x_2318_ = l_Lean_profileitIOUnsafe___redArg(v_category_2310_, v_opts_2311_, v___x_2317_, v_decl_2313_);
return v___x_2318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v_category_2319_, lean_object* v_opts_2320_, lean_object* v_act_2321_, lean_object* v_decl_2322_, lean_object* v___y_2323_, lean_object* v___y_2324_, lean_object* v___y_2325_){
_start:
{
lean_object* v_res_2326_; 
v_res_2326_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0___redArg(v_category_2319_, v_opts_2320_, v_act_2321_, v_decl_2322_, v___y_2323_, v___y_2324_);
lean_dec(v___y_2324_);
lean_dec_ref(v___y_2323_);
lean_dec_ref(v_opts_2320_);
lean_dec_ref(v_category_2319_);
return v_res_2326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_2327_, lean_object* v_category_2328_, lean_object* v_opts_2329_, lean_object* v_act_2330_, lean_object* v_decl_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_){
_start:
{
lean_object* v___x_2335_; 
v___x_2335_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0___redArg(v_category_2328_, v_opts_2329_, v_act_2330_, v_decl_2331_, v___y_2332_, v___y_2333_);
return v___x_2335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_2336_, lean_object* v_category_2337_, lean_object* v_opts_2338_, lean_object* v_act_2339_, lean_object* v_decl_2340_, lean_object* v___y_2341_, lean_object* v___y_2342_, lean_object* v___y_2343_){
_start:
{
lean_object* v_res_2344_; 
v_res_2344_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0(v_00_u03b1_2336_, v_category_2337_, v_opts_2338_, v_act_2339_, v_decl_2340_, v___y_2341_, v___y_2342_);
lean_dec(v___y_2342_);
lean_dec_ref(v___y_2341_);
lean_dec_ref(v_opts_2338_);
lean_dec_ref(v_category_2337_);
return v_res_2344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_(lean_object* v_src_2345_, lean_object* v_stx_2346_, uint8_t v_kind_2347_, lean_object* v___y_2348_, lean_object* v___y_2349_){
_start:
{
lean_object* v___x_2351_; 
lean_inc(v_src_2345_);
v___x_2351_ = lp_mathlib_Mathlib_Tactic_Translate_elabTranslationAttr(v_src_2345_, v_stx_2346_, v___y_2348_, v___y_2349_);
if (lean_obj_tag(v___x_2351_) == 0)
{
lean_object* v_a_2352_; lean_object* v___x_2353_; lean_object* v___x_2354_; 
v_a_2352_ = lean_ctor_get(v___x_2351_, 0);
lean_inc(v_a_2352_);
lean_dec_ref_known(v___x_2351_, 1);
v___x_2353_ = lp_mathlib_Mathlib_Tactic_ToDual_data;
v___x_2354_ = lp_mathlib_Mathlib_Tactic_Translate_addTranslationAttr(v___x_2353_, v_src_2345_, v_a_2352_, v_kind_2347_, v___y_2348_, v___y_2349_);
return v___x_2354_;
}
else
{
lean_object* v_a_2355_; lean_object* v___x_2357_; uint8_t v_isShared_2358_; uint8_t v_isSharedCheck_2362_; 
lean_dec(v_src_2345_);
v_a_2355_ = lean_ctor_get(v___x_2351_, 0);
v_isSharedCheck_2362_ = !lean_is_exclusive(v___x_2351_);
if (v_isSharedCheck_2362_ == 0)
{
v___x_2357_ = v___x_2351_;
v_isShared_2358_ = v_isSharedCheck_2362_;
goto v_resetjp_2356_;
}
else
{
lean_inc(v_a_2355_);
lean_dec(v___x_2351_);
v___x_2357_ = lean_box(0);
v_isShared_2358_ = v_isSharedCheck_2362_;
goto v_resetjp_2356_;
}
v_resetjp_2356_:
{
lean_object* v___x_2360_; 
if (v_isShared_2358_ == 0)
{
v___x_2360_ = v___x_2357_;
goto v_reusejp_2359_;
}
else
{
lean_object* v_reuseFailAlloc_2361_; 
v_reuseFailAlloc_2361_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2361_, 0, v_a_2355_);
v___x_2360_ = v_reuseFailAlloc_2361_;
goto v_reusejp_2359_;
}
v_reusejp_2359_:
{
return v___x_2360_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed(lean_object* v_src_2363_, lean_object* v_stx_2364_, lean_object* v_kind_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_){
_start:
{
uint8_t v_kind_boxed_2369_; lean_object* v_res_2370_; 
v_kind_boxed_2369_ = lean_unbox(v_kind_2365_);
v_res_2370_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_(v_src_2363_, v_stx_2364_, v_kind_boxed_2369_, v___y_2366_, v___y_2367_);
lean_dec(v___y_2367_);
lean_dec_ref(v___y_2366_);
return v_res_2370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_(lean_object* v___x_2371_, lean_object* v___x_2372_, lean_object* v_src_2373_, lean_object* v_stx_2374_, uint8_t v_kind_2375_, lean_object* v___y_2376_, lean_object* v___y_2377_){
_start:
{
lean_object* v_options_2379_; lean_object* v___x_2380_; lean_object* v___f_2381_; lean_object* v___x_2382_; 
v_options_2379_ = lean_ctor_get(v___y_2376_, 2);
v___x_2380_ = lean_box(v_kind_2375_);
v___f_2381_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed), 6, 3);
lean_closure_set(v___f_2381_, 0, v_src_2373_);
lean_closure_set(v___f_2381_, 1, v_stx_2374_);
lean_closure_set(v___f_2381_, 2, v___x_2380_);
v___x_2382_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2__spec__0___redArg(v___x_2371_, v_options_2379_, v___f_2381_, v___x_2372_, v___y_2376_, v___y_2377_);
if (lean_obj_tag(v___x_2382_) == 0)
{
lean_object* v___x_2384_; uint8_t v_isShared_2385_; uint8_t v_isSharedCheck_2390_; 
v_isSharedCheck_2390_ = !lean_is_exclusive(v___x_2382_);
if (v_isSharedCheck_2390_ == 0)
{
lean_object* v_unused_2391_; 
v_unused_2391_ = lean_ctor_get(v___x_2382_, 0);
lean_dec(v_unused_2391_);
v___x_2384_ = v___x_2382_;
v_isShared_2385_ = v_isSharedCheck_2390_;
goto v_resetjp_2383_;
}
else
{
lean_dec(v___x_2382_);
v___x_2384_ = lean_box(0);
v_isShared_2385_ = v_isSharedCheck_2390_;
goto v_resetjp_2383_;
}
v_resetjp_2383_:
{
lean_object* v___x_2386_; lean_object* v___x_2388_; 
v___x_2386_ = lean_box(0);
if (v_isShared_2385_ == 0)
{
lean_ctor_set(v___x_2384_, 0, v___x_2386_);
v___x_2388_ = v___x_2384_;
goto v_reusejp_2387_;
}
else
{
lean_object* v_reuseFailAlloc_2389_; 
v_reuseFailAlloc_2389_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2389_, 0, v___x_2386_);
v___x_2388_ = v_reuseFailAlloc_2389_;
goto v_reusejp_2387_;
}
v_reusejp_2387_:
{
return v___x_2388_;
}
}
}
else
{
lean_object* v_a_2392_; lean_object* v___x_2394_; uint8_t v_isShared_2395_; uint8_t v_isSharedCheck_2399_; 
v_a_2392_ = lean_ctor_get(v___x_2382_, 0);
v_isSharedCheck_2399_ = !lean_is_exclusive(v___x_2382_);
if (v_isSharedCheck_2399_ == 0)
{
v___x_2394_ = v___x_2382_;
v_isShared_2395_ = v_isSharedCheck_2399_;
goto v_resetjp_2393_;
}
else
{
lean_inc(v_a_2392_);
lean_dec(v___x_2382_);
v___x_2394_ = lean_box(0);
v_isShared_2395_ = v_isSharedCheck_2399_;
goto v_resetjp_2393_;
}
v_resetjp_2393_:
{
lean_object* v___x_2397_; 
if (v_isShared_2395_ == 0)
{
v___x_2397_ = v___x_2394_;
goto v_reusejp_2396_;
}
else
{
lean_object* v_reuseFailAlloc_2398_; 
v_reuseFailAlloc_2398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2398_, 0, v_a_2392_);
v___x_2397_ = v_reuseFailAlloc_2398_;
goto v_reusejp_2396_;
}
v_reusejp_2396_:
{
return v___x_2397_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed(lean_object* v___x_2400_, lean_object* v___x_2401_, lean_object* v_src_2402_, lean_object* v_stx_2403_, lean_object* v_kind_2404_, lean_object* v___y_2405_, lean_object* v___y_2406_, lean_object* v___y_2407_){
_start:
{
uint8_t v_kind_boxed_2408_; lean_object* v_res_2409_; 
v_kind_boxed_2408_ = lean_unbox(v_kind_2404_);
v_res_2409_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_(v___x_2400_, v___x_2401_, v_src_2402_, v_stx_2403_, v_kind_boxed_2408_, v___y_2405_, v___y_2406_);
lean_dec(v___y_2406_);
lean_dec_ref(v___y_2405_);
lean_dec_ref(v___x_2400_);
return v_res_2409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_(lean_object* v___x_2410_, lean_object* v_decl_2411_, lean_object* v___y_2412_, lean_object* v___y_2413_){
_start:
{
lean_object* v___x_2415_; lean_object* v___x_2416_; lean_object* v___x_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; lean_object* v___x_2420_; 
v___x_2415_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_2416_ = l_Lean_MessageData_ofName(v___x_2410_);
v___x_2417_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2417_, 0, v___x_2415_);
lean_ctor_set(v___x_2417_, 1, v___x_2416_);
v___x_2418_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_);
v___x_2419_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2419_, 0, v___x_2417_);
lean_ctor_set(v___x_2419_, 1, v___x_2418_);
v___x_2420_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1___redArg(v___x_2419_, v___y_2412_, v___y_2413_);
return v___x_2420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed(lean_object* v___x_2421_, lean_object* v_decl_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_){
_start:
{
lean_object* v_res_2426_; 
v_res_2426_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_(v___x_2421_, v_decl_2422_, v___y_2423_, v___y_2424_);
lean_dec(v___y_2424_);
lean_dec_ref(v___y_2423_);
lean_dec(v_decl_2422_);
return v_res_2426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2455_; lean_object* v___x_2456_; 
v___x_2455_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_));
v___x_2456_ = l_Lean_registerBuiltinAttribute(v___x_2455_);
return v___x_2456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2____boxed(lean_object* v_a_2457_){
_start:
{
lean_object* v_res_2458_; 
v_res_2458_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_();
return v_res_2458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1___redArg(lean_object* v_env_2481_, lean_object* v___y_2482_){
_start:
{
lean_object* v___x_2484_; lean_object* v_messages_2485_; lean_object* v_scopes_2486_; lean_object* v_usedQuotCtxts_2487_; lean_object* v_nextMacroScope_2488_; lean_object* v_maxRecDepth_2489_; lean_object* v_ngen_2490_; lean_object* v_auxDeclNGen_2491_; lean_object* v_infoState_2492_; lean_object* v_traceState_2493_; lean_object* v_snapshotTasks_2494_; lean_object* v_prevLinterStates_2495_; lean_object* v___x_2497_; uint8_t v_isShared_2498_; uint8_t v_isSharedCheck_2505_; 
v___x_2484_ = lean_st_ref_take(v___y_2482_);
v_messages_2485_ = lean_ctor_get(v___x_2484_, 1);
v_scopes_2486_ = lean_ctor_get(v___x_2484_, 2);
v_usedQuotCtxts_2487_ = lean_ctor_get(v___x_2484_, 3);
v_nextMacroScope_2488_ = lean_ctor_get(v___x_2484_, 4);
v_maxRecDepth_2489_ = lean_ctor_get(v___x_2484_, 5);
v_ngen_2490_ = lean_ctor_get(v___x_2484_, 6);
v_auxDeclNGen_2491_ = lean_ctor_get(v___x_2484_, 7);
v_infoState_2492_ = lean_ctor_get(v___x_2484_, 8);
v_traceState_2493_ = lean_ctor_get(v___x_2484_, 9);
v_snapshotTasks_2494_ = lean_ctor_get(v___x_2484_, 10);
v_prevLinterStates_2495_ = lean_ctor_get(v___x_2484_, 11);
v_isSharedCheck_2505_ = !lean_is_exclusive(v___x_2484_);
if (v_isSharedCheck_2505_ == 0)
{
lean_object* v_unused_2506_; 
v_unused_2506_ = lean_ctor_get(v___x_2484_, 0);
lean_dec(v_unused_2506_);
v___x_2497_ = v___x_2484_;
v_isShared_2498_ = v_isSharedCheck_2505_;
goto v_resetjp_2496_;
}
else
{
lean_inc(v_prevLinterStates_2495_);
lean_inc(v_snapshotTasks_2494_);
lean_inc(v_traceState_2493_);
lean_inc(v_infoState_2492_);
lean_inc(v_auxDeclNGen_2491_);
lean_inc(v_ngen_2490_);
lean_inc(v_maxRecDepth_2489_);
lean_inc(v_nextMacroScope_2488_);
lean_inc(v_usedQuotCtxts_2487_);
lean_inc(v_scopes_2486_);
lean_inc(v_messages_2485_);
lean_dec(v___x_2484_);
v___x_2497_ = lean_box(0);
v_isShared_2498_ = v_isSharedCheck_2505_;
goto v_resetjp_2496_;
}
v_resetjp_2496_:
{
lean_object* v___x_2500_; 
if (v_isShared_2498_ == 0)
{
lean_ctor_set(v___x_2497_, 0, v_env_2481_);
v___x_2500_ = v___x_2497_;
goto v_reusejp_2499_;
}
else
{
lean_object* v_reuseFailAlloc_2504_; 
v_reuseFailAlloc_2504_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_2504_, 0, v_env_2481_);
lean_ctor_set(v_reuseFailAlloc_2504_, 1, v_messages_2485_);
lean_ctor_set(v_reuseFailAlloc_2504_, 2, v_scopes_2486_);
lean_ctor_set(v_reuseFailAlloc_2504_, 3, v_usedQuotCtxts_2487_);
lean_ctor_set(v_reuseFailAlloc_2504_, 4, v_nextMacroScope_2488_);
lean_ctor_set(v_reuseFailAlloc_2504_, 5, v_maxRecDepth_2489_);
lean_ctor_set(v_reuseFailAlloc_2504_, 6, v_ngen_2490_);
lean_ctor_set(v_reuseFailAlloc_2504_, 7, v_auxDeclNGen_2491_);
lean_ctor_set(v_reuseFailAlloc_2504_, 8, v_infoState_2492_);
lean_ctor_set(v_reuseFailAlloc_2504_, 9, v_traceState_2493_);
lean_ctor_set(v_reuseFailAlloc_2504_, 10, v_snapshotTasks_2494_);
lean_ctor_set(v_reuseFailAlloc_2504_, 11, v_prevLinterStates_2495_);
v___x_2500_ = v_reuseFailAlloc_2504_;
goto v_reusejp_2499_;
}
v_reusejp_2499_:
{
lean_object* v___x_2501_; lean_object* v___x_2502_; lean_object* v___x_2503_; 
v___x_2501_ = lean_st_ref_set(v___y_2482_, v___x_2500_);
v___x_2502_ = lean_box(0);
v___x_2503_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2503_, 0, v___x_2502_);
return v___x_2503_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1___redArg___boxed(lean_object* v_env_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_){
_start:
{
lean_object* v_res_2510_; 
v_res_2510_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1___redArg(v_env_2507_, v___y_2508_);
lean_dec(v___y_2508_);
return v_res_2510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1___redArg(lean_object* v_msgData_2511_, lean_object* v___y_2512_){
_start:
{
lean_object* v___x_2514_; lean_object* v_env_2515_; lean_object* v___x_2516_; lean_object* v_scopes_2517_; lean_object* v___x_2518_; lean_object* v___x_2519_; lean_object* v_opts_2520_; lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; lean_object* v___x_2527_; 
v___x_2514_ = lean_st_ref_get(v___y_2512_);
v_env_2515_ = lean_ctor_get(v___x_2514_, 0);
lean_inc_ref(v_env_2515_);
lean_dec(v___x_2514_);
v___x_2516_ = lean_st_ref_get(v___y_2512_);
v_scopes_2517_ = lean_ctor_get(v___x_2516_, 2);
lean_inc(v_scopes_2517_);
lean_dec(v___x_2516_);
v___x_2518_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2519_ = l_List_head_x21___redArg(v___x_2518_, v_scopes_2517_);
lean_dec(v_scopes_2517_);
v_opts_2520_ = lean_ctor_get(v___x_2519_, 1);
lean_inc_ref(v_opts_2520_);
lean_dec(v___x_2519_);
v___x_2521_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__2);
v___x_2522_ = lean_unsigned_to_nat(32u);
v___x_2523_ = lean_mk_empty_array_with_capacity(v___x_2522_);
lean_dec_ref(v___x_2523_);
v___x_2524_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__1_spec__2___closed__5);
v___x_2525_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2525_, 0, v_env_2515_);
lean_ctor_set(v___x_2525_, 1, v___x_2521_);
lean_ctor_set(v___x_2525_, 2, v___x_2524_);
lean_ctor_set(v___x_2525_, 3, v_opts_2520_);
v___x_2526_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2526_, 0, v___x_2525_);
lean_ctor_set(v___x_2526_, 1, v_msgData_2511_);
v___x_2527_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2527_, 0, v___x_2526_);
return v___x_2527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_msgData_2528_, lean_object* v___y_2529_, lean_object* v___y_2530_){
_start:
{
lean_object* v_res_2531_; 
v_res_2531_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1___redArg(v_msgData_2528_, v___y_2529_);
lean_dec(v___y_2529_);
return v_res_2531_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__0(void){
_start:
{
lean_object* v___x_2532_; lean_object* v___x_2533_; 
v___x_2532_ = lean_box(1);
v___x_2533_ = l_Lean_MessageData_ofFormat(v___x_2532_);
return v___x_2533_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__3(void){
_start:
{
lean_object* v___x_2537_; lean_object* v___x_2538_; 
v___x_2537_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__2));
v___x_2538_ = l_Lean_MessageData_ofFormat(v___x_2537_);
return v___x_2538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5(lean_object* v_x_2539_, lean_object* v_x_2540_){
_start:
{
if (lean_obj_tag(v_x_2540_) == 0)
{
return v_x_2539_;
}
else
{
lean_object* v_head_2541_; lean_object* v_tail_2542_; lean_object* v___x_2544_; uint8_t v_isShared_2545_; uint8_t v_isSharedCheck_2564_; 
v_head_2541_ = lean_ctor_get(v_x_2540_, 0);
v_tail_2542_ = lean_ctor_get(v_x_2540_, 1);
v_isSharedCheck_2564_ = !lean_is_exclusive(v_x_2540_);
if (v_isSharedCheck_2564_ == 0)
{
v___x_2544_ = v_x_2540_;
v_isShared_2545_ = v_isSharedCheck_2564_;
goto v_resetjp_2543_;
}
else
{
lean_inc(v_tail_2542_);
lean_inc(v_head_2541_);
lean_dec(v_x_2540_);
v___x_2544_ = lean_box(0);
v_isShared_2545_ = v_isSharedCheck_2564_;
goto v_resetjp_2543_;
}
v_resetjp_2543_:
{
lean_object* v_before_2546_; lean_object* v___x_2548_; uint8_t v_isShared_2549_; uint8_t v_isSharedCheck_2562_; 
v_before_2546_ = lean_ctor_get(v_head_2541_, 0);
v_isSharedCheck_2562_ = !lean_is_exclusive(v_head_2541_);
if (v_isSharedCheck_2562_ == 0)
{
lean_object* v_unused_2563_; 
v_unused_2563_ = lean_ctor_get(v_head_2541_, 1);
lean_dec(v_unused_2563_);
v___x_2548_ = v_head_2541_;
v_isShared_2549_ = v_isSharedCheck_2562_;
goto v_resetjp_2547_;
}
else
{
lean_inc(v_before_2546_);
lean_dec(v_head_2541_);
v___x_2548_ = lean_box(0);
v_isShared_2549_ = v_isSharedCheck_2562_;
goto v_resetjp_2547_;
}
v_resetjp_2547_:
{
lean_object* v___x_2550_; lean_object* v___x_2552_; 
v___x_2550_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__0);
if (v_isShared_2549_ == 0)
{
lean_ctor_set_tag(v___x_2548_, 7);
lean_ctor_set(v___x_2548_, 1, v___x_2550_);
lean_ctor_set(v___x_2548_, 0, v_x_2539_);
v___x_2552_ = v___x_2548_;
goto v_reusejp_2551_;
}
else
{
lean_object* v_reuseFailAlloc_2561_; 
v_reuseFailAlloc_2561_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2561_, 0, v_x_2539_);
lean_ctor_set(v_reuseFailAlloc_2561_, 1, v___x_2550_);
v___x_2552_ = v_reuseFailAlloc_2561_;
goto v_reusejp_2551_;
}
v_reusejp_2551_:
{
lean_object* v___x_2553_; lean_object* v___x_2555_; 
v___x_2553_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__3);
if (v_isShared_2545_ == 0)
{
lean_ctor_set_tag(v___x_2544_, 7);
lean_ctor_set(v___x_2544_, 1, v___x_2553_);
lean_ctor_set(v___x_2544_, 0, v___x_2552_);
v___x_2555_ = v___x_2544_;
goto v_reusejp_2554_;
}
else
{
lean_object* v_reuseFailAlloc_2560_; 
v_reuseFailAlloc_2560_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2560_, 0, v___x_2552_);
lean_ctor_set(v_reuseFailAlloc_2560_, 1, v___x_2553_);
v___x_2555_ = v_reuseFailAlloc_2560_;
goto v_reusejp_2554_;
}
v_reusejp_2554_:
{
lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; 
v___x_2556_ = l_Lean_MessageData_ofSyntax(v_before_2546_);
v___x_2557_ = l_Lean_indentD(v___x_2556_);
v___x_2558_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2558_, 0, v___x_2555_);
lean_ctor_set(v___x_2558_, 1, v___x_2557_);
v_x_2539_ = v___x_2558_;
v_x_2540_ = v_tail_2542_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__4(lean_object* v_opts_2565_, lean_object* v_opt_2566_){
_start:
{
lean_object* v_name_2567_; lean_object* v_defValue_2568_; lean_object* v_map_2569_; lean_object* v___x_2570_; 
v_name_2567_ = lean_ctor_get(v_opt_2566_, 0);
v_defValue_2568_ = lean_ctor_get(v_opt_2566_, 1);
v_map_2569_ = lean_ctor_get(v_opts_2565_, 0);
v___x_2570_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2569_, v_name_2567_);
if (lean_obj_tag(v___x_2570_) == 0)
{
uint8_t v___x_2571_; 
v___x_2571_ = lean_unbox(v_defValue_2568_);
return v___x_2571_;
}
else
{
lean_object* v_val_2572_; 
v_val_2572_ = lean_ctor_get(v___x_2570_, 0);
lean_inc(v_val_2572_);
lean_dec_ref_known(v___x_2570_, 1);
if (lean_obj_tag(v_val_2572_) == 1)
{
uint8_t v_v_2573_; 
v_v_2573_ = lean_ctor_get_uint8(v_val_2572_, 0);
lean_dec_ref_known(v_val_2572_, 0);
return v_v_2573_;
}
else
{
uint8_t v___x_2574_; 
lean_dec(v_val_2572_);
v___x_2574_ = lean_unbox(v_defValue_2568_);
return v___x_2574_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v_opts_2575_, lean_object* v_opt_2576_){
_start:
{
uint8_t v_res_2577_; lean_object* v_r_2578_; 
v_res_2577_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__4(v_opts_2575_, v_opt_2576_);
lean_dec_ref(v_opt_2576_);
lean_dec_ref(v_opts_2575_);
v_r_2578_ = lean_box(v_res_2577_);
return v_r_2578_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_2582_; lean_object* v___x_2583_; 
v___x_2582_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__1));
v___x_2583_ = l_Lean_MessageData_ofFormat(v___x_2582_);
return v___x_2583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg(lean_object* v_msgData_2584_, lean_object* v_macroStack_2585_, lean_object* v___y_2586_){
_start:
{
lean_object* v___x_2588_; lean_object* v_scopes_2589_; lean_object* v___x_2590_; lean_object* v___x_2591_; lean_object* v_opts_2592_; lean_object* v___x_2593_; uint8_t v___x_2594_; 
v___x_2588_ = lean_st_ref_get(v___y_2586_);
v_scopes_2589_ = lean_ctor_get(v___x_2588_, 2);
lean_inc(v_scopes_2589_);
lean_dec(v___x_2588_);
v___x_2590_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2591_ = l_List_head_x21___redArg(v___x_2590_, v_scopes_2589_);
lean_dec(v_scopes_2589_);
v_opts_2592_ = lean_ctor_get(v___x_2591_, 1);
lean_inc_ref(v_opts_2592_);
lean_dec(v___x_2591_);
v___x_2593_ = l_Lean_Elab_pp_macroStack;
v___x_2594_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__4(v_opts_2592_, v___x_2593_);
lean_dec_ref(v_opts_2592_);
if (v___x_2594_ == 0)
{
lean_object* v___x_2595_; 
lean_dec(v_macroStack_2585_);
v___x_2595_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2595_, 0, v_msgData_2584_);
return v___x_2595_;
}
else
{
if (lean_obj_tag(v_macroStack_2585_) == 0)
{
lean_object* v___x_2596_; 
v___x_2596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2596_, 0, v_msgData_2584_);
return v___x_2596_;
}
else
{
lean_object* v_head_2597_; lean_object* v_after_2598_; lean_object* v___x_2600_; uint8_t v_isShared_2601_; uint8_t v_isSharedCheck_2613_; 
v_head_2597_ = lean_ctor_get(v_macroStack_2585_, 0);
lean_inc(v_head_2597_);
v_after_2598_ = lean_ctor_get(v_head_2597_, 1);
v_isSharedCheck_2613_ = !lean_is_exclusive(v_head_2597_);
if (v_isSharedCheck_2613_ == 0)
{
lean_object* v_unused_2614_; 
v_unused_2614_ = lean_ctor_get(v_head_2597_, 0);
lean_dec(v_unused_2614_);
v___x_2600_ = v_head_2597_;
v_isShared_2601_ = v_isSharedCheck_2613_;
goto v_resetjp_2599_;
}
else
{
lean_inc(v_after_2598_);
lean_dec(v_head_2597_);
v___x_2600_ = lean_box(0);
v_isShared_2601_ = v_isSharedCheck_2613_;
goto v_resetjp_2599_;
}
v_resetjp_2599_:
{
lean_object* v___x_2602_; lean_object* v___x_2604_; 
v___x_2602_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5___closed__0);
if (v_isShared_2601_ == 0)
{
lean_ctor_set_tag(v___x_2600_, 7);
lean_ctor_set(v___x_2600_, 1, v___x_2602_);
lean_ctor_set(v___x_2600_, 0, v_msgData_2584_);
v___x_2604_ = v___x_2600_;
goto v_reusejp_2603_;
}
else
{
lean_object* v_reuseFailAlloc_2612_; 
v_reuseFailAlloc_2612_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2612_, 0, v_msgData_2584_);
lean_ctor_set(v_reuseFailAlloc_2612_, 1, v___x_2602_);
v___x_2604_ = v_reuseFailAlloc_2612_;
goto v_reusejp_2603_;
}
v_reusejp_2603_:
{
lean_object* v___x_2605_; lean_object* v___x_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; lean_object* v_msgData_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; 
v___x_2605_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___closed__2);
v___x_2606_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2606_, 0, v___x_2604_);
lean_ctor_set(v___x_2606_, 1, v___x_2605_);
v___x_2607_ = l_Lean_MessageData_ofSyntax(v_after_2598_);
v___x_2608_ = l_Lean_indentD(v___x_2607_);
v_msgData_2609_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_2609_, 0, v___x_2606_);
lean_ctor_set(v_msgData_2609_, 1, v___x_2608_);
v___x_2610_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2_spec__5(v_msgData_2609_, v_macroStack_2585_);
v___x_2611_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2611_, 0, v___x_2610_);
return v___x_2611_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_msgData_2615_, lean_object* v_macroStack_2616_, lean_object* v___y_2617_, lean_object* v___y_2618_){
_start:
{
lean_object* v_res_2619_; 
v_res_2619_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg(v_msgData_2615_, v_macroStack_2616_, v___y_2617_);
lean_dec(v___y_2617_);
return v_res_2619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0___redArg(lean_object* v_msg_2620_, lean_object* v___y_2621_, lean_object* v___y_2622_){
_start:
{
lean_object* v___x_2624_; 
v___x_2624_ = l_Lean_Elab_Command_getRef___redArg(v___y_2621_);
if (lean_obj_tag(v___x_2624_) == 0)
{
lean_object* v_a_2625_; lean_object* v_macroStack_2626_; lean_object* v___x_2627_; lean_object* v_a_2628_; lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v_a_2631_; lean_object* v___x_2633_; uint8_t v_isShared_2634_; uint8_t v_isSharedCheck_2639_; 
v_a_2625_ = lean_ctor_get(v___x_2624_, 0);
lean_inc(v_a_2625_);
lean_dec_ref_known(v___x_2624_, 1);
v_macroStack_2626_ = lean_ctor_get(v___y_2621_, 4);
v___x_2627_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1___redArg(v_msg_2620_, v___y_2622_);
v_a_2628_ = lean_ctor_get(v___x_2627_, 0);
lean_inc(v_a_2628_);
lean_dec_ref(v___x_2627_);
v___x_2629_ = l_Lean_Elab_getBetterRef(v_a_2625_, v_macroStack_2626_);
lean_dec(v_a_2625_);
lean_inc(v_macroStack_2626_);
v___x_2630_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg(v_a_2628_, v_macroStack_2626_, v___y_2622_);
v_a_2631_ = lean_ctor_get(v___x_2630_, 0);
v_isSharedCheck_2639_ = !lean_is_exclusive(v___x_2630_);
if (v_isSharedCheck_2639_ == 0)
{
v___x_2633_ = v___x_2630_;
v_isShared_2634_ = v_isSharedCheck_2639_;
goto v_resetjp_2632_;
}
else
{
lean_inc(v_a_2631_);
lean_dec(v___x_2630_);
v___x_2633_ = lean_box(0);
v_isShared_2634_ = v_isSharedCheck_2639_;
goto v_resetjp_2632_;
}
v_resetjp_2632_:
{
lean_object* v___x_2635_; lean_object* v___x_2637_; 
v___x_2635_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2635_, 0, v___x_2629_);
lean_ctor_set(v___x_2635_, 1, v_a_2631_);
if (v_isShared_2634_ == 0)
{
lean_ctor_set_tag(v___x_2633_, 1);
lean_ctor_set(v___x_2633_, 0, v___x_2635_);
v___x_2637_ = v___x_2633_;
goto v_reusejp_2636_;
}
else
{
lean_object* v_reuseFailAlloc_2638_; 
v_reuseFailAlloc_2638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2638_, 0, v___x_2635_);
v___x_2637_ = v_reuseFailAlloc_2638_;
goto v_reusejp_2636_;
}
v_reusejp_2636_:
{
return v___x_2637_;
}
}
}
else
{
lean_object* v_a_2640_; lean_object* v___x_2642_; uint8_t v_isShared_2643_; uint8_t v_isSharedCheck_2647_; 
lean_dec_ref(v_msg_2620_);
v_a_2640_ = lean_ctor_get(v___x_2624_, 0);
v_isSharedCheck_2647_ = !lean_is_exclusive(v___x_2624_);
if (v_isSharedCheck_2647_ == 0)
{
v___x_2642_ = v___x_2624_;
v_isShared_2643_ = v_isSharedCheck_2647_;
goto v_resetjp_2641_;
}
else
{
lean_inc(v_a_2640_);
lean_dec(v___x_2624_);
v___x_2642_ = lean_box(0);
v_isShared_2643_ = v_isSharedCheck_2647_;
goto v_resetjp_2641_;
}
v_resetjp_2641_:
{
lean_object* v___x_2645_; 
if (v_isShared_2643_ == 0)
{
v___x_2645_ = v___x_2642_;
goto v_reusejp_2644_;
}
else
{
lean_object* v_reuseFailAlloc_2646_; 
v_reuseFailAlloc_2646_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2646_, 0, v_a_2640_);
v___x_2645_ = v_reuseFailAlloc_2646_;
goto v_reusejp_2644_;
}
v_reusejp_2644_:
{
return v___x_2645_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0___redArg___boxed(lean_object* v_msg_2648_, lean_object* v___y_2649_, lean_object* v___y_2650_, lean_object* v___y_2651_){
_start:
{
lean_object* v_res_2652_; 
v_res_2652_ = lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0___redArg(v_msg_2648_, v___y_2649_, v___y_2650_);
lean_dec(v___y_2650_);
lean_dec_ref(v___y_2649_);
return v_res_2652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0___redArg(lean_object* v_ext_2653_, lean_object* v_k_2654_, lean_object* v_v_2655_, lean_object* v___y_2656_, lean_object* v___y_2657_){
_start:
{
lean_object* v___x_2659_; lean_object* v_env_2660_; lean_object* v___x_2661_; 
v___x_2659_ = lean_st_ref_get(v___y_2657_);
v_env_2660_ = lean_ctor_get(v___x_2659_, 0);
lean_inc_ref(v_env_2660_);
lean_dec(v___x_2659_);
v___x_2661_ = lp_batteries_Lean_NameMapExtension_find_x3f___redArg(v_ext_2653_, v_env_2660_, v_k_2654_);
if (lean_obj_tag(v___x_2661_) == 1)
{
lean_object* v_name_2662_; lean_object* v___x_2663_; lean_object* v___x_2664_; lean_object* v___x_2665_; lean_object* v___x_2666_; lean_object* v___x_2667_; lean_object* v___x_2668_; lean_object* v___x_2669_; lean_object* v___x_2670_; 
lean_dec_ref_known(v___x_2661_, 1);
lean_dec(v_v_2655_);
v_name_2662_ = lean_ctor_get(v_ext_2653_, 1);
lean_inc(v_name_2662_);
lean_dec_ref(v_ext_2653_);
v___x_2663_ = lean_obj_once(&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__1, &lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__1);
v___x_2664_ = l_Lean_MessageData_ofName(v_name_2662_);
v___x_2665_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2665_, 0, v___x_2663_);
lean_ctor_set(v___x_2665_, 1, v___x_2664_);
v___x_2666_ = lean_obj_once(&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__3, &lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__3_once, _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2__spec__0___redArg___closed__3);
v___x_2667_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2667_, 0, v___x_2665_);
lean_ctor_set(v___x_2667_, 1, v___x_2666_);
v___x_2668_ = l_Lean_MessageData_ofName(v_k_2654_);
v___x_2669_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2669_, 0, v___x_2667_);
lean_ctor_set(v___x_2669_, 1, v___x_2668_);
v___x_2670_ = lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0___redArg(v___x_2669_, v___y_2656_, v___y_2657_);
return v___x_2670_;
}
else
{
lean_object* v___x_2671_; lean_object* v_toEnvExtension_2672_; lean_object* v_env_2673_; lean_object* v_asyncMode_2674_; lean_object* v___x_2675_; lean_object* v___x_2676_; lean_object* v___x_2677_; lean_object* v___x_2678_; 
lean_dec(v___x_2661_);
v___x_2671_ = lean_st_ref_get(v___y_2657_);
v_toEnvExtension_2672_ = lean_ctor_get(v_ext_2653_, 0);
v_env_2673_ = lean_ctor_get(v___x_2671_, 0);
lean_inc_ref(v_env_2673_);
lean_dec(v___x_2671_);
v_asyncMode_2674_ = lean_ctor_get(v_toEnvExtension_2672_, 2);
lean_inc(v_asyncMode_2674_);
v___x_2675_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2675_, 0, v_k_2654_);
lean_ctor_set(v___x_2675_, 1, v_v_2655_);
v___x_2676_ = lean_box(0);
v___x_2677_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v_ext_2653_, v_env_2673_, v___x_2675_, v_asyncMode_2674_, v___x_2676_);
lean_dec(v_asyncMode_2674_);
v___x_2678_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1___redArg(v___x_2677_, v___y_2657_);
return v___x_2678_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0___redArg___boxed(lean_object* v_ext_2679_, lean_object* v_k_2680_, lean_object* v_v_2681_, lean_object* v___y_2682_, lean_object* v___y_2683_, lean_object* v___y_2684_){
_start:
{
lean_object* v_res_2685_; 
v_res_2685_ = lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0___redArg(v_ext_2679_, v_k_2680_, v_v_2681_, v___y_2682_, v___y_2683_);
lean_dec(v___y_2683_);
lean_dec_ref(v___y_2682_);
return v_res_2685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1(lean_object* v_x_2696_, lean_object* v_a_2697_, lean_object* v_a_2698_){
_start:
{
lean_object* v___x_2700_; uint8_t v___x_2701_; 
v___x_2700_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_commandInsert__to__dual__translation_____00__closed__1));
lean_inc(v_x_2696_);
v___x_2701_ = l_Lean_Syntax_isOfKind(v_x_2696_, v___x_2700_);
if (v___x_2701_ == 0)
{
lean_object* v___x_2702_; 
lean_dec(v_x_2696_);
v___x_2702_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___redArg();
return v___x_2702_;
}
else
{
lean_object* v___x_2703_; lean_object* v_src_2704_; lean_object* v___x_2705_; lean_object* v_tgt_2706_; lean_object* v___x_2707_; lean_object* v___x_2708_; lean_object* v___x_2709_; lean_object* v___x_2710_; lean_object* v___x_2711_; lean_object* v___x_2712_; lean_object* v___x_2713_; 
v___x_2703_ = lean_unsigned_to_nat(1u);
v_src_2704_ = l_Lean_Syntax_getArg(v_x_2696_, v___x_2703_);
v___x_2705_ = lean_unsigned_to_nat(2u);
v_tgt_2706_ = l_Lean_Syntax_getArg(v_x_2696_, v___x_2705_);
lean_dec(v_x_2696_);
v___x_2707_ = lp_mathlib_Mathlib_Tactic_ToDual_translations;
v___x_2708_ = l_Lean_TSyntax_getId(v_src_2704_);
lean_dec(v_src_2704_);
v___x_2709_ = l_Lean_TSyntax_getId(v_tgt_2706_);
lean_dec(v_tgt_2706_);
v___x_2710_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__2));
v___x_2711_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___closed__3));
lean_inc(v___x_2709_);
v___x_2712_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2712_, 0, v___x_2709_);
lean_ctor_set(v___x_2712_, 1, v___x_2710_);
lean_ctor_set(v___x_2712_, 2, v___x_2711_);
lean_inc(v___x_2708_);
v___x_2713_ = lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0___redArg(v___x_2707_, v___x_2708_, v___x_2712_, v_a_2697_, v_a_2698_);
if (lean_obj_tag(v___x_2713_) == 0)
{
lean_object* v___x_2714_; lean_object* v___x_2715_; 
lean_dec_ref_known(v___x_2713_, 1);
v___x_2714_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2714_, 0, v___x_2708_);
lean_ctor_set(v___x_2714_, 1, v___x_2710_);
lean_ctor_set(v___x_2714_, 2, v___x_2711_);
v___x_2715_ = lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0___redArg(v___x_2707_, v___x_2709_, v___x_2714_, v_a_2697_, v_a_2698_);
return v___x_2715_;
}
else
{
lean_dec(v___x_2709_);
lean_dec(v___x_2708_);
return v___x_2713_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1___boxed(lean_object* v_x_2716_, lean_object* v_a_2717_, lean_object* v_a_2718_, lean_object* v_a_2719_){
_start:
{
lean_object* v_res_2720_; 
v_res_2720_ = lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1(v_x_2716_, v_a_2717_, v_a_2718_);
lean_dec(v_a_2718_);
lean_dec_ref(v_a_2717_);
return v_res_2720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1(lean_object* v_env_2721_, lean_object* v___y_2722_, lean_object* v___y_2723_){
_start:
{
lean_object* v___x_2725_; 
v___x_2725_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1___redArg(v_env_2721_, v___y_2723_);
return v___x_2725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1___boxed(lean_object* v_env_2726_, lean_object* v___y_2727_, lean_object* v___y_2728_, lean_object* v___y_2729_){
_start:
{
lean_object* v_res_2730_; 
v_res_2730_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__1(v_env_2726_, v___y_2727_, v___y_2728_);
lean_dec(v___y_2728_);
lean_dec_ref(v___y_2727_);
return v_res_2730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0(lean_object* v_00_u03b1_2731_, lean_object* v_ext_2732_, lean_object* v_k_2733_, lean_object* v_v_2734_, lean_object* v___y_2735_, lean_object* v___y_2736_){
_start:
{
lean_object* v___x_2738_; 
v___x_2738_ = lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0___redArg(v_ext_2732_, v_k_2733_, v_v_2734_, v___y_2735_, v___y_2736_);
return v___x_2738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0___boxed(lean_object* v_00_u03b1_2739_, lean_object* v_ext_2740_, lean_object* v_k_2741_, lean_object* v_v_2742_, lean_object* v___y_2743_, lean_object* v___y_2744_, lean_object* v___y_2745_){
_start:
{
lean_object* v_res_2746_; 
v_res_2746_ = lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0(v_00_u03b1_2739_, v_ext_2740_, v_k_2741_, v_v_2742_, v___y_2743_, v___y_2744_);
lean_dec(v___y_2744_);
lean_dec_ref(v___y_2743_);
return v_res_2746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1(lean_object* v_msgData_2747_, lean_object* v___y_2748_, lean_object* v___y_2749_){
_start:
{
lean_object* v___x_2751_; 
v___x_2751_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1___redArg(v_msgData_2747_, v___y_2749_);
return v___x_2751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1___boxed(lean_object* v_msgData_2752_, lean_object* v___y_2753_, lean_object* v___y_2754_, lean_object* v___y_2755_){
_start:
{
lean_object* v_res_2756_; 
v_res_2756_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__1(v_msgData_2752_, v___y_2753_, v___y_2754_);
lean_dec(v___y_2754_);
lean_dec_ref(v___y_2753_);
return v_res_2756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0(lean_object* v_00_u03b1_2757_, lean_object* v_msg_2758_, lean_object* v___y_2759_, lean_object* v___y_2760_){
_start:
{
lean_object* v___x_2762_; 
v___x_2762_ = lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0___redArg(v_msg_2758_, v___y_2759_, v___y_2760_);
return v___x_2762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0___boxed(lean_object* v_00_u03b1_2763_, lean_object* v_msg_2764_, lean_object* v___y_2765_, lean_object* v___y_2766_, lean_object* v___y_2767_){
_start:
{
lean_object* v_res_2768_; 
v_res_2768_ = lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0(v_00_u03b1_2763_, v_msg_2764_, v___y_2765_, v___y_2766_);
lean_dec(v___y_2766_);
lean_dec_ref(v___y_2765_);
return v_res_2768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2(lean_object* v_msgData_2769_, lean_object* v_macroStack_2770_, lean_object* v___y_2771_, lean_object* v___y_2772_){
_start:
{
lean_object* v___x_2774_; 
v___x_2774_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___redArg(v_msgData_2769_, v_macroStack_2770_, v___y_2772_);
return v___x_2774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2___boxed(lean_object* v_msgData_2775_, lean_object* v_macroStack_2776_, lean_object* v___y_2777_, lean_object* v___y_2778_, lean_object* v___y_2779_){
_start:
{
lean_object* v_res_2780_; 
v_res_2780_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandInsert__to__dual__translation______1_spec__0_spec__0_spec__2(v_msgData_2775_, v_macroStack_2776_, v___y_2777_, v___y_2778_);
lean_dec(v___y_2778_);
lean_dec_ref(v___y_2777_);
return v_res_2780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1_spec__0(lean_object* v_as_2814_, size_t v_sz_2815_, size_t v_i_2816_, lean_object* v_b_2817_, lean_object* v___y_2818_, lean_object* v___y_2819_){
_start:
{
uint8_t v___x_2821_; 
v___x_2821_ = lean_usize_dec_lt(v_i_2816_, v_sz_2815_);
if (v___x_2821_ == 0)
{
lean_object* v___x_2822_; 
v___x_2822_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2822_, 0, v_b_2817_);
return v___x_2822_;
}
else
{
lean_object* v___x_2823_; lean_object* v___x_2824_; lean_object* v_a_2825_; lean_object* v___x_2826_; lean_object* v___x_2827_; lean_object* v___x_2828_; lean_object* v___x_2829_; 
v___x_2823_ = lean_unsigned_to_nat(0u);
v___x_2824_ = lean_unsigned_to_nat(1u);
v_a_2825_ = lean_array_uget_borrowed(v_as_2814_, v_i_2816_);
v___x_2826_ = lp_mathlib_Mathlib_Tactic_ToDual_guessNameExt;
v___x_2827_ = l_Lean_Syntax_getArg(v_a_2825_, v___x_2823_);
v___x_2828_ = l_Lean_Syntax_getArg(v_a_2825_, v___x_2824_);
v___x_2829_ = lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation(v___x_2826_, v___x_2827_, v___x_2828_, v___y_2818_, v___y_2819_);
if (lean_obj_tag(v___x_2829_) == 0)
{
lean_object* v___x_2830_; 
lean_dec_ref_known(v___x_2829_, 1);
v___x_2830_ = lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation(v___x_2826_, v___x_2828_, v___x_2827_, v___y_2818_, v___y_2819_);
lean_dec(v___x_2827_);
lean_dec(v___x_2828_);
if (lean_obj_tag(v___x_2830_) == 0)
{
lean_object* v___x_2831_; size_t v___x_2832_; size_t v___x_2833_; 
lean_dec_ref_known(v___x_2830_, 1);
v___x_2831_ = lean_box(0);
v___x_2832_ = ((size_t)1ULL);
v___x_2833_ = lean_usize_add(v_i_2816_, v___x_2832_);
v_i_2816_ = v___x_2833_;
v_b_2817_ = v___x_2831_;
goto _start;
}
else
{
return v___x_2830_;
}
}
else
{
lean_dec(v___x_2828_);
lean_dec(v___x_2827_);
return v___x_2829_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1_spec__0___boxed(lean_object* v_as_2835_, lean_object* v_sz_2836_, lean_object* v_i_2837_, lean_object* v_b_2838_, lean_object* v___y_2839_, lean_object* v___y_2840_, lean_object* v___y_2841_){
_start:
{
size_t v_sz_boxed_2842_; size_t v_i_boxed_2843_; lean_object* v_res_2844_; 
v_sz_boxed_2842_ = lean_unbox_usize(v_sz_2836_);
lean_dec(v_sz_2836_);
v_i_boxed_2843_ = lean_unbox_usize(v_i_2837_);
lean_dec(v_i_2837_);
v_res_2844_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1_spec__0(v_as_2835_, v_sz_boxed_2842_, v_i_boxed_2843_, v_b_2838_, v___y_2839_, v___y_2840_);
lean_dec(v___y_2840_);
lean_dec_ref(v___y_2839_);
lean_dec_ref(v_as_2835_);
return v_res_2844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1(lean_object* v_x_2845_, lean_object* v_a_2846_, lean_object* v_a_2847_){
_start:
{
lean_object* v___x_2849_; uint8_t v___x_2850_; 
v___x_2849_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToDual_commandTo__dual__name__hint_____x2c_x2c___closed__1));
lean_inc(v_x_2845_);
v___x_2850_ = l_Lean_Syntax_isOfKind(v_x_2845_, v___x_2849_);
if (v___x_2850_ == 0)
{
lean_object* v___x_2851_; 
lean_dec(v_x_2845_);
v___x_2851_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__insert__cast___x3a_x3d____1_spec__0___redArg();
return v___x_2851_;
}
else
{
lean_object* v___x_2852_; lean_object* v___x_2853_; lean_object* v_hints_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; size_t v_sz_2857_; size_t v___x_2858_; lean_object* v___x_2859_; 
v___x_2852_ = lean_unsigned_to_nat(1u);
v___x_2853_ = l_Lean_Syntax_getArg(v_x_2845_, v___x_2852_);
lean_dec(v_x_2845_);
v_hints_2854_ = l_Lean_Syntax_getArgs(v___x_2853_);
lean_dec(v___x_2853_);
v___x_2855_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_hints_2854_);
lean_dec_ref(v_hints_2854_);
v___x_2856_ = lean_box(0);
v_sz_2857_ = lean_array_size(v___x_2855_);
v___x_2858_ = ((size_t)0ULL);
v___x_2859_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1_spec__0(v___x_2855_, v_sz_2857_, v___x_2858_, v___x_2856_, v_a_2846_, v_a_2847_);
lean_dec_ref(v___x_2855_);
if (lean_obj_tag(v___x_2859_) == 0)
{
lean_object* v___x_2861_; uint8_t v_isShared_2862_; uint8_t v_isSharedCheck_2866_; 
v_isSharedCheck_2866_ = !lean_is_exclusive(v___x_2859_);
if (v_isSharedCheck_2866_ == 0)
{
lean_object* v_unused_2867_; 
v_unused_2867_ = lean_ctor_get(v___x_2859_, 0);
lean_dec(v_unused_2867_);
v___x_2861_ = v___x_2859_;
v_isShared_2862_ = v_isSharedCheck_2866_;
goto v_resetjp_2860_;
}
else
{
lean_dec(v___x_2859_);
v___x_2861_ = lean_box(0);
v_isShared_2862_ = v_isSharedCheck_2866_;
goto v_resetjp_2860_;
}
v_resetjp_2860_:
{
lean_object* v___x_2864_; 
if (v_isShared_2862_ == 0)
{
lean_ctor_set(v___x_2861_, 0, v___x_2856_);
v___x_2864_ = v___x_2861_;
goto v_reusejp_2863_;
}
else
{
lean_object* v_reuseFailAlloc_2865_; 
v_reuseFailAlloc_2865_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2865_, 0, v___x_2856_);
v___x_2864_ = v_reuseFailAlloc_2865_;
goto v_reusejp_2863_;
}
v_reusejp_2863_:
{
return v___x_2864_;
}
}
}
else
{
return v___x_2859_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1___boxed(lean_object* v_x_2868_, lean_object* v_a_2869_, lean_object* v_a_2870_, lean_object* v_a_2871_){
_start:
{
lean_object* v_res_2872_; 
v_res_2872_ = lp_mathlib_Mathlib_Tactic_ToDual___aux__Mathlib__Tactic__Translate__ToDual______elabRules__Mathlib__Tactic__ToDual__commandTo__dual__name__hint_____x2c_x2c__1(v_x_2868_, v_a_2869_, v_a_2870_);
lean_dec(v_a_2870_);
lean_dec_ref(v_a_2869_);
return v_res_2872_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Translate_TagUnfoldBoundary(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Translate_ToDual(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Translate_TagUnfoldBoundary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Translate_ToDual(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ToDual_to__dual = _init_lp_mathlib_Mathlib_Tactic_ToDual_to__dual();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToDual_to__dual);
lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f__ = _init_lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f__();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToDual_attrTo__dual_x3f__);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_78335030____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_ToDual_ignoreArgsAttr = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToDual_ignoreArgsAttr);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_678359338____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_ToDual_unfoldBoundaries = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToDual_unfoldBoundaries);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_2699740242____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_ToDual_doTranslateAttr = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToDual_doTranslateAttr);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_4009335259____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_549782961____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_ToDual_translations = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToDual_translations);
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ToDual_nameDict = _init_lp_mathlib_Mathlib_Tactic_ToDual_nameDict();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToDual_nameDict);
lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict = _init_lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToDual_abbreviationDict);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_3035231656____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_ToDual_guessNameExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToDual_guessNameExt);
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ToDual_data = _init_lp_mathlib_Mathlib_Tactic_ToDual_data();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToDual_data);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToDual_0__Mathlib_Tactic_ToDual_initFn_00___x40_Mathlib_Tactic_Translate_ToDual_1319075495____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Translate_TagUnfoldBoundary(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Translate_ToDual(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Translate_TagUnfoldBoundary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Translate_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Translate_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Translate_ToDual(builtin);
}
#ifdef __cplusplus
}
#endif
