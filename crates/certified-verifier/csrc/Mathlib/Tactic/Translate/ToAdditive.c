// Lean compiler output
// Module: Mathlib.Tactic.Translate.ToAdditive
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Translate.Core
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_registerNameMapExtension___redArg(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
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
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_Translate_attrArgs;
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Translate_elabTranslationAttr(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Translate_addTranslationAttr(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_profileitIOUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "ToAdditive"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "to_additive_ignore_args"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 195, 156, 165, 47, 235, 27, 126)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__3_value),LEAN_SCALAR_PTR_LITERAL(48, 244, 47, 1, 140, 164, 111, 213)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__8_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__10_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__13_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "to_additive_do_translate"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 195, 156, 165, 47, 235, 27, 126)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(40, 9, 237, 220, 166, 250, 156, 23)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "to_additive_dont_translate"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 195, 156, 165, 47, 235, 27, 126)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(11, 199, 118, 124, 242, 246, 216, 90)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "to_additive"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 195, 156, 165, 47, 235, 27, 126)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__0_value),LEAN_SCALAR_PTR_LITERAL(86, 237, 250, 219, 205, 129, 25, 153)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__3_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "attrTo_additive\?_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 195, 156, 165, 47, 235, 27, 126)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(199, 31, 234, 124, 115, 70, 125, 94)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "to_additive\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f__;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__3_value),LEAN_SCALAR_PTR_LITERAL(75, 200, 83, 217, 234, 132, 86, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2____boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__3_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ignoreArgsAttr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 195, 156, 165, 47, 235, 27, 126)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(50, 156, 104, 71, 66, 118, 65, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 90, .m_capacity = 90, .m_length = 89, .m_data = "Auxiliary attribute for `to_additive` stating that certain arguments are not additivized."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_ignoreArgsAttr;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "doTranslateAttr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 195, 156, 165, 47, 235, 27, 126)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(157, 135, 158, 33, 55, 255, 131, 178)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_doTranslateAttr;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Already exists entry for "};
static const lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Translate"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(104, 225, 249, 99, 81, 122, 117, 142)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 211, 230, 110, 81, 221, 27, 112)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(158, 56, 12, 33, 186, 64, 246, 113)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__9_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(87, 117, 144, 190, 228, 120, 83, 220)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__9_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__9_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__10_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__9_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(150, 71, 123, 19, 196, 240, 113, 221)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__10_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__10_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__11_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__10_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(205, 9, 209, 41, 91, 27, 133, 43)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__11_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__11_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__12_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__12_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__12_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__13_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__11_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__12_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(36, 145, 63, 69, 157, 61, 224, 16)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__13_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__13_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__14_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__14_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__14_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__15_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__13_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__14_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(45, 186, 41, 74, 113, 101, 117, 52)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__15_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__15_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__16_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__15_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(40, 97, 176, 71, 224, 38, 1, 239)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__16_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__16_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__17_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__16_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(85, 96, 81, 86, 74, 173, 125, 4)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__17_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__17_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__18_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__17_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(202, 136, 177, 164, 219, 70, 134, 19)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__18_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__18_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__19_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__18_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(233, 53, 244, 208, 121, 185, 110, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__19_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__19_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__21_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__21_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__21_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__23_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__23_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__23_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__26_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__do__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(51, 251, 182, 57, 49, 178, 10, 11)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__26_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__26_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__27_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__26_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__27_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__27_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__28_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 101, .m_capacity = 101, .m_length = 100, .m_data = "Auxiliary attribute for `to_additive` stating that the operations on this type should be translated."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__28_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__28_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__31_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__31_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__31_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__32_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__dont__translate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(240, 74, 86, 238, 112, 210, 124, 194)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__32_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__32_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__33_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__32_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__33_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__33_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__34_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 105, .m_capacity = 105, .m_length = 104, .m_data = "Auxiliary attribute for `to_additive` stating that the operations on this type should not be translated."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__34_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__34_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "translations"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 195, 156, 165, 47, 235, 27, 126)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(32, 8, 12, 8, 232, 53, 73, 143)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_translations;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "gpfree"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "APFree"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "quantale"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Add"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Quantale"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "square"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Even"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "mconv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "irreducible"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Irreducible"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "mlconvolution"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "LConvolution"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__23_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__26_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__22_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "conj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__34_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__35_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__33_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__36_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "commutator"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Commutator"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__39_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__40_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__38_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__41_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "rootable"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__43_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Divisible"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__44_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__43_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__45_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__46_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "zpowers"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "ZMultiples"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__48_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__48_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__47_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__49_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__50_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "powers"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__51_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Multiples"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__52_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__52_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__53_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__51_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__53_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__54_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "multipliable"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__55_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Summable"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__56_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__56_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__57_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__55_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__57_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__58_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__58_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__59_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__54_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__59_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__60_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__50_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__60_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__61_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__46_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__61_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__62_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__42_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__62_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__63_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__37_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__63_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__64_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cyclic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__65_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Cyclic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__66_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__66_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__67_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__67_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__68_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__65_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__68_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__69_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "semigrp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__70_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Semigrp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__71_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__71_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__72_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__72_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__73_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__70_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__73_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__74_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "grp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__75_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Grp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__76_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__76_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__77_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__77_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__78_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__75_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__78_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__79_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "commute"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__80 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__80_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Commute"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__81 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__81_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__81_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__82 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__82_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__82_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__83 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__83_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__80_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__83_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__84_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "semiconj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__85_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiconj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__86_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__86_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__87 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__87_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__87_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__88 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__88_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__85_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__88_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__89 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__89_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "conjugates"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__90 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__90_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Conjugates"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__91 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__91_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__91_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__92 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__92_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__92_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__93 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__93_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__94_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__90_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__93_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__94 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__94_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__94_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__64_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__95 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__95_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__89_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__95_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__96 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__96_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__84_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__96_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__97 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__97_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__79_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__97_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__98 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__98_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__74_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__98_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__99 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__99_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__69_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__99_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__100 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__100_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "magma"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__101 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__101_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Magma"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__102 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__102_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__102_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__103 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__103_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__104_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__103_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__104 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__104_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__105_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__101_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__104_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__105 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__105_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__106_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "haar"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__106 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__106_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__107_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Haar"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__107 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__107_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__108_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__107_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__108 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__108_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__109_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__108_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__109 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__109_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__110_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__106_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__109_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__110 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__110_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__111_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "prehaar"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__111 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__111_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__112_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Prehaar"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__112 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__112_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__113_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__112_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__113 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__113_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__114_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__113_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__114 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__114_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__115_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__111_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__114_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__115 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__115_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__116_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "unit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__116 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__116_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__117_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Unit"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__117 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__117_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__118_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__117_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__118 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__118_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__119_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__118_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__119 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__119_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__120_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__116_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__119_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__120 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__120_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__121_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "units"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__121 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__121_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__122_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Units"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__122 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__122_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__123_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__122_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__123 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__123_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__124_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__123_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__124 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__124_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__125_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__121_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__124_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__125 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__125_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__126_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__125_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__100_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__126 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__126_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__127_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__120_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__126_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__127 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__127_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__128_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__115_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__127_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__128 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__128_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__129_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__110_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__128_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__129 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__129_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__130_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__105_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__129_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__130 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__130_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__131_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "monoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__131 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__131_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__132_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Monoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__132 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__132_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__133_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__132_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__133 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__133_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__134_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__133_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__134 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__134_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__135_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__131_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__134_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__135 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__135_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__136_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "submonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__136 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__136_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__137_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Submonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__137 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__137_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__138_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__137_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__138 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__138_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__139_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__138_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__139 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__139_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__140_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__136_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__139_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__140 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__140_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__141_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__141 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__141_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__142_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__142 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__142_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__143_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__142_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__143 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__143_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__144_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__143_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__144 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__144_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__145_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__141_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__144_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__145 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__145_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__146_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "subgroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__146 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__146_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__147_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Subgroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__147 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__147_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__148_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__147_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__148 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__148_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__149_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__148_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__149 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__149_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__150_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__146_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__149_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__150 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__150_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__151_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "semigroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__151 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__151_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__152_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Semigroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__152 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__152_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__153_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__152_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__153 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__153_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__154_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__153_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__154 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__154_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__155_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__151_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__154_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__155 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__155_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__156_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "torsor"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__156 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__156_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__157_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Torsor"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__157 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__157_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__158_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__157_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__158 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__158_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__159_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__158_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__159 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__159_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__160_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__156_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__159_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__160 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__160_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__161_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__160_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__130_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__161 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__161_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__162_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__155_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__161_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__162 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__162_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__163_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__150_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__162_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__163 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__163_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__164_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__145_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__163_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__164 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__164_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__165_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__140_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__164_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__165 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__165_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__166_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__135_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__165_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__166 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__166_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__167_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "finprod"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__167 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__167_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__168_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finsum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__168 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__168_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__169_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__168_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__169 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__169_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__170_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__167_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__169_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__170 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__170_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__171_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "tprod"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__171 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__171_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__172_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "TSum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__172 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__172_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__173_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__172_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__173 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__173_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__174_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__171_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__173_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__174 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__174_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__175_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "pow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__175 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__175_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__176_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "NSMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__176 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__176_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__177_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__176_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__177 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__177_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__178_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__175_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__177_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__178 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__178_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__179_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "npow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__179 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__179_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__180_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__179_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__177_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__180 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__180_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__181_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "zpow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__181 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__181_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__182_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ZSMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__182 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__182_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__183_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__182_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__183 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__183_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__184_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__181_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__183_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__184 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__184_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__185_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "mabs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__185 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__185_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__186_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Abs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__186 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__186_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__187_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__186_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__187 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__187_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__188_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__185_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__187_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__188 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__188_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__189_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__188_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__166_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__189 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__189_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__190_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__184_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__189_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__190 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__190_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__191_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__180_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__190_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__191 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__191_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__192_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__178_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__191_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__192 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__192_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__193_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__174_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__192_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__193 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__193_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__194_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__170_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__193_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__194 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__194_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__195_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "sdiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__195 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__195_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__196_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "VSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__196 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__196_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__197_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__196_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__197 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__197_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__198_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__195_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__197_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__198 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__198_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__199_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "prod"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__199 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__199_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__200_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Sum"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__200 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__200_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__201_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__200_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__201 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__201_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__202_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__199_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__201_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__202 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__202_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__203_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hmul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__203 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__203_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__204_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__204 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__204_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__205_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__204_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__205 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__205_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__206_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__203_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__205_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__206 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__206_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__207_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "hsmul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__207 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__207_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__208_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "HVAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__208 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__208_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__209_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__208_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__209 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__209_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__210_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__207_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__209_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__210 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__210_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__211_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hdiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__211 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__211_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__212_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__212 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__212_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__213_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__212_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__213 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__213_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__214_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__211_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__213_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__214 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__214_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__215_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hpow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__215 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__215_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__216_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "HSMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__216 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__216_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__217_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__216_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__217 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__217_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__218_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__215_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__217_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__218 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__218_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__219_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__218_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__194_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__219 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__219_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__220_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__214_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__219_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__220 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__220_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__221_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__210_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__220_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__221 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__221_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__222_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__206_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__221_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__222 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__222_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__223_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__202_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__222_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__223 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__223_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__224_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__198_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__223_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__224 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__224_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__225_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__225 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__225_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__226_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__226 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__226_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__227_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__226_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__227 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__227_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__228_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__225_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__227_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__228 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__228_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__229_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__229 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__229_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__230_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__230 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__230_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__231_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__229_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__230_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__231 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__231_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__232_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "smul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__232 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__232_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__233_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "VAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__233 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__233_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__234_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__233_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__234 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__234_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__235_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__232_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__234_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__235 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__235_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__236_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__236 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__236_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__237_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__237 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__237_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__238_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__237_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__238 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__238_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__239_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__236_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__238_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__239 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__239_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__240_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__240 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__240_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__241_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Sub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__241 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__241_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__242_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__241_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__242 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__242_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__243_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__240_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__242_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__243 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__243_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__244_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__243_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__224_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__244 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__244_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__245_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__239_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__244_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__245 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__245_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__246_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__235_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__245_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__246 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__246_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__247_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__231_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__246_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__247 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__247_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__248_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__228_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__247_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__248 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__248_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__249_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__249;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__250_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__250;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__251_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__251;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "isModHom"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "IsAddModHom"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mapMod"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "MapAddMod"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "modObj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "AddModObj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "yonedaMon"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "YonedaAddMon"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "conGen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "AddConGen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unoneD"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "unzeroD"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "unone"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unzero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "addShift"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Shift"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__28_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "addSubshift"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Subshift"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__31_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "isQuotientCoveringMap"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "IsAddQuotientCoveringMap"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__34_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__35_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "addExact"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Exact"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__37_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__38_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "isMonHom"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "IsAddMonHom"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__40_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__41_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mapMon"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__43_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "MapAddMon"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__43_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__44_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__45_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "monObj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__46_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "AddMonObj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__47_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__46_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__47_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__48_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__48_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__45_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__49_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__50_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__42_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__50_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__51_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__39_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__51_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__52_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__36_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__52_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__53_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__33_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__53_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__54_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__30_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__54_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__55_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "isOfFinOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__56_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "IsOfFinAddOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__57_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__56_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__57_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__58_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isCentralScalar"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__59_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "IsCentralVAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__60_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__59_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__60_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__61_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "function_addSemiconj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__62_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Function_semiconj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__63_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__62_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__63_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__64_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "function_addCommute"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__65_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Function_commute"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__66_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__65_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__66_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__67_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "divisionAddMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__68_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "SubtractionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__69_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__68_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__69_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__70_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "subNegZeroAddMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__71_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "SubNegZeroMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__72_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__71_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__72_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__73_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "modularCharacter"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__74_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "AddModularCharacter"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__75_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__74_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__75_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__76_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__76_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__55_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__77_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__73_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__77_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__78_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__70_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__78_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__79_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__67_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__79_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__80 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__80_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__64_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__80_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__81 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__81_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__61_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__81_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__82 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__82_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__58_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__82_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__83 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__83_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "quotientMeasure"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__84_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "AddQuotientMeasure"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__85_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__84_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__85_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__86_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "negFun"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__87 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__87_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "InvFun"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__88 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__88_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__87_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__88_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__89 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__89_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "uniqueProds"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__90 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__90_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "UniqueSums"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__91 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__91_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__90_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__91_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__92 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__92_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "orderOf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__93 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__93_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__94_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "AddOrderOf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__94 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__94_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__93_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__94_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__95 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__95_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "zeroLePart"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__96 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__96_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "PosPart"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__97 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__97_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__96_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__97_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__98 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__98_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "leZeroPart"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__99 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__99_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NegPart"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__100 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__100_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__99_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__100_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__101 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__101_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isScalarTower"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__102 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__102_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "VAddAssocClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__103 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__103_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__104_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__102_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__103_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__104 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__104_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__105_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__104_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__83_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__105 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__105_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__106_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__101_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__105_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__106 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__106_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__107_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__98_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__106_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__107 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__107_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__108_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__95_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__107_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__108 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__108_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__109_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__92_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__108_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__109 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__109_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__110_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__89_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__109_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__110 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__110_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__111_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__86_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__110_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__111 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__111_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__112_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "addSpanning"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__112 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__112_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__113_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Spanning"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__113 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__113_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__114_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__112_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__113_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__114 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__114_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__115_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "addIndicator"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__115 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__115_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__116_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Indicator"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__116 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__116_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__117_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__115_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__116_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__117 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__117_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__118_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "isEven"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__118 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__118_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__119_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__118_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__119 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__119_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__120_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "isRegular"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__120 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__120_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__121_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "IsAddRegular"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__121 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__121_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__122_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__120_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__121_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__122 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__122_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__123_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "isLeftRegular"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__123 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__123_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__124_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "IsAddLeftRegular"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__124 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__124_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__125_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__123_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__124_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__125 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__125_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__126_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "isRightRegular"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__126 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__126_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__127_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "IsAddRightRegular"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__127 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__127_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__128_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__126_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__127_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__128 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__128_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__129_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "hasFundamentalDomain"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__129 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__129_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__130_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "HasAddFundamentalDomain"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__130 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__130_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__131_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__129_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__130_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__131 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__131_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__132_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__131_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__111_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__132 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__132_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__133_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__128_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__132_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__133 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__133_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__134_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__125_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__133_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__134 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__134_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__135_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__122_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__134_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__135 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__135_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__136_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__119_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__135_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__136 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__136_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__137_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__117_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__136_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__137 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__137_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__138_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__114_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__137_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__138 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__138_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__139_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ltzero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__139 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__139_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__140_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__139_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__237_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__140 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__140_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__141_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "lt_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__141 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__141_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__142_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__141_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__237_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__142 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__142_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__143_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "addAntidiagonal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__143 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__143_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__144_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Antidiagonal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__144 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__144_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__145_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__143_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__144_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__145 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__145_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__146_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "addSingle"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__146 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__146_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__147_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Single"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__147 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__147_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__148_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__146_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__147_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__148 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__148_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__149_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "addSupport"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__149 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__149_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__150_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Support"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__150 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__150_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__151_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__149_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__150_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__151 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__151_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__152_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "addTSupport"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__152 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__152_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__153_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "TSupport"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__153 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__153_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__154_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__152_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__153_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__154 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__154_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__155_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "addPointed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__155 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__155_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__156_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Pointed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__156 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__156_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__157_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__155_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__156_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__157 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__157_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__158_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__157_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__138_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__158 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__158_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__159_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__154_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__158_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__159 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__159_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__160_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__151_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__159_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__160 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__160_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__161_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__148_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__160_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__161 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__161_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__162_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__145_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__161_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__162 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__162_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__163_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__142_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__162_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__163 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__163_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__164_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__140_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__163_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__164 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__164_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__165_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "commAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__165 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__165_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__166_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "AddComm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__166 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__166_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__167_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__165_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__166_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__167 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__167_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__168_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "zero_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__168 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__168_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__169_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Nonneg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__169 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__169_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__170_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__168_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__169_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__170 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__170_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__171_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "zeroLE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__171 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__171_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__172_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__171_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__169_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__172 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__172_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__173_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "zero_lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__173 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__173_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__174_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Pos"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__174 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__174_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__175_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__173_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__174_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__175 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__175_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__176_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "zeroLT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__176 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__176_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__177_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__176_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__174_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__177 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__177_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__178_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "lezero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__178 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__178_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__179_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Nonpos"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__179 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__179_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__180_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__178_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__179_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__180 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__180_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__181_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "le_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__181 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__181_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__182_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__181_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__179_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__182 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__182_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__183_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__182_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__164_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__183 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__183_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__184_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__180_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__183_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__184 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__184_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__185_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__177_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__184_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__185 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__185_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__186_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__175_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__185_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__186 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__186_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__187_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__172_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__186_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__187 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__187_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__188_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__170_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__187_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__188 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__188_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__189_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__167_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__188_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__189 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__189_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__190_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "isCancelAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__190 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__190_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__191_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "IsCancelAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__191 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__191_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__192_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__190_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__191_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__192 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__192_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__193_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "isLeftCancelAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__193 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__193_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__194_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "IsLeftCancelAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__194 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__194_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__195_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__193_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__194_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__195 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__195_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__196_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "isRightCancelAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__196 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__196_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__197_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "IsRightCancelAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__197 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__197_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__198_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__196_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__197_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__198 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__198_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__199_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "cancelAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__199 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__199_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__200_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "AddCancel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__200 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__200_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__201_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__199_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__200_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__201 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__201_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__202_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "leftCancelAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__202 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__202_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__203_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "AddLeftCancel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__203 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__203_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__204_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__202_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__203_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__204 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__204_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__205_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "rightCancelAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__205 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__205_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__206_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "AddRightCancel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__206 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__206_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__207_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__205_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__206_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__207 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__207_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__208_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "cancelCommAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__208 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__208_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__209_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "AddCancelComm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__209 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__209_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__210_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__208_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__209_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__210 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__210_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__211_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__210_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__189_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__211 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__211_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__212_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__207_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__211_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__212 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__212_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__213_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__204_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__212_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__213 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__213_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__214_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__201_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__213_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__214 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__214_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__215_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__198_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__214_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__215 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__215_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__216_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__195_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__215_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__216 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__216_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__217_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__192_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__216_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__217 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__217_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__218_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__218;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_guessNameExt;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 223, 190, 146, 58, 73, 225, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_data;
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__19_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1826194073) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(129, 40, 102, 219, 151, 18, 212, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__21_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(194, 169, 191, 246, 230, 3, 205, 169)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__23_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(78, 37, 244, 153, 91, 115, 26, 14)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(159, 61, 80, 226, 192, 105, 246, 189)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed, .m_arity = 8, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Transport multiplicative to additive"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__6_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__7_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__4_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "commandInsert_to_additive_translation__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 195, 156, 165, 47, 235, 27, 126)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(182, 195, 34, 211, 132, 189, 16, 62)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "insert_to_additive_translation"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation____ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__5___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "commandTo_additive_name_hint__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 195, 156, 165, 47, 235, 27, 126)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 11, 106, 1, 154, 52, 237, 59)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "to_additive_name_hint"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint____ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandTo__additive__name__hint______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandTo__additive__name__hint______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__9(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_95_ = lp_mathlib_Mathlib_Tactic_Translate_attrArgs;
v___x_96_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__8));
v___x_97_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6));
v___x_98_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v___x_96_);
lean_ctor_set(v___x_98_, 2, v___x_95_);
return v___x_98_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__10(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_99_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__9, &lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__9);
v___x_100_ = lean_unsigned_to_nat(1022u);
v___x_101_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1));
v___x_102_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v___x_100_);
lean_ctor_set(v___x_102_, 2, v___x_99_);
return v___x_102_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive(void){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__10, &lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__10);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__4(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_114_ = lp_mathlib_Mathlib_Tactic_Translate_attrArgs;
v___x_115_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__3));
v___x_116_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__6));
v___x_117_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v___x_115_);
lean_ctor_set(v___x_117_, 2, v___x_114_);
return v___x_117_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__5(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_118_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__4, &lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__4);
v___x_119_ = lean_unsigned_to_nat(1022u);
v___x_120_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1));
v___x_121_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v___x_119_);
lean_ctor_set(v___x_121_, 2, v___x_118_);
return v___x_121_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f__(void){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__5, &lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__5);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1(lean_object* v_x_126_, lean_object* v_a_127_, lean_object* v_a_128_){
_start:
{
lean_object* v___x_129_; uint8_t v___x_130_; 
v___x_129_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f___00__closed__1));
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
v___x_138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__0));
v___x_139_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__1));
lean_inc_n(v___x_137_, 3);
v___x_140_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_137_);
lean_ctor_set(v___x_140_, 1, v___x_138_);
v___x_141_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1___closed__1));
v___x_142_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive___closed__5));
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1___boxed(lean_object* v_x_147_, lean_object* v_a_148_, lean_object* v_a_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______macroRules__Mathlib__Tactic__ToAdditive__attrTo__additive_x3f____1(v_x_147_, v_a_148_, v_a_149_);
lean_dec_ref(v_a_148_);
return v_res_150_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___closed__0(void){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg(){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_155_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___closed__0);
v___x_156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_156_, 0, v___x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v___y_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg();
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_159_, lean_object* v___y_160_, lean_object* v___y_161_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg();
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0(v_00_u03b1_164_, v___y_165_, v___y_166_);
lean_dec(v___y_166_);
lean_dec_ref(v___y_165_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__2(size_t v_sz_169_, size_t v_i_170_, lean_object* v_bs_171_){
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
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__2___boxed(lean_object* v_sz_183_, lean_object* v_i_184_, lean_object* v_bs_185_){
_start:
{
size_t v_sz_boxed_186_; size_t v_i_boxed_187_; lean_object* v_res_188_; 
v_sz_boxed_186_ = lean_unbox_usize(v_sz_183_);
lean_dec(v_sz_183_);
v_i_boxed_187_ = lean_unbox_usize(v_i_184_);
lean_dec(v_i_184_);
v_res_188_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__2(v_sz_boxed_186_, v_i_boxed_187_, v_bs_185_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__1(size_t v_sz_189_, size_t v_i_190_, lean_object* v_bs_191_){
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
v___x_195_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive__ignore__args___closed__14));
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
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__1___boxed(lean_object* v_sz_204_, lean_object* v_i_205_, lean_object* v_bs_206_){
_start:
{
size_t v_sz_boxed_207_; size_t v_i_boxed_208_; lean_object* v_res_209_; 
v_sz_boxed_207_ = lean_unbox_usize(v_sz_204_);
lean_dec(v_sz_204_);
v_i_boxed_208_ = lean_unbox_usize(v_i_205_);
lean_dec(v_i_205_);
v_res_209_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__1(v_sz_boxed_207_, v_i_boxed_208_, v_bs_206_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_(lean_object* v___x_210_, lean_object* v___x_211_, lean_object* v___x_212_, lean_object* v___x_213_, lean_object* v_x_214_, lean_object* v_stx_215_, lean_object* v___y_216_, lean_object* v___y_217_){
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
v___x_225_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg();
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
v___x_239_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__1(v_sz_237_, v___x_238_, v___x_236_);
if (lean_obj_tag(v___x_239_) == 0)
{
lean_object* v___x_240_; lean_object* v_a_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_248_; 
v___x_240_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg();
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
v___x_251_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__2(v_sz_250_, v___x_238_, v_val_249_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2____boxed(lean_object* v___x_252_, lean_object* v___x_253_, lean_object* v___x_254_, lean_object* v___x_255_, lean_object* v_x_256_, lean_object* v_stx_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_(v___x_252_, v___x_253_, v___x_254_, v___x_255_, v_x_256_, v_stx_257_, v___y_258_, v___y_259_);
lean_dec(v___y_259_);
lean_dec_ref(v___y_258_);
lean_dec(v_x_256_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; 
v___x_282_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__5_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_));
v___x_283_ = lp_batteries_Lean_registerNameMapAttribute___redArg(v___x_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2____boxed(lean_object* v_a_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_();
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_293_; lean_object* v___x_294_; 
v___x_293_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2_));
v___x_294_ = lp_batteries_Lean_registerNameMapExtension___redArg(v___x_293_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2____boxed(lean_object* v_a_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2_();
return v_res_296_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__0(void){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_297_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_298_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__0);
v___x_299_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_299_, 0, v___x_298_);
return v___x_299_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__2(void){
_start:
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_300_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__1);
v___x_301_ = lean_unsigned_to_nat(0u);
v___x_302_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_302_, 0, v___x_301_);
lean_ctor_set(v___x_302_, 1, v___x_301_);
lean_ctor_set(v___x_302_, 2, v___x_301_);
lean_ctor_set(v___x_302_, 3, v___x_301_);
lean_ctor_set(v___x_302_, 4, v___x_300_);
lean_ctor_set(v___x_302_, 5, v___x_300_);
lean_ctor_set(v___x_302_, 6, v___x_300_);
lean_ctor_set(v___x_302_, 7, v___x_300_);
lean_ctor_set(v___x_302_, 8, v___x_300_);
lean_ctor_set(v___x_302_, 9, v___x_300_);
return v___x_302_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__3(void){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v___x_303_ = lean_unsigned_to_nat(32u);
v___x_304_ = lean_mk_empty_array_with_capacity(v___x_303_);
v___x_305_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_305_, 0, v___x_304_);
return v___x_305_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__4(void){
_start:
{
size_t v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; 
v___x_306_ = ((size_t)5ULL);
v___x_307_ = lean_unsigned_to_nat(0u);
v___x_308_ = lean_unsigned_to_nat(32u);
v___x_309_ = lean_mk_empty_array_with_capacity(v___x_308_);
v___x_310_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__3);
v___x_311_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_311_, 0, v___x_310_);
lean_ctor_set(v___x_311_, 1, v___x_309_);
lean_ctor_set(v___x_311_, 2, v___x_307_);
lean_ctor_set(v___x_311_, 3, v___x_307_);
lean_ctor_set_usize(v___x_311_, 4, v___x_306_);
return v___x_311_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__5(void){
_start:
{
lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; 
v___x_312_ = lean_box(1);
v___x_313_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__4);
v___x_314_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__1);
v___x_315_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_315_, 0, v___x_314_);
lean_ctor_set(v___x_315_, 1, v___x_313_);
lean_ctor_set(v___x_315_, 2, v___x_312_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2(lean_object* v_msgData_316_, lean_object* v___y_317_, lean_object* v___y_318_){
_start:
{
lean_object* v___x_320_; lean_object* v_env_321_; lean_object* v_options_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
v___x_320_ = lean_st_ref_get(v___y_318_);
v_env_321_ = lean_ctor_get(v___x_320_, 0);
lean_inc_ref(v_env_321_);
lean_dec(v___x_320_);
v_options_322_ = lean_ctor_get(v___y_317_, 2);
v___x_323_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__2);
v___x_324_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__5);
lean_inc_ref(v_options_322_);
v___x_325_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_325_, 0, v_env_321_);
lean_ctor_set(v___x_325_, 1, v___x_323_);
lean_ctor_set(v___x_325_, 2, v___x_324_);
lean_ctor_set(v___x_325_, 3, v_options_322_);
v___x_326_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_326_, 0, v___x_325_);
lean_ctor_set(v___x_326_, 1, v_msgData_316_);
v___x_327_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_327_, 0, v___x_326_);
return v___x_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object* v_msgData_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_){
_start:
{
lean_object* v_res_332_; 
v_res_332_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2(v_msgData_328_, v___y_329_, v___y_330_);
lean_dec(v___y_330_);
lean_dec_ref(v___y_329_);
return v_res_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___redArg(lean_object* v_msg_333_, lean_object* v___y_334_, lean_object* v___y_335_){
_start:
{
lean_object* v_ref_337_; lean_object* v___x_338_; lean_object* v_a_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_347_; 
v_ref_337_ = lean_ctor_get(v___y_334_, 5);
v___x_338_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2(v_msg_333_, v___y_334_, v___y_335_);
v_a_339_ = lean_ctor_get(v___x_338_, 0);
v_isSharedCheck_347_ = !lean_is_exclusive(v___x_338_);
if (v_isSharedCheck_347_ == 0)
{
v___x_341_ = v___x_338_;
v_isShared_342_ = v_isSharedCheck_347_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_a_339_);
lean_dec(v___x_338_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_347_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
lean_object* v___x_343_; lean_object* v___x_345_; 
lean_inc(v_ref_337_);
v___x_343_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_343_, 0, v_ref_337_);
lean_ctor_set(v___x_343_, 1, v_a_339_);
if (v_isShared_342_ == 0)
{
lean_ctor_set_tag(v___x_341_, 1);
lean_ctor_set(v___x_341_, 0, v___x_343_);
v___x_345_ = v___x_341_;
goto v_reusejp_344_;
}
else
{
lean_object* v_reuseFailAlloc_346_; 
v_reuseFailAlloc_346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_346_, 0, v___x_343_);
v___x_345_ = v_reuseFailAlloc_346_;
goto v_reusejp_344_;
}
v_reusejp_344_:
{
return v___x_345_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object* v_msg_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_){
_start:
{
lean_object* v_res_352_; 
v_res_352_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___redArg(v_msg_348_, v___y_349_, v___y_350_);
lean_dec(v___y_350_);
lean_dec_ref(v___y_349_);
return v_res_352_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_353_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; 
v___x_354_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0, &lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__0);
v___x_355_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_355_, 0, v___x_354_);
return v___x_355_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_356_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1, &lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__1);
v___x_357_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_357_, 0, v___x_356_);
lean_ctor_set(v___x_357_, 1, v___x_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object* v_env_358_, lean_object* v___y_359_){
_start:
{
lean_object* v___x_361_; lean_object* v_nextMacroScope_362_; lean_object* v_ngen_363_; lean_object* v_auxDeclNGen_364_; lean_object* v_traceState_365_; lean_object* v_messages_366_; lean_object* v_infoState_367_; lean_object* v_snapshotTasks_368_; lean_object* v___x_370_; uint8_t v_isShared_371_; uint8_t v_isSharedCheck_379_; 
v___x_361_ = lean_st_ref_take(v___y_359_);
v_nextMacroScope_362_ = lean_ctor_get(v___x_361_, 1);
v_ngen_363_ = lean_ctor_get(v___x_361_, 2);
v_auxDeclNGen_364_ = lean_ctor_get(v___x_361_, 3);
v_traceState_365_ = lean_ctor_get(v___x_361_, 4);
v_messages_366_ = lean_ctor_get(v___x_361_, 6);
v_infoState_367_ = lean_ctor_get(v___x_361_, 7);
v_snapshotTasks_368_ = lean_ctor_get(v___x_361_, 8);
v_isSharedCheck_379_ = !lean_is_exclusive(v___x_361_);
if (v_isSharedCheck_379_ == 0)
{
lean_object* v_unused_380_; lean_object* v_unused_381_; 
v_unused_380_ = lean_ctor_get(v___x_361_, 5);
lean_dec(v_unused_380_);
v_unused_381_ = lean_ctor_get(v___x_361_, 0);
lean_dec(v_unused_381_);
v___x_370_ = v___x_361_;
v_isShared_371_ = v_isSharedCheck_379_;
goto v_resetjp_369_;
}
else
{
lean_inc(v_snapshotTasks_368_);
lean_inc(v_infoState_367_);
lean_inc(v_messages_366_);
lean_inc(v_traceState_365_);
lean_inc(v_auxDeclNGen_364_);
lean_inc(v_ngen_363_);
lean_inc(v_nextMacroScope_362_);
lean_dec(v___x_361_);
v___x_370_ = lean_box(0);
v_isShared_371_ = v_isSharedCheck_379_;
goto v_resetjp_369_;
}
v_resetjp_369_:
{
lean_object* v___x_372_; lean_object* v___x_374_; 
v___x_372_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2, &lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___closed__2);
if (v_isShared_371_ == 0)
{
lean_ctor_set(v___x_370_, 5, v___x_372_);
lean_ctor_set(v___x_370_, 0, v_env_358_);
v___x_374_ = v___x_370_;
goto v_reusejp_373_;
}
else
{
lean_object* v_reuseFailAlloc_378_; 
v_reuseFailAlloc_378_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_378_, 0, v_env_358_);
lean_ctor_set(v_reuseFailAlloc_378_, 1, v_nextMacroScope_362_);
lean_ctor_set(v_reuseFailAlloc_378_, 2, v_ngen_363_);
lean_ctor_set(v_reuseFailAlloc_378_, 3, v_auxDeclNGen_364_);
lean_ctor_set(v_reuseFailAlloc_378_, 4, v_traceState_365_);
lean_ctor_set(v_reuseFailAlloc_378_, 5, v___x_372_);
lean_ctor_set(v_reuseFailAlloc_378_, 6, v_messages_366_);
lean_ctor_set(v_reuseFailAlloc_378_, 7, v_infoState_367_);
lean_ctor_set(v_reuseFailAlloc_378_, 8, v_snapshotTasks_368_);
v___x_374_ = v_reuseFailAlloc_378_;
goto v_reusejp_373_;
}
v_reusejp_373_:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; 
v___x_375_ = lean_st_ref_set(v___y_359_, v___x_374_);
v___x_376_ = lean_box(0);
v___x_377_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_377_, 0, v___x_376_);
return v___x_377_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object* v_env_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg(v_env_382_, v___y_383_);
lean_dec(v___y_383_);
return v_res_385_;
}
}
static lean_object* _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_387_; lean_object* v___x_388_; 
v___x_387_ = ((lean_object*)(lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__0));
v___x_388_ = l_Lean_stringToMessageData(v___x_387_);
return v___x_388_;
}
}
static lean_object* _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_390_ = ((lean_object*)(lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__2));
v___x_391_ = l_Lean_stringToMessageData(v___x_390_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg(lean_object* v_ext_392_, lean_object* v_k_393_, lean_object* v_v_394_, lean_object* v___y_395_, lean_object* v___y_396_){
_start:
{
lean_object* v___x_398_; lean_object* v_env_399_; lean_object* v___x_400_; 
v___x_398_ = lean_st_ref_get(v___y_396_);
v_env_399_ = lean_ctor_get(v___x_398_, 0);
lean_inc_ref(v_env_399_);
lean_dec(v___x_398_);
v___x_400_ = lp_batteries_Lean_NameMapExtension_find_x3f___redArg(v_ext_392_, v_env_399_, v_k_393_);
if (lean_obj_tag(v___x_400_) == 1)
{
lean_object* v_name_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; 
lean_dec_ref_known(v___x_400_, 1);
lean_dec(v_v_394_);
v_name_401_ = lean_ctor_get(v_ext_392_, 1);
lean_inc(v_name_401_);
lean_dec_ref(v_ext_392_);
v___x_402_ = lean_obj_once(&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__1, &lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__1);
v___x_403_ = l_Lean_MessageData_ofName(v_name_401_);
v___x_404_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_404_, 0, v___x_402_);
lean_ctor_set(v___x_404_, 1, v___x_403_);
v___x_405_ = lean_obj_once(&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__3, &lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__3_once, _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__3);
v___x_406_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_406_, 0, v___x_404_);
lean_ctor_set(v___x_406_, 1, v___x_405_);
v___x_407_ = l_Lean_MessageData_ofName(v_k_393_);
v___x_408_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_406_);
lean_ctor_set(v___x_408_, 1, v___x_407_);
v___x_409_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___redArg(v___x_408_, v___y_395_, v___y_396_);
return v___x_409_;
}
else
{
lean_object* v___x_410_; lean_object* v_toEnvExtension_411_; lean_object* v_env_412_; lean_object* v_asyncMode_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; 
lean_dec(v___x_400_);
v___x_410_ = lean_st_ref_get(v___y_396_);
v_toEnvExtension_411_ = lean_ctor_get(v_ext_392_, 0);
v_env_412_ = lean_ctor_get(v___x_410_, 0);
lean_inc_ref(v_env_412_);
lean_dec(v___x_410_);
v_asyncMode_413_ = lean_ctor_get(v_toEnvExtension_411_, 2);
lean_inc(v_asyncMode_413_);
v___x_414_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_414_, 0, v_k_393_);
lean_ctor_set(v___x_414_, 1, v_v_394_);
v___x_415_ = lean_box(0);
v___x_416_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v_ext_392_, v_env_412_, v___x_414_, v_asyncMode_413_, v___x_415_);
lean_dec(v_asyncMode_413_);
v___x_417_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_416_, v___y_396_);
return v___x_417_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v_ext_418_, lean_object* v_k_419_, lean_object* v_v_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_){
_start:
{
lean_object* v_res_424_; 
v_res_424_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg(v_ext_418_, v_k_419_, v_v_420_, v___y_421_, v___y_422_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
return v_res_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(lean_object* v_name_425_, lean_object* v_x_426_, uint8_t v_x_427_, lean_object* v___y_428_, lean_object* v___y_429_){
_start:
{
lean_object* v___x_431_; uint8_t v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; 
v___x_431_ = lp_mathlib_Mathlib_Tactic_ToAdditive_doTranslateAttr;
v___x_432_ = 1;
v___x_433_ = lean_box(v___x_432_);
v___x_434_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg(v___x_431_, v_name_425_, v___x_433_, v___y_428_, v___y_429_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed(lean_object* v_name_435_, lean_object* v_x_436_, lean_object* v_x_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_){
_start:
{
uint8_t v_x_1766__boxed_441_; lean_object* v_res_442_; 
v_x_1766__boxed_441_ = lean_unbox(v_x_437_);
v_res_442_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(v_name_435_, v_x_436_, v_x_1766__boxed_441_, v___y_438_, v___y_439_);
lean_dec(v___y_439_);
lean_dec_ref(v___y_438_);
lean_dec(v_x_436_);
return v_res_442_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_444_; lean_object* v___x_445_; 
v___x_444_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_445_ = l_Lean_stringToMessageData(v___x_444_);
return v___x_445_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_447_; lean_object* v___x_448_; 
v___x_447_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_448_ = l_Lean_stringToMessageData(v___x_447_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(lean_object* v___x_449_, lean_object* v_decl_450_, lean_object* v___y_451_, lean_object* v___y_452_){
_start:
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; 
v___x_454_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_455_ = l_Lean_MessageData_ofName(v___x_449_);
v___x_456_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_456_, 0, v___x_454_);
lean_ctor_set(v___x_456_, 1, v___x_455_);
v___x_457_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_458_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_458_, 0, v___x_456_);
lean_ctor_set(v___x_458_, 1, v___x_457_);
v___x_459_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___redArg(v___x_458_, v___y_451_, v___y_452_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed(lean_object* v___x_460_, lean_object* v_decl_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_){
_start:
{
lean_object* v_res_465_; 
v_res_465_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(v___x_460_, v_decl_461_, v___y_462_, v___y_463_);
lean_dec(v___y_463_);
lean_dec_ref(v___y_462_);
lean_dec(v_decl_461_);
return v_res_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(lean_object* v_name_466_, lean_object* v_x_467_, uint8_t v_x_468_, lean_object* v___y_469_, lean_object* v___y_470_){
_start:
{
lean_object* v___x_472_; uint8_t v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; 
v___x_472_ = lp_mathlib_Mathlib_Tactic_ToAdditive_doTranslateAttr;
v___x_473_ = 0;
v___x_474_ = lean_box(v___x_473_);
v___x_475_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg(v___x_472_, v_name_466_, v___x_474_, v___y_469_, v___y_470_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed(lean_object* v_name_476_, lean_object* v_x_477_, lean_object* v_x_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_){
_start:
{
uint8_t v_x_1831__boxed_482_; lean_object* v_res_483_; 
v_x_1831__boxed_482_ = lean_unbox(v_x_478_);
v_res_483_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(v_name_476_, v_x_477_, v_x_1831__boxed_482_, v___y_479_, v___y_480_);
lean_dec(v___y_480_);
lean_dec_ref(v___y_479_);
lean_dec(v_x_477_);
return v_res_483_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_534_ = lean_unsigned_to_nat(3091569921u);
v___x_535_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__19_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_536_ = l_Lean_Name_num___override(v___x_535_, v___x_534_);
return v___x_536_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; 
v___x_538_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__21_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_539_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__20_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_540_ = l_Lean_Name_str___override(v___x_539_, v___x_538_);
return v___x_540_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_542_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__23_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_543_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__22_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_544_ = l_Lean_Name_str___override(v___x_543_, v___x_542_);
return v___x_544_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; 
v___x_545_ = lean_unsigned_to_nat(2u);
v___x_546_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__24_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_547_ = l_Lean_Name_num___override(v___x_546_, v___x_545_);
return v___x_547_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; 
v___x_553_ = 0;
v___x_554_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__28_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_555_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__26_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_556_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_557_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_557_, 0, v___x_556_);
lean_ctor_set(v___x_557_, 1, v___x_555_);
lean_ctor_set(v___x_557_, 2, v___x_554_);
lean_ctor_set_uint8(v___x_557_, sizeof(void*)*3, v___x_553_);
return v___x_557_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_558_; lean_object* v___f_559_; lean_object* v___x_560_; lean_object* v___x_561_; 
v___f_558_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__27_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___f_559_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_560_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__29_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_561_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_561_, 0, v___x_560_);
lean_ctor_set(v___x_561_, 1, v___f_559_);
lean_ctor_set(v___x_561_, 2, v___f_558_);
return v___x_561_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; 
v___x_568_ = 0;
v___x_569_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__34_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_570_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__32_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_571_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__25_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_572_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_572_, 0, v___x_571_);
lean_ctor_set(v___x_572_, 1, v___x_570_);
lean_ctor_set(v___x_572_, 2, v___x_569_);
lean_ctor_set_uint8(v___x_572_, sizeof(void*)*3, v___x_568_);
return v___x_572_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_573_; lean_object* v___f_574_; lean_object* v___x_575_; lean_object* v___x_576_; 
v___f_573_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__33_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___f_574_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__31_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_));
v___x_575_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__35_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_576_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_576_, 0, v___x_575_);
lean_ctor_set(v___x_576_, 1, v___f_574_);
lean_ctor_set(v___x_576_, 2, v___f_573_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_578_; lean_object* v___x_579_; 
v___x_578_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__30_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_579_ = l_Lean_registerBuiltinAttribute(v___x_578_);
if (lean_obj_tag(v___x_579_) == 0)
{
lean_object* v___x_580_; lean_object* v___x_581_; 
lean_dec_ref_known(v___x_579_, 1);
v___x_580_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__36_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_581_ = l_Lean_registerBuiltinAttribute(v___x_580_);
return v___x_581_;
}
else
{
return v___x_579_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2____boxed(lean_object* v_a_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_();
return v_res_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_env_584_, lean_object* v___y_585_, lean_object* v___y_586_){
_start:
{
lean_object* v___x_588_; 
v___x_588_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___redArg(v_env_584_, v___y_586_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_env_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0_spec__0(v_env_589_, v___y_590_, v___y_591_);
lean_dec(v___y_591_);
lean_dec_ref(v___y_590_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_594_, lean_object* v_ext_595_, lean_object* v_k_596_, lean_object* v_v_597_, lean_object* v___y_598_, lean_object* v___y_599_){
_start:
{
lean_object* v___x_601_; 
v___x_601_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg(v_ext_595_, v_k_596_, v_v_597_, v___y_598_, v___y_599_);
return v___x_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_602_, lean_object* v_ext_603_, lean_object* v_k_604_, lean_object* v_v_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0(v_00_u03b1_602_, v_ext_603_, v_k_604_, v_v_605_, v___y_606_, v___y_607_);
lean_dec(v___y_607_);
lean_dec_ref(v___y_606_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b1_610_, lean_object* v_msg_611_, lean_object* v___y_612_, lean_object* v___y_613_){
_start:
{
lean_object* v___x_615_; 
v___x_615_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___redArg(v_msg_611_, v___y_612_, v___y_613_);
return v___x_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___boxed(lean_object* v_00_u03b1_616_, lean_object* v_msg_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_){
_start:
{
lean_object* v_res_621_; 
v_res_621_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1(v_00_u03b1_616_, v_msg_617_, v___y_618_, v___y_619_);
lean_dec(v___y_619_);
lean_dec_ref(v___y_618_);
return v_res_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_629_; lean_object* v___x_630_; 
v___x_629_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2_));
v___x_630_ = lp_batteries_Lean_registerNameMapExtension___redArg(v___x_629_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2____boxed(lean_object* v_a_631_){
_start:
{
lean_object* v_res_632_; 
v_res_632_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2_();
return v_res_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5___redArg(lean_object* v_x_633_, lean_object* v_x_634_){
_start:
{
if (lean_obj_tag(v_x_634_) == 0)
{
return v_x_633_;
}
else
{
lean_object* v_key_635_; lean_object* v_value_636_; lean_object* v_tail_637_; lean_object* v___x_639_; uint8_t v_isShared_640_; uint8_t v_isSharedCheck_660_; 
v_key_635_ = lean_ctor_get(v_x_634_, 0);
v_value_636_ = lean_ctor_get(v_x_634_, 1);
v_tail_637_ = lean_ctor_get(v_x_634_, 2);
v_isSharedCheck_660_ = !lean_is_exclusive(v_x_634_);
if (v_isSharedCheck_660_ == 0)
{
v___x_639_ = v_x_634_;
v_isShared_640_ = v_isSharedCheck_660_;
goto v_resetjp_638_;
}
else
{
lean_inc(v_tail_637_);
lean_inc(v_value_636_);
lean_inc(v_key_635_);
lean_dec(v_x_634_);
v___x_639_ = lean_box(0);
v_isShared_640_ = v_isSharedCheck_660_;
goto v_resetjp_638_;
}
v_resetjp_638_:
{
lean_object* v___x_641_; uint64_t v___x_642_; uint64_t v___x_643_; uint64_t v___x_644_; uint64_t v_fold_645_; uint64_t v___x_646_; uint64_t v___x_647_; uint64_t v___x_648_; size_t v___x_649_; size_t v___x_650_; size_t v___x_651_; size_t v___x_652_; size_t v___x_653_; lean_object* v___x_654_; lean_object* v___x_656_; 
v___x_641_ = lean_array_get_size(v_x_633_);
v___x_642_ = lean_string_hash(v_key_635_);
v___x_643_ = 32ULL;
v___x_644_ = lean_uint64_shift_right(v___x_642_, v___x_643_);
v_fold_645_ = lean_uint64_xor(v___x_642_, v___x_644_);
v___x_646_ = 16ULL;
v___x_647_ = lean_uint64_shift_right(v_fold_645_, v___x_646_);
v___x_648_ = lean_uint64_xor(v_fold_645_, v___x_647_);
v___x_649_ = lean_uint64_to_usize(v___x_648_);
v___x_650_ = lean_usize_of_nat(v___x_641_);
v___x_651_ = ((size_t)1ULL);
v___x_652_ = lean_usize_sub(v___x_650_, v___x_651_);
v___x_653_ = lean_usize_land(v___x_649_, v___x_652_);
v___x_654_ = lean_array_uget_borrowed(v_x_633_, v___x_653_);
lean_inc(v___x_654_);
if (v_isShared_640_ == 0)
{
lean_ctor_set(v___x_639_, 2, v___x_654_);
v___x_656_ = v___x_639_;
goto v_reusejp_655_;
}
else
{
lean_object* v_reuseFailAlloc_659_; 
v_reuseFailAlloc_659_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_659_, 0, v_key_635_);
lean_ctor_set(v_reuseFailAlloc_659_, 1, v_value_636_);
lean_ctor_set(v_reuseFailAlloc_659_, 2, v___x_654_);
v___x_656_ = v_reuseFailAlloc_659_;
goto v_reusejp_655_;
}
v_reusejp_655_:
{
lean_object* v___x_657_; 
v___x_657_ = lean_array_uset(v_x_633_, v___x_653_, v___x_656_);
v_x_633_ = v___x_657_;
v_x_634_ = v_tail_637_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3___redArg(lean_object* v_i_661_, lean_object* v_source_662_, lean_object* v_target_663_){
_start:
{
lean_object* v___x_664_; uint8_t v___x_665_; 
v___x_664_ = lean_array_get_size(v_source_662_);
v___x_665_ = lean_nat_dec_lt(v_i_661_, v___x_664_);
if (v___x_665_ == 0)
{
lean_dec_ref(v_source_662_);
lean_dec(v_i_661_);
return v_target_663_;
}
else
{
lean_object* v_es_666_; lean_object* v___x_667_; lean_object* v_source_668_; lean_object* v_target_669_; lean_object* v___x_670_; lean_object* v___x_671_; 
v_es_666_ = lean_array_fget(v_source_662_, v_i_661_);
v___x_667_ = lean_box(0);
v_source_668_ = lean_array_fset(v_source_662_, v_i_661_, v___x_667_);
v_target_669_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5___redArg(v_target_663_, v_es_666_);
v___x_670_ = lean_unsigned_to_nat(1u);
v___x_671_ = lean_nat_add(v_i_661_, v___x_670_);
lean_dec(v_i_661_);
v_i_661_ = v___x_671_;
v_source_662_ = v_source_668_;
v_target_663_ = v_target_669_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2___redArg(lean_object* v_data_673_){
_start:
{
lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v_nbuckets_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; 
v___x_674_ = lean_array_get_size(v_data_673_);
v___x_675_ = lean_unsigned_to_nat(2u);
v_nbuckets_676_ = lean_nat_mul(v___x_674_, v___x_675_);
v___x_677_ = lean_unsigned_to_nat(0u);
v___x_678_ = lean_box(0);
v___x_679_ = lean_mk_array(v_nbuckets_676_, v___x_678_);
v___x_680_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3___redArg(v___x_677_, v_data_673_, v___x_679_);
return v___x_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__3___redArg(lean_object* v_a_681_, lean_object* v_b_682_, lean_object* v_x_683_){
_start:
{
if (lean_obj_tag(v_x_683_) == 0)
{
lean_dec(v_b_682_);
lean_dec_ref(v_a_681_);
return v_x_683_;
}
else
{
lean_object* v_key_684_; lean_object* v_value_685_; lean_object* v_tail_686_; lean_object* v___x_688_; uint8_t v_isShared_689_; uint8_t v_isSharedCheck_698_; 
v_key_684_ = lean_ctor_get(v_x_683_, 0);
v_value_685_ = lean_ctor_get(v_x_683_, 1);
v_tail_686_ = lean_ctor_get(v_x_683_, 2);
v_isSharedCheck_698_ = !lean_is_exclusive(v_x_683_);
if (v_isSharedCheck_698_ == 0)
{
v___x_688_ = v_x_683_;
v_isShared_689_ = v_isSharedCheck_698_;
goto v_resetjp_687_;
}
else
{
lean_inc(v_tail_686_);
lean_inc(v_value_685_);
lean_inc(v_key_684_);
lean_dec(v_x_683_);
v___x_688_ = lean_box(0);
v_isShared_689_ = v_isSharedCheck_698_;
goto v_resetjp_687_;
}
v_resetjp_687_:
{
uint8_t v___x_690_; 
v___x_690_ = lean_string_dec_eq(v_key_684_, v_a_681_);
if (v___x_690_ == 0)
{
lean_object* v___x_691_; lean_object* v___x_693_; 
v___x_691_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__3___redArg(v_a_681_, v_b_682_, v_tail_686_);
if (v_isShared_689_ == 0)
{
lean_ctor_set(v___x_688_, 2, v___x_691_);
v___x_693_ = v___x_688_;
goto v_reusejp_692_;
}
else
{
lean_object* v_reuseFailAlloc_694_; 
v_reuseFailAlloc_694_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_694_, 0, v_key_684_);
lean_ctor_set(v_reuseFailAlloc_694_, 1, v_value_685_);
lean_ctor_set(v_reuseFailAlloc_694_, 2, v___x_691_);
v___x_693_ = v_reuseFailAlloc_694_;
goto v_reusejp_692_;
}
v_reusejp_692_:
{
return v___x_693_;
}
}
else
{
lean_object* v___x_696_; 
lean_dec(v_value_685_);
lean_dec(v_key_684_);
if (v_isShared_689_ == 0)
{
lean_ctor_set(v___x_688_, 1, v_b_682_);
lean_ctor_set(v___x_688_, 0, v_a_681_);
v___x_696_ = v___x_688_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v_a_681_);
lean_ctor_set(v_reuseFailAlloc_697_, 1, v_b_682_);
lean_ctor_set(v_reuseFailAlloc_697_, 2, v_tail_686_);
v___x_696_ = v_reuseFailAlloc_697_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
return v___x_696_;
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1___redArg(lean_object* v_a_699_, lean_object* v_x_700_){
_start:
{
if (lean_obj_tag(v_x_700_) == 0)
{
uint8_t v___x_701_; 
v___x_701_ = 0;
return v___x_701_;
}
else
{
lean_object* v_key_702_; lean_object* v_tail_703_; uint8_t v___x_704_; 
v_key_702_ = lean_ctor_get(v_x_700_, 0);
v_tail_703_ = lean_ctor_get(v_x_700_, 2);
v___x_704_ = lean_string_dec_eq(v_key_702_, v_a_699_);
if (v___x_704_ == 0)
{
v_x_700_ = v_tail_703_;
goto _start;
}
else
{
return v___x_704_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_a_706_, lean_object* v_x_707_){
_start:
{
uint8_t v_res_708_; lean_object* v_r_709_; 
v_res_708_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1___redArg(v_a_706_, v_x_707_);
lean_dec(v_x_707_);
lean_dec_ref(v_a_706_);
v_r_709_ = lean_box(v_res_708_);
return v_r_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0___redArg(lean_object* v_m_710_, lean_object* v_a_711_, lean_object* v_b_712_){
_start:
{
lean_object* v_size_713_; lean_object* v_buckets_714_; lean_object* v___x_716_; uint8_t v_isShared_717_; uint8_t v_isSharedCheck_757_; 
v_size_713_ = lean_ctor_get(v_m_710_, 0);
v_buckets_714_ = lean_ctor_get(v_m_710_, 1);
v_isSharedCheck_757_ = !lean_is_exclusive(v_m_710_);
if (v_isSharedCheck_757_ == 0)
{
v___x_716_ = v_m_710_;
v_isShared_717_ = v_isSharedCheck_757_;
goto v_resetjp_715_;
}
else
{
lean_inc(v_buckets_714_);
lean_inc(v_size_713_);
lean_dec(v_m_710_);
v___x_716_ = lean_box(0);
v_isShared_717_ = v_isSharedCheck_757_;
goto v_resetjp_715_;
}
v_resetjp_715_:
{
lean_object* v___x_718_; uint64_t v___x_719_; uint64_t v___x_720_; uint64_t v___x_721_; uint64_t v_fold_722_; uint64_t v___x_723_; uint64_t v___x_724_; uint64_t v___x_725_; size_t v___x_726_; size_t v___x_727_; size_t v___x_728_; size_t v___x_729_; size_t v___x_730_; lean_object* v_bkt_731_; uint8_t v___x_732_; 
v___x_718_ = lean_array_get_size(v_buckets_714_);
v___x_719_ = lean_string_hash(v_a_711_);
v___x_720_ = 32ULL;
v___x_721_ = lean_uint64_shift_right(v___x_719_, v___x_720_);
v_fold_722_ = lean_uint64_xor(v___x_719_, v___x_721_);
v___x_723_ = 16ULL;
v___x_724_ = lean_uint64_shift_right(v_fold_722_, v___x_723_);
v___x_725_ = lean_uint64_xor(v_fold_722_, v___x_724_);
v___x_726_ = lean_uint64_to_usize(v___x_725_);
v___x_727_ = lean_usize_of_nat(v___x_718_);
v___x_728_ = ((size_t)1ULL);
v___x_729_ = lean_usize_sub(v___x_727_, v___x_728_);
v___x_730_ = lean_usize_land(v___x_726_, v___x_729_);
v_bkt_731_ = lean_array_uget_borrowed(v_buckets_714_, v___x_730_);
v___x_732_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1___redArg(v_a_711_, v_bkt_731_);
if (v___x_732_ == 0)
{
lean_object* v___x_733_; lean_object* v_size_x27_734_; lean_object* v___x_735_; lean_object* v_buckets_x27_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; uint8_t v___x_742_; 
v___x_733_ = lean_unsigned_to_nat(1u);
v_size_x27_734_ = lean_nat_add(v_size_713_, v___x_733_);
lean_dec(v_size_713_);
lean_inc(v_bkt_731_);
v___x_735_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_735_, 0, v_a_711_);
lean_ctor_set(v___x_735_, 1, v_b_712_);
lean_ctor_set(v___x_735_, 2, v_bkt_731_);
v_buckets_x27_736_ = lean_array_uset(v_buckets_714_, v___x_730_, v___x_735_);
v___x_737_ = lean_unsigned_to_nat(4u);
v___x_738_ = lean_nat_mul(v_size_x27_734_, v___x_737_);
v___x_739_ = lean_unsigned_to_nat(3u);
v___x_740_ = lean_nat_div(v___x_738_, v___x_739_);
lean_dec(v___x_738_);
v___x_741_ = lean_array_get_size(v_buckets_x27_736_);
v___x_742_ = lean_nat_dec_le(v___x_740_, v___x_741_);
lean_dec(v___x_740_);
if (v___x_742_ == 0)
{
lean_object* v_val_743_; lean_object* v___x_745_; 
v_val_743_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2___redArg(v_buckets_x27_736_);
if (v_isShared_717_ == 0)
{
lean_ctor_set(v___x_716_, 1, v_val_743_);
lean_ctor_set(v___x_716_, 0, v_size_x27_734_);
v___x_745_ = v___x_716_;
goto v_reusejp_744_;
}
else
{
lean_object* v_reuseFailAlloc_746_; 
v_reuseFailAlloc_746_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_746_, 0, v_size_x27_734_);
lean_ctor_set(v_reuseFailAlloc_746_, 1, v_val_743_);
v___x_745_ = v_reuseFailAlloc_746_;
goto v_reusejp_744_;
}
v_reusejp_744_:
{
return v___x_745_;
}
}
else
{
lean_object* v___x_748_; 
if (v_isShared_717_ == 0)
{
lean_ctor_set(v___x_716_, 1, v_buckets_x27_736_);
lean_ctor_set(v___x_716_, 0, v_size_x27_734_);
v___x_748_ = v___x_716_;
goto v_reusejp_747_;
}
else
{
lean_object* v_reuseFailAlloc_749_; 
v_reuseFailAlloc_749_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_749_, 0, v_size_x27_734_);
lean_ctor_set(v_reuseFailAlloc_749_, 1, v_buckets_x27_736_);
v___x_748_ = v_reuseFailAlloc_749_;
goto v_reusejp_747_;
}
v_reusejp_747_:
{
return v___x_748_;
}
}
}
else
{
lean_object* v___x_750_; lean_object* v_buckets_x27_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_755_; 
lean_inc(v_bkt_731_);
v___x_750_ = lean_box(0);
v_buckets_x27_751_ = lean_array_uset(v_buckets_714_, v___x_730_, v___x_750_);
v___x_752_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__3___redArg(v_a_711_, v_b_712_, v_bkt_731_);
v___x_753_ = lean_array_uset(v_buckets_x27_751_, v___x_730_, v___x_752_);
if (v_isShared_717_ == 0)
{
lean_ctor_set(v___x_716_, 1, v___x_753_);
v___x_755_ = v___x_716_;
goto v_reusejp_754_;
}
else
{
lean_object* v_reuseFailAlloc_756_; 
v_reuseFailAlloc_756_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_756_, 0, v_size_713_);
lean_ctor_set(v_reuseFailAlloc_756_, 1, v___x_753_);
v___x_755_ = v_reuseFailAlloc_756_;
goto v_reusejp_754_;
}
v_reusejp_754_:
{
return v___x_755_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1___redArg(lean_object* v_as_x27_758_, lean_object* v_b_759_){
_start:
{
if (lean_obj_tag(v_as_x27_758_) == 0)
{
return v_b_759_;
}
else
{
lean_object* v_head_760_; lean_object* v_tail_761_; lean_object* v_fst_762_; lean_object* v_snd_763_; lean_object* v_r_764_; 
v_head_760_ = lean_ctor_get(v_as_x27_758_, 0);
v_tail_761_ = lean_ctor_get(v_as_x27_758_, 1);
v_fst_762_ = lean_ctor_get(v_head_760_, 0);
v_snd_763_ = lean_ctor_get(v_head_760_, 1);
lean_inc(v_snd_763_);
lean_inc(v_fst_762_);
v_r_764_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0___redArg(v_b_759_, v_fst_762_, v_snd_763_);
v_as_x27_758_ = v_tail_761_;
v_b_759_ = v_r_764_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1___redArg___boxed(lean_object* v_as_x27_766_, lean_object* v_b_767_){
_start:
{
lean_object* v_res_768_; 
v_res_768_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1___redArg(v_as_x27_766_, v_b_767_);
lean_dec(v_as_x27_766_);
return v_res_768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0(lean_object* v_m_769_, lean_object* v_l_770_){
_start:
{
lean_object* v___x_771_; 
v___x_771_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1___redArg(v_l_770_, v_m_769_);
return v___x_771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0___boxed(lean_object* v_m_772_, lean_object* v_l_773_){
_start:
{
lean_object* v_res_774_; 
v_res_774_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0(v_m_772_, v_l_773_);
lean_dec(v_l_773_);
return v_res_774_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__249(void){
_start:
{
lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; 
v___x_1340_ = lean_box(0);
v___x_1341_ = lean_unsigned_to_nat(16u);
v___x_1342_ = lean_mk_array(v___x_1341_, v___x_1340_);
return v___x_1342_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__250(void){
_start:
{
lean_object* v___x_1343_; lean_object* v___x_1344_; lean_object* v___x_1345_; 
v___x_1343_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__249, &lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__249_once, _init_lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__249);
v___x_1344_ = lean_unsigned_to_nat(0u);
v___x_1345_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1345_, 0, v___x_1344_);
lean_ctor_set(v___x_1345_, 1, v___x_1343_);
return v___x_1345_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__251(void){
_start:
{
lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; 
v___x_1346_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__250, &lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__250_once, _init_lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__250);
v___x_1347_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__248));
v___x_1348_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1___redArg(v___x_1347_, v___x_1346_);
return v___x_1348_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict(void){
_start:
{
lean_object* v___x_1349_; 
v___x_1349_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__251, &lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__251_once, _init_lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__251);
return v___x_1349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0(lean_object* v_00_u03b2_1350_, lean_object* v_m_1351_, lean_object* v_a_1352_, lean_object* v_b_1353_){
_start:
{
lean_object* v___x_1354_; 
v___x_1354_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0___redArg(v_m_1351_, v_a_1352_, v_b_1353_);
return v___x_1354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1(lean_object* v_as_1355_, lean_object* v_as_x27_1356_, lean_object* v_b_1357_, lean_object* v_a_1358_){
_start:
{
lean_object* v___x_1359_; 
v___x_1359_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1___redArg(v_as_x27_1356_, v_b_1357_);
return v___x_1359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1___boxed(lean_object* v_as_1360_, lean_object* v_as_x27_1361_, lean_object* v_b_1362_, lean_object* v_a_1363_){
_start:
{
lean_object* v_res_1364_; 
v_res_1364_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__1(v_as_1360_, v_as_x27_1361_, v_b_1362_, v_a_1363_);
lean_dec(v_as_x27_1361_);
lean_dec(v_as_1360_);
return v_res_1364_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_1365_, lean_object* v_a_1366_, lean_object* v_x_1367_){
_start:
{
uint8_t v___x_1368_; 
v___x_1368_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1___redArg(v_a_1366_, v_x_1367_);
return v___x_1368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_1369_, lean_object* v_a_1370_, lean_object* v_x_1371_){
_start:
{
uint8_t v_res_1372_; lean_object* v_r_1373_; 
v_res_1372_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__1(v_00_u03b2_1369_, v_a_1370_, v_x_1371_);
lean_dec(v_x_1371_);
lean_dec_ref(v_a_1370_);
v_r_1373_ = lean_box(v_res_1372_);
return v_r_1373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_1374_, lean_object* v_data_1375_){
_start:
{
lean_object* v___x_1376_; 
v___x_1376_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2___redArg(v_data_1375_);
return v___x_1376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__3(lean_object* v_00_u03b2_1377_, lean_object* v_a_1378_, lean_object* v_b_1379_, lean_object* v_x_1380_){
_start:
{
lean_object* v___x_1381_; 
v___x_1381_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__3___redArg(v_a_1378_, v_b_1379_, v_x_1380_);
return v___x_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3(lean_object* v_00_u03b2_1382_, lean_object* v_i_1383_, lean_object* v_source_1384_, lean_object* v_target_1385_){
_start:
{
lean_object* v___x_1386_; 
v___x_1386_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3___redArg(v_i_1383_, v_source_1384_, v_target_1385_);
return v___x_1386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_1387_, lean_object* v_x_1388_, lean_object* v_x_1389_){
_start:
{
lean_object* v___x_1390_; 
v___x_1390_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0_spec__2_spec__3_spec__5___redArg(v_x_1388_, v_x_1389_);
return v___x_1390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0___redArg(lean_object* v_as_x27_1391_, lean_object* v_b_1392_){
_start:
{
if (lean_obj_tag(v_as_x27_1391_) == 0)
{
return v_b_1392_;
}
else
{
lean_object* v_head_1393_; lean_object* v_tail_1394_; lean_object* v_fst_1395_; lean_object* v_snd_1396_; lean_object* v_r_1397_; 
v_head_1393_ = lean_ctor_get(v_as_x27_1391_, 0);
v_tail_1394_ = lean_ctor_get(v_as_x27_1391_, 1);
v_fst_1395_ = lean_ctor_get(v_head_1393_, 0);
v_snd_1396_ = lean_ctor_get(v_head_1393_, 1);
lean_inc(v_snd_1396_);
lean_inc(v_fst_1395_);
v_r_1397_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_nameDict_spec__0_spec__0___redArg(v_b_1392_, v_fst_1395_, v_snd_1396_);
v_as_x27_1391_ = v_tail_1394_;
v_b_1392_ = v_r_1397_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0___redArg___boxed(lean_object* v_as_x27_1399_, lean_object* v_b_1400_){
_start:
{
lean_object* v_res_1401_; 
v_res_1401_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0___redArg(v_as_x27_1399_, v_b_1400_);
lean_dec(v_as_x27_1399_);
return v_res_1401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0(lean_object* v_m_1402_, lean_object* v_l_1403_){
_start:
{
lean_object* v___x_1404_; 
v___x_1404_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0___redArg(v_l_1403_, v_m_1402_);
return v___x_1404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0___boxed(lean_object* v_m_1405_, lean_object* v_l_1406_){
_start:
{
lean_object* v_res_1407_; 
v_res_1407_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0(v_m_1405_, v_l_1406_);
lean_dec(v_l_1406_);
return v_res_1407_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__218(void){
_start:
{
lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; 
v___x_1850_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__250, &lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__250_once, _init_lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict___closed__250);
v___x_1851_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__217));
v___x_1852_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0___redArg(v___x_1851_, v___x_1850_);
return v___x_1852_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict(void){
_start:
{
lean_object* v___x_1853_; 
v___x_1853_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__218, &lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__218_once, _init_lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict___closed__218);
return v___x_1853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0(lean_object* v_as_1854_, lean_object* v_as_x27_1855_, lean_object* v_b_1856_, lean_object* v_a_1857_){
_start:
{
lean_object* v___x_1858_; 
v___x_1858_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0___redArg(v_as_x27_1855_, v_b_1856_);
return v___x_1858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0___boxed(lean_object* v_as_1859_, lean_object* v_as_x27_1860_, lean_object* v_b_1861_, lean_object* v_a_1862_){
_start:
{
lean_object* v_res_1863_; 
v_res_1863_ = lp_mathlib_List_forIn_x27_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertMany___at___00Mathlib_Tactic_ToAdditive_abbreviationDict_spec__0_spec__0(v_as_1859_, v_as_x27_1860_, v_b_1861_, v_a_1862_);
lean_dec(v_as_x27_1860_);
lean_dec(v_as_1859_);
return v_res_1863_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; 
v___x_1864_ = lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict;
v___x_1865_ = lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict;
v___x_1866_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1866_, 0, v___x_1865_);
lean_ctor_set(v___x_1866_, 1, v___x_1864_);
return v___x_1866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1868_; lean_object* v___x_1869_; 
v___x_1868_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2_);
v___x_1869_ = lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt(v___x_1868_);
return v___x_1869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2____boxed(lean_object* v_a_1870_){
_start:
{
lean_object* v_res_1871_; 
v_res_1871_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2_();
return v_res_1871_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__1(void){
_start:
{
lean_object* v___x_1874_; uint8_t v___x_1875_; uint8_t v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; 
v___x_1874_ = lp_mathlib_Mathlib_Tactic_ToAdditive_guessNameExt;
v___x_1875_ = 0;
v___x_1876_ = 1;
v___x_1877_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__0));
v___x_1878_ = lp_mathlib_Mathlib_Tactic_ToAdditive_translations;
v___x_1879_ = lean_box(0);
v___x_1880_ = lp_mathlib_Mathlib_Tactic_ToAdditive_doTranslateAttr;
v___x_1881_ = lp_mathlib_Mathlib_Tactic_ToAdditive_ignoreArgsAttr;
v___x_1882_ = lean_alloc_ctor(0, 6, 2);
lean_ctor_set(v___x_1882_, 0, v___x_1881_);
lean_ctor_set(v___x_1882_, 1, v___x_1880_);
lean_ctor_set(v___x_1882_, 2, v___x_1879_);
lean_ctor_set(v___x_1882_, 3, v___x_1878_);
lean_ctor_set(v___x_1882_, 4, v___x_1877_);
lean_ctor_set(v___x_1882_, 5, v___x_1874_);
lean_ctor_set_uint8(v___x_1882_, sizeof(void*)*6, v___x_1876_);
lean_ctor_set_uint8(v___x_1882_, sizeof(void*)*6 + 1, v___x_1875_);
return v___x_1882_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ToAdditive_data(void){
_start:
{
lean_object* v___x_1883_; 
v___x_1883_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__1, &lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ToAdditive_data___closed__1);
return v___x_1883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0___redArg(lean_object* v_category_1884_, lean_object* v_opts_1885_, lean_object* v_act_1886_, lean_object* v_decl_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_){
_start:
{
lean_object* v___x_1891_; lean_object* v___x_1892_; 
lean_inc(v___y_1889_);
lean_inc_ref(v___y_1888_);
v___x_1891_ = lean_apply_2(v_act_1886_, v___y_1888_, v___y_1889_);
v___x_1892_ = l_Lean_profileitIOUnsafe___redArg(v_category_1884_, v_opts_1885_, v___x_1891_, v_decl_1887_);
return v___x_1892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v_category_1893_, lean_object* v_opts_1894_, lean_object* v_act_1895_, lean_object* v_decl_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_, lean_object* v___y_1899_){
_start:
{
lean_object* v_res_1900_; 
v_res_1900_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0___redArg(v_category_1893_, v_opts_1894_, v_act_1895_, v_decl_1896_, v___y_1897_, v___y_1898_);
lean_dec(v___y_1898_);
lean_dec_ref(v___y_1897_);
lean_dec_ref(v_opts_1894_);
lean_dec_ref(v_category_1893_);
return v_res_1900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_1901_, lean_object* v_category_1902_, lean_object* v_opts_1903_, lean_object* v_act_1904_, lean_object* v_decl_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_){
_start:
{
lean_object* v___x_1909_; 
v___x_1909_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0___redArg(v_category_1902_, v_opts_1903_, v_act_1904_, v_decl_1905_, v___y_1906_, v___y_1907_);
return v___x_1909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_1910_, lean_object* v_category_1911_, lean_object* v_opts_1912_, lean_object* v_act_1913_, lean_object* v_decl_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_){
_start:
{
lean_object* v_res_1918_; 
v_res_1918_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0(v_00_u03b1_1910_, v_category_1911_, v_opts_1912_, v_act_1913_, v_decl_1914_, v___y_1915_, v___y_1916_);
lean_dec(v___y_1916_);
lean_dec_ref(v___y_1915_);
lean_dec_ref(v_opts_1912_);
lean_dec_ref(v_category_1911_);
return v_res_1918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_(lean_object* v_src_1919_, lean_object* v_stx_1920_, uint8_t v_kind_1921_, lean_object* v___y_1922_, lean_object* v___y_1923_){
_start:
{
lean_object* v___x_1925_; 
lean_inc(v_src_1919_);
v___x_1925_ = lp_mathlib_Mathlib_Tactic_Translate_elabTranslationAttr(v_src_1919_, v_stx_1920_, v___y_1922_, v___y_1923_);
if (lean_obj_tag(v___x_1925_) == 0)
{
lean_object* v_a_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; 
v_a_1926_ = lean_ctor_get(v___x_1925_, 0);
lean_inc(v_a_1926_);
lean_dec_ref_known(v___x_1925_, 1);
v___x_1927_ = lp_mathlib_Mathlib_Tactic_ToAdditive_data;
v___x_1928_ = lp_mathlib_Mathlib_Tactic_Translate_addTranslationAttr(v___x_1927_, v_src_1919_, v_a_1926_, v_kind_1921_, v___y_1922_, v___y_1923_);
return v___x_1928_;
}
else
{
lean_object* v_a_1929_; lean_object* v___x_1931_; uint8_t v_isShared_1932_; uint8_t v_isSharedCheck_1936_; 
lean_dec(v_src_1919_);
v_a_1929_ = lean_ctor_get(v___x_1925_, 0);
v_isSharedCheck_1936_ = !lean_is_exclusive(v___x_1925_);
if (v_isSharedCheck_1936_ == 0)
{
v___x_1931_ = v___x_1925_;
v_isShared_1932_ = v_isSharedCheck_1936_;
goto v_resetjp_1930_;
}
else
{
lean_inc(v_a_1929_);
lean_dec(v___x_1925_);
v___x_1931_ = lean_box(0);
v_isShared_1932_ = v_isSharedCheck_1936_;
goto v_resetjp_1930_;
}
v_resetjp_1930_:
{
lean_object* v___x_1934_; 
if (v_isShared_1932_ == 0)
{
v___x_1934_ = v___x_1931_;
goto v_reusejp_1933_;
}
else
{
lean_object* v_reuseFailAlloc_1935_; 
v_reuseFailAlloc_1935_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1935_, 0, v_a_1929_);
v___x_1934_ = v_reuseFailAlloc_1935_;
goto v_reusejp_1933_;
}
v_reusejp_1933_:
{
return v___x_1934_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed(lean_object* v_src_1937_, lean_object* v_stx_1938_, lean_object* v_kind_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_){
_start:
{
uint8_t v_kind_boxed_1943_; lean_object* v_res_1944_; 
v_kind_boxed_1943_ = lean_unbox(v_kind_1939_);
v_res_1944_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_(v_src_1937_, v_stx_1938_, v_kind_boxed_1943_, v___y_1940_, v___y_1941_);
lean_dec(v___y_1941_);
lean_dec_ref(v___y_1940_);
return v_res_1944_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_(lean_object* v___x_1945_, lean_object* v___x_1946_, lean_object* v_src_1947_, lean_object* v_stx_1948_, uint8_t v_kind_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_){
_start:
{
lean_object* v_options_1953_; lean_object* v___x_1954_; lean_object* v___f_1955_; lean_object* v___x_1956_; 
v_options_1953_ = lean_ctor_get(v___y_1950_, 2);
v___x_1954_ = lean_box(v_kind_1949_);
v___f_1955_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__0_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed), 6, 3);
lean_closure_set(v___f_1955_, 0, v_src_1947_);
lean_closure_set(v___f_1955_, 1, v_stx_1948_);
lean_closure_set(v___f_1955_, 2, v___x_1954_);
v___x_1956_ = lp_mathlib_Lean_profileitM___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2__spec__0___redArg(v___x_1945_, v_options_1953_, v___f_1955_, v___x_1946_, v___y_1950_, v___y_1951_);
if (lean_obj_tag(v___x_1956_) == 0)
{
lean_object* v___x_1958_; uint8_t v_isShared_1959_; uint8_t v_isSharedCheck_1964_; 
v_isSharedCheck_1964_ = !lean_is_exclusive(v___x_1956_);
if (v_isSharedCheck_1964_ == 0)
{
lean_object* v_unused_1965_; 
v_unused_1965_ = lean_ctor_get(v___x_1956_, 0);
lean_dec(v_unused_1965_);
v___x_1958_ = v___x_1956_;
v_isShared_1959_ = v_isSharedCheck_1964_;
goto v_resetjp_1957_;
}
else
{
lean_dec(v___x_1956_);
v___x_1958_ = lean_box(0);
v_isShared_1959_ = v_isSharedCheck_1964_;
goto v_resetjp_1957_;
}
v_resetjp_1957_:
{
lean_object* v___x_1960_; lean_object* v___x_1962_; 
v___x_1960_ = lean_box(0);
if (v_isShared_1959_ == 0)
{
lean_ctor_set(v___x_1958_, 0, v___x_1960_);
v___x_1962_ = v___x_1958_;
goto v_reusejp_1961_;
}
else
{
lean_object* v_reuseFailAlloc_1963_; 
v_reuseFailAlloc_1963_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1963_, 0, v___x_1960_);
v___x_1962_ = v_reuseFailAlloc_1963_;
goto v_reusejp_1961_;
}
v_reusejp_1961_:
{
return v___x_1962_;
}
}
}
else
{
lean_object* v_a_1966_; lean_object* v___x_1968_; uint8_t v_isShared_1969_; uint8_t v_isSharedCheck_1973_; 
v_a_1966_ = lean_ctor_get(v___x_1956_, 0);
v_isSharedCheck_1973_ = !lean_is_exclusive(v___x_1956_);
if (v_isSharedCheck_1973_ == 0)
{
v___x_1968_ = v___x_1956_;
v_isShared_1969_ = v_isSharedCheck_1973_;
goto v_resetjp_1967_;
}
else
{
lean_inc(v_a_1966_);
lean_dec(v___x_1956_);
v___x_1968_ = lean_box(0);
v_isShared_1969_ = v_isSharedCheck_1973_;
goto v_resetjp_1967_;
}
v_resetjp_1967_:
{
lean_object* v___x_1971_; 
if (v_isShared_1969_ == 0)
{
v___x_1971_ = v___x_1968_;
goto v_reusejp_1970_;
}
else
{
lean_object* v_reuseFailAlloc_1972_; 
v_reuseFailAlloc_1972_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1972_, 0, v_a_1966_);
v___x_1971_ = v_reuseFailAlloc_1972_;
goto v_reusejp_1970_;
}
v_reusejp_1970_:
{
return v___x_1971_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed(lean_object* v___x_1974_, lean_object* v___x_1975_, lean_object* v_src_1976_, lean_object* v_stx_1977_, lean_object* v_kind_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_){
_start:
{
uint8_t v_kind_boxed_1982_; lean_object* v_res_1983_; 
v_kind_boxed_1982_ = lean_unbox(v_kind_1978_);
v_res_1983_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_(v___x_1974_, v___x_1975_, v_src_1976_, v_stx_1977_, v_kind_boxed_1982_, v___y_1979_, v___y_1980_);
lean_dec(v___y_1980_);
lean_dec_ref(v___y_1979_);
lean_dec_ref(v___x_1974_);
return v_res_1983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_(lean_object* v___x_1984_, lean_object* v_decl_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_){
_start:
{
lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; 
v___x_1989_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_1990_ = l_Lean_MessageData_ofName(v___x_1984_);
v___x_1991_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1991_, 0, v___x_1989_);
lean_ctor_set(v___x_1991_, 1, v___x_1990_);
v___x_1992_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_);
v___x_1993_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1993_, 0, v___x_1991_);
lean_ctor_set(v___x_1993_, 1, v___x_1992_);
v___x_1994_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1___redArg(v___x_1993_, v___y_1986_, v___y_1987_);
return v___x_1994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed(lean_object* v___x_1995_, lean_object* v_decl_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_){
_start:
{
lean_object* v_res_2000_; 
v_res_2000_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___lam__2_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_(v___x_1995_, v_decl_1996_, v___y_1997_, v___y_1998_);
lean_dec(v___y_1998_);
lean_dec_ref(v___y_1997_);
lean_dec(v_decl_1996_);
return v_res_2000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2029_; lean_object* v___x_2030_; 
v___x_2029_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn___closed__8_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_));
v___x_2030_ = l_Lean_registerBuiltinAttribute(v___x_2029_);
return v___x_2030_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2____boxed(lean_object* v_a_2031_){
_start:
{
lean_object* v_res_2032_; 
v_res_2032_ = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_();
return v_res_2032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0___redArg(){
_start:
{
lean_object* v___x_2061_; lean_object* v___x_2062_; 
v___x_2061_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2__spec__0___redArg___closed__0);
v___x_2062_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2062_, 0, v___x_2061_);
return v___x_2062_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0___redArg___boxed(lean_object* v___y_2063_){
_start:
{
lean_object* v_res_2064_; 
v_res_2064_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0___redArg();
return v_res_2064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0(lean_object* v_00_u03b1_2065_, lean_object* v___y_2066_, lean_object* v___y_2067_){
_start:
{
lean_object* v___x_2069_; 
v___x_2069_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0___redArg();
return v___x_2069_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0___boxed(lean_object* v_00_u03b1_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_){
_start:
{
lean_object* v_res_2074_; 
v_res_2074_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0(v_00_u03b1_2070_, v___y_2071_, v___y_2072_);
lean_dec(v___y_2072_);
lean_dec_ref(v___y_2071_);
return v_res_2074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2___redArg(lean_object* v_msgData_2075_, lean_object* v___y_2076_){
_start:
{
lean_object* v___x_2078_; lean_object* v_env_2079_; lean_object* v___x_2080_; lean_object* v_scopes_2081_; lean_object* v___x_2082_; lean_object* v___x_2083_; lean_object* v_opts_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2090_; lean_object* v___x_2091_; 
v___x_2078_ = lean_st_ref_get(v___y_2076_);
v_env_2079_ = lean_ctor_get(v___x_2078_, 0);
lean_inc_ref(v_env_2079_);
lean_dec(v___x_2078_);
v___x_2080_ = lean_st_ref_get(v___y_2076_);
v_scopes_2081_ = lean_ctor_get(v___x_2080_, 2);
lean_inc(v_scopes_2081_);
lean_dec(v___x_2080_);
v___x_2082_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2083_ = l_List_head_x21___redArg(v___x_2082_, v_scopes_2081_);
lean_dec(v_scopes_2081_);
v_opts_2084_ = lean_ctor_get(v___x_2083_, 1);
lean_inc_ref(v_opts_2084_);
lean_dec(v___x_2083_);
v___x_2085_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__2);
v___x_2086_ = lean_unsigned_to_nat(32u);
v___x_2087_ = lean_mk_empty_array_with_capacity(v___x_2086_);
lean_dec_ref(v___x_2087_);
v___x_2088_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__1_spec__2___closed__5);
v___x_2089_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2089_, 0, v_env_2079_);
lean_ctor_set(v___x_2089_, 1, v___x_2085_);
lean_ctor_set(v___x_2089_, 2, v___x_2088_);
lean_ctor_set(v___x_2089_, 3, v_opts_2084_);
v___x_2090_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2090_, 0, v___x_2089_);
lean_ctor_set(v___x_2090_, 1, v_msgData_2075_);
v___x_2091_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2091_, 0, v___x_2090_);
return v___x_2091_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_msgData_2092_, lean_object* v___y_2093_, lean_object* v___y_2094_){
_start:
{
lean_object* v_res_2095_; 
v_res_2095_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2___redArg(v_msgData_2092_, v___y_2093_);
lean_dec(v___y_2093_);
return v_res_2095_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__0(void){
_start:
{
lean_object* v___x_2096_; lean_object* v___x_2097_; 
v___x_2096_ = lean_box(1);
v___x_2097_ = l_Lean_MessageData_ofFormat(v___x_2096_);
return v___x_2097_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__3(void){
_start:
{
lean_object* v___x_2101_; lean_object* v___x_2102_; 
v___x_2101_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__2));
v___x_2102_ = l_Lean_MessageData_ofFormat(v___x_2101_);
return v___x_2102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6(lean_object* v_x_2103_, lean_object* v_x_2104_){
_start:
{
if (lean_obj_tag(v_x_2104_) == 0)
{
return v_x_2103_;
}
else
{
lean_object* v_head_2105_; lean_object* v_tail_2106_; lean_object* v___x_2108_; uint8_t v_isShared_2109_; uint8_t v_isSharedCheck_2128_; 
v_head_2105_ = lean_ctor_get(v_x_2104_, 0);
v_tail_2106_ = lean_ctor_get(v_x_2104_, 1);
v_isSharedCheck_2128_ = !lean_is_exclusive(v_x_2104_);
if (v_isSharedCheck_2128_ == 0)
{
v___x_2108_ = v_x_2104_;
v_isShared_2109_ = v_isSharedCheck_2128_;
goto v_resetjp_2107_;
}
else
{
lean_inc(v_tail_2106_);
lean_inc(v_head_2105_);
lean_dec(v_x_2104_);
v___x_2108_ = lean_box(0);
v_isShared_2109_ = v_isSharedCheck_2128_;
goto v_resetjp_2107_;
}
v_resetjp_2107_:
{
lean_object* v_before_2110_; lean_object* v___x_2112_; uint8_t v_isShared_2113_; uint8_t v_isSharedCheck_2126_; 
v_before_2110_ = lean_ctor_get(v_head_2105_, 0);
v_isSharedCheck_2126_ = !lean_is_exclusive(v_head_2105_);
if (v_isSharedCheck_2126_ == 0)
{
lean_object* v_unused_2127_; 
v_unused_2127_ = lean_ctor_get(v_head_2105_, 1);
lean_dec(v_unused_2127_);
v___x_2112_ = v_head_2105_;
v_isShared_2113_ = v_isSharedCheck_2126_;
goto v_resetjp_2111_;
}
else
{
lean_inc(v_before_2110_);
lean_dec(v_head_2105_);
v___x_2112_ = lean_box(0);
v_isShared_2113_ = v_isSharedCheck_2126_;
goto v_resetjp_2111_;
}
v_resetjp_2111_:
{
lean_object* v___x_2114_; lean_object* v___x_2116_; 
v___x_2114_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__0);
if (v_isShared_2113_ == 0)
{
lean_ctor_set_tag(v___x_2112_, 7);
lean_ctor_set(v___x_2112_, 1, v___x_2114_);
lean_ctor_set(v___x_2112_, 0, v_x_2103_);
v___x_2116_ = v___x_2112_;
goto v_reusejp_2115_;
}
else
{
lean_object* v_reuseFailAlloc_2125_; 
v_reuseFailAlloc_2125_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2125_, 0, v_x_2103_);
lean_ctor_set(v_reuseFailAlloc_2125_, 1, v___x_2114_);
v___x_2116_ = v_reuseFailAlloc_2125_;
goto v_reusejp_2115_;
}
v_reusejp_2115_:
{
lean_object* v___x_2117_; lean_object* v___x_2119_; 
v___x_2117_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__3);
if (v_isShared_2109_ == 0)
{
lean_ctor_set_tag(v___x_2108_, 7);
lean_ctor_set(v___x_2108_, 1, v___x_2117_);
lean_ctor_set(v___x_2108_, 0, v___x_2116_);
v___x_2119_ = v___x_2108_;
goto v_reusejp_2118_;
}
else
{
lean_object* v_reuseFailAlloc_2124_; 
v_reuseFailAlloc_2124_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2124_, 0, v___x_2116_);
lean_ctor_set(v_reuseFailAlloc_2124_, 1, v___x_2117_);
v___x_2119_ = v_reuseFailAlloc_2124_;
goto v_reusejp_2118_;
}
v_reusejp_2118_:
{
lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; 
v___x_2120_ = l_Lean_MessageData_ofSyntax(v_before_2110_);
v___x_2121_ = l_Lean_indentD(v___x_2120_);
v___x_2122_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2122_, 0, v___x_2119_);
lean_ctor_set(v___x_2122_, 1, v___x_2121_);
v_x_2103_ = v___x_2122_;
v_x_2104_ = v_tail_2106_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__5(lean_object* v_opts_2129_, lean_object* v_opt_2130_){
_start:
{
lean_object* v_name_2131_; lean_object* v_defValue_2132_; lean_object* v_map_2133_; lean_object* v___x_2134_; 
v_name_2131_ = lean_ctor_get(v_opt_2130_, 0);
v_defValue_2132_ = lean_ctor_get(v_opt_2130_, 1);
v_map_2133_ = lean_ctor_get(v_opts_2129_, 0);
v___x_2134_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2133_, v_name_2131_);
if (lean_obj_tag(v___x_2134_) == 0)
{
uint8_t v___x_2135_; 
v___x_2135_ = lean_unbox(v_defValue_2132_);
return v___x_2135_;
}
else
{
lean_object* v_val_2136_; 
v_val_2136_ = lean_ctor_get(v___x_2134_, 0);
lean_inc(v_val_2136_);
lean_dec_ref_known(v___x_2134_, 1);
if (lean_obj_tag(v_val_2136_) == 1)
{
uint8_t v_v_2137_; 
v_v_2137_ = lean_ctor_get_uint8(v_val_2136_, 0);
lean_dec_ref_known(v_val_2136_, 0);
return v_v_2137_;
}
else
{
uint8_t v___x_2138_; 
lean_dec(v_val_2136_);
v___x_2138_ = lean_unbox(v_defValue_2132_);
return v___x_2138_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__5___boxed(lean_object* v_opts_2139_, lean_object* v_opt_2140_){
_start:
{
uint8_t v_res_2141_; lean_object* v_r_2142_; 
v_res_2141_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__5(v_opts_2139_, v_opt_2140_);
lean_dec_ref(v_opt_2140_);
lean_dec_ref(v_opts_2139_);
v_r_2142_ = lean_box(v_res_2141_);
return v_r_2142_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_2146_; lean_object* v___x_2147_; 
v___x_2146_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__1));
v___x_2147_ = l_Lean_MessageData_ofFormat(v___x_2146_);
return v___x_2147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg(lean_object* v_msgData_2148_, lean_object* v_macroStack_2149_, lean_object* v___y_2150_){
_start:
{
lean_object* v___x_2152_; lean_object* v_scopes_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; lean_object* v_opts_2156_; lean_object* v___x_2157_; uint8_t v___x_2158_; 
v___x_2152_ = lean_st_ref_get(v___y_2150_);
v_scopes_2153_ = lean_ctor_get(v___x_2152_, 2);
lean_inc(v_scopes_2153_);
lean_dec(v___x_2152_);
v___x_2154_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2155_ = l_List_head_x21___redArg(v___x_2154_, v_scopes_2153_);
lean_dec(v_scopes_2153_);
v_opts_2156_ = lean_ctor_get(v___x_2155_, 1);
lean_inc_ref(v_opts_2156_);
lean_dec(v___x_2155_);
v___x_2157_ = l_Lean_Elab_pp_macroStack;
v___x_2158_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__5(v_opts_2156_, v___x_2157_);
lean_dec_ref(v_opts_2156_);
if (v___x_2158_ == 0)
{
lean_object* v___x_2159_; 
lean_dec(v_macroStack_2149_);
v___x_2159_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2159_, 0, v_msgData_2148_);
return v___x_2159_;
}
else
{
if (lean_obj_tag(v_macroStack_2149_) == 0)
{
lean_object* v___x_2160_; 
v___x_2160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2160_, 0, v_msgData_2148_);
return v___x_2160_;
}
else
{
lean_object* v_head_2161_; lean_object* v_after_2162_; lean_object* v___x_2164_; uint8_t v_isShared_2165_; uint8_t v_isSharedCheck_2177_; 
v_head_2161_ = lean_ctor_get(v_macroStack_2149_, 0);
lean_inc(v_head_2161_);
v_after_2162_ = lean_ctor_get(v_head_2161_, 1);
v_isSharedCheck_2177_ = !lean_is_exclusive(v_head_2161_);
if (v_isSharedCheck_2177_ == 0)
{
lean_object* v_unused_2178_; 
v_unused_2178_ = lean_ctor_get(v_head_2161_, 0);
lean_dec(v_unused_2178_);
v___x_2164_ = v_head_2161_;
v_isShared_2165_ = v_isSharedCheck_2177_;
goto v_resetjp_2163_;
}
else
{
lean_inc(v_after_2162_);
lean_dec(v_head_2161_);
v___x_2164_ = lean_box(0);
v_isShared_2165_ = v_isSharedCheck_2177_;
goto v_resetjp_2163_;
}
v_resetjp_2163_:
{
lean_object* v___x_2166_; lean_object* v___x_2168_; 
v___x_2166_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6___closed__0);
if (v_isShared_2165_ == 0)
{
lean_ctor_set_tag(v___x_2164_, 7);
lean_ctor_set(v___x_2164_, 1, v___x_2166_);
lean_ctor_set(v___x_2164_, 0, v_msgData_2148_);
v___x_2168_ = v___x_2164_;
goto v_reusejp_2167_;
}
else
{
lean_object* v_reuseFailAlloc_2176_; 
v_reuseFailAlloc_2176_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2176_, 0, v_msgData_2148_);
lean_ctor_set(v_reuseFailAlloc_2176_, 1, v___x_2166_);
v___x_2168_ = v_reuseFailAlloc_2176_;
goto v_reusejp_2167_;
}
v_reusejp_2167_:
{
lean_object* v___x_2169_; lean_object* v___x_2170_; lean_object* v___x_2171_; lean_object* v___x_2172_; lean_object* v_msgData_2173_; lean_object* v___x_2174_; lean_object* v___x_2175_; 
v___x_2169_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___closed__2);
v___x_2170_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2170_, 0, v___x_2168_);
lean_ctor_set(v___x_2170_, 1, v___x_2169_);
v___x_2171_ = l_Lean_MessageData_ofSyntax(v_after_2162_);
v___x_2172_ = l_Lean_indentD(v___x_2171_);
v_msgData_2173_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_2173_, 0, v___x_2170_);
lean_ctor_set(v_msgData_2173_, 1, v___x_2172_);
v___x_2174_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3_spec__6(v_msgData_2173_, v_macroStack_2149_);
v___x_2175_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2175_, 0, v___x_2174_);
return v___x_2175_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_msgData_2179_, lean_object* v_macroStack_2180_, lean_object* v___y_2181_, lean_object* v___y_2182_){
_start:
{
lean_object* v_res_2183_; 
v_res_2183_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg(v_msgData_2179_, v_macroStack_2180_, v___y_2181_);
lean_dec(v___y_2181_);
return v_res_2183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1___redArg(lean_object* v_msg_2184_, lean_object* v___y_2185_, lean_object* v___y_2186_){
_start:
{
lean_object* v___x_2188_; 
v___x_2188_ = l_Lean_Elab_Command_getRef___redArg(v___y_2185_);
if (lean_obj_tag(v___x_2188_) == 0)
{
lean_object* v_a_2189_; lean_object* v_macroStack_2190_; lean_object* v___x_2191_; lean_object* v_a_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v_a_2195_; lean_object* v___x_2197_; uint8_t v_isShared_2198_; uint8_t v_isSharedCheck_2203_; 
v_a_2189_ = lean_ctor_get(v___x_2188_, 0);
lean_inc(v_a_2189_);
lean_dec_ref_known(v___x_2188_, 1);
v_macroStack_2190_ = lean_ctor_get(v___y_2185_, 4);
v___x_2191_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2___redArg(v_msg_2184_, v___y_2186_);
v_a_2192_ = lean_ctor_get(v___x_2191_, 0);
lean_inc(v_a_2192_);
lean_dec_ref(v___x_2191_);
v___x_2193_ = l_Lean_Elab_getBetterRef(v_a_2189_, v_macroStack_2190_);
lean_dec(v_a_2189_);
lean_inc(v_macroStack_2190_);
v___x_2194_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg(v_a_2192_, v_macroStack_2190_, v___y_2186_);
v_a_2195_ = lean_ctor_get(v___x_2194_, 0);
v_isSharedCheck_2203_ = !lean_is_exclusive(v___x_2194_);
if (v_isSharedCheck_2203_ == 0)
{
v___x_2197_ = v___x_2194_;
v_isShared_2198_ = v_isSharedCheck_2203_;
goto v_resetjp_2196_;
}
else
{
lean_inc(v_a_2195_);
lean_dec(v___x_2194_);
v___x_2197_ = lean_box(0);
v_isShared_2198_ = v_isSharedCheck_2203_;
goto v_resetjp_2196_;
}
v_resetjp_2196_:
{
lean_object* v___x_2199_; lean_object* v___x_2201_; 
v___x_2199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2199_, 0, v___x_2193_);
lean_ctor_set(v___x_2199_, 1, v_a_2195_);
if (v_isShared_2198_ == 0)
{
lean_ctor_set_tag(v___x_2197_, 1);
lean_ctor_set(v___x_2197_, 0, v___x_2199_);
v___x_2201_ = v___x_2197_;
goto v_reusejp_2200_;
}
else
{
lean_object* v_reuseFailAlloc_2202_; 
v_reuseFailAlloc_2202_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2202_, 0, v___x_2199_);
v___x_2201_ = v_reuseFailAlloc_2202_;
goto v_reusejp_2200_;
}
v_reusejp_2200_:
{
return v___x_2201_;
}
}
}
else
{
lean_object* v_a_2204_; lean_object* v___x_2206_; uint8_t v_isShared_2207_; uint8_t v_isSharedCheck_2211_; 
lean_dec_ref(v_msg_2184_);
v_a_2204_ = lean_ctor_get(v___x_2188_, 0);
v_isSharedCheck_2211_ = !lean_is_exclusive(v___x_2188_);
if (v_isSharedCheck_2211_ == 0)
{
v___x_2206_ = v___x_2188_;
v_isShared_2207_ = v_isSharedCheck_2211_;
goto v_resetjp_2205_;
}
else
{
lean_inc(v_a_2204_);
lean_dec(v___x_2188_);
v___x_2206_ = lean_box(0);
v_isShared_2207_ = v_isSharedCheck_2211_;
goto v_resetjp_2205_;
}
v_resetjp_2205_:
{
lean_object* v___x_2209_; 
if (v_isShared_2207_ == 0)
{
v___x_2209_ = v___x_2206_;
goto v_reusejp_2208_;
}
else
{
lean_object* v_reuseFailAlloc_2210_; 
v_reuseFailAlloc_2210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2210_, 0, v_a_2204_);
v___x_2209_ = v_reuseFailAlloc_2210_;
goto v_reusejp_2208_;
}
v_reusejp_2208_:
{
return v___x_2209_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1___redArg___boxed(lean_object* v_msg_2212_, lean_object* v___y_2213_, lean_object* v___y_2214_, lean_object* v___y_2215_){
_start:
{
lean_object* v_res_2216_; 
v_res_2216_ = lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1___redArg(v_msg_2212_, v___y_2213_, v___y_2214_);
lean_dec(v___y_2214_);
lean_dec_ref(v___y_2213_);
return v_res_2216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2___redArg(lean_object* v_env_2217_, lean_object* v___y_2218_){
_start:
{
lean_object* v___x_2220_; lean_object* v_messages_2221_; lean_object* v_scopes_2222_; lean_object* v_usedQuotCtxts_2223_; lean_object* v_nextMacroScope_2224_; lean_object* v_maxRecDepth_2225_; lean_object* v_ngen_2226_; lean_object* v_auxDeclNGen_2227_; lean_object* v_infoState_2228_; lean_object* v_traceState_2229_; lean_object* v_snapshotTasks_2230_; lean_object* v_prevLinterStates_2231_; lean_object* v___x_2233_; uint8_t v_isShared_2234_; uint8_t v_isSharedCheck_2241_; 
v___x_2220_ = lean_st_ref_take(v___y_2218_);
v_messages_2221_ = lean_ctor_get(v___x_2220_, 1);
v_scopes_2222_ = lean_ctor_get(v___x_2220_, 2);
v_usedQuotCtxts_2223_ = lean_ctor_get(v___x_2220_, 3);
v_nextMacroScope_2224_ = lean_ctor_get(v___x_2220_, 4);
v_maxRecDepth_2225_ = lean_ctor_get(v___x_2220_, 5);
v_ngen_2226_ = lean_ctor_get(v___x_2220_, 6);
v_auxDeclNGen_2227_ = lean_ctor_get(v___x_2220_, 7);
v_infoState_2228_ = lean_ctor_get(v___x_2220_, 8);
v_traceState_2229_ = lean_ctor_get(v___x_2220_, 9);
v_snapshotTasks_2230_ = lean_ctor_get(v___x_2220_, 10);
v_prevLinterStates_2231_ = lean_ctor_get(v___x_2220_, 11);
v_isSharedCheck_2241_ = !lean_is_exclusive(v___x_2220_);
if (v_isSharedCheck_2241_ == 0)
{
lean_object* v_unused_2242_; 
v_unused_2242_ = lean_ctor_get(v___x_2220_, 0);
lean_dec(v_unused_2242_);
v___x_2233_ = v___x_2220_;
v_isShared_2234_ = v_isSharedCheck_2241_;
goto v_resetjp_2232_;
}
else
{
lean_inc(v_prevLinterStates_2231_);
lean_inc(v_snapshotTasks_2230_);
lean_inc(v_traceState_2229_);
lean_inc(v_infoState_2228_);
lean_inc(v_auxDeclNGen_2227_);
lean_inc(v_ngen_2226_);
lean_inc(v_maxRecDepth_2225_);
lean_inc(v_nextMacroScope_2224_);
lean_inc(v_usedQuotCtxts_2223_);
lean_inc(v_scopes_2222_);
lean_inc(v_messages_2221_);
lean_dec(v___x_2220_);
v___x_2233_ = lean_box(0);
v_isShared_2234_ = v_isSharedCheck_2241_;
goto v_resetjp_2232_;
}
v_resetjp_2232_:
{
lean_object* v___x_2236_; 
if (v_isShared_2234_ == 0)
{
lean_ctor_set(v___x_2233_, 0, v_env_2217_);
v___x_2236_ = v___x_2233_;
goto v_reusejp_2235_;
}
else
{
lean_object* v_reuseFailAlloc_2240_; 
v_reuseFailAlloc_2240_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_2240_, 0, v_env_2217_);
lean_ctor_set(v_reuseFailAlloc_2240_, 1, v_messages_2221_);
lean_ctor_set(v_reuseFailAlloc_2240_, 2, v_scopes_2222_);
lean_ctor_set(v_reuseFailAlloc_2240_, 3, v_usedQuotCtxts_2223_);
lean_ctor_set(v_reuseFailAlloc_2240_, 4, v_nextMacroScope_2224_);
lean_ctor_set(v_reuseFailAlloc_2240_, 5, v_maxRecDepth_2225_);
lean_ctor_set(v_reuseFailAlloc_2240_, 6, v_ngen_2226_);
lean_ctor_set(v_reuseFailAlloc_2240_, 7, v_auxDeclNGen_2227_);
lean_ctor_set(v_reuseFailAlloc_2240_, 8, v_infoState_2228_);
lean_ctor_set(v_reuseFailAlloc_2240_, 9, v_traceState_2229_);
lean_ctor_set(v_reuseFailAlloc_2240_, 10, v_snapshotTasks_2230_);
lean_ctor_set(v_reuseFailAlloc_2240_, 11, v_prevLinterStates_2231_);
v___x_2236_ = v_reuseFailAlloc_2240_;
goto v_reusejp_2235_;
}
v_reusejp_2235_:
{
lean_object* v___x_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; 
v___x_2237_ = lean_st_ref_set(v___y_2218_, v___x_2236_);
v___x_2238_ = lean_box(0);
v___x_2239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2239_, 0, v___x_2238_);
return v___x_2239_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2___redArg___boxed(lean_object* v_env_2243_, lean_object* v___y_2244_, lean_object* v___y_2245_){
_start:
{
lean_object* v_res_2246_; 
v_res_2246_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2___redArg(v_env_2243_, v___y_2244_);
lean_dec(v___y_2244_);
return v_res_2246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1___redArg(lean_object* v_ext_2247_, lean_object* v_k_2248_, lean_object* v_v_2249_, lean_object* v___y_2250_, lean_object* v___y_2251_){
_start:
{
lean_object* v___x_2253_; lean_object* v_env_2254_; lean_object* v___x_2255_; 
v___x_2253_ = lean_st_ref_get(v___y_2251_);
v_env_2254_ = lean_ctor_get(v___x_2253_, 0);
lean_inc_ref(v_env_2254_);
lean_dec(v___x_2253_);
v___x_2255_ = lp_batteries_Lean_NameMapExtension_find_x3f___redArg(v_ext_2247_, v_env_2254_, v_k_2248_);
if (lean_obj_tag(v___x_2255_) == 1)
{
lean_object* v_name_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; 
lean_dec_ref_known(v___x_2255_, 1);
lean_dec(v_v_2249_);
v_name_2256_ = lean_ctor_get(v_ext_2247_, 1);
lean_inc(v_name_2256_);
lean_dec_ref(v_ext_2247_);
v___x_2257_ = lean_obj_once(&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__1, &lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__1);
v___x_2258_ = l_Lean_MessageData_ofName(v_name_2256_);
v___x_2259_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2259_, 0, v___x_2257_);
lean_ctor_set(v___x_2259_, 1, v___x_2258_);
v___x_2260_ = lean_obj_once(&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__3, &lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__3_once, _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2__spec__0___redArg___closed__3);
v___x_2261_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2261_, 0, v___x_2259_);
lean_ctor_set(v___x_2261_, 1, v___x_2260_);
v___x_2262_ = l_Lean_MessageData_ofName(v_k_2248_);
v___x_2263_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2263_, 0, v___x_2261_);
lean_ctor_set(v___x_2263_, 1, v___x_2262_);
v___x_2264_ = lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1___redArg(v___x_2263_, v___y_2250_, v___y_2251_);
return v___x_2264_;
}
else
{
lean_object* v___x_2265_; lean_object* v_toEnvExtension_2266_; lean_object* v_env_2267_; lean_object* v_asyncMode_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v___x_2272_; 
lean_dec(v___x_2255_);
v___x_2265_ = lean_st_ref_get(v___y_2251_);
v_toEnvExtension_2266_ = lean_ctor_get(v_ext_2247_, 0);
v_env_2267_ = lean_ctor_get(v___x_2265_, 0);
lean_inc_ref(v_env_2267_);
lean_dec(v___x_2265_);
v_asyncMode_2268_ = lean_ctor_get(v_toEnvExtension_2266_, 2);
lean_inc(v_asyncMode_2268_);
v___x_2269_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2269_, 0, v_k_2248_);
lean_ctor_set(v___x_2269_, 1, v_v_2249_);
v___x_2270_ = lean_box(0);
v___x_2271_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v_ext_2247_, v_env_2267_, v___x_2269_, v_asyncMode_2268_, v___x_2270_);
lean_dec(v_asyncMode_2268_);
v___x_2272_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2___redArg(v___x_2271_, v___y_2251_);
return v___x_2272_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1___redArg___boxed(lean_object* v_ext_2273_, lean_object* v_k_2274_, lean_object* v_v_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_){
_start:
{
lean_object* v_res_2279_; 
v_res_2279_ = lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1___redArg(v_ext_2273_, v_k_2274_, v_v_2275_, v___y_2276_, v___y_2277_);
lean_dec(v___y_2277_);
lean_dec_ref(v___y_2276_);
return v_res_2279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1(lean_object* v_x_2290_, lean_object* v_a_2291_, lean_object* v_a_2292_){
_start:
{
lean_object* v___x_2294_; uint8_t v___x_2295_; 
v___x_2294_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_commandInsert__to__additive__translation_____00__closed__1));
lean_inc(v_x_2290_);
v___x_2295_ = l_Lean_Syntax_isOfKind(v_x_2290_, v___x_2294_);
if (v___x_2295_ == 0)
{
lean_object* v___x_2296_; 
lean_dec(v_x_2290_);
v___x_2296_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0___redArg();
return v___x_2296_;
}
else
{
lean_object* v___x_2297_; lean_object* v_src_2298_; lean_object* v___x_2299_; lean_object* v_tgt_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; lean_object* v___x_2305_; lean_object* v___x_2306_; lean_object* v___x_2307_; 
v___x_2297_ = lean_unsigned_to_nat(1u);
v_src_2298_ = l_Lean_Syntax_getArg(v_x_2290_, v___x_2297_);
v___x_2299_ = lean_unsigned_to_nat(2u);
v_tgt_2300_ = l_Lean_Syntax_getArg(v_x_2290_, v___x_2299_);
lean_dec(v_x_2290_);
v___x_2301_ = lp_mathlib_Mathlib_Tactic_ToAdditive_translations;
v___x_2302_ = l_Lean_TSyntax_getId(v_src_2298_);
lean_dec(v_src_2298_);
v___x_2303_ = l_Lean_TSyntax_getId(v_tgt_2300_);
lean_dec(v_tgt_2300_);
v___x_2304_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__2));
v___x_2305_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___closed__3));
v___x_2306_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2306_, 0, v___x_2303_);
lean_ctor_set(v___x_2306_, 1, v___x_2304_);
lean_ctor_set(v___x_2306_, 2, v___x_2305_);
v___x_2307_ = lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1___redArg(v___x_2301_, v___x_2302_, v___x_2306_, v_a_2291_, v_a_2292_);
return v___x_2307_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1___boxed(lean_object* v_x_2308_, lean_object* v_a_2309_, lean_object* v_a_2310_, lean_object* v_a_2311_){
_start:
{
lean_object* v_res_2312_; 
v_res_2312_ = lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1(v_x_2308_, v_a_2309_, v_a_2310_);
lean_dec(v_a_2310_);
lean_dec_ref(v_a_2309_);
return v_res_2312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2(lean_object* v_env_2313_, lean_object* v___y_2314_, lean_object* v___y_2315_){
_start:
{
lean_object* v___x_2317_; 
v___x_2317_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2___redArg(v_env_2313_, v___y_2315_);
return v___x_2317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2___boxed(lean_object* v_env_2318_, lean_object* v___y_2319_, lean_object* v___y_2320_, lean_object* v___y_2321_){
_start:
{
lean_object* v_res_2322_; 
v_res_2322_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__2(v_env_2318_, v___y_2319_, v___y_2320_);
lean_dec(v___y_2320_);
lean_dec_ref(v___y_2319_);
return v_res_2322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1(lean_object* v_00_u03b1_2323_, lean_object* v_ext_2324_, lean_object* v_k_2325_, lean_object* v_v_2326_, lean_object* v___y_2327_, lean_object* v___y_2328_){
_start:
{
lean_object* v___x_2330_; 
v___x_2330_ = lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1___redArg(v_ext_2324_, v_k_2325_, v_v_2326_, v___y_2327_, v___y_2328_);
return v___x_2330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1___boxed(lean_object* v_00_u03b1_2331_, lean_object* v_ext_2332_, lean_object* v_k_2333_, lean_object* v_v_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_, lean_object* v___y_2337_){
_start:
{
lean_object* v_res_2338_; 
v_res_2338_ = lp_mathlib_Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1(v_00_u03b1_2331_, v_ext_2332_, v_k_2333_, v_v_2334_, v___y_2335_, v___y_2336_);
lean_dec(v___y_2336_);
lean_dec_ref(v___y_2335_);
return v_res_2338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2(lean_object* v_msgData_2339_, lean_object* v___y_2340_, lean_object* v___y_2341_){
_start:
{
lean_object* v___x_2343_; 
v___x_2343_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2___redArg(v_msgData_2339_, v___y_2341_);
return v___x_2343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2___boxed(lean_object* v_msgData_2344_, lean_object* v___y_2345_, lean_object* v___y_2346_, lean_object* v___y_2347_){
_start:
{
lean_object* v_res_2348_; 
v_res_2348_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__2(v_msgData_2344_, v___y_2345_, v___y_2346_);
lean_dec(v___y_2346_);
lean_dec_ref(v___y_2345_);
return v_res_2348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1(lean_object* v_00_u03b1_2349_, lean_object* v_msg_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_){
_start:
{
lean_object* v___x_2354_; 
v___x_2354_ = lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1___redArg(v_msg_2350_, v___y_2351_, v___y_2352_);
return v___x_2354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1___boxed(lean_object* v_00_u03b1_2355_, lean_object* v_msg_2356_, lean_object* v___y_2357_, lean_object* v___y_2358_, lean_object* v___y_2359_){
_start:
{
lean_object* v_res_2360_; 
v_res_2360_ = lp_mathlib_Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1(v_00_u03b1_2355_, v_msg_2356_, v___y_2357_, v___y_2358_);
lean_dec(v___y_2358_);
lean_dec_ref(v___y_2357_);
return v_res_2360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3(lean_object* v_msgData_2361_, lean_object* v_macroStack_2362_, lean_object* v___y_2363_, lean_object* v___y_2364_){
_start:
{
lean_object* v___x_2366_; 
v___x_2366_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___redArg(v_msgData_2361_, v_macroStack_2362_, v___y_2364_);
return v___x_2366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3___boxed(lean_object* v_msgData_2367_, lean_object* v_macroStack_2368_, lean_object* v___y_2369_, lean_object* v___y_2370_, lean_object* v___y_2371_){
_start:
{
lean_object* v_res_2372_; 
v_res_2372_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_NameMapExtension_add___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__1_spec__1_spec__3(v_msgData_2367_, v_macroStack_2368_, v___y_2369_, v___y_2370_);
lean_dec(v___y_2370_);
lean_dec_ref(v___y_2369_);
return v_res_2372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandTo__additive__name__hint______1(lean_object* v_x_2395_, lean_object* v_a_2396_, lean_object* v_a_2397_){
_start:
{
lean_object* v___x_2399_; uint8_t v___x_2400_; 
v___x_2399_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ToAdditive_commandTo__additive__name__hint_____00__closed__1));
lean_inc(v_x_2395_);
v___x_2400_ = l_Lean_Syntax_isOfKind(v_x_2395_, v___x_2399_);
if (v___x_2400_ == 0)
{
lean_object* v___x_2401_; 
lean_dec(v_x_2395_);
v___x_2401_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandInsert__to__additive__translation______1_spec__0___redArg();
return v___x_2401_;
}
else
{
lean_object* v___x_2402_; lean_object* v_src_2403_; lean_object* v___x_2404_; lean_object* v_tgt_2405_; lean_object* v___x_2406_; lean_object* v___x_2407_; 
v___x_2402_ = lean_unsigned_to_nat(1u);
v_src_2403_ = l_Lean_Syntax_getArg(v_x_2395_, v___x_2402_);
v___x_2404_ = lean_unsigned_to_nat(2u);
v_tgt_2405_ = l_Lean_Syntax_getArg(v_x_2395_, v___x_2404_);
lean_dec(v_x_2395_);
v___x_2406_ = lp_mathlib_Mathlib_Tactic_ToAdditive_guessNameExt;
v___x_2407_ = lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation(v___x_2406_, v_src_2403_, v_tgt_2405_, v_a_2396_, v_a_2397_);
lean_dec(v_tgt_2405_);
lean_dec(v_src_2403_);
return v___x_2407_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandTo__additive__name__hint______1___boxed(lean_object* v_x_2408_, lean_object* v_a_2409_, lean_object* v_a_2410_, lean_object* v_a_2411_){
_start:
{
lean_object* v_res_2412_; 
v_res_2412_ = lp_mathlib_Mathlib_Tactic_ToAdditive___aux__Mathlib__Tactic__Translate__ToAdditive______elabRules__Mathlib__Tactic__ToAdditive__commandTo__additive__name__hint______1(v_x_2408_, v_a_2409_, v_a_2410_);
lean_dec(v_a_2410_);
lean_dec_ref(v_a_2409_);
return v_res_2412_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Translate_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Translate_ToAdditive(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Translate_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Translate_ToAdditive(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive = _init_lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToAdditive_to__additive);
lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f__ = _init_lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f__();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToAdditive_attrTo__additive_x3f__);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2527625364____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_ToAdditive_ignoreArgsAttr = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToAdditive_ignoreArgsAttr);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_2699740242____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_ToAdditive_doTranslateAttr = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToAdditive_doTranslateAttr);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3091569921____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1649600147____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_ToAdditive_translations = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToAdditive_translations);
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict = _init_lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToAdditive_nameDict);
lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict = _init_lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToAdditive_abbreviationDict);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_3035231656____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_ToAdditive_guessNameExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToAdditive_guessNameExt);
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ToAdditive_data = _init_lp_mathlib_Mathlib_Tactic_ToAdditive_data();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ToAdditive_data);
res = lp_mathlib___private_Mathlib_Tactic_Translate_ToAdditive_0__Mathlib_Tactic_ToAdditive_initFn_00___x40_Mathlib_Tactic_Translate_ToAdditive_1826194073____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Translate_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Translate_ToAdditive(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Translate_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Translate_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Translate_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Translate_ToAdditive(builtin);
}
#ifdef __cplusplus
}
#endif
