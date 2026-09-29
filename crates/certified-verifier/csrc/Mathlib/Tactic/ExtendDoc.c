// Lean compiler output
// Module: Mathlib.Tactic.ExtendDoc
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.DocString
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
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
extern lean_object* l_Lean_docStringExt;
lean_object* l_String_removeLeadingSpaces(lean_object*);
lean_object* l_Lean_MapDeclarationExtension_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_findDocString_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_TSyntax_getString(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "ExtendDocs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "commandExtend_docs__Before__After_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(238, 229, 215, 249, 190, 24, 240, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(20, 122, 111, 131, 30, 17, 221, 126)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "extend_docs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "before "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__21_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__20_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "after "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__27_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__29_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__26_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__33_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__33_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__2(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "invalid doc string, declaration `"};
static const lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "` is in an imported module"};
static const lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\n\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "expected at least one of 'before' or 'after'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_81_ = lean_box(0);
v___x_82_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_83_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_83_, 0, v___x_82_);
lean_ctor_set(v___x_83_, 1, v___x_81_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg(){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_85_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg___closed__0);
v___x_86_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_86_, 0, v___x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg___boxed(lean_object* v___y_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg();
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0(lean_object* v_00_u03b1_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg();
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___boxed(lean_object* v_00_u03b1_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0(v_00_u03b1_94_, v___y_95_, v___y_96_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__2(lean_object* v_msg_99_){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_100_ = lean_box(0);
v___x_101_ = lean_panic_fn_borrowed(v___x_100_, v_msg_99_);
return v___x_101_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_102_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_103_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__0);
v___x_104_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
return v___x_104_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_105_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__1);
v___x_106_ = lean_unsigned_to_nat(0u);
v___x_107_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v___x_106_);
lean_ctor_set(v___x_107_, 2, v___x_106_);
lean_ctor_set(v___x_107_, 3, v___x_106_);
lean_ctor_set(v___x_107_, 4, v___x_105_);
lean_ctor_set(v___x_107_, 5, v___x_105_);
lean_ctor_set(v___x_107_, 6, v___x_105_);
lean_ctor_set(v___x_107_, 7, v___x_105_);
lean_ctor_set(v___x_107_, 8, v___x_105_);
lean_ctor_set(v___x_107_, 9, v___x_105_);
return v___x_107_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__3(void){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_108_ = lean_unsigned_to_nat(32u);
v___x_109_ = lean_mk_empty_array_with_capacity(v___x_108_);
v___x_110_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__4(void){
_start:
{
size_t v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_111_ = ((size_t)5ULL);
v___x_112_ = lean_unsigned_to_nat(0u);
v___x_113_ = lean_unsigned_to_nat(32u);
v___x_114_ = lean_mk_empty_array_with_capacity(v___x_113_);
v___x_115_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__3);
v___x_116_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v___x_114_);
lean_ctor_set(v___x_116_, 2, v___x_112_);
lean_ctor_set(v___x_116_, 3, v___x_112_);
lean_ctor_set_usize(v___x_116_, 4, v___x_111_);
return v___x_116_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__5(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_117_ = lean_box(1);
v___x_118_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__4);
v___x_119_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__1);
v___x_120_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_120_, 0, v___x_119_);
lean_ctor_set(v___x_120_, 1, v___x_118_);
lean_ctor_set(v___x_120_, 2, v___x_117_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg(lean_object* v_msgData_121_, lean_object* v___y_122_){
_start:
{
lean_object* v___x_124_; lean_object* v_env_125_; lean_object* v___x_126_; lean_object* v_scopes_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v_opts_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_124_ = lean_st_ref_get(v___y_122_);
v_env_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc_ref(v_env_125_);
lean_dec(v___x_124_);
v___x_126_ = lean_st_ref_get(v___y_122_);
v_scopes_127_ = lean_ctor_get(v___x_126_, 2);
lean_inc(v_scopes_127_);
lean_dec(v___x_126_);
v___x_128_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_129_ = l_List_head_x21___redArg(v___x_128_, v_scopes_127_);
lean_dec(v_scopes_127_);
v_opts_130_ = lean_ctor_get(v___x_129_, 1);
lean_inc_ref(v_opts_130_);
lean_dec(v___x_129_);
v___x_131_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__2);
v___x_132_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___closed__5);
v___x_133_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_133_, 0, v_env_125_);
lean_ctor_set(v___x_133_, 1, v___x_131_);
lean_ctor_set(v___x_133_, 2, v___x_132_);
lean_ctor_set(v___x_133_, 3, v_opts_130_);
v___x_134_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_msgData_121_);
v___x_135_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg___boxed(lean_object* v_msgData_136_, lean_object* v___y_137_, lean_object* v___y_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg(v_msgData_136_, v___y_137_);
lean_dec(v___y_137_);
return v_res_139_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__0(void){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_140_ = lean_box(1);
v___x_141_ = l_Lean_MessageData_ofFormat(v___x_140_);
return v___x_141_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__3(void){
_start:
{
lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_145_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__2));
v___x_146_ = l_Lean_MessageData_ofFormat(v___x_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6(lean_object* v_x_147_, lean_object* v_x_148_){
_start:
{
if (lean_obj_tag(v_x_148_) == 0)
{
return v_x_147_;
}
else
{
lean_object* v_head_149_; lean_object* v_tail_150_; lean_object* v___x_152_; uint8_t v_isShared_153_; uint8_t v_isSharedCheck_172_; 
v_head_149_ = lean_ctor_get(v_x_148_, 0);
v_tail_150_ = lean_ctor_get(v_x_148_, 1);
v_isSharedCheck_172_ = !lean_is_exclusive(v_x_148_);
if (v_isSharedCheck_172_ == 0)
{
v___x_152_ = v_x_148_;
v_isShared_153_ = v_isSharedCheck_172_;
goto v_resetjp_151_;
}
else
{
lean_inc(v_tail_150_);
lean_inc(v_head_149_);
lean_dec(v_x_148_);
v___x_152_ = lean_box(0);
v_isShared_153_ = v_isSharedCheck_172_;
goto v_resetjp_151_;
}
v_resetjp_151_:
{
lean_object* v_before_154_; lean_object* v___x_156_; uint8_t v_isShared_157_; uint8_t v_isSharedCheck_170_; 
v_before_154_ = lean_ctor_get(v_head_149_, 0);
v_isSharedCheck_170_ = !lean_is_exclusive(v_head_149_);
if (v_isSharedCheck_170_ == 0)
{
lean_object* v_unused_171_; 
v_unused_171_ = lean_ctor_get(v_head_149_, 1);
lean_dec(v_unused_171_);
v___x_156_ = v_head_149_;
v_isShared_157_ = v_isSharedCheck_170_;
goto v_resetjp_155_;
}
else
{
lean_inc(v_before_154_);
lean_dec(v_head_149_);
v___x_156_ = lean_box(0);
v_isShared_157_ = v_isSharedCheck_170_;
goto v_resetjp_155_;
}
v_resetjp_155_:
{
lean_object* v___x_158_; lean_object* v___x_160_; 
v___x_158_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__0);
if (v_isShared_157_ == 0)
{
lean_ctor_set_tag(v___x_156_, 7);
lean_ctor_set(v___x_156_, 1, v___x_158_);
lean_ctor_set(v___x_156_, 0, v_x_147_);
v___x_160_ = v___x_156_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_169_; 
v_reuseFailAlloc_169_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_169_, 0, v_x_147_);
lean_ctor_set(v_reuseFailAlloc_169_, 1, v___x_158_);
v___x_160_ = v_reuseFailAlloc_169_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
lean_object* v___x_161_; lean_object* v___x_163_; 
v___x_161_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__3);
if (v_isShared_153_ == 0)
{
lean_ctor_set_tag(v___x_152_, 7);
lean_ctor_set(v___x_152_, 1, v___x_161_);
lean_ctor_set(v___x_152_, 0, v___x_160_);
v___x_163_ = v___x_152_;
goto v_reusejp_162_;
}
else
{
lean_object* v_reuseFailAlloc_168_; 
v_reuseFailAlloc_168_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_168_, 0, v___x_160_);
lean_ctor_set(v_reuseFailAlloc_168_, 1, v___x_161_);
v___x_163_ = v_reuseFailAlloc_168_;
goto v_reusejp_162_;
}
v_reusejp_162_:
{
lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_164_ = l_Lean_MessageData_ofSyntax(v_before_154_);
v___x_165_ = l_Lean_indentD(v___x_164_);
v___x_166_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_166_, 0, v___x_163_);
lean_ctor_set(v___x_166_, 1, v___x_165_);
v_x_147_ = v___x_166_;
v_x_148_ = v_tail_150_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__5(lean_object* v_opts_173_, lean_object* v_opt_174_){
_start:
{
lean_object* v_name_175_; lean_object* v_defValue_176_; lean_object* v_map_177_; lean_object* v___x_178_; 
v_name_175_ = lean_ctor_get(v_opt_174_, 0);
v_defValue_176_ = lean_ctor_get(v_opt_174_, 1);
v_map_177_ = lean_ctor_get(v_opts_173_, 0);
v___x_178_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_177_, v_name_175_);
if (lean_obj_tag(v___x_178_) == 0)
{
uint8_t v___x_179_; 
v___x_179_ = lean_unbox(v_defValue_176_);
return v___x_179_;
}
else
{
lean_object* v_val_180_; 
v_val_180_ = lean_ctor_get(v___x_178_, 0);
lean_inc(v_val_180_);
lean_dec_ref_known(v___x_178_, 1);
if (lean_obj_tag(v_val_180_) == 1)
{
uint8_t v_v_181_; 
v_v_181_ = lean_ctor_get_uint8(v_val_180_, 0);
lean_dec_ref_known(v_val_180_, 0);
return v_v_181_;
}
else
{
uint8_t v___x_182_; 
lean_dec(v_val_180_);
v___x_182_ = lean_unbox(v_defValue_176_);
return v___x_182_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__5___boxed(lean_object* v_opts_183_, lean_object* v_opt_184_){
_start:
{
uint8_t v_res_185_; lean_object* v_r_186_; 
v_res_185_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__5(v_opts_183_, v_opt_184_);
lean_dec_ref(v_opt_184_);
lean_dec_ref(v_opts_183_);
v_r_186_ = lean_box(v_res_185_);
return v_r_186_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_190_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__1));
v___x_191_ = l_Lean_MessageData_ofFormat(v___x_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg(lean_object* v_msgData_192_, lean_object* v_macroStack_193_, lean_object* v___y_194_){
_start:
{
lean_object* v___x_196_; lean_object* v_scopes_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v_opts_200_; lean_object* v___x_201_; uint8_t v___x_202_; 
v___x_196_ = lean_st_ref_get(v___y_194_);
v_scopes_197_ = lean_ctor_get(v___x_196_, 2);
lean_inc(v_scopes_197_);
lean_dec(v___x_196_);
v___x_198_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_199_ = l_List_head_x21___redArg(v___x_198_, v_scopes_197_);
lean_dec(v_scopes_197_);
v_opts_200_ = lean_ctor_get(v___x_199_, 1);
lean_inc_ref(v_opts_200_);
lean_dec(v___x_199_);
v___x_201_ = l_Lean_Elab_pp_macroStack;
v___x_202_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__5(v_opts_200_, v___x_201_);
lean_dec_ref(v_opts_200_);
if (v___x_202_ == 0)
{
lean_object* v___x_203_; 
lean_dec(v_macroStack_193_);
v___x_203_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_203_, 0, v_msgData_192_);
return v___x_203_;
}
else
{
if (lean_obj_tag(v_macroStack_193_) == 0)
{
lean_object* v___x_204_; 
v___x_204_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_204_, 0, v_msgData_192_);
return v___x_204_;
}
else
{
lean_object* v_head_205_; lean_object* v_after_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_221_; 
v_head_205_ = lean_ctor_get(v_macroStack_193_, 0);
lean_inc(v_head_205_);
v_after_206_ = lean_ctor_get(v_head_205_, 1);
v_isSharedCheck_221_ = !lean_is_exclusive(v_head_205_);
if (v_isSharedCheck_221_ == 0)
{
lean_object* v_unused_222_; 
v_unused_222_ = lean_ctor_get(v_head_205_, 0);
lean_dec(v_unused_222_);
v___x_208_ = v_head_205_;
v_isShared_209_ = v_isSharedCheck_221_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_after_206_);
lean_dec(v_head_205_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_221_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
lean_object* v___x_210_; lean_object* v___x_212_; 
v___x_210_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6___closed__0);
if (v_isShared_209_ == 0)
{
lean_ctor_set_tag(v___x_208_, 7);
lean_ctor_set(v___x_208_, 1, v___x_210_);
lean_ctor_set(v___x_208_, 0, v_msgData_192_);
v___x_212_ = v___x_208_;
goto v_reusejp_211_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v_msgData_192_);
lean_ctor_set(v_reuseFailAlloc_220_, 1, v___x_210_);
v___x_212_ = v_reuseFailAlloc_220_;
goto v_reusejp_211_;
}
v_reusejp_211_:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v_msgData_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_213_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___closed__2);
v___x_214_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_214_, 0, v___x_212_);
lean_ctor_set(v___x_214_, 1, v___x_213_);
v___x_215_ = l_Lean_MessageData_ofSyntax(v_after_206_);
v___x_216_ = l_Lean_indentD(v___x_215_);
v_msgData_217_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_217_, 0, v___x_214_);
lean_ctor_set(v_msgData_217_, 1, v___x_216_);
v___x_218_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4_spec__6(v_msgData_217_, v_macroStack_193_);
v___x_219_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_219_, 0, v___x_218_);
return v___x_219_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg___boxed(lean_object* v_msgData_223_, lean_object* v_macroStack_224_, lean_object* v___y_225_, lean_object* v___y_226_){
_start:
{
lean_object* v_res_227_; 
v_res_227_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg(v_msgData_223_, v_macroStack_224_, v___y_225_);
lean_dec(v___y_225_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3___redArg(lean_object* v_msg_228_, lean_object* v___y_229_, lean_object* v___y_230_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = l_Lean_Elab_Command_getRef___redArg(v___y_229_);
if (lean_obj_tag(v___x_232_) == 0)
{
lean_object* v_a_233_; lean_object* v_macroStack_234_; lean_object* v___x_235_; lean_object* v_a_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v_a_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_247_; 
v_a_233_ = lean_ctor_get(v___x_232_, 0);
lean_inc(v_a_233_);
lean_dec_ref_known(v___x_232_, 1);
v_macroStack_234_ = lean_ctor_get(v___y_229_, 4);
v___x_235_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg(v_msg_228_, v___y_230_);
v_a_236_ = lean_ctor_get(v___x_235_, 0);
lean_inc(v_a_236_);
lean_dec_ref(v___x_235_);
v___x_237_ = l_Lean_Elab_getBetterRef(v_a_233_, v_macroStack_234_);
lean_dec(v_a_233_);
lean_inc(v_macroStack_234_);
v___x_238_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg(v_a_236_, v_macroStack_234_, v___y_230_);
v_a_239_ = lean_ctor_get(v___x_238_, 0);
v_isSharedCheck_247_ = !lean_is_exclusive(v___x_238_);
if (v_isSharedCheck_247_ == 0)
{
v___x_241_ = v___x_238_;
v_isShared_242_ = v_isSharedCheck_247_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_a_239_);
lean_dec(v___x_238_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_247_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___x_243_; lean_object* v___x_245_; 
v___x_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_243_, 0, v___x_237_);
lean_ctor_set(v___x_243_, 1, v_a_239_);
if (v_isShared_242_ == 0)
{
lean_ctor_set_tag(v___x_241_, 1);
lean_ctor_set(v___x_241_, 0, v___x_243_);
v___x_245_ = v___x_241_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v___x_243_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
else
{
lean_object* v_a_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_255_; 
lean_dec_ref(v_msg_228_);
v_a_248_ = lean_ctor_get(v___x_232_, 0);
v_isSharedCheck_255_ = !lean_is_exclusive(v___x_232_);
if (v_isSharedCheck_255_ == 0)
{
v___x_250_ = v___x_232_;
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_a_248_);
lean_dec(v___x_232_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_253_; 
if (v_isShared_251_ == 0)
{
v___x_253_ = v___x_250_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v_a_248_);
v___x_253_ = v_reuseFailAlloc_254_;
goto v_reusejp_252_;
}
v_reusejp_252_:
{
return v___x_253_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3___redArg___boxed(lean_object* v_msg_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3___redArg(v_msg_256_, v___y_257_, v___y_258_);
lean_dec(v___y_258_);
lean_dec_ref(v___y_257_);
return v_res_260_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__1(void){
_start:
{
lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_262_ = ((lean_object*)(lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__0));
v___x_263_ = l_Lean_stringToMessageData(v___x_262_);
return v___x_263_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__3(void){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_265_ = ((lean_object*)(lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__2));
v___x_266_ = l_Lean_stringToMessageData(v___x_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1(lean_object* v_declName_267_, lean_object* v_docString_268_, lean_object* v___y_269_, lean_object* v___y_270_){
_start:
{
lean_object* v___y_273_; lean_object* v___x_300_; lean_object* v_env_301_; lean_object* v___x_302_; 
v___x_300_ = lean_st_ref_get(v___y_270_);
v_env_301_ = lean_ctor_get(v___x_300_, 0);
lean_inc_ref(v_env_301_);
lean_dec(v___x_300_);
v___x_302_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_301_, v_declName_267_);
lean_dec_ref(v_env_301_);
if (lean_obj_tag(v___x_302_) == 0)
{
v___y_273_ = v___y_270_;
goto v___jp_272_;
}
else
{
uint8_t v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
lean_dec_ref_known(v___x_302_, 1);
lean_dec_ref(v_docString_268_);
v___x_303_ = 0;
v___x_304_ = lean_obj_once(&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__1, &lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__1_once, _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__1);
v___x_305_ = l_Lean_MessageData_ofConstName(v_declName_267_, v___x_303_);
v___x_306_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_306_, 0, v___x_304_);
lean_ctor_set(v___x_306_, 1, v___x_305_);
v___x_307_ = lean_obj_once(&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__3, &lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__3_once, _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___closed__3);
v___x_308_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_308_, 0, v___x_306_);
lean_ctor_set(v___x_308_, 1, v___x_307_);
v___x_309_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3___redArg(v___x_308_, v___y_269_, v___y_270_);
return v___x_309_;
}
v___jp_272_:
{
lean_object* v___x_274_; lean_object* v_env_275_; lean_object* v_messages_276_; lean_object* v_scopes_277_; lean_object* v_usedQuotCtxts_278_; lean_object* v_nextMacroScope_279_; lean_object* v_maxRecDepth_280_; lean_object* v_ngen_281_; lean_object* v_auxDeclNGen_282_; lean_object* v_infoState_283_; lean_object* v_traceState_284_; lean_object* v_snapshotTasks_285_; lean_object* v_prevLinterStates_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_299_; 
v___x_274_ = lean_st_ref_take(v___y_273_);
v_env_275_ = lean_ctor_get(v___x_274_, 0);
v_messages_276_ = lean_ctor_get(v___x_274_, 1);
v_scopes_277_ = lean_ctor_get(v___x_274_, 2);
v_usedQuotCtxts_278_ = lean_ctor_get(v___x_274_, 3);
v_nextMacroScope_279_ = lean_ctor_get(v___x_274_, 4);
v_maxRecDepth_280_ = lean_ctor_get(v___x_274_, 5);
v_ngen_281_ = lean_ctor_get(v___x_274_, 6);
v_auxDeclNGen_282_ = lean_ctor_get(v___x_274_, 7);
v_infoState_283_ = lean_ctor_get(v___x_274_, 8);
v_traceState_284_ = lean_ctor_get(v___x_274_, 9);
v_snapshotTasks_285_ = lean_ctor_get(v___x_274_, 10);
v_prevLinterStates_286_ = lean_ctor_get(v___x_274_, 11);
v_isSharedCheck_299_ = !lean_is_exclusive(v___x_274_);
if (v_isSharedCheck_299_ == 0)
{
v___x_288_ = v___x_274_;
v_isShared_289_ = v_isSharedCheck_299_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_prevLinterStates_286_);
lean_inc(v_snapshotTasks_285_);
lean_inc(v_traceState_284_);
lean_inc(v_infoState_283_);
lean_inc(v_auxDeclNGen_282_);
lean_inc(v_ngen_281_);
lean_inc(v_maxRecDepth_280_);
lean_inc(v_nextMacroScope_279_);
lean_inc(v_usedQuotCtxts_278_);
lean_inc(v_scopes_277_);
lean_inc(v_messages_276_);
lean_inc(v_env_275_);
lean_dec(v___x_274_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_299_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_294_; 
v___x_290_ = l_Lean_docStringExt;
v___x_291_ = l_String_removeLeadingSpaces(v_docString_268_);
v___x_292_ = l_Lean_MapDeclarationExtension_insert___redArg(v___x_290_, v_env_275_, v_declName_267_, v___x_291_);
if (v_isShared_289_ == 0)
{
lean_ctor_set(v___x_288_, 0, v___x_292_);
v___x_294_ = v___x_288_;
goto v_reusejp_293_;
}
else
{
lean_object* v_reuseFailAlloc_298_; 
v_reuseFailAlloc_298_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_298_, 0, v___x_292_);
lean_ctor_set(v_reuseFailAlloc_298_, 1, v_messages_276_);
lean_ctor_set(v_reuseFailAlloc_298_, 2, v_scopes_277_);
lean_ctor_set(v_reuseFailAlloc_298_, 3, v_usedQuotCtxts_278_);
lean_ctor_set(v_reuseFailAlloc_298_, 4, v_nextMacroScope_279_);
lean_ctor_set(v_reuseFailAlloc_298_, 5, v_maxRecDepth_280_);
lean_ctor_set(v_reuseFailAlloc_298_, 6, v_ngen_281_);
lean_ctor_set(v_reuseFailAlloc_298_, 7, v_auxDeclNGen_282_);
lean_ctor_set(v_reuseFailAlloc_298_, 8, v_infoState_283_);
lean_ctor_set(v_reuseFailAlloc_298_, 9, v_traceState_284_);
lean_ctor_set(v_reuseFailAlloc_298_, 10, v_snapshotTasks_285_);
lean_ctor_set(v_reuseFailAlloc_298_, 11, v_prevLinterStates_286_);
v___x_294_ = v_reuseFailAlloc_298_;
goto v_reusejp_293_;
}
v_reusejp_293_:
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_295_ = lean_st_ref_set(v___y_273_, v___x_294_);
v___x_296_ = lean_box(0);
v___x_297_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
return v___x_297_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1___boxed(lean_object* v_declName_310_, lean_object* v_docString_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1(v_declName_310_, v_docString_311_, v___y_312_, v___y_313_);
lean_dec(v___y_313_);
lean_dec_ref(v___y_312_);
return v_res_315_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__5(void){
_start:
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_321_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__4));
v___x_322_ = lean_unsigned_to_nat(14u);
v___x_323_ = lean_unsigned_to_nat(22u);
v___x_324_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__3));
v___x_325_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__2));
v___x_326_ = l_mkPanicMessageWithDecl(v___x_325_, v___x_324_, v___x_323_, v___x_322_, v___x_321_);
return v___x_326_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__7(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_328_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__6));
v___x_329_ = l_Lean_stringToMessageData(v___x_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1(lean_object* v_x_330_, lean_object* v_a_331_, lean_object* v_a_332_){
_start:
{
lean_object* v___y_335_; lean_object* v___y_336_; lean_object* v___y_337_; lean_object* v___y_338_; lean_object* v___y_339_; lean_object* v___y_340_; lean_object* v___x_344_; uint8_t v___x_345_; 
v___x_344_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__4));
lean_inc(v_x_330_);
v___x_345_ = l_Lean_Syntax_isOfKind(v_x_330_, v___x_344_);
if (v___x_345_ == 0)
{
lean_object* v___x_346_; 
lean_dec(v_x_330_);
v___x_346_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg();
return v___x_346_;
}
else
{
lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; uint8_t v___x_350_; lean_object* v___y_352_; lean_object* v___y_353_; lean_object* v___y_354_; lean_object* v___y_355_; lean_object* v___y_356_; lean_object* v___y_380_; lean_object* v___y_381_; lean_object* v___y_382_; lean_object* v___y_383_; lean_object* v___y_384_; lean_object* v___y_385_; lean_object* v___y_389_; lean_object* v___y_390_; lean_object* v___y_391_; lean_object* v___y_392_; lean_object* v___y_393_; lean_object* v___y_401_; lean_object* v___y_402_; lean_object* v___y_403_; lean_object* v___y_404_; lean_object* v___y_405_; lean_object* v___y_410_; lean_object* v___y_411_; lean_object* v___y_412_; lean_object* v___y_413_; lean_object* v___y_433_; lean_object* v_aft_434_; lean_object* v___y_435_; lean_object* v___y_436_; 
v___x_347_ = lean_unsigned_to_nat(1u);
v___x_348_ = l_Lean_Syntax_getArg(v_x_330_, v___x_347_);
v___x_349_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__10));
lean_inc(v___x_348_);
v___x_350_ = l_Lean_Syntax_isOfKind(v___x_348_, v___x_349_);
if (v___x_350_ == 0)
{
lean_object* v___x_439_; 
lean_dec(v___x_348_);
lean_dec(v_x_330_);
v___x_439_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg();
return v___x_439_;
}
else
{
lean_object* v___x_440_; lean_object* v_bef_442_; lean_object* v___y_443_; lean_object* v___y_444_; lean_object* v___x_456_; uint8_t v___x_457_; 
v___x_440_ = lean_unsigned_to_nat(2u);
v___x_456_ = l_Lean_Syntax_getArg(v_x_330_, v___x_440_);
v___x_457_ = l_Lean_Syntax_isNone(v___x_456_);
if (v___x_457_ == 0)
{
uint8_t v___x_458_; 
lean_inc(v___x_456_);
v___x_458_ = l_Lean_Syntax_matchesNull(v___x_456_, v___x_440_);
if (v___x_458_ == 0)
{
lean_object* v___x_459_; 
lean_dec(v___x_456_);
lean_dec(v___x_348_);
lean_dec(v_x_330_);
v___x_459_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg();
return v___x_459_;
}
else
{
lean_object* v_bef_460_; lean_object* v___x_461_; uint8_t v___x_462_; 
v_bef_460_ = l_Lean_Syntax_getArg(v___x_456_, v___x_347_);
lean_dec(v___x_456_);
v___x_461_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__22));
lean_inc(v_bef_460_);
v___x_462_ = l_Lean_Syntax_isOfKind(v_bef_460_, v___x_461_);
if (v___x_462_ == 0)
{
lean_object* v___x_463_; 
lean_dec(v_bef_460_);
lean_dec(v___x_348_);
lean_dec(v_x_330_);
v___x_463_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg();
return v___x_463_;
}
else
{
lean_object* v___x_464_; 
v___x_464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_464_, 0, v_bef_460_);
v_bef_442_ = v___x_464_;
v___y_443_ = v_a_331_;
v___y_444_ = v_a_332_;
goto v___jp_441_;
}
}
}
else
{
lean_object* v___x_465_; 
lean_dec(v___x_456_);
v___x_465_ = lean_box(0);
v_bef_442_ = v___x_465_;
v___y_443_ = v_a_331_;
v___y_444_ = v_a_332_;
goto v___jp_441_;
}
v___jp_441_:
{
lean_object* v___x_445_; lean_object* v___x_446_; uint8_t v___x_447_; 
v___x_445_ = lean_unsigned_to_nat(3u);
v___x_446_ = l_Lean_Syntax_getArg(v_x_330_, v___x_445_);
lean_dec(v_x_330_);
v___x_447_ = l_Lean_Syntax_isNone(v___x_446_);
if (v___x_447_ == 0)
{
uint8_t v___x_448_; 
lean_inc(v___x_446_);
v___x_448_ = l_Lean_Syntax_matchesNull(v___x_446_, v___x_440_);
if (v___x_448_ == 0)
{
lean_object* v___x_449_; 
lean_dec(v___x_446_);
lean_dec(v_bef_442_);
lean_dec(v___x_348_);
v___x_449_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg();
return v___x_449_;
}
else
{
lean_object* v_aft_450_; lean_object* v___x_451_; uint8_t v___x_452_; 
v_aft_450_ = l_Lean_Syntax_getArg(v___x_446_, v___x_347_);
lean_dec(v___x_446_);
v___x_451_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs_commandExtend__docs____Before____After___00__closed__22));
lean_inc(v_aft_450_);
v___x_452_ = l_Lean_Syntax_isOfKind(v_aft_450_, v___x_451_);
if (v___x_452_ == 0)
{
lean_object* v___x_453_; 
lean_dec(v_aft_450_);
lean_dec(v_bef_442_);
lean_dec(v___x_348_);
v___x_453_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__0___redArg();
return v___x_453_;
}
else
{
lean_object* v___x_454_; 
v___x_454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_454_, 0, v_aft_450_);
v___y_433_ = v_bef_442_;
v_aft_434_ = v___x_454_;
v___y_435_ = v___y_443_;
v___y_436_ = v___y_444_;
goto v___jp_432_;
}
}
}
else
{
lean_object* v___x_455_; 
lean_dec(v___x_446_);
v___x_455_ = lean_box(0);
v___y_433_ = v_bef_442_;
v_aft_434_ = v___x_455_;
v___y_435_ = v___y_443_;
v___y_436_ = v___y_444_;
goto v___jp_432_;
}
}
}
v___jp_351_:
{
lean_object* v___x_357_; lean_object* v_env_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; 
v___x_357_ = lean_st_ref_get(v___y_355_);
v_env_358_ = lean_ctor_get(v___x_357_, 0);
lean_inc_ref(v_env_358_);
lean_dec(v___x_357_);
v___x_359_ = l_Lean_Options_empty;
v___x_360_ = lean_box(0);
v___x_361_ = lean_box(0);
lean_inc(v___y_354_);
v___x_362_ = l_Lean_findDocString_x3f(v_env_358_, v___y_354_, v___x_350_, v___x_359_, v___x_360_, v___x_361_);
if (lean_obj_tag(v___x_362_) == 0)
{
lean_object* v_a_363_; 
v_a_363_ = lean_ctor_get(v___x_362_, 0);
lean_inc(v_a_363_);
lean_dec_ref_known(v___x_362_, 1);
if (lean_obj_tag(v_a_363_) == 0)
{
lean_object* v___x_364_; 
v___x_364_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__0));
v___y_335_ = v___y_352_;
v___y_336_ = v___y_356_;
v___y_337_ = v___y_353_;
v___y_338_ = v___y_355_;
v___y_339_ = v___y_354_;
v___y_340_ = v___x_364_;
goto v___jp_334_;
}
else
{
lean_object* v_val_365_; 
v_val_365_ = lean_ctor_get(v_a_363_, 0);
lean_inc(v_val_365_);
lean_dec_ref_known(v_a_363_, 1);
v___y_335_ = v___y_352_;
v___y_336_ = v___y_356_;
v___y_337_ = v___y_353_;
v___y_338_ = v___y_355_;
v___y_339_ = v___y_354_;
v___y_340_ = v_val_365_;
goto v___jp_334_;
}
}
else
{
lean_object* v_a_366_; lean_object* v___x_368_; uint8_t v_isShared_369_; uint8_t v_isSharedCheck_378_; 
lean_dec_ref(v___y_356_);
lean_dec(v___y_354_);
lean_dec_ref(v___y_353_);
v_a_366_ = lean_ctor_get(v___x_362_, 0);
v_isSharedCheck_378_ = !lean_is_exclusive(v___x_362_);
if (v_isSharedCheck_378_ == 0)
{
v___x_368_ = v___x_362_;
v_isShared_369_ = v_isSharedCheck_378_;
goto v_resetjp_367_;
}
else
{
lean_inc(v_a_366_);
lean_dec(v___x_362_);
v___x_368_ = lean_box(0);
v_isShared_369_ = v_isSharedCheck_378_;
goto v_resetjp_367_;
}
v_resetjp_367_:
{
lean_object* v_ref_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_376_; 
v_ref_370_ = lean_ctor_get(v___y_352_, 7);
v___x_371_ = lean_io_error_to_string(v_a_366_);
v___x_372_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
v___x_373_ = l_Lean_MessageData_ofFormat(v___x_372_);
lean_inc(v_ref_370_);
v___x_374_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_374_, 0, v_ref_370_);
lean_ctor_set(v___x_374_, 1, v___x_373_);
if (v_isShared_369_ == 0)
{
lean_ctor_set(v___x_368_, 0, v___x_374_);
v___x_376_ = v___x_368_;
goto v_reusejp_375_;
}
else
{
lean_object* v_reuseFailAlloc_377_; 
v_reuseFailAlloc_377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_377_, 0, v___x_374_);
v___x_376_ = v_reuseFailAlloc_377_;
goto v_reusejp_375_;
}
v_reusejp_375_:
{
return v___x_376_;
}
}
}
}
v___jp_379_:
{
lean_object* v___x_386_; lean_object* v___x_387_; 
v___x_386_ = l_Lean_TSyntax_getString(v___y_385_);
lean_dec(v___y_385_);
lean_inc_ref(v___y_381_);
v___x_387_ = lean_string_append(v___y_381_, v___x_386_);
lean_dec_ref(v___x_386_);
v___y_352_ = v___y_380_;
v___y_353_ = v___y_382_;
v___y_354_ = v___y_384_;
v___y_355_ = v___y_383_;
v___y_356_ = v___x_387_;
goto v___jp_351_;
}
v___jp_388_:
{
if (lean_obj_tag(v___y_390_) == 0)
{
if (v___x_350_ == 0)
{
lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_394_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__1));
v___x_395_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__5, &lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__5);
v___x_396_ = lp_mathlib_panic___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__2(v___x_395_);
v___y_380_ = v___y_389_;
v___y_381_ = v___x_394_;
v___y_382_ = v___y_393_;
v___y_383_ = v___y_391_;
v___y_384_ = v___y_392_;
v___y_385_ = v___x_396_;
goto v___jp_379_;
}
else
{
lean_object* v___x_397_; 
v___x_397_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__0));
v___y_352_ = v___y_389_;
v___y_353_ = v___y_393_;
v___y_354_ = v___y_392_;
v___y_355_ = v___y_391_;
v___y_356_ = v___x_397_;
goto v___jp_351_;
}
}
else
{
lean_object* v_val_398_; lean_object* v___x_399_; 
v_val_398_ = lean_ctor_get(v___y_390_, 0);
lean_inc(v_val_398_);
lean_dec_ref_known(v___y_390_, 1);
v___x_399_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__1));
v___y_380_ = v___y_389_;
v___y_381_ = v___x_399_;
v___y_382_ = v___y_393_;
v___y_383_ = v___y_391_;
v___y_384_ = v___y_392_;
v___y_385_ = v_val_398_;
goto v___jp_379_;
}
}
v___jp_400_:
{
lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; 
v___x_406_ = l_Lean_TSyntax_getString(v___y_405_);
lean_dec(v___y_405_);
v___x_407_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__1));
v___x_408_ = lean_string_append(v___x_406_, v___x_407_);
v___y_389_ = v___y_401_;
v___y_390_ = v___y_402_;
v___y_391_ = v___y_404_;
v___y_392_ = v___y_403_;
v___y_393_ = v___x_408_;
goto v___jp_388_;
}
v___jp_409_:
{
lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_414_ = lean_box(0);
v___x_415_ = lean_alloc_closure((void*)(l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo___boxed), 5, 2);
lean_closure_set(v___x_415_, 0, v___x_348_);
lean_closure_set(v___x_415_, 1, v___x_414_);
v___x_416_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_415_, v___y_412_, v___y_413_);
if (lean_obj_tag(v___x_416_) == 0)
{
if (lean_obj_tag(v___y_410_) == 0)
{
if (v___x_350_ == 0)
{
lean_object* v_a_417_; lean_object* v___x_418_; lean_object* v___x_419_; 
v_a_417_ = lean_ctor_get(v___x_416_, 0);
lean_inc(v_a_417_);
lean_dec_ref_known(v___x_416_, 1);
v___x_418_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__5, &lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__5);
v___x_419_ = lp_mathlib_panic___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__2(v___x_418_);
v___y_401_ = v___y_412_;
v___y_402_ = v___y_411_;
v___y_403_ = v_a_417_;
v___y_404_ = v___y_413_;
v___y_405_ = v___x_419_;
goto v___jp_400_;
}
else
{
lean_object* v_a_420_; lean_object* v___x_421_; 
v_a_420_ = lean_ctor_get(v___x_416_, 0);
lean_inc(v_a_420_);
lean_dec_ref_known(v___x_416_, 1);
v___x_421_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__0));
v___y_389_ = v___y_412_;
v___y_390_ = v___y_411_;
v___y_391_ = v___y_413_;
v___y_392_ = v_a_420_;
v___y_393_ = v___x_421_;
goto v___jp_388_;
}
}
else
{
lean_object* v_a_422_; lean_object* v_val_423_; 
v_a_422_ = lean_ctor_get(v___x_416_, 0);
lean_inc(v_a_422_);
lean_dec_ref_known(v___x_416_, 1);
v_val_423_ = lean_ctor_get(v___y_410_, 0);
lean_inc(v_val_423_);
lean_dec_ref_known(v___y_410_, 1);
v___y_401_ = v___y_412_;
v___y_402_ = v___y_411_;
v___y_403_ = v_a_422_;
v___y_404_ = v___y_413_;
v___y_405_ = v_val_423_;
goto v___jp_400_;
}
}
else
{
lean_object* v_a_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_431_; 
lean_dec(v___y_411_);
lean_dec(v___y_410_);
v_a_424_ = lean_ctor_get(v___x_416_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_416_);
if (v_isSharedCheck_431_ == 0)
{
v___x_426_ = v___x_416_;
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_a_424_);
lean_dec(v___x_416_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_429_; 
if (v_isShared_427_ == 0)
{
v___x_429_ = v___x_426_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v_a_424_);
v___x_429_ = v_reuseFailAlloc_430_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
return v___x_429_;
}
}
}
}
v___jp_432_:
{
if (lean_obj_tag(v___y_433_) == 0)
{
if (v___x_350_ == 0)
{
v___y_410_ = v___y_433_;
v___y_411_ = v_aft_434_;
v___y_412_ = v___y_435_;
v___y_413_ = v___y_436_;
goto v___jp_409_;
}
else
{
if (lean_obj_tag(v_aft_434_) == 0)
{
lean_object* v___x_437_; lean_object* v___x_438_; 
lean_dec(v___x_348_);
v___x_437_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__7, &lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___closed__7);
v___x_438_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3___redArg(v___x_437_, v___y_435_, v___y_436_);
return v___x_438_;
}
else
{
v___y_410_ = v___y_433_;
v___y_411_ = v_aft_434_;
v___y_412_ = v___y_435_;
v___y_413_ = v___y_436_;
goto v___jp_409_;
}
}
}
else
{
v___y_410_ = v___y_433_;
v___y_411_ = v_aft_434_;
v___y_412_ = v___y_435_;
v___y_413_ = v___y_436_;
goto v___jp_409_;
}
}
}
v___jp_334_:
{
lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_341_ = lean_string_append(v___y_337_, v___y_340_);
lean_dec_ref(v___y_340_);
v___x_342_ = lean_string_append(v___x_341_, v___y_336_);
lean_dec_ref(v___y_336_);
v___x_343_ = lp_mathlib_Lean_addDocStringCore___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__1(v___y_339_, v___x_342_, v___y_335_, v___y_338_);
return v___x_343_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1___boxed(lean_object* v_x_466_, lean_object* v_a_467_, lean_object* v_a_468_, lean_object* v_a_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_mathlib_Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1(v_x_466_, v_a_467_, v_a_468_);
lean_dec(v_a_468_);
lean_dec_ref(v_a_467_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3(lean_object* v_msgData_471_, lean_object* v___y_472_, lean_object* v___y_473_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___redArg(v_msgData_471_, v___y_473_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3___boxed(lean_object* v_msgData_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__3(v_msgData_476_, v___y_477_, v___y_478_);
lean_dec(v___y_478_);
lean_dec_ref(v___y_477_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3(lean_object* v_00_u03b1_481_, lean_object* v_msg_482_, lean_object* v___y_483_, lean_object* v___y_484_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3___redArg(v_msg_482_, v___y_483_, v___y_484_);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3___boxed(lean_object* v_00_u03b1_487_, lean_object* v_msg_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_){
_start:
{
lean_object* v_res_492_; 
v_res_492_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3(v_00_u03b1_487_, v_msg_488_, v___y_489_, v___y_490_);
lean_dec(v___y_490_);
lean_dec_ref(v___y_489_);
return v_res_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4(lean_object* v_msgData_493_, lean_object* v_macroStack_494_, lean_object* v___y_495_, lean_object* v___y_496_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___redArg(v_msgData_493_, v_macroStack_494_, v___y_496_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4___boxed(lean_object* v_msgData_499_, lean_object* v_macroStack_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_){
_start:
{
lean_object* v_res_504_; 
v_res_504_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_ExtendDocs___aux__Mathlib__Tactic__ExtendDoc______elabRules__Mathlib__Tactic__ExtendDocs__commandExtend__docs____Before____After____1_spec__3_spec__4(v_msgData_499_, v_macroStack_500_, v___y_501_, v___y_502_);
lean_dec(v___y_502_);
lean_dec_ref(v___y_501_);
return v_res_504_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ExtendDoc(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_DocString(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ExtendDoc(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_DocString(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_DocString(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ExtendDoc(uint8_t builtin) {
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
res = initialize_Lean_DocString(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ExtendDoc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ExtendDoc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ExtendDoc(builtin);
}
#ifdef __cplusplus
}
#endif
