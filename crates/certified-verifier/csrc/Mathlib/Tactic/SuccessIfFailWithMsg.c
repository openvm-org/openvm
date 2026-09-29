// Lean compiler output
// Module: Mathlib.Tactic.SuccessIfFailWithMsg
// Imports: public import Init public meta import Init public meta import Lean.Elab.Eval public meta import Lean.Elab.Tactic.BuiltinTactic public import Mathlib.Init public meta import Lean.Meta.Tactic.TryThis
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
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* l_Lean_MessageData_toString___boxed(lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_String_Slice_trimAscii(lean_object*);
uint8_t l_String_Slice_beq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_throwErrorAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_toString(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_evalTerm___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTacticSeq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_toString(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withoutRecover___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "successIfFailWithMsg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(244, 58, 29, 249, 84, 193, 89, 5)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "success_if_fail_with_msg "};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__9_value),((lean_object*)(((size_t)(1024) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(13, 106, 54, 236, 164, 218, 24, 154)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMsg = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "Update with tactic error message"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__3___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "tactic '"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "' failed, but got different error message:\n\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "tactic failed, but got different error message:\n\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__5(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\""};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Update with tactic error message: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "' succeeded, but was expected to fail"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "tactic succeeded, but was expected to fail"};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "String"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(6, 130, 56, 8, 41, 104, 134, 43)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__0(lean_object* v_x_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__0___closed__0));
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__0___boxed(lean_object* v_x_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__0(v_x_42_);
lean_dec_ref(v_x_42_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__1(lean_object* v_toPure_44_, lean_object* v_____do__lift_45_){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_46_, 0, v_____do__lift_45_);
v___x_47_ = lean_apply_2(v_toPure_44_, lean_box(0), v___x_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__2(lean_object* v_inst_48_, lean_object* v_toBind_49_, lean_object* v___f_50_, lean_object* v_err_51_){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_52_ = l_Lean_Exception_toMessageData(v_err_51_);
v___x_53_ = lean_alloc_closure((void*)(l_Lean_MessageData_toString___boxed), 2, 1);
lean_closure_set(v___x_53_, 0, v___x_52_);
v___x_54_ = lean_apply_2(v_inst_48_, lean_box(0), v___x_53_);
v___x_55_ = lean_apply_4(v_toBind_49_, lean_box(0), lean_box(0), v___x_54_, v___f_50_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__3(lean_object* v_toPure_56_, lean_object* v_____x_57_){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_58_ = lean_box(0);
v___x_59_ = lean_apply_2(v_toPure_56_, lean_box(0), v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__3___boxed(lean_object* v_toPure_60_, lean_object* v_____x_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__3(v_toPure_60_, v_____x_61_);
lean_dec(v_____x_61_);
return v_res_62_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_64_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__0));
v___x_65_ = l_Lean_stringToMessageData(v___x_64_);
return v___x_65_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__3(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__2));
v___x_68_ = l_Lean_stringToMessageData(v___x_67_);
return v___x_68_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__5(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__4));
v___x_71_ = l_Lean_stringToMessageData(v___x_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4(lean_object* v_ref_72_, lean_object* v_val_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_____r_76_){
_start:
{
if (lean_obj_tag(v_ref_72_) == 1)
{
lean_object* v_val_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v_val_77_ = lean_ctor_get(v_ref_72_, 0);
lean_inc_n(v_val_77_, 2);
lean_dec_ref_known(v_ref_72_, 1);
v___x_78_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1);
v___x_79_ = l_Lean_MessageData_ofSyntax(v_val_77_);
v___x_80_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_80_, 0, v___x_78_);
lean_ctor_set(v___x_80_, 1, v___x_79_);
v___x_81_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__3, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__3);
v___x_82_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_82_, 0, v___x_80_);
lean_ctor_set(v___x_82_, 1, v___x_81_);
v___x_83_ = l_Lean_stringToMessageData(v_val_73_);
v___x_84_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_82_);
lean_ctor_set(v___x_84_, 1, v___x_83_);
v___x_85_ = l_Lean_throwErrorAt___redArg(v_inst_74_, v_inst_75_, v_val_77_, v___x_84_);
return v___x_85_;
}
else
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
lean_dec(v_ref_72_);
v___x_86_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__5, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__5);
v___x_87_ = l_Lean_stringToMessageData(v_val_73_);
v___x_88_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_86_);
lean_ctor_set(v___x_88_, 1, v___x_87_);
v___x_89_ = l_Lean_throwError___redArg(v_inst_74_, v_inst_75_, v___x_88_);
return v___x_89_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__5(lean_object* v___f_90_, lean_object* v_____r_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lean_apply_1(v___f_90_, v_____r_91_);
return v___x_92_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__3(void){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_96_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__2));
v___x_97_ = l_Lean_stringToMessageData(v___x_96_);
return v___x_97_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__5(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_99_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__4));
v___x_100_ = l_Lean_stringToMessageData(v___x_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6(lean_object* v_err_101_, lean_object* v_msg_102_, lean_object* v_ref_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_msgref_106_, lean_object* v___f_107_, lean_object* v_inst_108_, lean_object* v_toBind_109_, lean_object* v_toPure_110_, lean_object* v_____r_111_){
_start:
{
if (lean_obj_tag(v_err_101_) == 1)
{
lean_object* v_val_112_; lean_object* v___x_114_; uint8_t v_isShared_115_; uint8_t v_isSharedCheck_154_; 
v_val_112_ = lean_ctor_get(v_err_101_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v_err_101_);
if (v_isSharedCheck_154_ == 0)
{
v___x_114_ = v_err_101_;
v_isShared_115_ = v_isSharedCheck_154_;
goto v_resetjp_113_;
}
else
{
lean_inc(v_val_112_);
lean_dec(v_err_101_);
v___x_114_ = lean_box(0);
v_isShared_115_ = v_isSharedCheck_154_;
goto v_resetjp_113_;
}
v_resetjp_113_:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_116_ = lean_unsigned_to_nat(0u);
v___x_117_ = lean_string_utf8_byte_size(v_msg_102_);
v___x_118_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_118_, 0, v_msg_102_);
lean_ctor_set(v___x_118_, 1, v___x_116_);
lean_ctor_set(v___x_118_, 2, v___x_117_);
v___x_119_ = l_String_Slice_trimAscii(v___x_118_);
v___x_120_ = lean_string_utf8_byte_size(v_val_112_);
lean_inc(v_val_112_);
v___x_121_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_121_, 0, v_val_112_);
lean_ctor_set(v___x_121_, 1, v___x_116_);
lean_ctor_set(v___x_121_, 2, v___x_120_);
v___x_122_ = l_String_Slice_trimAscii(v___x_121_);
v___x_123_ = l_String_Slice_beq(v___x_119_, v___x_122_);
lean_dec_ref(v___x_119_);
if (v___x_123_ == 0)
{
lean_object* v___f_124_; 
lean_dec(v_toPure_110_);
lean_inc_ref(v_inst_105_);
lean_inc_ref(v_inst_104_);
lean_inc(v_val_112_);
lean_inc(v_ref_103_);
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4), 5, 4);
lean_closure_set(v___f_124_, 0, v_ref_103_);
lean_closure_set(v___f_124_, 1, v_val_112_);
lean_closure_set(v___f_124_, 2, v_inst_104_);
lean_closure_set(v___f_124_, 3, v_inst_105_);
if (lean_obj_tag(v_msgref_106_) == 1)
{
lean_object* v_val_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_149_; 
lean_dec(v_val_112_);
lean_dec_ref(v_inst_105_);
lean_dec_ref(v_inst_104_);
lean_dec(v_ref_103_);
v_val_125_ = lean_ctor_get(v_msgref_106_, 0);
v_isSharedCheck_149_ = !lean_is_exclusive(v_msgref_106_);
if (v_isSharedCheck_149_ == 0)
{
v___x_127_ = v_msgref_106_;
v_isShared_128_ = v_isSharedCheck_149_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_val_125_);
lean_dec(v_msgref_106_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_149_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___f_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_135_; 
v___f_129_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__5), 2, 1);
lean_closure_set(v___f_129_, 0, v___f_124_);
v___x_130_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__0));
v___x_131_ = l_String_Slice_toString(v___x_122_);
lean_dec_ref(v___x_122_);
v___x_132_ = lean_string_append(v___x_130_, v___x_131_);
lean_dec_ref(v___x_131_);
v___x_133_ = lean_string_append(v___x_132_, v___x_130_);
if (v_isShared_115_ == 0)
{
lean_ctor_set(v___x_114_, 0, v___x_133_);
v___x_135_ = v___x_114_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v___x_133_);
v___x_135_ = v_reuseFailAlloc_148_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
lean_object* v___x_136_; lean_object* v___x_138_; 
v___x_136_ = lean_box(0);
if (v_isShared_128_ == 0)
{
lean_ctor_set(v___x_127_, 0, v___f_107_);
v___x_138_ = v___x_127_;
goto v_reusejp_137_;
}
else
{
lean_object* v_reuseFailAlloc_147_; 
v_reuseFailAlloc_147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_147_, 0, v___f_107_);
v___x_138_ = v_reuseFailAlloc_147_;
goto v_reusejp_137_;
}
v_reusejp_137_:
{
lean_object* v_suggestion_139_; lean_object* v___x_140_; uint8_t v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v_suggestion_139_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_suggestion_139_, 0, v___x_135_);
lean_ctor_set(v_suggestion_139_, 1, v___x_136_);
lean_ctor_set(v_suggestion_139_, 2, v___x_136_);
lean_ctor_set(v_suggestion_139_, 3, v___x_136_);
lean_ctor_set(v_suggestion_139_, 4, v___x_136_);
lean_ctor_set(v_suggestion_139_, 5, v___x_138_);
v___x_140_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__1));
v___x_141_ = 4;
v___x_142_ = l_Lean_MessageData_nil;
v___x_143_ = lean_box(v___x_141_);
v___x_144_ = lean_alloc_closure((void*)(l_Lean_Meta_Tactic_TryThis_addSuggestion___boxed), 10, 7);
lean_closure_set(v___x_144_, 0, v_val_125_);
lean_closure_set(v___x_144_, 1, v_suggestion_139_);
lean_closure_set(v___x_144_, 2, v___x_136_);
lean_closure_set(v___x_144_, 3, v___x_140_);
lean_closure_set(v___x_144_, 4, v___x_136_);
lean_closure_set(v___x_144_, 5, v___x_143_);
lean_closure_set(v___x_144_, 6, v___x_142_);
v___x_145_ = lean_apply_2(v_inst_108_, lean_box(0), v___x_144_);
v___x_146_ = lean_apply_4(v_toBind_109_, lean_box(0), lean_box(0), v___x_145_, v___f_129_);
return v___x_146_;
}
}
}
}
else
{
lean_object* v___x_150_; lean_object* v___x_151_; 
lean_dec_ref(v___f_124_);
lean_dec_ref(v___x_122_);
lean_del_object(v___x_114_);
lean_dec(v_toBind_109_);
lean_dec(v_inst_108_);
lean_dec_ref(v___f_107_);
lean_dec(v_msgref_106_);
v___x_150_ = lean_box(0);
v___x_151_ = lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4(v_ref_103_, v_val_112_, v_inst_104_, v_inst_105_, v___x_150_);
return v___x_151_;
}
}
else
{
lean_object* v___x_152_; lean_object* v___x_153_; 
lean_dec_ref(v___x_122_);
lean_del_object(v___x_114_);
lean_dec(v_val_112_);
lean_dec(v_toBind_109_);
lean_dec(v_inst_108_);
lean_dec_ref(v___f_107_);
lean_dec(v_msgref_106_);
lean_dec_ref(v_inst_105_);
lean_dec_ref(v_inst_104_);
lean_dec(v_ref_103_);
v___x_152_ = lean_box(0);
v___x_153_ = lean_apply_2(v_toPure_110_, lean_box(0), v___x_152_);
return v___x_153_;
}
}
}
else
{
lean_dec(v_toPure_110_);
lean_dec(v_toBind_109_);
lean_dec(v_inst_108_);
lean_dec_ref(v___f_107_);
lean_dec(v_msgref_106_);
lean_dec_ref(v_msg_102_);
lean_dec(v_err_101_);
if (lean_obj_tag(v_ref_103_) == 1)
{
lean_object* v_val_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v_val_155_ = lean_ctor_get(v_ref_103_, 0);
lean_inc_n(v_val_155_, 2);
lean_dec_ref_known(v_ref_103_, 1);
v___x_156_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1);
v___x_157_ = l_Lean_MessageData_ofSyntax(v_val_155_);
v___x_158_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_158_, 0, v___x_156_);
lean_ctor_set(v___x_158_, 1, v___x_157_);
v___x_159_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__3, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__3);
v___x_160_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_158_);
lean_ctor_set(v___x_160_, 1, v___x_159_);
v___x_161_ = l_Lean_throwErrorAt___redArg(v_inst_104_, v_inst_105_, v_val_155_, v___x_160_);
return v___x_161_;
}
else
{
lean_object* v___x_162_; lean_object* v___x_163_; 
lean_dec(v_ref_103_);
v___x_162_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__5, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__5);
v___x_163_ = l_Lean_throwError___redArg(v_inst_104_, v_inst_105_, v___x_162_);
return v___x_163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__7(lean_object* v_msg_164_, lean_object* v_ref_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_msgref_168_, lean_object* v___f_169_, lean_object* v_inst_170_, lean_object* v_toBind_171_, lean_object* v_toPure_172_, lean_object* v_restoreState_173_, lean_object* v_s_174_, lean_object* v_err_175_){
_start:
{
lean_object* v___f_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
lean_inc(v_toBind_171_);
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6), 11, 10);
lean_closure_set(v___f_176_, 0, v_err_175_);
lean_closure_set(v___f_176_, 1, v_msg_164_);
lean_closure_set(v___f_176_, 2, v_ref_165_);
lean_closure_set(v___f_176_, 3, v_inst_166_);
lean_closure_set(v___f_176_, 4, v_inst_167_);
lean_closure_set(v___f_176_, 5, v_msgref_168_);
lean_closure_set(v___f_176_, 6, v___f_169_);
lean_closure_set(v___f_176_, 7, v_inst_170_);
lean_closure_set(v___f_176_, 8, v_toBind_171_);
lean_closure_set(v___f_176_, 9, v_toPure_172_);
v___x_177_ = lean_apply_1(v_restoreState_173_, v_s_174_);
v___x_178_ = lean_apply_4(v_toBind_171_, lean_box(0), lean_box(0), v___x_177_, v___f_176_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__8(lean_object* v_toMonadExceptOf_179_, lean_object* v_msg_180_, lean_object* v_ref_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_msgref_184_, lean_object* v___f_185_, lean_object* v_inst_186_, lean_object* v_toBind_187_, lean_object* v_toPure_188_, lean_object* v_restoreState_189_, lean_object* v_tacs_190_, lean_object* v___f_191_, lean_object* v___f_192_, lean_object* v_s_193_){
_start:
{
lean_object* v_tryCatch_194_; lean_object* v___f_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v_tryCatch_194_ = lean_ctor_get(v_toMonadExceptOf_179_, 1);
lean_inc(v_tryCatch_194_);
lean_dec_ref(v_toMonadExceptOf_179_);
lean_inc_n(v_toBind_187_, 2);
v___f_195_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__7), 12, 11);
lean_closure_set(v___f_195_, 0, v_msg_180_);
lean_closure_set(v___f_195_, 1, v_ref_181_);
lean_closure_set(v___f_195_, 2, v_inst_182_);
lean_closure_set(v___f_195_, 3, v_inst_183_);
lean_closure_set(v___f_195_, 4, v_msgref_184_);
lean_closure_set(v___f_195_, 5, v___f_185_);
lean_closure_set(v___f_195_, 6, v_inst_186_);
lean_closure_set(v___f_195_, 7, v_toBind_187_);
lean_closure_set(v___f_195_, 8, v_toPure_188_);
lean_closure_set(v___f_195_, 9, v_restoreState_189_);
lean_closure_set(v___f_195_, 10, v_s_193_);
v___x_196_ = lean_apply_4(v_toBind_187_, lean_box(0), lean_box(0), v_tacs_190_, v___f_191_);
v___x_197_ = lean_apply_3(v_tryCatch_194_, lean_box(0), v___x_196_, v___f_192_);
v___x_198_ = lean_apply_4(v_toBind_187_, lean_box(0), lean_box(0), v___x_197_, v___f_195_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg(lean_object* v_inst_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_msg_205_, lean_object* v_tacs_206_, lean_object* v_msgref_207_, lean_object* v_ref_208_){
_start:
{
lean_object* v_toApplicative_209_; lean_object* v_toBind_210_; lean_object* v_saveState_211_; lean_object* v_restoreState_212_; lean_object* v_toMonadExceptOf_213_; lean_object* v_toPure_214_; lean_object* v___f_215_; lean_object* v___f_216_; lean_object* v___f_217_; lean_object* v___f_218_; lean_object* v___f_219_; lean_object* v___x_220_; 
v_toApplicative_209_ = lean_ctor_get(v_inst_200_, 0);
v_toBind_210_ = lean_ctor_get(v_inst_200_, 1);
lean_inc_n(v_toBind_210_, 3);
v_saveState_211_ = lean_ctor_get(v_inst_203_, 0);
lean_inc(v_saveState_211_);
v_restoreState_212_ = lean_ctor_get(v_inst_203_, 1);
lean_inc(v_restoreState_212_);
lean_dec_ref(v_inst_203_);
v_toMonadExceptOf_213_ = lean_ctor_get(v_inst_204_, 0);
lean_inc_ref(v_toMonadExceptOf_213_);
v_toPure_214_ = lean_ctor_get(v_toApplicative_209_, 1);
lean_inc_n(v_toPure_214_, 3);
v___f_215_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___closed__0));
v___f_216_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__1), 2, 1);
lean_closure_set(v___f_216_, 0, v_toPure_214_);
v___f_217_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__2), 4, 3);
lean_closure_set(v___f_217_, 0, v_inst_201_);
lean_closure_set(v___f_217_, 1, v_toBind_210_);
lean_closure_set(v___f_217_, 2, v___f_216_);
v___f_218_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_218_, 0, v_toPure_214_);
v___f_219_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__8), 15, 14);
lean_closure_set(v___f_219_, 0, v_toMonadExceptOf_213_);
lean_closure_set(v___f_219_, 1, v_msg_205_);
lean_closure_set(v___f_219_, 2, v_ref_208_);
lean_closure_set(v___f_219_, 3, v_inst_200_);
lean_closure_set(v___f_219_, 4, v_inst_204_);
lean_closure_set(v___f_219_, 5, v_msgref_207_);
lean_closure_set(v___f_219_, 6, v___f_215_);
lean_closure_set(v___f_219_, 7, v_inst_202_);
lean_closure_set(v___f_219_, 8, v_toBind_210_);
lean_closure_set(v___f_219_, 9, v_toPure_214_);
lean_closure_set(v___f_219_, 10, v_restoreState_212_);
lean_closure_set(v___f_219_, 11, v_tacs_206_);
lean_closure_set(v___f_219_, 12, v___f_218_);
lean_closure_set(v___f_219_, 13, v___f_217_);
v___x_220_ = lean_apply_4(v_toBind_210_, lean_box(0), lean_box(0), v_saveState_211_, v___f_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage(lean_object* v_s_221_, lean_object* v_00_u03b1_222_, lean_object* v_m_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_msg_229_, lean_object* v_tacs_230_, lean_object* v_msgref_231_, lean_object* v_ref_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg(v_inst_224_, v_inst_225_, v_inst_226_, v_inst_227_, v_inst_228_, v_msg_229_, v_tacs_230_, v_msgref_231_, v_ref_232_);
return v___x_233_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__2(void){
_start:
{
lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_237_ = lean_box(0);
v___x_238_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__1));
v___x_239_ = l_Lean_Expr_const___override(v___x_238_, v___x_237_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1(lean_object* v_msg_240_, lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_, lean_object* v_a_244_, lean_object* v_a_245_, lean_object* v_a_246_){
_start:
{
lean_object* v___x_248_; uint8_t v___x_249_; lean_object* v___x_250_; 
v___x_248_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__2, &lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__2);
v___x_249_ = 1;
v___x_250_ = l_Lean_Elab_Term_evalTerm___redArg(v___x_248_, v_msg_240_, v___x_249_, v_a_241_, v_a_242_, v_a_243_, v_a_244_, v_a_245_, v_a_246_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___boxed(lean_object* v_msg_251_, lean_object* v_a_252_, lean_object* v_a_253_, lean_object* v_a_254_, lean_object* v_a_255_, lean_object* v_a_256_, lean_object* v_a_257_, lean_object* v_a_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1(v_msg_251_, v_a_252_, v_a_253_, v_a_254_, v_a_255_, v_a_256_, v_a_257_);
lean_dec(v_a_257_);
lean_dec_ref(v_a_256_);
lean_dec(v_a_255_);
lean_dec_ref(v_a_254_);
lean_dec(v_a_253_);
lean_dec_ref(v_a_252_);
return v_res_259_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_260_ = lean_box(0);
v___x_261_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_262_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_261_);
lean_ctor_set(v___x_262_, 1, v___x_260_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg(){
_start:
{
lean_object* v___x_264_; lean_object* v___x_265_; 
v___x_264_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg___closed__0);
v___x_265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_265_, 0, v___x_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg___boxed(lean_object* v___y_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg();
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0(lean_object* v_00_u03b1_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg();
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___boxed(lean_object* v_00_u03b1_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0(v_00_u03b1_279_, v___y_280_, v___y_281_, v___y_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_);
lean_dec(v___y_287_);
lean_dec_ref(v___y_286_);
lean_dec(v___y_285_);
lean_dec_ref(v___y_284_);
lean_dec(v___y_283_);
lean_dec_ref(v___y_282_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2___redArg(lean_object* v_a_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_){
_start:
{
lean_object* v___x_300_; lean_object* v___x_301_; 
lean_inc(v___y_292_);
lean_inc_ref(v___y_291_);
v___x_300_ = lean_apply_2(v_a_290_, v___y_291_, v___y_292_);
v___x_301_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v___x_300_, v___y_293_, v___y_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2___redArg___boxed(lean_object* v_a_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2___redArg(v_a_302_, v___y_303_, v___y_304_, v___y_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_);
lean_dec(v___y_310_);
lean_dec_ref(v___y_309_);
lean_dec(v___y_308_);
lean_dec_ref(v___y_307_);
lean_dec(v___y_306_);
lean_dec_ref(v___y_305_);
lean_dec(v___y_304_);
lean_dec_ref(v___y_303_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2(lean_object* v_00_u03b1_313_, lean_object* v_a_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2___redArg(v_a_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_, v___y_322_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2___boxed(lean_object* v_00_u03b1_325_, lean_object* v_a_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2(v_00_u03b1_325_, v_a_326_, v___y_327_, v___y_328_, v___y_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_);
lean_dec(v___y_334_);
lean_dec_ref(v___y_333_);
lean_dec(v___y_332_);
lean_dec_ref(v___y_331_);
lean_dec(v___y_330_);
lean_dec_ref(v___y_329_);
lean_dec(v___y_328_);
lean_dec_ref(v___y_327_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2_spec__4(lean_object* v_msgData_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_){
_start:
{
lean_object* v___x_343_; lean_object* v_env_344_; lean_object* v___x_345_; lean_object* v_mctx_346_; lean_object* v_lctx_347_; lean_object* v_options_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_343_ = lean_st_ref_get(v___y_341_);
v_env_344_ = lean_ctor_get(v___x_343_, 0);
lean_inc_ref(v_env_344_);
lean_dec(v___x_343_);
v___x_345_ = lean_st_ref_get(v___y_339_);
v_mctx_346_ = lean_ctor_get(v___x_345_, 0);
lean_inc_ref(v_mctx_346_);
lean_dec(v___x_345_);
v_lctx_347_ = lean_ctor_get(v___y_338_, 2);
v_options_348_ = lean_ctor_get(v___y_340_, 2);
lean_inc_ref(v_options_348_);
lean_inc_ref(v_lctx_347_);
v___x_349_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_349_, 0, v_env_344_);
lean_ctor_set(v___x_349_, 1, v_mctx_346_);
lean_ctor_set(v___x_349_, 2, v_lctx_347_);
lean_ctor_set(v___x_349_, 3, v_options_348_);
v___x_350_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_349_);
lean_ctor_set(v___x_350_, 1, v_msgData_337_);
v___x_351_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_351_, 0, v___x_350_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2_spec__4___boxed(lean_object* v_msgData_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_){
_start:
{
lean_object* v_res_358_; 
v_res_358_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2_spec__4(v_msgData_352_, v___y_353_, v___y_354_, v___y_355_, v___y_356_);
lean_dec(v___y_356_);
lean_dec_ref(v___y_355_);
lean_dec(v___y_354_);
lean_dec_ref(v___y_353_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___redArg(lean_object* v_msg_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_){
_start:
{
lean_object* v_ref_365_; lean_object* v___x_366_; lean_object* v_a_367_; lean_object* v___x_369_; uint8_t v_isShared_370_; uint8_t v_isSharedCheck_375_; 
v_ref_365_ = lean_ctor_get(v___y_362_, 5);
v___x_366_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2_spec__4(v_msg_359_, v___y_360_, v___y_361_, v___y_362_, v___y_363_);
v_a_367_ = lean_ctor_get(v___x_366_, 0);
v_isSharedCheck_375_ = !lean_is_exclusive(v___x_366_);
if (v_isSharedCheck_375_ == 0)
{
v___x_369_ = v___x_366_;
v_isShared_370_ = v_isSharedCheck_375_;
goto v_resetjp_368_;
}
else
{
lean_inc(v_a_367_);
lean_dec(v___x_366_);
v___x_369_ = lean_box(0);
v_isShared_370_ = v_isSharedCheck_375_;
goto v_resetjp_368_;
}
v_resetjp_368_:
{
lean_object* v___x_371_; lean_object* v___x_373_; 
lean_inc(v_ref_365_);
v___x_371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_371_, 0, v_ref_365_);
lean_ctor_set(v___x_371_, 1, v_a_367_);
if (v_isShared_370_ == 0)
{
lean_ctor_set_tag(v___x_369_, 1);
lean_ctor_set(v___x_369_, 0, v___x_371_);
v___x_373_ = v___x_369_;
goto v_reusejp_372_;
}
else
{
lean_object* v_reuseFailAlloc_374_; 
v_reuseFailAlloc_374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_374_, 0, v___x_371_);
v___x_373_ = v_reuseFailAlloc_374_;
goto v_reusejp_372_;
}
v_reusejp_372_:
{
return v___x_373_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___redArg___boxed(lean_object* v_msg_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___redArg(v_msg_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_);
lean_dec(v___y_380_);
lean_dec_ref(v___y_379_);
lean_dec(v___y_378_);
lean_dec_ref(v___y_377_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1___redArg(lean_object* v_ref_383_, lean_object* v_msg_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_){
_start:
{
lean_object* v_fileName_394_; lean_object* v_fileMap_395_; lean_object* v_options_396_; lean_object* v_currRecDepth_397_; lean_object* v_maxRecDepth_398_; lean_object* v_ref_399_; lean_object* v_currNamespace_400_; lean_object* v_openDecls_401_; lean_object* v_initHeartbeats_402_; lean_object* v_maxHeartbeats_403_; lean_object* v_quotContext_404_; lean_object* v_currMacroScope_405_; uint8_t v_diag_406_; lean_object* v_cancelTk_x3f_407_; uint8_t v_suppressElabErrors_408_; lean_object* v_inheritedTraceOptions_409_; lean_object* v_ref_410_; lean_object* v___x_411_; lean_object* v___x_412_; 
v_fileName_394_ = lean_ctor_get(v___y_391_, 0);
v_fileMap_395_ = lean_ctor_get(v___y_391_, 1);
v_options_396_ = lean_ctor_get(v___y_391_, 2);
v_currRecDepth_397_ = lean_ctor_get(v___y_391_, 3);
v_maxRecDepth_398_ = lean_ctor_get(v___y_391_, 4);
v_ref_399_ = lean_ctor_get(v___y_391_, 5);
v_currNamespace_400_ = lean_ctor_get(v___y_391_, 6);
v_openDecls_401_ = lean_ctor_get(v___y_391_, 7);
v_initHeartbeats_402_ = lean_ctor_get(v___y_391_, 8);
v_maxHeartbeats_403_ = lean_ctor_get(v___y_391_, 9);
v_quotContext_404_ = lean_ctor_get(v___y_391_, 10);
v_currMacroScope_405_ = lean_ctor_get(v___y_391_, 11);
v_diag_406_ = lean_ctor_get_uint8(v___y_391_, sizeof(void*)*14);
v_cancelTk_x3f_407_ = lean_ctor_get(v___y_391_, 12);
v_suppressElabErrors_408_ = lean_ctor_get_uint8(v___y_391_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_409_ = lean_ctor_get(v___y_391_, 13);
v_ref_410_ = l_Lean_replaceRef(v_ref_383_, v_ref_399_);
lean_inc_ref(v_inheritedTraceOptions_409_);
lean_inc(v_cancelTk_x3f_407_);
lean_inc(v_currMacroScope_405_);
lean_inc(v_quotContext_404_);
lean_inc(v_maxHeartbeats_403_);
lean_inc(v_initHeartbeats_402_);
lean_inc(v_openDecls_401_);
lean_inc(v_currNamespace_400_);
lean_inc(v_maxRecDepth_398_);
lean_inc(v_currRecDepth_397_);
lean_inc_ref(v_options_396_);
lean_inc_ref(v_fileMap_395_);
lean_inc_ref(v_fileName_394_);
v___x_411_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_411_, 0, v_fileName_394_);
lean_ctor_set(v___x_411_, 1, v_fileMap_395_);
lean_ctor_set(v___x_411_, 2, v_options_396_);
lean_ctor_set(v___x_411_, 3, v_currRecDepth_397_);
lean_ctor_set(v___x_411_, 4, v_maxRecDepth_398_);
lean_ctor_set(v___x_411_, 5, v_ref_410_);
lean_ctor_set(v___x_411_, 6, v_currNamespace_400_);
lean_ctor_set(v___x_411_, 7, v_openDecls_401_);
lean_ctor_set(v___x_411_, 8, v_initHeartbeats_402_);
lean_ctor_set(v___x_411_, 9, v_maxHeartbeats_403_);
lean_ctor_set(v___x_411_, 10, v_quotContext_404_);
lean_ctor_set(v___x_411_, 11, v_currMacroScope_405_);
lean_ctor_set(v___x_411_, 12, v_cancelTk_x3f_407_);
lean_ctor_set(v___x_411_, 13, v_inheritedTraceOptions_409_);
lean_ctor_set_uint8(v___x_411_, sizeof(void*)*14, v_diag_406_);
lean_ctor_set_uint8(v___x_411_, sizeof(void*)*14 + 1, v_suppressElabErrors_408_);
v___x_412_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___redArg(v_msg_384_, v___y_389_, v___y_390_, v___x_411_, v___y_392_);
lean_dec_ref_known(v___x_411_, 14);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1___redArg___boxed(lean_object* v_ref_413_, lean_object* v_msg_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_){
_start:
{
lean_object* v_res_424_; 
v_res_424_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1___redArg(v_ref_413_, v_msg_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_, v___y_419_, v___y_420_, v___y_421_, v___y_422_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
lean_dec(v___y_420_);
lean_dec_ref(v___y_419_);
lean_dec(v___y_418_);
lean_dec_ref(v___y_417_);
lean_dec(v___y_416_);
lean_dec_ref(v___y_415_);
lean_dec(v_ref_413_);
return v_res_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg(lean_object* v_msg_427_, lean_object* v_tacs_428_, lean_object* v_msgref_429_, lean_object* v_ref_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_){
_start:
{
lean_object* v___y_441_; lean_object* v___y_442_; lean_object* v___y_443_; lean_object* v___y_444_; lean_object* v___y_445_; lean_object* v___y_446_; lean_object* v___y_447_; lean_object* v___y_448_; lean_object* v___y_449_; lean_object* v___x_463_; 
v___x_463_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_432_, v___y_434_, v___y_436_, v___y_438_);
if (lean_obj_tag(v___x_463_) == 0)
{
lean_object* v_a_464_; lean_object* v_a_466_; lean_object* v___x_515_; 
v_a_464_ = lean_ctor_get(v___x_463_, 0);
lean_inc(v_a_464_);
lean_dec_ref_known(v___x_463_, 1);
lean_inc(v___y_438_);
lean_inc_ref(v___y_437_);
lean_inc(v___y_436_);
lean_inc_ref(v___y_435_);
lean_inc(v___y_434_);
lean_inc_ref(v___y_433_);
lean_inc(v___y_432_);
lean_inc_ref(v___y_431_);
v___x_515_ = lean_apply_9(v_tacs_428_, v___y_431_, v___y_432_, v___y_433_, v___y_434_, v___y_435_, v___y_436_, v___y_437_, v___y_438_, lean_box(0));
if (lean_obj_tag(v___x_515_) == 0)
{
lean_object* v___x_516_; 
lean_dec_ref_known(v___x_515_, 1);
v___x_516_ = lean_box(0);
v_a_466_ = v___x_516_;
goto v___jp_465_;
}
else
{
lean_object* v_a_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_531_; 
v_a_517_ = lean_ctor_get(v___x_515_, 0);
v_isSharedCheck_531_ = !lean_is_exclusive(v___x_515_);
if (v_isSharedCheck_531_ == 0)
{
v___x_519_ = v___x_515_;
v_isShared_520_ = v_isSharedCheck_531_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_a_517_);
lean_dec(v___x_515_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_531_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
uint8_t v___y_522_; uint8_t v___x_529_; 
v___x_529_ = l_Lean_Exception_isInterrupt(v_a_517_);
if (v___x_529_ == 0)
{
uint8_t v___x_530_; 
lean_inc(v_a_517_);
v___x_530_ = l_Lean_Exception_isRuntime(v_a_517_);
v___y_522_ = v___x_530_;
goto v___jp_521_;
}
else
{
v___y_522_ = v___x_529_;
goto v___jp_521_;
}
v___jp_521_:
{
if (v___y_522_ == 0)
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
lean_del_object(v___x_519_);
v___x_523_ = l_Lean_Exception_toMessageData(v_a_517_);
v___x_524_ = l_Lean_MessageData_toString(v___x_523_);
v___x_525_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_525_, 0, v___x_524_);
v_a_466_ = v___x_525_;
goto v___jp_465_;
}
else
{
lean_object* v___x_527_; 
lean_dec(v_a_464_);
lean_dec(v_ref_430_);
lean_dec(v_msgref_429_);
lean_dec_ref(v_msg_427_);
if (v_isShared_520_ == 0)
{
v___x_527_ = v___x_519_;
goto v_reusejp_526_;
}
else
{
lean_object* v_reuseFailAlloc_528_; 
v_reuseFailAlloc_528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_528_, 0, v_a_517_);
v___x_527_ = v_reuseFailAlloc_528_;
goto v_reusejp_526_;
}
v_reusejp_526_:
{
return v___x_527_;
}
}
}
}
}
v___jp_465_:
{
uint8_t v___x_467_; lean_object* v___x_468_; 
v___x_467_ = 0;
v___x_468_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_464_, v___x_467_, v___y_432_, v___y_433_, v___y_434_, v___y_435_, v___y_436_, v___y_437_, v___y_438_);
if (lean_obj_tag(v___x_468_) == 0)
{
lean_object* v___x_470_; uint8_t v_isShared_471_; uint8_t v_isSharedCheck_513_; 
v_isSharedCheck_513_ = !lean_is_exclusive(v___x_468_);
if (v_isSharedCheck_513_ == 0)
{
lean_object* v_unused_514_; 
v_unused_514_ = lean_ctor_get(v___x_468_, 0);
lean_dec(v_unused_514_);
v___x_470_ = v___x_468_;
v_isShared_471_ = v_isSharedCheck_513_;
goto v_resetjp_469_;
}
else
{
lean_dec(v___x_468_);
v___x_470_ = lean_box(0);
v_isShared_471_ = v_isSharedCheck_513_;
goto v_resetjp_469_;
}
v_resetjp_469_:
{
if (lean_obj_tag(v_a_466_) == 1)
{
lean_object* v_val_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; uint8_t v___x_480_; 
v_val_472_ = lean_ctor_get(v_a_466_, 0);
lean_inc_n(v_val_472_, 2);
lean_dec_ref_known(v_a_466_, 1);
v___x_473_ = lean_unsigned_to_nat(0u);
v___x_474_ = lean_string_utf8_byte_size(v_msg_427_);
v___x_475_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_475_, 0, v_msg_427_);
lean_ctor_set(v___x_475_, 1, v___x_473_);
lean_ctor_set(v___x_475_, 2, v___x_474_);
v___x_476_ = l_String_Slice_trimAscii(v___x_475_);
v___x_477_ = lean_string_utf8_byte_size(v_val_472_);
v___x_478_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_478_, 0, v_val_472_);
lean_ctor_set(v___x_478_, 1, v___x_473_);
lean_ctor_set(v___x_478_, 2, v___x_477_);
v___x_479_ = l_String_Slice_trimAscii(v___x_478_);
v___x_480_ = l_String_Slice_beq(v___x_476_, v___x_479_);
lean_dec_ref(v___x_476_);
if (v___x_480_ == 0)
{
lean_del_object(v___x_470_);
if (lean_obj_tag(v_msgref_429_) == 1)
{
lean_object* v_val_481_; lean_object* v___x_483_; uint8_t v_isShared_484_; uint8_t v_isSharedCheck_499_; 
v_val_481_ = lean_ctor_get(v_msgref_429_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v_msgref_429_);
if (v_isSharedCheck_499_ == 0)
{
v___x_483_ = v_msgref_429_;
v_isShared_484_ = v_isSharedCheck_499_;
goto v_resetjp_482_;
}
else
{
lean_inc(v_val_481_);
lean_dec(v_msgref_429_);
v___x_483_ = lean_box(0);
v_isShared_484_ = v_isSharedCheck_499_;
goto v_resetjp_482_;
}
v_resetjp_482_:
{
lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_490_; 
v___x_485_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__0));
v___x_486_ = l_String_Slice_toString(v___x_479_);
lean_dec_ref(v___x_479_);
v___x_487_ = lean_string_append(v___x_485_, v___x_486_);
lean_dec_ref(v___x_486_);
v___x_488_ = lean_string_append(v___x_487_, v___x_485_);
if (v_isShared_484_ == 0)
{
lean_ctor_set(v___x_483_, 0, v___x_488_);
v___x_490_ = v___x_483_;
goto v_reusejp_489_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v___x_488_);
v___x_490_ = v_reuseFailAlloc_498_;
goto v_reusejp_489_;
}
v_reusejp_489_:
{
lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v_suggestion_493_; lean_object* v___x_494_; uint8_t v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; 
v___x_491_ = lean_box(0);
v___x_492_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg___closed__0));
v_suggestion_493_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_suggestion_493_, 0, v___x_490_);
lean_ctor_set(v_suggestion_493_, 1, v___x_491_);
lean_ctor_set(v_suggestion_493_, 2, v___x_491_);
lean_ctor_set(v_suggestion_493_, 3, v___x_491_);
lean_ctor_set(v_suggestion_493_, 4, v___x_491_);
lean_ctor_set(v_suggestion_493_, 5, v___x_492_);
v___x_494_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__1));
v___x_495_ = 4;
v___x_496_ = l_Lean_MessageData_nil;
v___x_497_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_val_481_, v_suggestion_493_, v___x_491_, v___x_494_, v___x_491_, v___x_495_, v___x_496_, v___y_437_, v___y_438_);
if (lean_obj_tag(v___x_497_) == 0)
{
lean_dec_ref_known(v___x_497_, 1);
v___y_441_ = v_val_472_;
v___y_442_ = v___y_431_;
v___y_443_ = v___y_432_;
v___y_444_ = v___y_433_;
v___y_445_ = v___y_434_;
v___y_446_ = v___y_435_;
v___y_447_ = v___y_436_;
v___y_448_ = v___y_437_;
v___y_449_ = v___y_438_;
goto v___jp_440_;
}
else
{
lean_dec(v_val_472_);
lean_dec(v_ref_430_);
return v___x_497_;
}
}
}
}
else
{
lean_dec_ref(v___x_479_);
lean_dec(v_msgref_429_);
v___y_441_ = v_val_472_;
v___y_442_ = v___y_431_;
v___y_443_ = v___y_432_;
v___y_444_ = v___y_433_;
v___y_445_ = v___y_434_;
v___y_446_ = v___y_435_;
v___y_447_ = v___y_436_;
v___y_448_ = v___y_437_;
v___y_449_ = v___y_438_;
goto v___jp_440_;
}
}
else
{
lean_object* v___x_500_; lean_object* v___x_502_; 
lean_dec_ref(v___x_479_);
lean_dec(v_val_472_);
lean_dec(v_ref_430_);
lean_dec(v_msgref_429_);
v___x_500_ = lean_box(0);
if (v_isShared_471_ == 0)
{
lean_ctor_set(v___x_470_, 0, v___x_500_);
v___x_502_ = v___x_470_;
goto v_reusejp_501_;
}
else
{
lean_object* v_reuseFailAlloc_503_; 
v_reuseFailAlloc_503_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_503_, 0, v___x_500_);
v___x_502_ = v_reuseFailAlloc_503_;
goto v_reusejp_501_;
}
v_reusejp_501_:
{
return v___x_502_;
}
}
}
else
{
lean_del_object(v___x_470_);
lean_dec(v_a_466_);
lean_dec(v_msgref_429_);
lean_dec_ref(v_msg_427_);
if (lean_obj_tag(v_ref_430_) == 1)
{
lean_object* v_val_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; 
v_val_504_ = lean_ctor_get(v_ref_430_, 0);
lean_inc_n(v_val_504_, 2);
lean_dec_ref_known(v_ref_430_, 1);
v___x_505_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1);
v___x_506_ = l_Lean_MessageData_ofSyntax(v_val_504_);
v___x_507_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_507_, 0, v___x_505_);
lean_ctor_set(v___x_507_, 1, v___x_506_);
v___x_508_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__3, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__3);
v___x_509_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_509_, 0, v___x_507_);
lean_ctor_set(v___x_509_, 1, v___x_508_);
v___x_510_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1___redArg(v_val_504_, v___x_509_, v___y_431_, v___y_432_, v___y_433_, v___y_434_, v___y_435_, v___y_436_, v___y_437_, v___y_438_);
lean_dec(v_val_504_);
return v___x_510_;
}
else
{
lean_object* v___x_511_; lean_object* v___x_512_; 
lean_dec(v_ref_430_);
v___x_511_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__5, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__6___closed__5);
v___x_512_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___redArg(v___x_511_, v___y_435_, v___y_436_, v___y_437_, v___y_438_);
return v___x_512_;
}
}
}
}
else
{
lean_dec(v_a_466_);
lean_dec(v_ref_430_);
lean_dec(v_msgref_429_);
lean_dec_ref(v_msg_427_);
return v___x_468_;
}
}
}
else
{
lean_object* v_a_532_; lean_object* v___x_534_; uint8_t v_isShared_535_; uint8_t v_isSharedCheck_539_; 
lean_dec(v_ref_430_);
lean_dec(v_msgref_429_);
lean_dec_ref(v_tacs_428_);
lean_dec_ref(v_msg_427_);
v_a_532_ = lean_ctor_get(v___x_463_, 0);
v_isSharedCheck_539_ = !lean_is_exclusive(v___x_463_);
if (v_isSharedCheck_539_ == 0)
{
v___x_534_ = v___x_463_;
v_isShared_535_ = v_isSharedCheck_539_;
goto v_resetjp_533_;
}
else
{
lean_inc(v_a_532_);
lean_dec(v___x_463_);
v___x_534_ = lean_box(0);
v_isShared_535_ = v_isSharedCheck_539_;
goto v_resetjp_533_;
}
v_resetjp_533_:
{
lean_object* v___x_537_; 
if (v_isShared_535_ == 0)
{
v___x_537_ = v___x_534_;
goto v_reusejp_536_;
}
else
{
lean_object* v_reuseFailAlloc_538_; 
v_reuseFailAlloc_538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_538_, 0, v_a_532_);
v___x_537_ = v_reuseFailAlloc_538_;
goto v_reusejp_536_;
}
v_reusejp_536_:
{
return v___x_537_;
}
}
}
v___jp_440_:
{
if (lean_obj_tag(v_ref_430_) == 1)
{
lean_object* v_val_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; 
v_val_450_ = lean_ctor_get(v_ref_430_, 0);
lean_inc_n(v_val_450_, 2);
lean_dec_ref_known(v_ref_430_, 1);
v___x_451_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__1);
v___x_452_ = l_Lean_MessageData_ofSyntax(v_val_450_);
v___x_453_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_453_, 0, v___x_451_);
lean_ctor_set(v___x_453_, 1, v___x_452_);
v___x_454_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__3, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__3);
v___x_455_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_455_, 0, v___x_453_);
lean_ctor_set(v___x_455_, 1, v___x_454_);
v___x_456_ = l_Lean_stringToMessageData(v___y_441_);
v___x_457_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_457_, 0, v___x_455_);
lean_ctor_set(v___x_457_, 1, v___x_456_);
v___x_458_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1___redArg(v_val_450_, v___x_457_, v___y_442_, v___y_443_, v___y_444_, v___y_445_, v___y_446_, v___y_447_, v___y_448_, v___y_449_);
lean_dec(v_val_450_);
return v___x_458_;
}
else
{
lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; 
lean_dec(v_ref_430_);
v___x_459_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__5, &lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___redArg___lam__4___closed__5);
v___x_460_ = l_Lean_stringToMessageData(v___y_441_);
v___x_461_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_461_, 0, v___x_459_);
lean_ctor_set(v___x_461_, 1, v___x_460_);
v___x_462_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___redArg(v___x_461_, v___y_446_, v___y_447_, v___y_448_, v___y_449_);
return v___x_462_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg___boxed(lean_object* v_msg_540_, lean_object* v_tacs_541_, lean_object* v_msgref_542_, lean_object* v_ref_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_){
_start:
{
lean_object* v_res_553_; 
v_res_553_ = lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg(v_msg_540_, v_tacs_541_, v_msgref_542_, v_ref_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_, v___y_549_, v___y_550_, v___y_551_);
lean_dec(v___y_551_);
lean_dec_ref(v___y_550_);
lean_dec(v___y_549_);
lean_dec_ref(v___y_548_);
lean_dec(v___y_547_);
lean_dec_ref(v___y_546_);
lean_dec(v___y_545_);
lean_dec_ref(v___y_544_);
return v_res_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___lam__0(lean_object* v___x_554_, lean_object* v___x_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_){
_start:
{
lean_object* v___x_565_; uint8_t v___x_566_; lean_object* v___x_567_; 
v___x_565_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__2, &lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_SuccessIfFailWithMsg_0__Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_unsafe__1___closed__2);
v___x_566_ = 1;
lean_inc(v___x_554_);
v___x_567_ = l_Lean_Elab_Term_evalTerm___redArg(v___x_565_, v___x_554_, v___x_566_, v___y_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_, v___y_563_);
if (lean_obj_tag(v___x_567_) == 0)
{
lean_object* v_a_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; 
v_a_568_ = lean_ctor_get(v___x_567_, 0);
lean_inc(v_a_568_);
lean_dec_ref_known(v___x_567_, 1);
lean_inc(v___x_555_);
v___x_569_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTacticSeq___boxed), 10, 1);
lean_closure_set(v___x_569_, 0, v___x_555_);
v___x_570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_570_, 0, v___x_554_);
v___x_571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_571_, 0, v___x_555_);
v___x_572_ = lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg(v_a_568_, v___x_569_, v___x_570_, v___x_571_, v___y_556_, v___y_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_, v___y_563_);
return v___x_572_;
}
else
{
lean_object* v_a_573_; lean_object* v___x_575_; uint8_t v_isShared_576_; uint8_t v_isSharedCheck_580_; 
lean_dec(v___x_555_);
lean_dec(v___x_554_);
v_a_573_ = lean_ctor_get(v___x_567_, 0);
v_isSharedCheck_580_ = !lean_is_exclusive(v___x_567_);
if (v_isSharedCheck_580_ == 0)
{
v___x_575_ = v___x_567_;
v_isShared_576_ = v_isSharedCheck_580_;
goto v_resetjp_574_;
}
else
{
lean_inc(v_a_573_);
lean_dec(v___x_567_);
v___x_575_ = lean_box(0);
v_isShared_576_ = v_isSharedCheck_580_;
goto v_resetjp_574_;
}
v_resetjp_574_:
{
lean_object* v___x_578_; 
if (v_isShared_576_ == 0)
{
v___x_578_ = v___x_575_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_579_; 
v_reuseFailAlloc_579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_579_, 0, v_a_573_);
v___x_578_ = v_reuseFailAlloc_579_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
return v___x_578_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___lam__0___boxed(lean_object* v___x_581_, lean_object* v___x_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___lam__0(v___x_581_, v___x_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_, v___y_589_, v___y_590_);
lean_dec(v___y_590_);
lean_dec_ref(v___y_589_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v___y_584_);
lean_dec_ref(v___y_583_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1(lean_object* v_x_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_, lean_object* v_a_605_, lean_object* v_a_606_, lean_object* v_a_607_, lean_object* v_a_608_){
_start:
{
lean_object* v___x_610_; uint8_t v___x_611_; 
v___x_610_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_successIfFailWithMsg___closed__3));
lean_inc(v_x_600_);
v___x_611_ = l_Lean_Syntax_isOfKind(v_x_600_, v___x_610_);
if (v___x_611_ == 0)
{
lean_object* v___x_612_; 
lean_dec(v_x_600_);
v___x_612_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg();
return v___x_612_;
}
else
{
lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; uint8_t v___x_616_; 
v___x_613_ = lean_unsigned_to_nat(2u);
v___x_614_ = l_Lean_Syntax_getArg(v_x_600_, v___x_613_);
v___x_615_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___closed__2));
lean_inc(v___x_614_);
v___x_616_ = l_Lean_Syntax_isOfKind(v___x_614_, v___x_615_);
if (v___x_616_ == 0)
{
lean_object* v___x_617_; 
lean_dec(v___x_614_);
lean_dec(v_x_600_);
v___x_617_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__0___redArg();
return v___x_617_;
}
else
{
lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___f_620_; lean_object* v___x_621_; lean_object* v___x_622_; 
v___x_618_ = lean_unsigned_to_nat(1u);
v___x_619_ = l_Lean_Syntax_getArg(v_x_600_, v___x_618_);
lean_dec(v_x_600_);
v___f_620_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___lam__0___boxed), 11, 2);
lean_closure_set(v___f_620_, 0, v___x_619_);
lean_closure_set(v___f_620_, 1, v___x_614_);
v___x_621_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withoutRecover___boxed), 11, 2);
lean_closure_set(v___x_621_, 0, lean_box(0));
lean_closure_set(v___x_621_, 1, v___f_620_);
v___x_622_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__2___redArg(v___x_621_, v_a_601_, v_a_602_, v_a_603_, v_a_604_, v_a_605_, v_a_606_, v_a_607_, v_a_608_);
return v___x_622_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1___boxed(lean_object* v_x_623_, lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_, lean_object* v_a_630_, lean_object* v_a_631_, lean_object* v_a_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1(v_x_623_, v_a_624_, v_a_625_, v_a_626_, v_a_627_, v_a_628_, v_a_629_, v_a_630_, v_a_631_);
lean_dec(v_a_631_);
lean_dec_ref(v_a_630_);
lean_dec(v_a_629_);
lean_dec_ref(v_a_628_);
lean_dec(v_a_627_);
lean_dec_ref(v_a_626_);
lean_dec(v_a_625_);
lean_dec_ref(v_a_624_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1(lean_object* v_00_u03b1_634_, lean_object* v_msg_635_, lean_object* v_tacs_636_, lean_object* v_msgref_637_, lean_object* v_ref_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_){
_start:
{
lean_object* v___x_648_; 
v___x_648_ = lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___redArg(v_msg_635_, v_tacs_636_, v_msgref_637_, v_ref_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_);
return v___x_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1___boxed(lean_object* v_00_u03b1_649_, lean_object* v_msg_650_, lean_object* v_tacs_651_, lean_object* v_msgref_652_, lean_object* v_ref_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_){
_start:
{
lean_object* v_res_663_; 
v_res_663_ = lp_mathlib_Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1(v_00_u03b1_649_, v_msg_650_, v_tacs_651_, v_msgref_652_, v_ref_653_, v___y_654_, v___y_655_, v___y_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_, v___y_661_);
lean_dec(v___y_661_);
lean_dec_ref(v___y_660_);
lean_dec(v___y_659_);
lean_dec_ref(v___y_658_);
lean_dec(v___y_657_);
lean_dec_ref(v___y_656_);
lean_dec(v___y_655_);
lean_dec_ref(v___y_654_);
return v_res_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1(lean_object* v_00_u03b1_664_, lean_object* v_ref_665_, lean_object* v_msg_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_){
_start:
{
lean_object* v___x_676_; 
v___x_676_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1___redArg(v_ref_665_, v_msg_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_, v___y_672_, v___y_673_, v___y_674_);
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1___boxed(lean_object* v_00_u03b1_677_, lean_object* v_ref_678_, lean_object* v_msg_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_, lean_object* v___y_686_, lean_object* v___y_687_, lean_object* v___y_688_){
_start:
{
lean_object* v_res_689_; 
v_res_689_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__1(v_00_u03b1_677_, v_ref_678_, v_msg_679_, v___y_680_, v___y_681_, v___y_682_, v___y_683_, v___y_684_, v___y_685_, v___y_686_, v___y_687_);
lean_dec(v___y_687_);
lean_dec_ref(v___y_686_);
lean_dec(v___y_685_);
lean_dec_ref(v___y_684_);
lean_dec(v___y_683_);
lean_dec_ref(v___y_682_);
lean_dec(v___y_681_);
lean_dec_ref(v___y_680_);
lean_dec(v_ref_678_);
return v_res_689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2(lean_object* v_00_u03b1_690_, lean_object* v_msg_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_){
_start:
{
lean_object* v___x_701_; 
v___x_701_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___redArg(v_msg_691_, v___y_696_, v___y_697_, v___y_698_, v___y_699_);
return v___x_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2___boxed(lean_object* v_00_u03b1_702_, lean_object* v_msg_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_){
_start:
{
lean_object* v_res_713_; 
v_res_713_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_successIfFailWithMessage___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SuccessIfFailWithMsg______elabRules__Mathlib__Tactic__successIfFailWithMsg__1_spec__1_spec__2(v_00_u03b1_702_, v_msg_703_, v___y_704_, v___y_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_, v___y_710_, v___y_711_);
lean_dec(v___y_711_);
lean_dec_ref(v___y_710_);
lean_dec(v___y_709_);
lean_dec_ref(v___y_708_);
lean_dec(v___y_707_);
lean_dec_ref(v___y_706_);
lean_dec(v___y_705_);
lean_dec_ref(v___y_704_);
return v_res_713_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Eval(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_BuiltinTactic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_BuiltinTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Eval(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_BuiltinTactic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_BuiltinTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(builtin);
}
#ifdef __cplusplus
}
#endif
