// Lean compiler output
// Module: Mathlib.Util.CodeActions.BinderPlicity
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Linter.Header public meta import Lean.Server.CodeActions.Basic
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
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
lean_object* l_Lean_FileMap_utf8RangeToLspRange(lean_object*, lean_object*);
uint8_t l_Lean_Lsp_instOrdPosition_ord(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_unsetTrailing(lean_object*);
lean_object* l_Lean_Syntax_reprint(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(lean_object*);
lean_object* l_Lean_Lsp_WorkspaceEdit_ofTextEdit(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getHeadInfo(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Syntax_topDown(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_CodeAction_binderPlicity_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_CodeAction_binderPlicity_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Make "};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "refactor.rewrite"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "choice"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(59, 66, 148, 42, 181, 100, 85, 166)}};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "implicit"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__3 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "explicit"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__4 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__5 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "explicitBinder"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__9 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__9_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__8 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__7 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__7_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__6 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(49, 119, 193, 23, 170, 93, 183, 238)}};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__10 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__10_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "implicitBinder"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__11 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__12_value_aux_1),((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__12_value_aux_2),((lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(39, 181, 62, 102, 86, 14, 161, 96)}};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__12 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__12_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__13 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__13_value;
static const lean_string_object lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__14 = (const lean_object*)&lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__14_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_CodeAction_binderPlicity___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_CodeAction_binderPlicity___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CodeAction_binderPlicity___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CodeAction_binderPlicity(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CodeAction_binderPlicity___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_CodeAction_binderPlicity_spec__0(lean_object* v___y_1_){
_start:
{
lean_object* v_doc_3_; lean_object* v___x_4_; 
v_doc_3_ = lean_ctor_get(v___y_1_, 1);
lean_inc_ref(v_doc_3_);
v___x_4_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4_, 0, v_doc_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_CodeAction_binderPlicity_spec__0___boxed(lean_object* v___y_5_, lean_object* v___y_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_CodeAction_binderPlicity_spec__0(v___y_5_);
lean_dec_ref(v___y_5_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0(lean_object* v_a_13_, lean_object* v_kind_14_, lean_object* v_range_15_, lean_object* v_newText_16_){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_17_ = lean_box(0);
v___x_18_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__0));
v___x_19_ = lean_string_append(v___x_18_, v_kind_14_);
v___x_20_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__1));
v___x_21_ = lean_string_append(v___x_19_, v___x_20_);
v___x_22_ = lean_string_append(v___x_21_, v_newText_16_);
v___x_23_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___closed__3));
v___x_24_ = l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(v_a_13_);
v___x_25_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_25_, 0, v_range_15_);
lean_ctor_set(v___x_25_, 1, v_newText_16_);
lean_ctor_set(v___x_25_, 2, v___x_17_);
lean_ctor_set(v___x_25_, 3, v___x_17_);
v___x_26_ = l_Lean_Lsp_WorkspaceEdit_ofTextEdit(v___x_24_, v___x_25_);
v___x_27_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_27_, 0, v___x_26_);
v___x_28_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_28_, 0, v___x_17_);
lean_ctor_set(v___x_28_, 1, v___x_17_);
lean_ctor_set(v___x_28_, 2, v___x_22_);
lean_ctor_set(v___x_28_, 3, v___x_23_);
lean_ctor_set(v___x_28_, 4, v___x_17_);
lean_ctor_set(v___x_28_, 5, v___x_17_);
lean_ctor_set(v___x_28_, 6, v___x_17_);
lean_ctor_set(v___x_28_, 7, v___x_27_);
lean_ctor_set(v___x_28_, 8, v___x_17_);
lean_ctor_set(v___x_28_, 9, v___x_17_);
v___x_29_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_29_, 0, v___x_28_);
lean_ctor_set(v___x_29_, 1, v___x_17_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0___boxed(lean_object* v_a_30_, lean_object* v_kind_31_, lean_object* v_range_32_, lean_object* v_newText_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0(v_a_30_, v_kind_31_, v_range_32_, v_newText_33_);
lean_dec_ref(v_kind_31_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1(lean_object* v_a_59_, lean_object* v_params_60_, uint8_t v_firstChoiceOnly_61_, lean_object* v_stx_62_, lean_object* v_b_63_, lean_object* v___y_64_){
_start:
{
lean_object* v_b_67_; lean_object* v___y_71_; lean_object* v___y_72_; lean_object* v_a_98_; uint8_t v___x_108_; lean_object* v___x_109_; 
v___x_108_ = 0;
v___x_109_ = l_Lean_Syntax_getRange_x3f(v_stx_62_, v___x_108_);
if (lean_obj_tag(v___x_109_) == 1)
{
lean_object* v_toEditableDocumentCore_110_; lean_object* v_meta_111_; lean_object* v_val_112_; lean_object* v_text_113_; lean_object* v___x_114_; lean_object* v_range_115_; lean_object* v_start_116_; lean_object* v_end_117_; lean_object* v_start_118_; lean_object* v_end_119_; lean_object* v___x_121_; uint8_t v_isShared_122_; uint8_t v_isSharedCheck_216_; 
v_toEditableDocumentCore_110_ = lean_ctor_get(v_a_59_, 0);
v_meta_111_ = lean_ctor_get(v_toEditableDocumentCore_110_, 0);
v_val_112_ = lean_ctor_get(v___x_109_, 0);
lean_inc(v_val_112_);
lean_dec_ref_known(v___x_109_, 1);
v_text_113_ = lean_ctor_get(v_meta_111_, 3);
lean_inc_ref(v_text_113_);
v___x_114_ = l_Lean_FileMap_utf8RangeToLspRange(v_text_113_, v_val_112_);
v_range_115_ = lean_ctor_get(v_params_60_, 3);
lean_inc_ref(v_range_115_);
v_start_116_ = lean_ctor_get(v___x_114_, 0);
lean_inc_ref(v_start_116_);
v_end_117_ = lean_ctor_get(v___x_114_, 1);
lean_inc_ref(v_end_117_);
v_start_118_ = lean_ctor_get(v_range_115_, 0);
v_end_119_ = lean_ctor_get(v_range_115_, 1);
v_isSharedCheck_216_ = !lean_is_exclusive(v_range_115_);
if (v_isSharedCheck_216_ == 0)
{
v___x_121_ = v_range_115_;
v_isShared_122_ = v_isSharedCheck_216_;
goto v_resetjp_120_;
}
else
{
lean_inc(v_end_119_);
lean_inc(v_start_118_);
lean_dec(v_range_115_);
v___x_121_ = lean_box(0);
v_isShared_122_ = v_isSharedCheck_216_;
goto v_resetjp_120_;
}
v_resetjp_120_:
{
uint8_t v___x_123_; 
v___x_123_ = l_Lean_Lsp_instOrdPosition_ord(v_start_116_, v_end_119_);
lean_dec_ref(v_end_119_);
lean_dec_ref(v_start_116_);
if (v___x_123_ == 2)
{
lean_del_object(v___x_121_);
lean_dec_ref(v_start_118_);
lean_dec_ref(v_end_117_);
lean_dec_ref(v___x_114_);
v_a_98_ = v_b_63_;
goto v___jp_97_;
}
else
{
uint8_t v___x_124_; 
v___x_124_ = l_Lean_Lsp_instOrdPosition_ord(v_start_118_, v_end_117_);
lean_dec_ref(v_end_117_);
lean_dec_ref(v_start_118_);
if (v___x_124_ == 2)
{
lean_del_object(v___x_121_);
lean_dec_ref(v___x_114_);
v_a_98_ = v_b_63_;
goto v___jp_97_;
}
else
{
lean_object* v___y_126_; lean_object* v___y_134_; lean_object* v_info_135_; lean_object* v_kind_136_; lean_object* v_args_137_; lean_object* v___y_152_; lean_object* v___y_160_; lean_object* v_info_161_; lean_object* v_kind_162_; lean_object* v_args_163_; lean_object* v___x_175_; uint8_t v___x_176_; 
v___x_175_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__10));
lean_inc(v_stx_62_);
v___x_176_ = l_Lean_Syntax_isOfKind(v_stx_62_, v___x_175_);
if (v___x_176_ == 0)
{
lean_object* v___x_177_; uint8_t v___x_178_; 
lean_del_object(v___x_121_);
v___x_177_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__12));
lean_inc(v_stx_62_);
v___x_178_ = l_Lean_Syntax_isOfKind(v_stx_62_, v___x_177_);
if (v___x_178_ == 0)
{
lean_dec_ref(v___x_114_);
v_a_98_ = v_b_63_;
goto v___jp_97_;
}
else
{
if (lean_obj_tag(v_stx_62_) == 1)
{
lean_object* v_info_179_; lean_object* v_kind_180_; lean_object* v_args_181_; lean_object* v___x_182_; lean_object* v___x_183_; uint8_t v___x_184_; 
v_info_179_ = lean_ctor_get(v_stx_62_, 0);
v_kind_180_ = lean_ctor_get(v_stx_62_, 1);
v_args_181_ = lean_ctor_get(v_stx_62_, 2);
v___x_182_ = lean_unsigned_to_nat(0u);
v___x_183_ = lean_array_get_size(v_args_181_);
v___x_184_ = lean_nat_dec_lt(v___x_182_, v___x_183_);
if (v___x_184_ == 0)
{
lean_inc_ref(v_args_181_);
lean_inc(v_kind_180_);
lean_inc(v_info_179_);
lean_inc_ref(v_stx_62_);
v___y_160_ = v_stx_62_;
v_info_161_ = v_info_179_;
v_kind_162_ = v_kind_180_;
v_args_163_ = v_args_181_;
goto v___jp_159_;
}
else
{
lean_object* v_v_185_; lean_object* v___x_186_; lean_object* v_xs_x27_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; 
v_v_185_ = lean_array_fget_borrowed(v_args_181_, v___x_182_);
v___x_186_ = lean_box(0);
lean_inc_ref(v_args_181_);
v_xs_x27_187_ = lean_array_fset(v_args_181_, v___x_182_, v___x_186_);
v___x_188_ = l_Lean_Syntax_getHeadInfo(v_v_185_);
v___x_189_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__13));
v___x_190_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_190_, 0, v___x_188_);
lean_ctor_set(v___x_190_, 1, v___x_189_);
v___x_191_ = lean_array_fset(v_xs_x27_187_, v___x_182_, v___x_190_);
lean_inc_ref(v___x_191_);
lean_inc_n(v_kind_180_, 2);
lean_inc_n(v_info_179_, 2);
v___x_192_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_192_, 0, v_info_179_);
lean_ctor_set(v___x_192_, 1, v_kind_180_);
lean_ctor_set(v___x_192_, 2, v___x_191_);
v___y_160_ = v___x_192_;
v_info_161_ = v_info_179_;
v_kind_162_ = v_kind_180_;
v_args_163_ = v___x_191_;
goto v___jp_159_;
}
}
else
{
if (lean_obj_tag(v_stx_62_) == 1)
{
lean_object* v_info_193_; lean_object* v_kind_194_; lean_object* v_args_195_; 
v_info_193_ = lean_ctor_get(v_stx_62_, 0);
v_kind_194_ = lean_ctor_get(v_stx_62_, 1);
v_args_195_ = lean_ctor_get(v_stx_62_, 2);
lean_inc_ref(v_args_195_);
lean_inc(v_kind_194_);
lean_inc(v_info_193_);
lean_inc_ref(v_stx_62_);
v___y_160_ = v_stx_62_;
v_info_161_ = v_info_193_;
v_kind_162_ = v_kind_194_;
v_args_163_ = v_args_195_;
goto v___jp_159_;
}
else
{
lean_inc(v_stx_62_);
v___y_152_ = v_stx_62_;
goto v___jp_151_;
}
}
}
}
else
{
lean_object* v___x_196_; lean_object* v___x_197_; uint8_t v___x_198_; 
v___x_196_ = lean_unsigned_to_nat(3u);
v___x_197_ = l_Lean_Syntax_getArg(v_stx_62_, v___x_196_);
v___x_198_ = l_Lean_Syntax_isNone(v___x_197_);
lean_dec(v___x_197_);
if (v___x_198_ == 0)
{
lean_del_object(v___x_121_);
lean_dec_ref(v___x_114_);
v_a_98_ = v_b_63_;
goto v___jp_97_;
}
else
{
if (lean_obj_tag(v_stx_62_) == 1)
{
lean_object* v_info_199_; lean_object* v_kind_200_; lean_object* v_args_201_; lean_object* v___x_202_; lean_object* v___x_203_; uint8_t v___x_204_; 
v_info_199_ = lean_ctor_get(v_stx_62_, 0);
v_kind_200_ = lean_ctor_get(v_stx_62_, 1);
v_args_201_ = lean_ctor_get(v_stx_62_, 2);
v___x_202_ = lean_unsigned_to_nat(0u);
v___x_203_ = lean_array_get_size(v_args_201_);
v___x_204_ = lean_nat_dec_lt(v___x_202_, v___x_203_);
if (v___x_204_ == 0)
{
lean_inc_ref(v_args_201_);
lean_inc(v_kind_200_);
lean_inc(v_info_199_);
lean_inc_ref(v_stx_62_);
v___y_134_ = v_stx_62_;
v_info_135_ = v_info_199_;
v_kind_136_ = v_kind_200_;
v_args_137_ = v_args_201_;
goto v___jp_133_;
}
else
{
lean_object* v_v_205_; lean_object* v___x_206_; lean_object* v_xs_x27_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v_v_205_ = lean_array_fget_borrowed(v_args_201_, v___x_202_);
v___x_206_ = lean_box(0);
lean_inc_ref(v_args_201_);
v_xs_x27_207_ = lean_array_fset(v_args_201_, v___x_202_, v___x_206_);
v___x_208_ = l_Lean_Syntax_getHeadInfo(v_v_205_);
v___x_209_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__14));
v___x_210_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_210_, 0, v___x_208_);
lean_ctor_set(v___x_210_, 1, v___x_209_);
v___x_211_ = lean_array_fset(v_xs_x27_207_, v___x_202_, v___x_210_);
lean_inc_ref(v___x_211_);
lean_inc_n(v_kind_200_, 2);
lean_inc_n(v_info_199_, 2);
v___x_212_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_212_, 0, v_info_199_);
lean_ctor_set(v___x_212_, 1, v_kind_200_);
lean_ctor_set(v___x_212_, 2, v___x_211_);
v___y_134_ = v___x_212_;
v_info_135_ = v_info_199_;
v_kind_136_ = v_kind_200_;
v_args_137_ = v___x_211_;
goto v___jp_133_;
}
}
else
{
if (lean_obj_tag(v_stx_62_) == 1)
{
lean_object* v_info_213_; lean_object* v_kind_214_; lean_object* v_args_215_; 
v_info_213_ = lean_ctor_get(v_stx_62_, 0);
v_kind_214_ = lean_ctor_get(v_stx_62_, 1);
v_args_215_ = lean_ctor_get(v_stx_62_, 2);
lean_inc_ref(v_args_215_);
lean_inc(v_kind_214_);
lean_inc(v_info_213_);
lean_inc_ref(v_stx_62_);
v___y_134_ = v_stx_62_;
v_info_135_ = v_info_213_;
v_kind_136_ = v_kind_214_;
v_args_137_ = v_args_215_;
goto v___jp_133_;
}
else
{
lean_del_object(v___x_121_);
lean_inc(v_stx_62_);
v___y_126_ = v_stx_62_;
goto v___jp_125_;
}
}
}
}
v___jp_125_:
{
lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_127_ = l_Lean_Syntax_unsetTrailing(v___y_126_);
v___x_128_ = l_Lean_Syntax_reprint(v___x_127_);
if (lean_obj_tag(v___x_128_) == 1)
{
lean_object* v_val_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v_val_129_ = lean_ctor_get(v___x_128_, 0);
lean_inc(v_val_129_);
lean_dec_ref_known(v___x_128_, 1);
v___x_130_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__2));
lean_inc_ref(v_a_59_);
v___x_131_ = lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0(v_a_59_, v___x_130_, v___x_114_, v_val_129_);
v___x_132_ = lean_array_push(v_b_63_, v___x_131_);
v_a_98_ = v___x_132_;
goto v___jp_97_;
}
else
{
lean_dec(v___x_128_);
lean_dec_ref(v___x_114_);
v_a_98_ = v_b_63_;
goto v___jp_97_;
}
}
v___jp_133_:
{
lean_object* v___x_138_; lean_object* v___x_139_; uint8_t v___x_140_; 
v___x_138_ = lean_unsigned_to_nat(4u);
v___x_139_ = lean_array_get_size(v_args_137_);
v___x_140_ = lean_nat_dec_lt(v___x_138_, v___x_139_);
if (v___x_140_ == 0)
{
lean_dec_ref(v_args_137_);
lean_dec(v_kind_136_);
lean_dec(v_info_135_);
lean_del_object(v___x_121_);
v___y_126_ = v___y_134_;
goto v___jp_125_;
}
else
{
lean_object* v_v_141_; lean_object* v___x_142_; lean_object* v_xs_x27_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_147_; 
lean_dec(v___y_134_);
v_v_141_ = lean_array_fget(v_args_137_, v___x_138_);
v___x_142_ = lean_box(0);
v_xs_x27_143_ = lean_array_fset(v_args_137_, v___x_138_, v___x_142_);
v___x_144_ = l_Lean_Syntax_getHeadInfo(v_v_141_);
lean_dec(v_v_141_);
v___x_145_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__3));
if (v_isShared_122_ == 0)
{
lean_ctor_set_tag(v___x_121_, 2);
lean_ctor_set(v___x_121_, 1, v___x_145_);
lean_ctor_set(v___x_121_, 0, v___x_144_);
v___x_147_ = v___x_121_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_150_; 
v_reuseFailAlloc_150_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_150_, 0, v___x_144_);
lean_ctor_set(v_reuseFailAlloc_150_, 1, v___x_145_);
v___x_147_ = v_reuseFailAlloc_150_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_148_ = lean_array_fset(v_xs_x27_143_, v___x_138_, v___x_147_);
v___x_149_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_149_, 0, v_info_135_);
lean_ctor_set(v___x_149_, 1, v_kind_136_);
lean_ctor_set(v___x_149_, 2, v___x_148_);
v___y_126_ = v___x_149_;
goto v___jp_125_;
}
}
}
v___jp_151_:
{
lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_153_ = l_Lean_Syntax_unsetTrailing(v___y_152_);
v___x_154_ = l_Lean_Syntax_reprint(v___x_153_);
if (lean_obj_tag(v___x_154_) == 1)
{
lean_object* v_val_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v_val_155_ = lean_ctor_get(v___x_154_, 0);
lean_inc(v_val_155_);
lean_dec_ref_known(v___x_154_, 1);
v___x_156_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__4));
lean_inc_ref(v_a_59_);
v___x_157_ = lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___lam__0(v_a_59_, v___x_156_, v___x_114_, v_val_155_);
v___x_158_ = lean_array_push(v_b_63_, v___x_157_);
v_a_98_ = v___x_158_;
goto v___jp_97_;
}
else
{
lean_dec(v___x_154_);
lean_dec_ref(v___x_114_);
v_a_98_ = v_b_63_;
goto v___jp_97_;
}
}
v___jp_159_:
{
lean_object* v___x_164_; lean_object* v___x_165_; uint8_t v___x_166_; 
v___x_164_ = lean_unsigned_to_nat(3u);
v___x_165_ = lean_array_get_size(v_args_163_);
v___x_166_ = lean_nat_dec_lt(v___x_164_, v___x_165_);
if (v___x_166_ == 0)
{
lean_dec_ref(v_args_163_);
lean_dec(v_kind_162_);
lean_dec(v_info_161_);
v___y_152_ = v___y_160_;
goto v___jp_151_;
}
else
{
lean_object* v_v_167_; lean_object* v___x_168_; lean_object* v_xs_x27_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
lean_dec(v___y_160_);
v_v_167_ = lean_array_fget(v_args_163_, v___x_164_);
v___x_168_ = lean_box(0);
v_xs_x27_169_ = lean_array_fset(v_args_163_, v___x_164_, v___x_168_);
v___x_170_ = l_Lean_Syntax_getHeadInfo(v_v_167_);
lean_dec(v_v_167_);
v___x_171_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__5));
v___x_172_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_170_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
v___x_173_ = lean_array_fset(v_xs_x27_169_, v___x_164_, v___x_172_);
v___x_174_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_174_, 0, v_info_161_);
lean_ctor_set(v___x_174_, 1, v_kind_162_);
lean_ctor_set(v___x_174_, 2, v___x_173_);
v___y_152_ = v___x_174_;
goto v___jp_151_;
}
}
}
}
}
}
else
{
lean_dec(v___x_109_);
v_a_98_ = v_b_63_;
goto v___jp_97_;
}
v___jp_66_:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_68_, 0, v_b_67_);
v___x_69_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
return v___x_69_;
}
v___jp_70_:
{
lean_object* v___x_73_; lean_object* v___x_74_; size_t v_sz_75_; size_t v___x_76_; lean_object* v___x_77_; 
v___x_73_ = lean_box(0);
v___x_74_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v___y_72_);
v_sz_75_ = lean_array_size(v___y_71_);
v___x_76_ = ((size_t)0ULL);
v___x_77_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1_spec__1(v_a_59_, v_params_60_, v_firstChoiceOnly_61_, v___y_71_, v_sz_75_, v___x_76_, v___x_74_, v___y_64_);
lean_dec_ref(v___y_71_);
if (lean_obj_tag(v___x_77_) == 0)
{
lean_object* v_a_78_; lean_object* v___x_80_; uint8_t v_isShared_81_; uint8_t v_isSharedCheck_88_; 
v_a_78_ = lean_ctor_get(v___x_77_, 0);
v_isSharedCheck_88_ = !lean_is_exclusive(v___x_77_);
if (v_isSharedCheck_88_ == 0)
{
v___x_80_ = v___x_77_;
v_isShared_81_ = v_isSharedCheck_88_;
goto v_resetjp_79_;
}
else
{
lean_inc(v_a_78_);
lean_dec(v___x_77_);
v___x_80_ = lean_box(0);
v_isShared_81_ = v_isSharedCheck_88_;
goto v_resetjp_79_;
}
v_resetjp_79_:
{
lean_object* v_fst_82_; 
v_fst_82_ = lean_ctor_get(v_a_78_, 0);
if (lean_obj_tag(v_fst_82_) == 0)
{
lean_object* v_snd_83_; 
lean_del_object(v___x_80_);
v_snd_83_ = lean_ctor_get(v_a_78_, 1);
lean_inc(v_snd_83_);
lean_dec(v_a_78_);
v_b_67_ = v_snd_83_;
goto v___jp_66_;
}
else
{
lean_object* v_val_84_; lean_object* v___x_86_; 
lean_inc_ref(v_fst_82_);
lean_dec(v_a_78_);
v_val_84_ = lean_ctor_get(v_fst_82_, 0);
lean_inc(v_val_84_);
lean_dec_ref_known(v_fst_82_, 1);
if (v_isShared_81_ == 0)
{
lean_ctor_set(v___x_80_, 0, v_val_84_);
v___x_86_ = v___x_80_;
goto v_reusejp_85_;
}
else
{
lean_object* v_reuseFailAlloc_87_; 
v_reuseFailAlloc_87_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_87_, 0, v_val_84_);
v___x_86_ = v_reuseFailAlloc_87_;
goto v_reusejp_85_;
}
v_reusejp_85_:
{
return v___x_86_;
}
}
}
}
else
{
lean_object* v_a_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_96_; 
v_a_89_ = lean_ctor_get(v___x_77_, 0);
v_isSharedCheck_96_ = !lean_is_exclusive(v___x_77_);
if (v_isSharedCheck_96_ == 0)
{
v___x_91_ = v___x_77_;
v_isShared_92_ = v_isSharedCheck_96_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_a_89_);
lean_dec(v___x_77_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_96_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_94_; 
if (v_isShared_92_ == 0)
{
v___x_94_ = v___x_91_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v_a_89_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
}
}
v___jp_97_:
{
if (lean_obj_tag(v_stx_62_) == 1)
{
if (v_firstChoiceOnly_61_ == 0)
{
lean_object* v_args_99_; 
v_args_99_ = lean_ctor_get(v_stx_62_, 2);
lean_inc_ref(v_args_99_);
lean_dec_ref_known(v_stx_62_, 3);
v___y_71_ = v_args_99_;
v___y_72_ = v_a_98_;
goto v___jp_70_;
}
else
{
lean_object* v_kind_100_; lean_object* v_args_101_; lean_object* v___x_102_; uint8_t v___x_103_; 
v_kind_100_ = lean_ctor_get(v_stx_62_, 1);
lean_inc(v_kind_100_);
v_args_101_ = lean_ctor_get(v_stx_62_, 2);
lean_inc_ref(v_args_101_);
lean_dec_ref_known(v_stx_62_, 3);
v___x_102_ = ((lean_object*)(lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___closed__1));
v___x_103_ = lean_name_eq(v_kind_100_, v___x_102_);
lean_dec(v_kind_100_);
if (v___x_103_ == 0)
{
v___y_71_ = v_args_101_;
v___y_72_ = v_a_98_;
goto v___jp_70_;
}
else
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_104_ = lean_box(0);
v___x_105_ = lean_unsigned_to_nat(0u);
v___x_106_ = lean_array_get(v___x_104_, v_args_101_, v___x_105_);
lean_dec_ref(v_args_101_);
v_stx_62_ = v___x_106_;
v_b_63_ = v_a_98_;
goto _start;
}
}
}
else
{
lean_dec(v_stx_62_);
lean_dec_ref(v_params_60_);
lean_dec_ref(v_a_59_);
v_b_67_ = v_a_98_;
goto v___jp_66_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1_spec__1(lean_object* v_a_217_, lean_object* v_params_218_, uint8_t v_firstChoiceOnly_219_, lean_object* v_as_220_, size_t v_sz_221_, size_t v_i_222_, lean_object* v_b_223_, lean_object* v___y_224_){
_start:
{
uint8_t v___x_226_; 
v___x_226_ = lean_usize_dec_lt(v_i_222_, v_sz_221_);
if (v___x_226_ == 0)
{
lean_object* v___x_227_; 
lean_dec_ref(v_params_218_);
lean_dec_ref(v_a_217_);
v___x_227_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_227_, 0, v_b_223_);
return v___x_227_;
}
else
{
lean_object* v_snd_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_262_; 
v_snd_228_ = lean_ctor_get(v_b_223_, 1);
v_isSharedCheck_262_ = !lean_is_exclusive(v_b_223_);
if (v_isSharedCheck_262_ == 0)
{
lean_object* v_unused_263_; 
v_unused_263_ = lean_ctor_get(v_b_223_, 0);
lean_dec(v_unused_263_);
v___x_230_ = v_b_223_;
v_isShared_231_ = v_isSharedCheck_262_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_snd_228_);
lean_dec(v_b_223_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_262_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v_a_232_; lean_object* v___x_233_; 
v_a_232_ = lean_array_uget_borrowed(v_as_220_, v_i_222_);
lean_inc(v_snd_228_);
lean_inc(v_a_232_);
lean_inc_ref(v_params_218_);
lean_inc_ref(v_a_217_);
v___x_233_ = lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1(v_a_217_, v_params_218_, v_firstChoiceOnly_219_, v_a_232_, v_snd_228_, v___y_224_);
if (lean_obj_tag(v___x_233_) == 0)
{
lean_object* v_a_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_253_; 
v_a_234_ = lean_ctor_get(v___x_233_, 0);
v_isSharedCheck_253_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_253_ == 0)
{
v___x_236_ = v___x_233_;
v_isShared_237_ = v_isSharedCheck_253_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_a_234_);
lean_dec(v___x_233_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_253_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
if (lean_obj_tag(v_a_234_) == 0)
{
lean_object* v___x_238_; lean_object* v___x_240_; 
lean_dec_ref(v_params_218_);
lean_dec_ref(v_a_217_);
v___x_238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_238_, 0, v_a_234_);
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 0, v___x_238_);
v___x_240_ = v___x_230_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v___x_238_);
lean_ctor_set(v_reuseFailAlloc_244_, 1, v_snd_228_);
v___x_240_ = v_reuseFailAlloc_244_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
lean_object* v___x_242_; 
if (v_isShared_237_ == 0)
{
lean_ctor_set(v___x_236_, 0, v___x_240_);
v___x_242_ = v___x_236_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v___x_240_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
}
else
{
lean_object* v_a_245_; lean_object* v___x_246_; lean_object* v___x_248_; 
lean_del_object(v___x_236_);
lean_dec(v_snd_228_);
v_a_245_ = lean_ctor_get(v_a_234_, 0);
lean_inc(v_a_245_);
lean_dec_ref_known(v_a_234_, 1);
v___x_246_ = lean_box(0);
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 1, v_a_245_);
lean_ctor_set(v___x_230_, 0, v___x_246_);
v___x_248_ = v___x_230_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_252_; 
v_reuseFailAlloc_252_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_252_, 0, v___x_246_);
lean_ctor_set(v_reuseFailAlloc_252_, 1, v_a_245_);
v___x_248_ = v_reuseFailAlloc_252_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
size_t v___x_249_; size_t v___x_250_; 
v___x_249_ = ((size_t)1ULL);
v___x_250_ = lean_usize_add(v_i_222_, v___x_249_);
v_i_222_ = v___x_250_;
v_b_223_ = v___x_248_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_254_; lean_object* v___x_256_; uint8_t v_isShared_257_; uint8_t v_isSharedCheck_261_; 
lean_del_object(v___x_230_);
lean_dec(v_snd_228_);
lean_dec_ref(v_params_218_);
lean_dec_ref(v_a_217_);
v_a_254_ = lean_ctor_get(v___x_233_, 0);
v_isSharedCheck_261_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_261_ == 0)
{
v___x_256_ = v___x_233_;
v_isShared_257_ = v_isSharedCheck_261_;
goto v_resetjp_255_;
}
else
{
lean_inc(v_a_254_);
lean_dec(v___x_233_);
v___x_256_ = lean_box(0);
v_isShared_257_ = v_isSharedCheck_261_;
goto v_resetjp_255_;
}
v_resetjp_255_:
{
lean_object* v___x_259_; 
if (v_isShared_257_ == 0)
{
v___x_259_ = v___x_256_;
goto v_reusejp_258_;
}
else
{
lean_object* v_reuseFailAlloc_260_; 
v_reuseFailAlloc_260_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_260_, 0, v_a_254_);
v___x_259_ = v_reuseFailAlloc_260_;
goto v_reusejp_258_;
}
v_reusejp_258_:
{
return v___x_259_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1_spec__1___boxed(lean_object* v_a_264_, lean_object* v_params_265_, lean_object* v_firstChoiceOnly_266_, lean_object* v_as_267_, lean_object* v_sz_268_, lean_object* v_i_269_, lean_object* v_b_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
uint8_t v_firstChoiceOnly_boxed_273_; size_t v_sz_boxed_274_; size_t v_i_boxed_275_; lean_object* v_res_276_; 
v_firstChoiceOnly_boxed_273_ = lean_unbox(v_firstChoiceOnly_266_);
v_sz_boxed_274_ = lean_unbox_usize(v_sz_268_);
lean_dec(v_sz_268_);
v_i_boxed_275_ = lean_unbox_usize(v_i_269_);
lean_dec(v_i_269_);
v_res_276_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1_spec__1(v_a_264_, v_params_265_, v_firstChoiceOnly_boxed_273_, v_as_267_, v_sz_boxed_274_, v_i_boxed_275_, v_b_270_, v___y_271_);
lean_dec_ref(v___y_271_);
lean_dec_ref(v_as_267_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1___boxed(lean_object* v_a_277_, lean_object* v_params_278_, lean_object* v_firstChoiceOnly_279_, lean_object* v_stx_280_, lean_object* v_b_281_, lean_object* v___y_282_, lean_object* v___y_283_){
_start:
{
uint8_t v_firstChoiceOnly_boxed_284_; lean_object* v_res_285_; 
v_firstChoiceOnly_boxed_284_ = lean_unbox(v_firstChoiceOnly_279_);
v_res_285_ = lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1(v_a_277_, v_params_278_, v_firstChoiceOnly_boxed_284_, v_stx_280_, v_b_281_, v___y_282_);
lean_dec_ref(v___y_282_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CodeAction_binderPlicity(lean_object* v_params_288_, lean_object* v_snap_289_, lean_object* v_a_290_){
_start:
{
lean_object* v___x_292_; lean_object* v_a_293_; lean_object* v_stx_294_; uint8_t v___x_295_; lean_object* v___x_296_; uint8_t v_firstChoiceOnly_297_; lean_object* v_stx_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_292_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_CodeAction_binderPlicity_spec__0(v_a_290_);
v_a_293_ = lean_ctor_get(v___x_292_, 0);
lean_inc(v_a_293_);
lean_dec_ref(v___x_292_);
v_stx_294_ = lean_ctor_get(v_snap_289_, 0);
lean_inc(v_stx_294_);
lean_dec_ref(v_snap_289_);
v___x_295_ = 0;
v___x_296_ = l_Lean_Syntax_topDown(v_stx_294_, v___x_295_);
v_firstChoiceOnly_297_ = lean_ctor_get_uint8(v___x_296_, sizeof(void*)*1);
v_stx_298_ = lean_ctor_get(v___x_296_, 0);
lean_inc(v_stx_298_);
lean_dec_ref(v___x_296_);
v___x_299_ = ((lean_object*)(lp_mathlib_Mathlib_CodeAction_binderPlicity___closed__0));
v___x_300_ = lp_mathlib_Lean_Syntax_instForInTopDownOfMonad_loop___at___00Mathlib_CodeAction_binderPlicity_spec__1(v_a_293_, v_params_288_, v_firstChoiceOnly_297_, v_stx_298_, v___x_299_, v_a_290_);
if (lean_obj_tag(v___x_300_) == 0)
{
lean_object* v_a_301_; lean_object* v___x_303_; uint8_t v_isShared_304_; uint8_t v_isSharedCheck_309_; 
v_a_301_ = lean_ctor_get(v___x_300_, 0);
v_isSharedCheck_309_ = !lean_is_exclusive(v___x_300_);
if (v_isSharedCheck_309_ == 0)
{
v___x_303_ = v___x_300_;
v_isShared_304_ = v_isSharedCheck_309_;
goto v_resetjp_302_;
}
else
{
lean_inc(v_a_301_);
lean_dec(v___x_300_);
v___x_303_ = lean_box(0);
v_isShared_304_ = v_isSharedCheck_309_;
goto v_resetjp_302_;
}
v_resetjp_302_:
{
lean_object* v_a_305_; lean_object* v___x_307_; 
v_a_305_ = lean_ctor_get(v_a_301_, 0);
lean_inc(v_a_305_);
lean_dec(v_a_301_);
if (v_isShared_304_ == 0)
{
lean_ctor_set(v___x_303_, 0, v_a_305_);
v___x_307_ = v___x_303_;
goto v_reusejp_306_;
}
else
{
lean_object* v_reuseFailAlloc_308_; 
v_reuseFailAlloc_308_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_308_, 0, v_a_305_);
v___x_307_ = v_reuseFailAlloc_308_;
goto v_reusejp_306_;
}
v_reusejp_306_:
{
return v___x_307_;
}
}
}
else
{
lean_object* v_a_310_; lean_object* v___x_312_; uint8_t v_isShared_313_; uint8_t v_isSharedCheck_317_; 
v_a_310_ = lean_ctor_get(v___x_300_, 0);
v_isSharedCheck_317_ = !lean_is_exclusive(v___x_300_);
if (v_isSharedCheck_317_ == 0)
{
v___x_312_ = v___x_300_;
v_isShared_313_ = v_isSharedCheck_317_;
goto v_resetjp_311_;
}
else
{
lean_inc(v_a_310_);
lean_dec(v___x_300_);
v___x_312_ = lean_box(0);
v_isShared_313_ = v_isSharedCheck_317_;
goto v_resetjp_311_;
}
v_resetjp_311_:
{
lean_object* v___x_315_; 
if (v_isShared_313_ == 0)
{
v___x_315_ = v___x_312_;
goto v_reusejp_314_;
}
else
{
lean_object* v_reuseFailAlloc_316_; 
v_reuseFailAlloc_316_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_316_, 0, v_a_310_);
v___x_315_ = v_reuseFailAlloc_316_;
goto v_reusejp_314_;
}
v_reusejp_314_:
{
return v___x_315_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CodeAction_binderPlicity___boxed(lean_object* v_params_318_, lean_object* v_snap_319_, lean_object* v_a_320_, lean_object* v_a_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib_Mathlib_CodeAction_binderPlicity(v_params_318_, v_snap_319_, v_a_320_);
lean_dec_ref(v_a_320_);
return v_res_322_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_CodeActions_BinderPlicity(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_CodeActions_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_CodeActions_BinderPlicity(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_CodeActions_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Server_CodeActions_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_CodeActions_BinderPlicity(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_CodeActions_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_CodeActions_BinderPlicity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_CodeActions_BinderPlicity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_CodeActions_BinderPlicity(builtin);
}
#ifdef __cplusplus
}
#endif
