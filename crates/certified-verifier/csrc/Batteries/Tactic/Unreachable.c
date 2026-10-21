// Lean compiler output
// Module: Batteries.Tactic.Unreachable
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Basic
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
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Tactic_instInhabitedTacticM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_unreachable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries_Batteries_Tactic_unreachable___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_unreachable___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries_Batteries_Tactic_unreachable___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_unreachable___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "unreachable"};
static const lean_object* lp_batteries_Batteries_Tactic_unreachable___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_unreachable___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_unreachable___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_unreachable___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__2_value),LEAN_SCALAR_PTR_LITERAL(138, 69, 16, 178, 93, 143, 143, 50)}};
static const lean_object* lp_batteries_Batteries_Tactic_unreachable___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_unreachable___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "unreachable!"};
static const lean_object* lp_batteries_Batteries_Tactic_unreachable___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_unreachable___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_unreachable___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_unreachable___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__3_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_unreachable___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__6_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_unreachable = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__6_value;
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_panic___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Tactic_instInhabitedTacticM___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_panic___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__1___closed__0 = (const lean_object*)&lp_batteries_panic___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Batteries.Tactic.Unreachable"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 94, .m_capacity = 94, .m_length = 93, .m_data = "Batteries.Tactic._aux_Batteries_Tactic_Unreachable___elabRules_Batteries_Tactic_unreachable_1"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "unreachable tactic has been reached"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__3;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__4;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_unreachableConv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "unreachableConv"};
static const lean_object* lp_batteries_Batteries_Tactic_unreachableConv___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachableConv___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_unreachableConv___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_unreachableConv___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_unreachableConv___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_unreachableConv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_unreachableConv___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_unreachableConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(180, 51, 125, 100, 108, 230, 32, 33)}};
static const lean_object* lp_batteries_Batteries_Tactic_unreachableConv___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachableConv___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_unreachableConv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_unreachableConv___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_unreachableConv___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachableConv___closed__2_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_unreachableConv = (const lean_object*)&lp_batteries_Batteries_Tactic_unreachableConv___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "nestedTacticCore"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4_value_aux_3),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(169, 23, 62, 30, 134, 160, 158, 203)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "tactic'"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__6_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__8_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__8_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__8_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__10_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__10_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_unreachable___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__10_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__10_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__12_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_17_ = lean_box(0);
v___x_18_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_19_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_19_, 0, v___x_18_);
lean_ctor_set(v___x_19_, 1, v___x_17_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg(){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg___closed__0);
v___x_22_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg___boxed(lean_object* v___y_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg();
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0(lean_object* v_00_u03b1_25_, lean_object* v___y_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg();
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___boxed(lean_object* v_00_u03b1_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0(v_00_u03b1_36_, v___y_37_, v___y_38_, v___y_39_, v___y_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
lean_dec(v___y_40_);
lean_dec_ref(v___y_39_);
lean_dec(v___y_38_);
lean_dec_ref(v___y_37_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__1(lean_object* v_msg_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_){
_start:
{
lean_object* v___f_58_; lean_object* v___x_618__overap_59_; lean_object* v___x_60_; 
v___f_58_ = ((lean_object*)(lp_batteries_panic___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__1___closed__0));
v___x_618__overap_59_ = lean_panic_fn_borrowed(v___f_58_, v_msg_48_);
lean_inc(v___y_56_);
lean_inc_ref(v___y_55_);
lean_inc(v___y_54_);
lean_inc_ref(v___y_53_);
lean_inc(v___y_52_);
lean_inc_ref(v___y_51_);
lean_inc(v___y_50_);
lean_inc_ref(v___y_49_);
v___x_60_ = lean_apply_9(v___x_618__overap_59_, v___y_49_, v___y_50_, v___y_51_, v___y_52_, v___y_53_, v___y_54_, v___y_55_, v___y_56_, lean_box(0));
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__1___boxed(lean_object* v_msg_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_batteries_panic___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__1(v_msg_61_, v___y_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_);
lean_dec(v___y_69_);
lean_dec_ref(v___y_68_);
lean_dec(v___y_67_);
lean_dec_ref(v___y_66_);
lean_dec(v___y_65_);
lean_dec_ref(v___y_64_);
lean_dec(v___y_63_);
lean_dec_ref(v___y_62_);
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2_spec__2(lean_object* v_msgData_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_){
_start:
{
lean_object* v___x_78_; lean_object* v_env_79_; lean_object* v___x_80_; lean_object* v_mctx_81_; lean_object* v_lctx_82_; lean_object* v_options_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_78_ = lean_st_ref_get(v___y_76_);
v_env_79_ = lean_ctor_get(v___x_78_, 0);
lean_inc_ref(v_env_79_);
lean_dec(v___x_78_);
v___x_80_ = lean_st_ref_get(v___y_74_);
v_mctx_81_ = lean_ctor_get(v___x_80_, 0);
lean_inc_ref(v_mctx_81_);
lean_dec(v___x_80_);
v_lctx_82_ = lean_ctor_get(v___y_73_, 2);
v_options_83_ = lean_ctor_get(v___y_75_, 2);
lean_inc_ref(v_options_83_);
lean_inc_ref(v_lctx_82_);
v___x_84_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_84_, 0, v_env_79_);
lean_ctor_set(v___x_84_, 1, v_mctx_81_);
lean_ctor_set(v___x_84_, 2, v_lctx_82_);
lean_ctor_set(v___x_84_, 3, v_options_83_);
v___x_85_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set(v___x_85_, 1, v_msgData_72_);
v___x_86_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_86_, 0, v___x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2_spec__2___boxed(lean_object* v_msgData_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2_spec__2(v_msgData_87_, v___y_88_, v___y_89_, v___y_90_, v___y_91_);
lean_dec(v___y_91_);
lean_dec_ref(v___y_90_);
lean_dec(v___y_89_);
lean_dec_ref(v___y_88_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2___redArg(lean_object* v_msg_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_){
_start:
{
lean_object* v_ref_100_; lean_object* v___x_101_; lean_object* v_a_102_; lean_object* v___x_104_; uint8_t v_isShared_105_; uint8_t v_isSharedCheck_110_; 
v_ref_100_ = lean_ctor_get(v___y_97_, 5);
v___x_101_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2_spec__2(v_msg_94_, v___y_95_, v___y_96_, v___y_97_, v___y_98_);
v_a_102_ = lean_ctor_get(v___x_101_, 0);
v_isSharedCheck_110_ = !lean_is_exclusive(v___x_101_);
if (v_isSharedCheck_110_ == 0)
{
v___x_104_ = v___x_101_;
v_isShared_105_ = v_isSharedCheck_110_;
goto v_resetjp_103_;
}
else
{
lean_inc(v_a_102_);
lean_dec(v___x_101_);
v___x_104_ = lean_box(0);
v_isShared_105_ = v_isSharedCheck_110_;
goto v_resetjp_103_;
}
v_resetjp_103_:
{
lean_object* v___x_106_; lean_object* v___x_108_; 
lean_inc(v_ref_100_);
v___x_106_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_106_, 0, v_ref_100_);
lean_ctor_set(v___x_106_, 1, v_a_102_);
if (v_isShared_105_ == 0)
{
lean_ctor_set_tag(v___x_104_, 1);
lean_ctor_set(v___x_104_, 0, v___x_106_);
v___x_108_ = v___x_104_;
goto v_reusejp_107_;
}
else
{
lean_object* v_reuseFailAlloc_109_; 
v_reuseFailAlloc_109_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_109_, 0, v___x_106_);
v___x_108_ = v_reuseFailAlloc_109_;
goto v_reusejp_107_;
}
v_reusejp_107_:
{
return v___x_108_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2___redArg___boxed(lean_object* v_msg_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2___redArg(v_msg_111_, v___y_112_, v___y_113_, v___y_114_, v___y_115_);
lean_dec(v___y_115_);
lean_dec_ref(v___y_114_);
lean_dec(v___y_113_);
lean_dec_ref(v___y_112_);
return v_res_117_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__3(void){
_start:
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_121_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__2));
v___x_122_ = lean_unsigned_to_nat(2u);
v___x_123_ = lean_unsigned_to_nat(27u);
v___x_124_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__1));
v___x_125_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__0));
v___x_126_ = l_mkPanicMessageWithDecl(v___x_125_, v___x_124_, v___x_123_, v___x_122_, v___x_121_);
return v___x_126_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__4(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_127_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__2));
v___x_128_ = l_Lean_stringToMessageData(v___x_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1(lean_object* v_x_129_, lean_object* v_a_130_, lean_object* v_a_131_, lean_object* v_a_132_, lean_object* v_a_133_, lean_object* v_a_134_, lean_object* v_a_135_, lean_object* v_a_136_, lean_object* v_a_137_){
_start:
{
lean_object* v___x_139_; uint8_t v___x_140_; 
v___x_139_ = ((lean_object*)(lp_batteries_Batteries_Tactic_unreachable___closed__3));
v___x_140_ = l_Lean_Syntax_isOfKind(v_x_129_, v___x_139_);
if (v___x_140_ == 0)
{
lean_object* v___x_141_; 
v___x_141_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__0___redArg();
return v___x_141_;
}
else
{
lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_142_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__3, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__3_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__3);
v___x_143_ = lp_batteries_panic___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__1(v___x_142_, v_a_130_, v_a_131_, v_a_132_, v_a_133_, v_a_134_, v_a_135_, v_a_136_, v_a_137_);
if (lean_obj_tag(v___x_143_) == 0)
{
lean_object* v___x_144_; lean_object* v___x_145_; 
lean_dec_ref_known(v___x_143_, 1);
v___x_144_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__4, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__4_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___closed__4);
v___x_145_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2___redArg(v___x_144_, v_a_134_, v_a_135_, v_a_136_, v_a_137_);
return v___x_145_;
}
else
{
return v___x_143_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1___boxed(lean_object* v_x_146_, lean_object* v_a_147_, lean_object* v_a_148_, lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v_a_151_, lean_object* v_a_152_, lean_object* v_a_153_, lean_object* v_a_154_, lean_object* v_a_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1(v_x_146_, v_a_147_, v_a_148_, v_a_149_, v_a_150_, v_a_151_, v_a_152_, v_a_153_, v_a_154_);
lean_dec(v_a_154_);
lean_dec_ref(v_a_153_);
lean_dec(v_a_152_);
lean_dec_ref(v_a_151_);
lean_dec(v_a_150_);
lean_dec_ref(v_a_149_);
lean_dec(v_a_148_);
lean_dec_ref(v_a_147_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2(lean_object* v_00_u03b1_157_, lean_object* v_msg_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2___redArg(v_msg_158_, v___y_163_, v___y_164_, v___y_165_, v___y_166_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2___boxed(lean_object* v_00_u03b1_169_, lean_object* v_msg_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Unreachable______elabRules__Batteries__Tactic__unreachable__1_spec__2(v_00_u03b1_169_, v_msg_170_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
lean_dec(v___y_176_);
lean_dec_ref(v___y_175_);
lean_dec(v___y_174_);
lean_dec_ref(v___y_173_);
lean_dec(v___y_172_);
lean_dec_ref(v___y_171_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1(lean_object* v_x_218_, lean_object* v_a_219_, lean_object* v_a_220_){
_start:
{
lean_object* v___x_221_; uint8_t v___x_222_; 
v___x_221_ = ((lean_object*)(lp_batteries_Batteries_Tactic_unreachableConv___closed__1));
v___x_222_ = l_Lean_Syntax_isOfKind(v_x_218_, v___x_221_);
if (v___x_222_ == 0)
{
lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_223_ = lean_box(1);
v___x_224_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_223_);
lean_ctor_set(v___x_224_, 1, v_a_220_);
return v___x_224_;
}
else
{
lean_object* v_ref_225_; uint8_t v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; 
v_ref_225_ = lean_ctor_get(v_a_219_, 5);
v___x_226_ = 0;
v___x_227_ = l_Lean_SourceInfo_fromRef(v_ref_225_, v___x_226_);
v___x_228_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__4));
v___x_229_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__5));
lean_inc_n(v___x_227_, 7);
v___x_230_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_230_, 0, v___x_227_);
lean_ctor_set(v___x_230_, 1, v___x_229_);
v___x_231_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__6));
v___x_232_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_232_, 0, v___x_227_);
lean_ctor_set(v___x_232_, 1, v___x_231_);
v___x_233_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__8));
v___x_234_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__10));
v___x_235_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___closed__12));
v___x_236_ = ((lean_object*)(lp_batteries_Batteries_Tactic_unreachable___closed__3));
v___x_237_ = ((lean_object*)(lp_batteries_Batteries_Tactic_unreachable___closed__4));
v___x_238_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_227_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
v___x_239_ = l_Lean_Syntax_node1(v___x_227_, v___x_236_, v___x_238_);
v___x_240_ = l_Lean_Syntax_node1(v___x_227_, v___x_235_, v___x_239_);
v___x_241_ = l_Lean_Syntax_node1(v___x_227_, v___x_234_, v___x_240_);
v___x_242_ = l_Lean_Syntax_node1(v___x_227_, v___x_233_, v___x_241_);
v___x_243_ = l_Lean_Syntax_node3(v___x_227_, v___x_228_, v___x_230_, v___x_232_, v___x_242_);
v___x_244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_243_);
lean_ctor_set(v___x_244_, 1, v_a_220_);
return v___x_244_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1___boxed(lean_object* v_x_245_, lean_object* v_a_246_, lean_object* v_a_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Unreachable______macroRules__Batteries__Tactic__unreachableConv__1(v_x_245_, v_a_246_, v_a_247_);
lean_dec_ref(v_a_246_);
return v_res_248_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_Unreachable(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_Unreachable(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_Unreachable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Unreachable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_Unreachable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_Unreachable(builtin);
}
#ifdef __cplusplus
}
#endif
