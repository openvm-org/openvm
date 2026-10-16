// Lean compiler output
// Module: Mathlib.Tactic.UnsetOption
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Parser.Term public meta import Lean.Parser.Do public meta import Lean.Elab.Command
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
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
lean_object* l_Lean_Options_erase(lean_object*, lean_object*);
lean_object* l_Lean_Elab_addCompletionInfo___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Elab_Command_modifyScope___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "unsetOption"};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(177, 181, 244, 12, 1, 14, 170, 235)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__3_value),LEAN_SCALAR_PTR_LITERAL(232, 172, 155, 47, 75, 105, 166, 86)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unset_option "};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__7_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__9_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__10_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__11_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__8_value),((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__11_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__12_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Command_unsetOption___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__12_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__13_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Command_unsetOption = (const lean_object*)&lp_mathlib_Lean_Elab_Command_unsetOption___closed__13_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___redArg___lam__0(lean_object* v_optionName_1_, lean_object* v_toPure_2_, lean_object* v_____do__lift_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = l_Lean_Options_erase(v_____do__lift_3_, v_optionName_1_);
v___x_5_ = lean_apply_2(v_toPure_2_, lean_box(0), v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___redArg___lam__0___boxed(lean_object* v_optionName_6_, lean_object* v_toPure_7_, lean_object* v_____do__lift_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___redArg___lam__0(v_optionName_6_, v_toPure_7_, v_____do__lift_8_);
lean_dec(v_optionName_6_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___redArg(lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_optionName_12_){
_start:
{
lean_object* v_toApplicative_13_; lean_object* v_toBind_14_; lean_object* v_toPure_15_; lean_object* v___f_16_; lean_object* v___x_17_; 
v_toApplicative_13_ = lean_ctor_get(v_inst_10_, 0);
lean_inc_ref(v_toApplicative_13_);
v_toBind_14_ = lean_ctor_get(v_inst_10_, 1);
lean_inc(v_toBind_14_);
lean_dec_ref(v_inst_10_);
v_toPure_15_ = lean_ctor_get(v_toApplicative_13_, 1);
lean_inc(v_toPure_15_);
lean_dec_ref(v_toApplicative_13_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_16_, 0, v_optionName_12_);
lean_closure_set(v___f_16_, 1, v_toPure_15_);
v___x_17_ = lean_apply_4(v_toBind_14_, lean_box(0), lean_box(0), v_inst_11_, v___f_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption(lean_object* v_m_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_optionName_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___redArg(v_inst_19_, v_inst_20_, v_optionName_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___redArg___lam__0(lean_object* v_id_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_____r_26_){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_27_ = l_Lean_Syntax_getId(v_id_23_);
v___x_28_ = l_Lean_Name_eraseMacroScopes(v___x_27_);
lean_dec(v___x_27_);
v___x_29_ = lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___redArg(v_inst_24_, v_inst_25_, v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___redArg___lam__0___boxed(lean_object* v_id_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_____r_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_Lean_Elab_elabUnsetOption___redArg___lam__0(v_id_30_, v_inst_31_, v_inst_32_, v_____r_33_);
lean_dec(v_id_30_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___redArg___lam__1(lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_toBind_37_, lean_object* v___f_38_, lean_object* v_____do__lift_39_){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_40_ = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(v___x_40_, 0, v_____do__lift_39_);
v___x_41_ = l_Lean_Elab_addCompletionInfo___redArg(v_inst_35_, v_inst_36_, v___x_40_);
v___x_42_ = lean_apply_4(v_toBind_37_, lean_box(0), lean_box(0), v___x_41_, v___f_38_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___redArg(lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_id_47_){
_start:
{
lean_object* v_toBind_48_; lean_object* v_getRef_49_; lean_object* v___f_50_; lean_object* v___f_51_; lean_object* v___x_52_; 
v_toBind_48_ = lean_ctor_get(v_inst_43_, 1);
lean_inc_n(v_toBind_48_, 2);
v_getRef_49_ = lean_ctor_get(v_inst_45_, 0);
lean_inc(v_getRef_49_);
lean_dec_ref(v_inst_45_);
lean_inc_ref(v_inst_43_);
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_elabUnsetOption___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_50_, 0, v_id_47_);
lean_closure_set(v___f_50_, 1, v_inst_43_);
lean_closure_set(v___f_50_, 2, v_inst_44_);
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_elabUnsetOption___redArg___lam__1), 5, 4);
lean_closure_set(v___f_51_, 0, v_inst_43_);
lean_closure_set(v___f_51_, 1, v_inst_46_);
lean_closure_set(v___f_51_, 2, v_toBind_48_);
lean_closure_set(v___f_51_, 3, v___f_50_);
v___x_52_ = lean_apply_4(v_toBind_48_, lean_box(0), lean_box(0), v_getRef_49_, v___f_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption(lean_object* v_m_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_id_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_Lean_Elab_elabUnsetOption___redArg(v_inst_54_, v_inst_55_, v_inst_56_, v_inst_57_, v_id_58_);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_89_ = lean_box(0);
v___x_90_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_91_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v___x_89_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg(){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_93_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg___closed__0);
v___x_94_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg___boxed(lean_object* v___y_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg();
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0(lean_object* v_00_u03b1_97_, lean_object* v___y_98_, lean_object* v___y_99_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg();
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___boxed(lean_object* v_00_u03b1_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0(v_00_u03b1_102_, v___y_103_, v___y_104_);
lean_dec(v___y_104_);
lean_dec_ref(v___y_103_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__2(lean_object* v_opts_107_, lean_object* v_opt_108_){
_start:
{
lean_object* v_name_109_; lean_object* v_defValue_110_; lean_object* v_map_111_; lean_object* v___x_112_; 
v_name_109_ = lean_ctor_get(v_opt_108_, 0);
v_defValue_110_ = lean_ctor_get(v_opt_108_, 1);
v_map_111_ = lean_ctor_get(v_opts_107_, 0);
v___x_112_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_111_, v_name_109_);
if (lean_obj_tag(v___x_112_) == 0)
{
lean_inc(v_defValue_110_);
return v_defValue_110_;
}
else
{
lean_object* v_val_113_; 
v_val_113_ = lean_ctor_get(v___x_112_, 0);
lean_inc(v_val_113_);
lean_dec_ref_known(v___x_112_, 1);
if (lean_obj_tag(v_val_113_) == 3)
{
lean_object* v_v_114_; 
v_v_114_ = lean_ctor_get(v_val_113_, 0);
lean_inc(v_v_114_);
lean_dec_ref_known(v_val_113_, 1);
return v_v_114_;
}
else
{
lean_dec(v_val_113_);
lean_inc(v_defValue_110_);
return v_defValue_110_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__2___boxed(lean_object* v_opts_115_, lean_object* v_opt_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__2(v_opts_115_, v_opt_116_);
lean_dec_ref(v_opt_116_);
lean_dec_ref(v_opts_115_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1___lam__0(lean_object* v_a_118_, lean_object* v_scope_119_){
_start:
{
lean_object* v_header_120_; lean_object* v_currNamespace_121_; lean_object* v_openDecls_122_; lean_object* v_levelNames_123_; lean_object* v_varDecls_124_; lean_object* v_varUIds_125_; lean_object* v_includedVars_126_; lean_object* v_omittedVars_127_; uint8_t v_isNoncomputable_128_; uint8_t v_isPublic_129_; uint8_t v_isMeta_130_; lean_object* v_attrs_131_; lean_object* v___x_133_; uint8_t v_isShared_134_; uint8_t v_isSharedCheck_138_; 
v_header_120_ = lean_ctor_get(v_scope_119_, 0);
v_currNamespace_121_ = lean_ctor_get(v_scope_119_, 2);
v_openDecls_122_ = lean_ctor_get(v_scope_119_, 3);
v_levelNames_123_ = lean_ctor_get(v_scope_119_, 4);
v_varDecls_124_ = lean_ctor_get(v_scope_119_, 5);
v_varUIds_125_ = lean_ctor_get(v_scope_119_, 6);
v_includedVars_126_ = lean_ctor_get(v_scope_119_, 7);
v_omittedVars_127_ = lean_ctor_get(v_scope_119_, 8);
v_isNoncomputable_128_ = lean_ctor_get_uint8(v_scope_119_, sizeof(void*)*10);
v_isPublic_129_ = lean_ctor_get_uint8(v_scope_119_, sizeof(void*)*10 + 1);
v_isMeta_130_ = lean_ctor_get_uint8(v_scope_119_, sizeof(void*)*10 + 2);
v_attrs_131_ = lean_ctor_get(v_scope_119_, 9);
v_isSharedCheck_138_ = !lean_is_exclusive(v_scope_119_);
if (v_isSharedCheck_138_ == 0)
{
lean_object* v_unused_139_; 
v_unused_139_ = lean_ctor_get(v_scope_119_, 1);
lean_dec(v_unused_139_);
v___x_133_ = v_scope_119_;
v_isShared_134_ = v_isSharedCheck_138_;
goto v_resetjp_132_;
}
else
{
lean_inc(v_attrs_131_);
lean_inc(v_omittedVars_127_);
lean_inc(v_includedVars_126_);
lean_inc(v_varUIds_125_);
lean_inc(v_varDecls_124_);
lean_inc(v_levelNames_123_);
lean_inc(v_openDecls_122_);
lean_inc(v_currNamespace_121_);
lean_inc(v_header_120_);
lean_dec(v_scope_119_);
v___x_133_ = lean_box(0);
v_isShared_134_ = v_isSharedCheck_138_;
goto v_resetjp_132_;
}
v_resetjp_132_:
{
lean_object* v___x_136_; 
if (v_isShared_134_ == 0)
{
lean_ctor_set(v___x_133_, 1, v_a_118_);
v___x_136_ = v___x_133_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 10, 3);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v_header_120_);
lean_ctor_set(v_reuseFailAlloc_137_, 1, v_a_118_);
lean_ctor_set(v_reuseFailAlloc_137_, 2, v_currNamespace_121_);
lean_ctor_set(v_reuseFailAlloc_137_, 3, v_openDecls_122_);
lean_ctor_set(v_reuseFailAlloc_137_, 4, v_levelNames_123_);
lean_ctor_set(v_reuseFailAlloc_137_, 5, v_varDecls_124_);
lean_ctor_set(v_reuseFailAlloc_137_, 6, v_varUIds_125_);
lean_ctor_set(v_reuseFailAlloc_137_, 7, v_includedVars_126_);
lean_ctor_set(v_reuseFailAlloc_137_, 8, v_omittedVars_127_);
lean_ctor_set(v_reuseFailAlloc_137_, 9, v_attrs_131_);
lean_ctor_set_uint8(v_reuseFailAlloc_137_, sizeof(void*)*10, v_isNoncomputable_128_);
lean_ctor_set_uint8(v_reuseFailAlloc_137_, sizeof(void*)*10 + 1, v_isPublic_129_);
lean_ctor_set_uint8(v_reuseFailAlloc_137_, sizeof(void*)*10 + 2, v_isMeta_130_);
v___x_136_ = v_reuseFailAlloc_137_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
return v___x_136_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5___redArg(lean_object* v_t_140_, lean_object* v___y_141_){
_start:
{
lean_object* v___x_143_; lean_object* v_infoState_144_; uint8_t v_enabled_145_; 
v___x_143_ = lean_st_ref_get(v___y_141_);
v_infoState_144_ = lean_ctor_get(v___x_143_, 8);
lean_inc_ref(v_infoState_144_);
lean_dec(v___x_143_);
v_enabled_145_ = lean_ctor_get_uint8(v_infoState_144_, sizeof(void*)*3);
lean_dec_ref(v_infoState_144_);
if (v_enabled_145_ == 0)
{
lean_object* v___x_146_; lean_object* v___x_147_; 
lean_dec_ref(v_t_140_);
v___x_146_ = lean_box(0);
v___x_147_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
return v___x_147_;
}
else
{
lean_object* v___x_148_; lean_object* v_infoState_149_; lean_object* v_env_150_; lean_object* v_messages_151_; lean_object* v_scopes_152_; lean_object* v_usedQuotCtxts_153_; lean_object* v_nextMacroScope_154_; lean_object* v_maxRecDepth_155_; lean_object* v_ngen_156_; lean_object* v_auxDeclNGen_157_; lean_object* v_traceState_158_; lean_object* v_snapshotTasks_159_; lean_object* v_prevLinterStates_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_182_; 
v___x_148_ = lean_st_ref_take(v___y_141_);
v_infoState_149_ = lean_ctor_get(v___x_148_, 8);
v_env_150_ = lean_ctor_get(v___x_148_, 0);
v_messages_151_ = lean_ctor_get(v___x_148_, 1);
v_scopes_152_ = lean_ctor_get(v___x_148_, 2);
v_usedQuotCtxts_153_ = lean_ctor_get(v___x_148_, 3);
v_nextMacroScope_154_ = lean_ctor_get(v___x_148_, 4);
v_maxRecDepth_155_ = lean_ctor_get(v___x_148_, 5);
v_ngen_156_ = lean_ctor_get(v___x_148_, 6);
v_auxDeclNGen_157_ = lean_ctor_get(v___x_148_, 7);
v_traceState_158_ = lean_ctor_get(v___x_148_, 9);
v_snapshotTasks_159_ = lean_ctor_get(v___x_148_, 10);
v_prevLinterStates_160_ = lean_ctor_get(v___x_148_, 11);
v_isSharedCheck_182_ = !lean_is_exclusive(v___x_148_);
if (v_isSharedCheck_182_ == 0)
{
v___x_162_ = v___x_148_;
v_isShared_163_ = v_isSharedCheck_182_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_prevLinterStates_160_);
lean_inc(v_snapshotTasks_159_);
lean_inc(v_traceState_158_);
lean_inc(v_infoState_149_);
lean_inc(v_auxDeclNGen_157_);
lean_inc(v_ngen_156_);
lean_inc(v_maxRecDepth_155_);
lean_inc(v_nextMacroScope_154_);
lean_inc(v_usedQuotCtxts_153_);
lean_inc(v_scopes_152_);
lean_inc(v_messages_151_);
lean_inc(v_env_150_);
lean_dec(v___x_148_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_182_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
uint8_t v_enabled_164_; lean_object* v_assignment_165_; lean_object* v_lazyAssignment_166_; lean_object* v_trees_167_; lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_181_; 
v_enabled_164_ = lean_ctor_get_uint8(v_infoState_149_, sizeof(void*)*3);
v_assignment_165_ = lean_ctor_get(v_infoState_149_, 0);
v_lazyAssignment_166_ = lean_ctor_get(v_infoState_149_, 1);
v_trees_167_ = lean_ctor_get(v_infoState_149_, 2);
v_isSharedCheck_181_ = !lean_is_exclusive(v_infoState_149_);
if (v_isSharedCheck_181_ == 0)
{
v___x_169_ = v_infoState_149_;
v_isShared_170_ = v_isSharedCheck_181_;
goto v_resetjp_168_;
}
else
{
lean_inc(v_trees_167_);
lean_inc(v_lazyAssignment_166_);
lean_inc(v_assignment_165_);
lean_dec(v_infoState_149_);
v___x_169_ = lean_box(0);
v_isShared_170_ = v_isSharedCheck_181_;
goto v_resetjp_168_;
}
v_resetjp_168_:
{
lean_object* v___x_171_; lean_object* v___x_173_; 
v___x_171_ = l_Lean_PersistentArray_push___redArg(v_trees_167_, v_t_140_);
if (v_isShared_170_ == 0)
{
lean_ctor_set(v___x_169_, 2, v___x_171_);
v___x_173_ = v___x_169_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_180_; 
v_reuseFailAlloc_180_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_180_, 0, v_assignment_165_);
lean_ctor_set(v_reuseFailAlloc_180_, 1, v_lazyAssignment_166_);
lean_ctor_set(v_reuseFailAlloc_180_, 2, v___x_171_);
lean_ctor_set_uint8(v_reuseFailAlloc_180_, sizeof(void*)*3, v_enabled_164_);
v___x_173_ = v_reuseFailAlloc_180_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
lean_object* v___x_175_; 
if (v_isShared_163_ == 0)
{
lean_ctor_set(v___x_162_, 8, v___x_173_);
v___x_175_ = v___x_162_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v_env_150_);
lean_ctor_set(v_reuseFailAlloc_179_, 1, v_messages_151_);
lean_ctor_set(v_reuseFailAlloc_179_, 2, v_scopes_152_);
lean_ctor_set(v_reuseFailAlloc_179_, 3, v_usedQuotCtxts_153_);
lean_ctor_set(v_reuseFailAlloc_179_, 4, v_nextMacroScope_154_);
lean_ctor_set(v_reuseFailAlloc_179_, 5, v_maxRecDepth_155_);
lean_ctor_set(v_reuseFailAlloc_179_, 6, v_ngen_156_);
lean_ctor_set(v_reuseFailAlloc_179_, 7, v_auxDeclNGen_157_);
lean_ctor_set(v_reuseFailAlloc_179_, 8, v___x_173_);
lean_ctor_set(v_reuseFailAlloc_179_, 9, v_traceState_158_);
lean_ctor_set(v_reuseFailAlloc_179_, 10, v_snapshotTasks_159_);
lean_ctor_set(v_reuseFailAlloc_179_, 11, v_prevLinterStates_160_);
v___x_175_ = v_reuseFailAlloc_179_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_176_ = lean_st_ref_set(v___y_141_, v___x_175_);
v___x_177_ = lean_box(0);
v___x_178_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
return v___x_178_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5___redArg___boxed(lean_object* v_t_183_, lean_object* v___y_184_, lean_object* v___y_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5___redArg(v_t_183_, v___y_184_);
lean_dec(v___y_184_);
return v_res_186_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__0(void){
_start:
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; 
v___x_187_ = lean_unsigned_to_nat(32u);
v___x_188_ = lean_mk_empty_array_with_capacity(v___x_187_);
v___x_189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_189_, 0, v___x_188_);
return v___x_189_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__1(void){
_start:
{
size_t v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; 
v___x_190_ = ((size_t)5ULL);
v___x_191_ = lean_unsigned_to_nat(0u);
v___x_192_ = lean_unsigned_to_nat(32u);
v___x_193_ = lean_mk_empty_array_with_capacity(v___x_192_);
v___x_194_ = lean_obj_once(&lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__0, &lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__0_once, _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__0);
v___x_195_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_195_, 0, v___x_194_);
lean_ctor_set(v___x_195_, 1, v___x_193_);
lean_ctor_set(v___x_195_, 2, v___x_191_);
lean_ctor_set(v___x_195_, 3, v___x_191_);
lean_ctor_set_usize(v___x_195_, 4, v___x_190_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3(lean_object* v_t_196_, lean_object* v___y_197_, lean_object* v___y_198_){
_start:
{
lean_object* v___x_200_; lean_object* v_infoState_201_; uint8_t v_enabled_202_; 
v___x_200_ = lean_st_ref_get(v___y_198_);
v_infoState_201_ = lean_ctor_get(v___x_200_, 8);
lean_inc_ref(v_infoState_201_);
lean_dec(v___x_200_);
v_enabled_202_ = lean_ctor_get_uint8(v_infoState_201_, sizeof(void*)*3);
lean_dec_ref(v_infoState_201_);
if (v_enabled_202_ == 0)
{
lean_object* v___x_203_; lean_object* v___x_204_; 
lean_dec_ref(v_t_196_);
v___x_203_ = lean_box(0);
v___x_204_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
return v___x_204_;
}
else
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_205_ = lean_obj_once(&lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__1, &lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__1_once, _init_lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___closed__1);
v___x_206_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_206_, 0, v_t_196_);
lean_ctor_set(v___x_206_, 1, v___x_205_);
v___x_207_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5___redArg(v___x_206_, v___y_198_);
return v___x_207_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3___boxed(lean_object* v_t_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3(v_t_208_, v___y_209_, v___y_210_);
lean_dec(v___y_210_);
lean_dec_ref(v___y_209_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1(lean_object* v_info_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_217_ = lean_alloc_ctor(8, 1, 0);
lean_ctor_set(v___x_217_, 0, v_info_213_);
v___x_218_ = lp_mathlib_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3(v___x_217_, v___y_214_, v___y_215_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1___boxed(lean_object* v_info_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1(v_info_219_, v___y_220_, v___y_221_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2___redArg(lean_object* v_optionName_224_, lean_object* v___y_225_){
_start:
{
lean_object* v___x_227_; lean_object* v_scopes_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v_opts_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_227_ = lean_st_ref_get(v___y_225_);
v_scopes_228_ = lean_ctor_get(v___x_227_, 2);
lean_inc(v_scopes_228_);
lean_dec(v___x_227_);
v___x_229_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_230_ = l_List_head_x21___redArg(v___x_229_, v_scopes_228_);
lean_dec(v_scopes_228_);
v_opts_231_ = lean_ctor_get(v___x_230_, 1);
lean_inc_ref(v_opts_231_);
lean_dec(v___x_230_);
v___x_232_ = l_Lean_Options_erase(v_opts_231_, v_optionName_224_);
v___x_233_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_233_, 0, v___x_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2___redArg___boxed(lean_object* v_optionName_234_, lean_object* v___y_235_, lean_object* v___y_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2___redArg(v_optionName_234_, v___y_235_);
lean_dec(v___y_235_);
lean_dec(v_optionName_234_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1(lean_object* v_id_238_, lean_object* v___y_239_, lean_object* v___y_240_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = l_Lean_Elab_Command_getRef___redArg(v___y_239_);
if (lean_obj_tag(v___x_242_) == 0)
{
lean_object* v_a_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
v_a_243_ = lean_ctor_get(v___x_242_, 0);
lean_inc(v_a_243_);
lean_dec_ref_known(v___x_242_, 1);
v___x_244_ = lean_alloc_ctor(5, 1, 0);
lean_ctor_set(v___x_244_, 0, v_a_243_);
v___x_245_ = lp_mathlib_Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1(v___x_244_, v___y_239_, v___y_240_);
lean_dec_ref(v___x_245_);
v___x_246_ = l_Lean_Syntax_getId(v_id_238_);
v___x_247_ = l_Lean_Name_eraseMacroScopes(v___x_246_);
lean_dec(v___x_246_);
v___x_248_ = lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2___redArg(v___x_247_, v___y_240_);
lean_dec(v___x_247_);
return v___x_248_;
}
else
{
lean_object* v_a_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_256_; 
v_a_249_ = lean_ctor_get(v___x_242_, 0);
v_isSharedCheck_256_ = !lean_is_exclusive(v___x_242_);
if (v_isSharedCheck_256_ == 0)
{
v___x_251_ = v___x_242_;
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_a_249_);
lean_dec(v___x_242_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___x_254_; 
if (v_isShared_252_ == 0)
{
v___x_254_ = v___x_251_;
goto v_reusejp_253_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v_a_249_);
v___x_254_ = v_reuseFailAlloc_255_;
goto v_reusejp_253_;
}
v_reusejp_253_:
{
return v___x_254_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1___boxed(lean_object* v_id_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1(v_id_257_, v___y_258_, v___y_259_);
lean_dec(v___y_259_);
lean_dec_ref(v___y_258_);
lean_dec(v_id_257_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1(lean_object* v_x_262_, lean_object* v_a_263_, lean_object* v_a_264_){
_start:
{
lean_object* v___x_266_; uint8_t v___x_267_; 
v___x_266_ = ((lean_object*)(lp_mathlib_Lean_Elab_Command_unsetOption___closed__4));
lean_inc(v_x_262_);
v___x_267_ = l_Lean_Syntax_isOfKind(v_x_262_, v___x_266_);
if (v___x_267_ == 0)
{
lean_object* v___x_268_; 
lean_dec(v_x_262_);
v___x_268_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__0___redArg();
return v___x_268_;
}
else
{
lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_269_ = lean_unsigned_to_nat(1u);
v___x_270_ = l_Lean_Syntax_getArg(v_x_262_, v___x_269_);
lean_dec(v_x_262_);
v___x_271_ = lp_mathlib_Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1(v___x_270_, v_a_263_, v_a_264_);
lean_dec(v___x_270_);
if (lean_obj_tag(v___x_271_) == 0)
{
lean_object* v_a_272_; lean_object* v___x_273_; lean_object* v_env_274_; lean_object* v_messages_275_; lean_object* v_scopes_276_; lean_object* v_usedQuotCtxts_277_; lean_object* v_nextMacroScope_278_; lean_object* v_ngen_279_; lean_object* v_auxDeclNGen_280_; lean_object* v_infoState_281_; lean_object* v_traceState_282_; lean_object* v_snapshotTasks_283_; lean_object* v_prevLinterStates_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_296_; 
v_a_272_ = lean_ctor_get(v___x_271_, 0);
lean_inc(v_a_272_);
lean_dec_ref_known(v___x_271_, 1);
v___x_273_ = lean_st_ref_take(v_a_264_);
v_env_274_ = lean_ctor_get(v___x_273_, 0);
v_messages_275_ = lean_ctor_get(v___x_273_, 1);
v_scopes_276_ = lean_ctor_get(v___x_273_, 2);
v_usedQuotCtxts_277_ = lean_ctor_get(v___x_273_, 3);
v_nextMacroScope_278_ = lean_ctor_get(v___x_273_, 4);
v_ngen_279_ = lean_ctor_get(v___x_273_, 6);
v_auxDeclNGen_280_ = lean_ctor_get(v___x_273_, 7);
v_infoState_281_ = lean_ctor_get(v___x_273_, 8);
v_traceState_282_ = lean_ctor_get(v___x_273_, 9);
v_snapshotTasks_283_ = lean_ctor_get(v___x_273_, 10);
v_prevLinterStates_284_ = lean_ctor_get(v___x_273_, 11);
v_isSharedCheck_296_ = !lean_is_exclusive(v___x_273_);
if (v_isSharedCheck_296_ == 0)
{
lean_object* v_unused_297_; 
v_unused_297_ = lean_ctor_get(v___x_273_, 5);
lean_dec(v_unused_297_);
v___x_286_ = v___x_273_;
v_isShared_287_ = v_isSharedCheck_296_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_prevLinterStates_284_);
lean_inc(v_snapshotTasks_283_);
lean_inc(v_traceState_282_);
lean_inc(v_infoState_281_);
lean_inc(v_auxDeclNGen_280_);
lean_inc(v_ngen_279_);
lean_inc(v_nextMacroScope_278_);
lean_inc(v_usedQuotCtxts_277_);
lean_inc(v_scopes_276_);
lean_inc(v_messages_275_);
lean_inc(v_env_274_);
lean_dec(v___x_273_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_296_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_291_; 
v___x_288_ = l_Lean_maxRecDepth;
v___x_289_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__2(v_a_272_, v___x_288_);
if (v_isShared_287_ == 0)
{
lean_ctor_set(v___x_286_, 5, v___x_289_);
v___x_291_ = v___x_286_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_295_; 
v_reuseFailAlloc_295_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_295_, 0, v_env_274_);
lean_ctor_set(v_reuseFailAlloc_295_, 1, v_messages_275_);
lean_ctor_set(v_reuseFailAlloc_295_, 2, v_scopes_276_);
lean_ctor_set(v_reuseFailAlloc_295_, 3, v_usedQuotCtxts_277_);
lean_ctor_set(v_reuseFailAlloc_295_, 4, v_nextMacroScope_278_);
lean_ctor_set(v_reuseFailAlloc_295_, 5, v___x_289_);
lean_ctor_set(v_reuseFailAlloc_295_, 6, v_ngen_279_);
lean_ctor_set(v_reuseFailAlloc_295_, 7, v_auxDeclNGen_280_);
lean_ctor_set(v_reuseFailAlloc_295_, 8, v_infoState_281_);
lean_ctor_set(v_reuseFailAlloc_295_, 9, v_traceState_282_);
lean_ctor_set(v_reuseFailAlloc_295_, 10, v_snapshotTasks_283_);
lean_ctor_set(v_reuseFailAlloc_295_, 11, v_prevLinterStates_284_);
v___x_291_ = v_reuseFailAlloc_295_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
lean_object* v___x_292_; lean_object* v___f_293_; lean_object* v___x_294_; 
v___x_292_ = lean_st_ref_set(v_a_264_, v___x_291_);
v___f_293_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1___lam__0), 2, 1);
lean_closure_set(v___f_293_, 0, v_a_272_);
v___x_294_ = l_Lean_Elab_Command_modifyScope___redArg(v___f_293_, v_a_264_);
return v___x_294_;
}
}
}
else
{
lean_object* v_a_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_305_; 
v_a_298_ = lean_ctor_get(v___x_271_, 0);
v_isSharedCheck_305_ = !lean_is_exclusive(v___x_271_);
if (v_isSharedCheck_305_ == 0)
{
v___x_300_ = v___x_271_;
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_a_298_);
lean_dec(v___x_271_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_303_; 
if (v_isShared_301_ == 0)
{
v___x_303_ = v___x_300_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_304_; 
v_reuseFailAlloc_304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_304_, 0, v_a_298_);
v___x_303_ = v_reuseFailAlloc_304_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
return v___x_303_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1___boxed(lean_object* v_x_306_, lean_object* v_a_307_, lean_object* v_a_308_, lean_object* v_a_309_){
_start:
{
lean_object* v_res_310_; 
v_res_310_ = lp_mathlib_Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1(v_x_306_, v_a_307_, v_a_308_);
lean_dec(v_a_308_);
lean_dec_ref(v_a_307_);
return v_res_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2(lean_object* v_optionName_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2___redArg(v_optionName_311_, v___y_313_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2___boxed(lean_object* v_optionName_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_mathlib___private_Mathlib_Tactic_UnsetOption_0__Lean_Elab_elabUnsetOption_unsetOption___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__2(v_optionName_316_, v___y_317_, v___y_318_);
lean_dec(v___y_318_);
lean_dec_ref(v___y_317_);
lean_dec(v_optionName_316_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5(lean_object* v_t_321_, lean_object* v___y_322_, lean_object* v___y_323_){
_start:
{
lean_object* v___x_325_; 
v___x_325_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5___redArg(v_t_321_, v___y_323_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5___boxed(lean_object* v_t_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addCompletionInfo___at___00Lean_Elab_elabUnsetOption___at___00Lean_Elab_Command___aux__Mathlib__Tactic__UnsetOption______elabRules__Lean__Elab__Command__unsetOption__1_spec__1_spec__1_spec__3_spec__5(v_t_326_, v___y_327_, v___y_328_);
lean_dec(v___y_328_);
lean_dec_ref(v___y_327_);
return v_res_330_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_UnsetOption(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Parser_Term(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Do(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_UnsetOption(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Do(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Parser_Term(uint8_t builtin);
lean_object* initialize_Lean_Parser_Do(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_UnsetOption(uint8_t builtin) {
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
res = initialize_Lean_Parser_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Do(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_UnsetOption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_UnsetOption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_UnsetOption(builtin);
}
#ifdef __cplusplus
}
#endif
