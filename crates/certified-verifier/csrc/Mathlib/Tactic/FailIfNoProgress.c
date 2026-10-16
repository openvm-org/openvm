// Lean compiler output
// Module: Mathlib.Tactic.FailIfNoProgress
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.Basic public meta import Lean.Meta.Tactic.Util
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_Tactic_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toList___redArg(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
uint8_t l_Lean_LocalDecl_isLet(lean_object*, uint8_t);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_value(lean_object*, uint8_t);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "failIfNoProgress"};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__2_value),LEAN_SCALAR_PTR_LITERAL(238, 120, 52, 11, 174, 48, 92, 172)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "fail_if_no_progress "};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__8_value),LEAN_SCALAR_PTR_LITERAL(13, 106, 54, 236, 164, 218, 24, 154)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_failIfNoProgress = (const lean_object*)&lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_lctxIsDefEq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_lctxIsDefEq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_lctxIsDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_lctxIsDefEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FailIfNoProgress_0__Mathlib_Tactic_lctxIsDefEq_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FailIfNoProgress_0__Mathlib_Tactic_lctxIsDefEq_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "no progress made on\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0___redArg(lean_object* v_k_29_, uint8_t v_allowLevelAssignments_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_30_, v_k_29_, v___y_31_, v___y_32_, v___y_33_, v___y_34_);
if (lean_obj_tag(v___x_36_) == 0)
{
lean_object* v_a_37_; lean_object* v___x_39_; uint8_t v_isShared_40_; uint8_t v_isSharedCheck_44_; 
v_a_37_ = lean_ctor_get(v___x_36_, 0);
v_isSharedCheck_44_ = !lean_is_exclusive(v___x_36_);
if (v_isSharedCheck_44_ == 0)
{
v___x_39_ = v___x_36_;
v_isShared_40_ = v_isSharedCheck_44_;
goto v_resetjp_38_;
}
else
{
lean_inc(v_a_37_);
lean_dec(v___x_36_);
v___x_39_ = lean_box(0);
v_isShared_40_ = v_isSharedCheck_44_;
goto v_resetjp_38_;
}
v_resetjp_38_:
{
lean_object* v___x_42_; 
if (v_isShared_40_ == 0)
{
v___x_42_ = v___x_39_;
goto v_reusejp_41_;
}
else
{
lean_object* v_reuseFailAlloc_43_; 
v_reuseFailAlloc_43_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_43_, 0, v_a_37_);
v___x_42_ = v_reuseFailAlloc_43_;
goto v_reusejp_41_;
}
v_reusejp_41_:
{
return v___x_42_;
}
}
}
else
{
lean_object* v_a_45_; lean_object* v___x_47_; uint8_t v_isShared_48_; uint8_t v_isSharedCheck_52_; 
v_a_45_ = lean_ctor_get(v___x_36_, 0);
v_isSharedCheck_52_ = !lean_is_exclusive(v___x_36_);
if (v_isSharedCheck_52_ == 0)
{
v___x_47_ = v___x_36_;
v_isShared_48_ = v_isSharedCheck_52_;
goto v_resetjp_46_;
}
else
{
lean_inc(v_a_45_);
lean_dec(v___x_36_);
v___x_47_ = lean_box(0);
v_isShared_48_ = v_isSharedCheck_52_;
goto v_resetjp_46_;
}
v_resetjp_46_:
{
lean_object* v___x_50_; 
if (v_isShared_48_ == 0)
{
v___x_50_ = v___x_47_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v_a_45_);
v___x_50_ = v_reuseFailAlloc_51_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
return v___x_50_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0___redArg___boxed(lean_object* v_k_53_, lean_object* v_allowLevelAssignments_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_60_; lean_object* v_res_61_; 
v_allowLevelAssignments_boxed_60_ = lean_unbox(v_allowLevelAssignments_54_);
v_res_61_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0___redArg(v_k_53_, v_allowLevelAssignments_boxed_60_, v___y_55_, v___y_56_, v___y_57_, v___y_58_);
lean_dec(v___y_58_);
lean_dec_ref(v___y_57_);
lean_dec(v___y_56_);
lean_dec_ref(v___y_55_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0(lean_object* v_00_u03b1_62_, lean_object* v_k_63_, uint8_t v_allowLevelAssignments_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0___redArg(v_k_63_, v_allowLevelAssignments_64_, v___y_65_, v___y_66_, v___y_67_, v___y_68_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0___boxed(lean_object* v_00_u03b1_71_, lean_object* v_k_72_, lean_object* v_allowLevelAssignments_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_79_; lean_object* v_res_80_; 
v_allowLevelAssignments_boxed_79_ = lean_unbox(v_allowLevelAssignments_73_);
v_res_80_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0(v_00_u03b1_71_, v_k_72_, v_allowLevelAssignments_boxed_79_, v___y_74_, v___y_75_, v___y_76_, v___y_77_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_lctxIsDefEq___lam__0(lean_object* v___x_81_, lean_object* v___x_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = l_Lean_Meta_isExprDefEq(v___x_81_, v___x_82_, v___y_83_, v___y_84_, v___y_85_, v___y_86_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_lctxIsDefEq___lam__0___boxed(lean_object* v___x_89_, lean_object* v___x_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_Mathlib_Tactic_lctxIsDefEq___lam__0(v___x_89_, v___x_90_, v___y_91_, v___y_92_, v___y_93_, v___y_94_);
lean_dec(v___y_94_);
lean_dec_ref(v___y_93_);
lean_dec(v___y_92_);
lean_dec_ref(v___y_91_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_lctxIsDefEq(lean_object* v_x_97_, lean_object* v_x_98_, lean_object* v_a_99_, lean_object* v_a_100_, lean_object* v_a_101_, lean_object* v_a_102_){
_start:
{
if (lean_obj_tag(v_x_97_) == 0)
{
if (lean_obj_tag(v_x_98_) == 0)
{
uint8_t v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_108_ = 1;
v___x_109_ = lean_box(v___x_108_);
v___x_110_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
return v___x_110_;
}
else
{
lean_object* v_head_111_; 
v_head_111_ = lean_ctor_get(v_x_98_, 0);
if (lean_obj_tag(v_head_111_) == 0)
{
lean_object* v_tail_112_; 
v_tail_112_ = lean_ctor_get(v_x_98_, 1);
lean_inc(v_tail_112_);
lean_dec_ref_known(v_x_98_, 2);
v_x_98_ = v_tail_112_;
goto _start;
}
else
{
lean_dec_ref_known(v_x_98_, 2);
goto v___jp_104_;
}
}
}
else
{
lean_object* v_head_114_; 
v_head_114_ = lean_ctor_get(v_x_97_, 0);
lean_inc(v_head_114_);
if (lean_obj_tag(v_head_114_) == 0)
{
lean_object* v_tail_115_; 
v_tail_115_ = lean_ctor_get(v_x_97_, 1);
lean_inc(v_tail_115_);
lean_dec_ref_known(v_x_97_, 2);
v_x_97_ = v_tail_115_;
goto _start;
}
else
{
if (lean_obj_tag(v_x_98_) == 1)
{
lean_object* v_head_117_; 
v_head_117_ = lean_ctor_get(v_x_98_, 0);
lean_inc(v_head_117_);
if (lean_obj_tag(v_head_117_) == 0)
{
lean_object* v_tail_118_; 
lean_dec_ref_known(v_head_114_, 1);
v_tail_118_ = lean_ctor_get(v_x_98_, 1);
lean_inc(v_tail_118_);
lean_dec_ref_known(v_x_98_, 2);
v_x_98_ = v_tail_118_;
goto _start;
}
else
{
lean_object* v_tail_120_; lean_object* v_val_121_; lean_object* v___x_123_; uint8_t v_isShared_124_; uint8_t v_isSharedCheck_178_; 
v_tail_120_ = lean_ctor_get(v_x_97_, 1);
lean_inc(v_tail_120_);
lean_dec_ref_known(v_x_97_, 2);
v_val_121_ = lean_ctor_get(v_head_114_, 0);
v_isSharedCheck_178_ = !lean_is_exclusive(v_head_114_);
if (v_isSharedCheck_178_ == 0)
{
v___x_123_ = v_head_114_;
v_isShared_124_ = v_isSharedCheck_178_;
goto v_resetjp_122_;
}
else
{
lean_inc(v_val_121_);
lean_dec(v_head_114_);
v___x_123_ = lean_box(0);
v_isShared_124_ = v_isSharedCheck_178_;
goto v_resetjp_122_;
}
v_resetjp_122_:
{
lean_object* v_tail_125_; lean_object* v_val_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_177_; 
v_tail_125_ = lean_ctor_get(v_x_98_, 1);
lean_inc(v_tail_125_);
lean_dec_ref_known(v_x_98_, 2);
v_val_126_ = lean_ctor_get(v_head_117_, 0);
v_isSharedCheck_177_ = !lean_is_exclusive(v_head_117_);
if (v_isSharedCheck_177_ == 0)
{
v___x_128_ = v_head_117_;
v_isShared_129_ = v_isSharedCheck_177_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_val_126_);
lean_dec(v_head_117_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_177_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
uint8_t v___x_130_; uint8_t v___x_131_; uint8_t v___y_171_; uint8_t v___x_176_; 
v___x_130_ = 0;
v___x_131_ = l_Lean_LocalDecl_isLet(v_val_121_, v___x_130_);
v___x_176_ = l_Lean_LocalDecl_isLet(v_val_126_, v___x_130_);
if (v___x_131_ == 0)
{
if (v___x_176_ == 0)
{
lean_del_object(v___x_123_);
goto v___jp_132_;
}
else
{
v___y_171_ = v___x_131_;
goto v___jp_170_;
}
}
else
{
v___y_171_ = v___x_176_;
goto v___jp_170_;
}
v___jp_132_:
{
lean_object* v___x_133_; lean_object* v___x_134_; uint8_t v___x_135_; 
v___x_133_ = l_Lean_LocalDecl_fvarId(v_val_121_);
v___x_134_ = l_Lean_LocalDecl_fvarId(v_val_126_);
v___x_135_ = l_Lean_instBEqFVarId_beq(v___x_133_, v___x_134_);
lean_dec(v___x_134_);
lean_dec(v___x_133_);
if (v___x_135_ == 0)
{
lean_object* v___x_136_; lean_object* v___x_138_; 
lean_dec(v_val_126_);
lean_dec(v_tail_125_);
lean_dec(v_val_121_);
lean_dec(v_tail_120_);
v___x_136_ = lean_box(v___x_130_);
if (v_isShared_129_ == 0)
{
lean_ctor_set_tag(v___x_128_, 0);
lean_ctor_set(v___x_128_, 0, v___x_136_);
v___x_138_ = v___x_128_;
goto v_reusejp_137_;
}
else
{
lean_object* v_reuseFailAlloc_139_; 
v_reuseFailAlloc_139_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_139_, 0, v___x_136_);
v___x_138_ = v_reuseFailAlloc_139_;
goto v_reusejp_137_;
}
v_reusejp_137_:
{
return v___x_138_;
}
}
else
{
lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___f_142_; lean_object* v___x_143_; 
lean_del_object(v___x_128_);
v___x_140_ = l_Lean_LocalDecl_type(v_val_121_);
v___x_141_ = l_Lean_LocalDecl_type(v_val_126_);
v___f_142_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_lctxIsDefEq___lam__0___boxed), 7, 2);
lean_closure_set(v___f_142_, 0, v___x_140_);
lean_closure_set(v___f_142_, 1, v___x_141_);
v___x_143_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0___redArg(v___f_142_, v___x_130_, v_a_99_, v_a_100_, v_a_101_, v_a_102_);
if (lean_obj_tag(v___x_143_) == 0)
{
lean_object* v_a_144_; lean_object* v___x_146_; uint8_t v_isShared_147_; uint8_t v_isSharedCheck_169_; 
v_a_144_ = lean_ctor_get(v___x_143_, 0);
v_isSharedCheck_169_ = !lean_is_exclusive(v___x_143_);
if (v_isSharedCheck_169_ == 0)
{
v___x_146_ = v___x_143_;
v_isShared_147_ = v_isSharedCheck_169_;
goto v_resetjp_145_;
}
else
{
lean_inc(v_a_144_);
lean_dec(v___x_143_);
v___x_146_ = lean_box(0);
v_isShared_147_ = v_isSharedCheck_169_;
goto v_resetjp_145_;
}
v_resetjp_145_:
{
uint8_t v___x_148_; 
v___x_148_ = lean_unbox(v_a_144_);
lean_dec(v_a_144_);
if (v___x_148_ == 0)
{
lean_object* v___x_149_; lean_object* v___x_151_; 
lean_dec(v_val_126_);
lean_dec(v_tail_125_);
lean_dec(v_val_121_);
lean_dec(v_tail_120_);
v___x_149_ = lean_box(v___x_130_);
if (v_isShared_147_ == 0)
{
lean_ctor_set(v___x_146_, 0, v___x_149_);
v___x_151_ = v___x_146_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_152_; 
v_reuseFailAlloc_152_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_152_, 0, v___x_149_);
v___x_151_ = v_reuseFailAlloc_152_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
return v___x_151_;
}
}
else
{
lean_del_object(v___x_146_);
if (v___x_131_ == 0)
{
lean_dec(v_val_126_);
lean_dec(v_val_121_);
v_x_97_ = v_tail_120_;
v_x_98_ = v_tail_125_;
goto _start;
}
else
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___f_156_; lean_object* v___x_157_; 
v___x_154_ = l_Lean_LocalDecl_value(v_val_121_, v___x_130_);
lean_dec(v_val_121_);
v___x_155_ = l_Lean_LocalDecl_value(v_val_126_, v___x_130_);
lean_dec(v_val_126_);
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_lctxIsDefEq___lam__0___boxed), 7, 2);
lean_closure_set(v___f_156_, 0, v___x_154_);
lean_closure_set(v___f_156_, 1, v___x_155_);
v___x_157_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_lctxIsDefEq_spec__0___redArg(v___f_156_, v___x_130_, v_a_99_, v_a_100_, v_a_101_, v_a_102_);
if (lean_obj_tag(v___x_157_) == 0)
{
lean_object* v_a_158_; lean_object* v___x_160_; uint8_t v_isShared_161_; uint8_t v_isSharedCheck_168_; 
v_a_158_ = lean_ctor_get(v___x_157_, 0);
v_isSharedCheck_168_ = !lean_is_exclusive(v___x_157_);
if (v_isSharedCheck_168_ == 0)
{
v___x_160_ = v___x_157_;
v_isShared_161_ = v_isSharedCheck_168_;
goto v_resetjp_159_;
}
else
{
lean_inc(v_a_158_);
lean_dec(v___x_157_);
v___x_160_ = lean_box(0);
v_isShared_161_ = v_isSharedCheck_168_;
goto v_resetjp_159_;
}
v_resetjp_159_:
{
uint8_t v___x_162_; 
v___x_162_ = lean_unbox(v_a_158_);
lean_dec(v_a_158_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; lean_object* v___x_165_; 
lean_dec(v_tail_125_);
lean_dec(v_tail_120_);
v___x_163_ = lean_box(v___x_130_);
if (v_isShared_161_ == 0)
{
lean_ctor_set(v___x_160_, 0, v___x_163_);
v___x_165_ = v___x_160_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v___x_163_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
else
{
lean_del_object(v___x_160_);
v_x_97_ = v_tail_120_;
v_x_98_ = v_tail_125_;
goto _start;
}
}
}
else
{
lean_dec(v_tail_125_);
lean_dec(v_tail_120_);
return v___x_157_;
}
}
}
}
}
else
{
lean_dec(v_val_126_);
lean_dec(v_tail_125_);
lean_dec(v_val_121_);
lean_dec(v_tail_120_);
return v___x_143_;
}
}
}
v___jp_170_:
{
if (v___y_171_ == 0)
{
lean_object* v___x_172_; lean_object* v___x_174_; 
lean_del_object(v___x_128_);
lean_dec(v_val_126_);
lean_dec(v_tail_125_);
lean_dec(v_val_121_);
lean_dec(v_tail_120_);
v___x_172_ = lean_box(v___x_130_);
if (v_isShared_124_ == 0)
{
lean_ctor_set_tag(v___x_123_, 0);
lean_ctor_set(v___x_123_, 0, v___x_172_);
v___x_174_ = v___x_123_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v___x_172_);
v___x_174_ = v_reuseFailAlloc_175_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
return v___x_174_;
}
}
else
{
lean_del_object(v___x_123_);
goto v___jp_132_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_head_114_, 1);
lean_dec_ref_known(v_x_97_, 2);
lean_dec(v_x_98_);
goto v___jp_104_;
}
}
}
v___jp_104_:
{
uint8_t v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_105_ = 0;
v___x_106_ = lean_box(v___x_105_);
v___x_107_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
return v___x_107_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_lctxIsDefEq___boxed(lean_object* v_x_179_, lean_object* v_x_180_, lean_object* v_a_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Mathlib_Tactic_lctxIsDefEq(v_x_179_, v_x_180_, v_a_181_, v_a_182_, v_a_183_, v_a_184_);
lean_dec(v_a_184_);
lean_dec_ref(v_a_183_);
lean_dec(v_a_182_);
lean_dec_ref(v_a_181_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FailIfNoProgress_0__Mathlib_Tactic_lctxIsDefEq_match__1_splitter___redArg(lean_object* v_x_187_, lean_object* v_x_188_, lean_object* v_h__1_189_, lean_object* v_h__2_190_, lean_object* v_h__3_191_, lean_object* v_h__4_192_, lean_object* v_h__5_193_){
_start:
{
if (lean_obj_tag(v_x_187_) == 0)
{
lean_dec(v_h__3_191_);
lean_dec(v_h__1_189_);
if (lean_obj_tag(v_x_188_) == 0)
{
lean_object* v___x_194_; lean_object* v___x_195_; 
lean_dec(v_h__5_193_);
lean_dec(v_h__2_190_);
v___x_194_ = lean_box(0);
v___x_195_ = lean_apply_1(v_h__4_192_, v___x_194_);
return v___x_195_;
}
else
{
lean_object* v_head_196_; 
lean_dec(v_h__4_192_);
v_head_196_ = lean_ctor_get(v_x_188_, 0);
if (lean_obj_tag(v_head_196_) == 0)
{
lean_object* v_tail_197_; lean_object* v___x_198_; 
lean_dec(v_h__5_193_);
v_tail_197_ = lean_ctor_get(v_x_188_, 1);
lean_inc(v_tail_197_);
lean_dec_ref_known(v_x_188_, 2);
v___x_198_ = lean_apply_3(v_h__2_190_, v_x_187_, v_tail_197_, lean_box(0));
return v___x_198_;
}
else
{
lean_object* v___x_199_; 
lean_dec(v_h__2_190_);
v___x_199_ = lean_apply_6(v_h__5_193_, v_x_187_, v_x_188_, lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_199_;
}
}
}
else
{
lean_object* v_head_200_; 
lean_dec(v_h__4_192_);
v_head_200_ = lean_ctor_get(v_x_187_, 0);
if (lean_obj_tag(v_head_200_) == 0)
{
lean_object* v_tail_201_; lean_object* v___x_202_; 
lean_dec(v_h__5_193_);
lean_dec(v_h__3_191_);
lean_dec(v_h__2_190_);
v_tail_201_ = lean_ctor_get(v_x_187_, 1);
lean_inc(v_tail_201_);
lean_dec_ref_known(v_x_187_, 2);
v___x_202_ = lean_apply_2(v_h__1_189_, v_tail_201_, v_x_188_);
return v___x_202_;
}
else
{
lean_dec(v_h__1_189_);
if (lean_obj_tag(v_x_188_) == 1)
{
lean_object* v_head_203_; 
lean_dec(v_h__5_193_);
v_head_203_ = lean_ctor_get(v_x_188_, 0);
if (lean_obj_tag(v_head_203_) == 0)
{
lean_object* v_tail_204_; lean_object* v___x_205_; 
lean_dec(v_h__3_191_);
v_tail_204_ = lean_ctor_get(v_x_188_, 1);
lean_inc(v_tail_204_);
lean_dec_ref_known(v_x_188_, 2);
v___x_205_ = lean_apply_3(v_h__2_190_, v_x_187_, v_tail_204_, lean_box(0));
return v___x_205_;
}
else
{
lean_object* v_tail_206_; lean_object* v_val_207_; lean_object* v_tail_208_; lean_object* v_val_209_; lean_object* v___x_210_; 
lean_inc_ref(v_head_203_);
lean_inc_ref(v_head_200_);
lean_dec(v_h__2_190_);
v_tail_206_ = lean_ctor_get(v_x_187_, 1);
lean_inc(v_tail_206_);
lean_dec_ref_known(v_x_187_, 2);
v_val_207_ = lean_ctor_get(v_head_200_, 0);
lean_inc(v_val_207_);
lean_dec_ref_known(v_head_200_, 1);
v_tail_208_ = lean_ctor_get(v_x_188_, 1);
lean_inc(v_tail_208_);
lean_dec_ref_known(v_x_188_, 2);
v_val_209_ = lean_ctor_get(v_head_203_, 0);
lean_inc(v_val_209_);
lean_dec_ref_known(v_head_203_, 1);
v___x_210_ = lean_apply_4(v_h__3_191_, v_val_207_, v_tail_206_, v_val_209_, v_tail_208_);
return v___x_210_;
}
}
else
{
lean_object* v___x_211_; 
lean_dec(v_h__3_191_);
lean_dec(v_h__2_190_);
v___x_211_ = lean_apply_6(v_h__5_193_, v_x_187_, v_x_188_, lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_211_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FailIfNoProgress_0__Mathlib_Tactic_lctxIsDefEq_match__1_splitter(lean_object* v_motive_212_, lean_object* v_x_213_, lean_object* v_x_214_, lean_object* v_h__1_215_, lean_object* v_h__2_216_, lean_object* v_h__3_217_, lean_object* v_h__4_218_, lean_object* v_h__5_219_){
_start:
{
if (lean_obj_tag(v_x_213_) == 0)
{
lean_dec(v_h__3_217_);
lean_dec(v_h__1_215_);
if (lean_obj_tag(v_x_214_) == 0)
{
lean_object* v___x_220_; lean_object* v___x_221_; 
lean_dec(v_h__5_219_);
lean_dec(v_h__2_216_);
v___x_220_ = lean_box(0);
v___x_221_ = lean_apply_1(v_h__4_218_, v___x_220_);
return v___x_221_;
}
else
{
lean_object* v_head_222_; 
lean_dec(v_h__4_218_);
v_head_222_ = lean_ctor_get(v_x_214_, 0);
if (lean_obj_tag(v_head_222_) == 0)
{
lean_object* v_tail_223_; lean_object* v___x_224_; 
lean_dec(v_h__5_219_);
v_tail_223_ = lean_ctor_get(v_x_214_, 1);
lean_inc(v_tail_223_);
lean_dec_ref_known(v_x_214_, 2);
v___x_224_ = lean_apply_3(v_h__2_216_, v_x_213_, v_tail_223_, lean_box(0));
return v___x_224_;
}
else
{
lean_object* v___x_225_; 
lean_dec(v_h__2_216_);
v___x_225_ = lean_apply_6(v_h__5_219_, v_x_213_, v_x_214_, lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_225_;
}
}
}
else
{
lean_object* v_head_226_; 
lean_dec(v_h__4_218_);
v_head_226_ = lean_ctor_get(v_x_213_, 0);
if (lean_obj_tag(v_head_226_) == 0)
{
lean_object* v_tail_227_; lean_object* v___x_228_; 
lean_dec(v_h__5_219_);
lean_dec(v_h__3_217_);
lean_dec(v_h__2_216_);
v_tail_227_ = lean_ctor_get(v_x_213_, 1);
lean_inc(v_tail_227_);
lean_dec_ref_known(v_x_213_, 2);
v___x_228_ = lean_apply_2(v_h__1_215_, v_tail_227_, v_x_214_);
return v___x_228_;
}
else
{
lean_dec(v_h__1_215_);
if (lean_obj_tag(v_x_214_) == 1)
{
lean_object* v_head_229_; 
lean_dec(v_h__5_219_);
v_head_229_ = lean_ctor_get(v_x_214_, 0);
if (lean_obj_tag(v_head_229_) == 0)
{
lean_object* v_tail_230_; lean_object* v___x_231_; 
lean_dec(v_h__3_217_);
v_tail_230_ = lean_ctor_get(v_x_214_, 1);
lean_inc(v_tail_230_);
lean_dec_ref_known(v_x_214_, 2);
v___x_231_ = lean_apply_3(v_h__2_216_, v_x_213_, v_tail_230_, lean_box(0));
return v___x_231_;
}
else
{
lean_object* v_tail_232_; lean_object* v_val_233_; lean_object* v_tail_234_; lean_object* v_val_235_; lean_object* v___x_236_; 
lean_inc_ref(v_head_229_);
lean_inc_ref(v_head_226_);
lean_dec(v_h__2_216_);
v_tail_232_ = lean_ctor_get(v_x_213_, 1);
lean_inc(v_tail_232_);
lean_dec_ref_known(v_x_213_, 2);
v_val_233_ = lean_ctor_get(v_head_226_, 0);
lean_inc(v_val_233_);
lean_dec_ref_known(v_head_226_, 1);
v_tail_234_ = lean_ctor_get(v_x_214_, 1);
lean_inc(v_tail_234_);
lean_dec_ref_known(v_x_214_, 2);
v_val_235_ = lean_ctor_get(v_head_229_, 0);
lean_inc(v_val_235_);
lean_dec_ref_known(v_head_229_, 1);
v___x_236_ = lean_apply_4(v_h__3_217_, v_val_233_, v_tail_232_, v_val_235_, v_tail_234_);
return v___x_236_;
}
}
else
{
lean_object* v___x_237_; 
lean_dec(v_h__3_217_);
lean_dec(v_h__2_216_);
v___x_237_ = lean_apply_6(v_h__5_219_, v_x_213_, v_x_214_, lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_237_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg___lam__0(lean_object* v_k_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_){
_start:
{
lean_object* v___x_248_; 
lean_inc(v___y_242_);
lean_inc_ref(v___y_241_);
lean_inc(v___y_240_);
lean_inc_ref(v___y_239_);
v___x_248_ = lean_apply_9(v_k_238_, v___y_239_, v___y_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_, v___y_245_, v___y_246_, lean_box(0));
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg___lam__0___boxed(lean_object* v_k_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg___lam__0(v_k_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_);
lean_dec(v___y_253_);
lean_dec_ref(v___y_252_);
lean_dec(v___y_251_);
lean_dec_ref(v___y_250_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg(lean_object* v_k_260_, uint8_t v_allowLevelAssignments_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_){
_start:
{
lean_object* v___f_271_; lean_object* v___x_272_; 
lean_inc(v___y_265_);
lean_inc_ref(v___y_264_);
lean_inc(v___y_263_);
lean_inc_ref(v___y_262_);
v___f_271_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_271_, 0, v_k_260_);
lean_closure_set(v___f_271_, 1, v___y_262_);
lean_closure_set(v___f_271_, 2, v___y_263_);
lean_closure_set(v___f_271_, 3, v___y_264_);
lean_closure_set(v___f_271_, 4, v___y_265_);
v___x_272_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_261_, v___f_271_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
if (lean_obj_tag(v___x_272_) == 0)
{
return v___x_272_;
}
else
{
lean_object* v_a_273_; lean_object* v___x_275_; uint8_t v_isShared_276_; uint8_t v_isSharedCheck_280_; 
v_a_273_ = lean_ctor_get(v___x_272_, 0);
v_isSharedCheck_280_ = !lean_is_exclusive(v___x_272_);
if (v_isSharedCheck_280_ == 0)
{
v___x_275_ = v___x_272_;
v_isShared_276_ = v_isSharedCheck_280_;
goto v_resetjp_274_;
}
else
{
lean_inc(v_a_273_);
lean_dec(v___x_272_);
v___x_275_ = lean_box(0);
v_isShared_276_ = v_isSharedCheck_280_;
goto v_resetjp_274_;
}
v_resetjp_274_:
{
lean_object* v___x_278_; 
if (v_isShared_276_ == 0)
{
v___x_278_ = v___x_275_;
goto v_reusejp_277_;
}
else
{
lean_object* v_reuseFailAlloc_279_; 
v_reuseFailAlloc_279_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_279_, 0, v_a_273_);
v___x_278_ = v_reuseFailAlloc_279_;
goto v_reusejp_277_;
}
v_reusejp_277_:
{
return v___x_278_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg___boxed(lean_object* v_k_281_, lean_object* v_allowLevelAssignments_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_292_; lean_object* v_res_293_; 
v_allowLevelAssignments_boxed_292_ = lean_unbox(v_allowLevelAssignments_282_);
v_res_293_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg(v_k_281_, v_allowLevelAssignments_boxed_292_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_, v___y_290_);
lean_dec(v___y_290_);
lean_dec_ref(v___y_289_);
lean_dec(v___y_288_);
lean_dec_ref(v___y_287_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
return v_res_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1(lean_object* v_00_u03b1_294_, lean_object* v_k_295_, uint8_t v_allowLevelAssignments_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg(v_k_295_, v_allowLevelAssignments_296_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_, v___y_304_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___boxed(lean_object* v_00_u03b1_307_, lean_object* v_k_308_, lean_object* v_allowLevelAssignments_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_319_; lean_object* v_res_320_; 
v_allowLevelAssignments_boxed_319_ = lean_unbox(v_allowLevelAssignments_309_);
v_res_320_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1(v_00_u03b1_307_, v_k_308_, v_allowLevelAssignments_boxed_319_, v___y_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_, v___y_317_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
lean_dec(v___y_315_);
lean_dec_ref(v___y_314_);
lean_dec(v___y_313_);
lean_dec_ref(v___y_312_);
lean_dec(v___y_311_);
lean_dec_ref(v___y_310_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg___lam__0(lean_object* v_x_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_){
_start:
{
lean_object* v___x_331_; 
lean_inc(v___y_325_);
lean_inc_ref(v___y_324_);
lean_inc(v___y_323_);
lean_inc_ref(v___y_322_);
v___x_331_ = lean_apply_9(v_x_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_, v___y_329_, lean_box(0));
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg___lam__0___boxed(lean_object* v_x_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_){
_start:
{
lean_object* v_res_342_; 
v_res_342_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg___lam__0(v_x_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_, v___y_337_, v___y_338_, v___y_339_, v___y_340_);
lean_dec(v___y_336_);
lean_dec_ref(v___y_335_);
lean_dec(v___y_334_);
lean_dec_ref(v___y_333_);
return v_res_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg(lean_object* v_mvarId_343_, lean_object* v_x_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_){
_start:
{
lean_object* v___f_354_; lean_object* v___x_355_; 
lean_inc(v___y_348_);
lean_inc_ref(v___y_347_);
lean_inc(v___y_346_);
lean_inc_ref(v___y_345_);
v___f_354_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_354_, 0, v_x_344_);
lean_closure_set(v___f_354_, 1, v___y_345_);
lean_closure_set(v___f_354_, 2, v___y_346_);
lean_closure_set(v___f_354_, 3, v___y_347_);
lean_closure_set(v___f_354_, 4, v___y_348_);
v___x_355_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_343_, v___f_354_, v___y_349_, v___y_350_, v___y_351_, v___y_352_);
if (lean_obj_tag(v___x_355_) == 0)
{
return v___x_355_;
}
else
{
lean_object* v_a_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_363_; 
v_a_356_ = lean_ctor_get(v___x_355_, 0);
v_isSharedCheck_363_ = !lean_is_exclusive(v___x_355_);
if (v_isSharedCheck_363_ == 0)
{
v___x_358_ = v___x_355_;
v_isShared_359_ = v_isSharedCheck_363_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_a_356_);
lean_dec(v___x_355_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_363_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
lean_object* v___x_361_; 
if (v_isShared_359_ == 0)
{
v___x_361_ = v___x_358_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v_a_356_);
v___x_361_ = v_reuseFailAlloc_362_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
return v___x_361_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg___boxed(lean_object* v_mvarId_364_, lean_object* v_x_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_){
_start:
{
lean_object* v_res_375_; 
v_res_375_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg(v_mvarId_364_, v_x_365_, v___y_366_, v___y_367_, v___y_368_, v___y_369_, v___y_370_, v___y_371_, v___y_372_, v___y_373_);
lean_dec(v___y_373_);
lean_dec_ref(v___y_372_);
lean_dec(v___y_371_);
lean_dec_ref(v___y_370_);
lean_dec(v___y_369_);
lean_dec_ref(v___y_368_);
lean_dec(v___y_367_);
lean_dec_ref(v___y_366_);
return v_res_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2(lean_object* v_00_u03b1_376_, lean_object* v_mvarId_377_, lean_object* v_x_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg(v_mvarId_377_, v_x_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___boxed(lean_object* v_00_u03b1_389_, lean_object* v_mvarId_390_, lean_object* v_x_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_){
_start:
{
lean_object* v_res_401_; 
v_res_401_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2(v_00_u03b1_389_, v_mvarId_390_, v_x_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
lean_dec(v___y_397_);
lean_dec_ref(v___y_396_);
lean_dec(v___y_395_);
lean_dec_ref(v___y_394_);
lean_dec(v___y_393_);
lean_dec_ref(v___y_392_);
return v_res_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0_spec__0(lean_object* v_msgData_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_){
_start:
{
lean_object* v___x_408_; lean_object* v_env_409_; lean_object* v___x_410_; lean_object* v_mctx_411_; lean_object* v_lctx_412_; lean_object* v_options_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_408_ = lean_st_ref_get(v___y_406_);
v_env_409_ = lean_ctor_get(v___x_408_, 0);
lean_inc_ref(v_env_409_);
lean_dec(v___x_408_);
v___x_410_ = lean_st_ref_get(v___y_404_);
v_mctx_411_ = lean_ctor_get(v___x_410_, 0);
lean_inc_ref(v_mctx_411_);
lean_dec(v___x_410_);
v_lctx_412_ = lean_ctor_get(v___y_403_, 2);
v_options_413_ = lean_ctor_get(v___y_405_, 2);
lean_inc_ref(v_options_413_);
lean_inc_ref(v_lctx_412_);
v___x_414_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_414_, 0, v_env_409_);
lean_ctor_set(v___x_414_, 1, v_mctx_411_);
lean_ctor_set(v___x_414_, 2, v_lctx_412_);
lean_ctor_set(v___x_414_, 3, v_options_413_);
v___x_415_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_415_, 0, v___x_414_);
lean_ctor_set(v___x_415_, 1, v_msgData_402_);
v___x_416_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_416_, 0, v___x_415_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0_spec__0___boxed(lean_object* v_msgData_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_){
_start:
{
lean_object* v_res_423_; 
v_res_423_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0_spec__0(v_msgData_417_, v___y_418_, v___y_419_, v___y_420_, v___y_421_);
lean_dec(v___y_421_);
lean_dec_ref(v___y_420_);
lean_dec(v___y_419_);
lean_dec_ref(v___y_418_);
return v_res_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___redArg(lean_object* v_msg_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_){
_start:
{
lean_object* v_ref_430_; lean_object* v___x_431_; lean_object* v_a_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_440_; 
v_ref_430_ = lean_ctor_get(v___y_427_, 5);
v___x_431_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0_spec__0(v_msg_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
v_a_432_ = lean_ctor_get(v___x_431_, 0);
v_isSharedCheck_440_ = !lean_is_exclusive(v___x_431_);
if (v_isSharedCheck_440_ == 0)
{
v___x_434_ = v___x_431_;
v_isShared_435_ = v_isSharedCheck_440_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_a_432_);
lean_dec(v___x_431_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_440_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v___x_436_; lean_object* v___x_438_; 
lean_inc(v_ref_430_);
v___x_436_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_436_, 0, v_ref_430_);
lean_ctor_set(v___x_436_, 1, v_a_432_);
if (v_isShared_435_ == 0)
{
lean_ctor_set_tag(v___x_434_, 1);
lean_ctor_set(v___x_434_, 0, v___x_436_);
v___x_438_ = v___x_434_;
goto v_reusejp_437_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v___x_436_);
v___x_438_ = v_reuseFailAlloc_439_;
goto v_reusejp_437_;
}
v_reusejp_437_:
{
return v___x_438_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___redArg___boxed(lean_object* v_msg_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_){
_start:
{
lean_object* v_res_447_; 
v_res_447_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___redArg(v_msg_441_, v___y_442_, v___y_443_, v___y_444_, v___y_445_);
lean_dec(v___y_445_);
lean_dec_ref(v___y_444_);
lean_dec(v___y_443_);
lean_dec_ref(v___y_442_);
return v_res_447_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1(void){
_start:
{
lean_object* v___x_449_; lean_object* v___x_450_; 
v___x_449_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__0));
v___x_450_ = l_Lean_stringToMessageData(v___x_449_);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0(lean_object* v_x_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_){
_start:
{
lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v_a_463_; lean_object* v___x_465_; uint8_t v_isShared_466_; uint8_t v_isSharedCheck_470_; 
v___x_461_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1);
v___x_462_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___redArg(v___x_461_, v___y_456_, v___y_457_, v___y_458_, v___y_459_);
v_a_463_ = lean_ctor_get(v___x_462_, 0);
v_isSharedCheck_470_ = !lean_is_exclusive(v___x_462_);
if (v_isSharedCheck_470_ == 0)
{
v___x_465_ = v___x_462_;
v_isShared_466_ = v_isSharedCheck_470_;
goto v_resetjp_464_;
}
else
{
lean_inc(v_a_463_);
lean_dec(v___x_462_);
v___x_465_ = lean_box(0);
v_isShared_466_ = v_isSharedCheck_470_;
goto v_resetjp_464_;
}
v_resetjp_464_:
{
lean_object* v___x_468_; 
if (v_isShared_466_ == 0)
{
v___x_468_ = v___x_465_;
goto v_reusejp_467_;
}
else
{
lean_object* v_reuseFailAlloc_469_; 
v_reuseFailAlloc_469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_469_, 0, v_a_463_);
v___x_468_ = v_reuseFailAlloc_469_;
goto v_reusejp_467_;
}
v_reusejp_467_:
{
return v___x_468_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___boxed(lean_object* v_x_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_){
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0(v_x_471_, v___y_472_, v___y_473_, v___y_474_, v___y_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_);
lean_dec(v___y_479_);
lean_dec_ref(v___y_478_);
lean_dec(v___y_477_);
lean_dec_ref(v___y_476_);
lean_dec(v___y_475_);
lean_dec_ref(v___y_474_);
lean_dec(v___y_473_);
lean_dec_ref(v___y_472_);
lean_dec(v_x_471_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__1(uint8_t v___x_482_, lean_object* v___x_483_, lean_object* v___x_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_){
_start:
{
lean_object* v_keyedConfig_494_; uint8_t v_trackZetaDelta_495_; lean_object* v_zetaDeltaSet_496_; lean_object* v_lctx_497_; lean_object* v_localInstances_498_; lean_object* v_defEqCtx_x3f_499_; lean_object* v_synthPendingDepth_500_; lean_object* v_customCanUnfoldPredicate_x3f_501_; uint8_t v_univApprox_502_; uint8_t v_inTypeClassResolution_503_; uint8_t v_cacheInferType_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_521_; 
v_keyedConfig_494_ = lean_ctor_get(v___y_489_, 0);
v_trackZetaDelta_495_ = lean_ctor_get_uint8(v___y_489_, sizeof(void*)*7);
v_zetaDeltaSet_496_ = lean_ctor_get(v___y_489_, 1);
v_lctx_497_ = lean_ctor_get(v___y_489_, 2);
v_localInstances_498_ = lean_ctor_get(v___y_489_, 3);
v_defEqCtx_x3f_499_ = lean_ctor_get(v___y_489_, 4);
v_synthPendingDepth_500_ = lean_ctor_get(v___y_489_, 5);
v_customCanUnfoldPredicate_x3f_501_ = lean_ctor_get(v___y_489_, 6);
v_univApprox_502_ = lean_ctor_get_uint8(v___y_489_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_503_ = lean_ctor_get_uint8(v___y_489_, sizeof(void*)*7 + 2);
v_cacheInferType_504_ = lean_ctor_get_uint8(v___y_489_, sizeof(void*)*7 + 3);
v_isSharedCheck_521_ = !lean_is_exclusive(v___y_489_);
if (v_isSharedCheck_521_ == 0)
{
v___x_506_ = v___y_489_;
v_isShared_507_ = v_isSharedCheck_521_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_501_);
lean_inc(v_synthPendingDepth_500_);
lean_inc(v_defEqCtx_x3f_499_);
lean_inc(v_localInstances_498_);
lean_inc(v_lctx_497_);
lean_inc(v_zetaDeltaSet_496_);
lean_inc(v_keyedConfig_494_);
lean_dec(v___y_489_);
v___x_506_ = lean_box(0);
v_isShared_507_ = v_isSharedCheck_521_;
goto v_resetjp_505_;
}
v_resetjp_505_:
{
lean_object* v___x_508_; lean_object* v___x_510_; 
v___x_508_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_482_, v_keyedConfig_494_);
if (v_isShared_507_ == 0)
{
lean_ctor_set(v___x_506_, 0, v___x_508_);
v___x_510_ = v___x_506_;
goto v_reusejp_509_;
}
else
{
lean_object* v_reuseFailAlloc_520_; 
v_reuseFailAlloc_520_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_520_, 0, v___x_508_);
lean_ctor_set(v_reuseFailAlloc_520_, 1, v_zetaDeltaSet_496_);
lean_ctor_set(v_reuseFailAlloc_520_, 2, v_lctx_497_);
lean_ctor_set(v_reuseFailAlloc_520_, 3, v_localInstances_498_);
lean_ctor_set(v_reuseFailAlloc_520_, 4, v_defEqCtx_x3f_499_);
lean_ctor_set(v_reuseFailAlloc_520_, 5, v_synthPendingDepth_500_);
lean_ctor_set(v_reuseFailAlloc_520_, 6, v_customCanUnfoldPredicate_x3f_501_);
lean_ctor_set_uint8(v_reuseFailAlloc_520_, sizeof(void*)*7, v_trackZetaDelta_495_);
lean_ctor_set_uint8(v_reuseFailAlloc_520_, sizeof(void*)*7 + 1, v_univApprox_502_);
lean_ctor_set_uint8(v_reuseFailAlloc_520_, sizeof(void*)*7 + 2, v_inTypeClassResolution_503_);
lean_ctor_set_uint8(v_reuseFailAlloc_520_, sizeof(void*)*7 + 3, v_cacheInferType_504_);
v___x_510_ = v_reuseFailAlloc_520_;
goto v_reusejp_509_;
}
v_reusejp_509_:
{
lean_object* v___x_511_; 
v___x_511_ = lp_mathlib_Mathlib_Tactic_lctxIsDefEq(v___x_483_, v___x_484_, v___x_510_, v___y_490_, v___y_491_, v___y_492_);
lean_dec_ref(v___x_510_);
if (lean_obj_tag(v___x_511_) == 0)
{
lean_object* v_a_512_; lean_object* v___x_514_; uint8_t v_isShared_515_; uint8_t v_isSharedCheck_519_; 
v_a_512_ = lean_ctor_get(v___x_511_, 0);
v_isSharedCheck_519_ = !lean_is_exclusive(v___x_511_);
if (v_isSharedCheck_519_ == 0)
{
v___x_514_ = v___x_511_;
v_isShared_515_ = v_isSharedCheck_519_;
goto v_resetjp_513_;
}
else
{
lean_inc(v_a_512_);
lean_dec(v___x_511_);
v___x_514_ = lean_box(0);
v_isShared_515_ = v_isSharedCheck_519_;
goto v_resetjp_513_;
}
v_resetjp_513_:
{
lean_object* v___x_517_; 
if (v_isShared_515_ == 0)
{
v___x_517_ = v___x_514_;
goto v_reusejp_516_;
}
else
{
lean_object* v_reuseFailAlloc_518_; 
v_reuseFailAlloc_518_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_518_, 0, v_a_512_);
v___x_517_ = v_reuseFailAlloc_518_;
goto v_reusejp_516_;
}
v_reusejp_516_:
{
return v___x_517_;
}
}
}
else
{
return v___x_511_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__1___boxed(lean_object* v___x_522_, lean_object* v___x_523_, lean_object* v___x_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_){
_start:
{
uint8_t v___x_13022__boxed_534_; lean_object* v_res_535_; 
v___x_13022__boxed_534_ = lean_unbox(v___x_522_);
v_res_535_ = lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__1(v___x_13022__boxed_534_, v___x_523_, v___x_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_, v___y_529_, v___y_530_, v___y_531_, v___y_532_);
lean_dec(v___y_532_);
lean_dec_ref(v___y_531_);
lean_dec(v___y_530_);
lean_dec(v___y_528_);
lean_dec_ref(v___y_527_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_525_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__2(uint8_t v___x_536_, lean_object* v_a_537_, lean_object* v_a_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_){
_start:
{
lean_object* v_keyedConfig_548_; uint8_t v_trackZetaDelta_549_; lean_object* v_zetaDeltaSet_550_; lean_object* v_lctx_551_; lean_object* v_localInstances_552_; lean_object* v_defEqCtx_x3f_553_; lean_object* v_synthPendingDepth_554_; lean_object* v_customCanUnfoldPredicate_x3f_555_; uint8_t v_univApprox_556_; uint8_t v_inTypeClassResolution_557_; uint8_t v_cacheInferType_558_; lean_object* v___x_560_; uint8_t v_isShared_561_; uint8_t v_isSharedCheck_575_; 
v_keyedConfig_548_ = lean_ctor_get(v___y_543_, 0);
v_trackZetaDelta_549_ = lean_ctor_get_uint8(v___y_543_, sizeof(void*)*7);
v_zetaDeltaSet_550_ = lean_ctor_get(v___y_543_, 1);
v_lctx_551_ = lean_ctor_get(v___y_543_, 2);
v_localInstances_552_ = lean_ctor_get(v___y_543_, 3);
v_defEqCtx_x3f_553_ = lean_ctor_get(v___y_543_, 4);
v_synthPendingDepth_554_ = lean_ctor_get(v___y_543_, 5);
v_customCanUnfoldPredicate_x3f_555_ = lean_ctor_get(v___y_543_, 6);
v_univApprox_556_ = lean_ctor_get_uint8(v___y_543_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_557_ = lean_ctor_get_uint8(v___y_543_, sizeof(void*)*7 + 2);
v_cacheInferType_558_ = lean_ctor_get_uint8(v___y_543_, sizeof(void*)*7 + 3);
v_isSharedCheck_575_ = !lean_is_exclusive(v___y_543_);
if (v_isSharedCheck_575_ == 0)
{
v___x_560_ = v___y_543_;
v_isShared_561_ = v_isSharedCheck_575_;
goto v_resetjp_559_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_555_);
lean_inc(v_synthPendingDepth_554_);
lean_inc(v_defEqCtx_x3f_553_);
lean_inc(v_localInstances_552_);
lean_inc(v_lctx_551_);
lean_inc(v_zetaDeltaSet_550_);
lean_inc(v_keyedConfig_548_);
lean_dec(v___y_543_);
v___x_560_ = lean_box(0);
v_isShared_561_ = v_isSharedCheck_575_;
goto v_resetjp_559_;
}
v_resetjp_559_:
{
lean_object* v___x_562_; lean_object* v___x_564_; 
v___x_562_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_536_, v_keyedConfig_548_);
if (v_isShared_561_ == 0)
{
lean_ctor_set(v___x_560_, 0, v___x_562_);
v___x_564_ = v___x_560_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_574_; 
v_reuseFailAlloc_574_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_574_, 0, v___x_562_);
lean_ctor_set(v_reuseFailAlloc_574_, 1, v_zetaDeltaSet_550_);
lean_ctor_set(v_reuseFailAlloc_574_, 2, v_lctx_551_);
lean_ctor_set(v_reuseFailAlloc_574_, 3, v_localInstances_552_);
lean_ctor_set(v_reuseFailAlloc_574_, 4, v_defEqCtx_x3f_553_);
lean_ctor_set(v_reuseFailAlloc_574_, 5, v_synthPendingDepth_554_);
lean_ctor_set(v_reuseFailAlloc_574_, 6, v_customCanUnfoldPredicate_x3f_555_);
lean_ctor_set_uint8(v_reuseFailAlloc_574_, sizeof(void*)*7, v_trackZetaDelta_549_);
lean_ctor_set_uint8(v_reuseFailAlloc_574_, sizeof(void*)*7 + 1, v_univApprox_556_);
lean_ctor_set_uint8(v_reuseFailAlloc_574_, sizeof(void*)*7 + 2, v_inTypeClassResolution_557_);
lean_ctor_set_uint8(v_reuseFailAlloc_574_, sizeof(void*)*7 + 3, v_cacheInferType_558_);
v___x_564_ = v_reuseFailAlloc_574_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
lean_object* v___x_565_; 
v___x_565_ = l_Lean_Meta_isExprDefEq(v_a_537_, v_a_538_, v___x_564_, v___y_544_, v___y_545_, v___y_546_);
lean_dec_ref(v___x_564_);
if (lean_obj_tag(v___x_565_) == 0)
{
lean_object* v_a_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_573_; 
v_a_566_ = lean_ctor_get(v___x_565_, 0);
v_isSharedCheck_573_ = !lean_is_exclusive(v___x_565_);
if (v_isSharedCheck_573_ == 0)
{
v___x_568_ = v___x_565_;
v_isShared_569_ = v_isSharedCheck_573_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_a_566_);
lean_dec(v___x_565_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_573_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v___x_571_; 
if (v_isShared_569_ == 0)
{
v___x_571_ = v___x_568_;
goto v_reusejp_570_;
}
else
{
lean_object* v_reuseFailAlloc_572_; 
v_reuseFailAlloc_572_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_572_, 0, v_a_566_);
v___x_571_ = v_reuseFailAlloc_572_;
goto v_reusejp_570_;
}
v_reusejp_570_:
{
return v___x_571_;
}
}
}
else
{
return v___x_565_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__2___boxed(lean_object* v___x_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_){
_start:
{
uint8_t v___x_13093__boxed_588_; lean_object* v_res_589_; 
v___x_13093__boxed_588_ = lean_unbox(v___x_576_);
v_res_589_ = lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__2(v___x_13093__boxed_588_, v_a_577_, v_a_578_, v___y_579_, v___y_580_, v___y_581_, v___y_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v___y_584_);
lean_dec(v___y_582_);
lean_dec_ref(v___y_581_);
lean_dec(v___y_580_);
lean_dec_ref(v___y_579_);
return v_res_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__3(lean_object* v_goal_590_, lean_object* v_head_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_){
_start:
{
lean_object* v___x_601_; 
lean_inc(v_goal_590_);
v___x_601_ = l_Lean_MVarId_getDecl(v_goal_590_, v___y_596_, v___y_597_, v___y_598_, v___y_599_);
if (lean_obj_tag(v___x_601_) == 0)
{
lean_object* v_a_602_; lean_object* v___x_603_; 
v_a_602_ = lean_ctor_get(v___x_601_, 0);
lean_inc(v_a_602_);
lean_dec_ref_known(v___x_601_, 1);
lean_inc(v_head_591_);
v___x_603_ = l_Lean_MVarId_getDecl(v_head_591_, v___y_596_, v___y_597_, v___y_598_, v___y_599_);
if (lean_obj_tag(v___x_603_) == 0)
{
lean_object* v_lctx_604_; lean_object* v_a_605_; lean_object* v_lctx_606_; lean_object* v_decls_607_; lean_object* v_decls_608_; lean_object* v___x_609_; lean_object* v___x_610_; uint8_t v___x_611_; lean_object* v___x_612_; lean_object* v___f_613_; uint8_t v___x_614_; lean_object* v___x_659_; 
v_lctx_604_ = lean_ctor_get(v_a_602_, 1);
lean_inc_ref(v_lctx_604_);
lean_dec(v_a_602_);
v_a_605_ = lean_ctor_get(v___x_603_, 0);
lean_inc(v_a_605_);
lean_dec_ref_known(v___x_603_, 1);
v_lctx_606_ = lean_ctor_get(v_a_605_, 1);
lean_inc_ref(v_lctx_606_);
lean_dec(v_a_605_);
v_decls_607_ = lean_ctor_get(v_lctx_604_, 1);
lean_inc_ref(v_decls_607_);
lean_dec_ref(v_lctx_604_);
v_decls_608_ = lean_ctor_get(v_lctx_606_, 1);
lean_inc_ref(v_decls_608_);
lean_dec_ref(v_lctx_606_);
v___x_609_ = l_Lean_PersistentArray_toList___redArg(v_decls_607_);
lean_dec_ref(v_decls_607_);
v___x_610_ = l_Lean_PersistentArray_toList___redArg(v_decls_608_);
lean_dec_ref(v_decls_608_);
v___x_611_ = 2;
v___x_612_ = lean_box(v___x_611_);
v___f_613_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__1___boxed), 12, 3);
lean_closure_set(v___f_613_, 0, v___x_612_);
lean_closure_set(v___f_613_, 1, v___x_609_);
lean_closure_set(v___f_613_, 2, v___x_610_);
v___x_614_ = 0;
v___x_659_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg(v___f_613_, v___x_614_, v___y_592_, v___y_593_, v___y_594_, v___y_595_, v___y_596_, v___y_597_, v___y_598_, v___y_599_);
if (lean_obj_tag(v___x_659_) == 0)
{
lean_object* v_a_660_; uint8_t v___x_661_; 
v_a_660_ = lean_ctor_get(v___x_659_, 0);
lean_inc(v_a_660_);
lean_dec_ref_known(v___x_659_, 1);
v___x_661_ = lean_unbox(v_a_660_);
lean_dec(v_a_660_);
if (v___x_661_ == 0)
{
lean_object* v___x_662_; lean_object* v___x_663_; 
lean_dec(v_head_591_);
lean_dec(v_goal_590_);
v___x_662_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1);
v___x_663_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___redArg(v___x_662_, v___y_596_, v___y_597_, v___y_598_, v___y_599_);
return v___x_663_;
}
else
{
goto v___jp_615_;
}
}
else
{
lean_object* v_a_664_; lean_object* v___x_666_; uint8_t v_isShared_667_; uint8_t v_isSharedCheck_671_; 
lean_dec(v_head_591_);
lean_dec(v_goal_590_);
v_a_664_ = lean_ctor_get(v___x_659_, 0);
v_isSharedCheck_671_ = !lean_is_exclusive(v___x_659_);
if (v_isSharedCheck_671_ == 0)
{
v___x_666_ = v___x_659_;
v_isShared_667_ = v_isSharedCheck_671_;
goto v_resetjp_665_;
}
else
{
lean_inc(v_a_664_);
lean_dec(v___x_659_);
v___x_666_ = lean_box(0);
v_isShared_667_ = v_isSharedCheck_671_;
goto v_resetjp_665_;
}
v_resetjp_665_:
{
lean_object* v___x_669_; 
if (v_isShared_667_ == 0)
{
v___x_669_ = v___x_666_;
goto v_reusejp_668_;
}
else
{
lean_object* v_reuseFailAlloc_670_; 
v_reuseFailAlloc_670_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_670_, 0, v_a_664_);
v___x_669_ = v_reuseFailAlloc_670_;
goto v_reusejp_668_;
}
v_reusejp_668_:
{
return v___x_669_;
}
}
}
v___jp_615_:
{
lean_object* v___x_616_; 
v___x_616_ = l_Lean_MVarId_getType(v_head_591_, v___y_596_, v___y_597_, v___y_598_, v___y_599_);
if (lean_obj_tag(v___x_616_) == 0)
{
lean_object* v_a_617_; lean_object* v___x_618_; 
v_a_617_ = lean_ctor_get(v___x_616_, 0);
lean_inc(v_a_617_);
lean_dec_ref_known(v___x_616_, 1);
v___x_618_ = l_Lean_MVarId_getType(v_goal_590_, v___y_596_, v___y_597_, v___y_598_, v___y_599_);
if (lean_obj_tag(v___x_618_) == 0)
{
lean_object* v_a_619_; lean_object* v___x_620_; lean_object* v___f_621_; lean_object* v___x_622_; 
v_a_619_ = lean_ctor_get(v___x_618_, 0);
lean_inc(v_a_619_);
lean_dec_ref_known(v___x_618_, 1);
v___x_620_ = lean_box(v___x_611_);
v___f_621_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__2___boxed), 12, 3);
lean_closure_set(v___f_621_, 0, v___x_620_);
lean_closure_set(v___f_621_, 1, v_a_617_);
lean_closure_set(v___f_621_, 2, v_a_619_);
v___x_622_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__1___redArg(v___f_621_, v___x_614_, v___y_592_, v___y_593_, v___y_594_, v___y_595_, v___y_596_, v___y_597_, v___y_598_, v___y_599_);
if (lean_obj_tag(v___x_622_) == 0)
{
lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_634_; 
v_a_623_ = lean_ctor_get(v___x_622_, 0);
v_isSharedCheck_634_ = !lean_is_exclusive(v___x_622_);
if (v_isSharedCheck_634_ == 0)
{
v___x_625_ = v___x_622_;
v_isShared_626_ = v_isSharedCheck_634_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_622_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_634_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
uint8_t v___x_627_; 
v___x_627_ = lean_unbox(v_a_623_);
lean_dec(v_a_623_);
if (v___x_627_ == 0)
{
lean_object* v___x_628_; lean_object* v___x_629_; 
lean_del_object(v___x_625_);
v___x_628_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0___closed__1);
v___x_629_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___redArg(v___x_628_, v___y_596_, v___y_597_, v___y_598_, v___y_599_);
return v___x_629_;
}
else
{
lean_object* v___x_630_; lean_object* v___x_632_; 
v___x_630_ = lean_box(0);
if (v_isShared_626_ == 0)
{
lean_ctor_set(v___x_625_, 0, v___x_630_);
v___x_632_ = v___x_625_;
goto v_reusejp_631_;
}
else
{
lean_object* v_reuseFailAlloc_633_; 
v_reuseFailAlloc_633_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_633_, 0, v___x_630_);
v___x_632_ = v_reuseFailAlloc_633_;
goto v_reusejp_631_;
}
v_reusejp_631_:
{
return v___x_632_;
}
}
}
}
else
{
lean_object* v_a_635_; lean_object* v___x_637_; uint8_t v_isShared_638_; uint8_t v_isSharedCheck_642_; 
v_a_635_ = lean_ctor_get(v___x_622_, 0);
v_isSharedCheck_642_ = !lean_is_exclusive(v___x_622_);
if (v_isSharedCheck_642_ == 0)
{
v___x_637_ = v___x_622_;
v_isShared_638_ = v_isSharedCheck_642_;
goto v_resetjp_636_;
}
else
{
lean_inc(v_a_635_);
lean_dec(v___x_622_);
v___x_637_ = lean_box(0);
v_isShared_638_ = v_isSharedCheck_642_;
goto v_resetjp_636_;
}
v_resetjp_636_:
{
lean_object* v___x_640_; 
if (v_isShared_638_ == 0)
{
v___x_640_ = v___x_637_;
goto v_reusejp_639_;
}
else
{
lean_object* v_reuseFailAlloc_641_; 
v_reuseFailAlloc_641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_641_, 0, v_a_635_);
v___x_640_ = v_reuseFailAlloc_641_;
goto v_reusejp_639_;
}
v_reusejp_639_:
{
return v___x_640_;
}
}
}
}
else
{
lean_object* v_a_643_; lean_object* v___x_645_; uint8_t v_isShared_646_; uint8_t v_isSharedCheck_650_; 
lean_dec(v_a_617_);
v_a_643_ = lean_ctor_get(v___x_618_, 0);
v_isSharedCheck_650_ = !lean_is_exclusive(v___x_618_);
if (v_isSharedCheck_650_ == 0)
{
v___x_645_ = v___x_618_;
v_isShared_646_ = v_isSharedCheck_650_;
goto v_resetjp_644_;
}
else
{
lean_inc(v_a_643_);
lean_dec(v___x_618_);
v___x_645_ = lean_box(0);
v_isShared_646_ = v_isSharedCheck_650_;
goto v_resetjp_644_;
}
v_resetjp_644_:
{
lean_object* v___x_648_; 
if (v_isShared_646_ == 0)
{
v___x_648_ = v___x_645_;
goto v_reusejp_647_;
}
else
{
lean_object* v_reuseFailAlloc_649_; 
v_reuseFailAlloc_649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_649_, 0, v_a_643_);
v___x_648_ = v_reuseFailAlloc_649_;
goto v_reusejp_647_;
}
v_reusejp_647_:
{
return v___x_648_;
}
}
}
}
else
{
lean_object* v_a_651_; lean_object* v___x_653_; uint8_t v_isShared_654_; uint8_t v_isSharedCheck_658_; 
lean_dec(v_goal_590_);
v_a_651_ = lean_ctor_get(v___x_616_, 0);
v_isSharedCheck_658_ = !lean_is_exclusive(v___x_616_);
if (v_isSharedCheck_658_ == 0)
{
v___x_653_ = v___x_616_;
v_isShared_654_ = v_isSharedCheck_658_;
goto v_resetjp_652_;
}
else
{
lean_inc(v_a_651_);
lean_dec(v___x_616_);
v___x_653_ = lean_box(0);
v_isShared_654_ = v_isSharedCheck_658_;
goto v_resetjp_652_;
}
v_resetjp_652_:
{
lean_object* v___x_656_; 
if (v_isShared_654_ == 0)
{
v___x_656_ = v___x_653_;
goto v_reusejp_655_;
}
else
{
lean_object* v_reuseFailAlloc_657_; 
v_reuseFailAlloc_657_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_657_, 0, v_a_651_);
v___x_656_ = v_reuseFailAlloc_657_;
goto v_reusejp_655_;
}
v_reusejp_655_:
{
return v___x_656_;
}
}
}
}
}
else
{
lean_object* v_a_672_; lean_object* v___x_674_; uint8_t v_isShared_675_; uint8_t v_isSharedCheck_679_; 
lean_dec(v_a_602_);
lean_dec(v_head_591_);
lean_dec(v_goal_590_);
v_a_672_ = lean_ctor_get(v___x_603_, 0);
v_isSharedCheck_679_ = !lean_is_exclusive(v___x_603_);
if (v_isSharedCheck_679_ == 0)
{
v___x_674_ = v___x_603_;
v_isShared_675_ = v_isSharedCheck_679_;
goto v_resetjp_673_;
}
else
{
lean_inc(v_a_672_);
lean_dec(v___x_603_);
v___x_674_ = lean_box(0);
v_isShared_675_ = v_isSharedCheck_679_;
goto v_resetjp_673_;
}
v_resetjp_673_:
{
lean_object* v___x_677_; 
if (v_isShared_675_ == 0)
{
v___x_677_ = v___x_674_;
goto v_reusejp_676_;
}
else
{
lean_object* v_reuseFailAlloc_678_; 
v_reuseFailAlloc_678_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_678_, 0, v_a_672_);
v___x_677_ = v_reuseFailAlloc_678_;
goto v_reusejp_676_;
}
v_reusejp_676_:
{
return v___x_677_;
}
}
}
}
else
{
lean_object* v_a_680_; lean_object* v___x_682_; uint8_t v_isShared_683_; uint8_t v_isSharedCheck_687_; 
lean_dec(v_head_591_);
lean_dec(v_goal_590_);
v_a_680_ = lean_ctor_get(v___x_601_, 0);
v_isSharedCheck_687_ = !lean_is_exclusive(v___x_601_);
if (v_isSharedCheck_687_ == 0)
{
v___x_682_ = v___x_601_;
v_isShared_683_ = v_isSharedCheck_687_;
goto v_resetjp_681_;
}
else
{
lean_inc(v_a_680_);
lean_dec(v___x_601_);
v___x_682_ = lean_box(0);
v_isShared_683_ = v_isSharedCheck_687_;
goto v_resetjp_681_;
}
v_resetjp_681_:
{
lean_object* v___x_685_; 
if (v_isShared_683_ == 0)
{
v___x_685_ = v___x_682_;
goto v_reusejp_684_;
}
else
{
lean_object* v_reuseFailAlloc_686_; 
v_reuseFailAlloc_686_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_686_, 0, v_a_680_);
v___x_685_ = v_reuseFailAlloc_686_;
goto v_reusejp_684_;
}
v_reusejp_684_:
{
return v___x_685_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__3___boxed(lean_object* v_goal_688_, lean_object* v_head_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_){
_start:
{
lean_object* v_res_699_; 
v_res_699_ = lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__3(v_goal_688_, v_head_689_, v___y_690_, v___y_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_, v___y_696_, v___y_697_);
lean_dec(v___y_697_);
lean_dec_ref(v___y_696_);
lean_dec(v___y_695_);
lean_dec_ref(v___y_694_);
lean_dec(v___y_693_);
lean_dec_ref(v___y_692_);
lean_dec(v___y_691_);
lean_dec_ref(v___y_690_);
return v_res_699_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___closed__1(void){
_start:
{
lean_object* v___x_701_; lean_object* v___x_702_; 
v___x_701_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___closed__0));
v___x_702_ = l_Lean_stringToMessageData(v___x_701_);
return v___x_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress(lean_object* v_goal_703_, lean_object* v_tacs_704_, lean_object* v_a_705_, lean_object* v_a_706_, lean_object* v_a_707_, lean_object* v_a_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_, lean_object* v_a_712_){
_start:
{
lean_object* v___x_719_; 
lean_inc(v_goal_703_);
v___x_719_ = l_Lean_Elab_Tactic_run(v_goal_703_, v_tacs_704_, v_a_707_, v_a_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_);
if (lean_obj_tag(v___x_719_) == 0)
{
lean_object* v_a_720_; lean_object* v___x_721_; 
v_a_720_ = lean_ctor_get(v___x_719_, 0);
lean_inc(v_a_720_);
lean_dec_ref_known(v___x_719_, 1);
v___x_721_ = l_Lean_Elab_Tactic_saveState___redArg(v_a_706_, v_a_708_, v_a_710_, v_a_712_);
if (lean_obj_tag(v___x_721_) == 0)
{
lean_object* v_a_722_; lean_object* v___x_724_; uint8_t v_isShared_725_; uint8_t v_isSharedCheck_772_; 
v_a_722_ = lean_ctor_get(v___x_721_, 0);
v_isSharedCheck_772_ = !lean_is_exclusive(v___x_721_);
if (v_isSharedCheck_772_ == 0)
{
v___x_724_ = v___x_721_;
v_isShared_725_ = v_isSharedCheck_772_;
goto v_resetjp_723_;
}
else
{
lean_inc(v_a_722_);
lean_dec(v___x_721_);
v___x_724_ = lean_box(0);
v_isShared_725_ = v_isSharedCheck_772_;
goto v_resetjp_723_;
}
v_resetjp_723_:
{
lean_object* v___y_727_; uint8_t v___y_728_; lean_object* v_a_750_; lean_object* v___y_754_; 
if (lean_obj_tag(v_a_720_) == 1)
{
lean_object* v_tail_765_; 
v_tail_765_ = lean_ctor_get(v_a_720_, 1);
if (lean_obj_tag(v_tail_765_) == 0)
{
lean_object* v_head_766_; lean_object* v___f_767_; lean_object* v___x_768_; 
v_head_766_ = lean_ctor_get(v_a_720_, 0);
lean_inc(v_head_766_);
lean_inc_n(v_goal_703_, 2);
v___f_767_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__3___boxed), 11, 2);
lean_closure_set(v___f_767_, 0, v_goal_703_);
lean_closure_set(v___f_767_, 1, v_head_766_);
v___x_768_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__2___redArg(v_goal_703_, v___f_767_, v_a_705_, v_a_706_, v_a_707_, v_a_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_);
if (lean_obj_tag(v___x_768_) == 0)
{
lean_dec_ref_known(v___x_768_, 1);
lean_dec_ref_known(v_a_720_, 2);
lean_del_object(v___x_724_);
lean_dec(v_a_722_);
goto v___jp_714_;
}
else
{
lean_object* v_a_769_; 
lean_dec(v_goal_703_);
v_a_769_ = lean_ctor_get(v___x_768_, 0);
lean_inc(v_a_769_);
lean_dec_ref_known(v___x_768_, 1);
v_a_750_ = v_a_769_;
goto v___jp_749_;
}
}
else
{
lean_object* v___x_770_; 
v___x_770_ = lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0(v_a_720_, v_a_705_, v_a_706_, v_a_707_, v_a_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_);
v___y_754_ = v___x_770_;
goto v___jp_753_;
}
}
else
{
lean_object* v___x_771_; 
v___x_771_ = lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___lam__0(v_a_720_, v_a_705_, v_a_706_, v_a_707_, v_a_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_);
v___y_754_ = v___x_771_;
goto v___jp_753_;
}
v___jp_726_:
{
if (v___y_728_ == 0)
{
lean_object* v___x_729_; 
lean_dec_ref(v___y_727_);
lean_del_object(v___x_724_);
v___x_729_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_722_, v___y_728_, v_a_706_, v_a_707_, v_a_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_);
if (lean_obj_tag(v___x_729_) == 0)
{
lean_object* v___x_731_; uint8_t v_isShared_732_; uint8_t v_isSharedCheck_736_; 
v_isSharedCheck_736_ = !lean_is_exclusive(v___x_729_);
if (v_isSharedCheck_736_ == 0)
{
lean_object* v_unused_737_; 
v_unused_737_ = lean_ctor_get(v___x_729_, 0);
lean_dec(v_unused_737_);
v___x_731_ = v___x_729_;
v_isShared_732_ = v_isSharedCheck_736_;
goto v_resetjp_730_;
}
else
{
lean_dec(v___x_729_);
v___x_731_ = lean_box(0);
v_isShared_732_ = v_isSharedCheck_736_;
goto v_resetjp_730_;
}
v_resetjp_730_:
{
lean_object* v___x_734_; 
if (v_isShared_732_ == 0)
{
lean_ctor_set(v___x_731_, 0, v_a_720_);
v___x_734_ = v___x_731_;
goto v_reusejp_733_;
}
else
{
lean_object* v_reuseFailAlloc_735_; 
v_reuseFailAlloc_735_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_735_, 0, v_a_720_);
v___x_734_ = v_reuseFailAlloc_735_;
goto v_reusejp_733_;
}
v_reusejp_733_:
{
return v___x_734_;
}
}
}
else
{
lean_object* v_a_738_; lean_object* v___x_740_; uint8_t v_isShared_741_; uint8_t v_isSharedCheck_745_; 
lean_dec(v_a_720_);
v_a_738_ = lean_ctor_get(v___x_729_, 0);
v_isSharedCheck_745_ = !lean_is_exclusive(v___x_729_);
if (v_isSharedCheck_745_ == 0)
{
v___x_740_ = v___x_729_;
v_isShared_741_ = v_isSharedCheck_745_;
goto v_resetjp_739_;
}
else
{
lean_inc(v_a_738_);
lean_dec(v___x_729_);
v___x_740_ = lean_box(0);
v_isShared_741_ = v_isSharedCheck_745_;
goto v_resetjp_739_;
}
v_resetjp_739_:
{
lean_object* v___x_743_; 
if (v_isShared_741_ == 0)
{
v___x_743_ = v___x_740_;
goto v_reusejp_742_;
}
else
{
lean_object* v_reuseFailAlloc_744_; 
v_reuseFailAlloc_744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_744_, 0, v_a_738_);
v___x_743_ = v_reuseFailAlloc_744_;
goto v_reusejp_742_;
}
v_reusejp_742_:
{
return v___x_743_;
}
}
}
}
else
{
lean_object* v___x_747_; 
lean_dec(v_a_722_);
lean_dec(v_a_720_);
if (v_isShared_725_ == 0)
{
lean_ctor_set_tag(v___x_724_, 1);
lean_ctor_set(v___x_724_, 0, v___y_727_);
v___x_747_ = v___x_724_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v___y_727_);
v___x_747_ = v_reuseFailAlloc_748_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
return v___x_747_;
}
}
}
v___jp_749_:
{
uint8_t v___x_751_; 
v___x_751_ = l_Lean_Exception_isInterrupt(v_a_750_);
if (v___x_751_ == 0)
{
uint8_t v___x_752_; 
lean_inc_ref(v_a_750_);
v___x_752_ = l_Lean_Exception_isRuntime(v_a_750_);
v___y_727_ = v_a_750_;
v___y_728_ = v___x_752_;
goto v___jp_726_;
}
else
{
v___y_727_ = v_a_750_;
v___y_728_ = v___x_751_;
goto v___jp_726_;
}
}
v___jp_753_:
{
if (lean_obj_tag(v___y_754_) == 0)
{
lean_object* v_a_755_; lean_object* v___x_757_; uint8_t v_isShared_758_; uint8_t v_isSharedCheck_763_; 
lean_del_object(v___x_724_);
lean_dec(v_a_722_);
lean_dec(v_a_720_);
v_a_755_ = lean_ctor_get(v___y_754_, 0);
v_isSharedCheck_763_ = !lean_is_exclusive(v___y_754_);
if (v_isSharedCheck_763_ == 0)
{
v___x_757_ = v___y_754_;
v_isShared_758_ = v_isSharedCheck_763_;
goto v_resetjp_756_;
}
else
{
lean_inc(v_a_755_);
lean_dec(v___y_754_);
v___x_757_ = lean_box(0);
v_isShared_758_ = v_isSharedCheck_763_;
goto v_resetjp_756_;
}
v_resetjp_756_:
{
if (lean_obj_tag(v_a_755_) == 0)
{
lean_object* v_a_759_; lean_object* v___x_761_; 
lean_dec(v_goal_703_);
v_a_759_ = lean_ctor_get(v_a_755_, 0);
lean_inc(v_a_759_);
lean_dec_ref_known(v_a_755_, 1);
if (v_isShared_758_ == 0)
{
lean_ctor_set(v___x_757_, 0, v_a_759_);
v___x_761_ = v___x_757_;
goto v_reusejp_760_;
}
else
{
lean_object* v_reuseFailAlloc_762_; 
v_reuseFailAlloc_762_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_762_, 0, v_a_759_);
v___x_761_ = v_reuseFailAlloc_762_;
goto v_reusejp_760_;
}
v_reusejp_760_:
{
return v___x_761_;
}
}
else
{
lean_dec_ref_known(v_a_755_, 1);
lean_del_object(v___x_757_);
goto v___jp_714_;
}
}
}
else
{
lean_object* v_a_764_; 
lean_dec(v_goal_703_);
v_a_764_ = lean_ctor_get(v___y_754_, 0);
lean_inc(v_a_764_);
lean_dec_ref_known(v___y_754_, 1);
v_a_750_ = v_a_764_;
goto v___jp_749_;
}
}
}
}
else
{
lean_object* v_a_773_; lean_object* v___x_775_; uint8_t v_isShared_776_; uint8_t v_isSharedCheck_780_; 
lean_dec(v_a_720_);
lean_dec(v_goal_703_);
v_a_773_ = lean_ctor_get(v___x_721_, 0);
v_isSharedCheck_780_ = !lean_is_exclusive(v___x_721_);
if (v_isSharedCheck_780_ == 0)
{
v___x_775_ = v___x_721_;
v_isShared_776_ = v_isSharedCheck_780_;
goto v_resetjp_774_;
}
else
{
lean_inc(v_a_773_);
lean_dec(v___x_721_);
v___x_775_ = lean_box(0);
v_isShared_776_ = v_isSharedCheck_780_;
goto v_resetjp_774_;
}
v_resetjp_774_:
{
lean_object* v___x_778_; 
if (v_isShared_776_ == 0)
{
v___x_778_ = v___x_775_;
goto v_reusejp_777_;
}
else
{
lean_object* v_reuseFailAlloc_779_; 
v_reuseFailAlloc_779_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_779_, 0, v_a_773_);
v___x_778_ = v_reuseFailAlloc_779_;
goto v_reusejp_777_;
}
v_reusejp_777_:
{
return v___x_778_;
}
}
}
}
else
{
lean_dec(v_goal_703_);
return v___x_719_;
}
v___jp_714_:
{
lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; 
v___x_715_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___closed__1, &lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___closed__1);
v___x_716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_716_, 0, v_goal_703_);
v___x_717_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_717_, 0, v___x_715_);
lean_ctor_set(v___x_717_, 1, v___x_716_);
v___x_718_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___redArg(v___x_717_, v_a_709_, v_a_710_, v_a_711_, v_a_712_);
return v___x_718_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress___boxed(lean_object* v_goal_781_, lean_object* v_tacs_782_, lean_object* v_a_783_, lean_object* v_a_784_, lean_object* v_a_785_, lean_object* v_a_786_, lean_object* v_a_787_, lean_object* v_a_788_, lean_object* v_a_789_, lean_object* v_a_790_, lean_object* v_a_791_){
_start:
{
lean_object* v_res_792_; 
v_res_792_ = lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress(v_goal_781_, v_tacs_782_, v_a_783_, v_a_784_, v_a_785_, v_a_786_, v_a_787_, v_a_788_, v_a_789_, v_a_790_);
lean_dec(v_a_790_);
lean_dec_ref(v_a_789_);
lean_dec(v_a_788_);
lean_dec_ref(v_a_787_);
lean_dec(v_a_786_);
lean_dec_ref(v_a_785_);
lean_dec(v_a_784_);
lean_dec_ref(v_a_783_);
return v_res_792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0(lean_object* v_00_u03b1_793_, lean_object* v_msg_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_){
_start:
{
lean_object* v___x_804_; 
v___x_804_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___redArg(v_msg_794_, v___y_799_, v___y_800_, v___y_801_, v___y_802_);
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0___boxed(lean_object* v_00_u03b1_805_, lean_object* v_msg_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_){
_start:
{
lean_object* v_res_816_; 
v_res_816_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runAndFailIfNoProgress_spec__0(v_00_u03b1_805_, v_msg_806_, v___y_807_, v___y_808_, v___y_809_, v___y_810_, v___y_811_, v___y_812_, v___y_813_, v___y_814_);
lean_dec(v___y_814_);
lean_dec_ref(v___y_813_);
lean_dec(v___y_812_);
lean_dec_ref(v___y_811_);
lean_dec(v___y_810_);
lean_dec_ref(v___y_809_);
lean_dec(v___y_808_);
lean_dec_ref(v___y_807_);
return v_res_816_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_817_ = lean_box(0);
v___x_818_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_819_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_819_, 0, v___x_818_);
lean_ctor_set(v___x_819_, 1, v___x_817_);
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg(){
_start:
{
lean_object* v___x_821_; lean_object* v___x_822_; 
v___x_821_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg___closed__0);
v___x_822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_822_, 0, v___x_821_);
return v___x_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg___boxed(lean_object* v___y_823_){
_start:
{
lean_object* v_res_824_; 
v_res_824_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg();
return v_res_824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0(lean_object* v_00_u03b1_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_){
_start:
{
lean_object* v___x_835_; 
v___x_835_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg();
return v___x_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___boxed(lean_object* v_00_u03b1_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_){
_start:
{
lean_object* v_res_846_; 
v_res_846_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0(v_00_u03b1_836_, v___y_837_, v___y_838_, v___y_839_, v___y_840_, v___y_841_, v___y_842_, v___y_843_, v___y_844_);
lean_dec(v___y_844_);
lean_dec_ref(v___y_843_);
lean_dec(v___y_842_);
lean_dec_ref(v___y_841_);
lean_dec(v___y_840_);
lean_dec_ref(v___y_839_);
lean_dec(v___y_838_);
lean_dec_ref(v___y_837_);
return v_res_846_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1(lean_object* v_x_847_, lean_object* v_a_848_, lean_object* v_a_849_, lean_object* v_a_850_, lean_object* v_a_851_, lean_object* v_a_852_, lean_object* v_a_853_, lean_object* v_a_854_, lean_object* v_a_855_){
_start:
{
lean_object* v___x_857_; uint8_t v___x_858_; 
v___x_857_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_failIfNoProgress___closed__3));
lean_inc(v_x_847_);
v___x_858_ = l_Lean_Syntax_isOfKind(v_x_847_, v___x_857_);
if (v___x_858_ == 0)
{
lean_object* v___x_859_; 
lean_dec(v_x_847_);
v___x_859_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1_spec__0___redArg();
return v___x_859_;
}
else
{
lean_object* v___x_860_; 
v___x_860_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v_a_849_, v_a_852_, v_a_853_, v_a_854_, v_a_855_);
if (lean_obj_tag(v___x_860_) == 0)
{
lean_object* v_a_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; 
v_a_861_ = lean_ctor_get(v___x_860_, 0);
lean_inc(v_a_861_);
lean_dec_ref_known(v___x_860_, 1);
v___x_862_ = lean_unsigned_to_nat(1u);
v___x_863_ = l_Lean_Syntax_getArg(v_x_847_, v___x_862_);
lean_dec(v_x_847_);
v___x_864_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_864_, 0, v___x_863_);
v___x_865_ = lp_mathlib_Mathlib_Tactic_runAndFailIfNoProgress(v_a_861_, v___x_864_, v_a_848_, v_a_849_, v_a_850_, v_a_851_, v_a_852_, v_a_853_, v_a_854_, v_a_855_);
if (lean_obj_tag(v___x_865_) == 0)
{
lean_object* v_a_866_; lean_object* v___x_867_; 
v_a_866_ = lean_ctor_get(v___x_865_, 0);
lean_inc(v_a_866_);
lean_dec_ref_known(v___x_865_, 1);
v___x_867_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_866_, v_a_849_, v_a_852_, v_a_853_, v_a_854_, v_a_855_);
return v___x_867_;
}
else
{
lean_object* v_a_868_; lean_object* v___x_870_; uint8_t v_isShared_871_; uint8_t v_isSharedCheck_875_; 
v_a_868_ = lean_ctor_get(v___x_865_, 0);
v_isSharedCheck_875_ = !lean_is_exclusive(v___x_865_);
if (v_isSharedCheck_875_ == 0)
{
v___x_870_ = v___x_865_;
v_isShared_871_ = v_isSharedCheck_875_;
goto v_resetjp_869_;
}
else
{
lean_inc(v_a_868_);
lean_dec(v___x_865_);
v___x_870_ = lean_box(0);
v_isShared_871_ = v_isSharedCheck_875_;
goto v_resetjp_869_;
}
v_resetjp_869_:
{
lean_object* v___x_873_; 
if (v_isShared_871_ == 0)
{
v___x_873_ = v___x_870_;
goto v_reusejp_872_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v_a_868_);
v___x_873_ = v_reuseFailAlloc_874_;
goto v_reusejp_872_;
}
v_reusejp_872_:
{
return v___x_873_;
}
}
}
}
else
{
lean_object* v_a_876_; lean_object* v___x_878_; uint8_t v_isShared_879_; uint8_t v_isSharedCheck_883_; 
lean_dec(v_x_847_);
v_a_876_ = lean_ctor_get(v___x_860_, 0);
v_isSharedCheck_883_ = !lean_is_exclusive(v___x_860_);
if (v_isSharedCheck_883_ == 0)
{
v___x_878_ = v___x_860_;
v_isShared_879_ = v_isSharedCheck_883_;
goto v_resetjp_877_;
}
else
{
lean_inc(v_a_876_);
lean_dec(v___x_860_);
v___x_878_ = lean_box(0);
v_isShared_879_ = v_isSharedCheck_883_;
goto v_resetjp_877_;
}
v_resetjp_877_:
{
lean_object* v___x_881_; 
if (v_isShared_879_ == 0)
{
v___x_881_ = v___x_878_;
goto v_reusejp_880_;
}
else
{
lean_object* v_reuseFailAlloc_882_; 
v_reuseFailAlloc_882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_882_, 0, v_a_876_);
v___x_881_ = v_reuseFailAlloc_882_;
goto v_reusejp_880_;
}
v_reusejp_880_:
{
return v___x_881_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1___boxed(lean_object* v_x_884_, lean_object* v_a_885_, lean_object* v_a_886_, lean_object* v_a_887_, lean_object* v_a_888_, lean_object* v_a_889_, lean_object* v_a_890_, lean_object* v_a_891_, lean_object* v_a_892_, lean_object* v_a_893_){
_start:
{
lean_object* v_res_894_; 
v_res_894_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__FailIfNoProgress______elabRules__Mathlib__Tactic__failIfNoProgress__1(v_x_884_, v_a_885_, v_a_886_, v_a_887_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v_a_892_);
lean_dec(v_a_892_);
lean_dec_ref(v_a_891_);
lean_dec(v_a_890_);
lean_dec_ref(v_a_889_);
lean_dec(v_a_888_);
lean_dec_ref(v_a_887_);
lean_dec(v_a_886_);
lean_dec_ref(v_a_885_);
return v_res_894_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Util(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Util(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(builtin);
}
#ifdef __cplusplus
}
#endif
