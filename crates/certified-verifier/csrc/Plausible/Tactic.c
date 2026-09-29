// Lean compiler output
// Module: Plausible.Tactic
// Imports: public import Init public meta import Init public meta import Plausible.Testable
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_Level_ofNat(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_evalExpr___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_admitGoal(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_Decorations_addDecorations(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_mkApp10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
size_t lean_array_size(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_config;
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_elabConfig___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_revert(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_mkOptionalNode(lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_string_object lp_plausible_plausibleSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "plausibleSyntax"};
static const lean_object* lp_plausible_plausibleSyntax___closed__0 = (const lean_object*)&lp_plausible_plausibleSyntax___closed__0_value;
static const lean_ctor_object lp_plausible_plausibleSyntax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_plausibleSyntax___closed__0_value),LEAN_SCALAR_PTR_LITERAL(66, 195, 60, 67, 37, 37, 76, 3)}};
static const lean_object* lp_plausible_plausibleSyntax___closed__1 = (const lean_object*)&lp_plausible_plausibleSyntax___closed__1_value;
static const lean_string_object lp_plausible_plausibleSyntax___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_plausible_plausibleSyntax___closed__2 = (const lean_object*)&lp_plausible_plausibleSyntax___closed__2_value;
static const lean_ctor_object lp_plausible_plausibleSyntax___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_plausibleSyntax___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_plausible_plausibleSyntax___closed__3 = (const lean_object*)&lp_plausible_plausibleSyntax___closed__3_value;
static const lean_string_object lp_plausible_plausibleSyntax___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "plausible"};
static const lean_object* lp_plausible_plausibleSyntax___closed__4 = (const lean_object*)&lp_plausible_plausibleSyntax___closed__4_value;
static const lean_ctor_object lp_plausible_plausibleSyntax___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_plausible_plausibleSyntax___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_plausible_plausibleSyntax___closed__5 = (const lean_object*)&lp_plausible_plausibleSyntax___closed__5_value;
static const lean_string_object lp_plausible_plausibleSyntax___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_plausible_plausibleSyntax___closed__6 = (const lean_object*)&lp_plausible_plausibleSyntax___closed__6_value;
static const lean_ctor_object lp_plausible_plausibleSyntax___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_plausibleSyntax___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_plausible_plausibleSyntax___closed__7 = (const lean_object*)&lp_plausible_plausibleSyntax___closed__7_value;
static lean_once_cell_t lp_plausible_plausibleSyntax___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_plausibleSyntax___closed__8;
static lean_once_cell_t lp_plausible_plausibleSyntax___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_plausibleSyntax___closed__9;
static lean_once_cell_t lp_plausible_plausibleSyntax___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_plausibleSyntax___closed__10;
LEAN_EXPORT lean_object* lp_plausible_plausibleSyntax;
static const lean_string_object lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__0_value;
static const lean_string_object lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Core"};
static const lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__1_value;
static const lean_string_object lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "CoreM"};
static const lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__2 = (const lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__2_value;
static const lean_ctor_object lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__3_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(194, 126, 120, 188, 150, 235, 117, 203)}};
static const lean_ctor_object lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__3_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(115, 114, 191, 177, 45, 189, 121, 141)}};
static const lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__3 = (const lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__3_value;
static lean_once_cell_t lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__4;
static const lean_string_object lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "PUnit"};
static const lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__5 = (const lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__5_value;
static const lean_ctor_object lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(23, 153, 158, 141, 176, 162, 235, 153)}};
static const lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__6 = (const lean_object*)&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__6_value;
static lean_once_cell_t lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__7;
static lean_once_cell_t lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__8;
static lean_once_cell_t lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__9;
static lean_once_cell_t lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__10;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__0;
static const lean_string_object lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__1 = (const lean_object*)&lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__1_value;
static const lean_array_object lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__2 = (const lean_object*)&lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "decoration"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__0 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__0_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_plausibleSyntax___closed__4_value),LEAN_SCALAR_PTR_LITERAL(116, 81, 155, 51, 136, 153, 194, 244)}};
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__1_value_aux_0),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(142, 255, 214, 58, 49, 108, 229, 224)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__1 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__1_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__2 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__2_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__3 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__3_value;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__4;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "[testable decoration]\n  "};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__5 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__5_value;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__6;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__7 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__7_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__8 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__8_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9_value_aux_0),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__10 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__10_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11_value_aux_0),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(22, 245, 194, 28, 184, 9, 113, 128)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__12 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__12_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__12_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__13 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__13_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Option"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__14 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__14_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__15 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__15_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__14_value),LEAN_SCALAR_PTR_LITERAL(95, 234, 177, 188, 3, 226, 91, 252)}};
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__16_value_aux_0),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(149, 114, 34, 228, 75, 195, 143, 131)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__16 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__16_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "some"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__17 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__17_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__14_value),LEAN_SCALAR_PTR_LITERAL(95, 234, 177, 188, 3, 226, 91, 252)}};
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__18_value_aux_0),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__17_value),LEAN_SCALAR_PTR_LITERAL(89, 148, 40, 55, 221, 242, 231, 67)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__18 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__18_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "check"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__19 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__19_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Configuration"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__20 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__20_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__21 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__21_value;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__22;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__23;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Failed to create a `testable` instance for `"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__24 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__24_value;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__25;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 482, .m_capacity = 482, .m_length = 481, .m_data = "`.\nWhat to do:\n1. make sure that the types you are using have `Plausible.SampleableExt` instances\n (you can use `#sample my_type` if you are unsure);\n2. make sure that the relations and predicates that your proposition use are decidable;\n3. if your hypothesis is big consider increasing `set_option synthInstance.maxSize` to a\n    \n  higher power of two\n    \n4. make sure that instances of `Plausible.Testable` exist that, when combined,\n  apply to your decorated proposition:\n```\n"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__26 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__26_value;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__27;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 175, .m_capacity = 175, .m_length = 174, .m_data = "\n```\n\nUse `set_option trace.Meta.synthInstance true` to understand what instances are missing.\n\nTry this:\nset_option trace.Meta.synthInstance true\n#synth Plausible.Testable ("};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__28 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__28_value;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__29;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__30 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__30_value;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__31;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Plausible"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__32 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__32_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Testable"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__33 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__33_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__32_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__34_value_aux_0),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__33_value),LEAN_SCALAR_PTR_LITERAL(113, 137, 77, 13, 23, 202, 166, 43)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__34 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__34_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "candidates"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__35 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__35_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "shrink"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__36 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__36_value;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "steps"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__37 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__37_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_plausibleSyntax___closed__4_value),LEAN_SCALAR_PTR_LITERAL(116, 81, 155, 51, 136, 153, 194, 244)}};
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__38_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__38_value_aux_0),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__36_value),LEAN_SCALAR_PTR_LITERAL(155, 130, 163, 150, 112, 34, 123, 238)}};
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__38_value_aux_1),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__37_value),LEAN_SCALAR_PTR_LITERAL(88, 60, 4, 42, 29, 42, 104, 5)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__38 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__38_value;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__39;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "success"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__40 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__40_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_plausibleSyntax___closed__4_value),LEAN_SCALAR_PTR_LITERAL(116, 81, 155, 51, 136, 153, 194, 244)}};
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__41_value_aux_0),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__40_value),LEAN_SCALAR_PTR_LITERAL(63, 186, 199, 35, 41, 50, 129, 51)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__41 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__41_value;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__42;
static const lean_string_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "discarded"};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__43 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__43_value;
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_plausibleSyntax___closed__4_value),LEAN_SCALAR_PTR_LITERAL(116, 81, 155, 51, 136, 153, 194, 244)}};
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__44_value_aux_0),((lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__43_value),LEAN_SCALAR_PTR_LITERAL(24, 146, 225, 240, 2, 63, 149, 0)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__44 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__44_value;
static lean_once_cell_t lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__45;
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__8(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_plausible_Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1___closed__0 = (const lean_object*)&lp_plausible_Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(10) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___closed__0 = (const lean_object*)&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_plausible_plausibleSyntax___closed__8(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_14_ = l_Lean_Parser_Tactic_config;
v___x_15_ = ((lean_object*)(lp_plausible_plausibleSyntax___closed__7));
v___x_16_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_16_, 0, v___x_15_);
lean_ctor_set(v___x_16_, 1, v___x_14_);
return v___x_16_;
}
}
static lean_object* _init_lp_plausible_plausibleSyntax___closed__9(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_17_ = lean_obj_once(&lp_plausible_plausibleSyntax___closed__8, &lp_plausible_plausibleSyntax___closed__8_once, _init_lp_plausible_plausibleSyntax___closed__8);
v___x_18_ = ((lean_object*)(lp_plausible_plausibleSyntax___closed__5));
v___x_19_ = ((lean_object*)(lp_plausible_plausibleSyntax___closed__3));
v___x_20_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_20_, 0, v___x_19_);
lean_ctor_set(v___x_20_, 1, v___x_18_);
lean_ctor_set(v___x_20_, 2, v___x_17_);
return v___x_20_;
}
}
static lean_object* _init_lp_plausible_plausibleSyntax___closed__10(void){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_21_ = lean_obj_once(&lp_plausible_plausibleSyntax___closed__9, &lp_plausible_plausibleSyntax___closed__9_once, _init_lp_plausible_plausibleSyntax___closed__9);
v___x_22_ = lean_unsigned_to_nat(1022u);
v___x_23_ = ((lean_object*)(lp_plausible_plausibleSyntax___closed__1));
v___x_24_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_24_, 0, v___x_23_);
lean_ctor_set(v___x_24_, 1, v___x_22_);
lean_ctor_set(v___x_24_, 2, v___x_21_);
return v___x_24_;
}
}
static lean_object* _init_lp_plausible_plausibleSyntax(void){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_obj_once(&lp_plausible_plausibleSyntax___closed__10, &lp_plausible_plausibleSyntax___closed__10_once, _init_lp_plausible_plausibleSyntax___closed__10);
return v___x_25_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__4(void){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_33_ = lean_box(0);
v___x_34_ = ((lean_object*)(lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__3));
v___x_35_ = l_Lean_mkConst(v___x_34_, v___x_33_);
return v___x_35_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__7(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_39_ = lean_unsigned_to_nat(1u);
v___x_40_ = l_Lean_Level_ofNat(v___x_39_);
return v___x_40_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__8(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_41_ = lean_box(0);
v___x_42_ = lean_obj_once(&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__7, &lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__7_once, _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__7);
v___x_43_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_43_, 0, v___x_42_);
lean_ctor_set(v___x_43_, 1, v___x_41_);
return v___x_43_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__9(void){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_44_ = lean_obj_once(&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__8, &lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__8_once, _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__8);
v___x_45_ = ((lean_object*)(lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__6));
v___x_46_ = l_Lean_mkConst(v___x_45_, v___x_44_);
return v___x_46_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__10(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_47_ = lean_obj_once(&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__9, &lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__9_once, _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__9);
v___x_48_ = lean_obj_once(&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__4, &lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__4_once, _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__4);
v___x_49_ = l_Lean_Expr_app___override(v___x_48_, v___x_47_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1(lean_object* v_e_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_){
_start:
{
lean_object* v___x_56_; uint8_t v___x_57_; uint8_t v___x_58_; lean_object* v___x_59_; 
v___x_56_ = lean_obj_once(&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__10, &lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__10_once, _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__10);
v___x_57_ = 1;
v___x_58_ = 1;
v___x_59_ = l_Lean_Meta_evalExpr___redArg(v___x_56_, v_e_50_, v___x_57_, v___x_58_, v_a_51_, v_a_52_, v_a_53_, v_a_54_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___boxed(lean_object* v_e_60_, lean_object* v_a_61_, lean_object* v_a_62_, lean_object* v_a_63_, lean_object* v_a_64_, lean_object* v_a_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1(v_e_60_, v_a_61_, v_a_62_, v_a_63_, v_a_64_);
lean_dec(v_a_64_);
lean_dec_ref(v_a_63_);
lean_dec(v_a_62_);
lean_dec_ref(v_a_61_);
return v_res_66_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_67_ = lean_box(0);
v___x_68_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_69_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
lean_ctor_set(v___x_69_, 1, v___x_67_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg(){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_71_ = lean_obj_once(&lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg___closed__0, &lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg___closed__0_once, _init_lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg___closed__0);
v___x_72_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg___boxed(lean_object* v___y_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg();
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0(lean_object* v_00_u03b1_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg();
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___boxed(lean_object* v_00_u03b1_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0(v_00_u03b1_86_, v___y_87_, v___y_88_, v___y_89_, v___y_90_, v___y_91_, v___y_92_, v___y_93_, v___y_94_);
lean_dec(v___y_94_);
lean_dec_ref(v___y_93_);
lean_dec(v___y_92_);
lean_dec_ref(v___y_91_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
lean_dec(v___y_88_);
lean_dec_ref(v___y_87_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg___lam__0(lean_object* v_x_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
lean_object* v___x_107_; 
lean_inc(v___y_101_);
lean_inc_ref(v___y_100_);
lean_inc(v___y_99_);
lean_inc_ref(v___y_98_);
v___x_107_ = lean_apply_9(v_x_97_, v___y_98_, v___y_99_, v___y_100_, v___y_101_, v___y_102_, v___y_103_, v___y_104_, v___y_105_, lean_box(0));
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg___lam__0___boxed(lean_object* v_x_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg___lam__0(v_x_108_, v___y_109_, v___y_110_, v___y_111_, v___y_112_, v___y_113_, v___y_114_, v___y_115_, v___y_116_);
lean_dec(v___y_112_);
lean_dec_ref(v___y_111_);
lean_dec(v___y_110_);
lean_dec_ref(v___y_109_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg(lean_object* v_mvarId_119_, lean_object* v_x_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v___f_130_; lean_object* v___x_131_; 
lean_inc(v___y_124_);
lean_inc_ref(v___y_123_);
lean_inc(v___y_122_);
lean_inc_ref(v___y_121_);
v___f_130_ = lean_alloc_closure((void*)(lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_130_, 0, v_x_120_);
lean_closure_set(v___f_130_, 1, v___y_121_);
lean_closure_set(v___f_130_, 2, v___y_122_);
lean_closure_set(v___f_130_, 3, v___y_123_);
lean_closure_set(v___f_130_, 4, v___y_124_);
v___x_131_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_119_, v___f_130_, v___y_125_, v___y_126_, v___y_127_, v___y_128_);
if (lean_obj_tag(v___x_131_) == 0)
{
return v___x_131_;
}
else
{
lean_object* v_a_132_; lean_object* v___x_134_; uint8_t v_isShared_135_; uint8_t v_isSharedCheck_139_; 
v_a_132_ = lean_ctor_get(v___x_131_, 0);
v_isSharedCheck_139_ = !lean_is_exclusive(v___x_131_);
if (v_isSharedCheck_139_ == 0)
{
v___x_134_ = v___x_131_;
v_isShared_135_ = v_isSharedCheck_139_;
goto v_resetjp_133_;
}
else
{
lean_inc(v_a_132_);
lean_dec(v___x_131_);
v___x_134_ = lean_box(0);
v_isShared_135_ = v_isSharedCheck_139_;
goto v_resetjp_133_;
}
v_resetjp_133_:
{
lean_object* v___x_137_; 
if (v_isShared_135_ == 0)
{
v___x_137_ = v___x_134_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v_a_132_);
v___x_137_ = v_reuseFailAlloc_138_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
return v___x_137_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg___boxed(lean_object* v_mvarId_140_, lean_object* v_x_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg(v_mvarId_140_, v_x_141_, v___y_142_, v___y_143_, v___y_144_, v___y_145_, v___y_146_, v___y_147_, v___y_148_, v___y_149_);
lean_dec(v___y_149_);
lean_dec_ref(v___y_148_);
lean_dec(v___y_147_);
lean_dec_ref(v___y_146_);
lean_dec(v___y_145_);
lean_dec_ref(v___y_144_);
lean_dec(v___y_143_);
lean_dec_ref(v___y_142_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5(lean_object* v_00_u03b1_152_, lean_object* v_mvarId_153_, lean_object* v_x_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg(v_mvarId_153_, v_x_154_, v___y_155_, v___y_156_, v___y_157_, v___y_158_, v___y_159_, v___y_160_, v___y_161_, v___y_162_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___boxed(lean_object* v_00_u03b1_165_, lean_object* v_mvarId_166_, lean_object* v_x_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5(v_00_u03b1_165_, v_mvarId_166_, v_x_167_, v___y_168_, v___y_169_, v___y_170_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_);
lean_dec(v___y_175_);
lean_dec_ref(v___y_174_);
lean_dec(v___y_173_);
lean_dec_ref(v___y_172_);
lean_dec(v___y_171_);
lean_dec_ref(v___y_170_);
lean_dec(v___y_169_);
lean_dec_ref(v___y_168_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3_spec__4(lean_object* v_msgData_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_){
_start:
{
lean_object* v___x_184_; lean_object* v_env_185_; lean_object* v___x_186_; lean_object* v_mctx_187_; lean_object* v_lctx_188_; lean_object* v_options_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; 
v___x_184_ = lean_st_ref_get(v___y_182_);
v_env_185_ = lean_ctor_get(v___x_184_, 0);
lean_inc_ref(v_env_185_);
lean_dec(v___x_184_);
v___x_186_ = lean_st_ref_get(v___y_180_);
v_mctx_187_ = lean_ctor_get(v___x_186_, 0);
lean_inc_ref(v_mctx_187_);
lean_dec(v___x_186_);
v_lctx_188_ = lean_ctor_get(v___y_179_, 2);
v_options_189_ = lean_ctor_get(v___y_181_, 2);
lean_inc_ref(v_options_189_);
lean_inc_ref(v_lctx_188_);
v___x_190_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_190_, 0, v_env_185_);
lean_ctor_set(v___x_190_, 1, v_mctx_187_);
lean_ctor_set(v___x_190_, 2, v_lctx_188_);
lean_ctor_set(v___x_190_, 3, v_options_189_);
v___x_191_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v_msgData_178_);
v___x_192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_192_, 0, v___x_191_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3_spec__4___boxed(lean_object* v_msgData_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3_spec__4(v_msgData_193_, v___y_194_, v___y_195_, v___y_196_, v___y_197_);
lean_dec(v___y_197_);
lean_dec_ref(v___y_196_);
lean_dec(v___y_195_);
lean_dec_ref(v___y_194_);
return v_res_199_;
}
}
static double _init_lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_200_; double v___x_201_; 
v___x_200_ = lean_unsigned_to_nat(0u);
v___x_201_ = lean_float_of_nat(v___x_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg(lean_object* v_cls_205_, lean_object* v_msg_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_){
_start:
{
lean_object* v_ref_212_; lean_object* v___x_213_; lean_object* v_a_214_; lean_object* v___x_216_; uint8_t v_isShared_217_; uint8_t v_isSharedCheck_258_; 
v_ref_212_ = lean_ctor_get(v___y_209_, 5);
v___x_213_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3_spec__4(v_msg_206_, v___y_207_, v___y_208_, v___y_209_, v___y_210_);
v_a_214_ = lean_ctor_get(v___x_213_, 0);
v_isSharedCheck_258_ = !lean_is_exclusive(v___x_213_);
if (v_isSharedCheck_258_ == 0)
{
v___x_216_ = v___x_213_;
v_isShared_217_ = v_isSharedCheck_258_;
goto v_resetjp_215_;
}
else
{
lean_inc(v_a_214_);
lean_dec(v___x_213_);
v___x_216_ = lean_box(0);
v_isShared_217_ = v_isSharedCheck_258_;
goto v_resetjp_215_;
}
v_resetjp_215_:
{
lean_object* v___x_218_; lean_object* v_traceState_219_; lean_object* v_env_220_; lean_object* v_nextMacroScope_221_; lean_object* v_ngen_222_; lean_object* v_auxDeclNGen_223_; lean_object* v_cache_224_; lean_object* v_messages_225_; lean_object* v_infoState_226_; lean_object* v_snapshotTasks_227_; lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_257_; 
v___x_218_ = lean_st_ref_take(v___y_210_);
v_traceState_219_ = lean_ctor_get(v___x_218_, 4);
v_env_220_ = lean_ctor_get(v___x_218_, 0);
v_nextMacroScope_221_ = lean_ctor_get(v___x_218_, 1);
v_ngen_222_ = lean_ctor_get(v___x_218_, 2);
v_auxDeclNGen_223_ = lean_ctor_get(v___x_218_, 3);
v_cache_224_ = lean_ctor_get(v___x_218_, 5);
v_messages_225_ = lean_ctor_get(v___x_218_, 6);
v_infoState_226_ = lean_ctor_get(v___x_218_, 7);
v_snapshotTasks_227_ = lean_ctor_get(v___x_218_, 8);
v_isSharedCheck_257_ = !lean_is_exclusive(v___x_218_);
if (v_isSharedCheck_257_ == 0)
{
v___x_229_ = v___x_218_;
v_isShared_230_ = v_isSharedCheck_257_;
goto v_resetjp_228_;
}
else
{
lean_inc(v_snapshotTasks_227_);
lean_inc(v_infoState_226_);
lean_inc(v_messages_225_);
lean_inc(v_cache_224_);
lean_inc(v_traceState_219_);
lean_inc(v_auxDeclNGen_223_);
lean_inc(v_ngen_222_);
lean_inc(v_nextMacroScope_221_);
lean_inc(v_env_220_);
lean_dec(v___x_218_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_257_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
uint64_t v_tid_231_; lean_object* v_traces_232_; lean_object* v___x_234_; uint8_t v_isShared_235_; uint8_t v_isSharedCheck_256_; 
v_tid_231_ = lean_ctor_get_uint64(v_traceState_219_, sizeof(void*)*1);
v_traces_232_ = lean_ctor_get(v_traceState_219_, 0);
v_isSharedCheck_256_ = !lean_is_exclusive(v_traceState_219_);
if (v_isSharedCheck_256_ == 0)
{
v___x_234_ = v_traceState_219_;
v_isShared_235_ = v_isSharedCheck_256_;
goto v_resetjp_233_;
}
else
{
lean_inc(v_traces_232_);
lean_dec(v_traceState_219_);
v___x_234_ = lean_box(0);
v_isShared_235_ = v_isSharedCheck_256_;
goto v_resetjp_233_;
}
v_resetjp_233_:
{
lean_object* v___x_236_; double v___x_237_; uint8_t v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_246_; 
v___x_236_ = lean_box(0);
v___x_237_ = lean_float_once(&lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__0, &lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__0_once, _init_lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__0);
v___x_238_ = 0;
v___x_239_ = ((lean_object*)(lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__1));
v___x_240_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_240_, 0, v_cls_205_);
lean_ctor_set(v___x_240_, 1, v___x_236_);
lean_ctor_set(v___x_240_, 2, v___x_239_);
lean_ctor_set_float(v___x_240_, sizeof(void*)*3, v___x_237_);
lean_ctor_set_float(v___x_240_, sizeof(void*)*3 + 8, v___x_237_);
lean_ctor_set_uint8(v___x_240_, sizeof(void*)*3 + 16, v___x_238_);
v___x_241_ = ((lean_object*)(lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___closed__2));
v___x_242_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_242_, 0, v___x_240_);
lean_ctor_set(v___x_242_, 1, v_a_214_);
lean_ctor_set(v___x_242_, 2, v___x_241_);
lean_inc(v_ref_212_);
v___x_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_243_, 0, v_ref_212_);
lean_ctor_set(v___x_243_, 1, v___x_242_);
v___x_244_ = l_Lean_PersistentArray_push___redArg(v_traces_232_, v___x_243_);
if (v_isShared_235_ == 0)
{
lean_ctor_set(v___x_234_, 0, v___x_244_);
v___x_246_ = v___x_234_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v___x_244_);
lean_ctor_set_uint64(v_reuseFailAlloc_255_, sizeof(void*)*1, v_tid_231_);
v___x_246_ = v_reuseFailAlloc_255_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
lean_object* v___x_248_; 
if (v_isShared_230_ == 0)
{
lean_ctor_set(v___x_229_, 4, v___x_246_);
v___x_248_ = v___x_229_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v_env_220_);
lean_ctor_set(v_reuseFailAlloc_254_, 1, v_nextMacroScope_221_);
lean_ctor_set(v_reuseFailAlloc_254_, 2, v_ngen_222_);
lean_ctor_set(v_reuseFailAlloc_254_, 3, v_auxDeclNGen_223_);
lean_ctor_set(v_reuseFailAlloc_254_, 4, v___x_246_);
lean_ctor_set(v_reuseFailAlloc_254_, 5, v_cache_224_);
lean_ctor_set(v_reuseFailAlloc_254_, 6, v_messages_225_);
lean_ctor_set(v_reuseFailAlloc_254_, 7, v_infoState_226_);
lean_ctor_set(v_reuseFailAlloc_254_, 8, v_snapshotTasks_227_);
v___x_248_ = v_reuseFailAlloc_254_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_252_; 
v___x_249_ = lean_st_ref_set(v___y_210_, v___x_248_);
v___x_250_ = lean_box(0);
if (v_isShared_217_ == 0)
{
lean_ctor_set(v___x_216_, 0, v___x_250_);
v___x_252_ = v___x_216_;
goto v_reusejp_251_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v___x_250_);
v___x_252_ = v_reuseFailAlloc_253_;
goto v_reusejp_251_;
}
v_reusejp_251_:
{
return v___x_252_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg___boxed(lean_object* v_cls_259_, lean_object* v_msg_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg(v_cls_259_, v_msg_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4___redArg(lean_object* v_msg_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_){
_start:
{
lean_object* v_ref_273_; lean_object* v___x_274_; lean_object* v_a_275_; lean_object* v___x_277_; uint8_t v_isShared_278_; uint8_t v_isSharedCheck_283_; 
v_ref_273_ = lean_ctor_get(v___y_270_, 5);
v___x_274_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3_spec__4(v_msg_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_);
v_a_275_ = lean_ctor_get(v___x_274_, 0);
v_isSharedCheck_283_ = !lean_is_exclusive(v___x_274_);
if (v_isSharedCheck_283_ == 0)
{
v___x_277_ = v___x_274_;
v_isShared_278_ = v_isSharedCheck_283_;
goto v_resetjp_276_;
}
else
{
lean_inc(v_a_275_);
lean_dec(v___x_274_);
v___x_277_ = lean_box(0);
v_isShared_278_ = v_isSharedCheck_283_;
goto v_resetjp_276_;
}
v_resetjp_276_:
{
lean_object* v___x_279_; lean_object* v___x_281_; 
lean_inc(v_ref_273_);
v___x_279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_279_, 0, v_ref_273_);
lean_ctor_set(v___x_279_, 1, v_a_275_);
if (v_isShared_278_ == 0)
{
lean_ctor_set_tag(v___x_277_, 1);
lean_ctor_set(v___x_277_, 0, v___x_279_);
v___x_281_ = v___x_277_;
goto v_reusejp_280_;
}
else
{
lean_object* v_reuseFailAlloc_282_; 
v_reuseFailAlloc_282_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_282_, 0, v___x_279_);
v___x_281_ = v_reuseFailAlloc_282_;
goto v_reusejp_280_;
}
v_reusejp_280_:
{
return v___x_281_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4___redArg___boxed(lean_object* v_msg_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4___redArg(v_msg_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_);
lean_dec(v___y_288_);
lean_dec_ref(v___y_287_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
return v_res_290_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__4(void){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_298_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__1));
v___x_299_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__3));
v___x_300_ = l_Lean_Name_append(v___x_299_, v___x_298_);
return v___x_300_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__6(void){
_start:
{
lean_object* v___x_302_; lean_object* v___x_303_; 
v___x_302_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__5));
v___x_303_ = l_Lean_stringToMessageData(v___x_302_);
return v___x_303_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__22(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_328_ = lean_box(0);
v___x_329_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9));
v___x_330_ = l_Lean_mkConst(v___x_329_, v___x_328_);
return v___x_330_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__23(void){
_start:
{
lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_331_ = lean_box(0);
v___x_332_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11));
v___x_333_ = l_Lean_mkConst(v___x_332_, v___x_331_);
return v___x_333_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__25(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__24));
v___x_336_ = l_Lean_stringToMessageData(v___x_335_);
return v___x_336_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__27(void){
_start:
{
lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_338_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__26));
v___x_339_ = l_Lean_stringToMessageData(v___x_338_);
return v___x_339_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__29(void){
_start:
{
lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_341_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__28));
v___x_342_ = l_Lean_stringToMessageData(v___x_341_);
return v___x_342_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__31(void){
_start:
{
lean_object* v___x_344_; lean_object* v___x_345_; 
v___x_344_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__30));
v___x_345_ = l_Lean_stringToMessageData(v___x_344_);
return v___x_345_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__39(void){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_358_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__38));
v___x_359_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__3));
v___x_360_ = l_Lean_Name_append(v___x_359_, v___x_358_);
return v___x_360_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__42(void){
_start:
{
lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; 
v___x_365_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__41));
v___x_366_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__3));
v___x_367_ = l_Lean_Name_append(v___x_366_, v___x_365_);
return v___x_367_;
}
}
static lean_object* _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__45(void){
_start:
{
lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; 
v___x_372_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__44));
v___x_373_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__3));
v___x_374_ = l_Lean_Name_append(v___x_373_, v___x_372_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0(lean_object* v_snd_375_, uint8_t v___x_376_, lean_object* v___x_377_, lean_object* v_a_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_){
_start:
{
lean_object* v___y_389_; lean_object* v___y_390_; lean_object* v___y_391_; lean_object* v___y_392_; lean_object* v___y_393_; lean_object* v___x_408_; 
lean_inc(v_snd_375_);
v___x_408_ = l_Lean_MVarId_getType(v_snd_375_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
if (lean_obj_tag(v___x_408_) == 0)
{
lean_object* v_a_409_; lean_object* v___x_410_; 
v_a_409_ = lean_ctor_get(v___x_408_, 0);
lean_inc_n(v_a_409_, 2);
lean_dec_ref_known(v___x_408_, 1);
v___x_410_ = lp_plausible_Plausible_Decorations_addDecorations(v_a_409_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
if (lean_obj_tag(v___x_410_) == 0)
{
lean_object* v_options_411_; lean_object* v_a_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_796_; 
v_options_411_ = lean_ctor_get(v___y_385_, 2);
v_a_412_ = lean_ctor_get(v___x_410_, 0);
v_isSharedCheck_796_ = !lean_is_exclusive(v___x_410_);
if (v_isSharedCheck_796_ == 0)
{
v___x_414_ = v___x_410_;
v_isShared_415_ = v_isSharedCheck_796_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_a_412_);
lean_dec(v___x_410_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_796_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v_inheritedTraceOptions_416_; uint8_t v_hasTrace_417_; lean_object* v___x_418_; lean_object* v___y_420_; lean_object* v___y_421_; lean_object* v___y_422_; lean_object* v___y_423_; lean_object* v___y_452_; lean_object* v___y_453_; lean_object* v___y_454_; lean_object* v___y_455_; lean_object* v___y_456_; lean_object* v___y_457_; lean_object* v___y_458_; lean_object* v___y_459_; lean_object* v___y_460_; uint8_t v___y_461_; lean_object* v___y_462_; lean_object* v___y_463_; lean_object* v___y_464_; lean_object* v___y_465_; lean_object* v___y_466_; lean_object* v___y_474_; lean_object* v___y_475_; lean_object* v___y_476_; uint8_t v___y_477_; lean_object* v___y_478_; lean_object* v___y_479_; lean_object* v___y_480_; lean_object* v___y_481_; lean_object* v___y_482_; lean_object* v___y_483_; uint8_t v___y_484_; lean_object* v___y_485_; lean_object* v___y_486_; lean_object* v___y_487_; lean_object* v___y_488_; lean_object* v___y_494_; lean_object* v___y_495_; lean_object* v___y_496_; uint8_t v___y_497_; lean_object* v___y_498_; lean_object* v___y_499_; lean_object* v___y_500_; lean_object* v___y_501_; lean_object* v___y_502_; lean_object* v___y_503_; lean_object* v___y_504_; uint8_t v___y_505_; lean_object* v___y_506_; lean_object* v___y_507_; lean_object* v___y_508_; lean_object* v___y_524_; lean_object* v___y_525_; lean_object* v___y_526_; uint8_t v___y_527_; lean_object* v___y_528_; lean_object* v___y_529_; lean_object* v___y_530_; lean_object* v___y_531_; lean_object* v___y_532_; lean_object* v___y_533_; uint8_t v___y_534_; uint8_t v___y_535_; lean_object* v___y_536_; lean_object* v___y_537_; lean_object* v___y_538_; lean_object* v___y_544_; lean_object* v___y_545_; lean_object* v___y_546_; uint8_t v___y_547_; uint8_t v___y_548_; lean_object* v___y_549_; lean_object* v___y_550_; lean_object* v___y_551_; lean_object* v___y_552_; lean_object* v___y_553_; uint8_t v___y_554_; uint8_t v___y_555_; lean_object* v___y_556_; lean_object* v___y_557_; lean_object* v___y_558_; lean_object* v___y_564_; lean_object* v___y_565_; lean_object* v___y_566_; uint8_t v___y_567_; uint8_t v___y_568_; lean_object* v___y_569_; lean_object* v___y_570_; lean_object* v___y_571_; lean_object* v___y_572_; uint8_t v___y_573_; uint8_t v___y_574_; uint8_t v___y_575_; lean_object* v___y_576_; lean_object* v___y_577_; lean_object* v___y_578_; lean_object* v___y_584_; uint8_t v___y_585_; uint8_t v___y_586_; uint8_t v___y_587_; lean_object* v___y_588_; uint8_t v___y_589_; lean_object* v___y_590_; uint8_t v___y_591_; uint8_t v___y_592_; lean_object* v___y_593_; lean_object* v___y_594_; lean_object* v___y_595_; lean_object* v_a_596_; lean_object* v___y_611_; uint8_t v___y_612_; lean_object* v___y_613_; lean_object* v___y_614_; uint8_t v___y_615_; uint8_t v___y_616_; lean_object* v___y_617_; lean_object* v___y_618_; lean_object* v___y_619_; uint8_t v___y_620_; uint8_t v___y_621_; lean_object* v___y_622_; uint8_t v___y_623_; lean_object* v___y_624_; uint8_t v___y_625_; lean_object* v___y_662_; uint8_t v___y_663_; lean_object* v___y_664_; lean_object* v___y_665_; uint8_t v___y_666_; uint8_t v___y_667_; lean_object* v___y_668_; lean_object* v___y_669_; lean_object* v___y_670_; uint8_t v___y_671_; uint8_t v___y_672_; uint8_t v___y_673_; lean_object* v___y_674_; lean_object* v_a_675_; lean_object* v___y_679_; uint8_t v___y_680_; uint8_t v___y_681_; uint8_t v___y_682_; lean_object* v___y_683_; uint8_t v___y_684_; lean_object* v___y_685_; uint8_t v___y_686_; lean_object* v___y_687_; uint8_t v___y_688_; lean_object* v___y_712_; uint8_t v___y_713_; uint8_t v___y_714_; uint8_t v___y_715_; uint8_t v___y_716_; lean_object* v___y_717_; uint8_t v___y_718_; uint8_t v___y_719_; lean_object* v___y_720_; lean_object* v___y_721_; uint8_t v___y_722_; uint8_t v___y_724_; lean_object* v___y_725_; uint8_t v___y_726_; uint8_t v___y_727_; uint8_t v___y_728_; lean_object* v___y_729_; lean_object* v___y_730_; uint8_t v___y_731_; uint8_t v___y_732_; uint8_t v___y_733_; lean_object* v___y_734_; uint8_t v___y_735_; uint8_t v___y_737_; lean_object* v___y_738_; uint8_t v___y_739_; lean_object* v___y_740_; uint8_t v___y_741_; uint8_t v___y_742_; uint8_t v___y_743_; lean_object* v___y_744_; uint8_t v___y_745_; uint8_t v___y_746_; uint8_t v___y_747_; lean_object* v___y_748_; uint8_t v___y_749_; uint8_t v___y_751_; uint8_t v___y_752_; uint8_t v___y_753_; uint8_t v_a_754_; lean_object* v___y_775_; uint8_t v___y_776_; uint8_t v___y_777_; uint8_t v_a_778_; uint8_t v___y_785_; uint8_t v_a_786_; uint8_t v_a_791_; 
v_inheritedTraceOptions_416_ = lean_ctor_get(v___y_385_, 13);
v_hasTrace_417_ = lean_ctor_get_uint8(v_options_411_, sizeof(void*)*1);
v___x_418_ = ((lean_object*)(lp_plausible_plausibleSyntax___closed__4));
if (v_hasTrace_417_ == 0)
{
v_a_791_ = v_hasTrace_417_;
goto v___jp_790_;
}
else
{
lean_object* v___x_794_; uint8_t v___x_795_; 
v___x_794_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__45, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__45_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__45);
v___x_795_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_416_, v_options_411_, v___x_794_);
v_a_791_ = v___x_795_;
goto v___jp_790_;
}
v___jp_419_:
{
lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_424_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_424_, 0, v___y_423_);
lean_inc(v_a_412_);
v___x_425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_425_, 0, v_a_412_);
v___x_426_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_426_, 0, v___y_421_);
v___x_427_ = lean_unsigned_to_nat(4u);
v___x_428_ = lean_mk_empty_array_with_capacity(v___x_427_);
v___x_429_ = lean_array_push(v___x_428_, v___y_420_);
v___x_430_ = lean_array_push(v___x_429_, v___x_424_);
v___x_431_ = lean_array_push(v___x_430_, v___x_425_);
v___x_432_ = lean_array_push(v___x_431_, v___x_426_);
v___x_433_ = l_Lean_Meta_mkAppOptM(v___y_422_, v___x_432_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
if (lean_obj_tag(v___x_433_) == 0)
{
if (v_hasTrace_417_ == 0)
{
lean_object* v_a_434_; 
lean_dec(v_a_412_);
v_a_434_ = lean_ctor_get(v___x_433_, 0);
lean_inc(v_a_434_);
lean_dec_ref_known(v___x_433_, 1);
v___y_389_ = v_a_434_;
v___y_390_ = v___y_383_;
v___y_391_ = v___y_384_;
v___y_392_ = v___y_385_;
v___y_393_ = v___y_386_;
goto v___jp_388_;
}
else
{
lean_object* v_a_435_; lean_object* v___x_436_; lean_object* v___x_437_; uint8_t v___x_438_; 
v_a_435_ = lean_ctor_get(v___x_433_, 0);
lean_inc(v_a_435_);
lean_dec_ref_known(v___x_433_, 1);
v___x_436_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__1));
v___x_437_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__4, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__4_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__4);
v___x_438_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_416_, v_options_411_, v___x_437_);
if (v___x_438_ == 0)
{
lean_dec(v_a_412_);
v___y_389_ = v_a_435_;
v___y_390_ = v___y_383_;
v___y_391_ = v___y_384_;
v___y_392_ = v___y_385_;
v___y_393_ = v___y_386_;
goto v___jp_388_;
}
else
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_439_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__6, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__6_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__6);
v___x_440_ = l_Lean_MessageData_ofExpr(v_a_412_);
v___x_441_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_441_, 0, v___x_439_);
lean_ctor_set(v___x_441_, 1, v___x_440_);
v___x_442_ = lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg(v___x_436_, v___x_441_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
if (lean_obj_tag(v___x_442_) == 0)
{
lean_dec_ref_known(v___x_442_, 1);
v___y_389_ = v_a_435_;
v___y_390_ = v___y_383_;
v___y_391_ = v___y_384_;
v___y_392_ = v___y_385_;
v___y_393_ = v___y_386_;
goto v___jp_388_;
}
else
{
lean_dec(v_a_435_);
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
lean_dec(v_snd_375_);
return v___x_442_;
}
}
}
}
else
{
lean_object* v_a_443_; lean_object* v___x_445_; uint8_t v_isShared_446_; uint8_t v_isSharedCheck_450_; 
lean_dec(v_a_412_);
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
lean_dec(v_snd_375_);
v_a_443_ = lean_ctor_get(v___x_433_, 0);
v_isSharedCheck_450_ = !lean_is_exclusive(v___x_433_);
if (v_isSharedCheck_450_ == 0)
{
v___x_445_ = v___x_433_;
v_isShared_446_ = v_isSharedCheck_450_;
goto v_resetjp_444_;
}
else
{
lean_inc(v_a_443_);
lean_dec(v___x_433_);
v___x_445_ = lean_box(0);
v_isShared_446_ = v_isSharedCheck_450_;
goto v_resetjp_444_;
}
v_resetjp_444_:
{
lean_object* v___x_448_; 
if (v_isShared_446_ == 0)
{
v___x_448_ = v___x_445_;
goto v_reusejp_447_;
}
else
{
lean_object* v_reuseFailAlloc_449_; 
v_reuseFailAlloc_449_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_449_, 0, v_a_443_);
v___x_448_ = v_reuseFailAlloc_449_;
goto v_reusejp_447_;
}
v_reusejp_447_:
{
return v___x_448_;
}
}
}
}
v___jp_451_:
{
if (v___y_461_ == 0)
{
lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; 
v___x_467_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9));
v___x_468_ = l_Lean_mkConst(v___x_467_, v___y_457_);
lean_inc_ref(v___y_455_);
v___x_469_ = l_Lean_mkApp10(v___y_460_, v___y_454_, v___y_453_, v___y_452_, v___y_455_, v___y_458_, v___y_459_, v___y_462_, v___y_463_, v___y_466_, v___x_468_);
v___y_420_ = v___y_456_;
v___y_421_ = v___y_464_;
v___y_422_ = v___y_465_;
v___y_423_ = v___x_469_;
goto v___jp_419_;
}
else
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; 
v___x_470_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11));
v___x_471_ = l_Lean_mkConst(v___x_470_, v___y_457_);
lean_inc_ref(v___y_455_);
v___x_472_ = l_Lean_mkApp10(v___y_460_, v___y_454_, v___y_453_, v___y_452_, v___y_455_, v___y_458_, v___y_459_, v___y_462_, v___y_463_, v___y_466_, v___x_471_);
v___y_420_ = v___y_456_;
v___y_421_ = v___y_464_;
v___y_422_ = v___y_465_;
v___y_423_ = v___x_472_;
goto v___jp_419_;
}
}
v___jp_473_:
{
if (v___y_477_ == 0)
{
lean_object* v___x_489_; lean_object* v___x_490_; 
v___x_489_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9));
lean_inc(v___y_480_);
v___x_490_ = l_Lean_mkConst(v___x_489_, v___y_480_);
v___y_452_ = v___y_474_;
v___y_453_ = v___y_475_;
v___y_454_ = v___y_476_;
v___y_455_ = v___y_478_;
v___y_456_ = v___y_479_;
v___y_457_ = v___y_480_;
v___y_458_ = v___y_481_;
v___y_459_ = v___y_482_;
v___y_460_ = v___y_483_;
v___y_461_ = v___y_484_;
v___y_462_ = v___y_485_;
v___y_463_ = v___y_488_;
v___y_464_ = v___y_486_;
v___y_465_ = v___y_487_;
v___y_466_ = v___x_490_;
goto v___jp_451_;
}
else
{
lean_object* v___x_491_; lean_object* v___x_492_; 
v___x_491_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11));
lean_inc(v___y_480_);
v___x_492_ = l_Lean_mkConst(v___x_491_, v___y_480_);
v___y_452_ = v___y_474_;
v___y_453_ = v___y_475_;
v___y_454_ = v___y_476_;
v___y_455_ = v___y_478_;
v___y_456_ = v___y_479_;
v___y_457_ = v___y_480_;
v___y_458_ = v___y_481_;
v___y_459_ = v___y_482_;
v___y_460_ = v___y_483_;
v___y_461_ = v___y_484_;
v___y_462_ = v___y_485_;
v___y_463_ = v___y_488_;
v___y_464_ = v___y_486_;
v___y_465_ = v___y_487_;
v___y_466_ = v___x_492_;
goto v___jp_451_;
}
}
v___jp_493_:
{
lean_object* v___x_509_; lean_object* v_type_510_; 
v___x_509_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__13));
lean_inc(v___y_501_);
v_type_510_ = l_Lean_mkConst(v___x_509_, v___y_501_);
if (lean_obj_tag(v___y_498_) == 0)
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; 
v___x_511_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__16));
v___x_512_ = lean_box(0);
lean_inc(v___y_501_);
v___x_513_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_513_, 0, v___x_512_);
lean_ctor_set(v___x_513_, 1, v___y_501_);
v___x_514_ = l_Lean_mkConst(v___x_511_, v___x_513_);
v___x_515_ = l_Lean_Expr_app___override(v___x_514_, v_type_510_);
v___y_474_ = v___y_494_;
v___y_475_ = v___y_495_;
v___y_476_ = v___y_496_;
v___y_477_ = v___y_497_;
v___y_478_ = v___y_499_;
v___y_479_ = v___y_500_;
v___y_480_ = v___y_501_;
v___y_481_ = v___y_502_;
v___y_482_ = v___y_503_;
v___y_483_ = v___y_504_;
v___y_484_ = v___y_505_;
v___y_485_ = v___y_508_;
v___y_486_ = v___y_506_;
v___y_487_ = v___y_507_;
v___y_488_ = v___x_515_;
goto v___jp_473_;
}
else
{
lean_object* v_val_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; 
v_val_516_ = lean_ctor_get(v___y_498_, 0);
lean_inc(v_val_516_);
lean_dec_ref_known(v___y_498_, 1);
v___x_517_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__18));
v___x_518_ = lean_box(0);
lean_inc(v___y_501_);
v___x_519_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_519_, 0, v___x_518_);
lean_ctor_set(v___x_519_, 1, v___y_501_);
v___x_520_ = l_Lean_mkConst(v___x_517_, v___x_519_);
v___x_521_ = l_Lean_mkNatLit(v_val_516_);
v___x_522_ = l_Lean_mkAppB(v___x_520_, v_type_510_, v___x_521_);
v___y_474_ = v___y_494_;
v___y_475_ = v___y_495_;
v___y_476_ = v___y_496_;
v___y_477_ = v___y_497_;
v___y_478_ = v___y_499_;
v___y_479_ = v___y_500_;
v___y_480_ = v___y_501_;
v___y_481_ = v___y_502_;
v___y_482_ = v___y_503_;
v___y_483_ = v___y_504_;
v___y_484_ = v___y_505_;
v___y_485_ = v___y_508_;
v___y_486_ = v___y_506_;
v___y_487_ = v___y_507_;
v___y_488_ = v___x_522_;
goto v___jp_473_;
}
}
v___jp_523_:
{
if (v___y_535_ == 0)
{
lean_object* v___x_539_; lean_object* v___x_540_; 
v___x_539_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9));
lean_inc(v___y_531_);
v___x_540_ = l_Lean_mkConst(v___x_539_, v___y_531_);
v___y_494_ = v___y_524_;
v___y_495_ = v___y_525_;
v___y_496_ = v___y_526_;
v___y_497_ = v___y_527_;
v___y_498_ = v___y_528_;
v___y_499_ = v___y_529_;
v___y_500_ = v___y_530_;
v___y_501_ = v___y_531_;
v___y_502_ = v___y_532_;
v___y_503_ = v___y_538_;
v___y_504_ = v___y_533_;
v___y_505_ = v___y_534_;
v___y_506_ = v___y_536_;
v___y_507_ = v___y_537_;
v___y_508_ = v___x_540_;
goto v___jp_493_;
}
else
{
lean_object* v___x_541_; lean_object* v___x_542_; 
v___x_541_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11));
lean_inc(v___y_531_);
v___x_542_ = l_Lean_mkConst(v___x_541_, v___y_531_);
v___y_494_ = v___y_524_;
v___y_495_ = v___y_525_;
v___y_496_ = v___y_526_;
v___y_497_ = v___y_527_;
v___y_498_ = v___y_528_;
v___y_499_ = v___y_529_;
v___y_500_ = v___y_530_;
v___y_501_ = v___y_531_;
v___y_502_ = v___y_532_;
v___y_503_ = v___y_538_;
v___y_504_ = v___y_533_;
v___y_505_ = v___y_534_;
v___y_506_ = v___y_536_;
v___y_507_ = v___y_537_;
v___y_508_ = v___x_542_;
goto v___jp_493_;
}
}
v___jp_543_:
{
if (v___y_548_ == 0)
{
lean_object* v___x_559_; lean_object* v___x_560_; 
v___x_559_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9));
lean_inc(v___y_552_);
v___x_560_ = l_Lean_mkConst(v___x_559_, v___y_552_);
v___y_524_ = v___y_544_;
v___y_525_ = v___y_545_;
v___y_526_ = v___y_546_;
v___y_527_ = v___y_547_;
v___y_528_ = v___y_549_;
v___y_529_ = v___y_550_;
v___y_530_ = v___y_551_;
v___y_531_ = v___y_552_;
v___y_532_ = v___y_558_;
v___y_533_ = v___y_553_;
v___y_534_ = v___y_554_;
v___y_535_ = v___y_555_;
v___y_536_ = v___y_556_;
v___y_537_ = v___y_557_;
v___y_538_ = v___x_560_;
goto v___jp_523_;
}
else
{
lean_object* v___x_561_; lean_object* v___x_562_; 
v___x_561_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11));
lean_inc(v___y_552_);
v___x_562_ = l_Lean_mkConst(v___x_561_, v___y_552_);
v___y_524_ = v___y_544_;
v___y_525_ = v___y_545_;
v___y_526_ = v___y_546_;
v___y_527_ = v___y_547_;
v___y_528_ = v___y_549_;
v___y_529_ = v___y_550_;
v___y_530_ = v___y_551_;
v___y_531_ = v___y_552_;
v___y_532_ = v___y_558_;
v___y_533_ = v___y_553_;
v___y_534_ = v___y_554_;
v___y_535_ = v___y_555_;
v___y_536_ = v___y_556_;
v___y_537_ = v___y_557_;
v___y_538_ = v___x_562_;
goto v___jp_523_;
}
}
v___jp_563_:
{
if (v___y_573_ == 0)
{
lean_object* v___x_579_; lean_object* v___x_580_; 
v___x_579_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__9));
lean_inc(v___y_571_);
v___x_580_ = l_Lean_mkConst(v___x_579_, v___y_571_);
v___y_544_ = v___y_564_;
v___y_545_ = v___y_565_;
v___y_546_ = v___y_566_;
v___y_547_ = v___y_567_;
v___y_548_ = v___y_568_;
v___y_549_ = v___y_569_;
v___y_550_ = v___y_578_;
v___y_551_ = v___y_570_;
v___y_552_ = v___y_571_;
v___y_553_ = v___y_572_;
v___y_554_ = v___y_574_;
v___y_555_ = v___y_575_;
v___y_556_ = v___y_576_;
v___y_557_ = v___y_577_;
v___y_558_ = v___x_580_;
goto v___jp_543_;
}
else
{
lean_object* v___x_581_; lean_object* v___x_582_; 
v___x_581_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__11));
lean_inc(v___y_571_);
v___x_582_ = l_Lean_mkConst(v___x_581_, v___y_571_);
v___y_544_ = v___y_564_;
v___y_545_ = v___y_565_;
v___y_546_ = v___y_566_;
v___y_547_ = v___y_567_;
v___y_548_ = v___y_568_;
v___y_549_ = v___y_569_;
v___y_550_ = v___y_578_;
v___y_551_ = v___y_570_;
v___y_552_ = v___y_571_;
v___y_553_ = v___y_572_;
v___y_554_ = v___y_574_;
v___y_555_ = v___y_575_;
v___y_556_ = v___y_576_;
v___y_557_ = v___y_577_;
v___y_558_ = v___x_582_;
goto v___jp_543_;
}
}
v___jp_583_:
{
lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; 
v___x_597_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__19));
lean_inc_ref(v___y_588_);
lean_inc_ref_n(v___y_594_, 2);
v___x_598_ = l_Lean_Name_mkStr3(v___y_594_, v___y_588_, v___x_597_);
v___x_599_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_599_, 0, v_a_409_);
v___x_600_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__20));
v___x_601_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__21));
v___x_602_ = l_Lean_Name_mkStr3(v___y_594_, v___x_600_, v___x_601_);
v___x_603_ = lean_box(0);
v___x_604_ = l_Lean_mkConst(v___x_602_, v___x_603_);
v___x_605_ = l_Lean_mkNatLit(v___y_593_);
v___x_606_ = l_Lean_mkNatLit(v___y_590_);
v___x_607_ = l_Lean_mkNatLit(v___y_584_);
if (v___y_592_ == 0)
{
lean_object* v___x_608_; 
v___x_608_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__22, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__22_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__22);
v___y_564_ = v___x_607_;
v___y_565_ = v___x_606_;
v___y_566_ = v___x_605_;
v___y_567_ = v___y_587_;
v___y_568_ = v___y_591_;
v___y_569_ = v___y_595_;
v___y_570_ = v___x_599_;
v___y_571_ = v___x_603_;
v___y_572_ = v___x_604_;
v___y_573_ = v___y_585_;
v___y_574_ = v___y_586_;
v___y_575_ = v___y_589_;
v___y_576_ = v_a_596_;
v___y_577_ = v___x_598_;
v___y_578_ = v___x_608_;
goto v___jp_563_;
}
else
{
lean_object* v___x_609_; 
v___x_609_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__23, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__23_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__23);
v___y_564_ = v___x_607_;
v___y_565_ = v___x_606_;
v___y_566_ = v___x_605_;
v___y_567_ = v___y_587_;
v___y_568_ = v___y_591_;
v___y_569_ = v___y_595_;
v___y_570_ = v___x_599_;
v___y_571_ = v___x_603_;
v___y_572_ = v___x_604_;
v___y_573_ = v___y_585_;
v___y_574_ = v___y_586_;
v___y_575_ = v___y_589_;
v___y_576_ = v_a_596_;
v___y_577_ = v___x_598_;
v___y_578_ = v___x_609_;
goto v___jp_563_;
}
}
v___jp_610_:
{
lean_dec(v___y_624_);
lean_dec(v___y_619_);
lean_dec(v___y_617_);
lean_dec(v___y_611_);
if (v___y_625_ == 0)
{
lean_object* v___x_626_; 
lean_dec_ref(v___y_622_);
lean_del_object(v___x_414_);
v___x_626_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v___y_614_, v___y_625_, v___y_380_, v___y_381_, v___y_382_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
if (lean_obj_tag(v___x_626_) == 0)
{
lean_dec_ref_known(v___x_626_, 1);
if (v___y_621_ == 0)
{
lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v_a_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_647_; 
lean_dec(v_snd_375_);
v___x_627_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__25, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__25_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__25);
v___x_628_ = l_Lean_MessageData_ofExpr(v_a_409_);
v___x_629_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_629_, 0, v___x_627_);
lean_ctor_set(v___x_629_, 1, v___x_628_);
v___x_630_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__27, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__27_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__27);
v___x_631_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_631_, 0, v___x_629_);
lean_ctor_set(v___x_631_, 1, v___x_630_);
v___x_632_ = l_Lean_MessageData_ofExpr(v_a_412_);
lean_inc_ref(v___x_632_);
v___x_633_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_633_, 0, v___x_631_);
lean_ctor_set(v___x_633_, 1, v___x_632_);
v___x_634_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__29, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__29_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__29);
v___x_635_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_635_, 0, v___x_633_);
lean_ctor_set(v___x_635_, 1, v___x_634_);
v___x_636_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_636_, 0, v___x_635_);
lean_ctor_set(v___x_636_, 1, v___x_632_);
v___x_637_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__31, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__31_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__31);
v___x_638_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_638_, 0, v___x_636_);
lean_ctor_set(v___x_638_, 1, v___x_637_);
v___x_639_ = lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4___redArg(v___x_638_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
v_a_640_ = lean_ctor_get(v___x_639_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v___x_639_);
if (v_isSharedCheck_647_ == 0)
{
v___x_642_ = v___x_639_;
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_a_640_);
lean_dec(v___x_639_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v___x_645_; 
if (v_isShared_643_ == 0)
{
v___x_645_ = v___x_642_;
goto v_reusejp_644_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v_a_640_);
v___x_645_ = v_reuseFailAlloc_646_;
goto v_reusejp_644_;
}
v_reusejp_644_:
{
return v___x_645_;
}
}
}
else
{
lean_object* v___x_648_; 
lean_dec(v_a_412_);
lean_dec(v_a_409_);
v___x_648_ = l_Lean_Elab_admitGoal(v_snd_375_, v___x_376_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
if (lean_obj_tag(v___x_648_) == 0)
{
lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_656_; 
v_isSharedCheck_656_ = !lean_is_exclusive(v___x_648_);
if (v_isSharedCheck_656_ == 0)
{
lean_object* v_unused_657_; 
v_unused_657_ = lean_ctor_get(v___x_648_, 0);
lean_dec(v_unused_657_);
v___x_650_ = v___x_648_;
v_isShared_651_ = v_isSharedCheck_656_;
goto v_resetjp_649_;
}
else
{
lean_dec(v___x_648_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_656_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_652_; lean_object* v___x_654_; 
v___x_652_ = lean_box(0);
if (v_isShared_651_ == 0)
{
lean_ctor_set(v___x_650_, 0, v___x_652_);
v___x_654_ = v___x_650_;
goto v_reusejp_653_;
}
else
{
lean_object* v_reuseFailAlloc_655_; 
v_reuseFailAlloc_655_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_655_, 0, v___x_652_);
v___x_654_ = v_reuseFailAlloc_655_;
goto v_reusejp_653_;
}
v_reusejp_653_:
{
return v___x_654_;
}
}
}
else
{
return v___x_648_;
}
}
}
else
{
lean_dec(v_a_412_);
lean_dec(v_a_409_);
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
lean_dec(v_snd_375_);
return v___x_626_;
}
}
else
{
lean_object* v___x_659_; 
lean_dec_ref(v___y_614_);
lean_dec(v_a_412_);
lean_dec(v_a_409_);
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
lean_dec(v_snd_375_);
if (v_isShared_415_ == 0)
{
lean_ctor_set_tag(v___x_414_, 1);
lean_ctor_set(v___x_414_, 0, v___y_622_);
v___x_659_ = v___x_414_;
goto v_reusejp_658_;
}
else
{
lean_object* v_reuseFailAlloc_660_; 
v_reuseFailAlloc_660_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_660_, 0, v___y_622_);
v___x_659_ = v_reuseFailAlloc_660_;
goto v_reusejp_658_;
}
v_reusejp_658_:
{
return v___x_659_;
}
}
}
v___jp_661_:
{
uint8_t v___x_676_; 
v___x_676_ = l_Lean_Exception_isInterrupt(v_a_675_);
if (v___x_676_ == 0)
{
uint8_t v___x_677_; 
lean_inc_ref(v_a_675_);
v___x_677_ = l_Lean_Exception_isRuntime(v_a_675_);
v___y_611_ = v___y_662_;
v___y_612_ = v___y_663_;
v___y_613_ = v___y_664_;
v___y_614_ = v___y_665_;
v___y_615_ = v___y_666_;
v___y_616_ = v___y_667_;
v___y_617_ = v___y_668_;
v___y_618_ = v___y_669_;
v___y_619_ = v___y_670_;
v___y_620_ = v___y_671_;
v___y_621_ = v___y_672_;
v___y_622_ = v_a_675_;
v___y_623_ = v___y_673_;
v___y_624_ = v___y_674_;
v___y_625_ = v___x_677_;
goto v___jp_610_;
}
else
{
v___y_611_ = v___y_662_;
v___y_612_ = v___y_663_;
v___y_613_ = v___y_664_;
v___y_614_ = v___y_665_;
v___y_615_ = v___y_666_;
v___y_616_ = v___y_667_;
v___y_617_ = v___y_668_;
v___y_618_ = v___y_669_;
v___y_619_ = v___y_670_;
v___y_620_ = v___y_671_;
v___y_621_ = v___y_672_;
v___y_622_ = v_a_675_;
v___y_623_ = v___y_673_;
v___y_624_ = v___y_674_;
v___y_625_ = v___x_676_;
goto v___jp_610_;
}
}
v___jp_678_:
{
lean_object* v___x_689_; 
v___x_689_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_380_, v___y_382_, v___y_384_, v___y_386_);
if (lean_obj_tag(v___x_689_) == 0)
{
lean_object* v_a_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; 
v_a_690_ = lean_ctor_get(v___x_689_, 0);
lean_inc(v_a_690_);
lean_dec_ref_known(v___x_689_, 1);
v___x_691_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__32));
v___x_692_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__33));
v___x_693_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__34));
v___x_694_ = lean_unsigned_to_nat(1u);
v___x_695_ = lean_mk_empty_array_with_capacity(v___x_694_);
lean_inc(v_a_412_);
v___x_696_ = lean_array_push(v___x_695_, v_a_412_);
v___x_697_ = l_Lean_Meta_mkAppM(v___x_693_, v___x_696_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
if (lean_obj_tag(v___x_697_) == 0)
{
lean_object* v_a_698_; lean_object* v___x_699_; 
v_a_698_ = lean_ctor_get(v___x_697_, 0);
lean_inc(v_a_698_);
lean_dec_ref_known(v___x_697_, 1);
v___x_699_ = l_Lean_Meta_synthInstance(v_a_698_, v___x_377_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
if (lean_obj_tag(v___x_699_) == 0)
{
lean_object* v_a_700_; 
lean_dec(v_a_690_);
lean_del_object(v___x_414_);
v_a_700_ = lean_ctor_get(v___x_699_, 0);
lean_inc(v_a_700_);
lean_dec_ref_known(v___x_699_, 1);
v___y_584_ = v___y_679_;
v___y_585_ = v___y_680_;
v___y_586_ = v___y_682_;
v___y_587_ = v___y_681_;
v___y_588_ = v___x_692_;
v___y_589_ = v___y_688_;
v___y_590_ = v___y_683_;
v___y_591_ = v___y_684_;
v___y_592_ = v___y_686_;
v___y_593_ = v___y_685_;
v___y_594_ = v___x_691_;
v___y_595_ = v___y_687_;
v_a_596_ = v_a_700_;
goto v___jp_583_;
}
else
{
lean_object* v_a_701_; 
v_a_701_ = lean_ctor_get(v___x_699_, 0);
lean_inc(v_a_701_);
lean_dec_ref_known(v___x_699_, 1);
v___y_662_ = v___y_679_;
v___y_663_ = v___y_681_;
v___y_664_ = v___x_692_;
v___y_665_ = v_a_690_;
v___y_666_ = v___y_684_;
v___y_667_ = v___y_686_;
v___y_668_ = v___y_685_;
v___y_669_ = v___x_691_;
v___y_670_ = v___y_687_;
v___y_671_ = v___y_680_;
v___y_672_ = v___y_682_;
v___y_673_ = v___y_688_;
v___y_674_ = v___y_683_;
v_a_675_ = v_a_701_;
goto v___jp_661_;
}
}
else
{
lean_object* v_a_702_; 
lean_dec(v___x_377_);
v_a_702_ = lean_ctor_get(v___x_697_, 0);
lean_inc(v_a_702_);
lean_dec_ref_known(v___x_697_, 1);
v___y_662_ = v___y_679_;
v___y_663_ = v___y_681_;
v___y_664_ = v___x_692_;
v___y_665_ = v_a_690_;
v___y_666_ = v___y_684_;
v___y_667_ = v___y_686_;
v___y_668_ = v___y_685_;
v___y_669_ = v___x_691_;
v___y_670_ = v___y_687_;
v___y_671_ = v___y_680_;
v___y_672_ = v___y_682_;
v___y_673_ = v___y_688_;
v___y_674_ = v___y_683_;
v_a_675_ = v_a_702_;
goto v___jp_661_;
}
}
else
{
lean_object* v_a_703_; lean_object* v___x_705_; uint8_t v_isShared_706_; uint8_t v_isSharedCheck_710_; 
lean_dec(v___y_687_);
lean_dec(v___y_685_);
lean_dec(v___y_683_);
lean_dec(v___y_679_);
lean_del_object(v___x_414_);
lean_dec(v_a_412_);
lean_dec(v_a_409_);
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
lean_dec(v___x_377_);
lean_dec(v_snd_375_);
v_a_703_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_710_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_710_ == 0)
{
v___x_705_ = v___x_689_;
v_isShared_706_ = v_isSharedCheck_710_;
goto v_resetjp_704_;
}
else
{
lean_inc(v_a_703_);
lean_dec(v___x_689_);
v___x_705_ = lean_box(0);
v_isShared_706_ = v_isSharedCheck_710_;
goto v_resetjp_704_;
}
v_resetjp_704_:
{
lean_object* v___x_708_; 
if (v_isShared_706_ == 0)
{
v___x_708_ = v___x_705_;
goto v_reusejp_707_;
}
else
{
lean_object* v_reuseFailAlloc_709_; 
v_reuseFailAlloc_709_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_709_, 0, v_a_703_);
v___x_708_ = v_reuseFailAlloc_709_;
goto v_reusejp_707_;
}
v_reusejp_707_:
{
return v___x_708_;
}
}
}
}
v___jp_711_:
{
if (v___y_718_ == 0)
{
v___y_679_ = v___y_712_;
v___y_680_ = v___y_713_;
v___y_681_ = v___y_715_;
v___y_682_ = v___y_714_;
v___y_683_ = v___y_717_;
v___y_684_ = v___y_722_;
v___y_685_ = v___y_720_;
v___y_686_ = v___y_719_;
v___y_687_ = v___y_721_;
v___y_688_ = v___y_716_;
goto v___jp_678_;
}
else
{
v___y_679_ = v___y_712_;
v___y_680_ = v___y_713_;
v___y_681_ = v___y_715_;
v___y_682_ = v___y_714_;
v___y_683_ = v___y_717_;
v___y_684_ = v___y_722_;
v___y_685_ = v___y_720_;
v___y_686_ = v___y_719_;
v___y_687_ = v___y_721_;
v___y_688_ = v___x_376_;
goto v___jp_678_;
}
}
v___jp_723_:
{
if (v___y_724_ == 0)
{
v___y_712_ = v___y_725_;
v___y_713_ = v___y_735_;
v___y_714_ = v___y_727_;
v___y_715_ = v___y_726_;
v___y_716_ = v___y_728_;
v___y_717_ = v___y_729_;
v___y_718_ = v___y_732_;
v___y_719_ = v___y_731_;
v___y_720_ = v___y_730_;
v___y_721_ = v___y_734_;
v___y_722_ = v___y_733_;
goto v___jp_711_;
}
else
{
v___y_712_ = v___y_725_;
v___y_713_ = v___y_735_;
v___y_714_ = v___y_727_;
v___y_715_ = v___y_726_;
v___y_716_ = v___y_728_;
v___y_717_ = v___y_729_;
v___y_718_ = v___y_732_;
v___y_719_ = v___y_731_;
v___y_720_ = v___y_730_;
v___y_721_ = v___y_734_;
v___y_722_ = v___x_376_;
goto v___jp_711_;
}
}
v___jp_736_:
{
if (v___y_745_ == 0)
{
v___y_724_ = v___y_737_;
v___y_725_ = v___y_738_;
v___y_726_ = v___y_739_;
v___y_727_ = v___y_746_;
v___y_728_ = v___y_747_;
v___y_729_ = v___y_748_;
v___y_730_ = v___y_740_;
v___y_731_ = v___y_749_;
v___y_732_ = v___y_741_;
v___y_733_ = v___y_743_;
v___y_734_ = v___y_744_;
v___y_735_ = v___y_742_;
goto v___jp_723_;
}
else
{
v___y_724_ = v___y_737_;
v___y_725_ = v___y_738_;
v___y_726_ = v___y_739_;
v___y_727_ = v___y_746_;
v___y_728_ = v___y_747_;
v___y_729_ = v___y_748_;
v___y_730_ = v___y_740_;
v___y_731_ = v___y_749_;
v___y_732_ = v___y_741_;
v___y_733_ = v___y_743_;
v___y_734_ = v___y_744_;
v___y_735_ = v___x_376_;
goto v___jp_723_;
}
}
v___jp_750_:
{
uint8_t v_traceDiscarded_755_; 
v_traceDiscarded_755_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4);
if (v_traceDiscarded_755_ == 0)
{
lean_object* v_numInst_756_; lean_object* v_maxSize_757_; lean_object* v_numRetries_758_; uint8_t v_traceSuccesses_759_; uint8_t v_traceShrink_760_; uint8_t v_traceShrinkCandidates_761_; lean_object* v_randomSeed_762_; uint8_t v_quiet_763_; uint8_t v_sorryIfNoTestable_764_; 
v_numInst_756_ = lean_ctor_get(v_a_378_, 0);
lean_inc(v_numInst_756_);
v_maxSize_757_ = lean_ctor_get(v_a_378_, 1);
lean_inc(v_maxSize_757_);
v_numRetries_758_ = lean_ctor_get(v_a_378_, 2);
lean_inc(v_numRetries_758_);
v_traceSuccesses_759_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4 + 1);
v_traceShrink_760_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_761_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4 + 3);
v_randomSeed_762_ = lean_ctor_get(v_a_378_, 3);
lean_inc(v_randomSeed_762_);
v_quiet_763_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_764_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4 + 5);
lean_dec_ref(v_a_378_);
v___y_737_ = v_traceShrink_760_;
v___y_738_ = v_numRetries_758_;
v___y_739_ = v_quiet_763_;
v___y_740_ = v_numInst_756_;
v___y_741_ = v_traceShrinkCandidates_761_;
v___y_742_ = v___y_752_;
v___y_743_ = v___y_753_;
v___y_744_ = v_randomSeed_762_;
v___y_745_ = v_traceSuccesses_759_;
v___y_746_ = v_sorryIfNoTestable_764_;
v___y_747_ = v_a_754_;
v___y_748_ = v_maxSize_757_;
v___y_749_ = v___y_751_;
goto v___jp_736_;
}
else
{
lean_object* v_numInst_765_; lean_object* v_maxSize_766_; lean_object* v_numRetries_767_; uint8_t v_traceSuccesses_768_; uint8_t v_traceShrink_769_; uint8_t v_traceShrinkCandidates_770_; lean_object* v_randomSeed_771_; uint8_t v_quiet_772_; uint8_t v_sorryIfNoTestable_773_; 
v_numInst_765_ = lean_ctor_get(v_a_378_, 0);
lean_inc(v_numInst_765_);
v_maxSize_766_ = lean_ctor_get(v_a_378_, 1);
lean_inc(v_maxSize_766_);
v_numRetries_767_ = lean_ctor_get(v_a_378_, 2);
lean_inc(v_numRetries_767_);
v_traceSuccesses_768_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4 + 1);
v_traceShrink_769_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_770_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4 + 3);
v_randomSeed_771_ = lean_ctor_get(v_a_378_, 3);
lean_inc(v_randomSeed_771_);
v_quiet_772_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_773_ = lean_ctor_get_uint8(v_a_378_, sizeof(void*)*4 + 5);
lean_dec_ref(v_a_378_);
v___y_737_ = v_traceShrink_769_;
v___y_738_ = v_numRetries_767_;
v___y_739_ = v_quiet_772_;
v___y_740_ = v_numInst_765_;
v___y_741_ = v_traceShrinkCandidates_770_;
v___y_742_ = v___y_752_;
v___y_743_ = v___y_753_;
v___y_744_ = v_randomSeed_771_;
v___y_745_ = v_traceSuccesses_768_;
v___y_746_ = v_sorryIfNoTestable_773_;
v___y_747_ = v_a_754_;
v___y_748_ = v_maxSize_766_;
v___y_749_ = v___x_376_;
goto v___jp_736_;
}
}
v___jp_774_:
{
if (v_hasTrace_417_ == 0)
{
v___y_751_ = v___y_776_;
v___y_752_ = v___y_777_;
v___y_753_ = v_a_778_;
v_a_754_ = v_hasTrace_417_;
goto v___jp_750_;
}
else
{
lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; uint8_t v___x_783_; 
v___x_779_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__35));
lean_inc_ref(v___y_775_);
v___x_780_ = l_Lean_Name_mkStr3(v___x_418_, v___y_775_, v___x_779_);
v___x_781_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__3));
v___x_782_ = l_Lean_Name_append(v___x_781_, v___x_780_);
v___x_783_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_416_, v_options_411_, v___x_782_);
lean_dec(v___x_782_);
v___y_751_ = v___y_776_;
v___y_752_ = v___y_777_;
v___y_753_ = v_a_778_;
v_a_754_ = v___x_783_;
goto v___jp_750_;
}
}
v___jp_784_:
{
lean_object* v___x_787_; 
v___x_787_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__36));
if (v_hasTrace_417_ == 0)
{
v___y_775_ = v___x_787_;
v___y_776_ = v___y_785_;
v___y_777_ = v_a_786_;
v_a_778_ = v_hasTrace_417_;
goto v___jp_774_;
}
else
{
lean_object* v___x_788_; uint8_t v___x_789_; 
v___x_788_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__39, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__39_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__39);
v___x_789_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_416_, v_options_411_, v___x_788_);
v___y_775_ = v___x_787_;
v___y_776_ = v___y_785_;
v___y_777_ = v_a_786_;
v_a_778_ = v___x_789_;
goto v___jp_774_;
}
}
v___jp_790_:
{
if (v_hasTrace_417_ == 0)
{
v___y_785_ = v_a_791_;
v_a_786_ = v_hasTrace_417_;
goto v___jp_784_;
}
else
{
lean_object* v___x_792_; uint8_t v___x_793_; 
v___x_792_ = lean_obj_once(&lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__42, &lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__42_once, _init_lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___closed__42);
v___x_793_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_416_, v_options_411_, v___x_792_);
v___y_785_ = v_a_791_;
v_a_786_ = v___x_793_;
goto v___jp_784_;
}
}
}
}
else
{
lean_object* v_a_797_; lean_object* v___x_799_; uint8_t v_isShared_800_; uint8_t v_isSharedCheck_804_; 
lean_dec(v_a_409_);
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
lean_dec_ref(v_a_378_);
lean_dec(v___x_377_);
lean_dec(v_snd_375_);
v_a_797_ = lean_ctor_get(v___x_410_, 0);
v_isSharedCheck_804_ = !lean_is_exclusive(v___x_410_);
if (v_isSharedCheck_804_ == 0)
{
v___x_799_ = v___x_410_;
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
else
{
lean_inc(v_a_797_);
lean_dec(v___x_410_);
v___x_799_ = lean_box(0);
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
v_resetjp_798_:
{
lean_object* v___x_802_; 
if (v_isShared_800_ == 0)
{
v___x_802_ = v___x_799_;
goto v_reusejp_801_;
}
else
{
lean_object* v_reuseFailAlloc_803_; 
v_reuseFailAlloc_803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_803_, 0, v_a_797_);
v___x_802_ = v_reuseFailAlloc_803_;
goto v_reusejp_801_;
}
v_reusejp_801_:
{
return v___x_802_;
}
}
}
}
else
{
lean_object* v_a_805_; lean_object* v___x_807_; uint8_t v_isShared_808_; uint8_t v_isSharedCheck_812_; 
lean_dec(v___y_386_);
lean_dec_ref(v___y_385_);
lean_dec_ref(v_a_378_);
lean_dec(v___x_377_);
lean_dec(v_snd_375_);
v_a_805_ = lean_ctor_get(v___x_408_, 0);
v_isSharedCheck_812_ = !lean_is_exclusive(v___x_408_);
if (v_isSharedCheck_812_ == 0)
{
v___x_807_ = v___x_408_;
v_isShared_808_ = v_isSharedCheck_812_;
goto v_resetjp_806_;
}
else
{
lean_inc(v_a_805_);
lean_dec(v___x_408_);
v___x_807_ = lean_box(0);
v_isShared_808_ = v_isSharedCheck_812_;
goto v_resetjp_806_;
}
v_resetjp_806_:
{
lean_object* v___x_810_; 
if (v_isShared_808_ == 0)
{
v___x_810_ = v___x_807_;
goto v_reusejp_809_;
}
else
{
lean_object* v_reuseFailAlloc_811_; 
v_reuseFailAlloc_811_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_811_, 0, v_a_805_);
v___x_810_ = v_reuseFailAlloc_811_;
goto v_reusejp_809_;
}
v_reusejp_809_:
{
return v___x_810_;
}
}
}
v___jp_388_:
{
lean_object* v___x_394_; uint8_t v___x_395_; lean_object* v___x_396_; 
v___x_394_ = lean_obj_once(&lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__10, &lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__10_once, _init_lp_plausible___private_Plausible_Tactic_0____aux__Plausible__Tactic______elabRules__plausibleSyntax__1_unsafe__1___closed__10);
v___x_395_ = 1;
v___x_396_ = l_Lean_Meta_evalExpr___redArg(v___x_394_, v___y_389_, v___x_395_, v___x_376_, v___y_390_, v___y_391_, v___y_392_, v___y_393_);
if (lean_obj_tag(v___x_396_) == 0)
{
lean_object* v_a_397_; lean_object* v___x_398_; 
v_a_397_ = lean_ctor_get(v___x_396_, 0);
lean_inc(v_a_397_);
lean_dec_ref_known(v___x_396_, 1);
lean_inc(v___y_393_);
lean_inc_ref(v___y_392_);
v___x_398_ = lean_apply_3(v_a_397_, v___y_392_, v___y_393_, lean_box(0));
if (lean_obj_tag(v___x_398_) == 0)
{
lean_object* v___x_399_; 
lean_dec_ref_known(v___x_398_, 1);
v___x_399_ = l_Lean_Elab_admitGoal(v_snd_375_, v___x_376_, v___y_390_, v___y_391_, v___y_392_, v___y_393_);
lean_dec(v___y_393_);
lean_dec_ref(v___y_392_);
return v___x_399_;
}
else
{
lean_dec(v___y_393_);
lean_dec_ref(v___y_392_);
lean_dec(v_snd_375_);
return v___x_398_;
}
}
else
{
lean_object* v_a_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_407_; 
lean_dec(v___y_393_);
lean_dec_ref(v___y_392_);
lean_dec(v_snd_375_);
v_a_400_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_407_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_407_ == 0)
{
v___x_402_ = v___x_396_;
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_a_400_);
lean_dec(v___x_396_);
v___x_402_ = lean_box(0);
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
v_resetjp_401_:
{
lean_object* v___x_405_; 
if (v_isShared_403_ == 0)
{
v___x_405_ = v___x_402_;
goto v_reusejp_404_;
}
else
{
lean_object* v_reuseFailAlloc_406_; 
v_reuseFailAlloc_406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_406_, 0, v_a_400_);
v___x_405_ = v_reuseFailAlloc_406_;
goto v_reusejp_404_;
}
v_reusejp_404_:
{
return v___x_405_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___boxed(lean_object* v_snd_813_, lean_object* v___x_814_, lean_object* v___x_815_, lean_object* v_a_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_){
_start:
{
uint8_t v___x_29688__boxed_826_; lean_object* v_res_827_; 
v___x_29688__boxed_826_ = lean_unbox(v___x_814_);
v_res_827_ = lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0(v_snd_813_, v___x_29688__boxed_826_, v___x_815_, v_a_816_, v___y_817_, v___y_818_, v___y_819_, v___y_820_, v___y_821_, v___y_822_, v___y_823_, v___y_824_);
lean_dec(v___y_822_);
lean_dec_ref(v___y_821_);
lean_dec(v___y_820_);
lean_dec_ref(v___y_819_);
lean_dec(v___y_818_);
lean_dec_ref(v___y_817_);
return v_res_827_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10___redArg(lean_object* v_as_828_, size_t v_sz_829_, size_t v_i_830_, lean_object* v_b_831_){
_start:
{
uint8_t v___x_833_; 
v___x_833_ = lean_usize_dec_lt(v_i_830_, v_sz_829_);
if (v___x_833_ == 0)
{
lean_object* v___x_834_; 
v___x_834_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_834_, 0, v_b_831_);
return v___x_834_;
}
else
{
lean_object* v_snd_835_; lean_object* v___x_837_; uint8_t v_isShared_838_; uint8_t v_isSharedCheck_853_; 
v_snd_835_ = lean_ctor_get(v_b_831_, 1);
v_isSharedCheck_853_ = !lean_is_exclusive(v_b_831_);
if (v_isSharedCheck_853_ == 0)
{
lean_object* v_unused_854_; 
v_unused_854_ = lean_ctor_get(v_b_831_, 0);
lean_dec(v_unused_854_);
v___x_837_ = v_b_831_;
v_isShared_838_ = v_isSharedCheck_853_;
goto v_resetjp_836_;
}
else
{
lean_inc(v_snd_835_);
lean_dec(v_b_831_);
v___x_837_ = lean_box(0);
v_isShared_838_ = v_isSharedCheck_853_;
goto v_resetjp_836_;
}
v_resetjp_836_:
{
lean_object* v___x_839_; lean_object* v_a_841_; lean_object* v_a_848_; 
v___x_839_ = lean_box(0);
v_a_848_ = lean_array_uget_borrowed(v_as_828_, v_i_830_);
if (lean_obj_tag(v_a_848_) == 0)
{
v_a_841_ = v_snd_835_;
goto v___jp_840_;
}
else
{
lean_object* v_val_849_; uint8_t v___x_850_; 
v_val_849_ = lean_ctor_get(v_a_848_, 0);
v___x_850_ = l_Lean_LocalDecl_isImplementationDetail(v_val_849_);
if (v___x_850_ == 0)
{
lean_object* v___x_851_; lean_object* v___x_852_; 
lean_inc(v_val_849_);
v___x_851_ = l_Lean_LocalDecl_toExpr(v_val_849_);
v___x_852_ = lean_array_push(v_snd_835_, v___x_851_);
v_a_841_ = v___x_852_;
goto v___jp_840_;
}
else
{
v_a_841_ = v_snd_835_;
goto v___jp_840_;
}
}
v___jp_840_:
{
lean_object* v___x_843_; 
if (v_isShared_838_ == 0)
{
lean_ctor_set(v___x_837_, 1, v_a_841_);
lean_ctor_set(v___x_837_, 0, v___x_839_);
v___x_843_ = v___x_837_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_847_; 
v_reuseFailAlloc_847_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_847_, 0, v___x_839_);
lean_ctor_set(v_reuseFailAlloc_847_, 1, v_a_841_);
v___x_843_ = v_reuseFailAlloc_847_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
size_t v___x_844_; size_t v___x_845_; 
v___x_844_ = ((size_t)1ULL);
v___x_845_ = lean_usize_add(v_i_830_, v___x_844_);
v_i_830_ = v___x_845_;
v_b_831_ = v___x_843_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10___redArg___boxed(lean_object* v_as_855_, lean_object* v_sz_856_, lean_object* v_i_857_, lean_object* v_b_858_, lean_object* v___y_859_){
_start:
{
size_t v_sz_boxed_860_; size_t v_i_boxed_861_; lean_object* v_res_862_; 
v_sz_boxed_860_ = lean_unbox_usize(v_sz_856_);
lean_dec(v_sz_856_);
v_i_boxed_861_ = lean_unbox_usize(v_i_857_);
lean_dec(v_i_857_);
v_res_862_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10___redArg(v_as_855_, v_sz_boxed_860_, v_i_boxed_861_, v_b_858_);
lean_dec_ref(v_as_855_);
return v_res_862_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9(lean_object* v_as_863_, size_t v_sz_864_, size_t v_i_865_, lean_object* v_b_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_){
_start:
{
uint8_t v___x_876_; 
v___x_876_ = lean_usize_dec_lt(v_i_865_, v_sz_864_);
if (v___x_876_ == 0)
{
lean_object* v___x_877_; 
v___x_877_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_877_, 0, v_b_866_);
return v___x_877_;
}
else
{
lean_object* v_snd_878_; lean_object* v___x_880_; uint8_t v_isShared_881_; uint8_t v_isSharedCheck_896_; 
v_snd_878_ = lean_ctor_get(v_b_866_, 1);
v_isSharedCheck_896_ = !lean_is_exclusive(v_b_866_);
if (v_isSharedCheck_896_ == 0)
{
lean_object* v_unused_897_; 
v_unused_897_ = lean_ctor_get(v_b_866_, 0);
lean_dec(v_unused_897_);
v___x_880_ = v_b_866_;
v_isShared_881_ = v_isSharedCheck_896_;
goto v_resetjp_879_;
}
else
{
lean_inc(v_snd_878_);
lean_dec(v_b_866_);
v___x_880_ = lean_box(0);
v_isShared_881_ = v_isSharedCheck_896_;
goto v_resetjp_879_;
}
v_resetjp_879_:
{
lean_object* v___x_882_; lean_object* v_a_884_; lean_object* v_a_891_; 
v___x_882_ = lean_box(0);
v_a_891_ = lean_array_uget_borrowed(v_as_863_, v_i_865_);
if (lean_obj_tag(v_a_891_) == 0)
{
v_a_884_ = v_snd_878_;
goto v___jp_883_;
}
else
{
lean_object* v_val_892_; uint8_t v___x_893_; 
v_val_892_ = lean_ctor_get(v_a_891_, 0);
v___x_893_ = l_Lean_LocalDecl_isImplementationDetail(v_val_892_);
if (v___x_893_ == 0)
{
lean_object* v___x_894_; lean_object* v___x_895_; 
lean_inc(v_val_892_);
v___x_894_ = l_Lean_LocalDecl_toExpr(v_val_892_);
v___x_895_ = lean_array_push(v_snd_878_, v___x_894_);
v_a_884_ = v___x_895_;
goto v___jp_883_;
}
else
{
v_a_884_ = v_snd_878_;
goto v___jp_883_;
}
}
v___jp_883_:
{
lean_object* v___x_886_; 
if (v_isShared_881_ == 0)
{
lean_ctor_set(v___x_880_, 1, v_a_884_);
lean_ctor_set(v___x_880_, 0, v___x_882_);
v___x_886_ = v___x_880_;
goto v_reusejp_885_;
}
else
{
lean_object* v_reuseFailAlloc_890_; 
v_reuseFailAlloc_890_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_890_, 0, v___x_882_);
lean_ctor_set(v_reuseFailAlloc_890_, 1, v_a_884_);
v___x_886_ = v_reuseFailAlloc_890_;
goto v_reusejp_885_;
}
v_reusejp_885_:
{
size_t v___x_887_; size_t v___x_888_; lean_object* v___x_889_; 
v___x_887_ = ((size_t)1ULL);
v___x_888_ = lean_usize_add(v_i_865_, v___x_887_);
v___x_889_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10___redArg(v_as_863_, v_sz_864_, v___x_888_, v___x_886_);
return v___x_889_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9___boxed(lean_object* v_as_898_, lean_object* v_sz_899_, lean_object* v_i_900_, lean_object* v_b_901_, lean_object* v___y_902_, lean_object* v___y_903_, lean_object* v___y_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_, lean_object* v___y_910_){
_start:
{
size_t v_sz_boxed_911_; size_t v_i_boxed_912_; lean_object* v_res_913_; 
v_sz_boxed_911_ = lean_unbox_usize(v_sz_899_);
lean_dec(v_sz_899_);
v_i_boxed_912_ = lean_unbox_usize(v_i_900_);
lean_dec(v_i_900_);
v_res_913_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9(v_as_898_, v_sz_boxed_911_, v_i_boxed_912_, v_b_901_, v___y_902_, v___y_903_, v___y_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_, v___y_909_);
lean_dec(v___y_909_);
lean_dec_ref(v___y_908_);
lean_dec(v___y_907_);
lean_dec_ref(v___y_906_);
lean_dec(v___y_905_);
lean_dec_ref(v___y_904_);
lean_dec(v___y_903_);
lean_dec_ref(v___y_902_);
lean_dec_ref(v_as_898_);
return v_res_913_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3(lean_object* v_init_914_, lean_object* v_n_915_, lean_object* v_b_916_, lean_object* v___y_917_, lean_object* v___y_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_){
_start:
{
if (lean_obj_tag(v_n_915_) == 0)
{
lean_object* v_cs_926_; lean_object* v___x_927_; lean_object* v___x_928_; size_t v_sz_929_; size_t v___x_930_; lean_object* v___x_931_; 
v_cs_926_ = lean_ctor_get(v_n_915_, 0);
v___x_927_ = lean_box(0);
v___x_928_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_928_, 0, v___x_927_);
lean_ctor_set(v___x_928_, 1, v_b_916_);
v_sz_929_ = lean_array_size(v_cs_926_);
v___x_930_ = ((size_t)0ULL);
v___x_931_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__8(v_init_914_, v_cs_926_, v_sz_929_, v___x_930_, v___x_928_, v___y_917_, v___y_918_, v___y_919_, v___y_920_, v___y_921_, v___y_922_, v___y_923_, v___y_924_);
if (lean_obj_tag(v___x_931_) == 0)
{
lean_object* v_a_932_; lean_object* v___x_934_; uint8_t v_isShared_935_; uint8_t v_isSharedCheck_946_; 
v_a_932_ = lean_ctor_get(v___x_931_, 0);
v_isSharedCheck_946_ = !lean_is_exclusive(v___x_931_);
if (v_isSharedCheck_946_ == 0)
{
v___x_934_ = v___x_931_;
v_isShared_935_ = v_isSharedCheck_946_;
goto v_resetjp_933_;
}
else
{
lean_inc(v_a_932_);
lean_dec(v___x_931_);
v___x_934_ = lean_box(0);
v_isShared_935_ = v_isSharedCheck_946_;
goto v_resetjp_933_;
}
v_resetjp_933_:
{
lean_object* v_fst_936_; 
v_fst_936_ = lean_ctor_get(v_a_932_, 0);
if (lean_obj_tag(v_fst_936_) == 0)
{
lean_object* v_snd_937_; lean_object* v___x_938_; lean_object* v___x_940_; 
v_snd_937_ = lean_ctor_get(v_a_932_, 1);
lean_inc(v_snd_937_);
lean_dec(v_a_932_);
v___x_938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_938_, 0, v_snd_937_);
if (v_isShared_935_ == 0)
{
lean_ctor_set(v___x_934_, 0, v___x_938_);
v___x_940_ = v___x_934_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_941_; 
v_reuseFailAlloc_941_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_941_, 0, v___x_938_);
v___x_940_ = v_reuseFailAlloc_941_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
return v___x_940_;
}
}
else
{
lean_object* v_val_942_; lean_object* v___x_944_; 
lean_inc_ref(v_fst_936_);
lean_dec(v_a_932_);
v_val_942_ = lean_ctor_get(v_fst_936_, 0);
lean_inc(v_val_942_);
lean_dec_ref_known(v_fst_936_, 1);
if (v_isShared_935_ == 0)
{
lean_ctor_set(v___x_934_, 0, v_val_942_);
v___x_944_ = v___x_934_;
goto v_reusejp_943_;
}
else
{
lean_object* v_reuseFailAlloc_945_; 
v_reuseFailAlloc_945_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_945_, 0, v_val_942_);
v___x_944_ = v_reuseFailAlloc_945_;
goto v_reusejp_943_;
}
v_reusejp_943_:
{
return v___x_944_;
}
}
}
}
else
{
lean_object* v_a_947_; lean_object* v___x_949_; uint8_t v_isShared_950_; uint8_t v_isSharedCheck_954_; 
v_a_947_ = lean_ctor_get(v___x_931_, 0);
v_isSharedCheck_954_ = !lean_is_exclusive(v___x_931_);
if (v_isSharedCheck_954_ == 0)
{
v___x_949_ = v___x_931_;
v_isShared_950_ = v_isSharedCheck_954_;
goto v_resetjp_948_;
}
else
{
lean_inc(v_a_947_);
lean_dec(v___x_931_);
v___x_949_ = lean_box(0);
v_isShared_950_ = v_isSharedCheck_954_;
goto v_resetjp_948_;
}
v_resetjp_948_:
{
lean_object* v___x_952_; 
if (v_isShared_950_ == 0)
{
v___x_952_ = v___x_949_;
goto v_reusejp_951_;
}
else
{
lean_object* v_reuseFailAlloc_953_; 
v_reuseFailAlloc_953_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_953_, 0, v_a_947_);
v___x_952_ = v_reuseFailAlloc_953_;
goto v_reusejp_951_;
}
v_reusejp_951_:
{
return v___x_952_;
}
}
}
}
else
{
lean_object* v_vs_955_; lean_object* v___x_956_; lean_object* v___x_957_; size_t v_sz_958_; size_t v___x_959_; lean_object* v___x_960_; 
v_vs_955_ = lean_ctor_get(v_n_915_, 0);
v___x_956_ = lean_box(0);
v___x_957_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_957_, 0, v___x_956_);
lean_ctor_set(v___x_957_, 1, v_b_916_);
v_sz_958_ = lean_array_size(v_vs_955_);
v___x_959_ = ((size_t)0ULL);
v___x_960_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9(v_vs_955_, v_sz_958_, v___x_959_, v___x_957_, v___y_917_, v___y_918_, v___y_919_, v___y_920_, v___y_921_, v___y_922_, v___y_923_, v___y_924_);
if (lean_obj_tag(v___x_960_) == 0)
{
lean_object* v_a_961_; lean_object* v___x_963_; uint8_t v_isShared_964_; uint8_t v_isSharedCheck_975_; 
v_a_961_ = lean_ctor_get(v___x_960_, 0);
v_isSharedCheck_975_ = !lean_is_exclusive(v___x_960_);
if (v_isSharedCheck_975_ == 0)
{
v___x_963_ = v___x_960_;
v_isShared_964_ = v_isSharedCheck_975_;
goto v_resetjp_962_;
}
else
{
lean_inc(v_a_961_);
lean_dec(v___x_960_);
v___x_963_ = lean_box(0);
v_isShared_964_ = v_isSharedCheck_975_;
goto v_resetjp_962_;
}
v_resetjp_962_:
{
lean_object* v_fst_965_; 
v_fst_965_ = lean_ctor_get(v_a_961_, 0);
if (lean_obj_tag(v_fst_965_) == 0)
{
lean_object* v_snd_966_; lean_object* v___x_967_; lean_object* v___x_969_; 
v_snd_966_ = lean_ctor_get(v_a_961_, 1);
lean_inc(v_snd_966_);
lean_dec(v_a_961_);
v___x_967_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_967_, 0, v_snd_966_);
if (v_isShared_964_ == 0)
{
lean_ctor_set(v___x_963_, 0, v___x_967_);
v___x_969_ = v___x_963_;
goto v_reusejp_968_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v___x_967_);
v___x_969_ = v_reuseFailAlloc_970_;
goto v_reusejp_968_;
}
v_reusejp_968_:
{
return v___x_969_;
}
}
else
{
lean_object* v_val_971_; lean_object* v___x_973_; 
lean_inc_ref(v_fst_965_);
lean_dec(v_a_961_);
v_val_971_ = lean_ctor_get(v_fst_965_, 0);
lean_inc(v_val_971_);
lean_dec_ref_known(v_fst_965_, 1);
if (v_isShared_964_ == 0)
{
lean_ctor_set(v___x_963_, 0, v_val_971_);
v___x_973_ = v___x_963_;
goto v_reusejp_972_;
}
else
{
lean_object* v_reuseFailAlloc_974_; 
v_reuseFailAlloc_974_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_974_, 0, v_val_971_);
v___x_973_ = v_reuseFailAlloc_974_;
goto v_reusejp_972_;
}
v_reusejp_972_:
{
return v___x_973_;
}
}
}
}
else
{
lean_object* v_a_976_; lean_object* v___x_978_; uint8_t v_isShared_979_; uint8_t v_isSharedCheck_983_; 
v_a_976_ = lean_ctor_get(v___x_960_, 0);
v_isSharedCheck_983_ = !lean_is_exclusive(v___x_960_);
if (v_isSharedCheck_983_ == 0)
{
v___x_978_ = v___x_960_;
v_isShared_979_ = v_isSharedCheck_983_;
goto v_resetjp_977_;
}
else
{
lean_inc(v_a_976_);
lean_dec(v___x_960_);
v___x_978_ = lean_box(0);
v_isShared_979_ = v_isSharedCheck_983_;
goto v_resetjp_977_;
}
v_resetjp_977_:
{
lean_object* v___x_981_; 
if (v_isShared_979_ == 0)
{
v___x_981_ = v___x_978_;
goto v_reusejp_980_;
}
else
{
lean_object* v_reuseFailAlloc_982_; 
v_reuseFailAlloc_982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_982_, 0, v_a_976_);
v___x_981_ = v_reuseFailAlloc_982_;
goto v_reusejp_980_;
}
v_reusejp_980_:
{
return v___x_981_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__8(lean_object* v_init_984_, lean_object* v_as_985_, size_t v_sz_986_, size_t v_i_987_, lean_object* v_b_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_, lean_object* v___y_992_, lean_object* v___y_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_){
_start:
{
uint8_t v___x_998_; 
v___x_998_ = lean_usize_dec_lt(v_i_987_, v_sz_986_);
if (v___x_998_ == 0)
{
lean_object* v___x_999_; 
v___x_999_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_999_, 0, v_b_988_);
return v___x_999_;
}
else
{
lean_object* v_snd_1000_; lean_object* v___x_1002_; uint8_t v_isShared_1003_; uint8_t v_isSharedCheck_1034_; 
v_snd_1000_ = lean_ctor_get(v_b_988_, 1);
v_isSharedCheck_1034_ = !lean_is_exclusive(v_b_988_);
if (v_isSharedCheck_1034_ == 0)
{
lean_object* v_unused_1035_; 
v_unused_1035_ = lean_ctor_get(v_b_988_, 0);
lean_dec(v_unused_1035_);
v___x_1002_ = v_b_988_;
v_isShared_1003_ = v_isSharedCheck_1034_;
goto v_resetjp_1001_;
}
else
{
lean_inc(v_snd_1000_);
lean_dec(v_b_988_);
v___x_1002_ = lean_box(0);
v_isShared_1003_ = v_isSharedCheck_1034_;
goto v_resetjp_1001_;
}
v_resetjp_1001_:
{
lean_object* v_a_1004_; lean_object* v___x_1005_; 
v_a_1004_ = lean_array_uget_borrowed(v_as_985_, v_i_987_);
lean_inc(v_snd_1000_);
v___x_1005_ = lp_plausible_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3(v_init_984_, v_a_1004_, v_snd_1000_, v___y_989_, v___y_990_, v___y_991_, v___y_992_, v___y_993_, v___y_994_, v___y_995_, v___y_996_);
if (lean_obj_tag(v___x_1005_) == 0)
{
lean_object* v_a_1006_; lean_object* v___x_1008_; uint8_t v_isShared_1009_; uint8_t v_isSharedCheck_1025_; 
v_a_1006_ = lean_ctor_get(v___x_1005_, 0);
v_isSharedCheck_1025_ = !lean_is_exclusive(v___x_1005_);
if (v_isSharedCheck_1025_ == 0)
{
v___x_1008_ = v___x_1005_;
v_isShared_1009_ = v_isSharedCheck_1025_;
goto v_resetjp_1007_;
}
else
{
lean_inc(v_a_1006_);
lean_dec(v___x_1005_);
v___x_1008_ = lean_box(0);
v_isShared_1009_ = v_isSharedCheck_1025_;
goto v_resetjp_1007_;
}
v_resetjp_1007_:
{
if (lean_obj_tag(v_a_1006_) == 0)
{
lean_object* v___x_1010_; lean_object* v___x_1012_; 
v___x_1010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1010_, 0, v_a_1006_);
if (v_isShared_1003_ == 0)
{
lean_ctor_set(v___x_1002_, 0, v___x_1010_);
v___x_1012_ = v___x_1002_;
goto v_reusejp_1011_;
}
else
{
lean_object* v_reuseFailAlloc_1016_; 
v_reuseFailAlloc_1016_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1016_, 0, v___x_1010_);
lean_ctor_set(v_reuseFailAlloc_1016_, 1, v_snd_1000_);
v___x_1012_ = v_reuseFailAlloc_1016_;
goto v_reusejp_1011_;
}
v_reusejp_1011_:
{
lean_object* v___x_1014_; 
if (v_isShared_1009_ == 0)
{
lean_ctor_set(v___x_1008_, 0, v___x_1012_);
v___x_1014_ = v___x_1008_;
goto v_reusejp_1013_;
}
else
{
lean_object* v_reuseFailAlloc_1015_; 
v_reuseFailAlloc_1015_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1015_, 0, v___x_1012_);
v___x_1014_ = v_reuseFailAlloc_1015_;
goto v_reusejp_1013_;
}
v_reusejp_1013_:
{
return v___x_1014_;
}
}
}
else
{
lean_object* v_a_1017_; lean_object* v___x_1018_; lean_object* v___x_1020_; 
lean_del_object(v___x_1008_);
lean_dec(v_snd_1000_);
v_a_1017_ = lean_ctor_get(v_a_1006_, 0);
lean_inc(v_a_1017_);
lean_dec_ref_known(v_a_1006_, 1);
v___x_1018_ = lean_box(0);
if (v_isShared_1003_ == 0)
{
lean_ctor_set(v___x_1002_, 1, v_a_1017_);
lean_ctor_set(v___x_1002_, 0, v___x_1018_);
v___x_1020_ = v___x_1002_;
goto v_reusejp_1019_;
}
else
{
lean_object* v_reuseFailAlloc_1024_; 
v_reuseFailAlloc_1024_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1024_, 0, v___x_1018_);
lean_ctor_set(v_reuseFailAlloc_1024_, 1, v_a_1017_);
v___x_1020_ = v_reuseFailAlloc_1024_;
goto v_reusejp_1019_;
}
v_reusejp_1019_:
{
size_t v___x_1021_; size_t v___x_1022_; 
v___x_1021_ = ((size_t)1ULL);
v___x_1022_ = lean_usize_add(v_i_987_, v___x_1021_);
v_i_987_ = v___x_1022_;
v_b_988_ = v___x_1020_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1026_; lean_object* v___x_1028_; uint8_t v_isShared_1029_; uint8_t v_isSharedCheck_1033_; 
lean_del_object(v___x_1002_);
lean_dec(v_snd_1000_);
v_a_1026_ = lean_ctor_get(v___x_1005_, 0);
v_isSharedCheck_1033_ = !lean_is_exclusive(v___x_1005_);
if (v_isSharedCheck_1033_ == 0)
{
v___x_1028_ = v___x_1005_;
v_isShared_1029_ = v_isSharedCheck_1033_;
goto v_resetjp_1027_;
}
else
{
lean_inc(v_a_1026_);
lean_dec(v___x_1005_);
v___x_1028_ = lean_box(0);
v_isShared_1029_ = v_isSharedCheck_1033_;
goto v_resetjp_1027_;
}
v_resetjp_1027_:
{
lean_object* v___x_1031_; 
if (v_isShared_1029_ == 0)
{
v___x_1031_ = v___x_1028_;
goto v_reusejp_1030_;
}
else
{
lean_object* v_reuseFailAlloc_1032_; 
v_reuseFailAlloc_1032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1032_, 0, v_a_1026_);
v___x_1031_ = v_reuseFailAlloc_1032_;
goto v_reusejp_1030_;
}
v_reusejp_1030_:
{
return v___x_1031_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__8___boxed(lean_object* v_init_1036_, lean_object* v_as_1037_, lean_object* v_sz_1038_, lean_object* v_i_1039_, lean_object* v_b_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_){
_start:
{
size_t v_sz_boxed_1050_; size_t v_i_boxed_1051_; lean_object* v_res_1052_; 
v_sz_boxed_1050_ = lean_unbox_usize(v_sz_1038_);
lean_dec(v_sz_1038_);
v_i_boxed_1051_ = lean_unbox_usize(v_i_1039_);
lean_dec(v_i_1039_);
v_res_1052_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__8(v_init_1036_, v_as_1037_, v_sz_boxed_1050_, v_i_boxed_1051_, v_b_1040_, v___y_1041_, v___y_1042_, v___y_1043_, v___y_1044_, v___y_1045_, v___y_1046_, v___y_1047_, v___y_1048_);
lean_dec(v___y_1048_);
lean_dec_ref(v___y_1047_);
lean_dec(v___y_1046_);
lean_dec_ref(v___y_1045_);
lean_dec(v___y_1044_);
lean_dec_ref(v___y_1043_);
lean_dec(v___y_1042_);
lean_dec_ref(v___y_1041_);
lean_dec_ref(v_as_1037_);
lean_dec_ref(v_init_1036_);
return v_res_1052_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3___boxed(lean_object* v_init_1053_, lean_object* v_n_1054_, lean_object* v_b_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_){
_start:
{
lean_object* v_res_1065_; 
v_res_1065_ = lp_plausible_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3(v_init_1053_, v_n_1054_, v_b_1055_, v___y_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_, v___y_1062_, v___y_1063_);
lean_dec(v___y_1063_);
lean_dec_ref(v___y_1062_);
lean_dec(v___y_1061_);
lean_dec_ref(v___y_1060_);
lean_dec(v___y_1059_);
lean_dec_ref(v___y_1058_);
lean_dec(v___y_1057_);
lean_dec_ref(v___y_1056_);
lean_dec_ref(v_n_1054_);
lean_dec_ref(v_init_1053_);
return v_res_1065_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11___redArg(lean_object* v_as_1066_, size_t v_sz_1067_, size_t v_i_1068_, lean_object* v_b_1069_){
_start:
{
uint8_t v___x_1071_; 
v___x_1071_ = lean_usize_dec_lt(v_i_1068_, v_sz_1067_);
if (v___x_1071_ == 0)
{
lean_object* v___x_1072_; 
v___x_1072_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1072_, 0, v_b_1069_);
return v___x_1072_;
}
else
{
lean_object* v_snd_1073_; lean_object* v___x_1075_; uint8_t v_isShared_1076_; uint8_t v_isSharedCheck_1091_; 
v_snd_1073_ = lean_ctor_get(v_b_1069_, 1);
v_isSharedCheck_1091_ = !lean_is_exclusive(v_b_1069_);
if (v_isSharedCheck_1091_ == 0)
{
lean_object* v_unused_1092_; 
v_unused_1092_ = lean_ctor_get(v_b_1069_, 0);
lean_dec(v_unused_1092_);
v___x_1075_ = v_b_1069_;
v_isShared_1076_ = v_isSharedCheck_1091_;
goto v_resetjp_1074_;
}
else
{
lean_inc(v_snd_1073_);
lean_dec(v_b_1069_);
v___x_1075_ = lean_box(0);
v_isShared_1076_ = v_isSharedCheck_1091_;
goto v_resetjp_1074_;
}
v_resetjp_1074_:
{
lean_object* v___x_1077_; lean_object* v_a_1079_; lean_object* v_a_1086_; 
v___x_1077_ = lean_box(0);
v_a_1086_ = lean_array_uget_borrowed(v_as_1066_, v_i_1068_);
if (lean_obj_tag(v_a_1086_) == 0)
{
v_a_1079_ = v_snd_1073_;
goto v___jp_1078_;
}
else
{
lean_object* v_val_1087_; uint8_t v___x_1088_; 
v_val_1087_ = lean_ctor_get(v_a_1086_, 0);
v___x_1088_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1087_);
if (v___x_1088_ == 0)
{
lean_object* v___x_1089_; lean_object* v___x_1090_; 
lean_inc(v_val_1087_);
v___x_1089_ = l_Lean_LocalDecl_toExpr(v_val_1087_);
v___x_1090_ = lean_array_push(v_snd_1073_, v___x_1089_);
v_a_1079_ = v___x_1090_;
goto v___jp_1078_;
}
else
{
v_a_1079_ = v_snd_1073_;
goto v___jp_1078_;
}
}
v___jp_1078_:
{
lean_object* v___x_1081_; 
if (v_isShared_1076_ == 0)
{
lean_ctor_set(v___x_1075_, 1, v_a_1079_);
lean_ctor_set(v___x_1075_, 0, v___x_1077_);
v___x_1081_ = v___x_1075_;
goto v_reusejp_1080_;
}
else
{
lean_object* v_reuseFailAlloc_1085_; 
v_reuseFailAlloc_1085_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1085_, 0, v___x_1077_);
lean_ctor_set(v_reuseFailAlloc_1085_, 1, v_a_1079_);
v___x_1081_ = v_reuseFailAlloc_1085_;
goto v_reusejp_1080_;
}
v_reusejp_1080_:
{
size_t v___x_1082_; size_t v___x_1083_; 
v___x_1082_ = ((size_t)1ULL);
v___x_1083_ = lean_usize_add(v_i_1068_, v___x_1082_);
v_i_1068_ = v___x_1083_;
v_b_1069_ = v___x_1081_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11___redArg___boxed(lean_object* v_as_1093_, lean_object* v_sz_1094_, lean_object* v_i_1095_, lean_object* v_b_1096_, lean_object* v___y_1097_){
_start:
{
size_t v_sz_boxed_1098_; size_t v_i_boxed_1099_; lean_object* v_res_1100_; 
v_sz_boxed_1098_ = lean_unbox_usize(v_sz_1094_);
lean_dec(v_sz_1094_);
v_i_boxed_1099_ = lean_unbox_usize(v_i_1095_);
lean_dec(v_i_1095_);
v_res_1100_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11___redArg(v_as_1093_, v_sz_boxed_1098_, v_i_boxed_1099_, v_b_1096_);
lean_dec_ref(v_as_1093_);
return v_res_1100_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4(lean_object* v_as_1101_, size_t v_sz_1102_, size_t v_i_1103_, lean_object* v_b_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_){
_start:
{
uint8_t v___x_1114_; 
v___x_1114_ = lean_usize_dec_lt(v_i_1103_, v_sz_1102_);
if (v___x_1114_ == 0)
{
lean_object* v___x_1115_; 
v___x_1115_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1115_, 0, v_b_1104_);
return v___x_1115_;
}
else
{
lean_object* v_snd_1116_; lean_object* v___x_1118_; uint8_t v_isShared_1119_; uint8_t v_isSharedCheck_1134_; 
v_snd_1116_ = lean_ctor_get(v_b_1104_, 1);
v_isSharedCheck_1134_ = !lean_is_exclusive(v_b_1104_);
if (v_isSharedCheck_1134_ == 0)
{
lean_object* v_unused_1135_; 
v_unused_1135_ = lean_ctor_get(v_b_1104_, 0);
lean_dec(v_unused_1135_);
v___x_1118_ = v_b_1104_;
v_isShared_1119_ = v_isSharedCheck_1134_;
goto v_resetjp_1117_;
}
else
{
lean_inc(v_snd_1116_);
lean_dec(v_b_1104_);
v___x_1118_ = lean_box(0);
v_isShared_1119_ = v_isSharedCheck_1134_;
goto v_resetjp_1117_;
}
v_resetjp_1117_:
{
lean_object* v___x_1120_; lean_object* v_a_1122_; lean_object* v_a_1129_; 
v___x_1120_ = lean_box(0);
v_a_1129_ = lean_array_uget_borrowed(v_as_1101_, v_i_1103_);
if (lean_obj_tag(v_a_1129_) == 0)
{
v_a_1122_ = v_snd_1116_;
goto v___jp_1121_;
}
else
{
lean_object* v_val_1130_; uint8_t v___x_1131_; 
v_val_1130_ = lean_ctor_get(v_a_1129_, 0);
v___x_1131_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1130_);
if (v___x_1131_ == 0)
{
lean_object* v___x_1132_; lean_object* v___x_1133_; 
lean_inc(v_val_1130_);
v___x_1132_ = l_Lean_LocalDecl_toExpr(v_val_1130_);
v___x_1133_ = lean_array_push(v_snd_1116_, v___x_1132_);
v_a_1122_ = v___x_1133_;
goto v___jp_1121_;
}
else
{
v_a_1122_ = v_snd_1116_;
goto v___jp_1121_;
}
}
v___jp_1121_:
{
lean_object* v___x_1124_; 
if (v_isShared_1119_ == 0)
{
lean_ctor_set(v___x_1118_, 1, v_a_1122_);
lean_ctor_set(v___x_1118_, 0, v___x_1120_);
v___x_1124_ = v___x_1118_;
goto v_reusejp_1123_;
}
else
{
lean_object* v_reuseFailAlloc_1128_; 
v_reuseFailAlloc_1128_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1128_, 0, v___x_1120_);
lean_ctor_set(v_reuseFailAlloc_1128_, 1, v_a_1122_);
v___x_1124_ = v_reuseFailAlloc_1128_;
goto v_reusejp_1123_;
}
v_reusejp_1123_:
{
size_t v___x_1125_; size_t v___x_1126_; lean_object* v___x_1127_; 
v___x_1125_ = ((size_t)1ULL);
v___x_1126_ = lean_usize_add(v_i_1103_, v___x_1125_);
v___x_1127_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11___redArg(v_as_1101_, v_sz_1102_, v___x_1126_, v___x_1124_);
return v___x_1127_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4___boxed(lean_object* v_as_1136_, lean_object* v_sz_1137_, lean_object* v_i_1138_, lean_object* v_b_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_){
_start:
{
size_t v_sz_boxed_1149_; size_t v_i_boxed_1150_; lean_object* v_res_1151_; 
v_sz_boxed_1149_ = lean_unbox_usize(v_sz_1137_);
lean_dec(v_sz_1137_);
v_i_boxed_1150_ = lean_unbox_usize(v_i_1138_);
lean_dec(v_i_1138_);
v_res_1151_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4(v_as_1136_, v_sz_boxed_1149_, v_i_boxed_1150_, v_b_1139_, v___y_1140_, v___y_1141_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_);
lean_dec(v___y_1147_);
lean_dec_ref(v___y_1146_);
lean_dec(v___y_1145_);
lean_dec_ref(v___y_1144_);
lean_dec(v___y_1143_);
lean_dec_ref(v___y_1142_);
lean_dec(v___y_1141_);
lean_dec_ref(v___y_1140_);
lean_dec_ref(v_as_1136_);
return v_res_1151_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1(lean_object* v_t_1152_, lean_object* v_init_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_){
_start:
{
lean_object* v_root_1163_; lean_object* v_tail_1164_; lean_object* v___x_1165_; 
v_root_1163_ = lean_ctor_get(v_t_1152_, 0);
v_tail_1164_ = lean_ctor_get(v_t_1152_, 1);
lean_inc_ref(v_init_1153_);
v___x_1165_ = lp_plausible_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3(v_init_1153_, v_root_1163_, v_init_1153_, v___y_1154_, v___y_1155_, v___y_1156_, v___y_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_);
lean_dec_ref(v_init_1153_);
if (lean_obj_tag(v___x_1165_) == 0)
{
lean_object* v_a_1166_; lean_object* v___x_1168_; uint8_t v_isShared_1169_; uint8_t v_isSharedCheck_1202_; 
v_a_1166_ = lean_ctor_get(v___x_1165_, 0);
v_isSharedCheck_1202_ = !lean_is_exclusive(v___x_1165_);
if (v_isSharedCheck_1202_ == 0)
{
v___x_1168_ = v___x_1165_;
v_isShared_1169_ = v_isSharedCheck_1202_;
goto v_resetjp_1167_;
}
else
{
lean_inc(v_a_1166_);
lean_dec(v___x_1165_);
v___x_1168_ = lean_box(0);
v_isShared_1169_ = v_isSharedCheck_1202_;
goto v_resetjp_1167_;
}
v_resetjp_1167_:
{
if (lean_obj_tag(v_a_1166_) == 0)
{
lean_object* v_a_1170_; lean_object* v___x_1172_; 
v_a_1170_ = lean_ctor_get(v_a_1166_, 0);
lean_inc(v_a_1170_);
lean_dec_ref_known(v_a_1166_, 1);
if (v_isShared_1169_ == 0)
{
lean_ctor_set(v___x_1168_, 0, v_a_1170_);
v___x_1172_ = v___x_1168_;
goto v_reusejp_1171_;
}
else
{
lean_object* v_reuseFailAlloc_1173_; 
v_reuseFailAlloc_1173_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1173_, 0, v_a_1170_);
v___x_1172_ = v_reuseFailAlloc_1173_;
goto v_reusejp_1171_;
}
v_reusejp_1171_:
{
return v___x_1172_;
}
}
else
{
lean_object* v_a_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; size_t v_sz_1177_; size_t v___x_1178_; lean_object* v___x_1179_; 
lean_del_object(v___x_1168_);
v_a_1174_ = lean_ctor_get(v_a_1166_, 0);
lean_inc(v_a_1174_);
lean_dec_ref_known(v_a_1166_, 1);
v___x_1175_ = lean_box(0);
v___x_1176_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1176_, 0, v___x_1175_);
lean_ctor_set(v___x_1176_, 1, v_a_1174_);
v_sz_1177_ = lean_array_size(v_tail_1164_);
v___x_1178_ = ((size_t)0ULL);
v___x_1179_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4(v_tail_1164_, v_sz_1177_, v___x_1178_, v___x_1176_, v___y_1154_, v___y_1155_, v___y_1156_, v___y_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_);
if (lean_obj_tag(v___x_1179_) == 0)
{
lean_object* v_a_1180_; lean_object* v___x_1182_; uint8_t v_isShared_1183_; uint8_t v_isSharedCheck_1193_; 
v_a_1180_ = lean_ctor_get(v___x_1179_, 0);
v_isSharedCheck_1193_ = !lean_is_exclusive(v___x_1179_);
if (v_isSharedCheck_1193_ == 0)
{
v___x_1182_ = v___x_1179_;
v_isShared_1183_ = v_isSharedCheck_1193_;
goto v_resetjp_1181_;
}
else
{
lean_inc(v_a_1180_);
lean_dec(v___x_1179_);
v___x_1182_ = lean_box(0);
v_isShared_1183_ = v_isSharedCheck_1193_;
goto v_resetjp_1181_;
}
v_resetjp_1181_:
{
lean_object* v_fst_1184_; 
v_fst_1184_ = lean_ctor_get(v_a_1180_, 0);
if (lean_obj_tag(v_fst_1184_) == 0)
{
lean_object* v_snd_1185_; lean_object* v___x_1187_; 
v_snd_1185_ = lean_ctor_get(v_a_1180_, 1);
lean_inc(v_snd_1185_);
lean_dec(v_a_1180_);
if (v_isShared_1183_ == 0)
{
lean_ctor_set(v___x_1182_, 0, v_snd_1185_);
v___x_1187_ = v___x_1182_;
goto v_reusejp_1186_;
}
else
{
lean_object* v_reuseFailAlloc_1188_; 
v_reuseFailAlloc_1188_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1188_, 0, v_snd_1185_);
v___x_1187_ = v_reuseFailAlloc_1188_;
goto v_reusejp_1186_;
}
v_reusejp_1186_:
{
return v___x_1187_;
}
}
else
{
lean_object* v_val_1189_; lean_object* v___x_1191_; 
lean_inc_ref(v_fst_1184_);
lean_dec(v_a_1180_);
v_val_1189_ = lean_ctor_get(v_fst_1184_, 0);
lean_inc(v_val_1189_);
lean_dec_ref_known(v_fst_1184_, 1);
if (v_isShared_1183_ == 0)
{
lean_ctor_set(v___x_1182_, 0, v_val_1189_);
v___x_1191_ = v___x_1182_;
goto v_reusejp_1190_;
}
else
{
lean_object* v_reuseFailAlloc_1192_; 
v_reuseFailAlloc_1192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1192_, 0, v_val_1189_);
v___x_1191_ = v_reuseFailAlloc_1192_;
goto v_reusejp_1190_;
}
v_reusejp_1190_:
{
return v___x_1191_;
}
}
}
}
else
{
lean_object* v_a_1194_; lean_object* v___x_1196_; uint8_t v_isShared_1197_; uint8_t v_isSharedCheck_1201_; 
v_a_1194_ = lean_ctor_get(v___x_1179_, 0);
v_isSharedCheck_1201_ = !lean_is_exclusive(v___x_1179_);
if (v_isSharedCheck_1201_ == 0)
{
v___x_1196_ = v___x_1179_;
v_isShared_1197_ = v_isSharedCheck_1201_;
goto v_resetjp_1195_;
}
else
{
lean_inc(v_a_1194_);
lean_dec(v___x_1179_);
v___x_1196_ = lean_box(0);
v_isShared_1197_ = v_isSharedCheck_1201_;
goto v_resetjp_1195_;
}
v_resetjp_1195_:
{
lean_object* v___x_1199_; 
if (v_isShared_1197_ == 0)
{
v___x_1199_ = v___x_1196_;
goto v_reusejp_1198_;
}
else
{
lean_object* v_reuseFailAlloc_1200_; 
v_reuseFailAlloc_1200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1200_, 0, v_a_1194_);
v___x_1199_ = v_reuseFailAlloc_1200_;
goto v_reusejp_1198_;
}
v_reusejp_1198_:
{
return v___x_1199_;
}
}
}
}
}
}
else
{
lean_object* v_a_1203_; lean_object* v___x_1205_; uint8_t v_isShared_1206_; uint8_t v_isSharedCheck_1210_; 
v_a_1203_ = lean_ctor_get(v___x_1165_, 0);
v_isSharedCheck_1210_ = !lean_is_exclusive(v___x_1165_);
if (v_isSharedCheck_1210_ == 0)
{
v___x_1205_ = v___x_1165_;
v_isShared_1206_ = v_isSharedCheck_1210_;
goto v_resetjp_1204_;
}
else
{
lean_inc(v_a_1203_);
lean_dec(v___x_1165_);
v___x_1205_ = lean_box(0);
v_isShared_1206_ = v_isSharedCheck_1210_;
goto v_resetjp_1204_;
}
v_resetjp_1204_:
{
lean_object* v___x_1208_; 
if (v_isShared_1206_ == 0)
{
v___x_1208_ = v___x_1205_;
goto v_reusejp_1207_;
}
else
{
lean_object* v_reuseFailAlloc_1209_; 
v_reuseFailAlloc_1209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1209_, 0, v_a_1203_);
v___x_1208_ = v_reuseFailAlloc_1209_;
goto v_reusejp_1207_;
}
v_reusejp_1207_:
{
return v___x_1208_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1___boxed(lean_object* v_t_1211_, lean_object* v_init_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_, lean_object* v___y_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_){
_start:
{
lean_object* v_res_1222_; 
v_res_1222_ = lp_plausible_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1(v_t_1211_, v_init_1212_, v___y_1213_, v___y_1214_, v___y_1215_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_, v___y_1220_);
lean_dec(v___y_1220_);
lean_dec_ref(v___y_1219_);
lean_dec(v___y_1218_);
lean_dec_ref(v___y_1217_);
lean_dec(v___y_1216_);
lean_dec_ref(v___y_1215_);
lean_dec(v___y_1214_);
lean_dec_ref(v___y_1213_);
lean_dec_ref(v_t_1211_);
return v_res_1222_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1(lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_){
_start:
{
lean_object* v_lctx_1234_; lean_object* v_decls_1235_; lean_object* v_hs_1236_; lean_object* v___x_1237_; 
v_lctx_1234_ = lean_ctor_get(v___y_1229_, 2);
v_decls_1235_ = lean_ctor_get(v_lctx_1234_, 1);
v_hs_1236_ = ((lean_object*)(lp_plausible_Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1___closed__0));
v___x_1237_ = lp_plausible_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1(v_decls_1235_, v_hs_1236_, v___y_1225_, v___y_1226_, v___y_1227_, v___y_1228_, v___y_1229_, v___y_1230_, v___y_1231_, v___y_1232_);
return v___x_1237_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1___boxed(lean_object* v___y_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_){
_start:
{
lean_object* v_res_1247_; 
v_res_1247_ = lp_plausible_Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1(v___y_1238_, v___y_1239_, v___y_1240_, v___y_1241_, v___y_1242_, v___y_1243_, v___y_1244_, v___y_1245_);
lean_dec(v___y_1245_);
lean_dec_ref(v___y_1244_);
lean_dec(v___y_1243_);
lean_dec_ref(v___y_1242_);
lean_dec(v___y_1241_);
lean_dec_ref(v___y_1240_);
lean_dec(v___y_1239_);
lean_dec_ref(v___y_1238_);
return v_res_1247_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__2(size_t v_sz_1248_, size_t v_i_1249_, lean_object* v_bs_1250_){
_start:
{
uint8_t v___x_1251_; 
v___x_1251_ = lean_usize_dec_lt(v_i_1249_, v_sz_1248_);
if (v___x_1251_ == 0)
{
return v_bs_1250_;
}
else
{
lean_object* v_v_1252_; lean_object* v___x_1253_; lean_object* v_bs_x27_1254_; lean_object* v___x_1255_; size_t v___x_1256_; size_t v___x_1257_; lean_object* v___x_1258_; 
v_v_1252_ = lean_array_uget(v_bs_1250_, v_i_1249_);
v___x_1253_ = lean_unsigned_to_nat(0u);
v_bs_x27_1254_ = lean_array_uset(v_bs_1250_, v_i_1249_, v___x_1253_);
v___x_1255_ = l_Lean_Expr_fvarId_x21(v_v_1252_);
lean_dec(v_v_1252_);
v___x_1256_ = ((size_t)1ULL);
v___x_1257_ = lean_usize_add(v_i_1249_, v___x_1256_);
v___x_1258_ = lean_array_uset(v_bs_x27_1254_, v_i_1249_, v___x_1255_);
v_i_1249_ = v___x_1257_;
v_bs_1250_ = v___x_1258_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__2___boxed(lean_object* v_sz_1260_, lean_object* v_i_1261_, lean_object* v_bs_1262_){
_start:
{
size_t v_sz_boxed_1263_; size_t v_i_boxed_1264_; lean_object* v_res_1265_; 
v_sz_boxed_1263_ = lean_unbox_usize(v_sz_1260_);
lean_dec(v_sz_1260_);
v_i_boxed_1264_ = lean_unbox_usize(v_i_1261_);
lean_dec(v_i_1261_);
v_res_1265_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__2(v_sz_boxed_1263_, v_i_boxed_1264_, v_bs_1262_);
return v_res_1265_;
}
}
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__1(lean_object* v___x_1266_, lean_object* v___x_1267_, uint8_t v___x_1268_, uint8_t v___x_1269_, lean_object* v___x_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_){
_start:
{
lean_object* v___x_1280_; 
v___x_1280_ = lp_plausible_Plausible_elabConfig___redArg(v___x_1266_, v___x_1267_, v___x_1268_, v___y_1271_, v___y_1277_, v___y_1278_);
if (lean_obj_tag(v___x_1280_) == 0)
{
lean_object* v_a_1281_; lean_object* v___x_1282_; 
v_a_1281_ = lean_ctor_get(v___x_1280_, 0);
lean_inc(v_a_1281_);
lean_dec_ref_known(v___x_1280_, 1);
v___x_1282_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1272_, v___y_1275_, v___y_1276_, v___y_1277_, v___y_1278_);
if (lean_obj_tag(v___x_1282_) == 0)
{
lean_object* v_a_1283_; lean_object* v___x_1284_; 
v_a_1283_ = lean_ctor_get(v___x_1282_, 0);
lean_inc(v_a_1283_);
lean_dec_ref_known(v___x_1282_, 1);
v___x_1284_ = lp_plausible_Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1(v___y_1271_, v___y_1272_, v___y_1273_, v___y_1274_, v___y_1275_, v___y_1276_, v___y_1277_, v___y_1278_);
if (lean_obj_tag(v___x_1284_) == 0)
{
lean_object* v_a_1285_; size_t v_sz_1286_; size_t v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; 
v_a_1285_ = lean_ctor_get(v___x_1284_, 0);
lean_inc(v_a_1285_);
lean_dec_ref_known(v___x_1284_, 1);
v_sz_1286_ = lean_array_size(v_a_1285_);
v___x_1287_ = ((size_t)0ULL);
v___x_1288_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__2(v_sz_1286_, v___x_1287_, v_a_1285_);
v___x_1289_ = l_Lean_MVarId_revert(v_a_1283_, v___x_1288_, v___x_1269_, v___x_1269_, v___y_1275_, v___y_1276_, v___y_1277_, v___y_1278_);
if (lean_obj_tag(v___x_1289_) == 0)
{
lean_object* v_a_1290_; lean_object* v_snd_1291_; lean_object* v___x_1292_; lean_object* v___f_1293_; lean_object* v___x_1294_; 
v_a_1290_ = lean_ctor_get(v___x_1289_, 0);
lean_inc(v_a_1290_);
lean_dec_ref_known(v___x_1289_, 1);
v_snd_1291_ = lean_ctor_get(v_a_1290_, 1);
lean_inc_n(v_snd_1291_, 2);
lean_dec(v_a_1290_);
v___x_1292_ = lean_box(v___x_1268_);
v___f_1293_ = lean_alloc_closure((void*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__0___boxed), 13, 4);
lean_closure_set(v___f_1293_, 0, v_snd_1291_);
lean_closure_set(v___f_1293_, 1, v___x_1292_);
lean_closure_set(v___f_1293_, 2, v___x_1270_);
lean_closure_set(v___f_1293_, 3, v_a_1281_);
v___x_1294_ = lp_plausible_Lean_MVarId_withContext___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__5___redArg(v_snd_1291_, v___f_1293_, v___y_1271_, v___y_1272_, v___y_1273_, v___y_1274_, v___y_1275_, v___y_1276_, v___y_1277_, v___y_1278_);
return v___x_1294_;
}
else
{
lean_object* v_a_1295_; lean_object* v___x_1297_; uint8_t v_isShared_1298_; uint8_t v_isSharedCheck_1302_; 
lean_dec(v_a_1281_);
lean_dec(v___x_1270_);
v_a_1295_ = lean_ctor_get(v___x_1289_, 0);
v_isSharedCheck_1302_ = !lean_is_exclusive(v___x_1289_);
if (v_isSharedCheck_1302_ == 0)
{
v___x_1297_ = v___x_1289_;
v_isShared_1298_ = v_isSharedCheck_1302_;
goto v_resetjp_1296_;
}
else
{
lean_inc(v_a_1295_);
lean_dec(v___x_1289_);
v___x_1297_ = lean_box(0);
v_isShared_1298_ = v_isSharedCheck_1302_;
goto v_resetjp_1296_;
}
v_resetjp_1296_:
{
lean_object* v___x_1300_; 
if (v_isShared_1298_ == 0)
{
v___x_1300_ = v___x_1297_;
goto v_reusejp_1299_;
}
else
{
lean_object* v_reuseFailAlloc_1301_; 
v_reuseFailAlloc_1301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1301_, 0, v_a_1295_);
v___x_1300_ = v_reuseFailAlloc_1301_;
goto v_reusejp_1299_;
}
v_reusejp_1299_:
{
return v___x_1300_;
}
}
}
}
else
{
lean_object* v_a_1303_; lean_object* v___x_1305_; uint8_t v_isShared_1306_; uint8_t v_isSharedCheck_1310_; 
lean_dec(v_a_1283_);
lean_dec(v_a_1281_);
lean_dec(v___x_1270_);
v_a_1303_ = lean_ctor_get(v___x_1284_, 0);
v_isSharedCheck_1310_ = !lean_is_exclusive(v___x_1284_);
if (v_isSharedCheck_1310_ == 0)
{
v___x_1305_ = v___x_1284_;
v_isShared_1306_ = v_isSharedCheck_1310_;
goto v_resetjp_1304_;
}
else
{
lean_inc(v_a_1303_);
lean_dec(v___x_1284_);
v___x_1305_ = lean_box(0);
v_isShared_1306_ = v_isSharedCheck_1310_;
goto v_resetjp_1304_;
}
v_resetjp_1304_:
{
lean_object* v___x_1308_; 
if (v_isShared_1306_ == 0)
{
v___x_1308_ = v___x_1305_;
goto v_reusejp_1307_;
}
else
{
lean_object* v_reuseFailAlloc_1309_; 
v_reuseFailAlloc_1309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1309_, 0, v_a_1303_);
v___x_1308_ = v_reuseFailAlloc_1309_;
goto v_reusejp_1307_;
}
v_reusejp_1307_:
{
return v___x_1308_;
}
}
}
}
else
{
lean_object* v_a_1311_; lean_object* v___x_1313_; uint8_t v_isShared_1314_; uint8_t v_isSharedCheck_1318_; 
lean_dec(v_a_1281_);
lean_dec(v___x_1270_);
v_a_1311_ = lean_ctor_get(v___x_1282_, 0);
v_isSharedCheck_1318_ = !lean_is_exclusive(v___x_1282_);
if (v_isSharedCheck_1318_ == 0)
{
v___x_1313_ = v___x_1282_;
v_isShared_1314_ = v_isSharedCheck_1318_;
goto v_resetjp_1312_;
}
else
{
lean_inc(v_a_1311_);
lean_dec(v___x_1282_);
v___x_1313_ = lean_box(0);
v_isShared_1314_ = v_isSharedCheck_1318_;
goto v_resetjp_1312_;
}
v_resetjp_1312_:
{
lean_object* v___x_1316_; 
if (v_isShared_1314_ == 0)
{
v___x_1316_ = v___x_1313_;
goto v_reusejp_1315_;
}
else
{
lean_object* v_reuseFailAlloc_1317_; 
v_reuseFailAlloc_1317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1317_, 0, v_a_1311_);
v___x_1316_ = v_reuseFailAlloc_1317_;
goto v_reusejp_1315_;
}
v_reusejp_1315_:
{
return v___x_1316_;
}
}
}
}
else
{
lean_object* v_a_1319_; lean_object* v___x_1321_; uint8_t v_isShared_1322_; uint8_t v_isSharedCheck_1326_; 
lean_dec(v___x_1270_);
v_a_1319_ = lean_ctor_get(v___x_1280_, 0);
v_isSharedCheck_1326_ = !lean_is_exclusive(v___x_1280_);
if (v_isSharedCheck_1326_ == 0)
{
v___x_1321_ = v___x_1280_;
v_isShared_1322_ = v_isSharedCheck_1326_;
goto v_resetjp_1320_;
}
else
{
lean_inc(v_a_1319_);
lean_dec(v___x_1280_);
v___x_1321_ = lean_box(0);
v_isShared_1322_ = v_isSharedCheck_1326_;
goto v_resetjp_1320_;
}
v_resetjp_1320_:
{
lean_object* v___x_1324_; 
if (v_isShared_1322_ == 0)
{
v___x_1324_ = v___x_1321_;
goto v_reusejp_1323_;
}
else
{
lean_object* v_reuseFailAlloc_1325_; 
v_reuseFailAlloc_1325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1325_, 0, v_a_1319_);
v___x_1324_ = v_reuseFailAlloc_1325_;
goto v_reusejp_1323_;
}
v_reusejp_1323_:
{
return v___x_1324_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__1___boxed(lean_object* v___x_1327_, lean_object* v___x_1328_, lean_object* v___x_1329_, lean_object* v___x_1330_, lean_object* v___x_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_){
_start:
{
uint8_t v___x_31234__boxed_1341_; uint8_t v___x_31235__boxed_1342_; lean_object* v_res_1343_; 
v___x_31234__boxed_1341_ = lean_unbox(v___x_1329_);
v___x_31235__boxed_1342_ = lean_unbox(v___x_1330_);
v_res_1343_ = lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__1(v___x_1327_, v___x_1328_, v___x_31234__boxed_1341_, v___x_31235__boxed_1342_, v___x_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_, v___y_1336_, v___y_1337_, v___y_1338_, v___y_1339_);
lean_dec(v___y_1339_);
lean_dec_ref(v___y_1338_);
lean_dec(v___y_1337_);
lean_dec_ref(v___y_1336_);
lean_dec(v___y_1335_);
lean_dec_ref(v___y_1334_);
lean_dec(v___y_1333_);
lean_dec_ref(v___y_1332_);
return v_res_1343_;
}
}
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1(lean_object* v_x_1349_, lean_object* v_a_1350_, lean_object* v_a_1351_, lean_object* v_a_1352_, lean_object* v_a_1353_, lean_object* v_a_1354_, lean_object* v_a_1355_, lean_object* v_a_1356_, lean_object* v_a_1357_){
_start:
{
lean_object* v___x_1359_; uint8_t v___x_1360_; lean_object* v___y_1362_; lean_object* v___y_1363_; lean_object* v___y_1364_; lean_object* v___y_1365_; lean_object* v___y_1366_; lean_object* v___y_1367_; lean_object* v___y_1368_; lean_object* v___y_1369_; lean_object* v___y_1370_; 
v___x_1359_ = ((lean_object*)(lp_plausible_plausibleSyntax___closed__1));
lean_inc(v_x_1349_);
v___x_1360_ = l_Lean_Syntax_isOfKind(v_x_1349_, v___x_1359_);
if (v___x_1360_ == 0)
{
lean_object* v___x_1379_; 
lean_dec(v_x_1349_);
v___x_1379_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg();
return v___x_1379_;
}
else
{
lean_object* v___x_1380_; lean_object* v___x_1381_; uint8_t v___x_1382_; 
v___x_1380_ = lean_unsigned_to_nat(1u);
v___x_1381_ = l_Lean_Syntax_getArg(v_x_1349_, v___x_1380_);
lean_dec(v_x_1349_);
v___x_1382_ = l_Lean_Syntax_isNone(v___x_1381_);
if (v___x_1382_ == 0)
{
uint8_t v___x_1383_; 
lean_inc(v___x_1381_);
v___x_1383_ = l_Lean_Syntax_matchesNull(v___x_1381_, v___x_1380_);
if (v___x_1383_ == 0)
{
lean_object* v___x_1384_; 
lean_dec(v___x_1381_);
v___x_1384_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__0___redArg();
return v___x_1384_;
}
else
{
lean_object* v___x_1385_; lean_object* v_cfg_1386_; lean_object* v___x_1387_; 
v___x_1385_ = lean_unsigned_to_nat(0u);
v_cfg_1386_ = l_Lean_Syntax_getArg(v___x_1381_, v___x_1385_);
lean_dec(v___x_1381_);
v___x_1387_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1387_, 0, v_cfg_1386_);
v___y_1362_ = v_a_1356_;
v___y_1363_ = v_a_1355_;
v___y_1364_ = v_a_1353_;
v___y_1365_ = v_a_1350_;
v___y_1366_ = v_a_1357_;
v___y_1367_ = v_a_1352_;
v___y_1368_ = v_a_1351_;
v___y_1369_ = v_a_1354_;
v___y_1370_ = v___x_1387_;
goto v___jp_1361_;
}
}
else
{
lean_object* v___x_1388_; 
lean_dec(v___x_1381_);
v___x_1388_ = lean_box(0);
v___y_1362_ = v_a_1356_;
v___y_1363_ = v_a_1355_;
v___y_1364_ = v_a_1353_;
v___y_1365_ = v_a_1350_;
v___y_1366_ = v_a_1357_;
v___y_1367_ = v_a_1352_;
v___y_1368_ = v_a_1351_;
v___y_1369_ = v_a_1354_;
v___y_1370_ = v___x_1388_;
goto v___jp_1361_;
}
}
v___jp_1361_:
{
lean_object* v___x_1371_; uint8_t v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___f_1377_; lean_object* v___x_1378_; 
v___x_1371_ = l_Lean_mkOptionalNode(v___y_1370_);
v___x_1372_ = 0;
v___x_1373_ = lean_box(0);
v___x_1374_ = ((lean_object*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___closed__0));
v___x_1375_ = lean_box(v___x_1360_);
v___x_1376_ = lean_box(v___x_1372_);
v___f_1377_ = lean_alloc_closure((void*)(lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___lam__1___boxed), 14, 5);
lean_closure_set(v___f_1377_, 0, v___x_1371_);
lean_closure_set(v___f_1377_, 1, v___x_1374_);
lean_closure_set(v___f_1377_, 2, v___x_1375_);
lean_closure_set(v___f_1377_, 3, v___x_1376_);
lean_closure_set(v___f_1377_, 4, v___x_1373_);
v___x_1378_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1377_, v___y_1365_, v___y_1368_, v___y_1367_, v___y_1364_, v___y_1369_, v___y_1363_, v___y_1362_, v___y_1366_);
return v___x_1378_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1___boxed(lean_object* v_x_1389_, lean_object* v_a_1390_, lean_object* v_a_1391_, lean_object* v_a_1392_, lean_object* v_a_1393_, lean_object* v_a_1394_, lean_object* v_a_1395_, lean_object* v_a_1396_, lean_object* v_a_1397_, lean_object* v_a_1398_){
_start:
{
lean_object* v_res_1399_; 
v_res_1399_ = lp_plausible___aux__Plausible__Tactic______elabRules__plausibleSyntax__1(v_x_1389_, v_a_1390_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_, v_a_1395_, v_a_1396_, v_a_1397_);
lean_dec(v_a_1397_);
lean_dec_ref(v_a_1396_);
lean_dec(v_a_1395_);
lean_dec_ref(v_a_1394_);
lean_dec(v_a_1393_);
lean_dec_ref(v_a_1392_);
lean_dec(v_a_1391_);
lean_dec_ref(v_a_1390_);
return v_res_1399_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3(lean_object* v_cls_1400_, lean_object* v_msg_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_){
_start:
{
lean_object* v___x_1411_; 
v___x_1411_ = lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___redArg(v_cls_1400_, v_msg_1401_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_);
return v___x_1411_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3___boxed(lean_object* v_cls_1412_, lean_object* v_msg_1413_, lean_object* v___y_1414_, lean_object* v___y_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_, lean_object* v___y_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_){
_start:
{
lean_object* v_res_1423_; 
v_res_1423_ = lp_plausible_Lean_addTrace___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__3(v_cls_1412_, v_msg_1413_, v___y_1414_, v___y_1415_, v___y_1416_, v___y_1417_, v___y_1418_, v___y_1419_, v___y_1420_, v___y_1421_);
lean_dec(v___y_1421_);
lean_dec_ref(v___y_1420_);
lean_dec(v___y_1419_);
lean_dec_ref(v___y_1418_);
lean_dec(v___y_1417_);
lean_dec_ref(v___y_1416_);
lean_dec(v___y_1415_);
lean_dec_ref(v___y_1414_);
return v_res_1423_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4(lean_object* v_00_u03b1_1424_, lean_object* v_msg_1425_, lean_object* v___y_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_){
_start:
{
lean_object* v___x_1435_; 
v___x_1435_ = lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4___redArg(v_msg_1425_, v___y_1430_, v___y_1431_, v___y_1432_, v___y_1433_);
return v___x_1435_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4___boxed(lean_object* v_00_u03b1_1436_, lean_object* v_msg_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_, lean_object* v___y_1446_){
_start:
{
lean_object* v_res_1447_; 
v_res_1447_ = lp_plausible_Lean_throwError___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__4(v_00_u03b1_1436_, v_msg_1437_, v___y_1438_, v___y_1439_, v___y_1440_, v___y_1441_, v___y_1442_, v___y_1443_, v___y_1444_, v___y_1445_);
lean_dec(v___y_1445_);
lean_dec_ref(v___y_1444_);
lean_dec(v___y_1443_);
lean_dec_ref(v___y_1442_);
lean_dec(v___y_1441_);
lean_dec_ref(v___y_1440_);
lean_dec(v___y_1439_);
lean_dec_ref(v___y_1438_);
return v_res_1447_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11(lean_object* v_as_1448_, size_t v_sz_1449_, size_t v_i_1450_, lean_object* v_b_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_, lean_object* v___y_1455_, lean_object* v___y_1456_, lean_object* v___y_1457_, lean_object* v___y_1458_, lean_object* v___y_1459_){
_start:
{
lean_object* v___x_1461_; 
v___x_1461_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11___redArg(v_as_1448_, v_sz_1449_, v_i_1450_, v_b_1451_);
return v___x_1461_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11___boxed(lean_object* v_as_1462_, lean_object* v_sz_1463_, lean_object* v_i_1464_, lean_object* v_b_1465_, lean_object* v___y_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_){
_start:
{
size_t v_sz_boxed_1475_; size_t v_i_boxed_1476_; lean_object* v_res_1477_; 
v_sz_boxed_1475_ = lean_unbox_usize(v_sz_1463_);
lean_dec(v_sz_1463_);
v_i_boxed_1476_ = lean_unbox_usize(v_i_1464_);
lean_dec(v_i_1464_);
v_res_1477_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__4_spec__11(v_as_1462_, v_sz_boxed_1475_, v_i_boxed_1476_, v_b_1465_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_);
lean_dec(v___y_1473_);
lean_dec_ref(v___y_1472_);
lean_dec(v___y_1471_);
lean_dec_ref(v___y_1470_);
lean_dec(v___y_1469_);
lean_dec_ref(v___y_1468_);
lean_dec(v___y_1467_);
lean_dec_ref(v___y_1466_);
lean_dec_ref(v_as_1462_);
return v_res_1477_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10(lean_object* v_as_1478_, size_t v_sz_1479_, size_t v_i_1480_, lean_object* v_b_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_, lean_object* v___y_1484_, lean_object* v___y_1485_, lean_object* v___y_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_){
_start:
{
lean_object* v___x_1491_; 
v___x_1491_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10___redArg(v_as_1478_, v_sz_1479_, v_i_1480_, v_b_1481_);
return v___x_1491_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10___boxed(lean_object* v_as_1492_, lean_object* v_sz_1493_, lean_object* v_i_1494_, lean_object* v_b_1495_, lean_object* v___y_1496_, lean_object* v___y_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_){
_start:
{
size_t v_sz_boxed_1505_; size_t v_i_boxed_1506_; lean_object* v_res_1507_; 
v_sz_boxed_1505_ = lean_unbox_usize(v_sz_1493_);
lean_dec(v_sz_1493_);
v_i_boxed_1506_ = lean_unbox_usize(v_i_1494_);
lean_dec(v_i_1494_);
v_res_1507_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__aux__Plausible__Tactic______elabRules__plausibleSyntax__1_spec__1_spec__1_spec__3_spec__9_spec__10(v_as_1492_, v_sz_boxed_1505_, v_i_boxed_1506_, v_b_1495_, v___y_1496_, v___y_1497_, v___y_1498_, v___y_1499_, v___y_1500_, v___y_1501_, v___y_1502_, v___y_1503_);
lean_dec(v___y_1503_);
lean_dec_ref(v___y_1502_);
lean_dec(v___y_1501_);
lean_dec_ref(v___y_1500_);
lean_dec(v___y_1499_);
lean_dec_ref(v___y_1498_);
lean_dec(v___y_1497_);
lean_dec_ref(v___y_1496_);
lean_dec_ref(v_as_1492_);
return v_res_1507_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_plausible_Plausible_Tactic(uint8_t builtin) {
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
lean_object* runtime_initialize_plausible_Plausible_Testable(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_plausible_Plausible_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Testable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_plausible_plausibleSyntax = _init_lp_plausible_plausibleSyntax();
lean_mark_persistent(lp_plausible_plausibleSyntax);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_plausible_Plausible_Testable(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_plausible_Plausible_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible_Testable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_plausible_Plausible_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_plausible_Plausible_Tactic(builtin);
}
#ifdef __cplusplus
}
#endif
