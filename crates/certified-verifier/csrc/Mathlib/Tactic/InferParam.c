// Lean compiler output
// Module: Mathlib.Tactic.InferParam
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.Basic public meta import Lean.Meta.Tactic.Replace
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_consume_type_annotations(lean_object*);
lean_object* l_Lean_MVarId_replaceTargetDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Expr_getOptParamDefault_x3f(lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAutoParamTactic_x3f(lean_object*);
lean_object* l___private_Lean_Elab_Util_0__Lean_Elab_evalSyntaxConstantUnsafe(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_inferOptParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_inferOptParam___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_inferOptParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_inferOptParam___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_inferOptParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "inferOptParam"};
static const lean_object* lp_mathlib_Mathlib_Tactic_inferOptParam___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_inferOptParam___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_inferOptParam___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_inferOptParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(160, 148, 17, 69, 246, 45, 144, 29)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_inferOptParam___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_inferOptParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "infer_param"};
static const lean_object* lp_mathlib_Mathlib_Tactic_inferOptParam___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_inferOptParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_inferOptParam___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_inferOptParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__3_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_inferOptParam___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_inferOptParam = (const lean_object*)&lp_mathlib_Mathlib_Tactic_inferOptParam___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 84, .m_capacity = 84, .m_length = 83, .m_data = "`infer_param` only solves goals of the form `optParam _ _` or `autoParam _ _`, not "};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__1;
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg___closed__0(void){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg(){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg___closed__0);
v___x_22_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg___boxed(lean_object* v___y_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg();
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0(lean_object* v_00_u03b1_25_, lean_object* v___y_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg();
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___boxed(lean_object* v_00_u03b1_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0(v_00_u03b1_36_, v___y_37_, v___y_38_, v___y_39_, v___y_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5_spec__6___redArg(lean_object* v_x_47_, lean_object* v_x_48_, lean_object* v_x_49_, lean_object* v_x_50_){
_start:
{
lean_object* v_ks_51_; lean_object* v_vs_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_76_; 
v_ks_51_ = lean_ctor_get(v_x_47_, 0);
v_vs_52_ = lean_ctor_get(v_x_47_, 1);
v_isSharedCheck_76_ = !lean_is_exclusive(v_x_47_);
if (v_isSharedCheck_76_ == 0)
{
v___x_54_ = v_x_47_;
v_isShared_55_ = v_isSharedCheck_76_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_vs_52_);
lean_inc(v_ks_51_);
lean_dec(v_x_47_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_76_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___x_56_; uint8_t v___x_57_; 
v___x_56_ = lean_array_get_size(v_ks_51_);
v___x_57_ = lean_nat_dec_lt(v_x_48_, v___x_56_);
if (v___x_57_ == 0)
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_61_; 
lean_dec(v_x_48_);
v___x_58_ = lean_array_push(v_ks_51_, v_x_49_);
v___x_59_ = lean_array_push(v_vs_52_, v_x_50_);
if (v_isShared_55_ == 0)
{
lean_ctor_set(v___x_54_, 1, v___x_59_);
lean_ctor_set(v___x_54_, 0, v___x_58_);
v___x_61_ = v___x_54_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_62_; 
v_reuseFailAlloc_62_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_62_, 0, v___x_58_);
lean_ctor_set(v_reuseFailAlloc_62_, 1, v___x_59_);
v___x_61_ = v_reuseFailAlloc_62_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
return v___x_61_;
}
}
else
{
lean_object* v_k_x27_63_; uint8_t v___x_64_; 
v_k_x27_63_ = lean_array_fget_borrowed(v_ks_51_, v_x_48_);
v___x_64_ = l_Lean_instBEqMVarId_beq(v_x_49_, v_k_x27_63_);
if (v___x_64_ == 0)
{
lean_object* v___x_66_; 
if (v_isShared_55_ == 0)
{
v___x_66_ = v___x_54_;
goto v_reusejp_65_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v_ks_51_);
lean_ctor_set(v_reuseFailAlloc_70_, 1, v_vs_52_);
v___x_66_ = v_reuseFailAlloc_70_;
goto v_reusejp_65_;
}
v_reusejp_65_:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = lean_unsigned_to_nat(1u);
v___x_68_ = lean_nat_add(v_x_48_, v___x_67_);
lean_dec(v_x_48_);
v_x_47_ = v___x_66_;
v_x_48_ = v___x_68_;
goto _start;
}
}
else
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_74_; 
v___x_71_ = lean_array_fset(v_ks_51_, v_x_48_, v_x_49_);
v___x_72_ = lean_array_fset(v_vs_52_, v_x_48_, v_x_50_);
lean_dec(v_x_48_);
if (v_isShared_55_ == 0)
{
lean_ctor_set(v___x_54_, 1, v___x_72_);
lean_ctor_set(v___x_54_, 0, v___x_71_);
v___x_74_ = v___x_54_;
goto v_reusejp_73_;
}
else
{
lean_object* v_reuseFailAlloc_75_; 
v_reuseFailAlloc_75_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_75_, 0, v___x_71_);
lean_ctor_set(v_reuseFailAlloc_75_, 1, v___x_72_);
v___x_74_ = v_reuseFailAlloc_75_;
goto v_reusejp_73_;
}
v_reusejp_73_:
{
return v___x_74_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5___redArg(lean_object* v_n_77_, lean_object* v_k_78_, lean_object* v_v_79_){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_80_ = lean_unsigned_to_nat(0u);
v___x_81_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5_spec__6___redArg(v_n_77_, v___x_80_, v_k_78_, v_v_79_);
return v___x_81_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg(lean_object* v_x_83_, size_t v_x_84_, size_t v_x_85_, lean_object* v_x_86_, lean_object* v_x_87_){
_start:
{
if (lean_obj_tag(v_x_83_) == 0)
{
lean_object* v_es_88_; size_t v___x_89_; size_t v___x_90_; lean_object* v_j_91_; lean_object* v___x_92_; uint8_t v___x_93_; 
v_es_88_ = lean_ctor_get(v_x_83_, 0);
v___x_89_ = ((size_t)31ULL);
v___x_90_ = lean_usize_land(v_x_84_, v___x_89_);
v_j_91_ = lean_usize_to_nat(v___x_90_);
v___x_92_ = lean_array_get_size(v_es_88_);
v___x_93_ = lean_nat_dec_lt(v_j_91_, v___x_92_);
if (v___x_93_ == 0)
{
lean_dec(v_j_91_);
lean_dec(v_x_87_);
lean_dec(v_x_86_);
return v_x_83_;
}
else
{
lean_object* v___x_95_; uint8_t v_isShared_96_; uint8_t v_isSharedCheck_132_; 
lean_inc_ref(v_es_88_);
v_isSharedCheck_132_ = !lean_is_exclusive(v_x_83_);
if (v_isSharedCheck_132_ == 0)
{
lean_object* v_unused_133_; 
v_unused_133_ = lean_ctor_get(v_x_83_, 0);
lean_dec(v_unused_133_);
v___x_95_ = v_x_83_;
v_isShared_96_ = v_isSharedCheck_132_;
goto v_resetjp_94_;
}
else
{
lean_dec(v_x_83_);
v___x_95_ = lean_box(0);
v_isShared_96_ = v_isSharedCheck_132_;
goto v_resetjp_94_;
}
v_resetjp_94_:
{
lean_object* v_v_97_; lean_object* v___x_98_; lean_object* v_xs_x27_99_; lean_object* v___y_101_; 
v_v_97_ = lean_array_fget(v_es_88_, v_j_91_);
v___x_98_ = lean_box(0);
v_xs_x27_99_ = lean_array_fset(v_es_88_, v_j_91_, v___x_98_);
switch(lean_obj_tag(v_v_97_))
{
case 0:
{
lean_object* v_key_106_; lean_object* v_val_107_; lean_object* v___x_109_; uint8_t v_isShared_110_; uint8_t v_isSharedCheck_117_; 
v_key_106_ = lean_ctor_get(v_v_97_, 0);
v_val_107_ = lean_ctor_get(v_v_97_, 1);
v_isSharedCheck_117_ = !lean_is_exclusive(v_v_97_);
if (v_isSharedCheck_117_ == 0)
{
v___x_109_ = v_v_97_;
v_isShared_110_ = v_isSharedCheck_117_;
goto v_resetjp_108_;
}
else
{
lean_inc(v_val_107_);
lean_inc(v_key_106_);
lean_dec(v_v_97_);
v___x_109_ = lean_box(0);
v_isShared_110_ = v_isSharedCheck_117_;
goto v_resetjp_108_;
}
v_resetjp_108_:
{
uint8_t v___x_111_; 
v___x_111_ = l_Lean_instBEqMVarId_beq(v_x_86_, v_key_106_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; lean_object* v___x_113_; 
lean_del_object(v___x_109_);
v___x_112_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_106_, v_val_107_, v_x_86_, v_x_87_);
v___x_113_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
v___y_101_ = v___x_113_;
goto v___jp_100_;
}
else
{
lean_object* v___x_115_; 
lean_dec(v_val_107_);
lean_dec(v_key_106_);
if (v_isShared_110_ == 0)
{
lean_ctor_set(v___x_109_, 1, v_x_87_);
lean_ctor_set(v___x_109_, 0, v_x_86_);
v___x_115_ = v___x_109_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_116_; 
v_reuseFailAlloc_116_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_116_, 0, v_x_86_);
lean_ctor_set(v_reuseFailAlloc_116_, 1, v_x_87_);
v___x_115_ = v_reuseFailAlloc_116_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
v___y_101_ = v___x_115_;
goto v___jp_100_;
}
}
}
}
case 1:
{
lean_object* v_node_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_130_; 
v_node_118_ = lean_ctor_get(v_v_97_, 0);
v_isSharedCheck_130_ = !lean_is_exclusive(v_v_97_);
if (v_isSharedCheck_130_ == 0)
{
v___x_120_ = v_v_97_;
v_isShared_121_ = v_isSharedCheck_130_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_node_118_);
lean_dec(v_v_97_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_130_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
size_t v___x_122_; size_t v___x_123_; size_t v___x_124_; size_t v___x_125_; lean_object* v___x_126_; lean_object* v___x_128_; 
v___x_122_ = ((size_t)5ULL);
v___x_123_ = lean_usize_shift_right(v_x_84_, v___x_122_);
v___x_124_ = ((size_t)1ULL);
v___x_125_ = lean_usize_add(v_x_85_, v___x_124_);
v___x_126_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg(v_node_118_, v___x_123_, v___x_125_, v_x_86_, v_x_87_);
if (v_isShared_121_ == 0)
{
lean_ctor_set(v___x_120_, 0, v___x_126_);
v___x_128_ = v___x_120_;
goto v_reusejp_127_;
}
else
{
lean_object* v_reuseFailAlloc_129_; 
v_reuseFailAlloc_129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_129_, 0, v___x_126_);
v___x_128_ = v_reuseFailAlloc_129_;
goto v_reusejp_127_;
}
v_reusejp_127_:
{
v___y_101_ = v___x_128_;
goto v___jp_100_;
}
}
}
default: 
{
lean_object* v___x_131_; 
v___x_131_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_131_, 0, v_x_86_);
lean_ctor_set(v___x_131_, 1, v_x_87_);
v___y_101_ = v___x_131_;
goto v___jp_100_;
}
}
v___jp_100_:
{
lean_object* v___x_102_; lean_object* v___x_104_; 
v___x_102_ = lean_array_fset(v_xs_x27_99_, v_j_91_, v___y_101_);
lean_dec(v_j_91_);
if (v_isShared_96_ == 0)
{
lean_ctor_set(v___x_95_, 0, v___x_102_);
v___x_104_ = v___x_95_;
goto v_reusejp_103_;
}
else
{
lean_object* v_reuseFailAlloc_105_; 
v_reuseFailAlloc_105_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_105_, 0, v___x_102_);
v___x_104_ = v_reuseFailAlloc_105_;
goto v_reusejp_103_;
}
v_reusejp_103_:
{
return v___x_104_;
}
}
}
}
}
else
{
lean_object* v_ks_134_; lean_object* v_vs_135_; lean_object* v___x_137_; uint8_t v_isShared_138_; uint8_t v_isSharedCheck_155_; 
v_ks_134_ = lean_ctor_get(v_x_83_, 0);
v_vs_135_ = lean_ctor_get(v_x_83_, 1);
v_isSharedCheck_155_ = !lean_is_exclusive(v_x_83_);
if (v_isSharedCheck_155_ == 0)
{
v___x_137_ = v_x_83_;
v_isShared_138_ = v_isSharedCheck_155_;
goto v_resetjp_136_;
}
else
{
lean_inc(v_vs_135_);
lean_inc(v_ks_134_);
lean_dec(v_x_83_);
v___x_137_ = lean_box(0);
v_isShared_138_ = v_isSharedCheck_155_;
goto v_resetjp_136_;
}
v_resetjp_136_:
{
lean_object* v___x_140_; 
if (v_isShared_138_ == 0)
{
v___x_140_ = v___x_137_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v_ks_134_);
lean_ctor_set(v_reuseFailAlloc_154_, 1, v_vs_135_);
v___x_140_ = v_reuseFailAlloc_154_;
goto v_reusejp_139_;
}
v_reusejp_139_:
{
lean_object* v_newNode_141_; uint8_t v___y_143_; size_t v___x_149_; uint8_t v___x_150_; 
v_newNode_141_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5___redArg(v___x_140_, v_x_86_, v_x_87_);
v___x_149_ = ((size_t)7ULL);
v___x_150_ = lean_usize_dec_le(v___x_149_, v_x_85_);
if (v___x_150_ == 0)
{
lean_object* v___x_151_; lean_object* v___x_152_; uint8_t v___x_153_; 
v___x_151_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_141_);
v___x_152_ = lean_unsigned_to_nat(4u);
v___x_153_ = lean_nat_dec_lt(v___x_151_, v___x_152_);
lean_dec(v___x_151_);
v___y_143_ = v___x_153_;
goto v___jp_142_;
}
else
{
v___y_143_ = v___x_150_;
goto v___jp_142_;
}
v___jp_142_:
{
if (v___y_143_ == 0)
{
lean_object* v_ks_144_; lean_object* v_vs_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v_ks_144_ = lean_ctor_get(v_newNode_141_, 0);
lean_inc_ref(v_ks_144_);
v_vs_145_ = lean_ctor_get(v_newNode_141_, 1);
lean_inc_ref(v_vs_145_);
lean_dec_ref(v_newNode_141_);
v___x_146_ = lean_unsigned_to_nat(0u);
v___x_147_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg___closed__0);
v___x_148_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6___redArg(v_x_85_, v_ks_144_, v_vs_145_, v___x_146_, v___x_147_);
lean_dec_ref(v_vs_145_);
lean_dec_ref(v_ks_144_);
return v___x_148_;
}
else
{
return v_newNode_141_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6___redArg(size_t v_depth_156_, lean_object* v_keys_157_, lean_object* v_vals_158_, lean_object* v_i_159_, lean_object* v_entries_160_){
_start:
{
lean_object* v___x_161_; uint8_t v___x_162_; 
v___x_161_ = lean_array_get_size(v_keys_157_);
v___x_162_ = lean_nat_dec_lt(v_i_159_, v___x_161_);
if (v___x_162_ == 0)
{
lean_dec(v_i_159_);
return v_entries_160_;
}
else
{
lean_object* v_k_163_; lean_object* v_v_164_; uint64_t v___x_165_; size_t v_h_166_; size_t v___x_167_; lean_object* v___x_168_; size_t v___x_169_; size_t v___x_170_; size_t v___x_171_; size_t v_h_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v_k_163_ = lean_array_fget_borrowed(v_keys_157_, v_i_159_);
v_v_164_ = lean_array_fget_borrowed(v_vals_158_, v_i_159_);
v___x_165_ = l_Lean_instHashableMVarId_hash(v_k_163_);
v_h_166_ = lean_uint64_to_usize(v___x_165_);
v___x_167_ = ((size_t)5ULL);
v___x_168_ = lean_unsigned_to_nat(1u);
v___x_169_ = ((size_t)1ULL);
v___x_170_ = lean_usize_sub(v_depth_156_, v___x_169_);
v___x_171_ = lean_usize_mul(v___x_167_, v___x_170_);
v_h_172_ = lean_usize_shift_right(v_h_166_, v___x_171_);
v___x_173_ = lean_nat_add(v_i_159_, v___x_168_);
lean_dec(v_i_159_);
lean_inc(v_v_164_);
lean_inc(v_k_163_);
v___x_174_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg(v_entries_160_, v_h_172_, v_depth_156_, v_k_163_, v_v_164_);
v_i_159_ = v___x_173_;
v_entries_160_ = v___x_174_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6___redArg___boxed(lean_object* v_depth_176_, lean_object* v_keys_177_, lean_object* v_vals_178_, lean_object* v_i_179_, lean_object* v_entries_180_){
_start:
{
size_t v_depth_boxed_181_; lean_object* v_res_182_; 
v_depth_boxed_181_ = lean_unbox_usize(v_depth_176_);
lean_dec(v_depth_176_);
v_res_182_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6___redArg(v_depth_boxed_181_, v_keys_177_, v_vals_178_, v_i_179_, v_entries_180_);
lean_dec_ref(v_vals_178_);
lean_dec_ref(v_keys_177_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg___boxed(lean_object* v_x_183_, lean_object* v_x_184_, lean_object* v_x_185_, lean_object* v_x_186_, lean_object* v_x_187_){
_start:
{
size_t v_x_5581__boxed_188_; size_t v_x_5582__boxed_189_; lean_object* v_res_190_; 
v_x_5581__boxed_188_ = lean_unbox_usize(v_x_184_);
lean_dec(v_x_184_);
v_x_5582__boxed_189_ = lean_unbox_usize(v_x_185_);
lean_dec(v_x_185_);
v_res_190_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg(v_x_183_, v_x_5581__boxed_188_, v_x_5582__boxed_189_, v_x_186_, v_x_187_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3___redArg(lean_object* v_x_191_, lean_object* v_x_192_, lean_object* v_x_193_){
_start:
{
uint64_t v___x_194_; size_t v___x_195_; size_t v___x_196_; lean_object* v___x_197_; 
v___x_194_ = l_Lean_instHashableMVarId_hash(v_x_192_);
v___x_195_ = lean_uint64_to_usize(v___x_194_);
v___x_196_ = ((size_t)1ULL);
v___x_197_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg(v_x_191_, v___x_195_, v___x_196_, v_x_192_, v_x_193_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2___redArg(lean_object* v_mvarId_198_, lean_object* v_val_199_, lean_object* v___y_200_){
_start:
{
lean_object* v___x_202_; lean_object* v_mctx_203_; lean_object* v_cache_204_; lean_object* v_zetaDeltaFVarIds_205_; lean_object* v_postponed_206_; lean_object* v_diag_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_235_; 
v___x_202_ = lean_st_ref_take(v___y_200_);
v_mctx_203_ = lean_ctor_get(v___x_202_, 0);
v_cache_204_ = lean_ctor_get(v___x_202_, 1);
v_zetaDeltaFVarIds_205_ = lean_ctor_get(v___x_202_, 2);
v_postponed_206_ = lean_ctor_get(v___x_202_, 3);
v_diag_207_ = lean_ctor_get(v___x_202_, 4);
v_isSharedCheck_235_ = !lean_is_exclusive(v___x_202_);
if (v_isSharedCheck_235_ == 0)
{
v___x_209_ = v___x_202_;
v_isShared_210_ = v_isSharedCheck_235_;
goto v_resetjp_208_;
}
else
{
lean_inc(v_diag_207_);
lean_inc(v_postponed_206_);
lean_inc(v_zetaDeltaFVarIds_205_);
lean_inc(v_cache_204_);
lean_inc(v_mctx_203_);
lean_dec(v___x_202_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_235_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
lean_object* v_depth_211_; lean_object* v_levelAssignDepth_212_; lean_object* v_lmvarCounter_213_; lean_object* v_mvarCounter_214_; lean_object* v_lDecls_215_; lean_object* v_decls_216_; lean_object* v_userNames_217_; lean_object* v_lAssignment_218_; lean_object* v_eAssignment_219_; lean_object* v_dAssignment_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_234_; 
v_depth_211_ = lean_ctor_get(v_mctx_203_, 0);
v_levelAssignDepth_212_ = lean_ctor_get(v_mctx_203_, 1);
v_lmvarCounter_213_ = lean_ctor_get(v_mctx_203_, 2);
v_mvarCounter_214_ = lean_ctor_get(v_mctx_203_, 3);
v_lDecls_215_ = lean_ctor_get(v_mctx_203_, 4);
v_decls_216_ = lean_ctor_get(v_mctx_203_, 5);
v_userNames_217_ = lean_ctor_get(v_mctx_203_, 6);
v_lAssignment_218_ = lean_ctor_get(v_mctx_203_, 7);
v_eAssignment_219_ = lean_ctor_get(v_mctx_203_, 8);
v_dAssignment_220_ = lean_ctor_get(v_mctx_203_, 9);
v_isSharedCheck_234_ = !lean_is_exclusive(v_mctx_203_);
if (v_isSharedCheck_234_ == 0)
{
v___x_222_ = v_mctx_203_;
v_isShared_223_ = v_isSharedCheck_234_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_dAssignment_220_);
lean_inc(v_eAssignment_219_);
lean_inc(v_lAssignment_218_);
lean_inc(v_userNames_217_);
lean_inc(v_decls_216_);
lean_inc(v_lDecls_215_);
lean_inc(v_mvarCounter_214_);
lean_inc(v_lmvarCounter_213_);
lean_inc(v_levelAssignDepth_212_);
lean_inc(v_depth_211_);
lean_dec(v_mctx_203_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_234_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
lean_object* v___x_224_; lean_object* v___x_226_; 
v___x_224_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3___redArg(v_eAssignment_219_, v_mvarId_198_, v_val_199_);
if (v_isShared_223_ == 0)
{
lean_ctor_set(v___x_222_, 8, v___x_224_);
v___x_226_ = v___x_222_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v_depth_211_);
lean_ctor_set(v_reuseFailAlloc_233_, 1, v_levelAssignDepth_212_);
lean_ctor_set(v_reuseFailAlloc_233_, 2, v_lmvarCounter_213_);
lean_ctor_set(v_reuseFailAlloc_233_, 3, v_mvarCounter_214_);
lean_ctor_set(v_reuseFailAlloc_233_, 4, v_lDecls_215_);
lean_ctor_set(v_reuseFailAlloc_233_, 5, v_decls_216_);
lean_ctor_set(v_reuseFailAlloc_233_, 6, v_userNames_217_);
lean_ctor_set(v_reuseFailAlloc_233_, 7, v_lAssignment_218_);
lean_ctor_set(v_reuseFailAlloc_233_, 8, v___x_224_);
lean_ctor_set(v_reuseFailAlloc_233_, 9, v_dAssignment_220_);
v___x_226_ = v_reuseFailAlloc_233_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
lean_object* v___x_228_; 
if (v_isShared_210_ == 0)
{
lean_ctor_set(v___x_209_, 0, v___x_226_);
v___x_228_ = v___x_209_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v___x_226_);
lean_ctor_set(v_reuseFailAlloc_232_, 1, v_cache_204_);
lean_ctor_set(v_reuseFailAlloc_232_, 2, v_zetaDeltaFVarIds_205_);
lean_ctor_set(v_reuseFailAlloc_232_, 3, v_postponed_206_);
lean_ctor_set(v_reuseFailAlloc_232_, 4, v_diag_207_);
v___x_228_ = v_reuseFailAlloc_232_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_229_ = lean_st_ref_set(v___y_200_, v___x_228_);
v___x_230_ = lean_box(0);
v___x_231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_231_, 0, v___x_230_);
return v___x_231_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2___redArg___boxed(lean_object* v_mvarId_236_, lean_object* v_val_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2___redArg(v_mvarId_236_, v_val_237_, v___y_238_);
lean_dec(v___y_238_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__0(lean_object* v_val_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_243_, v___y_246_, v___y_247_, v___y_248_, v___y_249_);
if (lean_obj_tag(v___x_251_) == 0)
{
lean_object* v_a_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v_a_252_ = lean_ctor_get(v___x_251_, 0);
lean_inc(v_a_252_);
lean_dec_ref_known(v___x_251_, 1);
v___x_253_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2___redArg(v_a_252_, v_val_241_, v___y_247_);
lean_dec_ref(v___x_253_);
v___x_254_ = lean_box(0);
v___x_255_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_254_, v___y_243_, v___y_246_, v___y_247_, v___y_248_, v___y_249_);
if (lean_obj_tag(v___x_255_) == 0)
{
lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_263_; 
v_isSharedCheck_263_ = !lean_is_exclusive(v___x_255_);
if (v_isSharedCheck_263_ == 0)
{
lean_object* v_unused_264_; 
v_unused_264_ = lean_ctor_get(v___x_255_, 0);
lean_dec(v_unused_264_);
v___x_257_ = v___x_255_;
v_isShared_258_ = v_isSharedCheck_263_;
goto v_resetjp_256_;
}
else
{
lean_dec(v___x_255_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_263_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
lean_object* v___x_259_; lean_object* v___x_261_; 
v___x_259_ = lean_box(0);
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 0, v___x_259_);
v___x_261_ = v___x_257_;
goto v_reusejp_260_;
}
else
{
lean_object* v_reuseFailAlloc_262_; 
v_reuseFailAlloc_262_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_262_, 0, v___x_259_);
v___x_261_ = v_reuseFailAlloc_262_;
goto v_reusejp_260_;
}
v_reusejp_260_:
{
return v___x_261_;
}
}
}
else
{
return v___x_255_;
}
}
else
{
lean_object* v_a_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_272_; 
lean_dec_ref(v_val_241_);
v_a_265_ = lean_ctor_get(v___x_251_, 0);
v_isSharedCheck_272_ = !lean_is_exclusive(v___x_251_);
if (v_isSharedCheck_272_ == 0)
{
v___x_267_ = v___x_251_;
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_a_265_);
lean_dec(v___x_251_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v___x_270_; 
if (v_isShared_268_ == 0)
{
v___x_270_ = v___x_267_;
goto v_reusejp_269_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v_a_265_);
v___x_270_ = v_reuseFailAlloc_271_;
goto v_reusejp_269_;
}
v_reusejp_269_:
{
return v___x_270_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__0___boxed(lean_object* v_val_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__0(v_val_273_, v___y_274_, v___y_275_, v___y_276_, v___y_277_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
lean_dec(v___y_279_);
lean_dec_ref(v___y_278_);
lean_dec(v___y_277_);
lean_dec_ref(v___y_276_);
lean_dec(v___y_275_);
lean_dec_ref(v___y_274_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__1(lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_285_, v___y_288_, v___y_289_, v___y_290_, v___y_291_);
if (lean_obj_tag(v___x_293_) == 0)
{
lean_object* v_a_294_; lean_object* v___x_295_; 
v_a_294_ = lean_ctor_get(v___x_293_, 0);
lean_inc_n(v_a_294_, 2);
lean_dec_ref_known(v___x_293_, 1);
v___x_295_ = l_Lean_MVarId_getType(v_a_294_, v___y_288_, v___y_289_, v___y_290_, v___y_291_);
if (lean_obj_tag(v___x_295_) == 0)
{
lean_object* v_a_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v_a_296_ = lean_ctor_get(v___x_295_, 0);
lean_inc(v_a_296_);
lean_dec_ref_known(v___x_295_, 1);
v___x_297_ = lean_expr_consume_type_annotations(v_a_296_);
v___x_298_ = l_Lean_MVarId_replaceTargetDefEq(v_a_294_, v___x_297_, v___y_288_, v___y_289_, v___y_290_, v___y_291_);
if (lean_obj_tag(v___x_298_) == 0)
{
lean_object* v_a_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v_a_299_ = lean_ctor_get(v___x_298_, 0);
lean_inc(v_a_299_);
lean_dec_ref_known(v___x_298_, 1);
v___x_300_ = lean_box(0);
v___x_301_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_301_, 0, v_a_299_);
lean_ctor_set(v___x_301_, 1, v___x_300_);
v___x_302_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_301_, v___y_285_, v___y_288_, v___y_289_, v___y_290_, v___y_291_);
return v___x_302_;
}
else
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_310_; 
v_a_303_ = lean_ctor_get(v___x_298_, 0);
v_isSharedCheck_310_ = !lean_is_exclusive(v___x_298_);
if (v_isSharedCheck_310_ == 0)
{
v___x_305_ = v___x_298_;
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_298_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v___x_308_; 
if (v_isShared_306_ == 0)
{
v___x_308_ = v___x_305_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v_a_303_);
v___x_308_ = v_reuseFailAlloc_309_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
return v___x_308_;
}
}
}
}
else
{
lean_object* v_a_311_; lean_object* v___x_313_; uint8_t v_isShared_314_; uint8_t v_isSharedCheck_318_; 
lean_dec(v_a_294_);
v_a_311_ = lean_ctor_get(v___x_295_, 0);
v_isSharedCheck_318_ = !lean_is_exclusive(v___x_295_);
if (v_isSharedCheck_318_ == 0)
{
v___x_313_ = v___x_295_;
v_isShared_314_ = v_isSharedCheck_318_;
goto v_resetjp_312_;
}
else
{
lean_inc(v_a_311_);
lean_dec(v___x_295_);
v___x_313_ = lean_box(0);
v_isShared_314_ = v_isSharedCheck_318_;
goto v_resetjp_312_;
}
v_resetjp_312_:
{
lean_object* v___x_316_; 
if (v_isShared_314_ == 0)
{
v___x_316_ = v___x_313_;
goto v_reusejp_315_;
}
else
{
lean_object* v_reuseFailAlloc_317_; 
v_reuseFailAlloc_317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_317_, 0, v_a_311_);
v___x_316_ = v_reuseFailAlloc_317_;
goto v_reusejp_315_;
}
v_reusejp_315_:
{
return v___x_316_;
}
}
}
}
else
{
lean_object* v_a_319_; lean_object* v___x_321_; uint8_t v_isShared_322_; uint8_t v_isSharedCheck_326_; 
v_a_319_ = lean_ctor_get(v___x_293_, 0);
v_isSharedCheck_326_ = !lean_is_exclusive(v___x_293_);
if (v_isSharedCheck_326_ == 0)
{
v___x_321_ = v___x_293_;
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
else
{
lean_inc(v_a_319_);
lean_dec(v___x_293_);
v___x_321_ = lean_box(0);
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
v_resetjp_320_:
{
lean_object* v___x_324_; 
if (v_isShared_322_ == 0)
{
v___x_324_ = v___x_321_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_325_; 
v_reuseFailAlloc_325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_325_, 0, v_a_319_);
v___x_324_ = v_reuseFailAlloc_325_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
return v___x_324_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__1___boxed(lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__1(v___y_327_, v___y_328_, v___y_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1_spec__1(lean_object* v_msgData_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1_spec__1___boxed(lean_object* v_msgData_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_){
_start:
{
lean_object* v_res_358_; 
v_res_358_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1_spec__1(v_msgData_352_, v___y_353_, v___y_354_, v___y_355_, v___y_356_);
lean_dec(v___y_356_);
lean_dec_ref(v___y_355_);
lean_dec(v___y_354_);
lean_dec_ref(v___y_353_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1___redArg(lean_object* v_msg_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_){
_start:
{
lean_object* v_ref_365_; lean_object* v___x_366_; lean_object* v_a_367_; lean_object* v___x_369_; uint8_t v_isShared_370_; uint8_t v_isSharedCheck_375_; 
v_ref_365_ = lean_ctor_get(v___y_362_, 5);
v___x_366_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1_spec__1(v_msg_359_, v___y_360_, v___y_361_, v___y_362_, v___y_363_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1___redArg___boxed(lean_object* v_msg_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1___redArg(v_msg_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_);
lean_dec(v___y_380_);
lean_dec_ref(v___y_379_);
lean_dec(v___y_378_);
lean_dec_ref(v___y_377_);
return v_res_382_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__1(void){
_start:
{
lean_object* v___x_384_; lean_object* v___x_385_; 
v___x_384_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__0));
v___x_385_ = l_Lean_stringToMessageData(v___x_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1(lean_object* v_x_387_, lean_object* v_a_388_, lean_object* v_a_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_){
_start:
{
lean_object* v___x_397_; uint8_t v___x_398_; 
v___x_397_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_inferOptParam___closed__3));
v___x_398_ = l_Lean_Syntax_isOfKind(v_x_387_, v___x_397_);
if (v___x_398_ == 0)
{
lean_object* v___x_399_; 
v___x_399_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__0___redArg();
return v___x_399_;
}
else
{
lean_object* v___x_400_; 
v___x_400_ = l_Lean_Elab_Tactic_getMainTarget(v_a_388_, v_a_389_, v_a_390_, v_a_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
if (lean_obj_tag(v___x_400_) == 0)
{
lean_object* v_a_401_; lean_object* v___y_403_; lean_object* v___y_404_; lean_object* v___y_405_; lean_object* v___y_406_; lean_object* v___y_407_; lean_object* v___y_408_; lean_object* v___y_409_; lean_object* v___y_410_; lean_object* v___x_415_; 
v_a_401_ = lean_ctor_get(v___x_400_, 0);
lean_inc(v_a_401_);
lean_dec_ref_known(v___x_400_, 1);
v___x_415_ = l_Lean_Expr_getOptParamDefault_x3f(v_a_401_);
if (lean_obj_tag(v___x_415_) == 1)
{
lean_object* v_val_416_; lean_object* v___f_417_; lean_object* v___x_418_; 
lean_dec(v_a_401_);
v_val_416_ = lean_ctor_get(v___x_415_, 0);
lean_inc(v_val_416_);
lean_dec_ref_known(v___x_415_, 1);
v___f_417_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___lam__0___boxed), 10, 1);
lean_closure_set(v___f_417_, 0, v_val_416_);
v___x_418_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_417_, v_a_388_, v_a_389_, v_a_390_, v_a_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
return v___x_418_;
}
else
{
lean_object* v___x_419_; 
lean_dec(v___x_415_);
v___x_419_ = l_Lean_Expr_getAutoParamTactic_x3f(v_a_401_);
if (lean_obj_tag(v___x_419_) == 1)
{
lean_object* v_val_420_; 
v_val_420_ = lean_ctor_get(v___x_419_, 0);
lean_inc(v_val_420_);
lean_dec_ref_known(v___x_419_, 1);
if (lean_obj_tag(v_val_420_) == 4)
{
lean_object* v_declName_421_; lean_object* v___x_422_; lean_object* v_env_423_; lean_object* v_options_424_; lean_object* v___x_425_; 
lean_dec(v_a_401_);
v_declName_421_ = lean_ctor_get(v_val_420_, 0);
lean_inc(v_declName_421_);
lean_dec_ref_known(v_val_420_, 2);
v___x_422_ = lean_st_ref_get(v_a_395_);
v_env_423_ = lean_ctor_get(v___x_422_, 0);
lean_inc_ref(v_env_423_);
lean_dec(v___x_422_);
v_options_424_ = lean_ctor_get(v_a_394_, 2);
v___x_425_ = l___private_Lean_Elab_Util_0__Lean_Elab_evalSyntaxConstantUnsafe(v_env_423_, v_options_424_, v_declName_421_);
if (lean_obj_tag(v___x_425_) == 0)
{
lean_object* v_a_426_; lean_object* v___x_428_; uint8_t v_isShared_429_; uint8_t v_isSharedCheck_435_; 
v_a_426_ = lean_ctor_get(v___x_425_, 0);
v_isSharedCheck_435_ = !lean_is_exclusive(v___x_425_);
if (v_isSharedCheck_435_ == 0)
{
v___x_428_ = v___x_425_;
v_isShared_429_ = v_isSharedCheck_435_;
goto v_resetjp_427_;
}
else
{
lean_inc(v_a_426_);
lean_dec(v___x_425_);
v___x_428_ = lean_box(0);
v_isShared_429_ = v_isSharedCheck_435_;
goto v_resetjp_427_;
}
v_resetjp_427_:
{
lean_object* v___x_431_; 
if (v_isShared_429_ == 0)
{
lean_ctor_set_tag(v___x_428_, 3);
v___x_431_ = v___x_428_;
goto v_reusejp_430_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v_a_426_);
v___x_431_ = v_reuseFailAlloc_434_;
goto v_reusejp_430_;
}
v_reusejp_430_:
{
lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_432_ = l_Lean_MessageData_ofFormat(v___x_431_);
v___x_433_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1___redArg(v___x_432_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
return v___x_433_;
}
}
}
else
{
lean_object* v_a_436_; lean_object* v___f_437_; lean_object* v___x_438_; 
v_a_436_ = lean_ctor_get(v___x_425_, 0);
lean_inc(v_a_436_);
lean_dec_ref_known(v___x_425_, 1);
v___f_437_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__2));
v___x_438_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_437_, v_a_388_, v_a_389_, v_a_390_, v_a_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
if (lean_obj_tag(v___x_438_) == 0)
{
lean_object* v___x_439_; 
lean_dec_ref_known(v___x_438_, 1);
v___x_439_ = l_Lean_Elab_Tactic_evalTactic(v_a_436_, v_a_388_, v_a_389_, v_a_390_, v_a_391_, v_a_392_, v_a_393_, v_a_394_, v_a_395_);
return v___x_439_;
}
else
{
lean_dec(v_a_436_);
return v___x_438_;
}
}
}
else
{
lean_dec(v_val_420_);
v___y_403_ = v_a_388_;
v___y_404_ = v_a_389_;
v___y_405_ = v_a_390_;
v___y_406_ = v_a_391_;
v___y_407_ = v_a_392_;
v___y_408_ = v_a_393_;
v___y_409_ = v_a_394_;
v___y_410_ = v_a_395_;
goto v___jp_402_;
}
}
else
{
lean_dec(v___x_419_);
v___y_403_ = v_a_388_;
v___y_404_ = v_a_389_;
v___y_405_ = v_a_390_;
v___y_406_ = v_a_391_;
v___y_407_ = v_a_392_;
v___y_408_ = v_a_393_;
v___y_409_ = v_a_394_;
v___y_410_ = v_a_395_;
goto v___jp_402_;
}
}
v___jp_402_:
{
lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; 
v___x_411_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___closed__1);
v___x_412_ = l_Lean_MessageData_ofExpr(v_a_401_);
v___x_413_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_413_, 0, v___x_411_);
lean_ctor_set(v___x_413_, 1, v___x_412_);
v___x_414_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1___redArg(v___x_413_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
return v___x_414_;
}
}
else
{
lean_object* v_a_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_447_; 
v_a_440_ = lean_ctor_get(v___x_400_, 0);
v_isSharedCheck_447_ = !lean_is_exclusive(v___x_400_);
if (v_isSharedCheck_447_ == 0)
{
v___x_442_ = v___x_400_;
v_isShared_443_ = v_isSharedCheck_447_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_a_440_);
lean_dec(v___x_400_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_447_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___x_445_; 
if (v_isShared_443_ == 0)
{
v___x_445_ = v___x_442_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_446_; 
v_reuseFailAlloc_446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_446_, 0, v_a_440_);
v___x_445_ = v_reuseFailAlloc_446_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
return v___x_445_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1___boxed(lean_object* v_x_448_, lean_object* v_a_449_, lean_object* v_a_450_, lean_object* v_a_451_, lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_, lean_object* v_a_455_, lean_object* v_a_456_, lean_object* v_a_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1(v_x_448_, v_a_449_, v_a_450_, v_a_451_, v_a_452_, v_a_453_, v_a_454_, v_a_455_, v_a_456_);
lean_dec(v_a_456_);
lean_dec_ref(v_a_455_);
lean_dec(v_a_454_);
lean_dec_ref(v_a_453_);
lean_dec(v_a_452_);
lean_dec_ref(v_a_451_);
lean_dec(v_a_450_);
lean_dec_ref(v_a_449_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1(lean_object* v_00_u03b1_459_, lean_object* v_msg_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1___redArg(v_msg_460_, v___y_465_, v___y_466_, v___y_467_, v___y_468_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1___boxed(lean_object* v_00_u03b1_471_, lean_object* v_msg_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_){
_start:
{
lean_object* v_res_482_; 
v_res_482_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__1(v_00_u03b1_471_, v_msg_472_, v___y_473_, v___y_474_, v___y_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_);
lean_dec(v___y_480_);
lean_dec_ref(v___y_479_);
lean_dec(v___y_478_);
lean_dec_ref(v___y_477_);
lean_dec(v___y_476_);
lean_dec_ref(v___y_475_);
lean_dec(v___y_474_);
lean_dec_ref(v___y_473_);
return v_res_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2(lean_object* v_mvarId_483_, lean_object* v_val_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2___redArg(v_mvarId_483_, v_val_484_, v___y_486_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2___boxed(lean_object* v_mvarId_491_, lean_object* v_val_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_){
_start:
{
lean_object* v_res_498_; 
v_res_498_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2(v_mvarId_491_, v_val_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_);
lean_dec(v___y_496_);
lean_dec_ref(v___y_495_);
lean_dec(v___y_494_);
lean_dec_ref(v___y_493_);
return v_res_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3(lean_object* v_00_u03b2_499_, lean_object* v_x_500_, lean_object* v_x_501_, lean_object* v_x_502_){
_start:
{
lean_object* v___x_503_; 
v___x_503_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3___redArg(v_x_500_, v_x_501_, v_x_502_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4(lean_object* v_00_u03b2_504_, lean_object* v_x_505_, size_t v_x_506_, size_t v_x_507_, lean_object* v_x_508_, lean_object* v_x_509_){
_start:
{
lean_object* v___x_510_; 
v___x_510_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___redArg(v_x_505_, v_x_506_, v_x_507_, v_x_508_, v_x_509_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4___boxed(lean_object* v_00_u03b2_511_, lean_object* v_x_512_, lean_object* v_x_513_, lean_object* v_x_514_, lean_object* v_x_515_, lean_object* v_x_516_){
_start:
{
size_t v_x_6209__boxed_517_; size_t v_x_6210__boxed_518_; lean_object* v_res_519_; 
v_x_6209__boxed_517_ = lean_unbox_usize(v_x_513_);
lean_dec(v_x_513_);
v_x_6210__boxed_518_ = lean_unbox_usize(v_x_514_);
lean_dec(v_x_514_);
v_res_519_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4(v_00_u03b2_511_, v_x_512_, v_x_6209__boxed_517_, v_x_6210__boxed_518_, v_x_515_, v_x_516_);
return v_res_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5(lean_object* v_00_u03b2_520_, lean_object* v_n_521_, lean_object* v_k_522_, lean_object* v_v_523_){
_start:
{
lean_object* v___x_524_; 
v___x_524_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5___redArg(v_n_521_, v_k_522_, v_v_523_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6(lean_object* v_00_u03b2_525_, size_t v_depth_526_, lean_object* v_keys_527_, lean_object* v_vals_528_, lean_object* v_heq_529_, lean_object* v_i_530_, lean_object* v_entries_531_){
_start:
{
lean_object* v___x_532_; 
v___x_532_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6___redArg(v_depth_526_, v_keys_527_, v_vals_528_, v_i_530_, v_entries_531_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6___boxed(lean_object* v_00_u03b2_533_, lean_object* v_depth_534_, lean_object* v_keys_535_, lean_object* v_vals_536_, lean_object* v_heq_537_, lean_object* v_i_538_, lean_object* v_entries_539_){
_start:
{
size_t v_depth_boxed_540_; lean_object* v_res_541_; 
v_depth_boxed_540_ = lean_unbox_usize(v_depth_534_);
lean_dec(v_depth_534_);
v_res_541_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__6(v_00_u03b2_533_, v_depth_boxed_540_, v_keys_535_, v_vals_536_, v_heq_537_, v_i_538_, v_entries_539_);
lean_dec_ref(v_vals_536_);
lean_dec_ref(v_keys_535_);
return v_res_541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5_spec__6(lean_object* v_00_u03b2_542_, lean_object* v_x_543_, lean_object* v_x_544_, lean_object* v_x_545_, lean_object* v_x_546_){
_start:
{
lean_object* v___x_547_; 
v___x_547_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Tactic__InferParam______elabRules__Mathlib__Tactic__inferOptParam__1_spec__2_spec__3_spec__4_spec__5_spec__6___redArg(v_x_543_, v_x_544_, v_x_545_, v_x_546_);
return v___x_547_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_InferParam(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Meta_Tactic_Replace(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_InferParam(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Replace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Replace(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_InferParam(uint8_t builtin) {
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
res = initialize_Lean_Meta_Tactic_Replace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_InferParam(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_InferParam(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_InferParam(builtin);
}
#ifdef __cplusplus
}
#endif
