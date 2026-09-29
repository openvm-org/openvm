// Lean compiler output
// Module: Mathlib.Tactic.ApplyAt
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.ElabTerm public meta import Mathlib.Lean.Meta.Basic public import Mathlib.Init
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTermForApply(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_LocalContext_findFromUserName_x3f(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_zip___redArg(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_inferInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
uint8_t l_Lean_BinderInfo_isInstImplicit(uint8_t);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Lean_MVarId_note(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_MVarId_tryClear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "tacticApply_At_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(233, 71, 158, 146, 216, 26, 31, 241)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "apply "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " at "};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_tacticApply__At__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__19_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Identifier "};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = " not found"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_46_ = lean_box(0);
v___x_47_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_48_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_48_, 0, v___x_47_);
lean_ctor_set(v___x_48_, 1, v___x_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg(){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg___closed__0);
v___x_51_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_51_, 0, v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg___boxed(lean_object* v___y_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg();
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0(lean_object* v_00_u03b1_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg();
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___boxed(lean_object* v_00_u03b1_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0(v_00_u03b1_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_, v___y_70_, v___y_71_, v___y_72_, v___y_73_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
lean_dec(v___y_71_);
lean_dec_ref(v___y_70_);
lean_dec(v___y_69_);
lean_dec_ref(v___y_68_);
lean_dec(v___y_67_);
lean_dec_ref(v___y_66_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6_spec__8(lean_object* v_msgData_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
lean_object* v___x_82_; lean_object* v_env_83_; lean_object* v___x_84_; lean_object* v_mctx_85_; lean_object* v_lctx_86_; lean_object* v_options_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_82_ = lean_st_ref_get(v___y_80_);
v_env_83_ = lean_ctor_get(v___x_82_, 0);
lean_inc_ref(v_env_83_);
lean_dec(v___x_82_);
v___x_84_ = lean_st_ref_get(v___y_78_);
v_mctx_85_ = lean_ctor_get(v___x_84_, 0);
lean_inc_ref(v_mctx_85_);
lean_dec(v___x_84_);
v_lctx_86_ = lean_ctor_get(v___y_77_, 2);
v_options_87_ = lean_ctor_get(v___y_79_, 2);
lean_inc_ref(v_options_87_);
lean_inc_ref(v_lctx_86_);
v___x_88_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_88_, 0, v_env_83_);
lean_ctor_set(v___x_88_, 1, v_mctx_85_);
lean_ctor_set(v___x_88_, 2, v_lctx_86_);
lean_ctor_set(v___x_88_, 3, v_options_87_);
v___x_89_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_89_, 0, v___x_88_);
lean_ctor_set(v___x_89_, 1, v_msgData_76_);
v___x_90_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6_spec__8___boxed(lean_object* v_msgData_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6_spec__8(v_msgData_91_, v___y_92_, v___y_93_, v___y_94_, v___y_95_);
lean_dec(v___y_95_);
lean_dec_ref(v___y_94_);
lean_dec(v___y_93_);
lean_dec_ref(v___y_92_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6___redArg(lean_object* v_msg_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_){
_start:
{
lean_object* v_ref_104_; lean_object* v___x_105_; lean_object* v_a_106_; lean_object* v___x_108_; uint8_t v_isShared_109_; uint8_t v_isSharedCheck_114_; 
v_ref_104_ = lean_ctor_get(v___y_101_, 5);
v___x_105_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6_spec__8(v_msg_98_, v___y_99_, v___y_100_, v___y_101_, v___y_102_);
v_a_106_ = lean_ctor_get(v___x_105_, 0);
v_isSharedCheck_114_ = !lean_is_exclusive(v___x_105_);
if (v_isSharedCheck_114_ == 0)
{
v___x_108_ = v___x_105_;
v_isShared_109_ = v_isSharedCheck_114_;
goto v_resetjp_107_;
}
else
{
lean_inc(v_a_106_);
lean_dec(v___x_105_);
v___x_108_ = lean_box(0);
v_isShared_109_ = v_isSharedCheck_114_;
goto v_resetjp_107_;
}
v_resetjp_107_:
{
lean_object* v___x_110_; lean_object* v___x_112_; 
lean_inc(v_ref_104_);
v___x_110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_110_, 0, v_ref_104_);
lean_ctor_set(v___x_110_, 1, v_a_106_);
if (v_isShared_109_ == 0)
{
lean_ctor_set_tag(v___x_108_, 1);
lean_ctor_set(v___x_108_, 0, v___x_110_);
v___x_112_ = v___x_108_;
goto v_reusejp_111_;
}
else
{
lean_object* v_reuseFailAlloc_113_; 
v_reuseFailAlloc_113_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_113_, 0, v___x_110_);
v___x_112_ = v_reuseFailAlloc_113_;
goto v_reusejp_111_;
}
v_reusejp_111_:
{
return v___x_112_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6___redArg___boxed(lean_object* v_msg_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6___redArg(v_msg_115_, v___y_116_, v___y_117_, v___y_118_, v___y_119_);
lean_dec(v___y_119_);
lean_dec_ref(v___y_118_);
lean_dec(v___y_117_);
lean_dec_ref(v___y_116_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5___redArg(lean_object* v_ref_122_, lean_object* v_msg_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
lean_object* v_fileName_133_; lean_object* v_fileMap_134_; lean_object* v_options_135_; lean_object* v_currRecDepth_136_; lean_object* v_maxRecDepth_137_; lean_object* v_ref_138_; lean_object* v_currNamespace_139_; lean_object* v_openDecls_140_; lean_object* v_initHeartbeats_141_; lean_object* v_maxHeartbeats_142_; lean_object* v_quotContext_143_; lean_object* v_currMacroScope_144_; uint8_t v_diag_145_; lean_object* v_cancelTk_x3f_146_; uint8_t v_suppressElabErrors_147_; lean_object* v_inheritedTraceOptions_148_; lean_object* v_ref_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v_fileName_133_ = lean_ctor_get(v___y_130_, 0);
v_fileMap_134_ = lean_ctor_get(v___y_130_, 1);
v_options_135_ = lean_ctor_get(v___y_130_, 2);
v_currRecDepth_136_ = lean_ctor_get(v___y_130_, 3);
v_maxRecDepth_137_ = lean_ctor_get(v___y_130_, 4);
v_ref_138_ = lean_ctor_get(v___y_130_, 5);
v_currNamespace_139_ = lean_ctor_get(v___y_130_, 6);
v_openDecls_140_ = lean_ctor_get(v___y_130_, 7);
v_initHeartbeats_141_ = lean_ctor_get(v___y_130_, 8);
v_maxHeartbeats_142_ = lean_ctor_get(v___y_130_, 9);
v_quotContext_143_ = lean_ctor_get(v___y_130_, 10);
v_currMacroScope_144_ = lean_ctor_get(v___y_130_, 11);
v_diag_145_ = lean_ctor_get_uint8(v___y_130_, sizeof(void*)*14);
v_cancelTk_x3f_146_ = lean_ctor_get(v___y_130_, 12);
v_suppressElabErrors_147_ = lean_ctor_get_uint8(v___y_130_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_148_ = lean_ctor_get(v___y_130_, 13);
v_ref_149_ = l_Lean_replaceRef(v_ref_122_, v_ref_138_);
lean_inc_ref(v_inheritedTraceOptions_148_);
lean_inc(v_cancelTk_x3f_146_);
lean_inc(v_currMacroScope_144_);
lean_inc(v_quotContext_143_);
lean_inc(v_maxHeartbeats_142_);
lean_inc(v_initHeartbeats_141_);
lean_inc(v_openDecls_140_);
lean_inc(v_currNamespace_139_);
lean_inc(v_maxRecDepth_137_);
lean_inc(v_currRecDepth_136_);
lean_inc_ref(v_options_135_);
lean_inc_ref(v_fileMap_134_);
lean_inc_ref(v_fileName_133_);
v___x_150_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_150_, 0, v_fileName_133_);
lean_ctor_set(v___x_150_, 1, v_fileMap_134_);
lean_ctor_set(v___x_150_, 2, v_options_135_);
lean_ctor_set(v___x_150_, 3, v_currRecDepth_136_);
lean_ctor_set(v___x_150_, 4, v_maxRecDepth_137_);
lean_ctor_set(v___x_150_, 5, v_ref_149_);
lean_ctor_set(v___x_150_, 6, v_currNamespace_139_);
lean_ctor_set(v___x_150_, 7, v_openDecls_140_);
lean_ctor_set(v___x_150_, 8, v_initHeartbeats_141_);
lean_ctor_set(v___x_150_, 9, v_maxHeartbeats_142_);
lean_ctor_set(v___x_150_, 10, v_quotContext_143_);
lean_ctor_set(v___x_150_, 11, v_currMacroScope_144_);
lean_ctor_set(v___x_150_, 12, v_cancelTk_x3f_146_);
lean_ctor_set(v___x_150_, 13, v_inheritedTraceOptions_148_);
lean_ctor_set_uint8(v___x_150_, sizeof(void*)*14, v_diag_145_);
lean_ctor_set_uint8(v___x_150_, sizeof(void*)*14 + 1, v_suppressElabErrors_147_);
v___x_151_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6___redArg(v_msg_123_, v___y_128_, v___y_129_, v___x_150_, v___y_131_);
lean_dec_ref_known(v___x_150_, 14);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5___redArg___boxed(lean_object* v_ref_152_, lean_object* v_msg_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5___redArg(v_ref_152_, v_msg_153_, v___y_154_, v___y_155_, v___y_156_, v___y_157_, v___y_158_, v___y_159_, v___y_160_, v___y_161_);
lean_dec(v___y_161_);
lean_dec_ref(v___y_160_);
lean_dec(v___y_159_);
lean_dec_ref(v___y_158_);
lean_dec(v___y_157_);
lean_dec_ref(v___y_156_);
lean_dec(v___y_155_);
lean_dec_ref(v___y_154_);
lean_dec(v_ref_152_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__4(lean_object* v_a_164_, lean_object* v_a_165_){
_start:
{
if (lean_obj_tag(v_a_164_) == 0)
{
lean_object* v___x_166_; 
v___x_166_ = l_List_reverse___redArg(v_a_165_);
return v___x_166_;
}
else
{
lean_object* v_head_167_; lean_object* v_tail_168_; lean_object* v___x_170_; uint8_t v_isShared_171_; uint8_t v_isSharedCheck_177_; 
v_head_167_ = lean_ctor_get(v_a_164_, 0);
v_tail_168_ = lean_ctor_get(v_a_164_, 1);
v_isSharedCheck_177_ = !lean_is_exclusive(v_a_164_);
if (v_isSharedCheck_177_ == 0)
{
v___x_170_ = v_a_164_;
v_isShared_171_ = v_isSharedCheck_177_;
goto v_resetjp_169_;
}
else
{
lean_inc(v_tail_168_);
lean_inc(v_head_167_);
lean_dec(v_a_164_);
v___x_170_ = lean_box(0);
v_isShared_171_ = v_isSharedCheck_177_;
goto v_resetjp_169_;
}
v_resetjp_169_:
{
lean_object* v___x_172_; lean_object* v___x_174_; 
v___x_172_ = l_Lean_Expr_mvarId_x21(v_head_167_);
lean_dec(v_head_167_);
if (v_isShared_171_ == 0)
{
lean_ctor_set(v___x_170_, 1, v_a_165_);
lean_ctor_set(v___x_170_, 0, v___x_172_);
v___x_174_ = v___x_170_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v___x_172_);
lean_ctor_set(v_reuseFailAlloc_176_, 1, v_a_165_);
v___x_174_ = v_reuseFailAlloc_176_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
v_a_164_ = v_tail_168_;
v_a_165_ = v___x_174_;
goto _start;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7___redArg(lean_object* v_keys_178_, lean_object* v_i_179_, lean_object* v_k_180_){
_start:
{
lean_object* v___x_181_; uint8_t v___x_182_; 
v___x_181_ = lean_array_get_size(v_keys_178_);
v___x_182_ = lean_nat_dec_lt(v_i_179_, v___x_181_);
if (v___x_182_ == 0)
{
lean_dec(v_i_179_);
return v___x_182_;
}
else
{
lean_object* v_k_x27_183_; uint8_t v___x_184_; 
v_k_x27_183_ = lean_array_fget_borrowed(v_keys_178_, v_i_179_);
v___x_184_ = l_Lean_instBEqMVarId_beq(v_k_180_, v_k_x27_183_);
if (v___x_184_ == 0)
{
lean_object* v___x_185_; lean_object* v___x_186_; 
v___x_185_ = lean_unsigned_to_nat(1u);
v___x_186_ = lean_nat_add(v_i_179_, v___x_185_);
lean_dec(v_i_179_);
v_i_179_ = v___x_186_;
goto _start;
}
else
{
lean_dec(v_i_179_);
return v___x_184_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7___redArg___boxed(lean_object* v_keys_188_, lean_object* v_i_189_, lean_object* v_k_190_){
_start:
{
uint8_t v_res_191_; lean_object* v_r_192_; 
v_res_191_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7___redArg(v_keys_188_, v_i_189_, v_k_190_);
lean_dec(v_k_190_);
lean_dec_ref(v_keys_188_);
v_r_192_ = lean_box(v_res_191_);
return v_r_192_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2___redArg(lean_object* v_x_193_, size_t v_x_194_, lean_object* v_x_195_){
_start:
{
if (lean_obj_tag(v_x_193_) == 0)
{
lean_object* v_es_196_; lean_object* v___x_197_; size_t v___x_198_; size_t v___x_199_; lean_object* v_j_200_; lean_object* v___x_201_; 
v_es_196_ = lean_ctor_get(v_x_193_, 0);
v___x_197_ = lean_box(2);
v___x_198_ = ((size_t)31ULL);
v___x_199_ = lean_usize_land(v_x_194_, v___x_198_);
v_j_200_ = lean_usize_to_nat(v___x_199_);
v___x_201_ = lean_array_get_borrowed(v___x_197_, v_es_196_, v_j_200_);
lean_dec(v_j_200_);
switch(lean_obj_tag(v___x_201_))
{
case 0:
{
lean_object* v_key_202_; uint8_t v___x_203_; 
v_key_202_ = lean_ctor_get(v___x_201_, 0);
v___x_203_ = l_Lean_instBEqMVarId_beq(v_x_195_, v_key_202_);
return v___x_203_;
}
case 1:
{
lean_object* v_node_204_; size_t v___x_205_; size_t v___x_206_; 
v_node_204_ = lean_ctor_get(v___x_201_, 0);
v___x_205_ = ((size_t)5ULL);
v___x_206_ = lean_usize_shift_right(v_x_194_, v___x_205_);
v_x_193_ = v_node_204_;
v_x_194_ = v___x_206_;
goto _start;
}
default: 
{
uint8_t v___x_208_; 
v___x_208_ = 0;
return v___x_208_;
}
}
}
else
{
lean_object* v_ks_209_; lean_object* v___x_210_; uint8_t v___x_211_; 
v_ks_209_ = lean_ctor_get(v_x_193_, 0);
v___x_210_ = lean_unsigned_to_nat(0u);
v___x_211_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7___redArg(v_ks_209_, v___x_210_, v_x_195_);
return v___x_211_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_x_212_, lean_object* v_x_213_, lean_object* v_x_214_){
_start:
{
size_t v_x_10619__boxed_215_; uint8_t v_res_216_; lean_object* v_r_217_; 
v_x_10619__boxed_215_ = lean_unbox_usize(v_x_213_);
lean_dec(v_x_213_);
v_res_216_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2___redArg(v_x_212_, v_x_10619__boxed_215_, v_x_214_);
lean_dec(v_x_214_);
lean_dec_ref(v_x_212_);
v_r_217_ = lean_box(v_res_216_);
return v_r_217_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1___redArg(lean_object* v_x_218_, lean_object* v_x_219_){
_start:
{
uint64_t v___x_220_; size_t v___x_221_; uint8_t v___x_222_; 
v___x_220_ = l_Lean_instHashableMVarId_hash(v_x_219_);
v___x_221_ = lean_uint64_to_usize(v___x_220_);
v___x_222_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2___redArg(v_x_218_, v___x_221_, v_x_219_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1___redArg___boxed(lean_object* v_x_223_, lean_object* v_x_224_){
_start:
{
uint8_t v_res_225_; lean_object* v_r_226_; 
v_res_225_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1___redArg(v_x_223_, v_x_224_);
lean_dec(v_x_224_);
lean_dec_ref(v_x_223_);
v_r_226_ = lean_box(v_res_225_);
return v_r_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1___redArg(lean_object* v_mvarId_227_, lean_object* v___y_228_){
_start:
{
lean_object* v___x_230_; lean_object* v_mctx_231_; lean_object* v_eAssignment_232_; uint8_t v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_230_ = lean_st_ref_get(v___y_228_);
v_mctx_231_ = lean_ctor_get(v___x_230_, 0);
lean_inc_ref(v_mctx_231_);
lean_dec(v___x_230_);
v_eAssignment_232_ = lean_ctor_get(v_mctx_231_, 8);
lean_inc_ref(v_eAssignment_232_);
lean_dec_ref(v_mctx_231_);
v___x_233_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1___redArg(v_eAssignment_232_, v_mvarId_227_);
lean_dec_ref(v_eAssignment_232_);
v___x_234_ = lean_box(v___x_233_);
v___x_235_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_235_, 0, v___x_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1___redArg___boxed(lean_object* v_mvarId_236_, lean_object* v___y_237_, lean_object* v___y_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1___redArg(v_mvarId_236_, v___y_237_);
lean_dec(v___y_237_);
lean_dec(v_mvarId_236_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__2(lean_object* v_as_240_, size_t v_sz_241_, size_t v_i_242_, lean_object* v_b_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_){
_start:
{
lean_object* v_a_254_; uint8_t v___x_258_; 
v___x_258_ = lean_usize_dec_lt(v_i_242_, v_sz_241_);
if (v___x_258_ == 0)
{
lean_object* v___x_259_; 
v___x_259_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_259_, 0, v_b_243_);
return v___x_259_;
}
else
{
lean_object* v_a_260_; lean_object* v_fst_261_; lean_object* v_snd_262_; lean_object* v___x_263_; lean_object* v___x_264_; 
v_a_260_ = lean_array_uget_borrowed(v_as_240_, v_i_242_);
v_fst_261_ = lean_ctor_get(v_a_260_, 0);
v_snd_262_ = lean_ctor_get(v_a_260_, 1);
v___x_263_ = l_Lean_Expr_mvarId_x21(v_fst_261_);
v___x_264_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1___redArg(v___x_263_, v___y_249_);
if (lean_obj_tag(v___x_264_) == 0)
{
lean_object* v_a_265_; lean_object* v___x_266_; lean_object* v___y_268_; lean_object* v___y_269_; uint8_t v___y_270_; uint8_t v___y_273_; uint8_t v___x_288_; uint8_t v___x_289_; 
v_a_265_ = lean_ctor_get(v___x_264_, 0);
lean_inc(v_a_265_);
lean_dec_ref_known(v___x_264_, 1);
v___x_266_ = lean_box(0);
v___x_288_ = lean_unbox(v_snd_262_);
v___x_289_ = l_Lean_BinderInfo_isInstImplicit(v___x_288_);
if (v___x_289_ == 0)
{
lean_dec(v_a_265_);
v___y_273_ = v___x_289_;
goto v___jp_272_;
}
else
{
uint8_t v___x_290_; 
v___x_290_ = lean_unbox(v_a_265_);
lean_dec(v_a_265_);
if (v___x_290_ == 0)
{
v___y_273_ = v___x_289_;
goto v___jp_272_;
}
else
{
lean_dec(v___x_263_);
v_a_254_ = v___x_266_;
goto v___jp_253_;
}
}
v___jp_267_:
{
if (v___y_270_ == 0)
{
lean_object* v___x_271_; 
lean_dec_ref(v___y_269_);
v___x_271_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v___y_268_, v___y_270_, v___y_245_, v___y_246_, v___y_247_, v___y_248_, v___y_249_, v___y_250_, v___y_251_);
if (lean_obj_tag(v___x_271_) == 0)
{
lean_dec_ref_known(v___x_271_, 1);
v_a_254_ = v___x_266_;
goto v___jp_253_;
}
else
{
return v___x_271_;
}
}
else
{
lean_dec_ref(v___y_268_);
return v___y_269_;
}
}
v___jp_272_:
{
if (v___y_273_ == 0)
{
lean_dec(v___x_263_);
v_a_254_ = v___x_266_;
goto v___jp_253_;
}
else
{
lean_object* v___x_274_; 
v___x_274_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_245_, v___y_247_, v___y_249_, v___y_251_);
if (lean_obj_tag(v___x_274_) == 0)
{
lean_object* v_a_275_; lean_object* v___x_276_; 
v_a_275_ = lean_ctor_get(v___x_274_, 0);
lean_inc(v_a_275_);
lean_dec_ref_known(v___x_274_, 1);
v___x_276_ = l_Lean_MVarId_inferInstance(v___x_263_, v___y_248_, v___y_249_, v___y_250_, v___y_251_);
if (lean_obj_tag(v___x_276_) == 0)
{
lean_dec_ref_known(v___x_276_, 1);
lean_dec(v_a_275_);
v_a_254_ = v___x_266_;
goto v___jp_253_;
}
else
{
lean_object* v_a_277_; uint8_t v___x_278_; 
v_a_277_ = lean_ctor_get(v___x_276_, 0);
lean_inc(v_a_277_);
v___x_278_ = l_Lean_Exception_isInterrupt(v_a_277_);
if (v___x_278_ == 0)
{
uint8_t v___x_279_; 
v___x_279_ = l_Lean_Exception_isRuntime(v_a_277_);
v___y_268_ = v_a_275_;
v___y_269_ = v___x_276_;
v___y_270_ = v___x_279_;
goto v___jp_267_;
}
else
{
lean_dec(v_a_277_);
v___y_268_ = v_a_275_;
v___y_269_ = v___x_276_;
v___y_270_ = v___x_278_;
goto v___jp_267_;
}
}
}
else
{
lean_object* v_a_280_; lean_object* v___x_282_; uint8_t v_isShared_283_; uint8_t v_isSharedCheck_287_; 
lean_dec(v___x_263_);
v_a_280_ = lean_ctor_get(v___x_274_, 0);
v_isSharedCheck_287_ = !lean_is_exclusive(v___x_274_);
if (v_isSharedCheck_287_ == 0)
{
v___x_282_ = v___x_274_;
v_isShared_283_ = v_isSharedCheck_287_;
goto v_resetjp_281_;
}
else
{
lean_inc(v_a_280_);
lean_dec(v___x_274_);
v___x_282_ = lean_box(0);
v_isShared_283_ = v_isSharedCheck_287_;
goto v_resetjp_281_;
}
v_resetjp_281_:
{
lean_object* v___x_285_; 
if (v_isShared_283_ == 0)
{
v___x_285_ = v___x_282_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_286_; 
v_reuseFailAlloc_286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_286_, 0, v_a_280_);
v___x_285_ = v_reuseFailAlloc_286_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
return v___x_285_;
}
}
}
}
}
}
else
{
lean_object* v_a_291_; lean_object* v___x_293_; uint8_t v_isShared_294_; uint8_t v_isSharedCheck_298_; 
lean_dec(v___x_263_);
v_a_291_ = lean_ctor_get(v___x_264_, 0);
v_isSharedCheck_298_ = !lean_is_exclusive(v___x_264_);
if (v_isSharedCheck_298_ == 0)
{
v___x_293_ = v___x_264_;
v_isShared_294_ = v_isSharedCheck_298_;
goto v_resetjp_292_;
}
else
{
lean_inc(v_a_291_);
lean_dec(v___x_264_);
v___x_293_ = lean_box(0);
v_isShared_294_ = v_isSharedCheck_298_;
goto v_resetjp_292_;
}
v_resetjp_292_:
{
lean_object* v___x_296_; 
if (v_isShared_294_ == 0)
{
v___x_296_ = v___x_293_;
goto v_reusejp_295_;
}
else
{
lean_object* v_reuseFailAlloc_297_; 
v_reuseFailAlloc_297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_297_, 0, v_a_291_);
v___x_296_ = v_reuseFailAlloc_297_;
goto v_reusejp_295_;
}
v_reusejp_295_:
{
return v___x_296_;
}
}
}
}
v___jp_253_:
{
size_t v___x_255_; size_t v___x_256_; 
v___x_255_ = ((size_t)1ULL);
v___x_256_ = lean_usize_add(v_i_242_, v___x_255_);
v_i_242_ = v___x_256_;
v_b_243_ = v_a_254_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__2___boxed(lean_object* v_as_299_, lean_object* v_sz_300_, lean_object* v_i_301_, lean_object* v_b_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_){
_start:
{
size_t v_sz_boxed_312_; size_t v_i_boxed_313_; lean_object* v_res_314_; 
v_sz_boxed_312_ = lean_unbox_usize(v_sz_300_);
lean_dec(v_sz_300_);
v_i_boxed_313_ = lean_unbox_usize(v_i_301_);
lean_dec(v_i_301_);
v_res_314_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__2(v_as_299_, v_sz_boxed_312_, v_i_boxed_313_, v_b_302_, v___y_303_, v___y_304_, v___y_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_);
lean_dec(v___y_310_);
lean_dec_ref(v___y_309_);
lean_dec(v___y_308_);
lean_dec_ref(v___y_307_);
lean_dec(v___y_306_);
lean_dec_ref(v___y_305_);
lean_dec(v___y_304_);
lean_dec_ref(v___y_303_);
lean_dec_ref(v_as_299_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__3(size_t v_sz_315_, size_t v_i_316_, lean_object* v_bs_317_){
_start:
{
uint8_t v___x_318_; 
v___x_318_ = lean_usize_dec_lt(v_i_316_, v_sz_315_);
if (v___x_318_ == 0)
{
return v_bs_317_;
}
else
{
lean_object* v_v_319_; lean_object* v___x_320_; lean_object* v_bs_x27_321_; lean_object* v___x_322_; size_t v___x_323_; size_t v___x_324_; lean_object* v___x_325_; 
v_v_319_ = lean_array_uget(v_bs_317_, v_i_316_);
v___x_320_ = lean_unsigned_to_nat(0u);
v_bs_x27_321_ = lean_array_uset(v_bs_317_, v_i_316_, v___x_320_);
v___x_322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_322_, 0, v_v_319_);
v___x_323_ = ((size_t)1ULL);
v___x_324_ = lean_usize_add(v_i_316_, v___x_323_);
v___x_325_ = lean_array_uset(v_bs_x27_321_, v_i_316_, v___x_322_);
v_i_316_ = v___x_324_;
v_bs_317_ = v___x_325_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__3___boxed(lean_object* v_sz_327_, lean_object* v_i_328_, lean_object* v_bs_329_){
_start:
{
size_t v_sz_boxed_330_; size_t v_i_boxed_331_; lean_object* v_res_332_; 
v_sz_boxed_330_ = lean_unbox_usize(v_sz_327_);
lean_dec(v_sz_327_);
v_i_boxed_331_ = lean_unbox_usize(v_i_328_);
lean_dec(v_i_328_);
v_res_332_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__3(v_sz_boxed_330_, v_i_boxed_331_, v_bs_329_);
return v_res_332_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_334_; lean_object* v___x_335_; 
v___x_334_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__0));
v___x_335_ = l_Lean_stringToMessageData(v___x_334_);
return v___x_335_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__3(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_337_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__2));
v___x_338_ = l_Lean_stringToMessageData(v___x_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0(lean_object* v___x_339_, uint8_t v___x_340_, lean_object* v_i_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_){
_start:
{
lean_object* v___x_351_; 
v___x_351_ = l_Lean_Elab_Tactic_elabTermForApply(v___x_339_, v___x_340_, v___y_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_351_) == 0)
{
lean_object* v_a_352_; lean_object* v_lctx_353_; lean_object* v___x_354_; lean_object* v___x_355_; 
v_a_352_ = lean_ctor_get(v___x_351_, 0);
lean_inc(v_a_352_);
lean_dec_ref_known(v___x_351_, 1);
v_lctx_353_ = lean_ctor_get(v___y_346_, 2);
v___x_354_ = l_Lean_TSyntax_getId(v_i_341_);
v___x_355_ = l_Lean_LocalContext_findFromUserName_x3f(v_lctx_353_, v___x_354_);
lean_dec(v___x_354_);
if (lean_obj_tag(v___x_355_) == 1)
{
lean_object* v_val_356_; lean_object* v___x_357_; 
lean_dec(v_i_341_);
v_val_356_ = lean_ctor_get(v___x_355_, 0);
lean_inc(v_val_356_);
lean_dec_ref_known(v___x_355_, 1);
lean_inc(v___y_349_);
lean_inc_ref(v___y_348_);
lean_inc(v___y_347_);
lean_inc_ref(v___y_346_);
lean_inc(v_a_352_);
v___x_357_ = lean_infer_type(v_a_352_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_357_) == 0)
{
lean_object* v_a_358_; lean_object* v___x_359_; uint8_t v___x_360_; lean_object* v___x_361_; 
v_a_358_ = lean_ctor_get(v___x_357_, 0);
lean_inc(v_a_358_);
lean_dec_ref_known(v___x_357_, 1);
v___x_359_ = l_Lean_LocalDecl_type(v_val_356_);
v___x_360_ = 0;
v___x_361_ = lp_mathlib_Lean_Meta_forallMetaTelescopeReducingUntilDefEq(v_a_358_, v___x_359_, v___x_360_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_361_) == 0)
{
lean_object* v_a_362_; lean_object* v_snd_363_; lean_object* v_fst_364_; lean_object* v_fst_365_; lean_object* v___x_366_; lean_object* v___x_367_; size_t v_sz_368_; size_t v___x_369_; lean_object* v___x_370_; 
v_a_362_ = lean_ctor_get(v___x_361_, 0);
lean_inc(v_a_362_);
lean_dec_ref_known(v___x_361_, 1);
v_snd_363_ = lean_ctor_get(v_a_362_, 1);
lean_inc(v_snd_363_);
v_fst_364_ = lean_ctor_get(v_a_362_, 0);
lean_inc(v_fst_364_);
lean_dec(v_a_362_);
v_fst_365_ = lean_ctor_get(v_snd_363_, 0);
lean_inc(v_fst_365_);
lean_dec(v_snd_363_);
v___x_366_ = l_Array_zip___redArg(v_fst_364_, v_fst_365_);
lean_dec(v_fst_365_);
v___x_367_ = lean_box(0);
v_sz_368_ = lean_array_size(v___x_366_);
v___x_369_ = ((size_t)0ULL);
v___x_370_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__2(v___x_366_, v_sz_368_, v___x_369_, v___x_367_, v___y_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
lean_dec_ref(v___x_366_);
if (lean_obj_tag(v___x_370_) == 0)
{
lean_object* v___x_371_; 
lean_dec_ref_known(v___x_370_, 1);
v___x_371_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_343_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_371_) == 0)
{
lean_object* v_a_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; size_t v_sz_376_; lean_object* v___x_377_; lean_object* v___x_378_; 
v_a_372_ = lean_ctor_get(v___x_371_, 0);
lean_inc(v_a_372_);
lean_dec_ref_known(v___x_371_, 1);
v___x_373_ = lean_array_pop(v_fst_364_);
lean_inc(v_val_356_);
v___x_374_ = l_Lean_LocalDecl_toExpr(v_val_356_);
lean_inc_ref(v___x_373_);
v___x_375_ = lean_array_push(v___x_373_, v___x_374_);
v_sz_376_ = lean_array_size(v___x_375_);
v___x_377_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__3(v_sz_376_, v___x_369_, v___x_375_);
v___x_378_ = l_Lean_Meta_mkAppOptM_x27(v_a_352_, v___x_377_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_378_) == 0)
{
lean_object* v_a_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
v_a_379_ = lean_ctor_get(v___x_378_, 0);
lean_inc(v_a_379_);
lean_dec_ref_known(v___x_378_, 1);
v___x_380_ = l_Lean_LocalDecl_userName(v_val_356_);
v___x_381_ = lean_box(0);
v___x_382_ = l_Lean_MVarId_note(v_a_372_, v___x_380_, v_a_379_, v___x_381_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_382_) == 0)
{
lean_object* v_a_383_; lean_object* v_snd_384_; lean_object* v___x_386_; uint8_t v_isShared_387_; uint8_t v_isSharedCheck_407_; 
v_a_383_ = lean_ctor_get(v___x_382_, 0);
lean_inc(v_a_383_);
lean_dec_ref_known(v___x_382_, 1);
v_snd_384_ = lean_ctor_get(v_a_383_, 1);
v_isSharedCheck_407_ = !lean_is_exclusive(v_a_383_);
if (v_isSharedCheck_407_ == 0)
{
lean_object* v_unused_408_; 
v_unused_408_ = lean_ctor_get(v_a_383_, 0);
lean_dec(v_unused_408_);
v___x_386_ = v_a_383_;
v_isShared_387_ = v_isSharedCheck_407_;
goto v_resetjp_385_;
}
else
{
lean_inc(v_snd_384_);
lean_dec(v_a_383_);
v___x_386_ = lean_box(0);
v_isShared_387_ = v_isSharedCheck_407_;
goto v_resetjp_385_;
}
v_resetjp_385_:
{
lean_object* v___x_388_; lean_object* v___x_389_; 
v___x_388_ = l_Lean_LocalDecl_fvarId(v_val_356_);
lean_dec(v_val_356_);
v___x_389_ = l_Lean_MVarId_tryClear(v_snd_384_, v___x_388_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_389_) == 0)
{
lean_object* v_a_390_; lean_object* v___x_391_; lean_object* v___x_393_; 
v_a_390_ = lean_ctor_get(v___x_389_, 0);
lean_inc(v_a_390_);
lean_dec_ref_known(v___x_389_, 1);
v___x_391_ = lean_box(0);
if (v_isShared_387_ == 0)
{
lean_ctor_set_tag(v___x_386_, 1);
lean_ctor_set(v___x_386_, 1, v___x_391_);
lean_ctor_set(v___x_386_, 0, v_a_390_);
v___x_393_ = v___x_386_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v_a_390_);
lean_ctor_set(v_reuseFailAlloc_398_, 1, v___x_391_);
v___x_393_ = v_reuseFailAlloc_398_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_394_ = lean_array_to_list(v___x_373_);
v___x_395_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__4(v___x_394_, v___x_391_);
v___x_396_ = l_List_appendTR___redArg(v___x_393_, v___x_395_);
v___x_397_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_396_, v___y_343_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
return v___x_397_;
}
}
else
{
lean_object* v_a_399_; lean_object* v___x_401_; uint8_t v_isShared_402_; uint8_t v_isSharedCheck_406_; 
lean_del_object(v___x_386_);
lean_dec_ref(v___x_373_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
v_a_399_ = lean_ctor_get(v___x_389_, 0);
v_isSharedCheck_406_ = !lean_is_exclusive(v___x_389_);
if (v_isSharedCheck_406_ == 0)
{
v___x_401_ = v___x_389_;
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
else
{
lean_inc(v_a_399_);
lean_dec(v___x_389_);
v___x_401_ = lean_box(0);
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
v_resetjp_400_:
{
lean_object* v___x_404_; 
if (v_isShared_402_ == 0)
{
v___x_404_ = v___x_401_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v_a_399_);
v___x_404_ = v_reuseFailAlloc_405_;
goto v_reusejp_403_;
}
v_reusejp_403_:
{
return v___x_404_;
}
}
}
}
}
else
{
lean_object* v_a_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_416_; 
lean_dec_ref(v___x_373_);
lean_dec(v_val_356_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
v_a_409_ = lean_ctor_get(v___x_382_, 0);
v_isSharedCheck_416_ = !lean_is_exclusive(v___x_382_);
if (v_isSharedCheck_416_ == 0)
{
v___x_411_ = v___x_382_;
v_isShared_412_ = v_isSharedCheck_416_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_a_409_);
lean_dec(v___x_382_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_416_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
lean_object* v___x_414_; 
if (v_isShared_412_ == 0)
{
v___x_414_ = v___x_411_;
goto v_reusejp_413_;
}
else
{
lean_object* v_reuseFailAlloc_415_; 
v_reuseFailAlloc_415_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_415_, 0, v_a_409_);
v___x_414_ = v_reuseFailAlloc_415_;
goto v_reusejp_413_;
}
v_reusejp_413_:
{
return v___x_414_;
}
}
}
}
else
{
lean_object* v_a_417_; lean_object* v___x_419_; uint8_t v_isShared_420_; uint8_t v_isSharedCheck_424_; 
lean_dec_ref(v___x_373_);
lean_dec(v_a_372_);
lean_dec(v_val_356_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
v_a_417_ = lean_ctor_get(v___x_378_, 0);
v_isSharedCheck_424_ = !lean_is_exclusive(v___x_378_);
if (v_isSharedCheck_424_ == 0)
{
v___x_419_ = v___x_378_;
v_isShared_420_ = v_isSharedCheck_424_;
goto v_resetjp_418_;
}
else
{
lean_inc(v_a_417_);
lean_dec(v___x_378_);
v___x_419_ = lean_box(0);
v_isShared_420_ = v_isSharedCheck_424_;
goto v_resetjp_418_;
}
v_resetjp_418_:
{
lean_object* v___x_422_; 
if (v_isShared_420_ == 0)
{
v___x_422_ = v___x_419_;
goto v_reusejp_421_;
}
else
{
lean_object* v_reuseFailAlloc_423_; 
v_reuseFailAlloc_423_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_423_, 0, v_a_417_);
v___x_422_ = v_reuseFailAlloc_423_;
goto v_reusejp_421_;
}
v_reusejp_421_:
{
return v___x_422_;
}
}
}
}
else
{
lean_object* v_a_425_; lean_object* v___x_427_; uint8_t v_isShared_428_; uint8_t v_isSharedCheck_432_; 
lean_dec(v_fst_364_);
lean_dec(v_val_356_);
lean_dec(v_a_352_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
v_a_425_ = lean_ctor_get(v___x_371_, 0);
v_isSharedCheck_432_ = !lean_is_exclusive(v___x_371_);
if (v_isSharedCheck_432_ == 0)
{
v___x_427_ = v___x_371_;
v_isShared_428_ = v_isSharedCheck_432_;
goto v_resetjp_426_;
}
else
{
lean_inc(v_a_425_);
lean_dec(v___x_371_);
v___x_427_ = lean_box(0);
v_isShared_428_ = v_isSharedCheck_432_;
goto v_resetjp_426_;
}
v_resetjp_426_:
{
lean_object* v___x_430_; 
if (v_isShared_428_ == 0)
{
v___x_430_ = v___x_427_;
goto v_reusejp_429_;
}
else
{
lean_object* v_reuseFailAlloc_431_; 
v_reuseFailAlloc_431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_431_, 0, v_a_425_);
v___x_430_ = v_reuseFailAlloc_431_;
goto v_reusejp_429_;
}
v_reusejp_429_:
{
return v___x_430_;
}
}
}
}
else
{
lean_dec(v_fst_364_);
lean_dec(v_val_356_);
lean_dec(v_a_352_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
return v___x_370_;
}
}
else
{
lean_object* v_a_433_; lean_object* v___x_435_; uint8_t v_isShared_436_; uint8_t v_isSharedCheck_440_; 
lean_dec(v_val_356_);
lean_dec(v_a_352_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
v_a_433_ = lean_ctor_get(v___x_361_, 0);
v_isSharedCheck_440_ = !lean_is_exclusive(v___x_361_);
if (v_isSharedCheck_440_ == 0)
{
v___x_435_ = v___x_361_;
v_isShared_436_ = v_isSharedCheck_440_;
goto v_resetjp_434_;
}
else
{
lean_inc(v_a_433_);
lean_dec(v___x_361_);
v___x_435_ = lean_box(0);
v_isShared_436_ = v_isSharedCheck_440_;
goto v_resetjp_434_;
}
v_resetjp_434_:
{
lean_object* v___x_438_; 
if (v_isShared_436_ == 0)
{
v___x_438_ = v___x_435_;
goto v_reusejp_437_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v_a_433_);
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
else
{
lean_object* v_a_441_; lean_object* v___x_443_; uint8_t v_isShared_444_; uint8_t v_isSharedCheck_448_; 
lean_dec(v_val_356_);
lean_dec(v_a_352_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
v_a_441_ = lean_ctor_get(v___x_357_, 0);
v_isSharedCheck_448_ = !lean_is_exclusive(v___x_357_);
if (v_isSharedCheck_448_ == 0)
{
v___x_443_ = v___x_357_;
v_isShared_444_ = v_isSharedCheck_448_;
goto v_resetjp_442_;
}
else
{
lean_inc(v_a_441_);
lean_dec(v___x_357_);
v___x_443_ = lean_box(0);
v_isShared_444_ = v_isSharedCheck_448_;
goto v_resetjp_442_;
}
v_resetjp_442_:
{
lean_object* v___x_446_; 
if (v_isShared_444_ == 0)
{
v___x_446_ = v___x_443_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_447_; 
v_reuseFailAlloc_447_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_447_, 0, v_a_441_);
v___x_446_ = v_reuseFailAlloc_447_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
return v___x_446_;
}
}
}
}
else
{
lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; 
lean_dec(v___x_355_);
lean_dec(v_a_352_);
v___x_449_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__1);
lean_inc(v_i_341_);
v___x_450_ = l_Lean_MessageData_ofSyntax(v_i_341_);
v___x_451_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_451_, 0, v___x_449_);
lean_ctor_set(v___x_451_, 1, v___x_450_);
v___x_452_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___closed__3);
v___x_453_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_453_, 0, v___x_451_);
lean_ctor_set(v___x_453_, 1, v___x_452_);
v___x_454_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5___redArg(v_i_341_, v___x_453_, v___y_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
lean_dec(v_i_341_);
return v___x_454_;
}
}
else
{
lean_object* v_a_455_; lean_object* v___x_457_; uint8_t v_isShared_458_; uint8_t v_isSharedCheck_462_; 
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
lean_dec(v_i_341_);
v_a_455_ = lean_ctor_get(v___x_351_, 0);
v_isSharedCheck_462_ = !lean_is_exclusive(v___x_351_);
if (v_isSharedCheck_462_ == 0)
{
v___x_457_ = v___x_351_;
v_isShared_458_ = v_isSharedCheck_462_;
goto v_resetjp_456_;
}
else
{
lean_inc(v_a_455_);
lean_dec(v___x_351_);
v___x_457_ = lean_box(0);
v_isShared_458_ = v_isSharedCheck_462_;
goto v_resetjp_456_;
}
v_resetjp_456_:
{
lean_object* v___x_460_; 
if (v_isShared_458_ == 0)
{
v___x_460_ = v___x_457_;
goto v_reusejp_459_;
}
else
{
lean_object* v_reuseFailAlloc_461_; 
v_reuseFailAlloc_461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_461_, 0, v_a_455_);
v___x_460_ = v_reuseFailAlloc_461_;
goto v_reusejp_459_;
}
v_reusejp_459_:
{
return v___x_460_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___boxed(lean_object* v___x_463_, lean_object* v___x_464_, lean_object* v_i_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_){
_start:
{
uint8_t v___x_10832__boxed_475_; lean_object* v_res_476_; 
v___x_10832__boxed_475_ = lean_unbox(v___x_464_);
v_res_476_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0(v___x_463_, v___x_10832__boxed_475_, v_i_465_, v___y_466_, v___y_467_, v___y_468_, v___y_469_, v___y_470_, v___y_471_, v___y_472_, v___y_473_);
lean_dec(v___y_469_);
lean_dec_ref(v___y_468_);
lean_dec(v___y_467_);
lean_dec_ref(v___y_466_);
return v_res_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1(lean_object* v_x_477_, lean_object* v_a_478_, lean_object* v_a_479_, lean_object* v_a_480_, lean_object* v_a_481_, lean_object* v_a_482_, lean_object* v_a_483_, lean_object* v_a_484_, lean_object* v_a_485_){
_start:
{
lean_object* v___x_487_; uint8_t v___x_488_; 
v___x_487_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticApply__At___00__closed__3));
lean_inc(v_x_477_);
v___x_488_ = l_Lean_Syntax_isOfKind(v_x_477_, v___x_487_);
if (v___x_488_ == 0)
{
lean_object* v___x_489_; 
lean_dec(v_x_477_);
v___x_489_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__0___redArg();
return v___x_489_;
}
else
{
lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v_i_493_; lean_object* v___x_494_; lean_object* v___f_495_; uint8_t v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; 
v___x_490_ = lean_unsigned_to_nat(1u);
v___x_491_ = l_Lean_Syntax_getArg(v_x_477_, v___x_490_);
v___x_492_ = lean_unsigned_to_nat(3u);
v_i_493_ = l_Lean_Syntax_getArg(v_x_477_, v___x_492_);
lean_dec(v_x_477_);
v___x_494_ = lean_box(v___x_488_);
v___f_495_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___lam__0___boxed), 12, 3);
lean_closure_set(v___f_495_, 0, v___x_491_);
lean_closure_set(v___f_495_, 1, v___x_494_);
lean_closure_set(v___f_495_, 2, v_i_493_);
v___x_496_ = 1;
lean_inc(v_a_479_);
lean_inc_ref(v_a_478_);
v___x_497_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withMainContext___boxed), 11, 4);
lean_closure_set(v___x_497_, 0, lean_box(0));
lean_closure_set(v___x_497_, 1, v___f_495_);
lean_closure_set(v___x_497_, 2, v_a_478_);
lean_closure_set(v___x_497_, 3, v_a_479_);
v___x_498_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_497_, v___x_496_, v_a_480_, v_a_481_, v_a_482_, v_a_483_, v_a_484_, v_a_485_);
return v___x_498_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1___boxed(lean_object* v_x_499_, lean_object* v_a_500_, lean_object* v_a_501_, lean_object* v_a_502_, lean_object* v_a_503_, lean_object* v_a_504_, lean_object* v_a_505_, lean_object* v_a_506_, lean_object* v_a_507_, lean_object* v_a_508_){
_start:
{
lean_object* v_res_509_; 
v_res_509_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1(v_x_499_, v_a_500_, v_a_501_, v_a_502_, v_a_503_, v_a_504_, v_a_505_, v_a_506_, v_a_507_);
lean_dec(v_a_507_);
lean_dec_ref(v_a_506_);
lean_dec(v_a_505_);
lean_dec_ref(v_a_504_);
lean_dec(v_a_503_);
lean_dec_ref(v_a_502_);
lean_dec(v_a_501_);
lean_dec_ref(v_a_500_);
return v_res_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1(lean_object* v_mvarId_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_){
_start:
{
lean_object* v___x_520_; 
v___x_520_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1___redArg(v_mvarId_510_, v___y_516_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1___boxed(lean_object* v_mvarId_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_){
_start:
{
lean_object* v_res_531_; 
v_res_531_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1(v_mvarId_521_, v___y_522_, v___y_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_, v___y_529_);
lean_dec(v___y_529_);
lean_dec_ref(v___y_528_);
lean_dec(v___y_527_);
lean_dec_ref(v___y_526_);
lean_dec(v___y_525_);
lean_dec_ref(v___y_524_);
lean_dec(v___y_523_);
lean_dec_ref(v___y_522_);
lean_dec(v_mvarId_521_);
return v_res_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5(lean_object* v_00_u03b1_532_, lean_object* v_ref_533_, lean_object* v_msg_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_){
_start:
{
lean_object* v___x_544_; 
v___x_544_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5___redArg(v_ref_533_, v_msg_534_, v___y_535_, v___y_536_, v___y_537_, v___y_538_, v___y_539_, v___y_540_, v___y_541_, v___y_542_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5___boxed(lean_object* v_00_u03b1_545_, lean_object* v_ref_546_, lean_object* v_msg_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_){
_start:
{
lean_object* v_res_557_; 
v_res_557_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5(v_00_u03b1_545_, v_ref_546_, v_msg_547_, v___y_548_, v___y_549_, v___y_550_, v___y_551_, v___y_552_, v___y_553_, v___y_554_, v___y_555_);
lean_dec(v___y_555_);
lean_dec_ref(v___y_554_);
lean_dec(v___y_553_);
lean_dec_ref(v___y_552_);
lean_dec(v___y_551_);
lean_dec_ref(v___y_550_);
lean_dec(v___y_549_);
lean_dec_ref(v___y_548_);
lean_dec(v_ref_546_);
return v_res_557_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1(lean_object* v_00_u03b2_558_, lean_object* v_x_559_, lean_object* v_x_560_){
_start:
{
uint8_t v___x_561_; 
v___x_561_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1___redArg(v_x_559_, v_x_560_);
return v___x_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1___boxed(lean_object* v_00_u03b2_562_, lean_object* v_x_563_, lean_object* v_x_564_){
_start:
{
uint8_t v_res_565_; lean_object* v_r_566_; 
v_res_565_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1(v_00_u03b2_562_, v_x_563_, v_x_564_);
lean_dec(v_x_564_);
lean_dec_ref(v_x_563_);
v_r_566_ = lean_box(v_res_565_);
return v_r_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6(lean_object* v_00_u03b1_567_, lean_object* v_msg_568_, lean_object* v___y_569_, lean_object* v___y_570_, lean_object* v___y_571_, lean_object* v___y_572_, lean_object* v___y_573_, lean_object* v___y_574_, lean_object* v___y_575_, lean_object* v___y_576_){
_start:
{
lean_object* v___x_578_; 
v___x_578_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6___redArg(v_msg_568_, v___y_573_, v___y_574_, v___y_575_, v___y_576_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6___boxed(lean_object* v_00_u03b1_579_, lean_object* v_msg_580_, lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__5_spec__6(v_00_u03b1_579_, v_msg_580_, v___y_581_, v___y_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v___y_584_);
lean_dec_ref(v___y_583_);
lean_dec(v___y_582_);
lean_dec_ref(v___y_581_);
return v_res_590_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2(lean_object* v_00_u03b2_591_, lean_object* v_x_592_, size_t v_x_593_, lean_object* v_x_594_){
_start:
{
uint8_t v___x_595_; 
v___x_595_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2___redArg(v_x_592_, v_x_593_, v_x_594_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b2_596_, lean_object* v_x_597_, lean_object* v_x_598_, lean_object* v_x_599_){
_start:
{
size_t v_x_11226__boxed_600_; uint8_t v_res_601_; lean_object* v_r_602_; 
v_x_11226__boxed_600_ = lean_unbox_usize(v_x_598_);
lean_dec(v_x_598_);
v_res_601_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2(v_00_u03b2_596_, v_x_597_, v_x_11226__boxed_600_, v_x_599_);
lean_dec(v_x_599_);
lean_dec_ref(v_x_597_);
v_r_602_ = lean_box(v_res_601_);
return v_r_602_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7(lean_object* v_00_u03b2_603_, lean_object* v_keys_604_, lean_object* v_vals_605_, lean_object* v_heq_606_, lean_object* v_i_607_, lean_object* v_k_608_){
_start:
{
uint8_t v___x_609_; 
v___x_609_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7___redArg(v_keys_604_, v_i_607_, v_k_608_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7___boxed(lean_object* v_00_u03b2_610_, lean_object* v_keys_611_, lean_object* v_vals_612_, lean_object* v_heq_613_, lean_object* v_i_614_, lean_object* v_k_615_){
_start:
{
uint8_t v_res_616_; lean_object* v_r_617_; 
v_res_616_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyAt______elabRules__Mathlib__Tactic__tacticApply__At____1_spec__1_spec__1_spec__2_spec__7(v_00_u03b2_610_, v_keys_611_, v_vals_612_, v_heq_613_, v_i_614_, v_k_615_);
lean_dec(v_k_615_);
lean_dec_ref(v_vals_612_);
lean_dec_ref(v_keys_611_);
v_r_617_ = lean_box(v_res_616_);
return v_r_617_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyAt(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ApplyAt(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ApplyAt(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyAt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ApplyAt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ApplyAt(builtin);
}
#ifdef __cplusplus
}
#endif
